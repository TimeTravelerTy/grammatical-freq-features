import torch
import torch.nn.functional as F
from collections.abc import Mapping

from src.activations import SparseActivation
from src.probing.utils import logprob_sum_from_logits

TRACER_KWARGS = {'scan' : False, 'validate' : False}


def _resolve_device(model, submodules, dictionaries):
    for module in submodules:
        for param in module.parameters():
            if param.device.type != "meta":
                return param.device
    for dictionary in dictionaries.values():
        for param in dictionary.parameters():
            if param.device.type != "meta":
                return param.device
    model_root = getattr(model, "model", model)
    for param in model_root.parameters():
        if param.device.type != "meta":
            return param.device
    device = getattr(model, "device", None)
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device) and device.type != "meta":
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_inputs(prefix, device):
    if isinstance(prefix, Mapping):
        input_ids = prefix.get("input_ids")
        attention_mask = prefix.get("attention_mask")
        if input_ids is None:
            raise KeyError("input_ids")
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
        }
    prefix = torch.cat([prefix], dim=0).to(device)
    return {
        "input_ids": prefix,
        "attention_mask": torch.ones_like(prefix, device=device),
    }


def _align_tensor(clean_tensor, patch_tensor):
    if clean_tensor is None or patch_tensor is None:
        return patch_tensor
    if clean_tensor.shape == patch_tensor.shape:
        return patch_tensor
    if clean_tensor.dim() == 3 and patch_tensor.dim() == 3:
        clean_len = clean_tensor.shape[1]
        patch_len = patch_tensor.shape[1]
        if patch_len > clean_len:
            return patch_tensor[:, :clean_len, :]
        if patch_len < clean_len:
            pad_len = clean_len - patch_len
            return F.pad(patch_tensor, (0, 0, 0, pad_len))
    if clean_tensor.dim() == 2 and patch_tensor.dim() == 2:
        clean_len = clean_tensor.shape[0]
        patch_len = patch_tensor.shape[0]
        if patch_len > clean_len:
            return patch_tensor[:clean_len, :]
        if patch_len < clean_len:
            pad_len = clean_len - patch_len
            return F.pad(patch_tensor, (0, 0, 0, pad_len))
    return patch_tensor


def attribution_patching(
    clean_prefix,
    model,
    probe,
    submodules,
    dictionaries,
    patch_prefix=None,
    steps=10,
    metric_fn=None,
    metric_kwargs=None,
):
    device = _resolve_device(model, submodules, dictionaries)
    if hasattr(probe, "to"):
        probe = probe.to(device)
    if steps is None:
        steps = 1
    steps = int(steps)
    if steps < 1:
        steps = 1

    clean_inputs = _prepare_inputs(clean_prefix, device)
    patch_inputs = None
    if patch_prefix is not None:
        patch_inputs = _prepare_inputs(patch_prefix, device)

    def _default_metric_fn(model, submodule, probe, **_):
        # Metric for attribution patching: Negative logit of label 1
        acts = submodule.output[0]
        n_features = probe.linear.weight.shape[1]
        if acts.dim() == 3:
            if acts.shape[-1] == n_features:
                acts_gathered = acts.sum(1)
            elif acts.shape[-2] == n_features:
                acts_gathered = acts.sum(-1)
            else:
                acts_gathered = acts.sum(1)
        else:
            acts_gathered = acts
        if hasattr(probe, "parameters"):
            probe_param = next(probe.parameters())
            acts_gathered = acts_gathered.to(probe_param.device, dtype=probe_param.dtype)
        metric = - probe(acts_gathered)
        return metric
    
    if metric_fn is None:
        metric_fn = _default_metric_fn
    metric_kwargs = dict(metric_kwargs or {})
    
    # first run through a test input to figure out which hidden states are tuples
    is_tuple = {}
    with model.trace("_", **TRACER_KWARGS):
        for submodule in submodules:
            is_tuple[submodule] = isinstance(submodule.output, tuple)

    def _collect_hidden_states(inputs, logprob_positions=None):
        logits = None
        saved_outputs = {}
        with model.trace(inputs, **TRACER_KWARGS), torch.no_grad():
            for submodule in submodules:
                x = submodule.output
                if is_tuple[submodule]:
                    x = x[0]
                saved_outputs[submodule] = x.save()
            if logprob_positions is not None:
                logits = model.output[0].save()

        states = {}
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            saved = saved_outputs[submodule]
            x = saved.value if hasattr(saved, "value") else saved
            if hasattr(x, "device") and x.device.type != "meta":
                dict_param = next(dictionary.parameters())
                dict_device = dict_param.device
                dict_dtype = dict_param.dtype
                if dict_device != x.device or dict_dtype != x.dtype:
                    dictionary = dictionary.to(device=x.device, dtype=x.dtype)
                    dictionaries[submodule] = dictionary
            f = dictionary.encode(x)
            f = f.to(dtype=next(dictionary.parameters()).dtype)
            x_hat = dictionary.decode(f)
            residual = x - x_hat
            states[submodule] = SparseActivation(act=f, res=residual)

        logprob = None
        if logits is not None:
            logprob = logprob_sum_from_logits(
                logits.value,
                inputs["input_ids"],
                logprob_positions,
            )
        return states, logprob

    hidden_states_clean, _ = _collect_hidden_states(clean_inputs)
    if patch_inputs is None:
        hidden_states_patch = {
            k: SparseActivation(act=torch.zeros_like(v.act), res=torch.zeros_like(v.res))
            for k, v in hidden_states_clean.items()
        }
    else:
        patch_positions = metric_kwargs.get("patch_positions")
        hidden_states_patch, patch_logprob = _collect_hidden_states(
            patch_inputs,
            logprob_positions=patch_positions,
        )
        if patch_logprob is not None:
            metric_kwargs["patch_logprob"] = patch_logprob
        for submodule in submodules:
            clean_state = hidden_states_clean[submodule]
            patch_state = hidden_states_patch[submodule]
            aligned_act = _align_tensor(clean_state.act, patch_state.act)
            aligned_res = _align_tensor(clean_state.res, patch_state.res)
            hidden_states_patch[submodule] = SparseActivation(act=aligned_act, res=aligned_res)
    total_effect = None

    effects = {}
    deltas = {}
    grads = {}
    for submodule in submodules:
        dictionary = dictionaries[submodule]
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]
        dict_device = next(dictionary.parameters()).device

        clean_act = clean_state.act
        clean_res = clean_state.res
        patch_act = patch_state.act
        patch_res = patch_state.res
        if clean_act.dim() >= 3 and clean_act.shape[0] == 1:
            clean_act = clean_act.squeeze(0)
            clean_res = clean_res.squeeze(0)
            patch_act = patch_act.squeeze(0)
            patch_res = patch_res.squeeze(0)

        alphas = torch.linspace(0, 1, steps, device=dict_device).view(steps, 1, 1)
        f_act = ((1 - alphas) * clean_act + alphas * patch_act)
        f_res = ((1 - alphas) * clean_res + alphas * patch_res)
        dict_dtype = next(dictionary.parameters()).dtype
        f_act = f_act.to(dict_device, dtype=dict_dtype).detach().requires_grad_(True)
        f_res = f_res.to(dict_device, dtype=dict_dtype).detach().requires_grad_(True)

        input_ids = clean_inputs["input_ids"]
        attention_mask = clean_inputs["attention_mask"]
        batch_inputs = {
            "input_ids": input_ids.repeat(steps, 1),
            "attention_mask": attention_mask.repeat(steps, 1),
        }

        if metric_kwargs.get("_log_batch", False):
            print(
                "attribution_patching batched",
                {
                    "steps": steps,
                    "input_shape": tuple(batch_inputs["input_ids"].shape),
                    "f_act_shape": tuple(f_act.shape),
                    "f_res_shape": tuple(f_res.shape),
                },
            )

        with model.trace(batch_inputs, **TRACER_KWARGS):
            if is_tuple[submodule]:
                submodule.output[0][:] = dictionary.decode(f_act) + f_res
            else:
                submodule.output = dictionary.decode(f_act) + f_res
            metric = metric_fn(
                model,
                submodule,
                probe,
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                **metric_kwargs,
            )
            metric.sum().backward()

        if f_act.grad is None or f_res.grad is None:
            raise RuntimeError("Missing gradients for attribution patching batch.")

        mean_grad = f_act.grad.mean(dim=0)
        mean_residual_grad = f_res.grad.mean(dim=0)
        grad = SparseActivation(act=mean_grad, res=mean_residual_grad)
        delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
        effect = grad @ delta

        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad

    return (effects, deltas, grads, total_effect)
