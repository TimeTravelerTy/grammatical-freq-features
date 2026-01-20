import torch
from collections.abc import Mapping

from src.activations import SparseActivation

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


def attribution_patching(
    clean_prefix,
    model,
    probe,
    submodules,
    dictionaries,
    steps=10,
    metric_kwargs=dict(),
):
    device = _resolve_device(model, submodules, dictionaries)
    if hasattr(probe, "to"):
        probe = probe.to(device)
    if steps is None:
        steps = 1
    steps = int(steps)
    if steps < 1:
        steps = 1

    if isinstance(clean_prefix, Mapping):
        input_ids = clean_prefix.get("input_ids")
        attention_mask = clean_prefix.get("attention_mask")
        if input_ids is None:
            raise KeyError("input_ids")
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        clean_inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
        }
    else:
        clean_prefix = torch.cat([clean_prefix], dim=0).to(device)
        clean_inputs = {
            "input_ids": clean_prefix,
            "attention_mask": torch.ones_like(clean_prefix, device=device),
        }

    def metric_fn(model, submodule, probe):
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
    
    # first run through a test input to figure out which hidden states are tuples
    is_tuple = {}
    with model.trace("_", **TRACER_KWARGS):
        for submodule in submodules:
            is_tuple[submodule] = isinstance(submodule.output, tuple)

    hidden_states_clean = {}
    with model.trace(clean_inputs, **TRACER_KWARGS), torch.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]
            if hasattr(x, "device") and x.device.type != "meta":
                dict_param = next(dictionary.parameters())
                dict_device = dict_param.device
                dict_dtype = dict_param.dtype
                if dict_device != x.device or dict_dtype != x.dtype:
                    dictionary = dictionary.to(device=x.device, dtype=x.dtype)
                    dictionaries[submodule] = dictionary
                if hasattr(probe, "to"):
                    probe = probe.to(x.device)
            f = dictionary.encode(x)
            f = f.to(dtype=next(dictionary.parameters()).dtype)
            x_hat = dictionary.decode(f)
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseActivation(act=f.save(), res=residual.save())
    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}

    hidden_states_patch = {
        k : SparseActivation(act=torch.zeros_like(v.act), res=torch.zeros_like(v.res)) for k, v in hidden_states_clean.items()
    }
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
            metric = metric_fn(model, submodule, probe, **metric_kwargs)
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
