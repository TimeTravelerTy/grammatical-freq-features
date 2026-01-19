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
            probe_device = next(probe.parameters()).device
            acts_gathered = acts_gathered.to(probe_device)
        metric = - probe(acts_gathered.float())
        return metric
    
    # first run through a test input to figure out which hidden states are tuples
    is_tuple = {}
    with model.trace("_", **TRACER_KWARGS):
        for submodule in submodules:
            is_tuple[submodule] = True # type(submodule.output) == tuple

    hidden_states_clean = {}
    with model.trace(clean_inputs, **TRACER_KWARGS), torch.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]
            if hasattr(x, "device") and x.device.type != "meta":
                dict_device = next(dictionary.parameters()).device
                if dict_device != x.device:
                    dictionary = dictionary.to(x.device)
                    dictionaries[submodule] = dictionary
                if hasattr(probe, "to"):
                    probe = probe.to(x.device)
            f = dictionary.encode(x)
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
        act_grad_sum = None
        res_grad_sum = None
        for step in range(steps):
            alpha = step / steps
            f = (1 - alpha) * clean_state + alpha * patch_state
            f_act = f.act.to(dict_device).detach().clone().requires_grad_(True)
            f_res = f.res.to(dict_device).detach().clone().requires_grad_(True)
            f = SparseActivation(act=f_act, res=f_res)
            with model.trace(clean_inputs, **TRACER_KWARGS):
                if is_tuple[submodule]:
                    submodule.output[0][:] = dictionary.decode(f.act) + f.res
                else:
                    submodule.output = dictionary.decode(f.act) + f.res
                metric = metric_fn(model, submodule, probe, **metric_kwargs)
                metric.sum().backward()
            if f.act.grad is None or f.res.grad is None:
                raise RuntimeError("Missing gradients for attribution patching step.")
            if act_grad_sum is None:
                act_grad_sum = f.act.grad
                res_grad_sum = f.res.grad
            else:
                act_grad_sum = act_grad_sum + f.act.grad
                res_grad_sum = res_grad_sum + f.res.grad
        mean_grad = act_grad_sum / steps
        mean_residual_grad = res_grad_sum / steps
        grad = SparseActivation(act=mean_grad, res=mean_residual_grad)
        delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
        effect = grad @ delta

        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad

    return (effects, deltas, grads, total_effect)
