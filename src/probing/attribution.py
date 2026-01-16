from collections.abc import Mapping
import torch

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

    def is_text_input(value):
        if isinstance(value, str):
            return True
        if isinstance(value, (list, tuple)) and value:
            return all(isinstance(item, str) for item in value)
        return False

    def get_input_value(inputs, key):
        if isinstance(inputs, Mapping):
            return inputs.get(key)
        if hasattr(inputs, "data") and isinstance(inputs.data, dict):
            return inputs.data.get(key)
        try:
            return inputs[key] if key in inputs else None
        except Exception:
            return None

    if is_text_input(clean_prefix):
        input_ids = None
    else:
        input_ids = get_input_value(clean_prefix, "input_ids")
    attention_mask = None if input_ids is None else get_input_value(clean_prefix, "attention_mask")
    if input_ids is not None:
        input_ids = input_ids if input_ids.dim() > 1 else torch.cat([input_ids], dim=0)
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask if attention_mask.dim() > 1 else torch.cat([attention_mask], dim=0)
            attention_mask = attention_mask.to(device)
            clean_prefix = {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            clean_prefix = input_ids
    elif not is_text_input(clean_prefix):
        clean_prefix = clean_prefix if clean_prefix.dim() > 1 else torch.cat([clean_prefix], dim=0)
        clean_prefix = clean_prefix.to(device)

    def metric_fn(model, submodule, probe):
        # Metric for attribution patching: Negative logit of label 1
        acts_gathered = submodule.output[0].sum(1)
        metric = - probe(acts_gathered.float())
        return metric
    
    # first run through a test input to figure out which hidden states are tuples
    is_tuple = {}
    with model.trace("_", **TRACER_KWARGS):
        for submodule in submodules:
            is_tuple[submodule] = True # type(submodule.output) == tuple

    hidden_states_clean = {}
    with model.trace(clean_prefix, **TRACER_KWARGS), torch.no_grad():
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
        with model.trace(**TRACER_KWARGS) as tracer:
            metrics = []
            fs = []
            for step in range(steps):
                alpha = step / steps
                f = (1 - alpha) * clean_state + alpha * patch_state
                f.act.retain_grad()
                f.res.retain_grad()
                fs.append(f)
                with tracer.invoke(clean_prefix, scan=TRACER_KWARGS['scan']):
                    if is_tuple[submodule]:
                        submodule.output[0][:] = dictionary.decode(f.act) + f.res
                    else:
                        submodule.output = dictionary.decode(f.act) + f.res
                    metrics.append(metric_fn(model, submodule, probe, **metric_kwargs))
            if not metrics:
                raise RuntimeError("No metrics collected; check steps and tracing inputs.")
            metric = metrics[0]
            for m in metrics[1:]:
                metric = metric + m
            metric.sum().backward(retain_graph=True) 

        mean_grad = sum([f.act.grad for f in fs]) / steps
        mean_residual_grad = sum([f.res.grad for f in fs]) / steps
        grad = SparseActivation(act=mean_grad, res=mean_residual_grad)
        delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
        effect = grad @ delta

        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad

    return (effects, deltas, grads, total_effect)
