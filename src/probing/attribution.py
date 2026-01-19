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

    clean_prefix = torch.cat([clean_prefix], dim=0).to(device)
    clean_inputs = {
        "input_ids": clean_prefix,
        "attention_mask": torch.ones_like(clean_prefix, device=device),
    }

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
        with model.trace(clean_inputs, **TRACER_KWARGS) as tracer:
            metrics = []
            fs = []
            step_count = 0
            for step in range(steps):
                step_count += 1
                alpha = step / steps
                f = (1 - alpha) * clean_state + alpha * patch_state
                f.act.retain_grad()
                f.res.retain_grad()
                fs.append(f)
                with tracer.invoke(clean_inputs, scan=TRACER_KWARGS['scan']):
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
