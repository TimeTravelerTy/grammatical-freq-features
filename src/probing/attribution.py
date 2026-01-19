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
        acts_gathered = submodule.output[0].sum(1)
        metric = - probe(acts_gathered.float())
        return metric
    
    # first run through a test input to figure out which hidden states are tuples
    is_tuple = {}
    with model.trace("_", **TRACER_KWARGS):
        for submodule in submodules:
            is_tuple[submodule] = True # type(submodule.output) == tuple

    if metric_kwargs.get("_debug_trace", False):
        with model.trace(clean_inputs, **TRACER_KWARGS):
            test_metric = metric_fn(model, submodules[0], probe).save()
        print(
            "attribution_patching trace sanity",
            {
                "metric_shape": tuple(test_metric.value.shape) if hasattr(test_metric.value, "shape") else None,
                "input_ids_shape": tuple(clean_inputs["input_ids"].shape),
                "attention_mask_shape": tuple(clean_inputs["attention_mask"].shape),
            },
        )

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
        with model.trace(**TRACER_KWARGS) as tracer:
            metrics = []
            fs = []
            step_count = 0
            appended = False
            debug_step = False
            debug_invoke = False
            for step in range(steps):
                step_count += 1
                alpha = step / steps
                f = (1 - alpha) * clean_state + alpha * patch_state
                f.act.retain_grad()
                f.res.retain_grad()
                fs.append(f)
                if not debug_step:
                    print(
                        "attribution_patching step entry",
                        {
                            "step": step,
                            "steps": steps,
                            "clean_ids_shape": tuple(clean_inputs["input_ids"].shape),
                            "mask_shape": tuple(clean_inputs["attention_mask"].shape),
                        },
                    )
                    debug_step = True
                print("attribution_patching invoke pre", {"step": step, "alpha": alpha})
                with tracer.invoke(clean_inputs, scan=TRACER_KWARGS['scan']):
                    print("attribution_patching invoke in", {"step": step, "alpha": alpha})
                    if is_tuple[submodule]:
                        submodule.output[0][:] = dictionary.decode(f.act) + f.res
                    else:
                        submodule.output = dictionary.decode(f.act) + f.res
                    if not debug_invoke:
                        print(
                            "attribution_patching invoke entered",
                            {
                                "step": step,
                                "alpha": alpha,
                                "f_act_shape": tuple(f.act.shape) if hasattr(f.act, "shape") else None,
                                "decoded_shape": tuple(dictionary.decode(f.act).shape),
                            },
                        )
                        debug_invoke = True
                    metric_val = metric_fn(model, submodule, probe, **metric_kwargs)
                    metrics.append(metric_val)
                    if not appended:
                        print(
                            "attribution_patching append",
                            {
                                "step": step,
                                "alpha": alpha,
                                "metric_shape": tuple(metric_val.shape) if hasattr(metric_val, "shape") else None,
                                "input_ids_shape": tuple(clean_inputs["input_ids"].shape),
                                "attention_mask_shape": tuple(clean_inputs["attention_mask"].shape),
                            },
                        )
                    appended = True
            if not appended:
                ids_shape = tuple(clean_inputs["input_ids"].shape)
                mask_shape = tuple(clean_inputs["attention_mask"].shape)
                raise RuntimeError(
                    f"No metrics collected; steps={steps}, step_count={step_count}, "
                    f"input_ids_shape={ids_shape}, attention_mask_shape={mask_shape}"
                )
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
