import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pyconll
import os


TRACER_KWARGS = {'scan': False, 'validate': False}


def extract_activations(model, dataloader, layer_num):
    """
    Extract activations for the entire dataset.
    """
    all_activations = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting activations"):
            text_batch = batch["sentence"]
            labels = batch["label"]

            with model.trace(text_batch, **TRACER_KWARGS):
                input = model.inputs.save()
                acts = model.model.layers[layer_num].output[0].save()
            
            # Remove padding tokens
            attn_mask = input[1]['attention_mask']
            acts = acts * attn_mask.unsqueeze(-1)

            pooled_acts = acts.sum(1)
            all_activations.append(pooled_acts.float().cpu().numpy())
            all_labels.append(labels.numpy())
    
    return np.vstack(all_activations), np.concatenate(all_labels)

def concept_filter(
    sentence,
    concept_key,
    concept_value,
    pos_tags=None,
    exclude_values=None,
    exclude_other_values=False,
    drop_conflicts=False,
):
    exclude_values = set(exclude_values or [])
    has_target = False
    has_excluded = False
    for token in sentence:
        if pos_tags and token.upos not in pos_tags:
            continue
        if concept_key not in token.feats:
            continue
        values = token.feats.get(concept_key, set())
        if concept_value in values:
            has_target = True
        if exclude_values and values.intersection(exclude_values):
            has_excluded = True
        if exclude_other_values and (values - {concept_value}):
            has_excluded = True
        if has_target and has_excluded:
            return None if drop_conflicts else False
    return has_target and not has_excluded


def concept_label_stats(
    conll_file,
    concept_key,
    concept_value,
    pos_tags=None,
    exclude_values=None,
    exclude_other_values=False,
    drop_conflicts=False,
):
    data = pyconll.load_from_file(conll_file)
    exclude_values = set(exclude_values or [])
    stats = {"only_target": 0, "only_excluded": 0, "both": 0, "neither": 0}
    for sentence in data:
        has_target = False
        has_excluded = False
        for token in sentence:
            if pos_tags and token.upos not in pos_tags:
                continue
            if concept_key not in token.feats:
                continue
            values = token.feats.get(concept_key, set())
            if concept_value in values:
                has_target = True
            if exclude_values and values.intersection(exclude_values):
                has_excluded = True
            if exclude_other_values and (values - {concept_value}):
                has_excluded = True
            if has_target and has_excluded:
                break
        if has_target and has_excluded:
            stats["both"] += 1
        elif has_target:
            stats["only_target"] += 1
        elif has_excluded:
            stats["only_excluded"] += 1
        else:
            stats["neither"] += 1
    stats["total"] = sum(stats.values())
    if drop_conflicts:
        stats["dropped"] = stats["both"]
    return stats

def get_features_and_values(conll_file):
    data = pyconll.load_from_file(conll_file)
    features = {}
    for sentence in data:
        for token in sentence:
            for feat, values in token.feats.items():
                if feat not in features:
                    features[feat] = set()
                features[feat].update(values)
    return features

class LogisticRegressionPyTorch(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegressionPyTorch, self).__init__()
        self.linear = nn.Linear(n_features, 1) 

    def forward(self, x):
        return self.linear(x) 
    
def convert_probe_to_pytorch(probe, device=None):
    """Convert sklearn probe to a PyTorch model."""
    coef = probe.coef_.ravel()
    bias = probe.intercept_
    torch_probe = LogisticRegressionPyTorch(coef.shape[0])
    with torch.no_grad():
        torch_probe.linear.weight.copy_(torch.tensor(coef, dtype=torch.float32).unsqueeze(0))
        torch_probe.linear.bias.copy_(torch.tensor(bias, dtype=torch.float32))
    if device is not None:
        torch_probe = torch_probe.to(device)
    return torch_probe
