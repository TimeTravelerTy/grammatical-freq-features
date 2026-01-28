import argparse
import csv
import glob
import json
import math
import os
import random
from collections import Counter, defaultdict
import heapq

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import HF_TOKEN


def _ensure_opensae_on_path():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    opensae_src = os.path.join(repo_root, "OpenSAE", "src")
    if opensae_src not in sys.path:
        sys.path.append(opensae_src)
    return opensae_src


import sys

_ensure_opensae_on_path()
try:
    from opensae import OpenSae, TransformerWithSae
    from opensae.sae_utils import torch_decode
    from opensae import config_utils as _opensae_config_utils
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Failed to import OpenSAE. Ensure OpenSAE is available in ./OpenSAE/src or installed."
    ) from exc


def _patch_opensae_dtype():
    def _fixed_get_torch_dtype(self):
        if isinstance(self.torch_dtype, torch.dtype):
            return self.torch_dtype
        dtype = getattr(torch, self.torch_dtype)
        assert isinstance(dtype, torch.dtype)
        return dtype

    _opensae_config_utils.PretrainedSaeConfig.get_torch_dtype = _fixed_get_torch_dtype


_patch_opensae_dtype()

DEFAULT_FILES = {
    "head": "data/freqBLiMP/freqBLiMP_head.jsonl",
    "tail": "data/freqBLiMP/freqBLiMP_tail.jsonl",
    "xtail": "data/freqBLiMP/freqBLiMP_xtail.jsonl",
}


def load_freqblimp_file(path, regime):
    examples = []
    if not os.path.exists(path):
        print(f"[warn] Missing file: {path} (skipping)")
        return examples
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            good = ex.get("good_rare")
            bad = ex.get("bad_rare")
            if not good or not bad:
                continue
            examples.append(
                {
                    "regime": regime,
                    "group": ex.get("group"),
                    "phenomenon": ex.get("phenomenon") or ex.get("group"),
                    "subtask": ex.get("subtask"),
                    "idx": ex.get("idx"),
                    "good": good,
                    "bad": bad,
                }
            )
    return examples


def _allocate_counts(total, group_sizes, rng):
    total_size = sum(group_sizes.values())
    if total_size == 0:
        return {k: 0 for k in group_sizes}
    raw = {k: total * (v / total_size) for k, v in group_sizes.items()}
    alloc = {k: min(int(raw[k]), group_sizes[k]) for k in group_sizes}
    remaining = total - sum(alloc.values())
    if remaining <= 0:
        return alloc
    frac = []
    for k, v in group_sizes.items():
        if alloc[k] >= v:
            continue
        frac.append((raw[k] - int(raw[k]), rng.random(), k))
    frac.sort(reverse=True)
    i = 0
    while remaining > 0 and frac:
        _, _, k = frac[i % len(frac)]
        if alloc[k] < group_sizes[k]:
            alloc[k] += 1
            remaining -= 1
        i += 1
    return alloc


def _sample_by_group(examples, n, key_fn, rng):
    if n is None or n >= len(examples):
        return list(examples)
    groups = defaultdict(list)
    for ex in examples:
        groups[key_fn(ex)].append(ex)
    group_sizes = {k: len(v) for k, v in groups.items()}
    alloc = _allocate_counts(n, group_sizes, rng)
    sampled = []
    for k, group in groups.items():
        rng.shuffle(group)
        sampled.extend(group[: alloc.get(k, 0)])
    rng.shuffle(sampled)
    return sampled


def load_dataset(regimes, max_pairs, seed):
    rng = random.Random(seed)
    all_examples = []
    by_regime = {}
    for regime in regimes:
        path = DEFAULT_FILES.get(regime, regime)
        exs = load_freqblimp_file(path, regime)
        if exs:
            by_regime[regime] = exs
            all_examples.extend(exs)
    if not all_examples:
        raise RuntimeError("No examples found; check data files/regimes.")
    if max_pairs is None or max_pairs >= len(all_examples):
        return all_examples
    regime_sizes = {k: len(v) for k, v in by_regime.items()}
    regime_alloc = _allocate_counts(max_pairs, regime_sizes, rng)
    sampled = []
    for regime, exs in by_regime.items():
        target = regime_alloc.get(regime, 0)
        sampled.extend(_sample_by_group(exs, target, lambda e: e.get("phenomenon"), rng))
    rng.shuffle(sampled)
    return sampled


def build_base_model_and_tokenizer(model_name, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HF_TOKEN,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    model.eval()
    if device:
        model.to(device)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model, tokenizer


def build_wrapper(base_model, sae_path, device, sae_dtype):
    sae_kwargs = {"dtype": sae_dtype}
    if HF_TOKEN:
        sae_kwargs["token"] = HF_TOKEN
    sae = OpenSae.from_pretrained(sae_path, **sae_kwargs)
    sae.config.decoder_impl = "torch"
    sae.decode_fn = torch_decode
    wrapper = TransformerWithSae(base_model, sae, device=device)
    wrapper.transformer.eval()
    wrapper.sae.eval()
    for param in wrapper.transformer.parameters():
        param.requires_grad_(False)
    for param in wrapper.sae.parameters():
        param.requires_grad_(False)
    return wrapper


def remove_hooks(wrapper):
    for handle in wrapper.forward_hook_handle.values():
        handle.remove()
    for handle in wrapper.backward_hook_handle.values():
        handle.remove()


def forward_with_features(wrapper, tokenizer, sentence, device, encoded=None):
    wrapper.clear_intermediates()
    if encoded is None:
        encoded = tokenizer(sentence, return_tensors="pt", padding=False)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
    else:
        input_ids = torch.tensor([encoded["input_ids"]], device=device, dtype=torch.long)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = torch.tensor([attention_mask], device=device, dtype=torch.long)
    with torch.inference_mode():
        outputs = wrapper.transformer(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
    feats = wrapper.saved_features
    if feats is None:
        raise RuntimeError("No SAE features captured; check hookpoints/config.")
    return outputs.logits, input_ids, attention_mask, feats


def forward_with_features_batch(wrapper, tokenizer, batch_items, device):
    wrapper.clear_intermediates()
    all_encoded = all(item.get("encoded") is not None for item in batch_items)
    if all_encoded:
        max_len = max(len(item["encoded"]["input_ids"]) for item in batch_items)
        pad_id = tokenizer.pad_token_id
        input_ids = torch.full((len(batch_items), max_len), pad_id, device=device, dtype=torch.long)
        attention_mask = torch.zeros((len(batch_items), max_len), device=device, dtype=torch.long)
        for i, item in enumerate(batch_items):
            ids = item["encoded"]["input_ids"]
            mask = item["encoded"].get("attention_mask")
            input_ids[i, : len(ids)] = torch.tensor(ids, device=device, dtype=torch.long)
            if mask is not None:
                attention_mask[i, : len(mask)] = torch.tensor(mask, device=device, dtype=torch.long)
            else:
                attention_mask[i, : len(ids)] = 1
    else:
        encoded = tokenizer(
            [item["sentence"] for item in batch_items],
            return_tensors="pt",
            padding=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
    with torch.inference_mode():
        outputs = wrapper.transformer(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
    feats = wrapper.saved_features
    if feats is None:
        raise RuntimeError("No SAE features captured; check hookpoints/config.")
    return input_ids, attention_mask, feats


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _activation_any_rate(feature, num_examples):
    if not num_examples:
        return 0.0
    return feature.get("example_count", 0) / num_examples


def _top_phenomena_from_profile(cluster_profile_row):
    items = [(k, float(v)) for k, v in cluster_profile_row.items() if k not in ("cluster", "size")]
    items = [(k, v) for k, v in items if v != 0.0]
    items.sort(key=lambda x: x[1], reverse=True)
    return [k for k, _ in items[:3]]


def _parse_layers(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _select_specialists(cluster_rows, cluster_stats, min_abs_mean_score):
    clusters_sorted = sorted(
        cluster_stats.items(), key=lambda x: (x[1]["mean_entropy"], -x[1]["size"])
    )
    chosen = [cid for cid, _ in clusters_sorted[:2]]
    selected = []
    for cid in chosen:
        feats = [r for r in cluster_rows if r["cluster"] == cid]
        feats = [
            r
            for r in feats
            if r["activation_any_rate"] <= 0.99 and abs(r["mean_score"]) >= min_abs_mean_score
        ]
        feats.sort(key=lambda r: r["entropy_positive"])
        selected.extend(feats[:4])
    return selected


def _select_generalists(cluster_rows, cluster_stats, min_abs_mean_score, target=5):
    clusters_sorted = sorted(
        cluster_stats.items(), key=lambda x: x[1]["mean_entropy"], reverse=True
    )
    selected = []
    for cid, _ in clusters_sorted:
        feats = [r for r in cluster_rows if r["cluster"] == cid]
        feats = [
            r
            for r in feats
            if r["activation_any_rate"] <= 0.99 and abs(r["mean_score"]) >= min_abs_mean_score
        ]
        feats.sort(key=lambda r: r["mean_score"], reverse=True)
        for r in feats:
            selected.append(r)
            if len(selected) >= target:
                return selected
    return selected


def _select_frequency_sensitive(
    feature_lookup,
    head_map,
    xtail_map,
    min_abs_mean_score,
    target=5,
):
    candidates = []
    for fid, info in feature_lookup.items():
        if info["activation_any_rate"] > 0.99:
            continue
        mean_score = info["mean_score"]
        if abs(mean_score) < min_abs_mean_score:
            continue
        head = head_map.get(fid, 0.0)
        xtail = xtail_map.get(fid, 0.0)
        diff = head - xtail
        candidates.append((fid, diff, abs(diff)))

    if not candidates:
        return []

    pos = sorted([c for c in candidates if c[1] > 0], key=lambda x: x[2], reverse=True)
    neg = sorted([c for c in candidates if c[1] < 0], key=lambda x: x[2], reverse=True)

    selected = []
    want_pos = math.ceil(target / 2)
    want_neg = target - want_pos
    for fid, diff, _ in pos[:want_pos]:
        selected.append((fid, diff))
    for fid, diff, _ in neg[:want_neg]:
        selected.append((fid, diff))

    if len(selected) < target:
        remaining = [c for c in candidates if (c[0], c[1]) not in selected]
        remaining.sort(key=lambda x: x[2], reverse=True)
        for fid, diff, _ in remaining:
            if len(selected) >= target:
                break
            selected.append((fid, diff))
    return selected


def _update_heap(heap, entry, k):
    if len(heap) < k:
        heapq.heappush(heap, (entry["activation_value"], entry))
        return
    if entry["activation_value"] > heap[0][0]:
        heapq.heapreplace(heap, (entry["activation_value"], entry))


def main():
    parser = argparse.ArgumentParser(
        description="Select hero SAE features per layer and extract top contexts."
    )
    parser.add_argument("--attribution_dir", type=str, default="outputs/attribution")
    parser.add_argument(
        "--phenomenon_profile_dir",
        type=str,
        default="outputs/attribution/phenomenon_profile",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/attribution/hero_features")
    parser.add_argument(
        "--layers",
        type=str,
        default="0,1,4,8,15",
        help="Comma-separated layer indices to analyze.",
    )
    parser.add_argument(
        "--min_abs_mean_score",
        type=float,
        default=1e-4,
        help="Minimum absolute overall mean_score to keep a feature.",
    )
    parser.add_argument(
        "--context_topk",
        type=int,
        default=20,
        help="Top contexts to keep per feature per regime.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for context scanning.",
    )
    parser.add_argument(
        "--sentence_field",
        type=str,
        default="both",
        choices=["good", "bad", "both"],
        help="Which sentence field(s) to scan for activations.",
    )
    parser.add_argument(
        "--regimes",
        type=str,
        default="head,tail,xtail",
        help="Comma-separated regimes or explicit file paths.",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Optional cap on total pairs (sampled proportionally per regime/phenomenon).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument(
        "--sae_dtype",
        type=str,
        default="float32",
        help="SAE dtype string (e.g., float32).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B",
        help="HF model name or path.",
    )
    parser.add_argument(
        "--sae_path_template",
        type=str,
        default="THU-KEG/OpenSAE-LLaMA-3.1-Layer_{layer:02d}",
        help="Template for SAE path (format with {layer}).",
    )
    parser.add_argument(
        "--cache_tokens",
        action="store_true",
        help="Pre-tokenize examples to reduce tokenizer overhead.",
    )
    parser.add_argument(
        "--skip_contexts",
        action="store_true",
        help="Only select hero features and write CSV; skip context extraction.",
    )
    parser.add_argument("--log_every", type=int, default=50)

    args = parser.parse_args()
    layers = _parse_layers(args.layers)
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(args.dtype.lower(), torch.float16)
    device = torch.device(args.device)
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        print("WARNING: float16/bfloat16 on CPU is unsupported; using float32.")
        dtype = torch.float32
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; pass --device cpu or use a GPU node.")

    examples = load_dataset(regimes, args.max_pairs, args.seed)
    print(f"Examples loaded: {len(examples)}")

    if args.cache_tokens:
        print("Caching tokenization...")
        base_model, tokenizer = build_base_model_and_tokenizer(args.model_name, device, dtype)
        for i, ex in enumerate(examples, start=1):
            cached = {}
            for key in ("good", "bad"):
                enc = tokenizer(ex[key], padding=False, return_attention_mask=True)
                cached[key] = {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc.get("attention_mask"),
                }
            ex["_tokens"] = cached
            if args.log_every and i % args.log_every == 0:
                print(f"  cached {i}/{len(examples)}")
    else:
        base_model, tokenizer = build_base_model_and_tokenizer(args.model_name, device, dtype)

    if not HF_TOKEN:
        print("WARNING: HF_TOKEN is empty; gated models may fail to load.")

    os.makedirs(args.output_dir, exist_ok=True)
    contexts_dir = os.path.join(args.output_dir, "contexts")
    os.makedirs(contexts_dir, exist_ok=True)

    hero_rows = []

    for layer in layers:
        attrib_path = os.path.join(
            args.attribution_dir, f"freqblimp_rare_attribution_layer{layer:02d}.json"
        )
        if not os.path.exists(attrib_path):
            print(f"[warn] Missing attribution file: {attrib_path} (skipping layer)")
            continue
        with open(attrib_path, "r", encoding="utf-8") as f:
            attrib = json.load(f)

        summary = attrib.get("summary", {})
        num_examples = summary.get("num_examples", 0)
        sae_path = summary.get("sae_path") or args.sae_path_template.format(layer=layer)

        features = attrib.get("features", [])
        feature_lookup = {}
        for feat in features:
            fid = feat.get("feature")
            if fid is None:
                continue
            feature_lookup[int(fid)] = {
                "mean_score": feat.get("mean_score", 0.0),
                "activation_any_rate": _activation_any_rate(feat, num_examples),
            }

        # Regime maps
        regime_features = attrib.get("regime_features", {})
        head_map = {int(f["feature"]): f.get("mean_score", 0.0) for f in regime_features.get("head", [])}
        tail_map = {int(f["feature"]): f.get("mean_score", 0.0) for f in regime_features.get("tail", [])}
        xtail_map = {int(f["feature"]): f.get("mean_score", 0.0) for f in regime_features.get("xtail", [])}

        cluster_path = os.path.join(
            args.phenomenon_profile_dir, f"feature_clusters_layer{layer:02d}.csv"
        )
        profile_path = os.path.join(
            args.phenomenon_profile_dir, f"cluster_profiles_layer{layer:02d}.csv"
        )
        if not os.path.exists(cluster_path) or not os.path.exists(profile_path):
            print(f"[warn] Missing clustering outputs for layer {layer}; skipping")
            continue

        cluster_rows_raw = _read_csv(cluster_path)
        cluster_rows = []
        for r in cluster_rows_raw:
            cluster_rows.append(
                {
                    "feature": int(r["feature"]),
                    "cluster": int(r["cluster"]),
                    "entropy_positive": float(r["entropy_positive"]),
                    "mean_score": float(r["mean_score"]),
                    "activation_any_rate": float(r["activation_any_rate"]),
                }
            )

        # Cluster stats
        cluster_stats = {}
        for r in cluster_rows:
            cid = r["cluster"]
            cluster_stats.setdefault(cid, {"size": 0, "entropy_sum": 0.0})
            cluster_stats[cid]["size"] += 1
            cluster_stats[cid]["entropy_sum"] += r["entropy_positive"]
        for cid, stats in cluster_stats.items():
            stats["mean_entropy"] = stats["entropy_sum"] / max(1, stats["size"])

        # Cluster top phenomena
        profile_rows = _read_csv(profile_path)
        cluster_top_phen = {}
        for r in profile_rows:
            cid = int(r["cluster"])
            cluster_top_phen[cid] = _top_phenomena_from_profile(r)

        specialists = _select_specialists(cluster_rows, cluster_stats, args.min_abs_mean_score)
        generalists = _select_generalists(cluster_rows, cluster_stats, args.min_abs_mean_score, target=5)
        freq_sel = _select_frequency_sensitive(
            feature_lookup, head_map, xtail_map, args.min_abs_mean_score, target=5
        )

        selected_features = {}

        def _add_selected(row, category):
            fid = row["feature"]
            entry = selected_features.get(fid)
            if entry is None:
                selected_features[fid] = {"row": row, "categories": {category}}
            else:
                entry["categories"].add(category)

        for r in specialists:
            _add_selected(r, "specialist")
        for r in generalists:
            _add_selected(r, "generalist")
        for fid, _ in freq_sel:
            # Build a row from lookup
            info = feature_lookup.get(fid, {"mean_score": 0.0, "activation_any_rate": 0.0})
            # If present in cluster rows, use that to get entropy/cluster
            match = next((r for r in cluster_rows if r["feature"] == fid), None)
            if match is None:
                match = {
                    "feature": fid,
                    "cluster": -1,
                    "entropy_positive": 0.0,
                    "mean_score": info["mean_score"],
                    "activation_any_rate": info["activation_any_rate"],
                }
            _add_selected(match, "frequency_sensitive")

        # Build CSV rows
        for fid, payload in selected_features.items():
            row = payload["row"]
            cluster_id = row.get("cluster", -1)
            hero_rows.append(
                {
                    "feature_id": fid,
                    "layer": layer,
                    "cluster_id": cluster_id,
                    "entropy": row.get("entropy_positive", 0.0),
                    "mean_score_overall": row.get("mean_score", 0.0),
                    "mean_score_head": head_map.get(fid, 0.0),
                    "mean_score_tail": tail_map.get(fid, 0.0),
                    "mean_score_xtail": xtail_map.get(fid, 0.0),
                    "cluster_top_phenomena": ",".join(cluster_top_phen.get(cluster_id, [])),
                    "category": ",".join(sorted(payload["categories"])),
                }
            )

        if args.skip_contexts:
            continue

        print(f"[Layer {layer}] Loading SAE: {sae_path}")
        wrapper = build_wrapper(base_model, sae_path, device, args.sae_dtype)

        selected_set = set(selected_features.keys())
        heaps = {fid: {reg: [] for reg in regimes} for fid in selected_set}

        fields = ["good", "bad"] if args.sentence_field == "both" else [args.sentence_field]

        items = []
        for ex in examples:
            regime = ex["regime"]
            if regime not in regimes:
                continue
            for field in fields:
                encoded = None
                if "_tokens" in ex:
                    encoded = ex["_tokens"].get(field)
                items.append(
                    {
                        "regime": regime,
                        "phenomenon": ex.get("phenomenon"),
                        "example_id": ex.get("idx"),
                        "sentence": ex[field],
                        "encoded": encoded,
                        "sentence_field": field,
                    }
                )

        for i in range(0, len(items), args.batch_size):
            batch = items[i : i + args.batch_size]
            input_ids, attention_mask, feats = forward_with_features_batch(
                wrapper, tokenizer, batch, device
            )
            idxs = feats.sparse_feature_indices.detach().cpu()
            acts = feats.sparse_feature_activations.detach().cpu()
            input_ids = input_ids.detach().cpu()
            attn = attention_mask.detach().cpu() if attention_mask is not None else None

            for b, item in enumerate(batch):
                if attn is not None:
                    seq_len = int(attn[b].sum().item())
                else:
                    seq_len = input_ids.shape[1]
                token_ids = input_ids[b, :seq_len].tolist()
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                idxs_list = idxs[b, :seq_len].tolist()
                acts_list = acts[b, :seq_len].tolist()

                max_by_feature = {}
                for t, (t_idxs, t_acts) in enumerate(zip(idxs_list, acts_list)):
                    for fid, act in zip(t_idxs, t_acts):
                        if fid not in selected_set:
                            continue
                        if act <= 0:
                            continue
                        prev = max_by_feature.get(fid)
                        if prev is None or act > prev[0]:
                            max_by_feature[fid] = (act, t)

                for fid, (act, t) in max_by_feature.items():
                    start = max(0, t - 4)
                    end = min(len(tokens), t + 5)
                    window_ids = token_ids[start:end]
                    window_text = tokenizer.decode(window_ids, clean_up_tokenization_spaces=True)
                    entry = {
                        "example_id": item["example_id"],
                        "phenomenon": item["phenomenon"],
                        "regime": item["regime"],
                        "sentence_field": item["sentence_field"],
                        "sentence": item["sentence"],
                        "token_index": t,
                        "token_str": tokens[t],
                        "activation_value": act,
                        "window_text": window_text,
                    }
                    _update_heap(heaps[fid][item["regime"]], entry, args.context_topk)

            if args.log_every and (i // args.batch_size + 1) % args.log_every == 0:
                print(f"  processed {min(i + args.batch_size, len(items))}/{len(items)}")

        # Save contexts per feature
        for fid in selected_set:
            payload = {
                "feature_id": fid,
                "layer": layer,
                "contexts": {},
            }
            for regime in regimes:
                heap = heaps[fid][regime]
                sorted_entries = [e for _, e in sorted(heap, key=lambda x: x[0], reverse=True)]
                payload["contexts"][regime] = sorted_entries
            out_path = os.path.join(contexts_dir, f"layer{layer:02d}_feature{fid}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        remove_hooks(wrapper)
        del wrapper
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Write hero features CSV
    out_csv = os.path.join(args.output_dir, "hero_features.csv")
    fieldnames = [
        "feature_id",
        "layer",
        "cluster_id",
        "entropy",
        "mean_score_overall",
        "mean_score_head",
        "mean_score_tail",
        "mean_score_xtail",
        "cluster_top_phenomena",
        "category",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in hero_rows:
            writer.writerow(row)

    print(f"Wrote hero feature summary to {out_csv}")


if __name__ == "__main__":
    main()
