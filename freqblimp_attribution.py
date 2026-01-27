import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import HF_TOKEN


def _ensure_opensae_on_path():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    opensae_src = os.path.join(repo_root, "OpenSAE", "src")
    if opensae_src not in sys.path:
        sys.path.append(opensae_src)
    return opensae_src


_ensure_opensae_on_path()
try:
    from opensae import OpenSae, TransformerWithSae
    from opensae.sae_utils import torch_decode
except Exception as exc:
    raise RuntimeError(
        "Failed to import OpenSAE. Ensure OpenSAE is available in ./OpenSAE/src or installed."
    ) from exc


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
    return model, tokenizer


def build_wrapper(base_model, sae_path, device, dtype):
    sae_kwargs = {"torch_dtype": dtype}
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


def sentence_nll(logits, input_ids, attention_mask):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )
    if attention_mask is not None:
        shift_mask = attention_mask[:, 1:].contiguous().view(-1)
        loss = loss * shift_mask
    return loss.sum()


def forward_with_features(wrapper, tokenizer, sentence, device, require_grad):
    wrapper.clear_intermediates()
    encoded = tokenizer(sentence, return_tensors="pt", padding=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if require_grad:
        outputs = wrapper.transformer(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
    else:
        with torch.no_grad():
            outputs = wrapper.transformer(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
    feats = wrapper.saved_features
    if feats is None:
        raise RuntimeError("No SAE features captured; check hookpoints/config.")
    return outputs.logits, input_ids, attention_mask, feats


def collect_active_features(wrapper, tokenizer, examples, device, min_count, topk, include_special):
    counts = Counter()
    special_ids = set(tokenizer.all_special_ids)
    for ex in examples:
        for key in ("good", "bad"):
            sentence = ex[key]
            _, input_ids, _, feats = forward_with_features(
                wrapper, tokenizer, sentence, device, require_grad=False
            )
            token_ids = input_ids[0].tolist()
            idx = feats.sparse_feature_indices.detach().cpu()
            if not include_special and special_ids:
                mask = torch.tensor([tid in special_ids for tid in token_ids], dtype=torch.bool)
                if mask.any():
                    idx = idx.clone()
                    idx[mask] = -1
            flat = idx.view(-1).tolist()
            counts.update(i for i in flat if i >= 0)
    if min_count > 0:
        counts = Counter({k: v for k, v in counts.items() if v >= min_count})
    if topk is not None and topk > 0 and len(counts) > topk:
        counts = Counter(dict(counts.most_common(topk)))
    return counts


def make_pos_mask(good_ids, bad_ids, special_ids, include_special):
    min_len = min(len(good_ids), len(bad_ids))
    if include_special or not special_ids:
        return [True] * min_len
    mask = []
    for i in range(min_len):
        if good_ids[i] in special_ids or bad_ids[i] in special_ids:
            mask.append(False)
        else:
            mask.append(True)
    return mask


def score_pair(
    wrapper,
    tokenizer,
    example,
    device,
    active_set,
    include_special,
):
    good_sentence = example["good"]
    bad_sentence = example["bad"]

    logits_good, input_ids_good, attn_good, feats_good = forward_with_features(
        wrapper, tokenizer, good_sentence, device, require_grad=True
    )
    good_acts = feats_good.sparse_feature_activations
    good_acts.retain_grad()
    nll_good = sentence_nll(logits_good, input_ids_good, attn_good)

    logits_bad, input_ids_bad, attn_bad, feats_bad = forward_with_features(
        wrapper, tokenizer, bad_sentence, device, require_grad=False
    )
    nll_bad = sentence_nll(logits_bad, input_ids_bad, attn_bad).detach()

    delta = nll_bad - nll_good
    wrapper.transformer.zero_grad(set_to_none=True)
    wrapper.sae.zero_grad(set_to_none=True)
    delta.backward()

    if good_acts.grad is None:
        raise RuntimeError("No gradients for SAE activations; check decoder_impl and hookpoints.")

    good_idx = feats_good.sparse_feature_indices.detach().cpu()
    good_act = good_acts.detach().cpu()
    good_grad = good_acts.grad.detach().cpu()
    bad_idx = feats_bad.sparse_feature_indices.detach().cpu()
    bad_act = feats_bad.sparse_feature_activations.detach().cpu()

    good_ids = input_ids_good[0].tolist()
    bad_ids = input_ids_bad[0].tolist()
    pos_mask = make_pos_mask(good_ids, bad_ids, set(tokenizer.all_special_ids), include_special)
    min_len = min(len(good_ids), len(bad_ids))

    scores = defaultdict(float)
    features_seen = set()
    for t in range(min_len):
        if not pos_mask[t]:
            continue
        g_idx = good_idx[t].tolist()
        g_act = good_act[t].tolist()
        g_grad = good_grad[t].tolist()
        b_idx = bad_idx[t].tolist()
        b_act = bad_act[t].tolist()
        b_map = {idx: act for idx, act in zip(b_idx, b_act)}
        for feat, act, grad in zip(g_idx, g_act, g_grad):
            if active_set is not None and feat not in active_set:
                continue
            diff = b_map.get(feat, 0.0) - act
            scores[feat] += diff * grad
            features_seen.add(feat)

    return scores, features_seen, nll_good.detach().item(), nll_bad.item(), delta.detach().item()


def build_group_topk(score_sums, feature_counts, group_size, active_counts, topk):
    results = []
    if group_size <= 0:
        return results
    for feat, total in score_sums.items():
        mean_score = total / group_size
        count = feature_counts.get(feat, 0)
        mean_active = total / count if count else 0.0
        results.append(
            {
                "feature": feat,
                "mean_score": mean_score,
                "mean_score_active": mean_active,
                "activation_count": active_counts.get(feat, 0),
                "example_count": count,
            }
        )
    results.sort(key=lambda x: abs(x["mean_score"]), reverse=True)
    if topk and topk > 0:
        return results[:topk]
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Attribution patching on OpenSAE latents for freqBLiMP rare pairs."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B",
        help="HF model name or path.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="0,1,4,8,15",
        help="Comma-separated layer indices to analyze.",
    )
    parser.add_argument(
        "--sae_path_template",
        type=str,
        default="THU-KEG/OpenSAE-LLaMA-3.1-Layer_{layer:02d}",
        help="Template for SAE path (format with {layer}).",
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
        "--active_topk",
        type=int,
        default=0,
        help="Keep top-N most active features before attribution ranking (0 disables).",
    )
    parser.add_argument(
        "--min_active_count",
        type=int,
        default=3,
        help="Drop features with activation count below this threshold.",
    )
    parser.add_argument(
        "--include_special_tokens",
        action="store_true",
        help="Include special tokens in attribution sums.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/attribution",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--phenomenon_topk",
        type=int,
        default=200,
        help="Top-K features to keep per phenomenon (0 disables).",
    )
    parser.add_argument(
        "--regime_topk",
        type=int,
        default=200,
        help="Top-K features to keep per frequency regime (0 disables).",
    )
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
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
    regime_counts = Counter(ex["regime"] for ex in examples)
    phenomenon_counts = Counter(ex.get("phenomenon") or "unknown" for ex in examples)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading base model...")
    base_model, tokenizer = build_base_model_and_tokenizer(args.model_name, device, dtype)
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN is empty; gated models may fail to load.")

    for layer in layers:
        sae_path = args.sae_path_template.format(layer=layer)
        print(f"\n[Layer {layer}] Loading SAE: {sae_path}")
        wrapper = build_wrapper(base_model, sae_path, device, dtype)

        print("[Layer {0}] Pre-filtering active features...".format(layer))
        active_counts = collect_active_features(
            wrapper,
            tokenizer,
            examples,
            device,
            args.min_active_count,
            args.active_topk,
            args.include_special_tokens,
        )
        active_set = set(active_counts.keys())
        print(f"[Layer {layer}] Active features: {len(active_set)}")

        score_sums = defaultdict(float)
        feature_example_counts = Counter()
        phenomenon_score_sums = defaultdict(lambda: defaultdict(float))
        phenomenon_feature_counts = defaultdict(Counter)
        regime_score_sums = defaultdict(lambda: defaultdict(float))
        regime_feature_counts = defaultdict(Counter)
        total_nll_good = 0.0
        total_nll_bad = 0.0
        total_delta = 0.0
        start = time.time()

        for i, ex in enumerate(examples, start=1):
            scores, features_seen, nll_good, nll_bad, delta = score_pair(
                wrapper,
                tokenizer,
                ex,
                device,
                active_set,
                args.include_special_tokens,
            )
            for feat, val in scores.items():
                score_sums[feat] += val
            phenomenon = ex.get("phenomenon") or "unknown"
            regime = ex.get("regime") or "unknown"
            for feat, val in scores.items():
                phenomenon_score_sums[phenomenon][feat] += val
                regime_score_sums[regime][feat] += val
            for feat in features_seen:
                feature_example_counts[feat] += 1
                phenomenon_feature_counts[phenomenon][feat] += 1
                regime_feature_counts[regime][feat] += 1
            total_nll_good += nll_good
            total_nll_bad += nll_bad
            total_delta += delta
            if args.log_every and i % args.log_every == 0:
                elapsed = time.time() - start
                print(
                    f"[Layer {layer}] {i}/{len(examples)} examples "
                    f"(+{elapsed:.1f}s, avg delta {total_delta / i:.4f})"
                )

        num_examples = len(examples)
        results = []
        for feat in active_set:
            total = score_sums.get(feat, 0.0)
            mean_score = total / num_examples
            mean_active = total / feature_example_counts[feat] if feature_example_counts[feat] else 0.0
            results.append(
                {
                    "feature": feat,
                    "mean_score": mean_score,
                    "mean_score_active": mean_active,
                    "activation_count": active_counts.get(feat, 0),
                    "example_count": feature_example_counts.get(feat, 0),
                }
            )
        results.sort(key=lambda x: abs(x["mean_score"]), reverse=True)

        phenomenon_topk = {}
        if args.phenomenon_topk and args.phenomenon_topk > 0:
            for phenomenon, sums in phenomenon_score_sums.items():
                phenomenon_topk[phenomenon] = build_group_topk(
                    sums,
                    phenomenon_feature_counts[phenomenon],
                    phenomenon_counts.get(phenomenon, 0),
                    active_counts,
                    args.phenomenon_topk,
                )

        regime_topk = {}
        if args.regime_topk and args.regime_topk > 0:
            for regime, sums in regime_score_sums.items():
                regime_topk[regime] = build_group_topk(
                    sums,
                    regime_feature_counts[regime],
                    regime_counts.get(regime, 0),
                    active_counts,
                    args.regime_topk,
                )

        summary = {
            "layer": layer,
            "sae_path": sae_path,
            "model_name": args.model_name,
            "num_examples": num_examples,
            "regime_counts": dict(regime_counts),
            "phenomenon_counts": dict(phenomenon_counts),
            "avg_nll_good": total_nll_good / num_examples,
            "avg_nll_bad": total_nll_bad / num_examples,
            "avg_delta": total_delta / num_examples,
            "active_features": len(active_set),
            "active_topk": args.active_topk,
            "min_active_count": args.min_active_count,
            "phenomenon_topk": args.phenomenon_topk,
            "regime_topk": args.regime_topk,
        }

        out_path = os.path.join(args.output_dir, f"freqblimp_rare_attribution_layer{layer:02d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary,
                    "features": results,
                    "phenomenon_features": phenomenon_topk,
                    "regime_features": regime_topk,
                },
                f,
                indent=2,
            )
        print(f"[Layer {layer}] Wrote results to {out_path}")
        remove_hooks(wrapper)
        del wrapper
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
