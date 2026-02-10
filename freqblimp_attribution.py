import argparse
import glob
import json
import os
import random
import sys
import time
import types
from collections import Counter, defaultdict

import torch
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
    from opensae import config_utils as _opensae_config_utils
except Exception as exc:
    raise RuntimeError(
        "Failed to import OpenSAE. Ensure OpenSAE is available in ./OpenSAE/src or installed."
    ) from exc


def _patch_opensae_dtype():
    # Patch OpenSAE dtype handling to avoid UnboundLocalError without modifying vendor code.
    def _fixed_get_torch_dtype(self):
        if isinstance(self.torch_dtype, torch.dtype):
            return self.torch_dtype
        dtype = getattr(torch, self.torch_dtype)
        assert isinstance(dtype, torch.dtype)
        return dtype

    _opensae_config_utils.PretrainedSaeConfig.get_torch_dtype = _fixed_get_torch_dtype


_patch_opensae_dtype()


DEFAULT_FILES = {
    "head": [
        "data/freqBLiMP/freqBLiMP_head.jsonl",
        "data/freqBLiMP/freq_blimp_head*.jsonl",
    ],
    "tail": [
        "data/freqBLiMP/freqBLiMP_tail.jsonl",
        "data/freqBLiMP/freq_blimp_tail*.jsonl",
    ],
    "xtail": [
        "data/freqBLiMP/freqBLiMP_xtail.jsonl",
        "data/freqBLiMP/freq_blimp_xtail*.jsonl",
    ],
}

PREFIX_LABEL_KEYS = (
    "prefix_eval",
    "prefix_evaluation",
    "prefix",
    "is_prefix",
    "is_prefix_eval",
    "one_prefix",
    "one_prefix_method",
    "use_prefix",
)


def _normalize_idx(value):
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip()
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(int(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y", "prefix", "one_prefix"}:
            return True
        if text in {"0", "false", "f", "no", "n", "non_prefix", "not_prefix"}:
            return False
    return None


def _extract_prefix_label(ex):
    containers = [ex]
    meta = ex.get("meta")
    if isinstance(meta, dict):
        containers.append(meta)
    labels = ex.get("labels")
    if isinstance(labels, dict):
        containers.append(labels)
    eval_info = ex.get("evaluation")
    if isinstance(eval_info, dict):
        containers.append(eval_info)

    for container in containers:
        for key in PREFIX_LABEL_KEYS:
            if key in container:
                label = _coerce_bool(container[key])
                if label is not None:
                    return label

    if isinstance(labels, (list, tuple, set)):
        lowered = {str(v).strip().lower() for v in labels}
        if "prefix" in lowered:
            return True
        if "non_prefix" in lowered:
            return False

    methods = ex.get("evaluation_methods")
    if isinstance(methods, (list, tuple, set)):
        lowered = {str(v).strip().lower() for v in methods}
        if "prefix" in lowered:
            return True
        if "full_sentence" in lowered:
            return False

    return None


def load_prefix_label_map(path):
    if not path:
        return {}, {"file": None, "rows": 0, "labeled_rows": 0}
    if not os.path.exists(path):
        print(f"[warn] BLiMP original file not found: {path}; continuing without labels")
        return {}, {"file": path, "rows": 0, "labeled_rows": 0}

    labels = {}
    rows = 0
    labeled_rows = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows += 1
            ex = json.loads(line)
            label = _extract_prefix_label(ex)
            if label is None:
                continue
            key = (
                ex.get("phenomenon"),
                ex.get("subtask"),
                _normalize_idx(ex.get("idx")),
            )
            labels[key] = label
            labeled_rows += 1
    return labels, {"file": path, "rows": rows, "labeled_rows": labeled_rows}


def _resolve_regime_path(regime_or_path):
    if os.path.exists(regime_or_path):
        return regime_or_path
    if any(ch in regime_or_path for ch in "*?[]"):
        matches = sorted(glob.glob(regime_or_path))
        if matches:
            return matches[0]
        return regime_or_path

    if regime_or_path in DEFAULT_FILES:
        for candidate in DEFAULT_FILES[regime_or_path]:
            if os.path.exists(candidate):
                return candidate
            if any(ch in candidate for ch in "*?[]"):
                matches = sorted(glob.glob(candidate))
                if matches:
                    return matches[0]
    return regime_or_path


def load_freqblimp_file(path, regime, prefix_label_map):
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
            good = ex.get("good_freq") or ex.get("good_rare") or ex.get("good")
            bad = ex.get("bad_freq") or ex.get("bad_rare") or ex.get("bad")
            if not good or not bad:
                continue

            key = (
                ex.get("phenomenon") or ex.get("group"),
                ex.get("subtask"),
                _normalize_idx(ex.get("idx")),
            )
            prefix_label = prefix_label_map.get(key)
            examples.append(
                {
                    "regime": regime,
                    "group": ex.get("group"),
                    "phenomenon": ex.get("phenomenon") or ex.get("group"),
                    "subtask": ex.get("subtask"),
                    "idx": ex.get("idx"),
                    "good": good,
                    "bad": bad,
                    "prefix_eval_label": prefix_label,
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


def load_dataset(regimes, max_pairs, seed, prefix_label_map):
    rng = random.Random(seed)
    all_examples = []
    by_regime = {}
    for regime in regimes:
        path = _resolve_regime_path(regime)
        exs = load_freqblimp_file(path, regime, prefix_label_map)
        if exs:
            by_regime[regime] = exs
            all_examples.extend(exs)
        else:
            print(f"[warn] No usable examples for regime '{regime}' from {path}")
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


def _tokenize_cached(tokenizer, sentence):
    encoded = tokenizer(sentence, padding=False, return_attention_mask=True)
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded.get("attention_mask"),
    }


def _single_token_diff_prefix(good_ids, bad_ids, tokenizer):
    if len(good_ids) != len(bad_ids):
        return None, "length_mismatch"
    differing = [i for i, (g, b) in enumerate(zip(good_ids, bad_ids)) if g != b]
    if len(differing) != 1:
        return None, f"diff_count_{len(differing)}"
    diff_idx = differing[0]
    prefix_ids = list(good_ids[:diff_idx])
    prefix_added_bos = False
    if not prefix_ids:
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            bos_id = tokenizer.eos_token_id
        if bos_id is None:
            return None, "empty_prefix_without_bos"
        prefix_ids = [bos_id]
        prefix_added_bos = True
    return (
        {
            "diff_index": diff_idx,
            "good_token": int(good_ids[diff_idx]),
            "bad_token": int(bad_ids[diff_idx]),
            "prefix_input_ids": prefix_ids,
            "prefix_attention_mask": [1] * len(prefix_ids),
            "prefix_added_bos": prefix_added_bos,
        },
        None,
    )


def prepare_prefix_examples(examples, tokenizer, prefix_mode):
    stats = Counter()
    prepared = []
    for ex in examples:
        good_encoded = ex.get("_tokens", {}).get("good")
        bad_encoded = ex.get("_tokens", {}).get("bad")
        if good_encoded is None:
            good_encoded = _tokenize_cached(tokenizer, ex["good"])
            ex.setdefault("_tokens", {})["good"] = good_encoded
        if bad_encoded is None:
            bad_encoded = _tokenize_cached(tokenizer, ex["bad"])
            ex.setdefault("_tokens", {})["bad"] = bad_encoded

        label = ex.get("prefix_eval_label")
        prefix_meta, reason = _single_token_diff_prefix(
            good_encoded["input_ids"], bad_encoded["input_ids"], tokenizer
        )

        keep = False
        if prefix_mode == "label":
            if label is True:
                keep = True
                stats["label_true"] += 1
            elif label is False:
                stats["label_false"] += 1
            else:
                stats["label_missing"] += 1
        elif prefix_mode == "off":
            keep = prefix_meta is not None
            if keep:
                stats["token_single_diff"] += 1
        else:
            # auto: use labels when present; fallback to token-based criterion when missing.
            if label is True:
                keep = True
                stats["label_true"] += 1
            elif label is False:
                stats["label_false"] += 1
            else:
                keep = prefix_meta is not None
                if keep:
                    stats["label_missing_fallback"] += 1
                else:
                    stats["label_missing_unusable"] += 1

        if not keep:
            continue
        if prefix_meta is None:
            stats[f"skip_{reason}"] += 1
            continue

        ex["_prefix"] = prefix_meta
        prepared.append(ex)
        stats["kept"] += 1

    return prepared, dict(stats)


class _IntegratedGradientIntervention:
    """
    Temporarily overrides OpenSAE intervention logic so we can inject an
    explicit sparse-activation tensor along the IG interpolation path.
    """

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self._orig_apply = None
        self._orig_intervention = None
        self._orig_indices = None

    def __enter__(self):
        self._orig_apply = self.wrapper._apply_intervention
        self._orig_intervention = self.wrapper.intervention_config.intervention
        self._orig_indices = self.wrapper.intervention_config.intervention_indices

        def _patched_apply(this):
            override = getattr(this, "_ig_override_sparse_acts", None)
            if override is not None:
                this.encoder_output.sparse_feature_activations = override

        self.wrapper._apply_intervention = types.MethodType(_patched_apply, self.wrapper)
        self.wrapper.intervention_config.intervention = True
        if self.wrapper.intervention_config.intervention_indices is None:
            self.wrapper.intervention_config.intervention_indices = [0]
        self.wrapper._ig_override_sparse_acts = None
        return self

    def __exit__(self, exc_type, exc, tb):
        self.wrapper._apply_intervention = self._orig_apply
        self.wrapper.intervention_config.intervention = self._orig_intervention
        self.wrapper.intervention_config.intervention_indices = self._orig_indices
        if hasattr(self.wrapper, "_ig_override_sparse_acts"):
            delattr(self.wrapper, "_ig_override_sparse_acts")


def score_pair(wrapper, tokenizer, example, device, include_special, ig_steps):
    prefix = example["_prefix"]
    prefix_ids = prefix["prefix_input_ids"]
    prefix_mask = prefix["prefix_attention_mask"]
    good_tok = prefix["good_token"]
    bad_tok = prefix["bad_token"]
    special_ids = set(tokenizer.all_special_ids or [])

    if not include_special and (good_tok in special_ids or bad_tok in special_ids):
        raise RuntimeError("Target token is special and include_special_tokens is disabled.")

    input_ids = torch.tensor([prefix_ids], device=device, dtype=torch.long)
    attention_mask = torch.tensor([prefix_mask], device=device, dtype=torch.long)

    with _IntegratedGradientIntervention(wrapper):
        wrapper.clear_intermediates()
        wrapper.transformer.zero_grad(set_to_none=True)
        wrapper.sae.zero_grad(set_to_none=True)
        outputs = wrapper.transformer(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        feats = wrapper.saved_features
        if feats is None:
            raise RuntimeError("No SAE features captured; check decoder_impl and hookpoints.")

        clean_idx = feats.sparse_feature_indices.detach()
        clean_act = feats.sparse_feature_activations.detach()
        if clean_act.dim() == 3 and clean_act.shape[0] == 1:
            clean_act = clean_act.squeeze(0)
            clean_idx = clean_idx.squeeze(0)

        logits = outputs.logits[:, -1, :]
        logit_good = logits[0, good_tok].detach()
        logit_bad = logits[0, bad_tok].detach()
        logit_diff = (logit_good - logit_bad).detach()

        grad_sum = torch.zeros_like(clean_act)
        for step in range(1, ig_steps + 1):
            alpha = float(step) / float(ig_steps)
            override_act = (clean_act * alpha).detach().requires_grad_(True)

            wrapper.clear_intermediates()
            wrapper.transformer.zero_grad(set_to_none=True)
            wrapper.sae.zero_grad(set_to_none=True)
            wrapper._ig_override_sparse_acts = override_act

            step_outputs = wrapper.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            step_logits = step_outputs.logits[:, -1, :]
            metric = step_logits[0, good_tok] - step_logits[0, bad_tok]
            metric.backward()
            if override_act.grad is None:
                raise RuntimeError("Missing gradient for IG override activation.")
            grad_sum = grad_sum + override_act.grad.detach()

        avg_grad = grad_sum / float(ig_steps)
        contributions = (clean_act * avg_grad).detach().cpu()
        feature_idx = clean_idx.detach().cpu()

    scores = defaultdict(float)
    features_seen = set()
    activation_counts = Counter()
    for pos, token_id in enumerate(prefix_ids):
        if not include_special and token_id in special_ids:
            continue
        feats_pos = feature_idx[pos].tolist()
        vals_pos = contributions[pos].tolist()
        for feat, value in zip(feats_pos, vals_pos):
            scores[feat] += value
            features_seen.add(feat)
            activation_counts[feat] += 1

    return (
        scores,
        features_seen,
        activation_counts,
        float(logit_good.item()),
        float(logit_bad.item()),
        float(logit_diff.item()),
    )


def build_group_topk(score_sums, feature_counts, group_size, activation_counts, topk):
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
                "activation_count": activation_counts.get(feat, 0),
                "example_count": count,
            }
        )
    results.sort(key=lambda x: abs(x["mean_score"]), reverse=True)
    if topk and topk > 0:
        return results[:topk]
    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Integrated-grad attribution on OpenSAE latents for freqBLiMP using "
            "single-token prefix logit differences."
        )
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
        "--blimp_original_file",
        type=str,
        default="data/freqBLiMP/blimp_original.jsonl",
        help="Original BLiMP JSONL used to inherit prefix-eval labels.",
    )
    parser.add_argument(
        "--prefix_mode",
        type=str,
        default="auto",
        choices=["auto", "label", "off"],
        help=(
            "How to filter to prefix-evaluable pairs: "
            "auto=use labels when present else fallback to token single-diff; "
            "label=require label==True; off=ignore labels and use token single-diff."
        ),
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
        "--cache_tokens",
        action="store_true",
        help="Pre-tokenize examples to reduce tokenizer overhead.",
    )
    parser.add_argument(
        "--ig_steps",
        type=int,
        default=16,
        help="Integrated gradients interpolation steps.",
    )
    parser.add_argument(
        "--include_special_tokens",
        action="store_true",
        help="Include special tokens in attribution sums and target-token eligibility.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/attribution",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--output_tag",
        type=str,
        default="prefix_ig",
        help="Tag inserted into output filename.",
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
    if args.ig_steps < 1:
        raise RuntimeError("--ig_steps must be >= 1")

    prefix_label_map, label_stats = load_prefix_label_map(args.blimp_original_file)
    print(
        "Prefix labels loaded:",
        f"{label_stats['labeled_rows']}/{label_stats['rows']}",
        f"(source: {label_stats['file']})",
    )

    raw_examples = load_dataset(regimes, args.max_pairs, args.seed, prefix_label_map)
    print(f"Raw examples loaded: {len(raw_examples)}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading base model...")
    base_model, tokenizer = build_base_model_and_tokenizer(args.model_name, device, dtype)
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN is empty; gated models may fail to load.")

    if args.cache_tokens:
        print("Caching tokenization...")
        start_cache = time.time()
        for i, ex in enumerate(raw_examples, start=1):
            cached = {}
            for key in ("good", "bad"):
                enc = tokenizer(ex[key], padding=False, return_attention_mask=True)
                cached[key] = {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc.get("attention_mask"),
                }
            ex["_tokens"] = cached
            if args.log_every and i % args.log_every == 0:
                elapsed = time.time() - start_cache
                rate = i / max(elapsed, 1e-6)
                remaining = max(len(raw_examples) - i, 0)
                eta = remaining / max(rate, 1e-6)
                print(
                    f"[cache] {i}/{len(raw_examples)} examples "
                    f"({rate:.2f} ex/s, ETA {eta/60:.1f}m)"
                )
        elapsed = time.time() - start_cache
        print(f"[cache] done in {elapsed/60:.1f}m")

    examples, prefix_filter_stats = prepare_prefix_examples(raw_examples, tokenizer, args.prefix_mode)
    if not examples:
        raise RuntimeError(
            "No prefix-evaluable examples after filtering. "
            "Try --prefix_mode off or verify BLiMP labels/tokenization."
        )
    print(f"Prefix-evaluable examples kept: {len(examples)}")
    print(f"Prefix filter stats: {prefix_filter_stats}")

    regime_counts = Counter(ex["regime"] for ex in examples)
    phenomenon_counts = Counter(ex.get("phenomenon") or "unknown" for ex in examples)
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")

    for layer in layers:
        sae_path = args.sae_path_template.format(layer=layer)
        print(f"\n[Layer {layer}] Loading SAE: {sae_path}")
        wrapper = build_wrapper(base_model, sae_path, device, args.sae_dtype)

        score_sums = defaultdict(float)
        feature_example_counts = Counter()
        activation_counts = Counter()
        phenomenon_score_sums = defaultdict(lambda: defaultdict(float))
        phenomenon_feature_counts = defaultdict(Counter)
        phenomenon_activation_counts = defaultdict(Counter)
        regime_score_sums = defaultdict(lambda: defaultdict(float))
        regime_feature_counts = defaultdict(Counter)
        regime_activation_counts = defaultdict(Counter)

        total_logit_good = 0.0
        total_logit_bad = 0.0
        total_logit_diff = 0.0
        start = time.time()

        for i, ex in enumerate(examples, start=1):
            (
                scores,
                features_seen,
                activation_counts_pair,
                logit_good,
                logit_bad,
                logit_diff,
            ) = score_pair(
                wrapper,
                tokenizer,
                ex,
                device,
                args.include_special_tokens,
                args.ig_steps,
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
            activation_counts.update(activation_counts_pair)
            phenomenon_activation_counts[phenomenon].update(activation_counts_pair)
            regime_activation_counts[regime].update(activation_counts_pair)

            total_logit_good += logit_good
            total_logit_bad += logit_bad
            total_logit_diff += logit_diff

            if args.log_every and i % args.log_every == 0:
                elapsed = time.time() - start
                rate = i / max(elapsed, 1e-6)
                remaining = max(len(examples) - i, 0)
                eta = remaining / max(rate, 1e-6)
                print(
                    f"[Layer {layer}] {i}/{len(examples)} examples "
                    f"({rate:.2f} ex/s, ETA {eta/60:.1f}m, avg logit_diff {total_logit_diff / i:.4f})"
                )

        num_examples = len(examples)
        results = []
        for feat, total in score_sums.items():
            mean_score = total / num_examples
            mean_active = total / feature_example_counts[feat] if feature_example_counts[feat] else 0.0
            results.append(
                {
                    "feature": feat,
                    "mean_score": mean_score,
                    "mean_score_active": mean_active,
                    "activation_count": activation_counts.get(feat, 0),
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
                    phenomenon_activation_counts[phenomenon],
                    args.phenomenon_topk,
                )

        regime_topk = {}
        if args.regime_topk and args.regime_topk > 0:
            for regime, sums in regime_score_sums.items():
                regime_topk[regime] = build_group_topk(
                    sums,
                    regime_feature_counts[regime],
                    regime_counts.get(regime, 0),
                    regime_activation_counts[regime],
                    args.regime_topk,
                )

        summary = {
            "layer": layer,
            "sae_path": sae_path,
            "model_name": args.model_name,
            "model_dtype": str(dtype).replace("torch.", ""),
            "sae_dtype": args.sae_dtype,
            "num_examples": num_examples,
            "regime_counts": dict(regime_counts),
            "phenomenon_counts": dict(phenomenon_counts),
            "avg_logit_good": total_logit_good / num_examples,
            "avg_logit_bad": total_logit_bad / num_examples,
            "avg_logit_diff": total_logit_diff / num_examples,
            "avg_delta": total_logit_diff / num_examples,
            "metric": "logit(good_token) - logit(bad_token) on shared prefix",
            "attribution_method": "integrated_gradients_over_sparse_activations",
            "ig_steps": args.ig_steps,
            "prefix_mode": args.prefix_mode,
            "prefix_filter_stats": prefix_filter_stats,
            "prefix_label_file": label_stats["file"],
            "prefix_labels_found": label_stats["labeled_rows"],
            "prefix_label_rows_total": label_stats["rows"],
            "phenomenon_topk": args.phenomenon_topk,
            "regime_topk": args.regime_topk,
            "run_timestamp": run_timestamp,
        }

        out_name = (
            f"freqblimp_{args.output_tag}_attribution_layer{layer:02d}_{run_timestamp}.json"
        )
        out_path = os.path.join(args.output_dir, out_name)
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
