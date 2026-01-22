import argparse
import heapq
import json
import os
import re
import time
from itertools import count

import torch
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import HF_TOKEN
from src.probing.sae_loader import load_local_sae
from src.utils import setup_autoencoder


TRACER_KWARGS = {'scan': False, 'validate': False}


def _has_meta_params(module):
    for param in module.parameters():
        if param.device.type == "meta":
            return True
    return False


def _build_model_with_transformers(model_name, device_map, dtype):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        token=HF_TOKEN,
    )
    tokenizer.pad_token = tokenizer.eos_token
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=False,
        token=HF_TOKEN,
    )
    if device_map not in ["auto", None]:
        hf_model.to(device_map)
    hf_model.eval()
    return LanguageModel(hf_model, tokenizer=tokenizer)


def resolve_submodule_device(model, submodule, fallback=None):
    for param in submodule.parameters():
        if param.device.type != "meta":
            return param.device
    model_root = getattr(model, "model", model)
    for param in model_root.parameters():
        if param.device.type != "meta":
            return param.device
    if isinstance(fallback, torch.device):
        return fallback
    if isinstance(fallback, str) and (fallback == "cpu" or fallback.startswith("cuda")):
        return torch.device(fallback)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_freqblimp_examples(path, n):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if n is not None and n > 0 and len(examples) >= n:
                break
    return examples


def summarize_concepts(concepts_path):
    if not os.path.exists(concepts_path):
        return None
    with open(concepts_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)
    total_values = sum(len(v) for v in concepts.values())
    return {"num_features": len(concepts), "num_values": total_values}


def load_concept_feature_indices(features_dir, concept_key, concept_value, k):
    if not concept_key or not concept_value:
        return None
    feature_path = os.path.join(features_dir, f"{concept_key}_{concept_value}.json")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Missing feature file: {feature_path}")
    with open(feature_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "top_1_percent" in data:
        top_features = [feature for feature, _ in data["top_1_percent"]]
    elif len(data) == 1:
        entry = next(iter(data.values()))
        if "top_1_percent" not in entry:
            raise KeyError(f"Missing top_1_percent in {feature_path}")
        top_features = [feature for feature, _ in entry["top_1_percent"]]
    else:
        raise KeyError(f"Unexpected feature file format: {feature_path}")
    if not top_features:
        return None
    k = min(k, len(top_features))
    return top_features[:k]


def parse_variants(args):
    if args.variants:
        return [v.strip() for v in args.variants.split(",") if v.strip()]
    return [args.variant]


def _extract_layer_index(hook_name):
    if not hook_name:
        return None
    match = re.search(r"(?:layers|blocks)\.(\d+)", hook_name)
    if match:
        return int(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(description="Sanity check for model+SAE on freqBLiMP.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path (e.g., meta-llama/Meta-Llama-3-8B)")
    parser.add_argument("--freqblimp_file", type=str, default="data/freqBLiMP/freqBLiMP_xtail.jsonl")
    parser.add_argument("--concepts_file", type=str, default="data/concepts.json")
    parser.add_argument("--autoencoder_path", type=str, default="autoencoders/llama-3-8b-layer16.pt")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu/auto (cuda default enforces GPU)")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--num_examples", type=int, default=2)
    parser.add_argument("--variant", type=str, default="good_original",
                        choices=["good_original", "bad_original", "good_rare", "bad_rare"])
    parser.add_argument("--variants", type=str, default=None,
                        help="Comma-separated list of variants (overrides --variant)")
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--top_sentences", type=int, default=5,
                        help="Top sentences to keep per feature (0 disables ranking mode).")
    parser.add_argument("--include_special_tokens", action="store_true",
                        help="Include special tokens when locating max activation.")
    parser.add_argument("--log_every", type=int, default=100,
                        help="Log progress every N examples (0 disables).")
    parser.add_argument("--concept_key", type=str, default=None)
    parser.add_argument("--concept_value", type=str, default=None)
    parser.add_argument("--features_dir", type=str, default="outputs/probing/features/llama")
    parser.add_argument("--skip_model", action="store_true")
    args = parser.parse_args()

    concept_summary = summarize_concepts(args.concepts_file)
    if concept_summary:
        print(f"Concepts loaded: {concept_summary['num_features']} features, {concept_summary['num_values']} values")
    else:
        print("No concepts file found yet (data/concepts.json).")

    examples = load_freqblimp_examples(args.freqblimp_file, args.num_examples)
    if not examples:
        raise RuntimeError(f"No examples found in {args.freqblimp_file}")

    variants = parse_variants(args)

    if args.skip_model:
        for ex in examples:
            header = f"[{ex.get('group')}:{ex.get('subtask')}] idx={ex.get('idx')}"
            print(header)
            for variant in variants:
                print(f"  {variant}: {ex.get(variant)}")
        return

    if not HF_TOKEN:
        print("WARNING: HF_TOKEN is empty; you may need to set it in src/config.py or via env.")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; pass --device cpu or use a GPU node.")

    dtype = torch.float32 if args.device == "cpu" else torch.float16
    model = None
    model_device_map = None if args.device in ["cuda", "cuda:0", "cpu"] else args.device
    try:
        model = LanguageModel(
            args.model_name,
            dtype=dtype,
            device_map=model_device_map,
            token=HF_TOKEN,
            low_cpu_mem_usage=False if model_device_map is None else True,
        )
        if model_device_map is None:
            try:
                model.to(args.device)
            except Exception:
                pass
        if _has_meta_params(model.model):
            model = _build_model_with_transformers(args.model_name, args.device, dtype)
    except Exception:
        model = _build_model_with_transformers(args.model_name, args.device, dtype)
    if _has_meta_params(model.model):
        raise RuntimeError("Model parameters are still on meta device after loading; check device/device_map.")

    use_local_sae = os.path.isdir(args.autoencoder_path)
    if use_local_sae:
        autoencoder = load_local_sae(args.autoencoder_path, device=args.device)
        hook_name = getattr(getattr(autoencoder, "cfg", None), "hook_name", None)
        layer_index = _extract_layer_index(hook_name) or args.layer
    else:
        layer_index = args.layer
    submodule = model.model.layers[layer_index]
    ae_device = resolve_submodule_device(model, submodule, fallback=args.device)
    if use_local_sae:
        if hasattr(autoencoder, "to"):
            autoencoder = autoencoder.to(ae_device)
    else:
        autoencoder = setup_autoencoder(checkpoint_path=args.autoencoder_path, device=ae_device)

    def _dev(mod):
        try:
            return next(mod.parameters()).device
        except Exception:
            return None

    print("CUDA available:", torch.cuda.is_available())
    print("Model param device:", _dev(model.model))
    print("Submodule param device:", _dev(submodule))
    print("AE param device:", _dev(autoencoder))
    print(f"Examples loaded: {len(examples)}")
    print(f"Variants: {', '.join(variants)}")

    concept_features = load_concept_feature_indices(
        args.features_dir, args.concept_key, args.concept_value, args.topk
    )
    if concept_features:
        concept_features = [idx for idx in concept_features if idx < autoencoder.dict_size]
        if not concept_features:
            raise RuntimeError("No concept features fall within the autoencoder dictionary size.")

    if args.top_sentences > 0:
        top_n = args.top_sentences
        include_special_tokens = args.include_special_tokens
        special_ids = set(model.tokenizer.all_special_ids)
        heaps = {idx: [] for idx in concept_features} if concept_features else {}
        counter = count()
        start = time.time()
        last_log = start

        def push_top(heap, score, meta, limit):
            if limit <= 0:
                return
            entry = (score, next(counter), meta)
            if len(heap) < limit:
                heapq.heappush(heap, entry)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, entry)

        for ex_i, ex in enumerate(examples, start=1):
            if args.log_every and ex_i % args.log_every == 0:
                now = time.time()
                elapsed = now - last_log
                total = now - start
                print(f"Processed {ex_i}/{len(examples)} examples (+{elapsed:.1f}s, total {total:.1f}s)")
                last_log = now
            for variant in variants:
                sentence = ex.get(variant)
                if not sentence:
                    continue
                tokenized = model.tokenizer(sentence, return_tensors="pt", padding=False)
                input_ids = tokenized["input_ids"]
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                token_ids = input_ids[0].tolist()
                token_strings = model.tokenizer.convert_ids_to_tokens(token_ids)
                if hasattr(model, "device"):
                    input_ids = input_ids.to(model.device)
                with model.trace(input_ids, **TRACER_KWARGS), torch.no_grad():
                    internal_acts = submodule.output[0].save()
                acts = internal_acts
                if hasattr(acts, "dim"):
                    if acts.dim() == 3:
                        acts = acts[0]
                    elif acts.dim() != 2:
                        raise RuntimeError(f"Unexpected activations shape: {tuple(acts.shape)}")
                ae_device = next(autoencoder.parameters()).device
                acts = acts.to(ae_device)
                feats = autoencoder.encode(acts)
                if feats.numel() == 0:
                    continue
                if feats.dim() == 1:
                    feats = feats.unsqueeze(0)
                if feats.dim() != 2:
                    continue

                if not include_special_tokens and special_ids:
                    mask = torch.tensor([tid in special_ids for tid in token_ids], device=feats.device)
                    if mask.any():
                        feats = feats.clone()
                        feats[mask, :] = float("-inf")

                if concept_features:
                    vals = feats[:, concept_features]
                    max_vals, max_pos = vals.max(dim=0)
                    max_vals = max_vals.detach().cpu().tolist()
                    max_pos = max_pos.detach().cpu().tolist()
                    feature_ids = concept_features
                else:
                    feat_max = feats.max(dim=0).values
                    if feat_max.numel() == 0:
                        continue
                    k = min(args.topk, feat_max.numel())
                    max_vals, feature_ids = torch.topk(feat_max, k)
                    feature_ids = feature_ids.detach().cpu().tolist()
                    max_vals = max_vals.detach().cpu().tolist()
                    vals = feats[:, feature_ids]
                    _, max_pos_tensor = vals.max(dim=0)
                    max_pos = max_pos_tensor.detach().cpu().tolist()

                for feat_idx, val, pos in zip(feature_ids, max_vals, max_pos):
                    if val == float("-inf"):
                        continue
                    token = token_strings[pos] if 0 <= pos < len(token_strings) else None
                    meta = {
                        "sentence": sentence,
                        "variant": variant,
                        "group": ex.get("group"),
                        "subtask": ex.get("subtask"),
                        "idx": ex.get("idx"),
                        "token": token,
                        "token_pos": pos,
                    }
                    heap = heaps.setdefault(feat_idx, [])
                    push_top(heap, val, meta, top_n)

        print(f"\nTop {top_n} sentences per feature:")
        for feat_idx in sorted(heaps.keys()):
            print(f"\nFeature {feat_idx}:")
            heap = heaps.get(feat_idx, [])
            if not heap:
                print("  (no matches)")
                continue
            for score, _, meta in sorted(heap, key=lambda x: x[0], reverse=True):
                token = meta.get("token")
                token_pos = meta.get("token_pos")
                header = f"[{meta.get('group')}:{meta.get('subtask')}] idx={meta.get('idx')} {meta.get('variant')}"
                print(f"  {score:.4f} {header} token={token} pos={token_pos}")
                print(f"    {meta.get('sentence')}")
        return

    for ex in examples:
        header = f"[{ex.get('group')}:{ex.get('subtask')}] idx={ex.get('idx')}"
        print(f"\n{header}")
        for variant in variants:
            sentence = ex.get(variant)
            if not sentence:
                continue
            tokens = model.tokenizer(sentence, return_tensors="pt", padding=False).input_ids
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            if hasattr(model, "device"):
                tokens = tokens.to(model.device)
            with model.trace(tokens, **TRACER_KWARGS), torch.no_grad():
                internal_acts = submodule.output[0].save()
            acts = internal_acts
            if hasattr(acts, "dim"):
                if acts.dim() == 3:
                    acts = acts[0]
                elif acts.dim() != 2:
                    raise RuntimeError(f"Unexpected activations shape: {tuple(acts.shape)}")
            ae_device = next(autoencoder.parameters()).device
            acts = acts.to(ae_device)
            feats = autoencoder.encode(acts)
            if feats.dim() == 1:
                feat_max = feats.detach().cpu()
            else:
                feat_max = feats.max(dim=0).values.detach().cpu()
            if feat_max.numel() == 0:
                print("    WARNING: empty feature vector; check autoencoder checkpoint/device.")
                continue

            if concept_features:
                valid = [idx for idx in concept_features if idx < feat_max.numel()]
                if not valid:
                    print("    WARNING: no concept features within autoencoder dictionary size.")
                    continue
                values = [(idx, feat_max[idx].item()) for idx in valid]
                values.sort(key=lambda x: x[1], reverse=True)
                k = min(args.topk, len(values))
                print(f"  {variant}: {sentence}")
                print(f"    Concept features (top {k} of {len(values)}):")
                for idx, val in values[:k]:
                    print(f"      {idx}: {val:.4f}")
            else:
                k = min(args.topk, feat_max.numel())
                top_vals, top_idx = torch.topk(feat_max, k)
                print(f"  {variant}: {sentence}")
                print(f"    Top {k} features (index, activation):")
                for idx, val in zip(top_idx.tolist(), top_vals.tolist()):
                    print(f"      {idx}: {val:.4f}")


if __name__ == "__main__":
    main()
