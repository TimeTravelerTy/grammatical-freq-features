import argparse
import json
import os

import torch
from nnsight import LanguageModel

from src.config import HF_TOKEN
from src.utils import setup_autoencoder


TRACER_KWARGS = {'scan': False, 'validate': False}


def load_freqblimp_examples(path, n):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if len(examples) >= n:
                break
    return examples


def summarize_concepts(concepts_path):
    if not os.path.exists(concepts_path):
        return None
    with open(concepts_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)
    total_values = sum(len(v) for v in concepts.values())
    return {"num_features": len(concepts), "num_values": total_values}


def load_concept_feature_indices(features_dir, concept_key, concept_value, language, k):
    if not concept_key or not concept_value:
        return None
    feature_path = os.path.join(features_dir, f"{concept_key}_{concept_value}.json")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Missing feature file: {feature_path}")
    with open(feature_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if language not in data:
        raise KeyError(f"Language '{language}' not found in {feature_path}")
    top_features = [feature for feature, _ in data[language]["top_1_percent"]]
    if not top_features:
        return None
    k = min(k, len(top_features))
    return top_features[:k]


def parse_variants(args):
    if args.variants:
        return [v.strip() for v in args.variants.split(",") if v.strip()]
    return [args.variant]


def main():
    parser = argparse.ArgumentParser(description="Sanity check for model+SAE on freqBLiMP.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path (e.g., meta-llama/Meta-Llama-3-8B)")
    parser.add_argument("--freqblimp_file", type=str, default="data/freqBLiMP/freqBLiMP_xtail.jsonl")
    parser.add_argument("--concepts_file", type=str, default="data/concepts.json")
    parser.add_argument("--autoencoder_path", type=str, default="autoencoders/llama-3-8b-layer16.pt")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or auto/cuda")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--num_examples", type=int, default=2)
    parser.add_argument("--variant", type=str, default="good_original",
                        choices=["good_original", "bad_original", "good_rare", "bad_rare"])
    parser.add_argument("--variants", type=str, default=None,
                        help="Comma-separated list of variants (overrides --variant)")
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--concept_key", type=str, default=None)
    parser.add_argument("--concept_value", type=str, default=None)
    parser.add_argument("--language", type=str, default="English")
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

    torch_dtype = torch.float32 if args.device == "cpu" else torch.float16
    model = LanguageModel(args.model_name, torch_dtype=torch_dtype, device_map=args.device, token=HF_TOKEN)
    submodule = model.model.layers[args.layer]
    autoencoder = setup_autoencoder(checkpoint_path=args.autoencoder_path, device=args.device)

    concept_features = load_concept_feature_indices(
        args.features_dir, args.concept_key, args.concept_value, args.language, args.topk
    )

    for ex in examples:
        header = f"[{ex.get('group')}:{ex.get('subtask')}] idx={ex.get('idx')}"
        print(f"\n{header}")
        for variant in variants:
            sentence = ex.get(variant)
            if not sentence:
                continue
            tokens = model.tokenizer(sentence, return_tensors="pt", padding=False).input_ids
            if hasattr(model, "device"):
                tokens = tokens.to(model.device)
            with model.trace(tokens, **TRACER_KWARGS), torch.no_grad():
                internal_acts = submodule.output[0].save()
            acts = internal_acts[0].to(autoencoder.device)
            feats = autoencoder.encode(acts)
            feat_max = feats.max(dim=0).values.detach().cpu()

            if concept_features:
                values = [(idx, feat_max[idx].item()) for idx in concept_features]
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
