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
    parser.add_argument("--topk", type=int, default=32)
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

    if args.skip_model:
        for ex in examples:
            print(f"[{ex.get('group')}:{ex.get('subtask')}] {ex.get(args.variant)}")
        return

    if not HF_TOKEN:
        print("WARNING: HF_TOKEN is empty; you may need to set it in src/config.py or via env.")

    torch_dtype = torch.float32 if args.device == "cpu" else torch.float16
    model = LanguageModel(args.model_name, torch_dtype=torch_dtype, device_map=args.device, token=HF_TOKEN)
    submodule = model.model.layers[args.layer]
    autoencoder = setup_autoencoder(checkpoint_path=args.autoencoder_path, device=args.device)

    for ex in examples:
        sentence = ex.get(args.variant)
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
        k = min(args.topk, feat_max.numel())
        top_vals, top_idx = torch.topk(feat_max, k)

        print(f"\nSentence: {sentence}")
        print(f"Top {k} features (index, activation):")
        for idx, val in zip(top_idx.tolist(), top_vals.tolist()):
            print(f"  {idx}: {val:.4f}")


if __name__ == "__main__":
    main()
