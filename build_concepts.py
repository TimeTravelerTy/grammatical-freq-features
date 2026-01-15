import argparse
import json
import os
from collections import Counter, defaultdict

import pyconll


def extract_feature_counts(conll_file):
    counts = defaultdict(Counter)
    data = pyconll.load_from_file(conll_file)
    for sentence in data:
        for token in sentence:
            for feat, values in token.feats.items():
                for value in values:
                    counts[feat][value] += 1
    return counts


def filter_concepts(counts, min_count):
    concepts = {}
    for feat, counter in counts.items():
        values = [value for value, count in counter.items() if count >= min_count]
        if values:
            concepts[feat] = sorted(values)
    return concepts


def main():
    parser = argparse.ArgumentParser(description="Build a concept list from a UD .conllu file.")
    parser.add_argument("--ud_file", type=str, required=True, help="Path to a UD .conllu file (e.g., en_ewt-ud-train.conllu)")
    parser.add_argument("--min_count", type=int, default=1, help="Minimum count for a feature value to be included")
    parser.add_argument("--out", type=str, default="data/concepts.json", help="Output path for concept list JSON")
    parser.add_argument("--counts_out", type=str, default=None, help="Optional output path for feature/value counts")
    args = parser.parse_args()

    counts = extract_feature_counts(args.ud_file)
    concepts = filter_concepts(counts, args.min_count)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(concepts, f, indent=2)

    if args.counts_out:
        counts_serializable = {feat: dict(counter) for feat, counter in counts.items()}
        os.makedirs(os.path.dirname(args.counts_out), exist_ok=True)
        with open(args.counts_out, "w") as f:
            json.dump(counts_serializable, f, indent=2)

    print(f"Wrote concepts to {args.out}")
    if args.counts_out:
        print(f"Wrote counts to {args.counts_out}")


if __name__ == "__main__":
    main()
