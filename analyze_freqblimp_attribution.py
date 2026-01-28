import argparse
import csv
import glob
import json
import math
import os
from collections import OrderedDict

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None


DEFAULT_PATTERN = "freqblimp_rare_attribution_layer*.json"


def _parse_topk_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return [int(p) for p in parts]


def _layer_from_path(path):
    base = os.path.basename(path)
    digits = "".join(ch for ch in base if ch.isdigit())
    if digits:
        return int(digits)
    return None


def _sorted_features(features):
    return sorted(features, key=lambda x: x.get("mean_score", 0.0), reverse=True)


def _positive_mass(features):
    return [max(0.0, f.get("mean_score", 0.0)) for f in features]


def _concentration_by_topk(sorted_features, topk_list):
    positives = _positive_mass(sorted_features)
    total_positive = sum(positives)
    if total_positive <= 0:
        return total_positive, {k: 0.0 for k in topk_list}
    topk_fractions = {}
    for k in topk_list:
        if k <= 0:
            topk_fractions[k] = 0.0
            continue
        topk_sum = sum(positives[: min(k, len(positives))])
        topk_fractions[k] = topk_sum / total_positive
    return total_positive, topk_fractions


def _hhi_and_effective_count(sorted_features):
    positives = _positive_mass(sorted_features)
    total = sum(positives)
    if total <= 0:
        return 0.0, 0.0
    probs = [v / total for v in positives if v > 0]
    hhi = sum(p * p for p in probs)
    eff = 1.0 / hhi if hhi > 0 else 0.0
    return hhi, eff


def _activation_rate(feature, num_examples):
    if not num_examples:
        return 0.0
    return feature.get("activation_count", 0) / num_examples


def _activation_any_rate(feature, num_examples):
    if not num_examples:
        return 0.0
    return feature.get("example_count", 0) / num_examples


def _passes_activation_filters(feature, num_examples, max_any_rate, max_avg_rate):
    if max_any_rate is not None:
        if _activation_any_rate(feature, num_examples) > max_any_rate:
            return False
    if max_avg_rate is not None:
        if _activation_rate(feature, num_examples) > max_avg_rate:
            return False
    return True


def _regime_overlap(regime_features, k):
    if not regime_features:
        return {}
    regimes = [r for r in ("head", "tail", "xtail") if r in regime_features]
    if len(regimes) < 2:
        return {}
    top_sets = {}
    for regime in regimes:
        feats = regime_features.get(regime, [])
        top = feats[: min(k, len(feats))]
        top_sets[regime] = {f["feature"] for f in top}
    overlaps = {}
    pairs = [("head", "tail"), ("head", "xtail"), ("tail", "xtail")]
    for a, b in pairs:
        if a not in top_sets or b not in top_sets:
            continue
        inter = top_sets[a].intersection(top_sets[b])
        union = top_sets[a].union(top_sets[b])
        overlaps[f"{a}_vs_{b}"] = {
            "intersection": len(inter),
            "union": len(union),
            "jaccard": (len(inter) / len(union)) if union else 0.0,
        }
    if all(r in top_sets for r in ("head", "tail", "xtail")):
        inter_all = top_sets["head"].intersection(top_sets["tail"]).intersection(top_sets["xtail"])
        union_all = top_sets["head"].union(top_sets["tail"]).union(top_sets["xtail"])
        overlaps["head_tail_xtail"] = {
            "intersection": len(inter_all),
            "union": len(union_all),
            "jaccard": (len(inter_all) / len(union_all)) if union_all else 0.0,
        }
    overlaps["_available_k"] = {r: len(s) for r, s in top_sets.items()}
    return overlaps


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Overview analysis for freqBLiMP attribution outputs."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="outputs/attribution",
        help="Directory with freqblimp attribution JSON files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PATTERN,
        help="Glob pattern for attribution JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/attribution/analysis",
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--topk",
        type=str,
        default="10,50,200",
        help="Comma-separated list of top-k cutoffs for concentration.",
    )
    parser.add_argument(
        "--regime_topk",
        type=int,
        default=50,
        help="Top-k cutoff for regime overlap stats.",
    )
    parser.add_argument(
        "--filter_activation_any_rate",
        type=float,
        default=0.99,
        help="Drop features with activation_any_rate above this threshold (set <0 to disable).",
    )
    parser.add_argument(
        "--filter_activation_rate",
        type=float,
        default=None,
        help="Drop features with activation_rate above this threshold (None disables).",
    )
    parser.add_argument(
        "--save_ranked_topk",
        type=int,
        default=200,
        help="Save top-k ranked features per layer to CSV (0 disables).",
    )

    args = parser.parse_args()
    topk_list = _parse_topk_list(args.topk)
    max_any_rate = args.filter_activation_any_rate
    if max_any_rate is not None and max_any_rate < 0:
        max_any_rate = None
    max_avg_rate = args.filter_activation_rate

    pattern = os.path.join(args.input_dir, args.pattern)
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No attribution files found for pattern: {pattern}")

    layer_summaries = []
    top10_rows = []
    overlap_rows = []

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("summary", {})
        layer = summary.get("layer")
        if layer is None:
            layer = _layer_from_path(path)
        features = data.get("features", [])
        if not features:
            continue
        sorted_features = _sorted_features(features)
        num_examples = summary.get("num_examples", 0)
        filtered_features = [
            f
            for f in sorted_features
            if _passes_activation_filters(f, num_examples, max_any_rate, max_avg_rate)
        ]

        total_positive, topk_fractions = _concentration_by_topk(sorted_features, topk_list)
        hhi, eff = _hhi_and_effective_count(sorted_features)
        filtered_positive, filtered_topk_fractions = _concentration_by_topk(
            filtered_features, topk_list
        )
        hhi_filtered, eff_filtered = _hhi_and_effective_count(filtered_features)

        row = OrderedDict()
        row["layer"] = layer
        row["num_examples"] = summary.get("num_examples", 0)
        row["num_features"] = len(sorted_features)
        row["positive_mass"] = total_positive
        row["hhi_positive"] = hhi
        row["effective_positive_features"] = eff
        row["num_features_filtered"] = len(filtered_features)
        row["positive_mass_filtered"] = filtered_positive
        row["hhi_positive_filtered"] = hhi_filtered
        row["effective_positive_features_filtered"] = eff_filtered
        for k in topk_list:
            row[f"top{k}_positive_fraction"] = topk_fractions.get(k, 0.0)
            row[f"top{k}_positive_fraction_filtered"] = filtered_topk_fractions.get(k, 0.0)
        layer_summaries.append(row)

        for rank, feat in enumerate(sorted_features[:10], start=1):
            top10_rows.append(
                {
                    "layer": layer,
                    "rank": rank,
                    "feature": feat.get("feature"),
                    "mean_score": feat.get("mean_score"),
                    "activation_rate": _activation_rate(feat, num_examples),
                    "activation_any_rate": _activation_any_rate(feat, num_examples),
                    "activation_count": feat.get("activation_count", 0),
                    "example_count": feat.get("example_count", 0),
                }
            )

        regime_features = data.get("regime_features", {})
        overlap = _regime_overlap(regime_features, args.regime_topk)
        if overlap:
            base = {
                "layer": layer,
                "regime_topk": args.regime_topk,
            }
            available = overlap.pop("_available_k", {})
            for regime, count in available.items():
                base[f"available_{regime}"] = count
            for key, stats in overlap.items():
                overlap_rows.append(
                    {
                        **base,
                        "pair": key,
                        "intersection": stats.get("intersection", 0),
                        "union": stats.get("union", 0),
                        "jaccard": stats.get("jaccard", 0.0),
                    }
                )

        if args.save_ranked_topk and args.save_ranked_topk > 0:
            topk = min(args.save_ranked_topk, len(sorted_features))
            out_rows = []
            for rank, feat in enumerate(sorted_features[:topk], start=1):
                out_rows.append(
                    {
                        "rank": rank,
                        "feature": feat.get("feature"),
                        "mean_score": feat.get("mean_score"),
                        "mean_score_active": feat.get("mean_score_active"),
                        "activation_count": feat.get("activation_count", 0),
                        "example_count": feat.get("example_count", 0),
                        "activation_any_rate": _activation_any_rate(feat, num_examples),
                    }
                )
            out_path = os.path.join(
                args.output_dir, f"top{topk}_features_layer{int(layer):02d}.csv"
            )
            _write_csv(
                out_path,
                out_rows,
                [
                    "rank",
                    "feature",
                    "mean_score",
                    "mean_score_active",
                    "activation_count",
                    "example_count",
                    "activation_any_rate",
                ],
            )

    layer_summaries.sort(key=lambda x: x["layer"])
    top10_rows.sort(key=lambda x: (x["layer"], x["rank"]))
    overlap_rows.sort(key=lambda x: (x["layer"], x["pair"]))

    os.makedirs(args.output_dir, exist_ok=True)
    _write_csv(
        os.path.join(args.output_dir, "layer_summary.csv"),
        layer_summaries,
        list(layer_summaries[0].keys()),
    )
    _write_csv(
        os.path.join(args.output_dir, "top10_features.csv"),
        top10_rows,
        [
            "layer",
            "rank",
            "feature",
            "mean_score",
            "activation_rate",
            "activation_any_rate",
            "activation_count",
            "example_count",
        ],
    )
    if overlap_rows:
        _write_csv(
            os.path.join(args.output_dir, "regime_overlap_topk.csv"),
            overlap_rows,
            [
                "layer",
                "regime_topk",
                "available_head",
                "available_tail",
                "available_xtail",
                "pair",
                "intersection",
                "union",
                "jaccard",
            ],
        )

    # Plot top-50 concentration by layer (or first available topk)
    top50_key = "top50_positive_fraction"
    if top50_key not in layer_summaries[0]:
        fallback = None
        for k in topk_list:
            cand = f"top{k}_positive_fraction"
            if cand in layer_summaries[0]:
                fallback = cand
                break
        top50_key = fallback

    if top50_key and plt is not None:
        layers = [row["layer"] for row in layer_summaries]
        values = [row[top50_key] for row in layer_summaries]
        plt.figure(figsize=(8, 4))
        plt.bar([str(l) for l in layers], values, color="#3b6ea8")
        plt.ylabel(top50_key)
        plt.xlabel("layer")
        plt.title("Positive mass concentration by layer")
        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "concentration_by_layer.png")
        plt.savefig(plot_path, dpi=160)
        plt.close()
    elif top50_key:
        print("[warn] matplotlib not available; skipping concentration plot.")

    # Scatter plots: head vs xtail mean scores for layers 4 and 15
    if plt is not None:
        for target_layer in (4, 15):
            match = next((p for p in files if f"layer{target_layer:02d}" in p), None)
            if match is None:
                continue
            with open(match, "r", encoding="utf-8") as f:
                data = json.load(f)
            regime_features = data.get("regime_features", {})
            head = regime_features.get("head", [])
            xtail = regime_features.get("xtail", [])
            if not head or not xtail:
                continue
            head_map = {f["feature"]: f.get("mean_score", 0.0) for f in head}
            xtail_map = {f["feature"]: f.get("mean_score", 0.0) for f in xtail}
            union = sorted(set(head_map) | set(xtail_map))
            xs = [head_map.get(fid, 0.0) for fid in union]
            ys = [xtail_map.get(fid, 0.0) for fid in union]
            plt.figure(figsize=(5, 5))
            plt.scatter(xs, ys, s=10, alpha=0.6, color="#3b6ea8")
            plt.axhline(0, color="#999999", linewidth=0.6)
            plt.axvline(0, color="#999999", linewidth=0.6)
            plt.xlabel("head mean_score")
            plt.ylabel("xtail mean_score")
            plt.title(f"Head vs xtail mean scores (layer {target_layer})")
            plt.tight_layout()
            out_path = os.path.join(
                args.output_dir, f"head_xtail_scatter_layer{target_layer:02d}.png"
            )
            plt.savefig(out_path, dpi=160)
            plt.close()
    else:
        print("[warn] matplotlib not available; skipping head vs xtail scatter plots.")

    print(f"Wrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
