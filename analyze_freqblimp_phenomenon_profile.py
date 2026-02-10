import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for this analysis") from exc

try:
    from sklearn.cluster import KMeans
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for this analysis") from exc

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional
    plt = None


DEFAULT_PATTERN = "freqblimp_*_attribution_layer*.json"


def _layer_from_path(path):
    base = os.path.basename(path)
    match = re.search(r"layer(\d+)", base)
    if match:
        return int(match.group(1))
    digits = "".join(ch for ch in base if ch.isdigit())
    if digits:
        return int(digits[:2])
    return None


def _latest_file_per_layer(paths):
    latest = {}
    for path in paths:
        layer = _layer_from_path(path)
        if layer is None:
            continue
        prev = latest.get(layer)
        if prev is None or os.path.getmtime(path) > os.path.getmtime(prev):
            latest[layer] = path
    return [latest[layer] for layer in sorted(latest)]


def _activation_any_rate(feature, num_examples):
    if not num_examples:
        return 0.0
    return feature.get("example_count", 0) / num_examples


def _load_layer(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    summary = data.get("summary", {})
    features = data.get("features", [])
    phenomenon_features = data.get("phenomenon_features", {})
    layer = summary.get("layer")
    if layer is None:
        layer = _layer_from_path(path)
    return layer, summary, features, phenomenon_features


def _build_feature_lookup(features, num_examples):
    lookup = {}
    for feat in features:
        fid = feat.get("feature")
        if fid is None:
            continue
        lookup[fid] = {
            "mean_score": feat.get("mean_score", 0.0),
            "activation_any_rate": _activation_any_rate(feat, num_examples),
        }
    return lookup


def _union_topk_features(phenomenon_features, topk):
    union = set()
    for feats in phenomenon_features.values():
        if not feats:
            continue
        union.update([f.get("feature") for f in feats[:topk] if f.get("feature") is not None])
    return union


def _build_matrix(feature_ids, phenomenon_list, phenomenon_features, topk):
    feat_index = {fid: i for i, fid in enumerate(feature_ids)}
    phen_index = {p: i for i, p in enumerate(phenomenon_list)}
    mat = np.zeros((len(feature_ids), len(phenomenon_list)), dtype=float)
    for phen, feats in phenomenon_features.items():
        if phen not in phen_index:
            continue
        j = phen_index[phen]
        for feat in feats[:topk]:
            fid = feat.get("feature")
            if fid not in feat_index:
                continue
            i = feat_index[fid]
            mat[i, j] = feat.get("mean_score", 0.0)
    return mat


def _l2_normalize(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _entropy_positive(row):
    positives = row[row > 0]
    if positives.size == 0:
        return 0.0
    total = positives.sum()
    if total <= 0:
        return 0.0
    probs = positives / total
    return float(-np.sum(probs * np.log(probs)))


def _top_phenomena_for_feature(row, phenomenon_list, k=3):
    if row.size == 0:
        return []
    idx = np.argsort(row)[::-1]
    top = []
    for i in idx:
        if row[i] == 0:
            break
        top.append(phenomenon_list[i])
        if len(top) >= k:
            break
    return top


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _cluster_and_report(
    layer,
    summary,
    features,
    phenomenon_features,
    out_dir,
    topk,
    max_any_rate,
    k,
    seed,
):
    num_examples = summary.get("num_examples", 0)
    if not phenomenon_features:
        print(f"[warn] layer {layer}: no phenomenon_features; skipping")
        return

    lookup = _build_feature_lookup(features, num_examples)
    union = _union_topk_features(phenomenon_features, topk)
    if max_any_rate is not None:
        union = {fid for fid in union if lookup.get(fid, {}).get("activation_any_rate", 0.0) <= max_any_rate}

    if not union:
        print(f"[warn] layer {layer}: no features after filtering; skipping")
        return

    phenomenon_list = sorted(phenomenon_features.keys())
    feature_ids = sorted(union)

    matrix = _build_matrix(feature_ids, phenomenon_list, phenomenon_features, topk)
    matrix_norm = _l2_normalize(matrix)

    k_use = min(k, len(feature_ids))
    if k_use < k:
        print(f"[warn] layer {layer}: reducing k from {k} to {k_use} due to feature count")

    kmeans = KMeans(n_clusters=k_use, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(matrix_norm)

    # Cluster profiles (unnormalized)
    cluster_profiles = np.zeros((k_use, len(phenomenon_list)), dtype=float)
    cluster_sizes = np.zeros(k_use, dtype=int)
    for i, lbl in enumerate(labels):
        cluster_profiles[lbl] += matrix[i]
        cluster_sizes[lbl] += 1
    for c in range(k_use):
        if cluster_sizes[c] > 0:
            cluster_profiles[c] /= cluster_sizes[c]

    # Feature-level metrics
    mean_scores = np.array([lookup.get(fid, {}).get("mean_score", 0.0) for fid in feature_ids])
    entropies = np.array([_entropy_positive(row) for row in matrix])
    activation_any = np.array([lookup.get(fid, {}).get("activation_any_rate", 0.0) for fid in feature_ids])

    # Save assignments CSV
    assignment_rows = []
    for i, fid in enumerate(feature_ids):
        tops = _top_phenomena_for_feature(matrix[i], phenomenon_list, k=3)
        assignment_rows.append(
            {
                "feature": fid,
                "cluster": int(labels[i]),
                "entropy_positive": entropies[i],
                "mean_score": mean_scores[i],
                "activation_any_rate": activation_any[i],
                "top_phenomena": ",".join(tops),
            }
        )

    out_assign = os.path.join(out_dir, f"feature_clusters_layer{layer:02d}.csv")
    _write_csv(
        out_assign,
        assignment_rows,
        ["feature", "cluster", "entropy_positive", "mean_score", "activation_any_rate", "top_phenomena"],
    )

    # Save cluster summary text
    summary_lines = []
    summary_lines.append(f"Layer {layer} cluster summary (k={k_use})")
    summary_lines.append(f"Features used: {len(feature_ids)} (filtered max activation_any_rate={max_any_rate})")
    summary_lines.append("")

    for c in range(k_use):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        cluster_entropy = float(entropies[idx].mean())
        profile = cluster_profiles[c]
        top_phens = _top_phenomena_for_feature(profile, phenomenon_list, k=3)

        # Representative features
        by_mean = idx[np.argsort(mean_scores[idx])[::-1]]
        by_entropy = idx[np.argsort(entropies[idx])]
        top_feat = feature_ids[by_mean[0]] if by_mean.size else None
        low_entropy_feat = feature_ids[by_entropy[0]] if by_entropy.size else None
        high_entropy_feat = feature_ids[by_entropy[-1]] if by_entropy.size else None

        summary_lines.append(f"Cluster {c} (n={idx.size})")
        summary_lines.append(f"  top_phenomena: {', '.join(top_phens) if top_phens else 'none'}")
        summary_lines.append(f"  mean_entropy_positive: {cluster_entropy:.4f}")
        summary_lines.append(
            "  reps: top_mean_score=%s, lowest_entropy=%s, highest_entropy=%s"
            % (top_feat, low_entropy_feat, high_entropy_feat)
        )
        summary_lines.append("")

    out_summary = os.path.join(out_dir, f"cluster_summary_layer{layer:02d}.txt")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_summary, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # Save cluster profiles as CSV
    profile_rows = []
    for c in range(k_use):
        row = {"cluster": c, "size": int(cluster_sizes[c])}
        for j, phen in enumerate(phenomenon_list):
            row[phen] = cluster_profiles[c, j]
        profile_rows.append(row)
    out_profile = os.path.join(out_dir, f"cluster_profiles_layer{layer:02d}.csv")
    _write_csv(out_profile, profile_rows, ["cluster", "size"] + phenomenon_list)

    # Heatmap
    if plt is not None:
        fig = plt.figure(figsize=(max(6, len(phenomenon_list) * 0.6), 2 + k_use * 0.6))
        ax = fig.add_subplot(111)
        im = ax.imshow(cluster_profiles, aspect="auto", cmap="coolwarm")
        ax.set_yticks(range(k_use))
        ax.set_yticklabels([f"C{c}" for c in range(k_use)])
        ax.set_xticks(range(len(phenomenon_list)))
        ax.set_xticklabels(phenomenon_list, rotation=45, ha="right")
        ax.set_xlabel("phenomenon")
        ax.set_ylabel("cluster")
        ax.set_title(f"Phenomenon profiles (layer {layer})")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        out_fig = os.path.join(out_dir, f"phenomenon_cluster_heatmap_layer{layer:02d}.png")
        fig.savefig(out_fig, dpi=160)
        plt.close(fig)
    else:
        print("[warn] matplotlib not available; skipping heatmap plots")


def main():
    parser = argparse.ArgumentParser(
        description="Phenomenon attribution profile clustering per layer."
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
        default="outputs/attribution/phenomenon_profile",
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--topk_per_phenomenon",
        type=int,
        default=200,
        help="Top-k per phenomenon to form the union.",
    )
    parser.add_argument(
        "--filter_activation_any_rate",
        type=float,
        default=0.99,
        help="Drop features with activation_any_rate above this threshold (set <0 to disable).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of clusters.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for k-means.",
    )

    args = parser.parse_args()
    max_any_rate = args.filter_activation_any_rate
    if max_any_rate is not None and max_any_rate < 0:
        max_any_rate = None

    pattern = os.path.join(args.input_dir, args.pattern)
    all_files = sorted(glob.glob(pattern))
    if not all_files:
        raise SystemExit(f"No attribution files found for pattern: {pattern}")
    files = _latest_file_per_layer(all_files)
    if len(files) < len(all_files):
        print(f"[info] Using latest run per layer: {len(files)} files from {len(all_files)} matches")

    for path in files:
        layer, summary, features, phenomenon_features = _load_layer(path)
        if layer is None:
            print(f"[warn] Could not parse layer from {path}; skipping")
            continue
        _cluster_and_report(
            layer,
            summary,
            features,
            phenomenon_features,
            args.output_dir,
            args.topk_per_phenomenon,
            max_any_rate,
            args.k,
            args.seed,
        )

    print(f"Wrote phenomenon profile analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
