#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def normalize_class_name(name: str) -> str:
    if not name:
        return ""
    name = str(name).strip().lower()
    name = re.sub(r"^\d+\.", "", name)
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_score_key(sample_item: dict, preferred: str | None = None) -> str:
    if preferred:
        return preferred
    candidates = [
        "mean_similarity",
        "mean_similarity_to_fewshot_backbone",
        "score",
        "similarity",
        "weight",
    ]
    for key in candidates:
        if key in sample_item:
            return key
    raise KeyError(f"Could not find a usable score key in item keys: {list(sample_item.keys())}")


def extract_scores(data: dict, score_key: str | None = None) -> np.ndarray:
    ec = data.get("Ec", [])
    if not ec:
        raise ValueError("Input JSON has no 'Ec' entries.")
    key = choose_score_key(ec[0], score_key)
    values = []
    for item in ec:
        if key not in item:
            raise KeyError(f"Missing key '{key}' in an Ec item.")
        values.append(float(item[key]))
    return np.asarray(values, dtype=float), key


def align_profile(scores: np.ndarray, target_len: int) -> np.ndarray:
    if len(scores) == target_len:
        return scores.copy()
    x_old = np.linspace(0.0, 1.0, num=len(scores))
    x_new = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(x_new, x_old, scores)


def compute_metrics(ref_scores: np.ndarray, prop_scores: np.ndarray):
    target_len = max(len(ref_scores), len(prop_scores))
    ref_aligned = align_profile(ref_scores, target_len)
    prop_aligned = align_profile(prop_scores, target_len)

    score_gap = float(np.mean(np.abs(ref_aligned - prop_aligned)))
    if np.std(ref_aligned) < 1e-12 or np.std(prop_aligned) < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(ref_aligned, prop_aligned)[0, 1])

    return {
        "profile_bins": target_len,
        "score_gap": score_gap,
        "score_correlation": corr,
        "ref_aligned": ref_aligned,
        "prop_aligned": prop_aligned,
    }


def main():
    parser = argparse.ArgumentParser(description="Plot score profiles for reference vs proposed Ec JSON files.")
    parser.add_argument("--ref", required=True, help="Path to reference/standard Ec JSON")
    parser.add_argument("--prop", required=True, help="Path to proposed Ec JSON")
    parser.add_argument("--ref-score-key", default=None, help="Optional explicit score key for reference JSON")
    parser.add_argument("--prop-score-key", default=None, help="Optional explicit score key for proposed JSON")
    parser.add_argument("--out", default="ec_profile_plot.png", help="Output image path")
    parser.add_argument("--title", default=None, help="Custom plot title")
    parser.add_argument("--sort-desc", action="store_true", help="Sort scores descending before plotting")
    args = parser.parse_args()

    ref_data = load_json(args.ref)
    prop_data = load_json(args.prop)

    ref_scores, ref_key = extract_scores(ref_data, args.ref_score_key)
    prop_scores, prop_key = extract_scores(prop_data, args.prop_score_key)

    if args.sort_desc:
        ref_scores = np.sort(ref_scores)[::-1]
        prop_scores = np.sort(prop_scores)[::-1]

    metrics = compute_metrics(ref_scores, prop_scores)
    ref_aligned = metrics["ref_aligned"]
    prop_aligned = metrics["prop_aligned"]
    x = np.arange(1, len(ref_aligned) + 1)

    ref_name = normalize_class_name(ref_data.get("class_name", "Reference"))
    prop_name = normalize_class_name(prop_data.get("class_name", "Proposed"))

    title = args.title
    if title is None:
        if ref_name and prop_name and ref_name == prop_name:
            title = f"Ec Score Profile Comparison: {ref_name.title()}"
        else:
            title = "Ec Score Profile Comparison"

    plt.figure(figsize=(8, 5))
    plt.plot(x, ref_aligned, linewidth=2, label=f"Standard Ec ({ref_key})")
    plt.plot(x, prop_aligned, linewidth=2, label=f"Proposed Ec ({prop_key})")
    plt.xlabel("Aligned exemplar rank")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close()

    result = {
        "reference_class_name": ref_data.get("class_name", ""),
        "proposed_class_name": prop_data.get("class_name", ""),
        "ref_score_key": ref_key,
        "prop_score_key": prop_key,
        "n_ref": int(len(ref_scores)),
        "n_prop": int(len(prop_scores)),
        "profile_bins": metrics["profile_bins"],
        "score_gap": metrics["score_gap"],
        "score_correlation": metrics["score_correlation"],
        "plot_path": str(Path(args.out).resolve()),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
