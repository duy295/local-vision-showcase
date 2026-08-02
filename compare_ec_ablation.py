import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_name(name: Optional[str]) -> str:
    if not name:
        return ''
    return ''.join(ch.lower() for ch in name if ch.isalnum())


def detect_score_key(ec_items: List[dict], preferred: Optional[str] = None) -> str:
    if preferred:
        return preferred
    candidates = [
        'mean_similarity_to_fewshot_backbone',
        'mean_similarity',
        'score',
        'similarity',
    ]
    for key in candidates:
        if any(key in item for item in ec_items):
            return key
    raise KeyError('Could not detect a usable score key in Ec items.')


def extract_scores(ec_items: List[dict], score_key: str) -> np.ndarray:
    values = []
    for item in ec_items:
        if score_key in item:
            values.append(float(item[score_key]))
    if not values:
        raise ValueError(f'No scores found with key={score_key!r}')
    arr = np.asarray(values, dtype=np.float64)
    # Sort descending so profile comparison follows exemplar quality/rank order.
    return np.sort(arr)[::-1]


def extract_vectors(
    ec_items: List[dict],
    embeddings_map: Optional[Dict[str, List[float]]] = None,
) -> Optional[np.ndarray]:
    vecs = []
    if embeddings_map is not None:
        for item in ec_items:
            p = item.get('path')
            if p in embeddings_map:
                vecs.append(np.asarray(embeddings_map[p], dtype=np.float64))
        return np.vstack(vecs) if vecs else None

    for item in ec_items:
        if 'embedding' in item:
            vecs.append(np.asarray(item['embedding'], dtype=np.float64))
    return np.vstack(vecs) if vecs else None



def quantile_profile(scores: np.ndarray, n_bins: int) -> np.ndarray:
    if len(scores) == n_bins:
        return scores.astype(np.float64)
    if len(scores) == 1:
        return np.repeat(scores[0], n_bins).astype(np.float64)

    # Interpolate on normalized rank positions [0, 1].
    x_old = np.linspace(0.0, 1.0, num=len(scores))
    x_new = np.linspace(0.0, 1.0, num=n_bins)
    return np.interp(x_new, x_old, scores).astype(np.float64)



def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size:
        raise ValueError('Pearson correlation requires arrays of equal length.')
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
    if denom == 0:
        return 0.0
    return float(np.dot(a_centered, b_centered) / denom)



def compute_metrics(
    ref_data: dict,
    prop_data: dict,
    ref_score_key: Optional[str] = None,
    prop_score_key: Optional[str] = None,
    n_bins: Optional[int] = None,
    embeddings_map: Optional[Dict[str, List[float]]] = None,
) -> dict:
    ref_items = ref_data['Ec']
    prop_items = prop_data['Ec']

    ref_score_key = detect_score_key(ref_items, ref_score_key)
    prop_score_key = detect_score_key(prop_items, prop_score_key)

    ref_scores = extract_scores(ref_items, ref_score_key)
    prop_scores = extract_scores(prop_items, prop_score_key)

    if n_bins is None:
        n_bins = max(len(ref_scores), len(prop_scores))

    ref_profile = quantile_profile(ref_scores, n_bins)
    prop_profile = quantile_profile(prop_scores, n_bins)

    # 1) Centroid distance.
    ref_vecs = extract_vectors(ref_items, embeddings_map)
    prop_vecs = extract_vectors(prop_items, embeddings_map)
    if ref_vecs is not None and prop_vecs is not None and len(ref_vecs) > 0 and len(prop_vecs) > 0:
        ref_centroid = ref_vecs.mean(axis=0)
        prop_centroid = prop_vecs.mean(axis=0)
        centroid_distance = float(np.linalg.norm(ref_centroid - prop_centroid))
        centroid_note = 'real feature-centroid distance'
    else:
        # Fallback proxy when only scalar scores are available.
        centroid_distance = float(abs(ref_scores.mean() - prop_scores.mean()))
        centroid_note = 'score-centroid proxy (no embeddings provided)'

    # 2) Score gap on aligned score profiles.
    score_gap = float(np.mean(np.abs(ref_profile - prop_profile)))

    # 3) Pearson correlation on aligned score profiles.
    score_correlation = pearson_corr(ref_profile, prop_profile)

    return {
        'reference_class_name': ref_data.get('class_name'),
        'proposed_class_name': prop_data.get('class_name'),
        'same_class_name_after_normalization': normalize_name(ref_data.get('class_name')) == normalize_name(prop_data.get('class_name')),
        'ref_score_key': ref_score_key,
        'prop_score_key': prop_score_key,
        'n_ref': len(ref_scores),
        'n_prop': len(prop_scores),
        'profile_bins': n_bins,
        'centroid_distance': centroid_distance,
        'centroid_note': centroid_note,
        'score_gap': score_gap,
        'score_correlation': score_correlation,
        'ref_score_mean': float(ref_scores.mean()),
        'prop_score_mean': float(prop_scores.mean()),
        'ref_score_std': float(ref_scores.std(ddof=0)),
        'prop_score_std': float(prop_scores.std(ddof=0)),
    }



def maybe_load_embeddings(path: Optional[str]) -> Optional[Dict[str, List[float]]]:
    if not path:
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)



def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compare two Ec JSON files for novel-exemplar approximation ablation.'
    )
    parser.add_argument('--ref', required=True, help='Path to the standard/reference Ec JSON file.')
    parser.add_argument('--prop', required=True, help='Path to the proposed/method Ec JSON file.')
    parser.add_argument('--bins', type=int, default=None, help='Number of bins for aligned score profiles. Default: max(len(ref), len(prop)).')
    parser.add_argument('--ref-score-key', default=None, help='Optional override for the reference score key.')
    parser.add_argument('--prop-score-key', default=None, help='Optional override for the proposed score key.')
    parser.add_argument('--embeddings-json', default=None, help='Optional JSON mapping image path -> embedding list for real centroid distance.')
    parser.add_argument('--save-json', default=None, help='Optional path to save the metrics as JSON.')
    args = parser.parse_args()

    ref_data = load_json(args.ref)
    prop_data = load_json(args.prop)
    embeddings_map = maybe_load_embeddings(args.embeddings_json)

    metrics = compute_metrics(
        ref_data=ref_data,
        prop_data=prop_data,
        ref_score_key=args.ref_score_key,
        prop_score_key=args.prop_score_key,
        n_bins=args.bins,
        embeddings_map=embeddings_map,
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print('\nLaTeX row example:')
    print(
        f"Proposed Transfer & {metrics['centroid_distance']:.4f} & {metrics['score_gap']:.4f} & {metrics['score_correlation']:.4f} \\\\"
    )

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f'\nSaved metrics to: {out_path}')


if __name__ == '__main__':
    main()
