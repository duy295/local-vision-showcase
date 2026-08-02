import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def normalize_path(p: str) -> str:
    return os.path.normpath(str(p)).replace('\\', '/').lower()


def normalize_class_name(name: Optional[str]) -> str:
    if not name:
        return ''
    s = str(name).lower().replace('_', ' ').replace('-', ' ')
    s = re.sub(r'^\s*\d+[\.\-_\s]*', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'[^a-z0-9]+', '', s)
    return s


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def get_scores_from_ec(data: dict, preferred_key: str) -> np.ndarray:
    ec = data['Ec']
    if not ec:
        raise ValueError('Ec list is empty.')
    if preferred_key in ec[0]:
        return np.array([float(x[preferred_key]) for x in ec], dtype=float)

    fallback_keys = ['mean_similarity', 'mean_similarity_to_fewshot_backbone']
    for k in fallback_keys:
        if k in ec[0]:
            return np.array([float(x[k]) for x in ec], dtype=float)
    raise ValueError(f'Cannot find score key in Ec. Preferred={preferred_key}')


def get_paths_from_ec(data: dict) -> List[str]:
    return [normalize_path(x['path']) for x in data['Ec'] if 'path' in x]


def get_fewshot_paths_from_prop(data: dict) -> List[str]:
    out = []
    for item in data.get('Ec', []):
        if str(item.get('source', '')).lower() == 'fewshot' and 'path' in item:
            out.append(normalize_path(item['path']))
    return out


def align_scores(scores: np.ndarray, target_len: int) -> np.ndarray:
    if len(scores) == target_len:
        return scores
    old_x = np.linspace(0.0, 1.0, len(scores))
    new_x = np.linspace(0.0, 1.0, target_len)
    return np.interp(new_x, old_x, scores)


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T


def load_embeddings_map(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        return {}
    raw = load_json(path)
    return {normalize_path(k): np.asarray(v, dtype=np.float32) for k, v in raw.items()}


def save_embeddings_map(emb_map: Dict[str, np.ndarray], path: str):
    serializable = {k: v.astype(float).tolist() for k, v in emb_map.items()}
    save_json(serializable, path)


def build_encoder(device: str = 'auto', model_name: str = 'resnet50'):
    import torch
    import torchvision.models as models

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_name = model_name.lower()
    if model_name == 'resnet50':
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            model = models.resnet50(weights=weights)
        except Exception:
            eprint('[Warn] Could not load pretrained ResNet50 weights. Falling back to random init.')
            model = models.resnet50(weights=None)
        model.fc = torch.nn.Identity()
        input_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif model_name == 'resnet18':
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
        except Exception:
            eprint('[Warn] Could not load pretrained ResNet18 weights. Falling back to random init.')
            model = models.resnet18(weights=None)
        model.fc = torch.nn.Identity()
        input_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError(f'Unsupported model: {model_name}. Use resnet50 or resnet18.')

    model = model.to(device)
    model.eval()

    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return model, transform, device


def encode_missing_images(paths: List[str], emb_map: Dict[str, np.ndarray], model_name: str, device: str) -> Dict[str, np.ndarray]:
    missing = [p for p in paths if p not in emb_map]
    if not missing:
        print('[Info] No missing embeddings. Reusing embeddings map.')
        return emb_map

    print(f'[Info] Building embeddings for {len(missing)} missing images using {model_name}...')
    import torch
    from PIL import Image

    model, transform, resolved_device = build_encoder(device=device, model_name=model_name)
    for idx, p in enumerate(missing, 1):
        try:
            img = Image.open(p).convert('RGB')
            x = transform(img).unsqueeze(0).to(resolved_device)
            with torch.no_grad():
                feat = model(x).squeeze(0).cpu().numpy().astype(np.float32)
            emb_map[p] = feat
        except Exception as e:
            eprint(f'[Warn] Skip embedding for {p}: {e}')
        if idx % 50 == 0 or idx == len(missing):
            print(f'[Info] Embedded {idx}/{len(missing)} images')
    return emb_map


def compute_feature_centroid_distance(ref_paths: List[str], prop_paths: List[str], emb_map: Dict[str, np.ndarray]) -> Tuple[Optional[float], str]:
    ref_vecs = [emb_map[p] for p in ref_paths if p in emb_map]
    prop_vecs = [emb_map[p] for p in prop_paths if p in emb_map]
    if not ref_vecs or not prop_vecs:
        return None, 'feature centroid distance unavailable (missing embeddings)'
    mu_ref = np.mean(np.stack(ref_vecs, axis=0), axis=0)
    mu_prop = np.mean(np.stack(prop_vecs, axis=0), axis=0)
    return float(np.linalg.norm(mu_ref - mu_prop)), 'feature centroid distance'


def plot_score_profile(ref_aligned: np.ndarray, prop_aligned: np.ndarray, save_path: str, class_label: str):
    import matplotlib.pyplot as plt
    target_len = len(ref_aligned)
    x = np.arange(1, target_len + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(x, ref_aligned, label='Standard Ec', linewidth=2)
    plt.plot(x, prop_aligned, label='Proposed Ec', linewidth=2)
    plt.xlabel('Aligned exemplar rank')
    plt.ylabel('Score')
    plt.title(f'Score Profile Comparison - {class_label}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_similarity_heatmap(Q: np.ndarray, E_ref: np.ndarray, E_prop: np.ndarray, save_path: str, class_label: str):
    import matplotlib.pyplot as plt
    H_ref = cosine_similarity_matrix(Q, E_ref)
    H_prop = cosine_similarity_matrix(Q, E_prop)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axes[0].imshow(H_ref, aspect='auto')
    axes[0].set_title('Few-shot Query vs Standard Ec')
    axes[0].set_xlabel('Standard exemplar index')
    axes[0].set_ylabel('Few-shot query index')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(H_prop, aspect='auto')
    axes[1].set_title('Few-shot Query vs Proposed Ec')
    axes[1].set_xlabel('Proposed exemplar index')
    axes[1].set_ylabel('Few-shot query index')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f'Similarity Heatmap - {class_label}', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_umap(ref_paths: List[str], prop_paths: List[str], fewshot_paths: List[str], emb_map: Dict[str, np.ndarray], save_path: str, class_label: str) -> bool:
    try:
        import matplotlib.pyplot as plt
        import umap
    except Exception as e:
        eprint(f'[Warn] UMAP skipped: {e}')
        return False

    X = []
    labels = []
    for p in ref_paths:
        if p in emb_map:
            X.append(emb_map[p])
            labels.append('Standard Ec')
    for p in prop_paths:
        if p in emb_map:
            X.append(emb_map[p])
            labels.append('Proposed Ec')
    for p in fewshot_paths:
        if p in emb_map:
            X.append(emb_map[p])
            labels.append('Few-shot')

    if len(X) < 3:
        eprint('[Warn] UMAP skipped: not enough embedded points.')
        return False

    X = np.stack(X, axis=0)
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.15, metric='cosine', random_state=42)
    X_2d = reducer.fit_transform(X)
    labels = np.array(labels)

    plt.figure(figsize=(8, 6))
    for name, marker in [('Standard Ec', 'o'), ('Proposed Ec', '^'), ('Few-shot', 's')]:
        idx = labels == name
        if idx.sum() > 0:
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=name, marker=marker, alpha=0.8)
    plt.title(f'UMAP - {class_label}')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description='Full local analysis for Standard Ec vs Proposed Ec.')
    parser.add_argument('--ref', required=True, help='Path to reference/standard Ec JSON')
    parser.add_argument('--prop', required=True, help='Path to proposed Ec JSON')
    parser.add_argument('--embeddings-json', default='embeddings_map.json', help='Path to embeddings cache JSON')
    parser.add_argument('--output-dir', default='ec_analysis_output', help='Directory to save metrics and figures')
    parser.add_argument('--ref-score-key', default='mean_similarity', help='Score key for reference Ec')
    parser.add_argument('--prop-score-key', default='mean_similarity', help='Score key for proposed Ec')
    parser.add_argument('--embedding-model', default='resnet50', choices=['resnet50', 'resnet18'], help='Backbone for building embeddings')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Device for embedding extraction')
    parser.add_argument('--skip-embeddings', action='store_true', help='Skip embedding extraction and embedding-based figures/metrics')
    parser.add_argument('--skip-umap', action='store_true', help='Skip UMAP plot')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ref_data = load_json(args.ref)
    prop_data = load_json(args.prop)

    ref_class_name = ref_data.get('class_name', '')
    prop_class_name = prop_data.get('class_name', '')
    same_class_norm = normalize_class_name(ref_class_name) == normalize_class_name(prop_class_name)
    class_label = prop_class_name or ref_class_name or 'UnknownClass'

    ref_scores = np.sort(get_scores_from_ec(ref_data, args.ref_score_key))[::-1]
    prop_scores = np.sort(get_scores_from_ec(prop_data, args.prop_score_key))[::-1]

    target_len = max(len(ref_scores), len(prop_scores))
    ref_aligned = align_scores(ref_scores, target_len)
    prop_aligned = align_scores(prop_scores, target_len)

    score_gap = float(np.mean(np.abs(ref_aligned - prop_aligned)))
    if np.std(ref_aligned) < 1e-12 or np.std(prop_aligned) < 1e-12:
        score_corr = 0.0
        corr_note = 'one profile is nearly constant'
    else:
        score_corr = float(np.corrcoef(ref_aligned, prop_aligned)[0, 1])
        corr_note = ''

    score_centroid_proxy = float(abs(ref_scores.mean() - prop_scores.mean()))

    ref_paths = get_paths_from_ec(ref_data)
    prop_paths = get_paths_from_ec(prop_data)
    fewshot_paths = get_fewshot_paths_from_prop(prop_data)
    all_paths = list(dict.fromkeys(ref_paths + prop_paths + fewshot_paths))

    emb_map: Dict[str, np.ndarray] = {}
    feature_centroid_distance = None
    feature_centroid_note = 'feature centroid distance unavailable'
    heatmap_saved = False
    umap_saved = False

    if not args.skip_embeddings:
        emb_map = load_embeddings_map(args.embeddings_json)
        print(f'[Info] Loaded {len(emb_map)} cached embeddings from {args.embeddings_json}')
        emb_map = encode_missing_images(all_paths, emb_map, model_name=args.embedding_model, device=args.device)
        save_embeddings_map(emb_map, args.embeddings_json)
        print(f'[Info] Saved embeddings cache to {args.embeddings_json}')

        feature_centroid_distance, feature_centroid_note = compute_feature_centroid_distance(ref_paths, prop_paths, emb_map)

        ref_emb_paths = [p for p in ref_paths if p in emb_map]
        prop_emb_paths = [p for p in prop_paths if p in emb_map]
        fewshot_emb_paths = [p for p in fewshot_paths if p in emb_map]

        if ref_emb_paths and prop_emb_paths and fewshot_emb_paths:
            Q = np.stack([emb_map[p] for p in fewshot_emb_paths], axis=0)
            E_ref = np.stack([emb_map[p] for p in ref_emb_paths], axis=0)
            E_prop = np.stack([emb_map[p] for p in prop_emb_paths], axis=0)
            heatmap_path = os.path.join(args.output_dir, 'similarity_heatmap.png')
            plot_similarity_heatmap(Q, E_ref, E_prop, heatmap_path, class_label)
            heatmap_saved = True
        else:
            eprint('[Warn] Heatmap skipped: missing embeddings for few-shot/ref/prop sets.')

        if not args.skip_umap:
            umap_path = os.path.join(args.output_dir, 'umap.png')
            umap_saved = plot_umap(ref_paths, prop_paths, fewshot_paths, emb_map, umap_path, class_label)

    score_plot_path = os.path.join(args.output_dir, 'score_profile.png')
    plot_score_profile(ref_aligned, prop_aligned, score_plot_path, class_label)

    metrics = {
        'reference_class_name': ref_class_name,
        'proposed_class_name': prop_class_name,
        'same_class_name_after_normalization': same_class_norm,
        'ref_score_key': args.ref_score_key,
        'prop_score_key': args.prop_score_key,
        'n_ref': int(len(ref_scores)),
        'n_prop': int(len(prop_scores)),
        'profile_bins': int(target_len),
        'score_centroid_proxy': score_centroid_proxy,
        'score_centroid_proxy_note': 'absolute difference between score means',
        'feature_centroid_distance': feature_centroid_distance,
        'feature_centroid_note': feature_centroid_note,
        'score_gap': score_gap,
        'score_correlation': score_corr,
        'score_correlation_note': corr_note,
        'ref_score_mean': float(ref_scores.mean()),
        'prop_score_mean': float(prop_scores.mean()),
        'ref_score_std': float(ref_scores.std()),
        'prop_score_std': float(prop_scores.std()),
        'figures': {
            'score_profile': score_plot_path,
            'similarity_heatmap': os.path.join(args.output_dir, 'similarity_heatmap.png') if heatmap_saved else None,
            'umap': os.path.join(args.output_dir, 'umap.png') if umap_saved else None,
        },
        'latex_row': f"{class_label} & {score_centroid_proxy:.4f} & {score_gap:.4f} & {score_corr:.4f} \\",
    }

    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    save_json(metrics, metrics_path)

    print('\n===== SUMMARY =====')
    print(json.dumps(metrics, indent=2))
    print('\nSaved files:')
    print(f'- Metrics JSON      : {metrics_path}')
    print(f'- Score profile PNG : {score_plot_path}')
    if heatmap_saved:
        print(f'- Heatmap PNG       : {os.path.join(args.output_dir, "similarity_heatmap.png")}')
    if umap_saved:
        print(f'- UMAP PNG          : {os.path.join(args.output_dir, "umap.png")}')


if __name__ == '__main__':
    main()
