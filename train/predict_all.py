import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Ensure repo root is importable when running: python train/predict_all.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backbone.feature_extract import HybridResNetBackbone
from backbone.relation_net import BilinearRelationNet


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ScoreCombinerNet(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1),
        )
        self.temperature = 0.5

    def forward(self, scores1, scores2, scores3):
        combined_input = torch.stack([scores1, scores2, scores3], dim=1)
        output = self.net(combined_input)
        return torch.sigmoid(output.squeeze(-1) / self.temperature) * 0.98


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _is_image_file(path):
    return os.path.splitext(path)[1].lower() in SUPPORTED_IMAGE_EXTS


def _normalize_class_name(name):
    """
    Remove numeric prefix in class names.
    Example: '001.Black_footed_Albatross' -> 'Black_footed_Albatross'
    """
    if not name:
        return ""
    return re.sub(r"^\s*\d+\s*[\.\-_]*\s*", "", str(name)).strip()


def _canonical_class_key(name: Optional[str]) -> str:
    """
    Canonical key for class matching across EC/CLIP:
    - remove numeric prefix
    - lower + underscores
    """
    return _normalize_class_name(name).replace(" ", "_").lower() if name else ""


def _display_class_name(name: Optional[str]) -> str:
    """
    Convert any class name format to human-readable display format.
    Examples:
      'american_pipit'     -> 'American Pipit'
      'American_Pipit'     -> 'American Pipit'
      '104.American_Pipit' -> 'American Pipit'
      'American Pipit'     -> 'American Pipit'  (already correct, no change)
    """
    if not name:
        return ""
    name = _normalize_class_name(name)
    return name.replace("_", " ").title()


def _norm_path(p: str) -> str:
    return os.path.normcase(os.path.normpath(p))


def _load_clip_results(clip_topk_json: str) -> List[dict]:
    with open(clip_topk_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload.get("results"), list):
        return payload["results"]
    if isinstance(payload.get("result"), dict):
        return [payload["result"]]
    return []


def _index_clip_results(clip_results: List[dict]) -> Tuple[Dict[str, dict], Dict[str, List[dict]]]:
    by_path: Dict[str, dict] = {}
    by_base: Dict[str, List[dict]] = {}
    for item in clip_results:
        p = item.get("image_path", "") or ""
        if p:
            by_path[_norm_path(p)] = item
            base = os.path.basename(p).lower()
            by_base.setdefault(base, []).append(item)
    return by_path, by_base


def _filter_classes(exemplars, selected_classes):
    if not selected_classes:
        return exemplars
    selected_norm = {_normalize_class_name(x) for x in selected_classes}
    return {
        cls: paths
        for cls, paths in exemplars.items()
        if _normalize_class_name(cls) in selected_norm
    }


def _limit_classes_by_order(exemplars, class_limit=0):
    if not class_limit or class_limit <= 0:
        return exemplars
    selected_names = sorted(exemplars.keys())[:class_limit]
    return {name: exemplars[name] for name in selected_names}


def _select_segment_representatives(paths, num_segments=5):
    """
    Split ordered paths into segments and pick up to 3 representatives per segment:
    head, middle, tail (unique positions only).
    """
    if not paths:
        return []

    segment_count = max(1, int(num_segments))
    segment_count = min(segment_count, len(paths))
    reps = []
    n = len(paths)

    for i in range(segment_count):
        start = (i * n) // segment_count
        end = ((i + 1) * n) // segment_count
        segment = paths[start:end]
        if not segment:
            continue

        idx_candidates = [0, len(segment) // 2, len(segment) - 1]
        used_local_idx = set()
        for idx in idx_candidates:
            if idx in used_local_idx:
                continue
            used_local_idx.add(idx)
            reps.append(segment[idx])

    return reps


def load_models(weights_dir):
    print(f"Loading weights from: {weights_dir}")
    backbone = HybridResNetBackbone().to(device)
    relation = BilinearRelationNet().to(device)
    combiner = ScoreCombinerNet().to(device)

    backbone.load_state_dict(
        torch.load(os.path.join(weights_dir, "backbone_full.pth"), map_location=device)
    )
    relation.load_state_dict(
        torch.load(os.path.join(weights_dir, "relation_full.pth"), map_location=device)
    )
    combiner.load_state_dict(
        torch.load(os.path.join(weights_dir, "score_combiner_full.pth"), map_location=device)
    )

    backbone.eval()
    relation.eval()
    combiner.eval()
    return backbone, relation, combiner


def _apply_image_per_ec_limit(paths, image_per_ec=0):
    if not image_per_ec or image_per_ec <= 0:
        return paths
    return paths[:image_per_ec]


def get_class_exemplars(support_dir, num_segments=5, image_per_ec=0):
    """
    Fallback for support_dir mode:
    choose representative images by segmenting ordered images in each class.
    """
    classes = sorted(
        d for d in os.listdir(support_dir) if os.path.isdir(os.path.join(support_dir, d))
    )
    class_exemplars = {}

    for cls in classes:
        cls_path = os.path.join(support_dir, cls)
        images = []
        for p in sorted(glob.glob(os.path.join(cls_path, "*"))):
            if os.path.isfile(p) and _is_image_file(p):
                images.append(p)
        images = _apply_image_per_ec_limit(images, image_per_ec=image_per_ec)
        class_exemplars[cls] = _select_segment_representatives(images, num_segments)

    return class_exemplars


def _extract_paths_from_ec_items(ec_items, image_per_ec=0):
    """
    Read valid image paths from Ec items.
    Supports dict item {'path': ..., 'rank': ...} or string path.
    """
    items = ec_items or []
    if items and isinstance(items[0], dict) and "rank" in items[0]:
        items = sorted(items, key=lambda x: x.get("rank", 10**9))

    paths = []
    for item in items:
        if isinstance(item, dict):
            p = item.get("path")
        else:
            p = str(item)
        if p and os.path.exists(p):
            paths.append(p)
    return _apply_image_per_ec_limit(paths, image_per_ec=image_per_ec)


def get_class_exemplars_from_ec_json(ec_json_dir, num_segments=5, image_per_ec=0):
    """
    Load exemplars from Ec JSON, then pick representative images by segments.
    Supports:
      - Ec_all_classes.json
      - files *_Ec.json
    """
    class_exemplars = {}
    merged_path = os.path.join(ec_json_dir, "Ec_all_classes.json")

    if os.path.isfile(merged_path):
        with open(merged_path, "r", encoding="utf-8") as f:
            merged = json.load(f)

        for class_name, ec_items in merged.items():
            paths = _extract_paths_from_ec_items(ec_items, image_per_ec=image_per_ec)
            class_exemplars[class_name] = _select_segment_representatives(paths, num_segments)
        return class_exemplars

    json_files = sorted(glob.glob(os.path.join(ec_json_dir, "*_Ec.json")))
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            payload = json.load(f)

        class_name = payload.get("class_name") or os.path.basename(jf).replace(
            "_Ec.json", ""
        )
        ec_items = payload.get("Ec", [])
        paths = _extract_paths_from_ec_items(ec_items, image_per_ec=image_per_ec)
        class_exemplars[class_name] = _select_segment_representatives(paths, num_segments)

    return class_exemplars


def _load_exemplars(args):
    if args.ec_json_dir:
        if not os.path.isdir(args.ec_json_dir):
            print(f"Ec json dir not found: {args.ec_json_dir}")
            return {}
        exemplars = get_class_exemplars_from_ec_json(
            args.ec_json_dir, num_segments=args.num_segments, image_per_ec=args.image_per_ec
        )
        print(
            f"Using Ec json from: {args.ec_json_dir} "
            f"(image_per_ec={args.image_per_ec}, segment reps/class: up to {args.num_segments * 3})"
        )
    elif args.support_dir:
        if not os.path.isdir(args.support_dir):
            print(f"Support dir not found: {args.support_dir}")
            return {}
        exemplars = get_class_exemplars(
            args.support_dir, num_segments=args.num_segments, image_per_ec=args.image_per_ec
        )
        print(
            f"Using support_dir from: {args.support_dir} "
            f"(image_per_ec={args.image_per_ec}, segment reps/class: up to {args.num_segments * 3})"
        )
    else:
        print("Please provide --ec_json_dir (preferred) or --support_dir")
        return {}

    exemplars = _filter_classes(exemplars, args.ec_classes)
    if args.ec_classes:
        print(f"Filtering classes from args: {', '.join(args.ec_classes)}")
    exemplars = _limit_classes_by_order(exemplars, args.ec_class_limit)
    if args.ec_class_limit and args.ec_class_limit > 0:
        print(f"Limiting Ec classes to first {args.ec_class_limit} (sorted by class name)")
    return exemplars


def _build_support_cache(exemplars, backbone):
    support_cache = {}
    print(f"Preparing support features for {len(exemplars)} classes...")

    for class_name, img_paths in tqdm(exemplars.items(), desc="Preparing support"):
        if not img_paths:
            continue

        support_imgs = []
        for p in img_paths:
            if not os.path.exists(p):
                continue
            try:
                img = transform(Image.open(p).convert("RGB"))
                support_imgs.append(img)
            except Exception as exc:
                print(f"[WARN] Skip support image {p}: {exc}")

        if not support_imgs:
            continue

        support_batch = torch.stack(support_imgs).to(device)
        with torch.no_grad():
            s_feat, s_global, s_patch = backbone(support_batch)

        support_cache[class_name] = {
            "s_feat": s_feat,
            "s_global": s_global,
            "s_patch": s_patch,
            "k": s_feat.shape[0],
        }

    return support_cache


def _score_query_features(q_feat, q_global, q_patch, support_cache, relation, combiner):
    results = []
    all_scores_sum = 0.0
    all_scores_count = 0

    for class_name, cache_item in support_cache.items():
        s_feat = cache_item["s_feat"]
        s_global = cache_item["s_global"]
        s_patch = cache_item["s_patch"]
        k = cache_item["k"]

        curr_q_feat = q_feat.repeat(k, 1)
        curr_q_global = q_global.repeat(k, 1)
        curr_q_patch = q_patch.repeat(k, 1)

        scores1 = relation(curr_q_feat, s_feat)
        scores2 = relation(curr_q_global, s_global)
        score3 = relation(curr_q_patch, s_patch)

        final_scores = combiner(scores1, scores2, score3)
        class_total_score = torch.sum(final_scores).item()
        class_mean_score = torch.mean(final_scores).item()
        results.append((class_name, class_mean_score, class_total_score, k))

        all_scores_sum += class_total_score
        all_scores_count += k

    results.sort(key=lambda x: x[1], reverse=True)
    return results, all_scores_sum, all_scores_count


def _score_query_features_subset(
    q_feat,
    q_global,
    q_patch,
    support_cache: dict,
    relation,
    combiner,
    allowed_class_names: List[str],
):
    results = []
    all_scores_sum = 0.0
    all_scores_count = 0

    for class_name in allowed_class_names:
        cache_item = support_cache.get(class_name)
        if not cache_item:
            continue

        s_feat = cache_item["s_feat"]
        s_global = cache_item["s_global"]
        s_patch = cache_item["s_patch"]
        k = cache_item["k"]

        curr_q_feat = q_feat.repeat(k, 1)
        curr_q_global = q_global.repeat(k, 1)
        curr_q_patch = q_patch.repeat(k, 1)

        scores1 = relation(curr_q_feat, s_feat)
        scores2 = relation(curr_q_global, s_global)
        score3 = relation(curr_q_patch, s_patch)

        final_scores = combiner(scores1, scores2, score3)
        class_total_score = torch.sum(final_scores).item()
        class_mean_score = torch.mean(final_scores).item()
        results.append((class_name, class_mean_score, class_total_score, k))

        all_scores_sum += class_total_score
        all_scores_count += k

    results.sort(key=lambda x: x[1], reverse=True)
    return results, all_scores_sum, all_scores_count


def _predict_single_image_scores(query_path, backbone, relation, combiner, support_cache):
    query_img = transform(Image.open(query_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        q_feat, q_global, q_patch = backbone(query_img)

    results, all_scores_sum, all_scores_count = _score_query_features(
        q_feat, q_global, q_patch, support_cache, relation, combiner
    )
    if not results or all_scores_count == 0:
        return None

    overall_mean_all_scores = all_scores_sum / all_scores_count
    overall_mean_class_scores = sum(r[1] for r in results) / len(results)
    return {
        "results": results,
        "overall_mean_all_scores": overall_mean_all_scores,
        "overall_mean_class_scores": overall_mean_class_scores,
    }


def _predict_single_image_scores_on_classes(
    query_path: str,
    backbone,
    relation,
    combiner,
    support_cache: dict,
    allowed_class_names: List[str],
):
    query_img = transform(Image.open(query_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        q_feat, q_global, q_patch = backbone(query_img)

    results, all_scores_sum, all_scores_count = _score_query_features_subset(
        q_feat, q_global, q_patch, support_cache, relation, combiner, allowed_class_names
    )
    if not results or all_scores_count == 0:
        return None

    overall_mean_all_scores = all_scores_sum / all_scores_count
    overall_mean_class_scores = sum(r[1] for r in results) / len(results)

    # Convert mean_scores -> probabilities (softmax over selected classes).
    mean_scores = torch.tensor([r[1] for r in results], dtype=torch.float32)
    probs = torch.softmax(mean_scores, dim=0).tolist()
    return {
        "results": results,
        "probs": probs,
        "overall_mean_all_scores": overall_mean_all_scores,
        "overall_mean_class_scores": overall_mean_class_scores,
    }


def _print_single_image_results(results_payload, top_k):
    results = results_payload["results"]

    print("\n" + "=" * 90)
    print("Scores of query image against all Ec classes (sorted by mean score)")
    print("=" * 90)
    print(
        "OverallMean(all image-pair scores across input Ec classes): "
        f"{results_payload['overall_mean_all_scores']:.6f}"
    )
    print(
        "OverallMean(per-class means across input Ec classes): "
        f"{results_payload['overall_mean_class_scores']:.6f}"
    )

    for rank, (cls_name, mean_score, total_score, used_k) in enumerate(results, 1):
        print(
            f"#{rank:>3} | Class: {cls_name:<30} | "
            f"MeanScore: {mean_score:.6f} | TotalScore(sum): {total_score:.6f} | reps_used: {used_k}"
        )

    print("=" * 90)
    print("\n" + "=" * 70)
    print(f"Top {top_k} classes by mean score")
    print("=" * 70)

    top_results = results[:top_k]
    for rank, (cls_name, mean_score, total_score, used_k) in enumerate(top_results, 1):
        print(
            f"#{rank} | Class: {cls_name:<30} | MeanScore: {mean_score:.6f} | "
            f"TotalScore: {total_score:.6f} | reps_used: {used_k}"
        )
    print("=" * 70)


def _collect_test_samples(test_dir, test_class_limit=0, test_images_per_class=0):
    samples = []
    class_folders_all = sorted(
        d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))
    )
    if test_class_limit and test_class_limit > 0:
        class_folders = class_folders_all[:test_class_limit]
    else:
        class_folders = class_folders_all

    for class_folder in class_folders:
        class_dir = os.path.join(test_dir, class_folder)
        class_images = []
        for p in sorted(glob.glob(os.path.join(class_dir, "*"))):
            if os.path.isfile(p) and _is_image_file(p):
                class_images.append(p)

        if test_images_per_class and test_images_per_class > 0:
            class_images = class_images[:test_images_per_class]

        for p in class_images:
            samples.append((p, class_folder))

    return samples, class_folders


def _predict_on_test_dir(args, backbone, relation, combiner, support_cache):
    if not os.path.isdir(args.test_dir):
        print(f"Test dir not found: {args.test_dir}")
        return

    samples, selected_class_folders = _collect_test_samples(
        args.test_dir,
        test_class_limit=args.test_class_limit,
        test_images_per_class=args.test_images_per_class,
    )
    if not samples:
        print("No image files found under test_dir class folders.")
        return

    print(
        f"Selected {len(selected_class_folders)} classes, "
        f"found {len(samples)} test images in {args.test_dir}"
    )
    all_items = []
    skipped_items = []
    num_correct_top1 = 0
    num_correct_final_top1 = 0
    num_correct_global_top1 = 0

    clip_by_path = None
    clip_by_base = None
    canonical_to_support_name: Dict[str, str] = {}
    if args.clip_topk_json:
        clip_results = _load_clip_results(args.clip_topk_json)
        clip_by_path, clip_by_base = _index_clip_results(clip_results)
        for support_class_name in support_cache.keys():
            key = _canonical_class_key(support_class_name)
            if key and key not in canonical_to_support_name:
                canonical_to_support_name[key] = support_class_name
        print(
            f"[INFO] Using CLIP top-k filter from: {args.clip_topk_json} "
            f"(compare both final_topk and global_topk)"
        )

    num_correct_final_top1 = 0
    num_correct_global_top1 = 0

    for image_path, true_class_raw in tqdm(samples, desc="Predicting test images"):
        try:
            if args.clip_topk_json:
                clip_item = None
                if clip_by_path is not None:
                    clip_item = clip_by_path.get(_norm_path(image_path))
                if clip_item is None and clip_by_base is not None:
                    cands = clip_by_base.get(os.path.basename(image_path).lower(), [])
                    clip_item = cands[0] if len(cands) == 1 else None
                if clip_item is None:
                    skipped_items.append(
                        {"image_path": image_path, "reason": "clip_item_not_found"}
                    )
                    continue

                def _build_allowed_and_predict(source_key: str):
                    topk_list = clip_item.get(source_key, []) or []
                    clip_candidates = [
                        {
                            "clip_rank": int(x.get("rank", i + 1)),
                            "clip_class_name": x.get("class_name"),
                            "clip_similarity": float(x.get("similarity", 0.0)),
                        }
                        for i, x in enumerate(topk_list)
                        if isinstance(x, dict) and x.get("class_name")
                    ]
                    allowed: List[str] = []
                    clip_info_by_support: Dict[str, dict] = {}
                    for c in clip_candidates:
                        ck = c["clip_class_name"]
                        support_name = canonical_to_support_name.get(_canonical_class_key(ck))
                        if not support_name:
                            continue
                        if support_name not in clip_info_by_support:
                            clip_info_by_support[support_name] = c
                            allowed.append(support_name)

                    # Fallback: if CLIP candidates don't match any support cache class,
                    # use all support cache classes instead of skipping.
                    if not allowed:
                        allowed = list(support_cache.keys())
                        clip_info_by_support = {}

                    pred = _predict_single_image_scores_on_classes(
                        image_path, backbone, relation, combiner, support_cache, allowed
                    )
                    return pred, allowed, clip_info_by_support, topk_list

                pred_final, allowed_final, clip_info_final, topk_list_final = _build_allowed_and_predict("final_topk")
                pred_global, allowed_global, clip_info_global, topk_list_global = _build_allowed_and_predict("global_topk")

                if pred_final is None and pred_global is None:
                    skipped_items.append(
                        {"image_path": image_path, "reason": "no_valid_scores"}
                    )
                    continue

                def _make_topk_entries(pred, clip_info_by_support, max_k):
                    if pred is None:
                        return []
                    entries = []
                    sliced = pred["results"][: max_k]
                    probs = pred.get("probs", None)
                    for local_idx, (cls_name, mean_score, total_score, used_k) in enumerate(sliced, 1):
                        prob = None
                        if probs is not None and (local_idx - 1) < len(probs):
                            prob = float(probs[local_idx - 1])
                        clip_info = clip_info_by_support.get(cls_name) if clip_info_by_support else None
                        entries.append({
                            "rank": local_idx,
                            "class_raw": cls_name,
                            "class": _display_class_name(cls_name),
                            "mean_score": float(mean_score),
                            "total_score": float(total_score),
                            "reps_used": int(used_k),
                            "prob": prob,
                            "clip_rank": int(clip_info["clip_rank"]) if clip_info else None,
                            "clip_class_name": _display_class_name(clip_info["clip_class_name"]) if clip_info else None,
                            "clip_similarity": float(clip_info["clip_similarity"]) if clip_info else None,
                        })
                    return entries

                true_class_norm = _display_class_name(true_class_raw)
                top_k_final = _make_topk_entries(pred_final, clip_info_final, args.top_k)
                top_k_global = _make_topk_entries(pred_global, clip_info_global, args.top_k)

                pred_final_top1_raw = pred_final["results"][0][0] if pred_final else ""
                pred_global_top1_raw = pred_global["results"][0][0] if pred_global else ""
                pred_final_top1_norm = _display_class_name(pred_final_top1_raw)
                pred_global_top1_norm = _display_class_name(pred_global_top1_raw)
                is_correct_final = pred_final_top1_norm == true_class_norm
                is_correct_global = pred_global_top1_norm == true_class_norm
                if is_correct_final:
                    num_correct_final_top1 += 1
                if is_correct_global:
                    num_correct_global_top1 += 1

                item_out = {
                    "image_path": image_path,
                    "true_class_raw": true_class_raw,
                    "true_class": true_class_norm,
                    "final_topk": {
                        "pred_top1_raw": pred_final_top1_raw,
                        "pred_top1": pred_final_top1_norm,
                        "is_correct_top1": is_correct_final,
                        "top_k": top_k_final,
                        "clip_topk": clip_item.get("final_topk", []) if clip_item else [],
                        "num_clip_candidates": len(topk_list_final),
                        "num_matched_ec_classes": len(allowed_final) if allowed_final else 0,
                    },
                    "global_topk": {
                        "pred_top1_raw": pred_global_top1_raw,
                        "pred_top1": pred_global_top1_norm,
                        "is_correct_top1": is_correct_global,
                        "top_k": top_k_global,
                        "clip_topk": clip_item.get("global_topk", []) if clip_item else [],
                        "num_clip_candidates": len(topk_list_global),
                        "num_matched_ec_classes": len(allowed_global) if allowed_global else 0,
                    },
                }
                all_items.append(item_out)
            else:
                pred = _predict_single_image_scores(
                    image_path, backbone, relation, combiner, support_cache
                )
                if pred is None:
                    skipped_items.append({"image_path": image_path, "reason": "no_valid_scores"})
                    continue
                top1_class_raw = pred["results"][0][0]
                true_class_norm = _display_class_name(true_class_raw)
                top1_class_norm = _display_class_name(top1_class_raw)
                is_correct_top1 = top1_class_norm == true_class_norm
                if is_correct_top1:
                    num_correct_top1 += 1
                top_k_entries = []
                for rank, (cls_name, mean_score, total_score, used_k) in enumerate(pred["results"][: args.top_k], 1):
                    top_k_entries.append({
                        "rank": rank,
                        "class_raw": cls_name,
                        "class": _display_class_name(cls_name),
                        "mean_score": float(mean_score),
                        "total_score": float(total_score),
                        "reps_used": int(used_k),
                        "prob": None,
                    })
                item_out = {
                    "image_path": image_path,
                    "true_class_raw": true_class_raw,
                    "true_class": true_class_norm,
                    "pred_top1_raw": top1_class_raw,
                    "pred_top1": top1_class_norm,
                    "is_correct_top1": is_correct_top1,
                    "top_k": top_k_entries,
                }
                all_items.append(item_out)
        except Exception as exc:
            skipped_items.append({"image_path": image_path, "reason": str(exc)})
            continue

    num_images = len(all_items)
    top1_accuracy = (num_correct_top1 / num_images) if num_images > 0 else 0.0
    top1_accuracy_final = (num_correct_final_top1 / num_images) if num_images > 0 and args.clip_topk_json else 0.0
    top1_accuracy_global = (num_correct_global_top1 / num_images) if num_images > 0 and args.clip_topk_json else 0.0

    output_payload = {
        "mode": "test_dir",
        "test_dir": os.path.abspath(args.test_dir),
        "weights_dir": os.path.abspath(args.weights),
        "ec_json_dir": args.ec_json_dir or None,
        "support_dir": args.support_dir or None,
        "clip_topk_json": os.path.abspath(args.clip_topk_json) if args.clip_topk_json else None,
        "image_per_ec": args.image_per_ec,
        "ec_class_limit": args.ec_class_limit,
        "selected_ec_classes": sorted(list(support_cache.keys())),
        "num_segments": args.num_segments,
        "top_k": args.top_k,
        "test_class_limit": args.test_class_limit,
        "test_images_per_class": args.test_images_per_class,
        "selected_test_classes": selected_class_folders,
        "num_images": num_images,
        "num_correct_top1": num_correct_top1,
        "top1_accuracy": top1_accuracy,
        "num_skipped": len(skipped_items),
        "skipped": skipped_items,
        "results": all_items,
    }
    if args.clip_topk_json:
        output_payload["num_correct_final_top1"] = num_correct_final_top1
        output_payload["num_correct_global_top1"] = num_correct_global_top1
        output_payload["top1_accuracy_final"] = top1_accuracy_final
        output_payload["top1_accuracy_global"] = top1_accuracy_global

    output_path = args.output_json
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("Batch prediction completed")
    print(f"Processed images: {num_images}")
    if args.clip_topk_json:
        print(f"Final top-k:  correct={num_correct_final_top1}, accuracy={top1_accuracy_final:.4f}")
        print(f"Global top-k: correct={num_correct_global_top1}, accuracy={top1_accuracy_global:.4f}")
    else:
        print(f"Top-1 correct: {num_correct_top1}")
        print(f"Top-1 accuracy: {top1_accuracy:.4f}")
    print(f"Skipped: {len(skipped_items)}")
    print(f"Saved JSON: {output_path}")
    print("=" * 80)


def predict(args):
    if args.k_shot is not None:
        if args.image_per_ec == 0:
            args.image_per_ec = args.k_shot
            print(
                "[INFO] --k_shot is deprecated; mapped to --image_per_ec="
                f"{args.image_per_ec}"
            )
        else:
            print("[INFO] --k_shot is deprecated and ignored because --image_per_ec is set.")

    backbone, relation, combiner = load_models(args.weights)

    exemplars = _load_exemplars(args)
    if not exemplars:
        print("No class exemplars loaded.")
        return

    support_cache = _build_support_cache(exemplars, backbone)
    if not support_cache:
        print("No valid support features were prepared.")
        return

    print(f"Ready to predict with {len(support_cache)} classes.")

    if args.test_dir:
        _predict_on_test_dir(args, backbone, relation, combiner, support_cache)
        return

    if not os.path.exists(args.test_image):
        print(f"Test image not found: {args.test_image}")
        return

    print(f"Processing query image: {os.path.basename(args.test_image)}")
    result_payload = _predict_single_image_scores(
        args.test_image, backbone, relation, combiner, support_cache
    )
    if result_payload is None:
        print("No valid class results. Check paths in Ec json/support_dir.")
        return

    _print_single_image_results(result_payload, args.top_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSCIL Prediction Script")
    parser.add_argument(
        "--test_image",
        type=str,
        default="",
        help="Path to a single test image (single-image mode)",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="",
        help="Path to test root dir containing class folders (batch mode)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="output_json/test_predictions_top1.json",
        help="Output json path for --test_dir mode",
    )
    parser.add_argument(
        "--ec_json_dir",
        type=str,
        default="",
        help="Directory containing Ec json files (preferred)",
    )
    parser.add_argument(
        "--support_dir",
        type=str,
        default="",
        help="Fallback directory containing class folders",
    )
    parser.add_argument("--weights", type=str, default="weights", help="Directory containing .pth")
    parser.add_argument(
        "--num_segments",
        type=int,
        default=5,
        help="Number of ordered segments per class; each segment uses head/middle/tail reps",
    )
    parser.add_argument(
        "--image_per_ec",
        type=int,
        default=0,
        help="0 = use all Ec images per class; k > 0 = use top-k Ec images before segmenting",
    )
    parser.add_argument(
        "--k_shot",
        type=int,
        default=None,
        help="Deprecated. Kept only for backward compatibility.",
    )
    parser.add_argument(
        "--ec_classes",
        nargs="*",
        default=None,
        help="List of Ec class names to compare (example: --ec_classes classA classB)",
    )
    parser.add_argument(
        "--ec_class_limit",
        type=int,
        default=0,
        help="Number of first Ec classes (sorted by class name) to compare (0 = all)",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Show/store top-k classes")
    parser.add_argument(
        "--clip_topk_json",
        type=str,
        default="",
        help=(
            "Optional: CLIP top-k result JSON (e.g. output_json/clip_top5_test_first100classes.json). "
            "If provided in --test_dir mode, prediction will only score classes from CLIP top-k."
        ),
    )
    parser.add_argument(
        "--clip_topk_source",
        type=str,
        default="final_topk",
        choices=["final_topk", "global_topk"],
        help="Which CLIP top-k list to use from clip_topk_json.",
    )
    parser.add_argument(
        "--test_class_limit",
        type=int,
        default=0,
        help="Number of first class folders to evaluate in --test_dir mode (0 = all)",
    )
    parser.add_argument(
        "--test_images_per_class",
        type=int,
        default=0,
        help="Number of first images per class in --test_dir mode (0 = all)",
    )

    args = parser.parse_args()

    if not args.test_image and not args.test_dir:
        parser.error("Please provide one of --test_image or --test_dir")
    if args.test_image and args.test_dir:
        parser.error("Please provide only one of --test_image or --test_dir")
    if args.num_segments <= 0:
        parser.error("--num_segments must be > 0")
    if args.image_per_ec < 0:
        parser.error("--image_per_ec must be >= 0")
    if args.ec_class_limit < 0:
        parser.error("--ec_class_limit must be >= 0")
    if args.test_class_limit < 0:
        parser.error("--test_class_limit must be >= 0")
    if args.test_images_per_class < 0:
        parser.error("--test_images_per_class must be >= 0")

    predict(args)