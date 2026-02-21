import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_IMAGE_EXTS


def _normalize_class_name(name: Optional[str]) -> str:
    """
    Remove numeric prefix like '106.Horned_Puffin' -> 'Horned_Puffin'
    Keep the rest intact (spaces, underscores, etc.), trimming whitespace.
    """
    if not name:
        return ""
    return re.sub(r"^\s*\d+\s*[\.\-_]*\s*", "", str(name)).strip()


def _canonical_class_name(name: Optional[str]) -> str:
    """
    Canonical key for matching: strip numeric prefix, replace spaces with '_', lower.
    """
    return _normalize_class_name(name).replace(" ", "_").lower()


def _find_embedding_files(embeddings_dir: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
      final_files:  {raw_class_key_from_filename: path}
      global_files: {raw_class_key_from_filename: path}

    Note: keys are derived from filenames (without suffix), e.g.
      'horned_puffin_final.json' -> key 'horned_puffin'
    """
    final_files: Dict[str, str] = {}
    global_files: Dict[str, str] = {}

    for root, _, files in os.walk(embeddings_dir):
        for filename in files:
            lower = filename.lower()
            full_path = os.path.join(root, filename)

            if lower.endswith("_final.json"):
                class_name = filename[: -len("_final.json")]
                final_files[class_name] = full_path
            elif lower.endswith("_global.json"):
                class_name = filename[: -len("_global.json")]
                global_files[class_name] = full_path

    return final_files, global_files


def _load_embedding_vector(path: str, device: torch.device) -> torch.Tensor:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Embedding file is not a list: {path}")

    vec = torch.tensor(data, dtype=torch.float32, device=device)
    if vec.ndim != 1:
        raise ValueError(f"Embedding vector must be 1D, got shape {tuple(vec.shape)} in {path}")
    return F.normalize(vec, p=2, dim=0)


def _load_clip_model(model_name: str, pretrained: str, device: torch.device):
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "open_clip is not installed in this interpreter.\n"
            f"Interpreter: {sys.executable}\n"
            "Use your venv python to run this script, or install with:\n"
            f"  {sys.executable} -m pip install open-clip-torch"
        ) from exc

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name, pretrained=pretrained
    )
    model = model.to(device).eval()
    return model, preprocess


def _encode_image_with_clip(
    image_path: str,
    model,
    preprocess,
    device: torch.device,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_emb = model.encode_image(image_input)
    image_emb = F.normalize(image_emb, p=2, dim=1)
    return image_emb.squeeze(0)


def _list_dir_files(dir_path: str) -> List[str]:
    return [os.path.join(dir_path, name) for name in os.listdir(dir_path)]


def _collect_test_samples(
    test_dir: str,
    test_class_limit: int = 0,
    test_images_per_class: int = 0,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    samples: list of (image_path, true_class_raw=class_folder_name)
    selected_class_folders: list of class_folder names selected (raw folder names)
    """
    samples: List[Tuple[str, str]] = []
    class_folders_all = sorted(
        d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))
    )

    class_folders = (
        class_folders_all[:test_class_limit]
        if test_class_limit and test_class_limit > 0
        else class_folders_all
    )

    for class_folder in class_folders:
        class_dir = os.path.join(test_dir, class_folder)
        class_images: List[str] = []
        for p in sorted(_list_dir_files(class_dir)):
            if os.path.isfile(p) and _is_image_file(p):
                class_images.append(p)

        if test_images_per_class and test_images_per_class > 0:
            class_images = class_images[:test_images_per_class]

        for p in class_images:
            samples.append((p, class_folder))

    return samples, class_folders


def _build_allowed_keys_from_test_classes(selected_test_classes: List[str]) -> List[str]:
    """
    Convert raw folder names like '106.Horned_Puffin' to canonical keys 'horned_puffin'.
    Preserve order and de-duplicate.
    """
    seen: Set[str] = set()
    ordered: List[str] = []
    for c in selected_test_classes:
        k = _canonical_class_name(c)
        if k and k not in seen:
            seen.add(k)
            ordered.append(k)
    return ordered


def _prepare_embedding_bank_from_allowed_keys(
    class_to_path: Dict[str, str],
    expected_dim: int,
    device: torch.device,
    allowed_keys: List[str],
) -> Tuple[dict, dict]:
    """
    Build bank strictly from allowed_keys (canonical keys from test_dir classes).
    - We match embedding dict keys via canonicalization too.
    - If an allowed key is missing in embeddings_dir, it is counted as missing.
    """
    # Build lookup: canonical_key -> (raw_key_in_dict, path)
    # Note: embedding filenames might already be canonical; still do canonical mapping to be safe.
    canonical_to_entry: Dict[str, Tuple[str, str]] = {}
    for raw_key, path in class_to_path.items():
        ck = _canonical_class_name(raw_key)
        if ck and ck not in canonical_to_entry:
            canonical_to_entry[ck] = (raw_key, path)

    loaded_names: List[str] = []   # store canonical class names (for ranking output)
    loaded_paths: List[str] = []
    loaded_vecs: List[torch.Tensor] = []

    missing = 0
    skipped = 0
    selected = 0

    for ck in allowed_keys:
        selected += 1
        entry = canonical_to_entry.get(ck)
        if entry is None:
            missing += 1
            continue

        _, path = entry
        try:
            vec = _load_embedding_vector(path, device)
        except Exception:
            skipped += 1
            continue

        if vec.shape[0] != expected_dim:
            skipped += 1
            continue

        loaded_names.append(ck)
        loaded_paths.append(path)
        loaded_vecs.append(vec)

    bank = {
        "names": loaded_names,  # canonical keys
        "paths": loaded_paths,
        "matrix": torch.stack(loaded_vecs, dim=0) if loaded_vecs else None,
    }
    summary = {
        "files_found": len(class_to_path),
        "allowed_classes": allowed_keys,
        "files_selected": selected,
        "files_loaded": len(loaded_vecs),
        "files_missing": missing,
        "files_skipped": skipped,
        "selected_classes": loaded_names,  # loaded canonical keys
    }
    return bank, summary


def _prepare_embedding_bank_sorted(
    class_to_path: Dict[str, str],
    expected_dim: int,
    device: torch.device,
    embedding_class_limit: int = 0,
) -> Tuple[dict, dict]:
    """
    Fallback for single-image mode (no test_dir classes to restrict).
    Keep old behavior: take sorted keys then limit.
    """
    class_names = sorted(class_to_path.keys())
    if embedding_class_limit and embedding_class_limit > 0:
        class_names = class_names[:embedding_class_limit]

    loaded_names: List[str] = []
    loaded_paths: List[str] = []
    loaded_vecs: List[torch.Tensor] = []
    skipped = 0

    for raw_key in class_names:
        path = class_to_path[raw_key]
        try:
            vec = _load_embedding_vector(path, device)
        except Exception:
            skipped += 1
            continue

        if vec.shape[0] != expected_dim:
            skipped += 1
            continue

        loaded_names.append(_canonical_class_name(raw_key))
        loaded_paths.append(path)
        loaded_vecs.append(vec)

    bank = {
        "names": loaded_names,
        "paths": loaded_paths,
        "matrix": torch.stack(loaded_vecs, dim=0) if loaded_vecs else None,
    }
    summary = {
        "files_found": len(class_to_path),
        "files_selected": len(class_names),
        "files_loaded": len(loaded_vecs),
        "files_skipped": skipped,
        "selected_classes": loaded_names,
    }
    return bank, summary


def _rank_with_bank(image_emb: torch.Tensor, bank: dict, top_k: int) -> List[dict]:
    matrix = bank["matrix"]
    if matrix is None or matrix.shape[0] == 0:
        return []

    sims = torch.matmul(matrix, image_emb)
    values, indices = torch.topk(sims, k=min(top_k, sims.shape[0]), largest=True)

    result: List[dict] = []
    for rank, (sim, idx) in enumerate(zip(values.tolist(), indices.tolist()), 1):
        result.append(
            {
                "rank": rank,
                "class_name": bank["names"][idx],  # canonical
                "similarity": float(sim),
                "embedding_file": bank["paths"][idx],
            }
        )
    return result


def _predict_single_item(
    image_path: str,
    true_class_raw: Optional[str],
    image_emb: torch.Tensor,
    final_bank: dict,
    global_bank: dict,
    top_k: int,
) -> dict:
    final_topk = _rank_with_bank(image_emb, final_bank, top_k)
    global_topk = _rank_with_bank(image_emb, global_bank, top_k)

    final_top1_raw = final_topk[0]["class_name"] if final_topk else None
    global_top1_raw = global_topk[0]["class_name"] if global_topk else None

    true_class = _normalize_class_name(true_class_raw) if true_class_raw else None
    pred_final_top1 = _normalize_class_name(final_top1_raw) if final_top1_raw else None
    pred_global_top1 = _normalize_class_name(global_top1_raw) if global_top1_raw else None

    item = {
        "image_path": image_path,
        "true_class_raw": true_class_raw,
        "true_class": true_class,
        "pred_final_top1_raw": final_top1_raw,
        "pred_final_top1": pred_final_top1,
        "pred_global_top1_raw": global_top1_raw,
        "pred_global_top1": pred_global_top1,
        "final_topk": final_topk,
        "global_topk": global_topk,
    }

    if true_class_raw is not None:
        true_key = _canonical_class_name(true_class_raw)
        item["is_correct_final_top1"] = (
            _canonical_class_name(final_top1_raw) == true_key if final_top1_raw else False
        )
        item["is_correct_global_top1"] = (
            _canonical_class_name(global_top1_raw) == true_key if global_top1_raw else False
        )

    return item


def run(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    if not os.path.isdir(args.embeddings_dir):
        raise FileNotFoundError(f"Embeddings dir not found: {args.embeddings_dir}")
    if args.top_k <= 0:
        raise ValueError("--top_k must be > 0")

    mode = "single_image" if args.test_image else "test_dir"

    if mode == "single_image":
        if not os.path.isfile(args.test_image):
            raise FileNotFoundError(f"Test image not found: {args.test_image}")
        samples = [(args.test_image, None)]
        selected_test_classes: List[str] = []
        allowed_keys: List[str] = []
    else:
        if not os.path.isdir(args.test_dir):
            raise FileNotFoundError(f"Test dir not found: {args.test_dir}")
        samples, selected_test_classes = _collect_test_samples(
            args.test_dir,
            test_class_limit=args.test_class_limit,
            test_images_per_class=args.test_images_per_class,
        )
        if not samples:
            raise RuntimeError("No image files found under test_dir class folders.")
        allowed_keys = _build_allowed_keys_from_test_classes(selected_test_classes)

    print(f"Using device: {device}")
    print(f"Loading CLIP model: model={args.clip_model}, pretrained={args.clip_pretrained}")
    clip_model, preprocess = _load_clip_model(
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
        device=device,
    )

    first_image_path = samples[0][0]
    first_image_emb = _encode_image_with_clip(
        image_path=first_image_path,
        model=clip_model,
        preprocess=preprocess,
        device=device,
    )

    final_files, global_files = _find_embedding_files(args.embeddings_dir)
    if not final_files and not global_files:
        raise RuntimeError(
            "No *_final.json or *_global.json found in embeddings_dir "
            f"(searched recursively): {args.embeddings_dir}"
        )

    print(f"Found embeddings (recursive): final={len(final_files)}, global={len(global_files)}")

    # Build banks:
    # - test_dir mode: strictly use classes from test_dir (after stripping numeric prefix)
    # - single_image mode: keep old behavior (sorted + limit)
    if mode == "test_dir":
        print(f"Restricting embedding banks to test_dir classes: {len(allowed_keys)} classes")
        final_bank, final_summary = _prepare_embedding_bank_from_allowed_keys(
            class_to_path=final_files,
            expected_dim=first_image_emb.shape[0],
            device=device,
            allowed_keys=allowed_keys,
        )
        global_bank, global_summary = _prepare_embedding_bank_from_allowed_keys(
            class_to_path=global_files,
            expected_dim=first_image_emb.shape[0],
            device=device,
            allowed_keys=allowed_keys,
        )
    else:
        final_bank, final_summary = _prepare_embedding_bank_sorted(
            class_to_path=final_files,
            expected_dim=first_image_emb.shape[0],
            device=device,
            embedding_class_limit=args.embedding_class_limit,
        )
        global_bank, global_summary = _prepare_embedding_bank_sorted(
            class_to_path=global_files,
            expected_dim=first_image_emb.shape[0],
            device=device,
            embedding_class_limit=args.embedding_class_limit,
        )

    if final_summary.get("files_loaded", 0) == 0 and global_summary.get("files_loaded", 0) == 0:
        raise RuntimeError(
            "No embedding vectors loaded (likely dimension mismatch, invalid JSON, or no overlap with test classes)."
        )

    if mode == "test_dir":
        # Nice warnings
        if final_summary.get("files_missing", 0) > 0 or global_summary.get("files_missing", 0) > 0:
            print(
                f"[WARN] Missing embeddings for some test classes: "
                f"final_missing={final_summary.get('files_missing', 0)}, "
                f"global_missing={global_summary.get('files_missing', 0)}"
            )
        print(
            f"Loaded banks: final={final_summary.get('files_loaded', 0)} / {final_summary.get('files_selected', 0)}, "
            f"global={global_summary.get('files_loaded', 0)} / {global_summary.get('files_selected', 0)}"
        )

    results: List[dict] = []
    skipped: List[dict] = []

    iterator = enumerate(samples)
    if mode == "test_dir":
        iterator = enumerate(tqdm(samples, desc="Predicting with CLIP top-k"))

    for idx, (image_path, true_class_raw) in iterator:
        try:
            image_emb = (
                first_image_emb
                if idx == 0
                else _encode_image_with_clip(
                    image_path=image_path,
                    model=clip_model,
                    preprocess=preprocess,
                    device=device,
                )
            )
            item = _predict_single_item(
                image_path=image_path,
                true_class_raw=true_class_raw,
                image_emb=image_emb,
                final_bank=final_bank,
                global_bank=global_bank,
                top_k=args.top_k,
            )
            results.append(item)
        except Exception as exc:
            skipped.append({"image_path": image_path, "reason": str(exc)})

    output: dict = {
        "mode": mode,
        "embeddings_dir": os.path.abspath(args.embeddings_dir),
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "device": str(device),
        "top_k": args.top_k,
        "embedding_class_limit": args.embedding_class_limit,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "final_summary": final_summary,
        "global_summary": global_summary,
    }

    if mode == "single_image":
        output["test_image"] = os.path.abspath(args.test_image)
        output["num_images"] = len(results)
        output["num_skipped"] = len(skipped)
        output["skipped"] = skipped
        output["result"] = results[0] if results else None
        output["final_topk"] = results[0]["final_topk"] if results else []
        output["global_topk"] = results[0]["global_topk"] if results else []
    else:
        num_final_correct = sum(1 for x in results if x.get("is_correct_final_top1", False))
        num_global_correct = sum(1 for x in results if x.get("is_correct_global_top1", False))
        num_images = len(results)

        output.update(
            {
                "test_dir": os.path.abspath(args.test_dir),
                "test_class_limit": args.test_class_limit,
                "test_images_per_class": args.test_images_per_class,
                "selected_test_classes": selected_test_classes,
                "canonical_test_classes": allowed_keys,
                "num_images": num_images,
                "num_skipped": len(skipped),
                "skipped": skipped,
                "num_correct_final_top1": num_final_correct,
                "num_correct_global_top1": num_global_correct,
                "final_top1_accuracy": (num_final_correct / num_images if num_images else 0.0),
                "global_top1_accuracy": (num_global_correct / num_images if num_images else 0.0),
                "results": results,
            }
        )

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Encode image(s) with CLIP and rank top classes by cosine similarity "
            "against *_final.json and *_global.json embeddings."
        )
    )
    parser.add_argument(
        "--test_image",
        type=str,
        default="",
        help="Path to one test image (single-image mode)",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="",
        help="Path to test root dir containing class folders (batch mode)",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        required=True,
        help="Root folder containing class embedding JSON files",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Top-K classes to keep")
    parser.add_argument(
        "--embedding_class_limit",
        type=int,
        default=100,
        help="(single-image mode only) Number of first embedding classes (sorted) to compare (0 = all)",
    )
    parser.add_argument(
        "--test_class_limit",
        type=int,
        default=100,
        help="Number of first test class folders to evaluate in --test_dir mode (0 = all)",
    )
    parser.add_argument(
        "--test_images_per_class",
        type=int,
        default=0,
        help="Number of first images per test class in --test_dir mode (0 = all)",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B-32",
        help="open_clip model name (must match your embedding pipeline)",
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="laion2b_s34b_b79k",
        help="open_clip pretrained tag (must match your embedding pipeline)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="output_json/clip_top5_result.json",
        help="Path to output result JSON",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    cli_args = parser.parse_args()

    if not cli_args.test_image and not cli_args.test_dir:
        parser.error("Please provide one of --test_image or --test_dir")
    if cli_args.test_image and cli_args.test_dir:
        parser.error("Please provide only one of --test_image or --test_dir")
    if cli_args.top_k <= 0:
        parser.error("--top_k must be > 0")
    if cli_args.embedding_class_limit < 0:
        parser.error("--embedding_class_limit must be >= 0")
    if cli_args.test_class_limit < 0:
        parser.error("--test_class_limit must be >= 0")
    if cli_args.test_images_per_class < 0:
        parser.error("--test_images_per_class must be >= 0")

    output_data = run(cli_args)

    print("=" * 80)
    print(f"Saved result JSON: {cli_args.output_json}")
    if output_data["mode"] == "single_image":
        top_final = output_data["final_topk"][0]["class_name"] if output_data["final_topk"] else "N/A"
        top_global = output_data["global_topk"][0]["class_name"] if output_data["global_topk"] else "N/A"
        print(f"Top final class:  {top_final}")
        print(f"Top global class: {top_global}")
    else:
        print(f"Processed images: {output_data['num_images']}")
        print(f"Final top1 accuracy:  {output_data['final_top1_accuracy']:.4f}")
        print(f"Global top1 accuracy: {output_data['global_top1_accuracy']:.4f}")
    print("=" * 80)
