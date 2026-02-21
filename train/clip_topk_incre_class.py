import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ====== Base-100 class list (ORDERED, exactly as you provided) ======
BASE100_CLASSES_RAW: List[str] = [
    "001.Black_footed_Albatross",
    "002.Laysan_Albatross",
    "003.Sooty_Albatross",
    "004.Groove_billed_Ani",
    "005.Crested_Auklet",
    "006.Least_Auklet",
    "007.Parakeet_Auklet",
    "008.Rhinoceros_Auklet",
    "009.Brewer_Blackbird",
    "010.Red_winged_Blackbird",
    "011.Rusty_Blackbird",
    "012.Yellow_headed_Blackbird",
    "013.Bobolink",
    "014.Indigo_Bunting",
    "015.Lazuli_Bunting",
    "016.Painted_Bunting",
    "017.Cardinal",
    "018.Spotted_Catbird",
    "019.Gray_Catbird",
    "020.Yellow_breasted_Chat",
    "021.Eastern_Towhee",
    "022.Chuck_will_Widow",
    "023.Brandt_Cormorant",
    "024.Red_faced_Cormorant",
    "025.Pelagic_Cormorant",
    "026.Bronzed_Cowbird",
    "027.Shiny_Cowbird",
    "028.Brown_Creeper",
    "029.American_Crow",
    "030.Fish_Crow",
    "031.Black_billed_Cuckoo",
    "032.Mangrove_Cuckoo",
    "033.Yellow_billed_Cuckoo",
    "034.Gray_crowned_Rosy_Finch",
    "035.Purple_Finch",
    "036.Northern_Flicker",
    "037.Acadian_Flycatcher",
    "038.Great_Crested_Flycatcher",
    "039.Least_Flycatcher",
    "040.Olive_sided_Flycatcher",
    "041.Scissor_tailed_Flycatcher",
    "042.Vermilion_Flycatcher",
    "043.Yellow_bellied_Flycatcher",
    "044.Frigatebird",
    "045.Northern_Fulmar",
    "046.Gadwall",
    "047.American_Goldfinch",
    "048.European_Goldfinch",
    "049.Boat_tailed_Grackle",
    "050.Eared_Grebe",
    "051.Horned_Grebe",
    "052.Pied_billed_Grebe",
    "053.Western_Grebe",
    "054.Blue_Grosbeak",
    "055.Evening_Grosbeak",
    "056.Pine_Grosbeak",
    "057.Rose_breasted_Grosbeak",
    "058.Pigeon_Guillemot",
    "059.California_Gull",
    "060.Glaucous_winged_Gull",
    "061.Heermann_Gull",
    "062.Herring_Gull",
    "063.Ivory_Gull",
    "064.Ring_billed_Gull",
    "065.Slaty_backed_Gull",
    "066.Western_Gull",
    "067.Anna_Hummingbird",
    "068.Ruby_throated_Hummingbird",
    "069.Rufous_Hummingbird",
    "070.Green_Violetear",
    "071.Long_tailed_Jaeger",
    "072.Pomarine_Jaeger",
    "073.Blue_Jay",
    "074.Florida_Jay",
    "075.Green_Jay",
    "076.Dark_eyed_Junco",
    "077.Tropical_Kingbird",
    "078.Gray_Kingbird",
    "079.Belted_Kingfisher",
    "080.Green_Kingfisher",
    "081.Pied_Kingfisher",
    "082.Ringed_Kingfisher",
    "083.White_breasted_Kingfisher",
    "084.Red_legged_Kittiwake",
    "085.Horned_Lark",
    "086.Pacific_Loon",
    "087.Mallard",
    "088.Western_Meadowlark",
    "089.Hooded_Merganser",
    "090.Red_breasted_Merganser",
    "091.Mockingbird",
    "092.Nighthawk",
    "093.Clark_Nutcracker",
    "094.White_breasted_Nuthatch",
    "095.Baltimore_Oriole",
    "096.Hooded_Oriole",
    "097.Orchard_Oriole",
    "098.Scott_Oriole",
    "099.Ovenbird",
    "100.Brown_Pelican",
]


def _is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_IMAGE_EXTS


def _normalize_class_name(name: Optional[str]) -> str:
    """
    Remove numeric prefix like '106.Horned_Puffin' -> 'Horned_Puffin'
    """
    if not name:
        return ""
    return re.sub(r"^\s*\d+\s*[\.\-_]*\s*", "", str(name)).strip()


def _canonical_class_name(name: Optional[str]) -> str:
    """
    Canonical key: strip numeric prefix, replace spaces with '_', lowercase.
    """
    return _normalize_class_name(name).replace(" ", "_").lower()


def _build_base100_keys() -> Tuple[List[str], Dict[str, str]]:
    """
    Returns:
      base_keys_ordered: canonical keys in the same order as BASE100_CLASSES_RAW
      key_to_display: canonical -> original '001.Name' display
    """
    base_keys_ordered: List[str] = []
    key_to_display: Dict[str, str] = {}
    seen = set()
    for raw in BASE100_CLASSES_RAW:
        k = _canonical_class_name(raw)
        if not k:
            continue
        if k in seen:
            continue
        seen.add(k)
        base_keys_ordered.append(k)
        key_to_display[k] = raw
    return base_keys_ordered, key_to_display


def _find_embedding_files(embeddings_dir: str) -> Tuple[Dict[str, str], Dict[str, str]]:
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


def _encode_image_with_clip(image_path: str, model, preprocess, device: torch.device) -> torch.Tensor:
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


def _prepare_base100_bank(
    class_to_path: Dict[str, str],
    expected_dim: int,
    device: torch.device,
    base_keys_ordered: List[str],
) -> Tuple[dict, dict]:
    """
    Build bank strictly from base_keys_ordered (canonical keys), in that order.
    Match embedding filenames via canonicalization.

    Returns bank:
      names: canonical keys (ordered)
      paths: matched embedding file paths (ordered)
      matrix: stacked vectors (ordered)
    """
    # canonical_key -> path (first occurrence wins)
    canonical_to_path: Dict[str, str] = {}
    for raw_key, path in class_to_path.items():
        ck = _canonical_class_name(raw_key)
        if ck and ck not in canonical_to_path:
            canonical_to_path[ck] = path

    loaded_names: List[str] = []
    loaded_paths: List[str] = []
    loaded_vecs: List[torch.Tensor] = []

    missing: List[str] = []
    skipped: List[Tuple[str, str]] = []  # (key, reason)

    for ck in base_keys_ordered:
        path = canonical_to_path.get(ck)
        if path is None:
            missing.append(ck)
            continue
        try:
            vec = _load_embedding_vector(path, device)
            if vec.shape[0] != expected_dim:
                skipped.append((ck, f"dim_mismatch expected={expected_dim} got={vec.shape[0]}"))
                continue
        except Exception as exc:
            skipped.append((ck, str(exc)))
            continue

        loaded_names.append(ck)
        loaded_paths.append(path)
        loaded_vecs.append(vec)

    bank = {
        "names": loaded_names,
        "paths": loaded_paths,
        "matrix": torch.stack(loaded_vecs, dim=0) if loaded_vecs else None,
    }
    summary = {
        "files_found": len(class_to_path),
        "base100_total": len(base_keys_ordered),
        "files_loaded": len(loaded_names),
        "files_missing": len(missing),
        "files_skipped": len(skipped),
        "missing_classes": missing,
        "skipped_classes": skipped,
        "selected_classes": loaded_names,  # canonical keys loaded
    }
    return bank, summary


def _rank_with_bank(image_emb: torch.Tensor, bank: dict, top_k: int, key_to_display: Dict[str, str]) -> List[dict]:
    matrix = bank["matrix"]
    if matrix is None or matrix.shape[0] == 0:
        return []

    sims = torch.matmul(matrix, image_emb)
    values, indices = torch.topk(sims, k=min(top_k, sims.shape[0]), largest=True)

    out: List[dict] = []
    for rank, (sim, idx) in enumerate(zip(values.tolist(), indices.tolist()), 1):
        ck = bank["names"][idx]
        out.append(
            {
                "rank": rank,
                "class_name": ck,
                #"class_key": ck,  # canonical
                #"class_display": key_to_display.get(ck, ck),  # '001.Name'
                "similarity": float(sim),
                "embedding_file": bank["paths"][idx],
            }
        )
    return out


def _predict_single_item(
    image_path: str,
    true_class_raw: Optional[str],
    image_emb: torch.Tensor,
    final_bank: dict,
    global_bank: dict,
    top_k: int,
    key_to_display: Dict[str, str],
) -> dict:
    final_topk = _rank_with_bank(image_emb, final_bank, top_k, key_to_display)
    global_topk = _rank_with_bank(image_emb, global_bank, top_k, key_to_display)

    final_top1_key = final_topk[0]["class_name"] if final_topk else None
    global_top1_key = global_topk[0]["class_name"] if global_topk else None

    item = {
        "image_path": image_path,
        "true_class_raw": true_class_raw,
        "true_class": _normalize_class_name(true_class_raw) if true_class_raw else None,
        "final_topk": final_topk,
        "global_topk": global_topk,
    }

    if true_class_raw is not None:
        true_key = _canonical_class_name(true_class_raw)
        item["is_correct_final_top1"] = (final_top1_key == true_key) if final_top1_key else False
        item["is_correct_global_top1"] = (global_top1_key == true_key) if global_top1_key else False

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

    print(f"Using device: {device}")
    print(f"Loading CLIP model: model={args.clip_model}, pretrained={args.clip_pretrained}")
    clip_model, preprocess = _load_clip_model(args.clip_model, args.clip_pretrained, device)

    # Encode one image to infer embedding dim
    first_image_path = samples[0][0]
    first_image_emb = _encode_image_with_clip(first_image_path, clip_model, preprocess, device)
    expected_dim = first_image_emb.shape[0]

    # Build base-100 class keys (ordered)
    base_keys_ordered, key_to_display = _build_base100_keys()
    print(f"Base100 classes (fixed): {len(base_keys_ordered)}")

    # Find all embedding files, then build banks STRICTLY from the base-100 list
    final_files, global_files = _find_embedding_files(args.embeddings_dir)
    if not final_files and not global_files:
        raise RuntimeError(
            "No *_final.json or *_global.json found in embeddings_dir "
            f"(searched recursively): {args.embeddings_dir}"
        )
    print(f"Found embeddings (recursive): final={len(final_files)}, global={len(global_files)}")

    final_bank, final_summary = _prepare_base100_bank(final_files, expected_dim, device, base_keys_ordered)
    global_bank, global_summary = _prepare_base100_bank(global_files, expected_dim, device, base_keys_ordered)

    if (final_bank["matrix"] is None or final_bank["matrix"].shape[0] == 0) and (
        global_bank["matrix"] is None or global_bank["matrix"].shape[0] == 0
    ):
        raise RuntimeError(
            "No embeddings loaded for base-100. Check embedding filenames and vector dims."
        )

    print(
        f"Loaded base100 banks: final={final_summary['files_loaded']}/{final_summary['base100_total']} "
        f"(missing={final_summary['files_missing']}, skipped={final_summary['files_skipped']}), "
        f"global={global_summary['files_loaded']}/{global_summary['base100_total']} "
        f"(missing={global_summary['files_missing']}, skipped={global_summary['files_skipped']})"
    )

    results: List[dict] = []
    skipped_imgs: List[dict] = []

    iterator = enumerate(samples)
    if mode == "test_dir":
        iterator = enumerate(tqdm(samples, desc="Predicting with CLIP top-k (base100-only)"))

    for idx, (image_path, true_class_raw) in iterator:
        try:
            image_emb = first_image_emb if idx == 0 else _encode_image_with_clip(
                image_path, clip_model, preprocess, device
            )
            item = _predict_single_item(
                image_path=image_path,
                true_class_raw=true_class_raw,
                image_emb=image_emb,
                final_bank=final_bank,
                global_bank=global_bank,
                top_k=args.top_k,
                key_to_display=key_to_display,
            )
            results.append(item)
        except Exception as exc:
            skipped_imgs.append({"image_path": image_path, "reason": str(exc)})

    output: dict = {
        "mode": mode,
        "embeddings_dir": os.path.abspath(args.embeddings_dir),
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "device": str(device),
        "top_k": args.top_k,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "base100_classes_raw": BASE100_CLASSES_RAW,
        "base100_classes_canonical": base_keys_ordered,
        "final_summary": final_summary,
        "global_summary": global_summary,
    }

    if mode == "single_image":
        output["test_image"] = os.path.abspath(args.test_image)
        output["num_images"] = len(results)
        output["num_skipped"] = len(skipped_imgs)
        output["skipped"] = skipped_imgs
        output["result"] = results[0] if results else None
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
                "num_images": num_images,
                "num_skipped": len(skipped_imgs),
                "skipped": skipped_imgs,
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
            "Base100-only CLIP ranking: compare images only against embeddings "
            "of the fixed first 100 CUB classes (001-100), for both *_final.json and *_global.json."
        )
    )
    parser.add_argument("--test_image", type=str, default="", help="Path to one test image (single-image mode)")
    parser.add_argument("--test_dir", type=str, default="", help="Path to test root dir containing class folders (batch mode)")
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Root folder containing class embedding JSON files")
    parser.add_argument("--top_k", type=int, default=10, help="Top-K classes to keep")
    parser.add_argument("--test_class_limit", type=int, default=0, help="Number of first test class folders (0=all)")
    parser.add_argument("--test_images_per_class", type=int, default=0, help="Number of first images per test class (0=all)")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32", help="open_clip model name (must match your embedding pipeline)")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k", help="open_clip pretrained tag (must match your embedding pipeline)")
    parser.add_argument("--output_json", type=str, default="output_json/clip_topk_base100_only.json", help="Path to output result JSON")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if not args.test_image and not args.test_dir:
        parser.error("Please provide one of --test_image or --test_dir")
    if args.test_image and args.test_dir:
        parser.error("Please provide only one of --test_image or --test_dir")
    if args.top_k <= 0:
        parser.error("--top_k must be > 0")
    if args.test_class_limit < 0:
        parser.error("--test_class_limit must be >= 0")
    if args.test_images_per_class < 0:
        parser.error("--test_images_per_class must be >= 0")

    out = run(args)

    print("=" * 80)
    print(f"Saved result JSON: {args.output_json}")
    if out["mode"] == "single_image":
        r = out.get("result") or {}
        ft = (r.get("final_topk") or [{}])[0].get("class_name", "N/A")
        gt = (r.get("global_topk") or [{}])[0].get("class_name", "N/A")
        print(f"Top final class:  {ft}")
        print(f"Top global class: {gt}")
    else:
        print(f"Processed images: {out['num_images']}")
        print(f"Final top1 accuracy:  {out['final_top1_accuracy']:.4f}")
        print(f"Global top1 accuracy: {out['global_top1_accuracy']:.4f}")
    print("=" * 80)
