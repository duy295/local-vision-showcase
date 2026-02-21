import argparse
import json
import os
import sys # Phải import sys trước
from collections import defaultdict

# Thêm đường dẫn vào sys.path TRƯỚC KHI import module nội bộ
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Bây giờ mới import các module từ thư mục backbone
from backbone.feature_extract import HybridResNetBackbone
from backbone.relation_net import BilinearRelationNet


class ScoreCombinerNet(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, scores1, scores2, scores3):
        combined_input = torch.stack([scores1, scores2, scores3], dim=1)
        output = self.net(combined_input)
        # SỬA Ở ĐÂY: Dùng Sigmoid và Scaling y hệt lúc Train
        return torch.sigmoid(output.squeeze(-1)) * 0.98
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageFolderWithPath(Dataset):
    def __init__(self, image_root, transform):
        self.dataset = ImageFolder(image_root)
        self.samples = self.dataset.samples
        self.transform = transform
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label, path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Ec set for each class by mean intra-class similarity."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to dataset root or images folder.",
    )
    parser.add_argument(
        "--backbone_pth",
        type=str,
        required=True,
        help="Path to trained backbone .pth",
    )
    parser.add_argument(
        "--relation_pth",
        type=str,
        required=True,
        help="Path to trained relation .pth",
    )
    parser.add_argument(
        "--score_combiner_pth",
        type=str,
        default="",
        help="Optional path to trained score_combiner .pth",
    )
    parser.add_argument(
        "--ec_size",
        type=int,
        required=True,
        help="Top-k images kept for Ec per class.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Ec_output",
        help="Directory to save per-class Ec files.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    return parser.parse_args()


'''def resolve_images_dir(root):
    if root.replace("\\", "/").endswith("/images") or root.replace("\\", "/").endswith("images"):
        return root
    return os.path.join(root, "images")
'''

def resolve_images_dir(root):
    if root.replace("\\", "/").endswith("/train") or root.replace("\\", "/").endswith("train"):
        return root
    return os.path.join(root, "train")


def load_state_dict_flexible(model, ckpt_path, model_name):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{model_name} checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, dict):
        cleaned = {}
        for key, val in state.items():
            if key.startswith("module."):
                cleaned[key[7:]] = val
            else:
                cleaned[key] = val
        state = cleaned

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)


@torch.no_grad()
def extract_all_features(backbone, dataloader, device):
    feat_all = []
    global_all = []
    patch_all = []
    labels_all = []
    paths_all = []

    for images, labels, paths in dataloader:
        images = images.to(device, non_blocking=True)
        feat, global_feat, patch_feat = backbone(images)

        feat_all.append(feat.cpu())
        global_all.append(global_feat.cpu())
        patch_all.append(patch_feat.cpu())
        labels_all.extend(labels.tolist())
        paths_all.extend(paths)

    feat_all = torch.cat(feat_all, dim=0)
    global_all = torch.cat(global_all, dim=0)
    patch_all = torch.cat(patch_all, dim=0)
    labels_all = torch.tensor(labels_all, dtype=torch.long)
    return feat_all, global_all, patch_all, labels_all, paths_all


@torch.no_grad()
def compute_class_rankings(
    class_indices,
    feat_all,
    global_all,
    patch_all,
    paths_all,
    relation,
    score_combiner,
    device,
):
    class_feat = feat_all[class_indices].to(device)
    class_global = global_all[class_indices].to(device)
    class_patch = patch_all[class_indices].to(device)

    n = class_feat.size(0)
    results = []
    for i in range(n):
        a_feat = class_feat[i].unsqueeze(0).expand(n, -1)
        a_global = class_global[i].unsqueeze(0).expand(n, -1)
        a_patch = class_patch[i].unsqueeze(0).expand(n, -1)
        def flatten(t):
            return t.view(t.size(0), -1)
        s1 = relation(flatten(a_feat), flatten(class_feat))
        s2 = relation(flatten(a_global), flatten(class_global))
        s3 = relation(flatten(a_patch), flatten(class_patch))

        if score_combiner is not None:
            final_scores = score_combiner(s1, s2, s3)
        else:
            final_scores = torch.clamp((s1 + s2 + s3) / 3.0, 0.0, 1.0)

        if n > 1:
            mask = torch.ones(n, dtype=torch.bool, device=device)
            mask[i] = False
            mean_sim = final_scores[mask].mean().item()
        else:
            mean_sim = final_scores[i].item()

        global_idx = class_indices[i]
        results.append(
            {
                "path": paths_all[global_idx],
                "mean_similarity": float(mean_sim),
            }
        )

    results.sort(key=lambda x: x["mean_similarity"], reverse=True)
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank
    return results


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but not available.")
    if args.ec_size <= 0:
        raise ValueError("--ec_size must be > 0")

    device = torch.device(args.device)
    images_dir = resolve_images_dir(args.root)
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImageFolderWithPath(images_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    backbone = HybridResNetBackbone().to(device)
    relation = BilinearRelationNet().to(device)
    score_combiner = None

    load_state_dict_flexible(backbone, args.backbone_pth, "Backbone")
    load_state_dict_flexible(relation, args.relation_pth, "Relation")

    if args.score_combiner_pth:
        score_combiner = ScoreCombinerNet(hidden_dim=64).to(device)
        load_state_dict_flexible(score_combiner, args.score_combiner_pth, "ScoreCombiner")
        score_combiner.eval()

    backbone.eval()
    relation.eval()

    feat_all, global_all, patch_all, labels_all, paths_all = extract_all_features(
        backbone, loader, device
    )

    class_to_indices = defaultdict(list)
    for idx, label in enumerate(labels_all.tolist()):
        class_to_indices[label].append(idx)

    os.makedirs(args.output_dir, exist_ok=True)
    merged_output = {}

    for class_idx, indices in class_to_indices.items():
        class_name = dataset.classes[class_idx]
        ranked_items = compute_class_rankings(
            indices,
            feat_all,
            global_all,
            patch_all,
            paths_all,
            relation,
            score_combiner,
            device,
        )
        ec_items = ranked_items[: min(args.ec_size, len(ranked_items))]
        payload = {
            "class_name": class_name,
            "class_idx": int(class_idx),
            "num_images_in_class": len(indices),
            "ec_size_requested": int(args.ec_size),
            "ec_size_actual": len(ec_items),
            "Ec": ec_items,
        }

        file_name = f"{class_name}_Ec.json"
        with open(os.path.join(args.output_dir, file_name), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        merged_output[class_name] = ec_items

    merged_path = os.path.join(args.output_dir, "Ec_all_classes.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged_output, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved Ec files to: {args.output_dir}")
    print(f"Merged file: {merged_path}")


if __name__ == "__main__":
    main()
