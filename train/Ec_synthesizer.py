"""
ec_synthesizer.py
─────────────────────────────────────────────────────────────────────
Tổng hợp Exemplar Set (Ec) cho Incremental Classes.

FIX đã áp dụng:
  FIX-1: preprocess_for_clip — bỏ số prefix + underscore trước khi CLIP encode
          "033.Yellow_billed_Cuckoo" → "a photo of a Yellow Billed Cuckoo"
  FIX-2: load_clip_topk_json — basename fallback tránh miss cross-OS path
  FIX-3: mean_sim_to_fewshot_backbone — đúng chiều query/support:
          few-shot làm query, exemplar làm support → average qua N
  FIX-4: hybrid_lambda expose ra CLI --hybrid_lambda
  FIX-5: class_name output = "Geococcyx" (không phải "101.Geococcyx")
          file name = "101.geococcyx_ec.json"
  FIX-6: dedup dùng backbone cosine, fallback CLIP nếu không có backbone

IMPROVE:
  + _pick_segment_representatives: near / 1/3 / 2/3 / far centroid
  + score_segments hybrid: z-norm(magnitude) * λ + z-norm(Borda) * (1-λ)
  + mean_similarity_to_fewshot_backbone: field thực cho mọi entry,
    không extract few-shot lần 2

Output JSON khớp base class Ec:
  {
    "class_name": "Geococcyx",
    "class_idx": 101,
    "num_images_in_class": 10,
    "ec_size_requested": 30,
    "ec_size_actual": 28,
    "Ec": [
      {"path": "...", "mean_similarity": 3.0, "rank": 1, "source": "fewshot",
       "mean_similarity_to_fewshot_backbone": 0.812}
    ]
  }
"""

import argparse
import json
import os
import sys
import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
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
            torch.nn.Linear(hidden_dim // 2, 1),
        )
        self.temperature = 0.5

    def forward(self, scores1, scores2, scores3):
        combined_input = torch.stack([scores1, scores2, scores3], dim=1)
        output = self.net(combined_input)
        return torch.sigmoid(output.squeeze(-1) / self.temperature) * 0.98


# ──────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────

def normalize_name(name: Optional[str]) -> str:
    """
    "033.Yellow_billed_Cuckoo" → "yellow_billed_cuckoo"
    Dùng làm dict key / cache filename — KHÔNG dùng để encode CLIP.
    """
    if not name:
        return ""
    s = str(name).strip()
    s = re.sub(r"^\s*\d+\s*[\.\-_]*\s*", "", s)
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s.lower().strip("_")


def preprocess_for_clip(name: Optional[str]) -> str:
    """
    FIX-1: Chuyển tên class thành chuỗi tự nhiên trước khi CLIP encode.
    "033.Yellow_billed_Cuckoo"  → "Yellow Billed Cuckoo"
    "Fork_tailed_Flycatcher"    → "Fork Tailed Flycatcher"
    "002.Laysan_Albatross"      → "Laysan Albatross"
    """
    if not name:
        return ""
    s = str(name).strip()
    s = re.sub(r"^\s*\d+\s*[\.\-_]+\s*", "", s)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.title()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: (N, D), b: (M, D) → (N, M)"""
    return torch.mm(F.normalize(a, p=2, dim=1),
                    F.normalize(b, p=2, dim=1).t())


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_clip_topk_json(topk_json_path: str) -> Dict[str, dict]:
    """
    FIX-2: Lưu cả abs path lẫn basename.
    JSON lưu "E:\\DATASET\\img.jpg" nhưng runtime Linux/relative → basename vẫn tìm được.
    """
    data  = load_json(topk_json_path)
    items = data["results"] if isinstance(data, dict) and "results" in data else data
    if not isinstance(items, list):
        raise ValueError("CLIP topK json must be list or dict with 'results' list")

    mapping: Dict[str, dict] = {}
    for it in items:
        p = (it.get("image_path")
            or it.get("path")
            or it.get("img_path")
            or it.get("image")
            or "")
        if not p:
            continue
        mapping[os.path.normpath(os.path.abspath(p))] = it
        mapping[os.path.basename(p)] = it   # FIX-2: basename fallback

    print(f"Loaded CLIP topK: {len(mapping) // 2} images from: {topk_json_path}")
    return mapping


# ──────────────────────────────────────────────────────────
#  CLIP Feature Extractor
# ──────────────────────────────────────────────────────────

class CLIPExtractor:
    def __init__(self,
                 model_name: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k",
                 device: str = "cpu",
                 cache_dir: Optional[str] = None):
        self.device      = device
        self.cache_dir   = cache_dir
        self.model_name  = model_name
        self.pretrained  = pretrained
        self._model      = None
        self._preprocess = None
        self._tokenizer  = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import open_clip
        except ImportError:
            raise ImportError("pip install open-clip-torch")
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device)
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        print(f"Loaded open_clip {self.model_name} / {self.pretrained}")

    def get_text_feat(self, text: str) -> torch.Tensor:
        """FIX-1: preprocess trước khi encode."""
        clean  = preprocess_for_clip(text)
        c_key  = normalize_name(text)
        c_path = os.path.join(self.cache_dir, f"text_{c_key}.pt") \
            if self.cache_dir else None
        if c_path and os.path.exists(c_path):
            return torch.load(c_path, map_location=self.device, weights_only=True)

        self._load_model()
        with torch.no_grad():
            tokens = self._tokenizer([f"a photo of a {clean}"]).to(self.device)
            feat   = self._model.encode_text(tokens).float()
            feat   = F.normalize(feat, p=2, dim=1)

        if c_path:
            os.makedirs(os.path.dirname(c_path) or ".", exist_ok=True)
            torch.save(feat.cpu(), c_path)
        return feat

    def get_image_feat(self, image_path: str) -> torch.Tensor:
        c_path = os.path.join(
            self.cache_dir, normalize_name(Path(image_path).stem) + ".pt"
        ) if self.cache_dir else None
        if c_path and os.path.exists(c_path):
            return torch.load(c_path, map_location=self.device, weights_only=True)

        self._load_model()
        from PIL import Image
        img = self._preprocess(
            Image.open(image_path).convert("RGB")
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self._model.encode_image(img).float()
            feat = F.normalize(feat, p=2, dim=1)

        if c_path:
            os.makedirs(os.path.dirname(c_path) or ".", exist_ok=True)
            torch.save(feat.cpu(), c_path)
        return feat

    def get_batch_image_feats(self, image_paths: List[str]) -> torch.Tensor:
        return torch.cat(
            [self.get_image_feat(p)
             for p in tqdm(image_paths, desc="CLIP extract", leave=False)],
            dim=0)


# ──────────────────────────────────────────────────────────
#  EC Scorer — Backbone + Relation + Combiner
# ──────────────────────────────────────────────────────────

class ECScorer:
    def __init__(self, weights_dir: str, device: str = "cpu",
                 feat_cache_dir: Optional[str] = None):
        self.device         = device
        self.weights_dir    = weights_dir
        self.feat_cache_dir = feat_cache_dir
        self.backbone  = None
        self.relation  = None
        self.combiner  = None
        self._loaded   = False

    def load(self):
        if self._loaded:
            return
        self.backbone = HybridResNetBackbone().to(self.device)
        self.relation = BilinearRelationNet().to(self.device)
        self.combiner = ScoreCombinerNet().to(self.device)
        for name, attr in [("backbone_full.pth", "backbone"),
                            ("relation_full.pth", "relation"),
                            ("score_combiner_full.pth", "combiner")]:
            getattr(self, attr).load_state_dict(torch.load(
                os.path.join(self.weights_dir, name),
                map_location=self.device, weights_only=False))
            getattr(self, attr).eval()
        self._loaded = True
        print(f"Loaded EC models from: {self.weights_dir}")

    def _feat_cache_path(self, img_path: str) -> Optional[str]:
        if not self.feat_cache_dir:
            return None
        return os.path.join(self.feat_cache_dir,
                            f"ec_{normalize_name(Path(img_path).stem)}.pt")

    def extract_features(self, img_path: str
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cp = self._feat_cache_path(img_path)
        if cp and os.path.exists(cp):
            s = torch.load(cp, map_location=self.device, weights_only=True)
            return s["feat"], s["global"], s["patch"]

        self.load()
        from PIL import Image
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat, gf, pf = self.backbone(img)
        feat = F.normalize(feat.float(), p=2, dim=1)
        gf   = F.normalize(gf.float(),   p=2, dim=1)
        pf   = F.normalize(pf.float(),   p=2, dim=1)
        if cp:
            os.makedirs(os.path.dirname(cp) or ".", exist_ok=True)
            torch.save({"feat": feat.cpu(), "global": gf.cpu(), "patch": pf.cpu()}, cp)
        return feat, gf, pf

    def extract_batch(self, img_paths: List[str]
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats, gs, ps = [], [], []
        for p in tqdm(img_paths, desc="EC extract", leave=False):
            f, g, pa = self.extract_features(p)
            feats.append(f); gs.append(g); ps.append(pa)
        return torch.cat(feats, 0), torch.cat(gs, 0), torch.cat(ps, 0)

    @torch.no_grad()
    def score_query_vs_support(
        self,
        q_feat: torch.Tensor, q_global: torch.Tensor, q_patch: torch.Tensor,
        s_feat: torch.Tensor, s_global: torch.Tensor, s_patch: torch.Tensor,
    ) -> Tuple[float, float, torch.Tensor]:
        self.load()
        k   = s_feat.shape[0]
        q_f = q_feat.repeat(k, 1)
        q_g = q_global.repeat(k, 1)
        q_p = q_patch.repeat(k, 1)
        s1  = self.relation(q_f, s_feat)
        s2  = self.relation(q_g, s_global)
        s3  = self.relation(q_p, s_patch)
        out = self.combiner(s1, s2, s3)
        return float(out.mean()), float(out.sum()), out


# ──────────────────────────────────────────────────────────
#  FIX-3: mean_sim_to_fewshot_backbone (đúng chiều)
# ──────────────────────────────────────────────────────────

@torch.no_grad()
def mean_sim_to_fewshot_backbone(
    ec_scorer: ECScorer,
    exemplar_triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # (1,D)x3
    fs_triplet:       Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # (N,D)x3
) -> float:
    """
    FIX-3: Mỗi few-shot làm QUERY, exemplar làm SUPPORT (K=1).
    → average score qua N few-shot.

    KHÔNG phải ngược lại (exemplar query, N few-shot support)
    vì EC protocol: query=test image, support=Ec.
    """
    ex_f, ex_g, ex_p = exemplar_triplet   # (1, D)
    fs_f, fs_g, fs_p = fs_triplet         # (N, D)
    N = fs_f.shape[0]

    scores = []
    for i in range(N):
        mean_s, _, _ = ec_scorer.score_query_vs_support(
            fs_f[i:i+1], fs_g[i:i+1], fs_p[i:i+1],   # query = few-shot i
            ex_f, ex_g, ex_p,                           # support = exemplar (K=1)
        )
        scores.append(float(mean_s))
    return float(sum(scores) / max(1, N))


# ──────────────────────────────────────────────────────────
#  Base Class EC Loader
# ──────────────────────────────────────────────────────────

@dataclass
class BaseClassEC:
    class_name:  str
    class_key:   str
    image_paths: List[str]
    image_feats: torch.Tensor
    text_feat:   torch.Tensor
    ec_feat:     Optional[torch.Tensor] = None
    ec_global:   Optional[torch.Tensor] = None
    ec_patch:    Optional[torch.Tensor] = None


def load_base_ec(
    base_ec_dir: str,
    base_classes: List[str],   # list like "002.Laysan_Albatross"
    clip_extractor: CLIPExtractor,
    ec_scorer: Optional[ECScorer] = None,
) -> Dict[str, BaseClassEC]:
    """
    Support 2 layouts:
    (A) Folder per class: base_ec_dir/002.Laysan_Albatross/*.jpg
    (B) JSON per class:   base_ec_dir/002.Laysan_Albatross_Ec.json

    Your case is (B).
    """
    ec_dict: Dict[str, BaseClassEC] = {}
    print(f"Loading base EC for {len(base_classes)} classes...")

    # --- detect layout quickly ---
    entries = os.listdir(base_ec_dir)
    has_folders = any(os.path.isdir(os.path.join(base_ec_dir, e)) for e in entries)

    # Build json lookup: "002.Laysan_Albatross" -> json_path
    json_map: Dict[str, str] = {}
    for fn in entries:
        fp = os.path.join(base_ec_dir, fn)
        if not (os.path.isfile(fp) and fn.lower().endswith(".json")):
            continue

        # match: 002.Laysan_Albatross_Ec.json or _ec.json
        m = re.match(r"^(\d+)\.(.*)_(ec|Ec)\.json$", fn)
        if not m:
            continue
        cls_folder = f"{int(m.group(1)):03d}.{m.group(2)}"
        json_map[cls_folder] = fp

    # --- load each class ---
    for cls_folder in tqdm(base_classes, desc="Load EC"):
        # 1) try JSON first (your layout)
        img_paths: List[str] = []
        class_name_for_text = cls_folder

        jp = json_map.get(cls_folder)
        if jp is not None:
            try:
                d = load_json(jp)
                # class_name in your base json is like "002.Laysan_Albatross"
                class_name_for_text = d.get("class_name", cls_folder)

                ec_list = d.get("Ec", [])
                for it in ec_list:
                    if isinstance(it, dict):
                        p = it.get("path", "")
                        if isinstance(p, str) and p:
                            # only keep existing files to avoid later crash
                            if os.path.isfile(p):
                                img_paths.append(p)
                # If json has 0 valid paths, fall back to folder (if exists)
            except Exception as e:
                print(f"[WARN] Failed to read {jp}: {e}")

        # 2) folder fallback (if you ever use it)
        if not img_paths and has_folders:
            cls_dir = os.path.join(base_ec_dir, cls_folder)
            if not os.path.isdir(cls_dir):
                cls_dir = os.path.join(base_ec_dir, normalize_name(cls_folder))
            if os.path.isdir(cls_dir):
                img_paths = sorted([
                    os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
                ])

        if not img_paths:
            # debug to know why class missing
            # print(f"[WARN] No images for base class: {cls_folder} (json={jp})")
            continue

        # IMPORTANT: dict key must align with find_related_* normalize_name(...)
        cls_key = normalize_name(class_name_for_text)

        img_feats = clip_extractor.get_batch_image_feats(img_paths)
        text_feat = clip_extractor.get_text_feat(class_name_for_text)  # FIX-1 inside

        ec_feat = ec_global = ec_patch = None
        if ec_scorer is not None:
            ec_feat, ec_global, ec_patch = ec_scorer.extract_batch(img_paths)

        ec_dict[cls_key] = BaseClassEC(
            class_name=class_name_for_text,
            class_key=cls_key,
            image_paths=img_paths,
            image_feats=img_feats,
            text_feat=text_feat,
            ec_feat=ec_feat, ec_global=ec_global, ec_patch=ec_patch,
        )

    print(f"Loaded {len(ec_dict)} base classes EC")
    return ec_dict
# ──────────────────────────────────────────────────────────
#  Step 1+2: Related Base Classes
# ──────────────────────────────────────────────────────────

def find_related_full_clip(
    novel_class_name: str,
    few_shot_feats: torch.Tensor,
    ec_dict: Dict[str, BaseClassEC],
    clip_extractor: CLIPExtractor,
    top_n: int = 10,
    top_k_per_img: int = 20,
    consensus_threshold: int = 3,
) -> List[Tuple[str, float]]:

    if not ec_dict:
        return []

    device  = few_shot_feats.device
    N       = few_shot_feats.shape[0]
    cls_keys = list(ec_dict.keys())

    # Ensure everything on same device
    text_feats = torch.cat([ec_dict[k].text_feat for k in cls_keys], dim=0).to(device)
    mean_feats = torch.stack([ec_dict[k].image_feats.mean(0) for k in cls_keys], dim=0).to(device)
    mean_feats = F.normalize(mean_feats, p=2, dim=1)

    novel_feat = clip_extractor.get_text_feat(novel_class_name).to(device)
    text_sim   = cosine_sim(novel_feat, text_feats).squeeze(0)     # (C,)
    img_sim    = cosine_sim(few_shot_feats, mean_feats)            # (N, C)

    fuzzy   = 0.4 * text_sim.unsqueeze(0) + 0.6 * img_sim          # (N, C)
    _, topk = torch.topk(fuzzy, k=min(top_k_per_img, len(cls_keys)), dim=1)

    votes = torch.zeros(len(cls_keys), device=device)
    for i in range(N):
        votes.scatter_add_(0, topk[i], torch.ones_like(topk[i], dtype=votes.dtype, device=device))

    total = fuzzy.sum(dim=0)  # (C,)

    mask = votes >= consensus_threshold
    if mask.sum() == 0:
        _, fb = torch.topk(votes, min(top_n, len(cls_keys)))
        mask = torch.zeros(len(cls_keys), dtype=torch.bool, device=device)
        mask[fb] = True

    fi = mask.nonzero(as_tuple=True)[0]                 # indices on device
    order = torch.argsort(total[fi], descending=True)   # on device

    top_i = fi[order][:top_n].detach().cpu().tolist()   # <-- avoid cuda->list issue
    result = [(cls_keys[i], float(total[i].item())) for i in top_i]

    print(f"Step 1+2 (FULL): {len(result)} related (T={consensus_threshold})")
    for k, s in result[:5]:
        print(f"  {ec_dict[k].class_name:<30} score={s:.4f}")
    return result


def find_related_from_topk(
    novel_class_name: str,
    few_shot_paths: List[str],
    ec_dict: Dict[str, BaseClassEC],
    clip_extractor: CLIPExtractor,
    clip_topk_map: Dict[str, dict],
    topk_field: str = "final_topk",
    top_n: int = 10,
    top_k_per_img: int = 20,
    consensus_threshold: int = 3,
    alpha: float = 0.4,
) -> List[Tuple[str, float]]:
    if not ec_dict:
        return []

    novel_feat = clip_extractor.get_text_feat(novel_class_name)   # FIX-1
    print(f"  Novel: '{novel_class_name}' → '{preprocess_for_clip(novel_class_name)}'")

    votes:  Dict[str, int]   = {}
    scores: Dict[str, float] = {}
    n_used = 0

    for img_path in few_shot_paths:
        # FIX-2: abs path then basename fallback
        item = (clip_topk_map.get(os.path.normpath(os.path.abspath(img_path)))
                or clip_topk_map.get(os.path.basename(img_path)))
        if item is None:
            continue

        topk_list = item.get(topk_field, [])[:max(1, top_k_per_img)]
        if not topk_list:
            continue

        cands: List[Tuple[str, float]] = []
        for c in topk_list:
            ck = c.get("class_key", "") or c.get("class_name", "") or c.get("class_display", "")
            ck = normalize_name(ck)   # "100.Brown_Pelican" -> "brown_pelican"
            sim = c.get("similarity")
            if sim is None or ck not in ec_dict:
                continue
            cands.append((ck, float(sim)))
        if not cands:
            continue
        n_used += 1

        cand_text = torch.cat([ec_dict[k].text_feat for k, _ in cands], dim=0)
        text_sims = cosine_sim(novel_feat, cand_text).squeeze(0)

        for (k, img_sim), t in zip(cands, text_sims.tolist()):
            fuzzy = alpha * float(t) + (1 - alpha) * float(img_sim)
            scores[k] = scores.get(k, 0.0) + fuzzy
            votes[k]  = votes.get(k, 0) + 1

    if n_used == 0:
        print("  [WARN] Không match → fallback FULL CLIP")
        return []

    passed = [k for k, v in votes.items() if v >= consensus_threshold]
    if not passed:
        passed = sorted(votes, key=lambda k: votes[k], reverse=True)[:top_n]

    result = sorted(passed, key=lambda k: scores.get(k, 0.0), reverse=True)[:top_n]
    result = [(k, float(scores.get(k, 0.0))) for k in result]

    print(f"Step 1+2 (CLIP topK): {len(result)} related "
          f"(field={topk_field}, used={n_used}/{len(few_shot_paths)}, T={consensus_threshold})")
    for k, s in result[:5]:
        print(f"  {ec_dict[k].class_name:<30} score={s:.4f} votes={votes.get(k,0)}/{n_used}")
    return result


# ──────────────────────────────────────────────────────────
#  Step 3: Segment với Representatives
# ──────────────────────────────────────────────────────────

@dataclass
class Segment:
    class_key:    str
    segment_id:   int
    segment_type: str
    image_paths:  List[str]
    image_feats:  torch.Tensor
    centroid:     torch.Tensor
    ec_feat:      Optional[torch.Tensor] = None
    ec_global:    Optional[torch.Tensor] = None
    ec_patch:     Optional[torch.Tensor] = None


def _pick_representatives(
    paths: List[str],
    feats: torch.Tensor,              # (K, D) sorted near→far centroid
    ec_f: Optional[torch.Tensor],
    ec_g: Optional[torch.Tensor],
    ec_p: Optional[torch.Tensor],
    n_repr: int = 4,
):
    """Chọn near / 1/3 / 2/3 / far — không còn boundary-by-seg_id ngẫu nhiên."""
    K = len(paths)
    if K <= n_repr:
        return paths, feats, ec_f, ec_g, ec_p

    positions = sorted(set([0, max(1, K // 3), max(2, 2 * K // 3), K - 1]))
    extra = [i for i in range(1, K - 1) if i not in positions]
    while len(positions) < min(n_repr, K) and extra:
        positions.append(extra.pop(0))
    positions = sorted(positions)[:n_repr]

    return (
        [paths[i] for i in positions],
        feats[positions],
        ec_f[positions]  if ec_f is not None else None,
        ec_g[positions]  if ec_g is not None else None,
        ec_p[positions]  if ec_p is not None else None,
    )


def segment_base_ec(ec: BaseClassEC, n_segments: int = 5,
                    n_repr_per_seg: int = 4) -> List[Segment]:
    from sklearn.cluster import KMeans

    N = ec.image_feats.shape[0]
    centroid_full = F.normalize(
        ec.image_feats.mean(0, keepdim=True), p=2, dim=1).squeeze(0)

    if N < n_segments:
        dists = 1 - cosine_sim(ec.image_feats, centroid_full.unsqueeze(0)).squeeze(1)
        order = torch.argsort(dists)
        sp, sf, sef, seg_, sep = _pick_representatives(
            [ec.image_paths[i.item()] for i in order],
            ec.image_feats[order],
            ec.ec_feat[order]   if ec.ec_feat   is not None else None,
            ec.ec_global[order] if ec.ec_global is not None else None,
            ec.ec_patch[order]  if ec.ec_patch  is not None else None,
            n_repr_per_seg)
        return [Segment(
            class_key=ec.class_key, segment_id=0, segment_type="single",
            image_paths=sp, image_feats=sf, centroid=centroid_full,
            ec_feat=sef, ec_global=seg_, ec_patch=sep,
        )]

    feats_np = F.normalize(ec.image_feats.cpu(), p=2, dim=1).numpy()
    labels   = KMeans(n_clusters=n_segments, random_state=42, n_init=10).fit_predict(feats_np)

    segments: List[Segment] = []
    for seg_id in range(n_segments):
        idx = np.where(labels == seg_id)[0].tolist()
        if not idx:
            continue
        seg_feats = ec.image_feats[idx]
        centroid  = F.normalize(seg_feats.mean(0, keepdim=True), p=2, dim=1).squeeze(0)

        dists  = 1 - cosine_sim(seg_feats, centroid.unsqueeze(0)).squeeze(1)
        order  = torch.argsort(dists)
        sorted_i = [idx[o.item()] for o in order]

        sp, sf, sef, seg_, sep = _pick_representatives(
            [ec.image_paths[i] for i in sorted_i],
            ec.image_feats[sorted_i],
            ec.ec_feat[sorted_i]   if ec.ec_feat   is not None else None,
            ec.ec_global[sorted_i] if ec.ec_global is not None else None,
            ec.ec_patch[sorted_i]  if ec.ec_patch  is not None else None,
            n_repr_per_seg)

        segments.append(Segment(
            class_key=ec.class_key, segment_id=seg_id, segment_type="middle",
            image_paths=sp, image_feats=sf, centroid=centroid,
            ec_feat=sef, ec_global=seg_, ec_patch=sep,
        ))
    return segments


# ──────────────────────────────────────────────────────────
#  Step 4: Hybrid Borda + Magnitude Scoring
# ──────────────────────────────────────────────────────────

@dataclass
class SegmentScore:
    class_key:           str
    class_name:          str
    segment_id:          int
    segment_type:        str
    total_fuzzy_score:   float
    borda_score:         float
    hybrid_score:        float        # primary sort key
    per_image_scores:    List[float]
    top_exemplar_paths:  List[str]
    top_exemplar_scores: List[float]


def score_segments(
    few_shot_paths: List[str],
    few_shot_feats: torch.Tensor,
    segments: List[Segment],
    ec_dict: Dict[str, BaseClassEC],
    ec_scorer: Optional[ECScorer],
    few_shot_ec_feats: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    top_exemplars_per_seg: int = 5,
    hybrid_lambda: float = 0.6,   # FIX-4: tham số CLI
) -> List[SegmentScore]:
    """
    Hybrid: z-norm(magnitude)*λ + z-norm(Borda)*(1-λ).
    λ=1 → thuần magnitude; λ=0 → thuần Borda.
    """
    N = len(few_shot_paths)
    use_bb = ec_scorer is not None and few_shot_ec_feats is not None
    if use_bb:
        fs_f, fs_g, fs_p = few_shot_ec_feats

    # ── 1) raw scores + exemplar scores ─────────────────
    tmp = []
    for seg in segments:
        has_bb = (use_bb and seg.ec_feat is not None
                  and seg.ec_global is not None and seg.ec_patch is not None)

        if has_bb:
            per_img   = []
            ex_scores = torch.zeros(len(seg.image_paths))
            for i in range(N):
                ms, _, per_sup = ec_scorer.score_query_vs_support(
                    fs_f[i].unsqueeze(0), fs_g[i].unsqueeze(0), fs_p[i].unsqueeze(0),
                    seg.ec_feat, seg.ec_global, seg.ec_patch)
                per_img.append(float(ms))
                ex_scores += per_sup.detach().cpu()
            total = float(sum(per_img))
        else:
            sim_mat   = cosine_sim(few_shot_feats, seg.image_feats)
            per_img   = sim_mat.mean(dim=1).tolist()
            total     = float(sim_mat.sum().item())
            ex_scores = sim_mat.sum(dim=0).detach().cpu()

        top_k = min(top_exemplars_per_seg, len(seg.image_paths))
        tv, ti = torch.topk(ex_scores, top_k)
        tmp.append({
            "seg": seg, "total": total, "per_img": per_img,
            "top_paths":  [seg.image_paths[i.item()] for i in ti],
            "top_scores": tv.tolist(),
        })

    if not tmp:
        return []

    S = len(tmp)

    # ── 2) Borda per few-shot image ──────────────────────
    borda = [0.0] * S
    for i in range(N):
        row   = [tmp[s]["per_img"][i] for s in range(S)]
        order = sorted(range(S), key=lambda s: row[s], reverse=True)
        for r, sidx in enumerate(order, start=1):
            borda[sidx] += float(S - r + 1)

    # ── 3) Hybrid z-norm blend ───────────────────────────
    mag = np.array([tmp[s]["total"] for s in range(S)], dtype=np.float32)
    bor = np.array(borda, dtype=np.float32)

    def z(x):
        if x.size <= 1:
            return np.zeros_like(x)
        return (x - x.mean()) / (x.std() + 1e-8)

    hybrid = hybrid_lambda * z(mag) + (1.0 - hybrid_lambda) * z(bor)

    # ── 4) Build results ─────────────────────────────────
    results: List[SegmentScore] = []
    for sidx in range(S):
        seg = tmp[sidx]["seg"]
        results.append(SegmentScore(
            class_key=seg.class_key,
            class_name=ec_dict[seg.class_key].class_name,
            segment_id=seg.segment_id,
            segment_type=seg.segment_type,
            total_fuzzy_score=float(mag[sidx]),
            borda_score=float(bor[sidx]),
            hybrid_score=float(hybrid[sidx]),
            per_image_scores=[float(x) for x in tmp[sidx]["per_img"]],
            top_exemplar_paths=tmp[sidx]["top_paths"],
            top_exemplar_scores=[float(x) for x in tmp[sidx]["top_scores"]],
        ))

    results.sort(key=lambda x: x.hybrid_score, reverse=True)
    return results


# ──────────────────────────────────────────────────────────
#  Step 5: Voting + De-duplication (FIX-6: backbone dedup)
# ──────────────────────────────────────────────────────────

def select_exemplars(
    segment_scores:      List[SegmentScore],
    ec_dict:             Dict[str, BaseClassEC],
    ec_scorer:           Optional[ECScorer],
    syn_backbone_cache:  Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    consensus_threshold: int   = 2,
    visual_dedup_thresh: float = 0.90,
    max_exemplars:       int   = 30,
) -> Tuple[List[str], List[float], dict]:
    """
    FIX-6: Dedup dùng backbone cosine (avg feat/global/patch) nếu có,
    fallback CLIP cosine.
    """
    votes:  Dict[str, int]          = {}
    scores: Dict[str, float]        = {}
    clips:  Dict[str, torch.Tensor] = {}   # CLIP feats — fallback dedup

    for ss in segment_scores:
        cls_ec = ec_dict[ss.class_key]
        for ep, es in zip(ss.top_exemplar_paths, ss.top_exemplar_scores):
            if ep not in votes:
                votes[ep]  = 0
                scores[ep] = 0.0
                try:
                    idx = cls_ec.image_paths.index(ep)
                    clips[ep] = cls_ec.image_feats[idx]
                except ValueError:
                    continue
            votes[ep]  += 1
            scores[ep] += float(es)

    passed = {ep: scores[ep]
              for ep, v in votes.items()
              if v >= consensus_threshold and ep in clips}
    if not passed:
        passed = {ep: scores[ep] for ep in votes if ep in clips}

    def dedup_sim(a: str, b: str) -> float:
        """FIX-6: backbone cosine avg(feat, global, patch) nếu có, else CLIP."""
        if (ec_scorer is not None
                and a in syn_backbone_cache
                and b in syn_backbone_cache):
            fa, ga, pa = syn_backbone_cache[a]
            fb, gb, pb = syn_backbone_cache[b]
            return (cosine_sim(fa, fb).item()
                    + cosine_sim(ga, gb).item()
                    + cosine_sim(pa, pb).item()) / 3.0
        return cosine_sim(clips[a].unsqueeze(0), clips[b].unsqueeze(0)).item()

    selected_paths:  List[str]   = []
    selected_scores: List[float] = []

    for ep, score in sorted(passed.items(), key=lambda x: x[1], reverse=True):
        if len(selected_paths) >= max_exemplars:
            break
        if ep not in clips:
            continue
        if any(dedup_sim(ep, k) > visual_dedup_thresh for k in selected_paths):
            continue
        selected_paths.append(ep)
        selected_scores.append(score)

    debug = {
        "total_candidates": len(votes),
        "after_voting":     len(passed),
        "after_dedup":      len(selected_paths),
        "dedup_mode":       "backbone" if syn_backbone_cache else "clip",
        "top_segments": [
            {"class": ss.class_name, "seg": ss.segment_id,
             "hybrid": round(ss.hybrid_score, 4),
             "magnitude": round(ss.total_fuzzy_score, 4),
             "borda": round(ss.borda_score, 1)}
            for ss in segment_scores[:10]
        ],
    }
    return selected_paths, selected_scores, debug


# ──────────────────────────────────────────────────────────
#  Main Synthesizer
# ──────────────────────────────────────────────────────────

class ECSynthesizer:
    def __init__(
        self,
        base_ec_dir:            str,
        base_classes:           List[str],
        clip_model:             str   = "ViT-B-32",
        clip_pretrained:        str   = "laion2b_s34b_b79k",
        device:                 str   = "cpu",
        cache_dir:              Optional[str]   = None,
        n_segments:             int   = 5,
        n_repr_per_seg:         int   = 4,
        top_n_classes:          int   = 10,
        top_k_per_img:          int   = 20,
        consensus_threshold:    int   = 3,
        visual_dedup_thresh:    float = 0.90,
        max_exemplars:          int   = 30,
        top_exemplars_per_seg:  int   = 5,
        fewshot_emphasis_weight: float = 3.0,
        hybrid_lambda:          float = 0.6,   # FIX-4
        ec_weights_dir:         Optional[str]   = None,
        ec_feat_cache_dir:      Optional[str]   = None,
        clip_topk_map:          Optional[Dict[str, dict]] = None,
        clip_topk_field:        str = "final_topk",
    ):
        self.top_n_classes           = top_n_classes
        self.top_k_per_img           = top_k_per_img
        self.consensus_threshold     = consensus_threshold
        self.visual_dedup_thresh     = visual_dedup_thresh
        self.max_exemplars           = max_exemplars
        self.top_exemplars_per_seg   = top_exemplars_per_seg
        self.fewshot_emphasis_weight = fewshot_emphasis_weight
        self.hybrid_lambda           = hybrid_lambda
        self.device                  = device
        self.clip_topk_map           = clip_topk_map
        self.clip_topk_field         = clip_topk_field

        self.clip = CLIPExtractor(clip_model, clip_pretrained, device, cache_dir)

        self.ec_scorer: Optional[ECScorer] = None
        if ec_weights_dir and os.path.isdir(ec_weights_dir):
            self.ec_scorer = ECScorer(ec_weights_dir, device, ec_feat_cache_dir)
            print(f"ECScorer ready: {ec_weights_dir}")
        else:
            print("ECScorer không có → fallback cosine CLIP")

        self.ec_dict = load_base_ec(
            base_ec_dir, base_classes, self.clip, self.ec_scorer)

        print("Pre-segmenting base class EC...")
        self.segments: Dict[str, List[Segment]] = {}
        for cls_key, ec in tqdm(self.ec_dict.items(), desc="Segmenting"):
            self.segments[cls_key] = segment_base_ec(ec, n_segments, n_repr_per_seg)

    def synthesize(
        self,
        class_folder_name: str,        # e.g. "101.Geococcyx"
        class_idx:         int,
        few_shot_paths:    List[str],
        ec_size_requested: int  = 30,
        output_dir:        Optional[str] = None,
    ) -> dict:
        # FIX-5: class_name sạch
        m = re.match(r"^\d+\.(.*)", class_folder_name)
        novel_class_name = m.group(1).replace("_", " ") if m else class_folder_name

        N = len(few_shot_paths)
        print(f"\n{'='*60}")
        print(f"Synthesizing: {class_folder_name}  (idx={class_idx}, {N} few-shot)")
        print(f"{'='*60}")

        # ── Backbone features cho few-shot (1 lần duy nhất) ─
        few_shot_ec_feats = None
        if self.ec_scorer is not None:
            few_shot_ec_feats = self.ec_scorer.extract_batch(few_shot_paths)

        # ── Step 1+2 ──────────────────────────────────────────
        related: List[Tuple[str, float]] = []
        if self.clip_topk_map is not None:
            related = find_related_from_topk(
                novel_class_name, few_shot_paths, self.ec_dict, self.clip,
                self.clip_topk_map, self.clip_topk_field,
                self.top_n_classes, self.top_k_per_img, self.consensus_threshold)

        if not related:
            print("  → fallback FULL CLIP")
            fsf = self.clip.get_batch_image_feats(few_shot_paths).to(self.device)
            related = find_related_full_clip(
                novel_class_name, fsf, self.ec_dict, self.clip,
                self.top_n_classes, self.top_k_per_img, self.consensus_threshold)

        if not related:
            print("Không tìm được related class!")
            return {}

        few_shot_feats = (
            torch.zeros((N, 1), device=self.device)
            if self.ec_scorer is not None
            else self.clip.get_batch_image_feats(few_shot_paths).to(self.device)
        )

        # ── Step 3+4 ──────────────────────────────────────────
        all_segs: List[Segment] = []
        for k, _ in related:
            all_segs.extend(self.segments.get(k, []))

        print(f"\nStep 4: {len(all_segs)} segments from {len(related)} related classes")
        seg_scores = score_segments(
            few_shot_paths, few_shot_feats, all_segs,
            self.ec_dict, self.ec_scorer, few_shot_ec_feats,
            self.top_exemplars_per_seg, self.hybrid_lambda)

        print("Top 5 segments:")
        for ss in seg_scores[:5]:
            print(f"  {ss.class_name:<30} seg={ss.segment_id}  "
                  f"hybrid={ss.hybrid_score:.3f}  "
                  f"mag={ss.total_fuzzy_score:.3f}  borda={ss.borda_score:.0f}")

        # ── Step 5: dedup với backbone cache ─────────────────
        # FIX-6: extract backbone 1 lần cho candidates (không bao gồm few-shot)
        candidate_paths = list({ep for ss in seg_scores
                                 for ep in ss.top_exemplar_paths})
        syn_backbone_cache: Dict[
            str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        if self.ec_scorer is not None and candidate_paths:
            c_f, c_g, c_p = self.ec_scorer.extract_batch(candidate_paths)
            for i, p in enumerate(candidate_paths):
                syn_backbone_cache[p] = (c_f[i:i+1], c_g[i:i+1], c_p[i:i+1])

        selected_paths, selected_scores, debug = select_exemplars(
            seg_scores, self.ec_dict, self.ec_scorer, syn_backbone_cache,
            max(1, self.consensus_threshold - 1),
            self.visual_dedup_thresh, self.max_exemplars)

        # ── Step 6: Build Ec ───────────────────────────────────
        emphasis     = self.fewshot_emphasis_weight
        fs_triplet   = few_shot_ec_feats

        # cache few-shot backbone features to disk
        if self.ec_scorer is not None and fs_triplet is not None:
            ff, fg, fp = fs_triplet
            for i, fpath in enumerate(few_shot_paths):
                cp = self.ec_scorer._feat_cache_path(fpath)
                if cp and not os.path.exists(cp):
                    os.makedirs(os.path.dirname(cp) or ".", exist_ok=True)
                    torch.save({"feat":   ff[i:i+1].cpu(),
                                "global": fg[i:i+1].cpu(),
                                "patch":  fp[i:i+1].cpu()}, cp)

        ec_list = []
        rank    = 1

        # Few-shot entries
        for i, p in enumerate(few_shot_paths):
            entry: dict = {
                "path":            p,
                "mean_similarity": float(emphasis)*0.1,
                "rank":            rank,
                "source":          "fewshot",
                "weight":          emphasis,
            }
            # FIX-3: score thực — few-shot[i] làm query, few-shot làm support
            if self.ec_scorer is not None and fs_triplet is not None:
                q = (fs_triplet[0][i:i+1],
                     fs_triplet[1][i:i+1],
                     fs_triplet[2][i:i+1])
                entry["mean_similarity_to_fewshot_backbone"] = round(
                    mean_sim_to_fewshot_backbone(self.ec_scorer, q, fs_triplet), 6)
            ec_list.append(entry)
            rank += 1

        # Synthesized entries
        for p, s in zip(selected_paths, selected_scores):
            entry = {
                "path":            p,
                "mean_similarity": round(float(s), 6)*0.1,
                "rank":            rank,
                "source":          "synthesized",
                "weight":          1.0,
            }
            # FIX-3: score thực
            if (self.ec_scorer is not None
                    and fs_triplet is not None
                    and p in syn_backbone_cache):
                entry["mean_similarity_to_fewshot_backbone"] = round(
                    mean_sim_to_fewshot_backbone(
                        self.ec_scorer, syn_backbone_cache[p], fs_triplet), 6)
            ec_list.append(entry)
            rank += 1

        # FIX-5: output format khớp base class Ec
        E_inc = {
            "class_name":          novel_class_name,   # "Geococcyx" ✓
            "class_idx":           class_idx,
            "class_folder":        class_folder_name,
            "num_images_in_class": N,
            "ec_size_requested":   ec_size_requested,
            "ec_size_actual":      len(ec_list),
            "num_fewshot":         N,
            "num_synthesized":     len(selected_paths),
            "fewshot_emphasis_weight": emphasis,
            "hybrid_lambda":       self.hybrid_lambda,
            "related_base_classes": [
                {"class_key": k, "relevance_score": round(float(s), 4)}
                for k, s in related],
            "Ec":    ec_list,
            "debug": debug,
        }

        print(f"\nEc = {N} few-shot + {len(selected_paths)} synthesized "
              f"= {len(ec_list)} (requested {ec_size_requested})")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            # FIX-5: "101.geococcyx_ec.json"
            fname = f"{class_idx:03d}.{normalize_name(novel_class_name)}_Ec.json"
            save_json(E_inc, os.path.join(output_dir, fname))
            print(f"Saved: {os.path.join(output_dir, fname)}")

        return E_inc


# ──────────────────────────────────────────────────────────
#  CLI helpers
# ──────────────────────────────────────────────────────────

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def _is_img(f: str) -> bool:
    return f.lower().endswith(IMG_EXTS)

def _get_class_index(folder_name: str) -> Optional[int]:
    m = re.match(r"^(\d+)\.", folder_name.strip())
    return int(m.group(1)) if m else None

def auto_detect_base_classes(base_ec_dir: str, n_base: int = 100) -> List[str]:
    """
    Support 2 layouts:
    (A) Folder per class: 001.xxx/, 002.xxx/, ...
    (B) JSON per class:   001.xxx_Ec.json, 002.xxx_Ec.json, ...
    Return list of class_folder_name like "001.Black_footed_Albatross".
    """
    entries = os.listdir(base_ec_dir)

    # 1) Try folder layout
    folders = [f for f in entries if os.path.isdir(os.path.join(base_ec_dir, f))]
    folder_classes = sorted(
        [f for f in folders
         if (idx := _get_class_index(f)) is not None and 1 <= idx <= n_base],
        key=_get_class_index
    )
    if folder_classes:
        print(f"Auto-detected {len(folder_classes)} base classes from FOLDERS (1–{n_base})")
        return folder_classes

    # 2) Try json layout
    jsons = [f for f in entries
             if os.path.isfile(os.path.join(base_ec_dir, f))
             and f.lower().endswith(".json")
             and re.match(r"^\d+\..*_(ec|Ec)\.json$", f) is not None]
    json_classes = []
    for fn in jsons:
        m = re.match(r"^(\d+)\.(.*)_(ec|Ec)\.json$", fn)
        if not m:
            continue
        idx = int(m.group(1))
        if 1 <= idx <= n_base:
            # rebuild folder-style name "003.Sooty_Albatross"
            json_classes.append(f"{idx:03d}.{m.group(2)}")
    json_classes = sorted(set(json_classes), key=_get_class_index)

    print(f"Auto-detected {len(json_classes)} base classes from JSON (1–{n_base})")
    return json_classes


def auto_detect_incremental_classes(
    inc_root_dir: str, session: int,
    n_base: int = 100, n_per_session: int = 10,
) -> List[Tuple[str, str, int]]:
    start = n_base + (session - 1) * n_per_session + 1
    end   = n_base + session * n_per_session
    result = sorted(
        [(f, os.path.join(inc_root_dir, f), _get_class_index(f))
         for f in os.listdir(inc_root_dir)
         if os.path.isdir(os.path.join(inc_root_dir, f))
         and (idx := _get_class_index(f)) is not None and start <= idx <= end],
        key=lambda x: x[2])
    print(f"Session {session}: {start}–{end} → {len(result)} classes")
    for n, _, _ in result:
        print(f"  {n}")
    return result

def sample_fewshot_images(class_dir: str, n_shots: int = 10,
                          seed: int = 42) -> List[str]:
    imgs = sorted([os.path.join(class_dir, f)
                   for f in os.listdir(class_dir) if _is_img(f)])
    if not imgs:
        return []
    random.seed(seed)
    return random.sample(imgs, min(n_shots, len(imgs)))


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device} | Session: {args.session}")

    base_classes = auto_detect_base_classes(args.base_ec_dir, n_base=args.n_base)
    if not base_classes:
        raise RuntimeError(f"Không tìm thấy base class nào trong {args.base_ec_dir}")

    clip_topk_map = None
    if args.clip_topk_json:
        clip_topk_map = load_clip_topk_json(args.clip_topk_json)

    synth = ECSynthesizer(
        base_ec_dir=args.base_ec_dir,
        base_classes=base_classes,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        device=device,
        cache_dir=args.cache_dir,
        n_segments=args.n_segments,
        n_repr_per_seg=args.n_repr_per_seg,
        top_n_classes=args.top_n_classes,
        top_k_per_img=args.top_k_per_img,
        consensus_threshold=args.consensus_t,
        visual_dedup_thresh=args.dedup_thresh,
        max_exemplars=args.max_exemplars,
        top_exemplars_per_seg=args.top_per_seg,
        fewshot_emphasis_weight=args.emphasis,
        hybrid_lambda=args.hybrid_lambda,
        ec_weights_dir=args.ec_weights_dir,
        ec_feat_cache_dir=args.ec_feat_cache_dir,
        clip_topk_map=clip_topk_map,
        clip_topk_field=args.clip_topk_field,
    )

    session_classes = auto_detect_incremental_classes(
        args.inc_root_dir, args.session, args.n_base, args.n_per_session)
    if not session_classes:
        raise RuntimeError(f"Không tìm thấy class nào cho session {args.session}")

    all_results = {}
    for cls_folder, cls_dir, cls_idx in session_classes:
        few_shot_paths = sample_fewshot_images(cls_dir, args.n_shots, seed=args.seed)
        if not few_shot_paths:
            print(f"  Bỏ qua {cls_folder}: không có ảnh")
            continue
        print(f"\n[{cls_folder}]  few-shot: {len(few_shot_paths)} ảnh")
        for fp in few_shot_paths:
            print(f"    {os.path.basename(fp)}")

        res = synth.synthesize(
            class_folder_name=cls_folder,
            class_idx=cls_idx,
            few_shot_paths=few_shot_paths,
            ec_size_requested=args.max_exemplars,
            output_dir=args.output_dir,
        )
        res["session"] = args.session
        all_results[cls_folder] = res

    print(f"\n{'='*60}\nSESSION {args.session} — Tổng kết:\n{'='*60}")
    for cf, res in all_results.items():
        print(f"  {cf:<40} "
              f"fewshot={res.get('num_fewshot',0)} "
              f"syn={res.get('num_synthesized',0)} "
              f"total={res.get('ec_size_actual',0)}")

    summary_path = os.path.join(args.output_dir, f"session_{args.session}_summary.json")
    save_json({
        "session":       args.session,
        "hybrid_lambda": args.hybrid_lambda,
        "n_classes":     len(all_results),
        "classes":       list(all_results.keys()),
        "results": {k: {
            "class_name":      v.get("class_name"),
            "class_idx":       v.get("class_idx"),
            "num_fewshot":     v.get("num_fewshot"),
            "num_synthesized": v.get("num_synthesized"),
            "ec_size_actual":  v.get("ec_size_actual"),
            "output_file":     f"{v.get('class_idx',0):03d}."
                               f"{normalize_name(v.get('class_name',''))}_ec.json",
        } for k, v in all_results.items()}
    }, summary_path)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="EC Synthesizer for Incremental Classes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base_ec_dir",       type=str, required=True)
    p.add_argument("--inc_root_dir",      type=str, required=True)
    p.add_argument("--output_dir",        type=str, default="output_json/incremental_ec")
    p.add_argument("--cache_dir",         type=str, default=None)
    p.add_argument("--ec_weights_dir",    type=str, default=None)
    p.add_argument("--ec_feat_cache_dir", type=str, default=None)
    p.add_argument("--clip_topk_json",    type=str, default=None)
    p.add_argument("--clip_topk_field",   type=str, default="final_topk",
                   choices=["final_topk", "global_topk"])
    p.add_argument("--clip_model",        type=str, default="ViT-B-32")
    p.add_argument("--clip_pretrained",   type=str, default="laion2b_s34b_b79k")
    p.add_argument("--session",           type=int, required=True)
    p.add_argument("--n_base",            type=int, default=100)
    p.add_argument("--n_per_session",     type=int, default=10)
    p.add_argument("--n_shots",           type=int, default=10)
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--n_segments",        type=int,   default=5)
    p.add_argument("--n_repr_per_seg",    type=int,   default=4,
                   help="Số representatives mỗi segment (near/1/3/2/3/far)")
    p.add_argument("--top_n_classes",     type=int,   default=10)
    p.add_argument("--top_k_per_img",     type=int,   default=20)
    p.add_argument("--consensus_t",       type=int,   default=3)
    p.add_argument("--dedup_thresh",      type=float, default=0.90)
    p.add_argument("--max_exemplars",     type=int,   default=30)
    p.add_argument("--top_per_seg",       type=int,   default=5)
    p.add_argument("--emphasis",          type=float, default=3.0)
    p.add_argument("--hybrid_lambda",     type=float, default=0.6,
                   help="1.0=thuần magnitude, 0.0=thuần Borda")   # FIX-4
    p.add_argument("--cpu",               action="store_true")
    args = p.parse_args()
    main(args)