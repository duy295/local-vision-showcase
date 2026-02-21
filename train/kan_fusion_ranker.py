import argparse
import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    s = str(name).strip()
    s = re.sub(r"^\s*\d+\s*[\.\-_]*\s*", "", s)
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s.lower().strip("_")


def _norm_path(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_results(payload: dict) -> List[dict]:
    if isinstance(payload.get("results"), list):
        return payload["results"]
    if isinstance(payload.get("result"), dict):
        return [payload["result"]]
    return []


# ─────────────────────────────────────────────
#  Extractors
#
#  Cấu trúc file EC thực tế:
#  {
#    "final_topk": {
#      "top_k":    [...],   <- EC final scores
#      "clip_topk":[...]    <- CLIP final scores
#    },
#    "global_topk": {
#      "top_k":    [...],   <- EC global scores
#      "clip_topk":[...]    <- CLIP global scores
#    }
#  }
# ─────────────────────────────────────────────

def _extract_ec_topk_from_section(section: dict) -> List[dict]:
    """Đọc EC scores từ section.top_k"""
    rows = section.get("top_k", []) or []
    out = []
    for row in rows:
        cls = row.get("class_raw") or row.get("class")
        if not cls:
            continue
        score = _safe_float(row.get("mean_score", row.get("total_score", row.get("score", 0.0))))
        rank = row.get("rank", len(out) + 1)
        out.append({
            "class_raw": cls,
            "class_key": normalize_name(cls),
            "score": score,
            "rank": int(rank),
        })
    out.sort(key=lambda x: x["rank"])
    return out


def _extract_clip_topk_from_section(section: dict) -> List[dict]:
    """Đọc CLIP scores từ section.clip_topk"""
    rows = section.get("clip_topk", []) or []
    out = []
    for row in rows:
        cls = row.get("class_name") or row.get("class_raw") or row.get("class")
        if not cls:
            continue
        out.append({
            "class_raw": cls,
            "class_key": normalize_name(cls),
            "score": _safe_float(row.get("similarity", row.get("score", 0.0))),
            "rank": int(row.get("rank", len(out) + 1)),
        })
    out.sort(key=lambda x: x["rank"])
    return out


def _extract_all_sources(item: dict) -> Tuple[List, List, List, List]:
    """Trả về (ec_final, ec_global, clip_final, clip_global) từ 1 item"""
    final_sec  = item.get("final_topk",  {}) or {}
    global_sec = item.get("global_topk", {}) or {}
    return (
        _extract_ec_topk_from_section(final_sec),
        _extract_ec_topk_from_section(global_sec),
        _extract_clip_topk_from_section(final_sec),
        _extract_clip_topk_from_section(global_sec),
    )


def _top1_top2_margin(rows: List[dict]) -> float:
    if not rows:
        return 0.0
    s1 = rows[0]["score"]
    s2 = rows[1]["score"] if len(rows) > 1 else 0.0
    return s1 - s2


def _find_rank_score(rows: List[dict], class_key: str) -> Tuple[int, float]:
    for r in rows:
        if r["class_key"] == class_key:
            return r["rank"], r["score"]
    return 10**6, 0.0


def _best_label(rows: List[dict]) -> str:
    return rows[0]["class_raw"] if rows else ""


def _build_candidate_keys(
    ec_f_rows, ec_g_rows, clip_f_rows, clip_g_rows,
    candidate_pool_k: int, force_include_key: str = "",
) -> List[str]:
    buckets = {}

    def _acc(rows):
        for r in rows[:candidate_pool_k]:
            key = r["class_key"]
            if key not in buckets:
                buckets[key] = {"best_rank": 10**6, "best_score": -1e9, "class_raw": r["class_raw"]}
            buckets[key]["best_rank"]  = min(buckets[key]["best_rank"],  r["rank"])
            buckets[key]["best_score"] = max(buckets[key]["best_score"], r["score"])

    _acc(ec_f_rows); _acc(ec_g_rows); _acc(clip_f_rows); _acc(clip_g_rows)

    if force_include_key and force_include_key not in buckets:
        buckets[force_include_key] = {"best_rank": 10**6, "best_score": -1e9, "class_raw": force_include_key}

    sorted_keys = sorted(buckets, key=lambda k: (buckets[k]["best_rank"], -buckets[k]["best_score"], k))

    if candidate_pool_k > 0:
        sorted_keys = sorted_keys[:candidate_pool_k]
        if force_include_key and force_include_key not in sorted_keys:
            sorted_keys[-1] = force_include_key

    return sorted_keys


@dataclass
class GroupSample:
    image_path: str
    true_class_raw: str
    true_key: str
    candidate_keys: List[str]
    candidate_raw: List[str]
    X: np.ndarray
    y_idx: int
    hard_weight: float
    ec_final_top1_raw: str
    ec_global_top1_raw: str
    clip_final_top1_raw: str
    clip_global_top1_raw: str
    true_ecf_rank: int = 10**6
    true_ecg_rank: int = 10**6
    true_cf_rank:  int = 10**6
    true_cg_rank:  int = 10**6
    final_global_disagree: bool = False


def _build_group_sample(item: dict, candidate_pool_k: int) -> Optional[GroupSample]:
    ec_f, ec_g, clip_f, clip_g = _extract_all_sources(item)

    if not ec_f and not ec_g and not clip_f and not clip_g:
        return None

    true_raw = item.get("true_class_raw") or item.get("true_class") or ""
    true_key = normalize_name(true_raw)
    if not true_key:
        return None

    ecf_top1  = _best_label(ec_f)
    ecg_top1  = _best_label(ec_g)
    cf_top1   = _best_label(clip_f)
    cg_top1   = _best_label(clip_g)

    candidate_keys = _build_candidate_keys(
        ec_f, ec_g, clip_f, clip_g,
        candidate_pool_k=candidate_pool_k,
        force_include_key=true_key,
    )
    if not candidate_keys:
        return None

    ecf_top1_score = ec_f[0]["score"]   if ec_f   else 0.0
    ecg_top1_score = ec_g[0]["score"]   if ec_g   else 0.0
    cf_top1_score  = clip_f[0]["score"] if clip_f else 0.0
    cg_top1_score  = clip_g[0]["score"] if clip_g else 0.0

    ecf_margin = _top1_top2_margin(ec_f)
    ecg_margin = _top1_top2_margin(ec_g)
    cf_margin  = _top1_top2_margin(clip_f)
    cg_margin  = _top1_top2_margin(clip_g)

    ecf_conf = ec_f[0]["score"]   / (ec_f[1]["score"]   + 1e-8) if len(ec_f)   > 1 else 1.0
    ecg_conf = ec_g[0]["score"]   / (ec_g[1]["score"]   + 1e-8) if len(ec_g)   > 1 else 1.0
    cf_conf  = clip_f[0]["score"] / (clip_f[1]["score"] + 1e-8) if len(clip_f) > 1 else 1.0
    cg_conf  = clip_g[0]["score"] / (clip_g[1]["score"] + 1e-8) if len(clip_g) > 1 else 1.0

    ecf_map = {r["class_key"]: r for r in ec_f}
    ecg_map = {r["class_key"]: r for r in ec_g}
    cf_map  = {r["class_key"]: r for r in clip_f}
    cg_map  = {r["class_key"]: r for r in clip_g}

    cand_raw, feats = [], []

    for ck in candidate_keys:
        efr = ecf_map.get(ck)
        egr = ecg_map.get(ck)
        fr  = cf_map.get(ck)
        gr  = cg_map.get(ck)

        ecf_present = 1.0 if efr else 0.0
        ecg_present = 1.0 if egr else 0.0
        cf_present  = 1.0 if fr  else 0.0
        cg_present  = 1.0 if gr  else 0.0

        ecf_rank = efr["rank"] if efr else 10**6
        ecg_rank = egr["rank"] if egr else 10**6
        cf_rank  = fr["rank"]  if fr  else 10**6
        cg_rank  = gr["rank"]  if gr  else 10**6

        ecf_score = efr["score"] if efr else 0.0
        ecg_score = egr["score"] if egr else 0.0
        cf_score  = fr["score"]  if fr  else 0.0
        cg_score  = gr["score"]  if gr  else 0.0

        vote_ecf   = 1.0 if normalize_name(ecf_top1) == ck else 0.0
        vote_ecg   = 1.0 if normalize_name(ecg_top1) == ck else 0.0
        vote_cf    = 1.0 if normalize_name(cf_top1)  == ck else 0.0
        vote_cg    = 1.0 if normalize_name(cg_top1)  == ck else 0.0
        vote_count = vote_ecf + vote_ecg + vote_cf + vote_cg

        available  = [s for s, p in [(ecf_score, ecf_present), (ecg_score, ecg_present),
                                      (cf_score, cf_present),  (cg_score,  cg_present)] if p > 0]
        score_mean = float(np.mean(available)) if available else 0.0
        score_std  = float(np.std(available))  if len(available) > 1 else 0.0

        ecf_rel = ecf_score / (ecf_top1_score + 1e-8)
        ecg_rel = ecg_score / (ecg_top1_score + 1e-8)
        cf_rel  = cf_score  / (cf_top1_score  + 1e-8)
        cg_rel  = cg_score  / (cg_top1_score  + 1e-8)

        ranks      = [r for r in [ecf_rank, ecg_rank, cf_rank, cg_rank] if r < 10**6]
        min_rank   = float(min(ranks)) if ranks else 999.0
        w_score    = (ecf_score * (1/ecf_rank if ecf_rank < 10**6 else 0) +
                      ecg_score * (1/ecg_rank if ecg_rank < 10**6 else 0) +
                      cf_score  * (1/cf_rank  if cf_rank  < 10**6 else 0) +
                      cg_score  * (1/cg_rank  if cg_rank  < 10**6 else 0))

        feat = [
            # EC final (6)
            ecf_score, 1/ecf_rank if ecf_rank<10**6 else 0, float(ecf_rank-1) if ecf_rank<10**6 else 999,
            ecf_present, ecf_score-ecf_top1_score, ecf_margin,
            # EC global (6)
            ecg_score, 1/ecg_rank if ecg_rank<10**6 else 0, float(ecg_rank-1) if ecg_rank<10**6 else 999,
            ecg_present, ecg_score-ecg_top1_score, ecg_margin,
            # CLIP final (6)
            cf_score, 1/cf_rank if cf_rank<10**6 else 0, float(cf_rank-1) if cf_rank<10**6 else 999,
            cf_present, cf_score-cf_top1_score, cf_margin,
            # CLIP global (6)
            cg_score, 1/cg_rank if cg_rank<10**6 else 0, float(cg_rank-1) if cg_rank<10**6 else 999,
            cg_present, cg_score-cg_top1_score, cg_margin,
            # Vote + stats (7)
            vote_ecf, vote_ecg, vote_cf, vote_cg, vote_count, score_mean, score_std,
            # Relative scores (4)
            ecf_rel, ecg_rel, cf_rel, cg_rel,
            # Confidence ratios (4)
            ecf_conf, ecg_conf, cf_conf, cg_conf,
            # Extra (2)
            min_rank, w_score,
        ]
        # Total: 6+6+6+6+7+4+4+2 = 41 dims
        feats.append(feat)
        cand_raw.append(
            (efr["class_raw"] if efr else "") or (egr["class_raw"] if egr else "") or
            (fr["class_raw"]  if fr  else "") or (gr["class_raw"]  if gr  else "") or ck
        )

    X     = np.asarray(feats, dtype=np.float32)
    y_idx = candidate_keys.index(true_key)

    true_ecf_rank, _ = _find_rank_score(ec_f,   true_key)
    true_ecg_rank, _ = _find_rank_score(ec_g,   true_key)
    true_cf_rank,  _ = _find_rank_score(clip_f, true_key)
    true_cg_rank,  _ = _find_rank_score(clip_g, true_key)

    hard_weight = (1.0
                   + (0.20 if true_ecf_rank > 1 else 0)
                   + (0.20 if true_ecg_rank > 1 else 0)
                   + (0.15 if true_cf_rank  > 1 else 0)
                   + (0.15 if true_cg_rank  > 1 else 0))

    return GroupSample(
        image_path=item.get("image_path", ""),
        true_class_raw=true_raw, true_key=true_key,
        candidate_keys=candidate_keys, candidate_raw=cand_raw,
        X=X, y_idx=y_idx, hard_weight=hard_weight,
        ec_final_top1_raw=ecf_top1, ec_global_top1_raw=ecg_top1,
        clip_final_top1_raw=cf_top1, clip_global_top1_raw=cg_top1,
        true_ecf_rank=true_ecf_rank, true_ecg_rank=true_ecg_rank,
        true_cf_rank=true_cf_rank,   true_cg_rank=true_cg_rank,
        final_global_disagree=(normalize_name(cf_top1) != normalize_name(cg_top1)),
    )


# ─────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────

class SplineTransform(nn.Module):
    def __init__(self, dim, grid_size=16, x_min=-3.5, x_max=3.5):
        super().__init__()
        self.dim = dim; self.grid_size = grid_size
        self.x_min = x_min; self.x_max = x_max
        self.register_buffer("grid", torch.linspace(x_min, x_max, grid_size))
        self.coeff      = nn.Parameter(torch.zeros(dim, grid_size))
        self.base_scale = nn.Parameter(torch.ones(dim))
        self.base_bias  = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.coeff, std=0.01)

    def forward(self, x):
        x    = torch.clamp(x, self.x_min, self.x_max)
        idx  = torch.bucketize(x, self.grid)
        idx0 = torch.clamp(idx - 1, 0, self.grid_size - 2)
        idx1 = idx0 + 1
        x0   = self.grid[idx0]; x1 = self.grid[idx1]
        t    = (x - x0) / (x1 - x0 + 1e-8)
        ce   = self.coeff.unsqueeze(0).expand(x.shape[0], -1, -1)
        y0   = torch.gather(ce, 2, idx0.unsqueeze(-1)).squeeze(-1)
        y1   = torch.gather(ce, 2, idx1.unsqueeze(-1)).squeeze(-1)
        return self.base_scale * x + self.base_bias + y0 + t * (y1 - y0)


class ModalityEncoder(nn.Module):
    def __init__(self, feat_dim=6, embed_dim=32, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, embed_dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
    def forward(self, x): return self.net(x)


class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, dropout=0.05, global_dim=17):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.mod_proj   = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(4)])
        self.global_proj = nn.Linear(global_dim, embed_dim)
        self.query_tok  = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff    = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, ecf, ecg, cf, cg, gf):
        N  = ecf.shape[0]
        kv = torch.cat([
            self.mod_proj[0](ecf).unsqueeze(1),
            self.mod_proj[1](ecg).unsqueeze(1),
            self.mod_proj[2](cf).unsqueeze(1),
            self.mod_proj[3](cg).unsqueeze(1),
            self.global_proj(gf).unsqueeze(1),
        ], dim=1)                                      # (N, 5, D)
        q  = self.query_tok.expand(N, -1, -1)          # (N, 1, D)
        a, _ = self.attn(q, kv, kv)
        a  = self.norm1(a + q)
        return self.norm2(self.ff(a) + a).squeeze(1)  # (N, D)


class KANRanker(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, grid_size=16, dropout=0.1,
                 use_feature_encoder=True, modality_embed_dim=32, attention_heads=4):
        super().__init__()
        self.use_feature_encoder = use_feature_encoder

        if use_feature_encoder:
            global_dim = in_dim - 24   # 4 modalities × 6 = 24
            self.ecf_enc = ModalityEncoder(6, modality_embed_dim, dropout * 0.5)
            self.ecg_enc = ModalityEncoder(6, modality_embed_dim, dropout * 0.5)
            self.cf_enc  = ModalityEncoder(6, modality_embed_dim, dropout * 0.5)
            self.cg_enc  = ModalityEncoder(6, modality_embed_dim, dropout * 0.5)
            self.cross   = CrossModalAttention(modality_embed_dim, attention_heads,
                                               dropout * 0.5, global_dim)
            self.fusion  = nn.Linear(modality_embed_dim, hidden_dim)
            kan_in = hidden_dim
        else:
            kan_in = in_dim

        self.sp1  = SplineTransform(kan_in, grid_size)
        self.fc1  = nn.Linear(kan_in, hidden_dim)
        self.sp2  = SplineTransform(hidden_dim, grid_size)
        self.fc2  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.out  = nn.Linear(hidden_dim // 2, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, return_embeddings=False):
        emb = {}
        if self.use_feature_encoder:
            ecf_e = self.ecf_enc(x[:, 0:6])
            ecg_e = self.ecg_enc(x[:, 6:12])
            cf_e  = self.cf_enc(x[:, 12:18])
            cg_e  = self.cg_enc(x[:, 18:24])
            gf    = x[:, 24:]
            if return_embeddings:
                emb.update({"ecf": ecf_e, "ecg": ecg_e, "cf": cf_e, "cg": cg_e})
            fused = self.cross(ecf_e, ecg_e, cf_e, cg_e, gf)
            if return_embeddings:
                emb["fused"] = fused
            h = self.fusion(fused)
        else:
            h = x

        h = self.sp1(h)
        h = F.silu(self.fc1(h))
        h = self.drop(h)
        h = self.sp2(h)
        h = F.silu(self.fc2(h))
        logits = self.out(h).squeeze(-1)
        return (logits, emb) if return_embeddings else logits


# ─────────────────────────────────────────────
#  Loss helpers
# ─────────────────────────────────────────────

def compute_auxiliary_loss(emb: dict, lambda_aux=0.1) -> torch.Tensor:
    keys = [k for k in ["ecf", "ecg", "cf", "cg"] if k in emb]
    if len(keys) < 2:
        return torch.tensor(0.0)
    norms = [F.normalize(emb[k], p=2, dim=1) for k in keys]
    total, count = 0.0, 0
    for i in range(len(norms)):
        for j in range(i + 1, len(norms)):
            total += (norms[i] * norms[j]).sum(dim=1).mean()
            count += 1
    return -lambda_aux * total / count


def compute_sample_loss_weight(g: GroupSample, alpha=0.15, beta=0.15,
                                gamma=0.25, rank_power=1.0) -> float:
    def rw(r): return min(r - 1, 5) if r < 10**6 else 0
    w = (1.0
         + alpha * (rw(g.true_ecf_rank) ** rank_power)
         + alpha * (rw(g.true_ecg_rank) ** rank_power)
         + beta  * (rw(g.true_cf_rank)  ** rank_power)
         + beta  * (rw(g.true_cg_rank)  ** rank_power))
    if g.final_global_disagree:
        w *= (1.0 + gamma)
    return float(min(w, 2.5))


# ─────────────────────────────────────────────
#  Train / Eval
# ─────────────────────────────────────────────

def _split_train_val(n, val_ratio, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = max(1, min(int(n * val_ratio), n - 1))
    return idx[n_val:], idx[:n_val]


def _compute_norm_stats(groups, indices):
    X    = np.concatenate([groups[i].X for i in indices], axis=0)
    mean = X.mean(0, keepdims=True)
    std  = X.std(0,  keepdims=True)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _evaluate_groups(model, groups, indices, X_mean, X_std, device,
                     alpha=0.15, beta=0.15, gamma=0.25, rp=1.0):
    if not indices: return 0.0, 0.0
    model.eval()
    total, correct = 0.0, 0
    with torch.no_grad():
        for i in indices:
            g = groups[i]
            x = torch.tensor((g.X - X_mean) / X_std, dtype=torch.float32, device=device)
            logits = model(x)
            target = torch.tensor([g.y_idx], dtype=torch.long, device=device)
            w = compute_sample_loss_weight(g, alpha, beta, gamma, rp)
            total += w * float(F.cross_entropy(logits.unsqueeze(0), target).item())
            if torch.argmax(logits).item() == g.y_idx:
                correct += 1
    return total / len(indices), correct / len(indices)


def train_kan_ranker(groups, epochs, val_ratio, lr, weight_decay,
                     hidden_dim, grid_size, dropout, patience, seed, device,
                     loss_alpha=0.15, loss_beta=0.15, loss_gamma=0.25,
                     loss_rank_power=1.0, use_feature_encoder=True, lambda_aux=0.1):
    if len(groups) < 5:
        raise ValueError("Need >= 5 samples.")

    train_idx, val_idx = _split_train_val(len(groups), val_ratio, seed)
    X_mean, X_std = _compute_norm_stats(groups, train_idx)
    in_dim = groups[0].X.shape[1]
    print(f"Feature dim: {in_dim}  (expected 41)")

    model = KANRanker(in_dim, hidden_dim, grid_size, dropout, use_feature_encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(ep):
        w = 5
        if ep < w: return (ep + 1) / w
        p = (ep - w) / max(1, epochs - w)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val_acc, best_state, bad = -1.0, None, 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_idx)
        tl, tc = 0.0, 0
        optimizer.zero_grad()

        for step, i in enumerate(train_idx):
            g = groups[i]
            x = torch.tensor((g.X - X_mean) / X_std, dtype=torch.float32, device=device)
            t = torch.tensor([g.y_idx], dtype=torch.long, device=device)

            if use_feature_encoder:
                logits, emb = model(x, return_embeddings=True)
                aux = compute_auxiliary_loss(emb, lambda_aux)
            else:
                logits = model(x); aux = torch.tensor(0.0, device=device)

            ce   = F.cross_entropy(logits.unsqueeze(0), t, label_smoothing=0.1)
            w    = compute_sample_loss_weight(g, loss_alpha, loss_beta, loss_gamma, loss_rank_power)
            loss = (w * ce + aux) / 2

            loss.backward()
            if (step + 1) % 2 == 0 or (step + 1) == len(train_idx):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad()

            tl += float(loss.item()) * 2
            if torch.argmax(logits).item() == g.y_idx: tc += 1

        scheduler.step()
        tl /= len(train_idx)
        ta  = tc / len(train_idx)
        vl, va = _evaluate_groups(model, groups, val_idx, X_mean, X_std, device,
                                   loss_alpha, loss_beta, loss_gamma, loss_rank_power)
        history.append({"epoch": epoch, "train_loss": tl, "train_acc": ta,
                         "val_loss": vl, "val_acc": va})
        print(f"Epoch {epoch:03d} | train_loss={tl:.4f} train_acc={ta:.4f} "
              f"| val_loss={vl:.4f} val_acc={va:.4f}")

        if va > best_val_acc:
            best_val_acc = va
            best_state   = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state: model.load_state_dict(best_state)
    return model, X_mean, X_std, {"best_val_acc": best_val_acc,
                                   "num_train": len(train_idx), "num_val": len(val_idx),
                                   "history": history}


# ─────────────────────────────────────────────
#  Inference / Evaluation
# ─────────────────────────────────────────────

def predict_with_kan(model, item, X_mean, X_std, candidate_pool_k=20,
                     top_k_out=5, device=torch.device("cpu")):
    g = _build_group_sample(item, candidate_pool_k)
    if g is None: return "", torch.empty(0), []
    model.eval()
    with torch.no_grad():
        x     = torch.tensor((g.X - X_mean) / X_std, dtype=torch.float32, device=device)
        logits = model(x)
        probs  = F.softmax(logits, dim=0)
    vals, idxs = torch.topk(probs, min(top_k_out, probs.shape[0]))
    cands = [{"rank": r+1, "class": g.candidate_raw[i], "class_key": g.candidate_keys[i],
               "prob": float(p), "logit": float(logits[i])}
              for r, (p, i) in enumerate(zip(vals.tolist(), idxs.tolist()))]
    return (cands[0]["class"] if cands else ""), probs, cands


def run_evaluation(model, data_list, X_mean, X_std, candidate_pool_k=20,
                   top_k_out=5, device=torch.device("cpu"), flip_log_limit=30):
    correct, total, flips = 0, 0, 0
    outputs = []
    print(f"Evaluating on {len(data_list)} items ...")
    for item in tqdm(data_list, desc="KAN eval"):
        pred, probs, cands = predict_with_kan(model, item, X_mean, X_std,
                                               candidate_pool_k, top_k_out, device)
        if not cands: continue
        true_raw  = item.get("true_class_raw") or item.get("true_class") or ""
        is_correct = normalize_name(pred) == normalize_name(true_raw)
        if is_correct: correct += 1
        total += 1

        ec_f  = _extract_ec_topk_from_section(item.get("final_topk", {}))
        ec_t1 = ec_f[0]["class_raw"] if ec_f else ""
        if normalize_name(ec_t1) != normalize_name(true_raw) and is_correct:
            if flips < flip_log_limit:
                p_max = float(torch.max(probs).item()) if probs.numel() > 0 else 0
                print(f"Flip: {os.path.basename(item.get('image_path',''))}")
                print(f"  EC final wrong: {ec_t1}  →  KAN correct: {pred}  p={p_max:.3f}")
            flips += 1

        outputs.append({"image_path": item.get("image_path", ""), "true_class_raw": true_raw,
                         "pred_class": pred, "is_correct": is_correct, "candidates": cands})

    acc = correct / total * 100 if total > 0 else 0
    print(f"KAN Accuracy: {acc:.2f}%  ({correct}/{total})")
    return {"accuracy": acc, "correct": correct, "total": total, "results": outputs}


# ─────────────────────────────────────────────
#  Checkpoint
# ─────────────────────────────────────────────

def _save_checkpoint(path, model, X_mean, X_std, config):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model_state": model.state_dict(), "X_mean": X_mean,
                "X_std": X_std, "config": config}, path)


def _load_checkpoint(path, device):
    ckpt  = torch.load(path, map_location=device,weights_only=False)
    cfg   = ckpt["config"]
    model = KANRanker(int(cfg["in_dim"]), int(cfg["hidden_dim"]), int(cfg["grid_size"]),
                      float(cfg["dropout"]), bool(cfg.get("use_feature_encoder", True))).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt["X_mean"], ckpt["X_std"], cfg


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    payload    = _load_json(args.ec_json)
    ec_results = _extract_results(payload)
    if not ec_results:
        raise RuntimeError("No results found in JSON.")
    print(f"Loaded {len(ec_results)} items")

    groups = [g for item in ec_results
              if (g := _build_group_sample(item, args.candidate_pool_k)) is not None]
    print(f"Groups built: {len(groups)}")
    if len(groups) < 5:
        raise RuntimeError("Too few valid groups.")

    # Debug check
    g0 = groups[0]
    print(f"\n=== Sample Check ===")
    print(f"True class   : {g0.true_class_raw}")
    print(f"EC final top1 : {g0.ec_final_top1_raw}")
    print(f"EC global top1: {g0.ec_global_top1_raw}")
    print(f"CLIP final top1 : {g0.clip_final_top1_raw}")
    print(f"CLIP global top1: {g0.clip_global_top1_raw}")
    print(f"Feature shape: {g0.X.shape}  (expected ({args.candidate_pool_k}, 41))")
    print(f"====================\n")

    if args.eval_only:
        model, X_mean, X_std, cfg = _load_checkpoint(args.model_path, device)
        train_info = {"loaded": cfg}
    else:
        model, X_mean, X_std, train_info = train_kan_ranker(
            groups, args.epochs, args.val_ratio, args.lr, args.weight_decay,
            args.hidden_dim, args.grid_size, args.dropout, args.patience,
            args.seed, device, args.loss_alpha, args.loss_beta, args.loss_gamma,
            args.loss_rank_power, args.use_feature_encoder, args.lambda_aux,
        )
        cfg = {"in_dim": int(groups[0].X.shape[1]), "hidden_dim": int(args.hidden_dim),
               "grid_size": int(args.grid_size), "dropout": float(args.dropout),
               "candidate_pool_k": int(args.candidate_pool_k),
               "loss_alpha": float(args.loss_alpha), "loss_beta": float(args.loss_beta),
               "loss_gamma": float(args.loss_gamma), "loss_rank_power": float(args.loss_rank_power),
               "use_feature_encoder": bool(args.use_feature_encoder),
               "lambda_aux": float(args.lambda_aux)}
        _save_checkpoint(args.model_path, model, X_mean, X_std, cfg)
        print(f"Saved: {args.model_path}")

    eval_out = run_evaluation(model, ec_results, X_mean, X_std,
                               args.candidate_pool_k, args.top_k_out, device, args.flip_log_limit)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"ec_json": os.path.abspath(args.ec_json), "num_groups": len(groups),
                   "train_info": train_info, "evaluation": eval_out}, f, ensure_ascii=False, indent=2)
    print(f"Saved eval: {args.output_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ec_json",     type=str, required=True,
                   help="EC JSON file (contains EC + CLIP scores for final & global)")
    p.add_argument("--model_path",  type=str, default="weights/kan_fusion_ranker.pth")
    p.add_argument("--output_json", type=str, default="output_json/kan_fusion_eval.json")
    p.add_argument("--candidate_pool_k", type=int,   default=20)
    p.add_argument("--top_k_out",        type=int,   default=5)
    p.add_argument("--epochs",           type=int,   default=150)
    p.add_argument("--val_ratio",        type=float, default=0.2) #=0.2
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--weight_decay",     type=float, default=3e-4)
    p.add_argument("--hidden_dim",       type=int,   default=64)
    p.add_argument("--grid_size",        type=int,   default=16)
    p.add_argument("--dropout",          type=float, default=0.1)
    p.add_argument("--flip_log_limit",   type=int,   default=30)
    p.add_argument("--patience",         type=int,   default=20)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--cpu",              action="store_true")
    p.add_argument("--eval_only",        action="store_true")
    p.add_argument("--loss_alpha",       type=float, default=0.15)
    p.add_argument("--loss_beta",        type=float, default=0.15)
    p.add_argument("--loss_gamma",       type=float, default=0.25)
    p.add_argument("--loss_rank_power",  type=float, default=1.0)
    p.add_argument("--use_feature_encoder", action="store_true", default=True)
    p.add_argument("--no_feature_encoder",  dest="use_feature_encoder", action="store_false")
    p.add_argument("--lambda_aux",       type=float, default=0.1)
    args = p.parse_args()
    main(args)