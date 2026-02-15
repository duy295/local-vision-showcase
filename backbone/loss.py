import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

class StructureAwareClipLoss(nn.Module):
    def __init__(self, output_json_path, dataset_name='cifar100', label_to_classname=None, 
                 alpha=0.2, alpha_soft=0.2, beta=0.8, max_rank_diff=50, device='cpu'):
        super(StructureAwareClipLoss, self).__init__()
        self.alpha = alpha
        self.alpha_soft = alpha_soft
        self.beta = beta
        self.max_rank_diff = float(max_rank_diff)
        self.device = device
        
        # 1. Setup path & mapping
        dataset_map = {'cub200': 'CUB200', 'imagenetr': 'ImageNetR', 'cifar100': 'cifar100'}
        self.dataset_path = os.path.join(output_json_path, dataset_map.get(dataset_name.lower(), 'cifar100'))
        
        if isinstance(label_to_classname, (list, tuple)):
            self.label_to_classname = {i: name for i, name in enumerate(label_to_classname)}
        else:
            self.label_to_classname = label_to_classname
        
        self.embedding_cache = {}

    def _load_embedding(self, label_id):
        cache_key = (label_id, 'final')
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        class_name = self.label_to_classname[label_id]
        json_path = os.path.join(self.dataset_path, f"{class_name}_final.json")
        
        if not os.path.exists(json_path):
            return torch.zeros(512, device=self.device)

        with open(json_path, 'r') as f:
            embedding = torch.tensor(json.load(f), dtype=torch.float32, device=self.device)
        
        embedding = F.normalize(embedding, p=2, dim=0)
        self.embedding_cache[cache_key] = embedding
        return embedding

    def forward(self, fuzzy_scores, feat1, feat2, rank1, rank2, label1, label2):
        # Chặn nhẹ scores để ổn định gradient
        fuzzy_scores = torch.clamp(fuzzy_scores, 1e-4, 1.0 - 1e-4)
        
        is_same = (label1 == label2).float()
        is_diff = 1.0 - is_same

        # Visual Similarity (Feature backbone)
        with torch.no_grad():
            feat_sim = F.cosine_similarity(feat1, feat2).detach()
            feat_sim = torch.clamp(feat_sim, 0.0, 1.0)

        # ====================================================
        # PHẦN 1: CÙNG CLASS (RANK + FEATURE)
        # ====================================================
        rank_diff = torch.abs(rank1 - rank2).float()
        rank_sim = torch.clamp(1.0 - (rank_diff / self.max_rank_diff), 0.0, 1.0)
        
        hybrid_sim_same = 0.5 * rank_sim + 0.5 * feat_sim
        target_same = self.beta + (0.987 - self.beta) * hybrid_sim_same
        
        # Loss MSE chính
        loss_same_mse = F.mse_loss(fuzzy_scores, target_same, reduction='none')
        
        # Penalty: Chỉ phạt khi score thấp hơn beta (SỬA LỖI TẠI ĐÂY)
        loss_penalty_same = torch.pow(F.relu(self.beta - fuzzy_scores), 2)
        
        loss_same_total = (loss_same_mse + 0.5 * loss_penalty_same) * is_same

        # ====================================================
        # PHẦN 2: KHÁC CLASS (CLIP + FEATURE)
        # ====================================================
        loss_diff_total = torch.zeros_like(fuzzy_scores)
        
        if is_diff.sum() > 0:
            batch_size = label1.shape[0]
            emb1, emb2 = [], []
            for i in range(batch_size):
                if is_diff[i] > 0:
                    emb1.append(self._load_embedding(label1[i].item()))
                    emb2.append(self._load_embedding(label2[i].item()))
                else:
                    emb1.append(torch.zeros(512, device=self.device))
                    emb2.append(torch.zeros(512, device=self.device))
            
            clip_sim = torch.sum(torch.stack(emb1) * torch.stack(emb2), dim=1)
            clip_sim = torch.clamp(clip_sim, 0.0, 1.0)
            
            # Hybrid Sim cho Khác Class
            hybrid_sim_diff = 0.5 * clip_sim + 0.5 * feat_sim
            target_diff = self.alpha + (self.alpha_soft * hybrid_sim_diff)
            
            # Loss MSE chính cho Diff class
            loss_diff_mse = F.mse_loss(fuzzy_scores, target_diff, reduction='none')
            
            # Penalty: Chỉ phạt khi score vượt quá alpha
            loss_penalty_diff = torch.pow(F.relu(fuzzy_scores - self.alpha), 2)
            
            loss_diff_total = (loss_diff_mse + 0.5 * loss_penalty_diff) * is_diff

        # ====================================================
        # PHẦN 3: CÂN BẰNG GRADIENT
        # ====================================================
        n_same = torch.clamp(is_same.sum(), min=1.0)
        n_diff = torch.clamp(is_diff.sum(), min=1.0)
        
        l_same = loss_same_total.sum() / n_same
        l_diff = loss_diff_total.sum() / n_diff

        return l_same + l_diff