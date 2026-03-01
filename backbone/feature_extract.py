from unicodedata import name
import torch
import torch.nn as nn
import torchvision.models as models

class HybridResNetBackbone(nn.Module):
    def __init__(self, output_dim=512):
        super(HybridResNetBackbone, self).__init__()
        
        # Load ResNet50
        resnet = models.resnet50(pretrained=True)
        # Bỏ lớp FC cuối, giữ lại đến Average Pool -> ra vector 2048
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
         # ❄️ FREEZE toàn bộ ResNet
        for name, p in self.backbone.named_parameters():
    # Thường backbone[6] là Layer 3, backbone[7] là Layer 4 trong Sequential
            if "6" in name or "7" in name: 
                p.requires_grad = True
            else:
                p.requires_grad = False
        # Linear để giảm chiều về output_dim (ví dụ 512 cho nhẹ)
        self.projector = nn.Linear(2048, output_dim)
        
        self.output_dim = output_dim

    def forward_single(self, x):
        # x: [B, 3, H, W]
        feat = self.backbone(x) # [B, 2048, 1, 1]
        feat = torch.flatten(feat, 1) # [B, 2048]
        feat = self.projector(feat) # [B, output_dim]
        return feat

    def forward(self, x):
        """
        1. Trích feature ảnh gốc (Global)
        2. Chia 9 patch -> ResNet -> Gộp 4 hướng (Horizontal, Rev-Horizontal, Vertical, Rev-Vertical)
        """
        batch_size = x.size(0)
        
        # --- TH1: Global Feature ---
        global_feat = self.forward_single(x) # [B, D]

        # --- TH2: Patch Features ---
        k = x.shape[2] // 3
        patches = x.unfold(2, k, k).unfold(3, k, k) 
        patches = patches.contiguous().view(batch_size, 3, 9, k, k)
        patches_flat = patches.permute(0, 2, 1, 3, 4).reshape(-1, 3, k, k)
        patch_feats_flat = self.forward_single(patches_flat) 
        
        # Reshape lại về [B, 9, D] để xử lý 4 hướng
        patch_feats = patch_feats_flat.view(batch_size, 9, -1)
        
        # Hướng 1: Row-major (Trái -> Phải)
        # Đại diện: Mean của hàng
        mat_1 = torch.mean(patch_feats, dim=1) # [B, D]
        
        # Hướng 2: Reverse Row-major (Phải -> Trái)
        mat_2 = torch.max(patch_feats, dim=1)[0] # [B, D]
        grid_feats = patch_feats.view(batch_size, 3, 3, -1)
        transposed_feats = grid_feats.permute(0, 2, 1, 3).contiguous()
        col_patch_feats = transposed_feats.view(batch_size, 9, -1)

        # Hướng 3: Column-major (Trên -> Dưới)
        mat_3 = torch.mean(col_patch_feats, dim=1) # [B, D]
        
        # Hướng 4: Reverse Column-major (Dưới -> Trên).
        mat_4 = torch.max(col_patch_feats, dim=1)[0] # [B, D]

        # Tổng hợp 4 hướng lại
        combined_patch_feat = mat_1 + mat_2 + mat_3 + mat_4
        final_feat = (global_feat + combined_patch_feat) / 2.0
        
        return final_feat, global_feat, combined_patch_feat