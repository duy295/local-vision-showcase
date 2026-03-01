import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearRelationNet(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super(BilinearRelationNet, self).__init__()
        # --- GIỮ NGUYÊN KHỐI NÀY ĐỂ LOAD ĐƯỢC WEIGHT ---
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.alpha = nn.Parameter(torch.tensor([0.75])) 
        self.beta = nn.Parameter(torch.tensor([0.25]))  
        self.bilinear_refiner = nn.Sequential(
            nn.Linear(hidden_dim // 2 * 3, 64), 
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # -----------------------------------------------

    def forward(self, x1, x2):
        """
        x1, x2: Input features (Batch_size, Input_dim)
        """
        if x1.dim() > 2:
            x1 = x1.view(x1.size(0), -1)
        if x2.dim() > 2:
            x2 = x2.view(x2.size(0), -1)
        h1 = self.proj(x1)
        h2 = self.proj(x2)
        s_math = (F.cosine_similarity(h1, h2, dim=1) + 1.0) / 2.0
        f_mul = h1 * h2
        f_dist = torch.abs(h1 - h2)
        f_add = h1 + h2
        combined = torch.cat([f_mul, f_dist, f_add], dim=1)
        s_learn = torch.sigmoid(self.bilinear_refiner(combined).view(-1))
        final_score = self.alpha * s_math + self.beta * s_learn

        return torch.sigmoid(final_score * 2) * 0.98  # Đảm bảo output luôn < 1.0