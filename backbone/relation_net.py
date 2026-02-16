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
        # 1. Làm phẳng (Flatten) nếu đầu vào là 4D (ví dụ patch feature)
        if x1.dim() > 2:
            x1 = x1.view(x1.size(0), -1)
        if x2.dim() > 2:
            x2 = x2.view(x2.size(0), -1)

        # 2. Projection (Đưa về không gian chung)
        h1 = self.proj(x1)
        h2 = self.proj(x2)

        # 3. Tính Score Toán học (Thay thế ma trận phức tạp bằng Cosine Similarity chuẩn)
        # F.cosine_similarity tự động chuẩn hóa vector, kết quả từ -1 đến 1
        # Ta dùng (s + 1) / 2 để đưa về [0, 1]
        s_math = (F.cosine_similarity(h1, h2, dim=1) + 1.0) / 2.0

        # 4. Tính Score Học máy (Bilinear)
        f_mul = h1 * h2
        f_dist = torch.abs(h1 - h2)
        f_add = h1 + h2
        combined = torch.cat([f_mul, f_dist, f_add], dim=1)
        
        # Output của refiner có thể rất lớn, ta dùng Sigmoid để ép về [0, 1]
        s_learn = torch.sigmoid(self.bilinear_refiner(combined).view(-1))

        # 5. Kết hợp (Weighted Sum)
        # Quan trọng: Không dùng clamp ngay, để xem giá trị thực tế
        final_score = self.alpha * s_math + self.beta * s_learn

        return torch.clamp(final_score, 0.0, 1.0)