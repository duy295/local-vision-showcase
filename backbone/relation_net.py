
import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearRelationNet(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, epsilon=1e-15):
        super(BilinearRelationNet, self).__init__()
        self.epsilon = epsilon
        
        # 1. Learnable Projection: Chuẩn bị đặc trưng tốt nhất trước khi đưa vào Fuzzy
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 2. Học cách kết hợp: Thay vì chỉ lấy kết quả toán học thuần túy,
        # model sẽ học cách "tin" vào toán học bao nhiêu phần trăm.
        self.alpha = nn.Parameter(torch.tensor([0.65])) # Trọng số cho Fuzzy toán học
        self.beta = nn.Parameter(torch.tensor([0.25]))  # Trọng số cho Bilinear học được
        
        # 3. Phần học bổ trợ (Bilinear) - Giúp bù đắp những gì toán học thuần túy bỏ sót
        # Input: concatenate [f_mul, f_dist, f_add] từ h1, h2 (mỗi cái hidden_dim // 2 = 128)
        # So 128 * 3 = 384
        self.bilinear_refiner = nn.Sequential(
            nn.Linear(hidden_dim // 2 * 3, 64),  # 128*3 = 384 -> 64
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def compute_fuzzy_score(self, X1, X2):
        """
        Đây chính là logic toán học chuẩn từ bài báo của sư đệ.
        Thực hiện Cosine Similarity bằng ma trận để đảm bảo tính đối xứng.
        """
        # S = X1 * X2^T
        S = torch.mm(X1, X2.t())
        
        # Tính chuẩn (Norms) cho X1
        D1 = torch.sqrt(torch.diag(torch.mm(X1, X1.t())))
        D1 = 1.0 / torch.clamp(D1, min=self.epsilon)
        D1 = torch.diag_embed(D1)
        
        # Tính chuẩn (Norms) cho X2
        D2 = torch.sqrt(torch.diag(torch.mm(X2, X2.t())))
        D2 = 1.0 / torch.clamp(D2, min=self.epsilon)
        D2 = torch.diag_embed(D2)
        
        # Kết quả S = D1 * S * D2 (Chuẩn hóa về dải [0, 1])
        S_fuzzy = torch.mm(torch.mm(D1, S), D2)
        return torch.clamp(S_fuzzy, 0.0, 1.0)

    def forward(self, x1, x2):
        # Bước 1: Trích xuất đặc trưng mới qua Projection
        h1 = self.proj(x1)
        h2 = self.proj(x2)
        
        # Bước 2: Tính toán Score theo kiểu TOÁN HỌC (Fuzzy)
        # Vì forward của chúng ta nhận cặp (x1, x2) tương ứng, ta lấy đường chéo của ma trận S
        fuzzy_mat = self.compute_fuzzy_score(h1, h2)
        s_math = torch.diag(fuzzy_mat) # Lấy score của từng cặp tương ứng
        
        # Bước 3: Tính toán Score theo kiểu HỌC MÁY (Bilinear Features)
        f_mul = h1 * h2
        f_dist = torch.abs(h1 - h2)
        f_add = h1 + h2
        combined = torch.cat([f_mul, f_dist, f_add], dim=1)
        s_learn = self.bilinear_refiner(combined).view(-1)
        
        # Bước 4: Kết hợp cả hai (Hybrid)
        # Sư huynh dùng Sigmoid cho phần learn để nó về [0, 1] đồng bộ với Fuzzy
        # Sau đó dùng trọng số alpha, beta để model tự cân bằng.
        final_score = self.alpha * s_math + self.beta * torch.sigmoid(s_learn)
        
        return torch.clamp(final_score, 0.0, 1.0)