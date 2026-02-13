import torch
import torch.nn as nn

class BilinearRelationNet(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super(BilinearRelationNet, self).__init__()
        # Projection layer để convert feature từ input_dim -> hidden_dim
        self.proj = nn.Linear(input_dim, hidden_dim)
        
        # Layer để compute relation score từ 2 features được nối
        self.layer = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
           # nn.Sigmoid() # Bắt buộc để ra range [0, 1]
        )
        # Classifier để compute final score từ 3 phép toán (mul, dist, add)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            #nn.Sigmoid() # Ra Score 0-1
        )

    def forward(self, x1, x2):
        # Project features
        h1 = self.proj(x1)
        h2 = self.proj(x2)
        
        # Compute 3 relation metrics
        feat_mul = h1 * h2        
        feat_dist = torch.abs(h1 - h2)
        feat_add = h1 + h2        
        
        # Combine và đưa vào classifier
        combined = torch.cat([feat_mul, feat_dist, feat_add], dim=1)
        x = self.classifier(combined) 
        return torch.sigmoid(x).view(-1)