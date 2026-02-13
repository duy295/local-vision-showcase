import torch
import torch.nn as nn
import argparse
from backbone.feature_extract import HybridResNetBackbone
from backbone.relation_net import RelationNetwork

class FuzzyModel(nn.Module):
    def __init__(self, args):
        super(FuzzyModel, self).__init__()
        self.backbone = HybridResNetBackbone(output_dim=args.feature_dim)
        self.relation = RelationNetwork(input_dim=args.feature_dim)
        
    def forward(self, x1, x2=None):
        """
        Nếu chỉ có x1 -> Trích feature (dùng cho inference/indexing)
        Nếu có x1, x2 -> Trích feature + So sánh (dùng cho training)
        """
        feat1 = self.backbone(x1)
        
        if x2 is not None:
            feat2 = self.backbone(x2)
            score = self.relation(feat1, feat2)
            return score
        
        return feat1

def get_args():
    parser = argparse.ArgumentParser(description="Fuzzy Learning Machine")
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--clip_path', type=str, default='./data/clip_similarity.json')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    model = FuzzyModel(args)
    print("Model loaded successfully with args:", args)