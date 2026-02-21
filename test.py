import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from backbone.feature_extract import HybridResNetBackbone
from backbone.relation_net import BilinearRelationNet
from backbone.loss import StructureAwareClipLoss
from utils.samplers import ClassSpecificBatchSampler

# --- SCORE COMBINER NEURAL NETWORK (match main.py) ---
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

def test_single_pair(path_img1, path_img2, backbone_weight_path, relation_weight_path, score_combiner_weight_path=None):
    # 1. Khởi tạo lại Models (đảm bảo trùng cấu trúc khi train)
    backbone = HybridResNetBackbone().to(device)
    relation = BilinearRelationNet().to(device)
    score_combiner = ScoreCombinerNet(hidden_dim=64).to(device)
    
    # 2. Load Weights đã lưu (separate files)
    if backbone_weight_path and torch.cuda.is_available() or True:
        backbone.load_state_dict(torch.load(backbone_weight_path, map_location=device))
    if relation_weight_path:
        relation.load_state_dict(torch.load(relation_weight_path, map_location=device))
    if score_combiner_weight_path:
        score_combiner.load_state_dict(torch.load(score_combiner_weight_path, map_location=device))
    
    backbone.eval()
    relation.eval()
    score_combiner.eval()

    # 3. Preprocess ảnh
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img1 = transform(Image.open(path_img1).convert('RGB')).unsqueeze(0).to(device)
    img2 = transform(Image.open(path_img2).convert('RGB')).unsqueeze(0).to(device)

    # 4. Chạy Inference (3-branch architecture)
    with torch.no_grad():
        # Backbone giờ return 3 types của features
        feat1, global_feat1, combined_patch_feat1 = backbone(img1)
        feat2, global_feat2, combined_patch_feat2 = backbone(img2)
        
        # Tính 3 scores từ 3 loại features khác nhau
        scores1 = relation(feat1, feat2)                                    # Local features score
        scores2 = relation(global_feat1, global_feat2)                      # Global features score
        score3 = relation(combined_patch_feat1, combined_patch_feat2)       # Combined patches score
        
        # Kết hợp 3 scores bằng neural network
        score = score_combiner(scores1, scores2, score3)
        
    print(f"--- KẾT QUẢ TEST ---")
    img1_path = r"E:\DATASET-FSCIL\CUB_200_2011 - Copy\images\001.Black_footed_Albatross\Black_Footed_Albatross_0074_59.jpg"
    img2_path = r"E:\DATASET-FSCIL\CUB_200_2011 - Copy\images\003.Sooty_Albatross\Sooty_Albatross_0013_796402.jpg"
    print(f"Ảnh 1: {img1_path}")
    print(f"Ảnh 2: {img2_path}")
    print(f"🔥 Score tương đồng: {score.item():.4f}")
    
    if score.item() > 0.86:
        print("✅ Kết luận: Cùng loài (Predict: SAME)")
    else:
        print("❌ Kết luận: Khác loài (Predict: DIFFERENT)")
path_img1=r"E:\DATASET-FSCIL\CUB_200_2011\images\001.Black_footed_Albatross\Black_Footed_Albatross_0074_59.jpg"
path_img2=r"E:\DATASET-FSCIL\CUB_200_2011\images\001.Black_footed_Albatross\Black_Footed_Albatross_0076_417.jpg"
# --- CHẠY THỬ ---
print(f"Ảnh 1: {path_img1}") # Dùng path_img1 thay vì hard-code string
print(f"Ảnh 2: {path_img2}")
test_single_pair(
    path_img1=r"E:\DATASET-FSCIL\CUB_200_2011\building model\train\001.Black_footed_Albatross\Black_Footed_Albatross_0010_796097.jpg",
    path_img2=r"E:\DATASET-FSCIL\CUB_200_2011\building model\train\200.Common_Yellowthroat\Common_Yellowthroat_0003_190521.jpg", # Chim sẻ
    backbone_weight_path="weights/backbone_full.pth",
    relation_weight_path="weights/relation_full.pth",
    score_combiner_weight_path="weights/score_combiner_full.pth"
)