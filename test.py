import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from backbone.feature_extract import HybridResNetBackbone
from backbone.relation_net import BilinearRelationNet
from backbone.loss import StructureAwareClipLoss
from utils.samplers import ClassSpecificBatchSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_single_pair(path_img1, path_img2, backbone_weight_path, relation_weight_path):
    # 1. Khởi tạo lại Model (đảm bảo trùng cấu trúc khi train)
    backbone = HybridResNetBackbone().to(device)
    relation = BilinearRelationNet().to(device)
    
    # 2. Load Weights đã lưu (separate files)
    if backbone_weight_path and torch.cuda.is_available() or True:
        backbone.load_state_dict(torch.load(backbone_weight_path, map_location=device))
    if relation_weight_path:
        relation.load_state_dict(torch.load(relation_weight_path, map_location=device))
    
    backbone.eval()
    relation.eval()

    # 3. Preprocess ảnh
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img1 = transform(Image.open(path_img1).convert('RGB')).unsqueeze(0).to(device)
    img2 = transform(Image.open(path_img2).convert('RGB')).unsqueeze(0).to(device)

    # 4. Chạy Inference
    with torch.no_grad():
        f1 = backbone(img1)
        f2 = backbone(img2)
        score = relation(f1, f2)
        
    print(f"--- KẾT QUẢ TEST ---")
    img1_path = r"E:\DATASET-FSCIL\CUB_200_2011\images\001.Black_footed_Albatross\Black_Footed_Albatross_0074_59.jpg"
    img2_path = r"E:\DATASET-FSCIL\CUB_200_2011\images\001.Black_footed_Albatross\Black_Footed_Albatross_0089_796069.jpg"
    print(f"Ảnh 1: {img1_path}")
    print(f"Ảnh 2: {img2_path}")
    print(f"🔥 Score tương đồng: {score.item():.4f}")
    
    if score.item() > 0.86:
        print("✅ Kết luận: Cùng loài (Predict: SAME)")
    else:
        print("❌ Kết luận: Khác loài (Predict: DIFFERENT)")

# --- CHẠY THỬ ---
test_single_pair(
    path_img1=r"E:\DATASET-FSCIL\CUB_200_2011\images\001.Black_footed_Albatross\Black_Footed_Albatross_0074_59.jpg",
    #path_img2=r"E:\DATASET-FSCIL\CUB_200_2011\images\001.Black_footed_Albatross\Black_Footed_Albatross_0076_417.jpg",
    path_img2=r"E:\DATASET-FSCIL\CUB_200_2011\images\001.Black_footed_Albatross\Black_Footed_Albatross_0024_796089.jpg",
    backbone_weight_path="weights/backbone_full.pth",
    relation_weight_path="weights/relation_full.pth"
)