"""
Fix cho main.py - Thêm Inference Mode & Improvements

Những sửa cần thiết để tối ưu model cho comparison task
"""

# ============================================================================
# MODIFICATION 1: Thêm --mode parameter vào argparse
# ============================================================================
# Tại function get_args() trong main.py, thêm:

"""
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'inference'],
                       help='Mode: train hoặc inference')
    
    parser.add_argument('--img1', type=str, default=None,
                       help='Path ảnh 1 cho inference mode')
    parser.add_argument('--img2', type=str, default=None,
                       help='Path ảnh 2 cho inference mode')
"""

# ============================================================================
# MODIFICATION 2: Cải tiến Feature Extraction với Normalization
# ============================================================================
# Thay đổi trong HybridResNetBackbone.forward_single():

"""
# BEFORE:
def forward_single(self, x):
    feat = self.backbone(x)
    feat = torch.flatten(feat, 1)
    feat = self.projector(feat)
    return feat

# AFTER (thêm normalization):
def forward_single(self, x):
    feat = self.backbone(x)
    feat = torch.flatten(feat, 1)
    feat = self.projector(feat)
    feat = F.normalize(feat, p=2, dim=1)  # ← ADD THIS
    return feat
"""

# ============================================================================
# MODIFICATION 3: Add inference() function vào main.py
# ============================================================================

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys, os

def inference_mode(args, backbone, relation, device):
    \"\"\"
    Inference function để so sánh 2 ảnh bất kỳ
    \"\"\"
    if args.img1 is None or args.img2 is None:
        print("❌ Inference mode cần --img1 và --img2")
        return
    
    print(f"🔍 Inference mode: So sánh 2 ảnh")
    print(f"   Image 1: {args.img1}")
    print(f"   Image 2: {args.img2}")
    
    # Load images
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        img1 = Image.open(args.img1).convert('RGB')
        img2 = Image.open(args.img2).convert('RGB')
    except Exception as e:
        print(f"❌ Lỗi load ảnh: {e}")
        return
    
    img1_tensor = transform(img1).unsqueeze(0).to(device)
    img2_tensor = transform(img2).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        feat1 = backbone.forward_single(img1_tensor)  # [1, D]
        feat2 = backbone.forward_single(img2_tensor)  # [1, D]
        
        # Get relation score
        score = relation(feat1, feat2).item()
        
        # Get cosine similarity
        cosine_sim = F.cosine_similarity(feat1, feat2).item()
    
    print("\\n" + "="*60)
    print("📊 COMPARISON RESULTS")
    print("="*60)
    print(f"Relation Network Score   : {score:.4f}")
    print(f"Cosine Similarity        : {cosine_sim:.4f}")
    print(f"Combined Score (avg)     : {(score + (cosine_sim+1)/2) / 2:.4f}")
    print("="*60 + "\\n")
    
    # Interpretation
    if score > 0.7:
        print("✅ RẤT GIỐNG NHAU (Very Similar)")
    elif score > 0.5:
        print("⚠️  GIỐNG MỘT PHẦN (Partially Similar)")
    else:
        print("❌ KHÁC NHAU (Different)")

# ============================================================================
# MODIFICATION 4: Update main() function - check mode
# ============================================================================

# Tại cuối function main(), thay phần này:
"""
# BEFORE:
if __name__ == "__main__":
    main()

# AFTER:
if __name__ == "__main__":
    args = get_args()
    
    if args.mode == 'inference':
        # Inference mode - không load dataset
        print("🚀 Starting INFERENCE mode...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = HybridResNetBackbone().to(device)
        relation = BilinearRelationNet().to(device)
        
        # Load trained weights
        if os.path.exists('weights/backbone_full.pth'):
            backbone.load_state_dict(torch.load('weights/backbone_full.pth', map_location=device))
        if os.path.exists('weights/relation_full.pth'):
            relation.load_state_dict(torch.load('weights/relation_full.pth', map_location=device))
        
        backbone.eval()
        relation.eval()
        
        inference_mode(args, backbone, relation, device)
    else:
        # Training mode
        print("🚀 Starting TRAINING mode...")
        main()
"""

# ============================================================================
# MODIFICATION 5: Thêm Model Checkpoint Management
# ============================================================================

# Thêm vào training loop:
"""
# Track best model
best_loss = float('inf')
patience_counter = 0

for epoch in range(args.epochs):
    # ... training code ...
    
    loss_epoch = train_one_epoch(...)
    
    # Save checkpoint if improved
    if loss_epoch < best_loss - args.min_delta:
        best_loss = loss_epoch
        patience_counter = 0
        
        # Save best model
        torch.save(backbone.state_dict(), 'weights/backbone_best.pth')
        torch.save(relation.state_dict(), 'weights/relation_best.pth')
        print(f"✅ Saved best checkpoint at epoch {epoch}")
    else:
        patience_counter += 1
        if patience_counter >= args.patience:
            print(f"⏹️ Early stopping at epoch {epoch}")
            break
"""

# ============================================================================
# MODIFICATION 6: Export Features untuk Retrieval
# ============================================================================

import json

def export_features(model, dataset, output_path='features.json', device='cuda'):
    \"\"\"
    Export features của tất cả ảnh trong dataset để dùng cho retrieval sau
    \"\"\"
    features_dict = {}
    
    with torch.no_grad():
        for idx, (img, label, _) in enumerate(dataset):
            img_tensor = img.unsqueeze(0).to(device)
            feat = model.forward_single(img_tensor)
            
            features_dict[f"img_{idx}"] = {
                'feature': feat.cpu().numpy().tolist(),
                'label': int(label)
            }
            
            if (idx + 1) % 100 == 0:
                print(f"Exported {idx + 1}/{len(dataset)} features")
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(features_dict, f)
    print(f"✅ Exported {len(features_dict)} features to {output_path}")

# ============================================================================
# MODIFICATION 7: Quick Comparison Helper
# ============================================================================

def quick_compare(img_path1, img_path2, backbone, relation, device):
    \"\"\"
    Quick helper function để compare 2 ảnh từ command line
    \"\"\"
    from torchvision import transforms
    from PIL import Image
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img1 = Image.open(img_path1).convert('RGB')
    img2 = Image.open(img_path2).convert('RGB')
    
    img1_t = transform(img1).unsqueeze(0).to(device)
    img2_t = transform(img2).unsqueeze(0).to(device)
    
    with torch.no_grad():
        feat1 = backbone.forward_single(img1_t)
        feat2 = backbone.forward_single(img2_t)
        score = relation(feat1, feat2).item()
    
    return score

# Usage:
# score = quick_compare('bird1.jpg', 'bird2.jpg', backbone, relation, device)
# print(f"Similarity: {score:.2%}")

# ============================================================================
# USAGE EXAMPLES:
# ============================================================================

\"\"\"
# TRAINING:
python train/main.py \\
    --root /path/to/CUB_200_2011/images \\
    --output_json_path /path/to/output_json \\
    --sorted_json_path /path/to/sorted_CUB200.json \\
    --epochs 80 \\
    --batch_size 1024

# INFERENCE (1 cặp ảnh):
python train/main.py \\
    --mode inference \\
    --img1 bird1.jpg \\
    --img2 bird2.jpg

# INFERENCE (dùng inference.py - RECOMMENDED):
python inference.py bird1.jpg bird2.jpg \\
    --backbone weights/backbone_full.pth \\
    --relation weights/relation_full.pth \\
    --concept-dir output_json/CUB200
\"\"\"
