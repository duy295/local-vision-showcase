import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import sys

# Đảm bảo python tìm thấy các file trong thư mục con
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- IMPORT MODULES ---
from backbone.feature_extract import HybridResNetBackbone
from backbone.relation_net import RelationNetwork
from backbone.loss import StructureAwareClipLoss
from utils.samplers import ClassSpecificBatchSampler
from utils.data_loader import CUB200_First10 
from torchvision import transforms

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./CUB_200_2011', help='Đường dẫn folder dataset')
    parser.add_argument('--clip_path', type=str, default='./data/cub_clip.json')
    parser.add_argument('--epochs', type=int, default=5, help='Chạy thử 5 epoch thôi')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running Test on: {device}")

    # 1. Setup Data & Model
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("⏳ Đang chuẩn bị dữ liệu 10 class...")
    # Class này sẽ tự check, nếu có folder CUB thật thì load 10 class, không thì tạo giả
    train_set = CUB200_First10(args.root, train=True, transform=transform)
    
    # Lấy label list cho Sampler
    # Lưu ý: Class loader ở bước 2 đã có property .data trả về dict chứa label
    labels = train_set.data['label']

    backbone = HybridResNetBackbone().to(device)
    relation = RelationNetwork().to(device)
    loss_fn = StructureAwareClipLoss(args.clip_path, device=device) # Tự tạo CLIP giả bên trong nếu thiếu
    
    optimizer = optim.Adam(list(backbone.parameters()) + list(relation.parameters()), lr=args.lr)

    # 2. Chiến thuật Train Test (Ngắn gọn)
    # Vì test nhanh 10 class nên ta chia phase ngắn lại
    phase1_end = 2  # 2 epoch đầu học cấu trúc
    phase2_end = 4  # 2 epoch sau học phân biệt
    # Epoch 5: Shuffle
    
    print("\n>>> BẮT ĐẦU TEST LUỒNG (SANITY CHECK)")
    
    for epoch in range(args.epochs):
        backbone.train()
        relation.train()
        
        # --- CHỌN SAMPLER ---
        if epoch < phase1_end:
            print(f"[Phase 1] Epoch {epoch+1}: Structure Learning (Same Class Batch)")
            sampler = ClassSpecificBatchSampler(labels, args.batch_size)
            loader = DataLoader(train_set, batch_sampler=sampler)
        elif epoch < phase2_end:
            print(f"[Phase 2] Epoch {epoch+1}: Discrimination (Mixed Batch)")
            loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        else:
            print(f"[Phase 3] Epoch {epoch+1}: Regularization (Shuffle)")
            loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

        total_loss = 0
        batch_count = 0
        
        for imgs, lbls, _ in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            if imgs.size(0) < 2: continue
            
            # Split Batch
            curr_bs = imgs.size(0)
            if curr_bs % 2 != 0: 
                imgs = imgs[:-1]; lbls = lbls[:-1]; curr_bs -= 1
            
            half = curr_bs // 2
            img1, img2 = imgs[:half], imgs[half:]
            lbl1, lbl2 = lbls[:half], lbls[half:]
            
            # Forward
            feat1 = backbone(img1)
            feat2 = backbone(img2)
            scores = relation(feat1, feat2)
            
            loss = loss_fn(scores, feat1, feat2, lbl1, lbl2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_count % 5 == 0:
                 print(f"   Iter {batch_count}: Loss {loss.item():.4f}")
            
        print(f"   ==> Avg Loss Epoch {epoch+1}: {total_loss/max(batch_count, 1):.4f}\n")

    print("✅ TEST THÀNH CÔNG! Model hoạt động tốt.")
    # Không cần lưu model hay tính Ec vì đây chỉ là chạy thử

if __name__ == "__main__":
    main()