#!/usr/bin/env python3
"""
🚀 HƯỚNG DẪN SỬ DỤNG LOSS.PY MỚI - QUICK REFERENCE

Lịch sử thay đổi:
✅ Thêm tự động phát hiện dataset (CIFAR100, CUB200, ImageNetR)
✅ Thêm loading json_global và json_final embeddings từ file
✅ Thêm caching mechanism để tối ưu tốc độ
"""

import torch
from backbone.loss import StructureAwareClipLoss

# ⚙️ STEP 1: Chuẩn bị danh sách class names
# Ví dụ cho CIFAR-100 (100 classes)
cifar100_classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
    'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    # ... (tổng 100 classes)
]

# ⚙️ STEP 2: Khởi tạo Loss Function
loss_fn = StructureAwareClipLoss(
    output_json_path=r'c:\Users\FPT SHOP\CODING PROBLEM\LLM via FSCIL\output_json',
    dataset_name='cifar100',              # 🎯 Tự động detect: CIFAR100
    label_to_classname=cifar100_classes,  # List hoặc Dict mapping
    alpha=0.05,    # Threshold cho khác class
    alpha_soft=0.2, # Threshold mềm
    beta=0.9,      # Min boundary
    device='cuda'  # GPU hoặc 'cpu'
)

# 📊 Output:
# 🔍 Đang sử dụng dataset: CIFAR100
# ✓ Đã tìm thấy thư mục: c:\Users\FPT SHOP\CODING PROBLEM\LLM via FSCIL\output_json\cifar100
# ✓ Số lượng class: 100

# ⚙️ STEP 3: Chuẩn bị dữ liệu
batch_size = 8
feat_dim = 512

fuzzy_scores = torch.randn(batch_size, device='cuda')  # [B] Output RelationNet
feat1 = torch.randn(batch_size, feat_dim, device='cuda')  # [B, D]
feat2 = torch.randn(batch_size, feat_dim, device='cuda')  # [B, D]
label1 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device='cuda')  # Label ảnh 1
label2 = torch.tensor([0, 1, 2, 8, 9, 10, 11, 12], device='cuda')  # Label ảnh 2

# ⚙️ STEP 4: Tính Loss
loss = loss_fn(fuzzy_scores, feat1, feat2, label1, label2)
print(f"Loss value: {loss.item():.4f}")

# ============================================================
# 📂 CÁCH 2: Cho CUB-200
# ============================================================
cub200_classes = [
    'acadian_flycatcher', 'american_crow', 'american_goldfinch',
    # ... (200 classes)
]

loss_fn_cub = StructureAwareClipLoss(
    output_json_path=r'c:\Users\FPT SHOP\CODING PROBLEM\LLM via FSCIL\output_json',
    dataset_name='CUB200',              # 🎯 Tự động detect: CUB200
    label_to_classname=cub200_classes,
    device='cuda'
)

# ============================================================
# 📂 CÁCH 3: Dùng Dict thay vì List
# ============================================================
label_mapping = {
    0: 'apple',
    1: 'aquarium_fish',
    2: 'baby',
    # ... (mapping cho tất cả labels)
}

loss_fn_dict = StructureAwareClipLoss(
    output_json_path=r'c:\Users\FPT SHOP\CODING PROBLEM\LLM via FSCIL\output_json',
    dataset_name='cifar100',
    label_to_classname=label_mapping,  # ✅ Cũng hoạt động tốt
    device='cuda'
)

# ============================================================
# 🔧 TÍNH NĂNG: Caching Embeddings
# ============================================================
# 💡 Tự động cache embeddings -> tránh đọc file lặp lại
# Cache key: (label_id, embedding_type='final'/'global')
# 
# Lần đầu load label 0 final: đọc từ file + cache
# Lần thứ 2 load label 0 final: lấy từ cache (nhanh hơn)

# ============================================================
# 🎯 DATASET DETECTION
# ============================================================
# Supported datasets:
# - 'cifar100' hoặc 'CIFAR100' -> output_json/cifar100/
# - 'cub200' hoặc 'CUB200'     -> output_json/CUB200/
# - 'imagenetr' hoặc 'ImageNetR' -> output_json/ImageNetR/

# ✅ Case-insensitive, tự động convert thành đúng folder name

# ============================================================
# 📋 STRUCTURE JSON FILES
# ============================================================
# Mỗi dataset folder có cấu trúc:
# 
# output_json/
# ├── cifar100/
# │   ├── apple_final.json       <- [float, float, ...]
# │   ├── apple_global.json      <- [float, float, ...]
# │   ├── aquarium_fish_final.json
# │   ├── aquarium_fish_global.json
# │   └── ...
# ├── CUB200/
# │   ├── acadian_flycatcher_final.json
# │   ├── acadian_flycatcher_global.json
# │   └── ...
# └── ImageNetR/

# ============================================================
# ⚠️ ERROR HANDLING
# ============================================================
# Dataset không hợp lệ? -> ValueError
# Thư mục không tồn tại? -> FileNotFoundError
# JSON file missing? -> FileNotFoundError
# label_to_classname is None? -> ValueError

# Example error messages:
# ❌ ValueError: Dataset 'invalid' không hợp lệ. Chọn: ['cifar100', 'cub200', 'imagenetr']
# ❌ FileNotFoundError: Không tìm thấy file: output_json/cifar100/apple_final.json

# ============================================================
# 🎓 TRAINING LOOP EXAMPLE
# ============================================================
def train_epoch(model, relation_net, loss_fn, dataloader, optimizer, device='cuda'):
    total_loss = 0
    
    for batch_idx, (
        query_img, support_img, query_label, support_label
    ) in enumerate(dataloader):
        # Move to device
        query_img = query_img.to(device)
        support_img = support_img.to(device)
        query_label = query_label.to(device)
        support_label = support_label.to(device)
        
        # Forward pass
        query_feat = model(query_img)  # [B, 512]
        support_feat = model(support_img)  # [B, 512]
        
        fuzzy_scores = relation_net(query_feat, support_feat)  # [B]
        
        # Compute loss
        loss = loss_fn(
            fuzzy_scores,
            query_feat,
            support_feat,
            query_label,
            support_label
        )
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch Average Loss: {avg_loss:.4f}")
    return avg_loss

# ============================================================
# 💡 TIPS & TRICKS
# ============================================================
# 1️⃣ Embeddings được normalize tự động
# 2️⃣ Caching hoạt động across batches - càng lâu cache càng tốt
# 3️⃣ label_to_classname là bắt buộc
# 4️⃣ Hỗ trợ cả List [class1, class2, ...] hoặc Dict {0: class1, ...}
# 5️⃣ Device parameter cho GPU/CPU acceleration

print("✅ Loss.py đã được cập nhật thành công!")
print("🎯 Sử dụng hàm StructureAwareClipLoss() với 3 tham số chính:")
print("   - output_json_path: Đường dẫn thư mục output_json")
print("   - dataset_name: Tên dataset (cifar100/cub200/imagenetr)")
print("   - label_to_classname: List hoặc Dict ánh xạ label -> class name")
