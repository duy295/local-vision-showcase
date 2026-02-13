"""
Ví dụ về cách sử dụng StructureAwareClipLoss với JSON embeddings

Cơ chế:
- Load CLIP embeddings (global và final) từ file JSON trong thư mục output_json
- Tự động phát hiện dataset đang sử dụng (cifar100, CUB200, ImageNetR)
- Caching embeddings để tránh load lặp lại
"""

import torch
from backbone.loss import StructureAwareClipLoss

# ============ CÁCH 1: Cho CIFAR-100 ============
# Định nghĩa danh sách class theo thứ tự (label 0, 1, 2, ..., 99)
cifar100_classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
    # ... (100 classes)
]

# Khởi tạo Loss Function
loss_fn = StructureAwareClipLoss(
    output_json_path='c:/Users/FPT SHOP/CODING PROBLEM/LLM via FSCIL/output_json',
    dataset_name='cifar100',  # Tự động tìm thư mục cifar100
    label_to_classname=cifar100_classes,
    alpha=0.05,
    alpha_soft=0.2,
    beta=0.9,
    device='cuda'
)

# Sử dụng trong training
fuzzy_scores = torch.randn(8)  # Output từ RelationNet [B]
feat1 = torch.randn(8, 512)    # Feature từ Backbone [B, D]
feat2 = torch.randn(8, 512)    # Feature từ Backbone [B, D]
label1 = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])  # Label của ảnh 1
label2 = torch.tensor([0, 1, 2, 5, 6, 7, 8, 9])  # Label của ảnh 2

loss = loss_fn(fuzzy_scores, feat1, feat2, label1, label2)
print(f"Loss: {loss.item()}")

# ============ CÁCH 2: Cho CUB-200 ============
cub200_classes = [
    'acadian_flycatcher', 'american_crow', 'american_goldfinch',
    # ... (200 classes)
]

loss_fn_cub = StructureAwareClipLoss(
    output_json_path='c:/Users/FPT SHOP/CODING PROBLEM/LLM via FSCIL/output_json',
    dataset_name='CUB200',  # Hoặc 'cub200' - sẽ tự động convert
    label_to_classname=cub200_classes,
    device='cuda'
)

# ============ CÁCH 3: Dùng Dict thay vì List ============
label_to_name = {
    0: 'apple',
    1: 'aquarium_fish',
    2: 'baby',
    # ...
}

loss_fn_dict = StructureAwareClipLoss(
    output_json_path='c:/Users/FPT SHOP/CODING PROBLEM/LLM via FSCIL/output_json',
    dataset_name='cifar100',
    label_to_classname=label_to_name,
    device='cuda'
)

# ============ THÔNG TIN DATASET ============
# Mỗi lần khởi tạo, sẽ in ra:
# 🔍 Đang sử dụng dataset: CIFAR100
# ✓ Đã tìm thấy thư mục: c:/Users/FPT SHOP/CODING PROBLEM/LLM via FSCIL/output_json/cifar100
# ✓ Số lượng class: 100

# ============ CẤU TRÚC CÁC FILE JSON ============
# Mỗi class có 3 file JSON:
# - {class_name}_final.json      (embedding vector cuối cùng)
# - {class_name}_global.json     (embedding vector global)
# - {class_name}_relation.json   (relation embedding)

# Ví dụ:
# output_json/
#   cifar100/
#     apple_final.json       <- [0.001, 0.002, -0.003, ...]
#     apple_global.json      <- [0.001, 0.002, -0.003, ...]
#     apple_relation.json
#     aquarium_fish_final.json
#     ...
#   CUB200/
#     acadian_flycatcher_final.json
#     ...
#   ImageNetR/
#     ...
