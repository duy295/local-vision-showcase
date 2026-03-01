import os
import json
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Sắp xếp ảnh dựa trên Feature Embedding (ResNet50)")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()

def get_model():
    # Sử dụng ResNet50 đã pre-train trên ImageNet làm "mắt nhìn"
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Loại bỏ lớp phân loại cuối cùng để lấy feature vector (2048 chiều)
    model = nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def main():
    args = parse_args()
    
    # Tự động đặt tên output nếu trống
    if args.output_file is None:
        base_name = os.path.basename(os.path.normpath(args.data_dir))
        args.output_file = f"{base_name}_feature_sorted.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()

    # Chuẩn bị transform cho ảnh (giống chuẩn ImageNet)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if not os.path.exists(args.data_dir):
        print(f"Lỗi: {args.data_dir} không tồn tại.")
        return

    classes = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    final_result = {}

    for class_name in tqdm(classes, desc="Extracting Features"):
        class_path = os.path.join(args.data_dir, class_name)
        img_paths = [os.path.join(class_path, f) for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not img_paths: continue

        features = []
        with torch.no_grad():
            for img_path in img_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_t = transform(img).unsqueeze(0).to(device)
                    feat = model(img_t).cpu().numpy().flatten()
                    features.append(feat)
                except:
                    continue
        
        if not features: continue
        
        features_np = np.array(features)
        # Tính Mean Vector
        class_mean = np.mean(features_np, axis=0)
        
        # Tính khoảng cách Euclidean từ từng ảnh tới tâm
        distances = np.linalg.norm(features_np - class_mean, axis=1)
        sorted_indices = np.argsort(distances)

        class_result = []
        for idx in sorted_indices:
            class_result.append({
                "path": img_paths[idx],
                "distance": float(distances[idx]),
                "rank": int(np.where(sorted_indices == idx)[0][0])
            })
        final_result[class_name] = class_result

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=4)

    print(f"\nXong! Đã lưu thứ tự sắp xếp vào: {args.output_file}")

if __name__ == "__main__":
    main()