import torch
from torchvision import transforms

def get_transforms(img_size=224):
    """
    Trả về pipeline preprocess: Resize -> CenterCrop -> Normalize
    """
    # ImageNet stats chuẩn
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((int(img_size * 1.14), int(img_size * 1.14))), # Resize lớn hơn chút
        transforms.CenterCrop((img_size, img_size)), # Chỉ lấy phần trung tâm
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform

# Hàm helper để cắt patch thủ công nếu cần
def crop_patches(image_tensor, grid_size=3):
    """
    Cắt ảnh tensor [C, H, W] thành 9 patches [9, C, h, w]
    """
    c, h, w = image_tensor.shape
    ph, pw = h // grid_size, w // grid_size
    patches = []
    for i in range(grid_size):
        for j in range(grid_size):
            patch = image_tensor[:, i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            patches.append(patch)
    return torch.stack(patches)