import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image

class CUB200_First10(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        
        # --- FIX 1: Xử lý đường dẫn thông minh ---
        # Nếu đường dẫn nhập vào đã kết thúc bằng 'images', thì không cộng thêm nữa
        if root.replace('\\', '/').endswith('images'):
            self.image_folder = root
        else:
            self.image_folder = os.path.join(root, 'images')

        # --- FIX 2: Logic Load Data & Biến self.data ---
        if os.path.exists(self.image_folder):
            self.use_fake = False
            print(f"✅ Tìm thấy dữ liệu tại: {self.image_folder}")
            
            # Load dataset thật
            full_dataset = ImageFolder(self.image_folder)
            
            # Lọc 10 class đầu (Label 0-9)
            self.samples = [s for s in full_dataset.samples if s[1] < 10]
            
            self.img_paths = [s[0] for s in self.samples]
            self.targets = [s[1] for s in self.samples]
            
            # QUAN TRỌNG: Gán trực tiếp vào biến self.data (Không dùng @property nữa)
            self.data = {'label': self.targets}
            
            # Cập nhật danh sách tên class
            if hasattr(full_dataset, 'classes'):
                self.classes = full_dataset.classes[:10]
            else:
                self.classes = [f"class_{i}" for i in range(10)]
                
            print(f"📊 Đã load: {len(self.samples)} ảnh (10 Class đầu).")
            
        else:
            # Chế độ Fake Data (Fallback)
            self.use_fake = True
            print(f"⚠️ Không tìm thấy '{self.image_folder}'. Đang dùng DỮ LIỆU GIẢ.")
            
            # Tạo 200 ảnh giả, label 0-9
            self.num_fake = 200
            fake_labels = torch.randint(0, 10, (self.num_fake,)).tolist()
            
            # Gán trực tiếp
            self.data = {'label': fake_labels}
            self.classes = [f"class_{i}" for i in range(10)]

    def __len__(self):
        if self.use_fake:
            return self.num_fake if hasattr(self, 'num_fake') else 100
        return len(self.samples)

    def __getitem__(self, idx):
        if self.use_fake:
            # Tạo ảnh nhiễu [3, 224, 224]
            img = torch.randn(3, 224, 224)
            label = self.data['label'][idx]
            # Trả về 3 giá trị để khớp với main.py: img, label, idx
            return img, label, idx
        else:
            # Load ảnh thật
            path = self.img_paths[idx]
            label = self.targets[idx]
            
            try:
                img = Image.open(path).convert('RGB')
            except:
                # Fallback nếu ảnh lỗi
                img = Image.new('RGB', (224, 224))
            
            if self.transform:
                img = self.transform(img)
                
            return img, label, idx


class CUB200_Full(Dataset):
    """Load toàn bộ 200 classes từ CUB-200-2011 dataset"""
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        
        # --- Xử lý đường dẫn thông minh ---
        if root.replace('\\', '/').endswith('images'):
            self.image_folder = root
        else:
            self.image_folder = os.path.join(root, 'images')

        # --- Load Data ---
        if os.path.exists(self.image_folder):
            self.use_fake = False
            print(f"✅ Tìm thấy dữ liệu tại: {self.image_folder}")
            
            # Load dataset thật - TẤT CẢ CLASSES
            full_dataset = ImageFolder(self.image_folder)
            
            # KHÔNG lọc - lấy tất cả
            self.samples = full_dataset.samples
            
            self.img_paths = [s[0] for s in self.samples]
            self.targets = [s[1] for s in self.samples]
            
            self.data = {'label': self.targets}
            
            # Lấy tất cả tên class
            if hasattr(full_dataset, 'classes'):
                self.classes = full_dataset.classes
            else:
                self.classes = [f"class_{i}" for i in range(200)]
                
            print(f"📊 Đã load: {len(self.samples)} ảnh ({len(self.classes)} Classes).")
            
        else:
            # Chế độ Fake Data (Fallback)
            self.use_fake = True
            print(f"⚠️ Không tìm thấy '{self.image_folder}'. Đang dùng DỮ LIỆU GIẢ (200 Classes).")
            
            # Tạo 2000 ảnh giả, label 0-199
            self.num_fake = 2000
            fake_labels = torch.randint(0, 200, (self.num_fake,)).tolist()
            
            self.data = {'label': fake_labels}
            self.classes = [f"class_{i}" for i in range(200)]

    def __len__(self):
        if self.use_fake:
            return self.num_fake if hasattr(self, 'num_fake') else 1000
        return len(self.samples)

    def __getitem__(self, idx):
        if self.use_fake:
            # Tạo ảnh nhiễu [3, 224, 224]
            img = torch.randn(3, 224, 224)
            label = self.data['label'][idx]
            return img, label, idx
        else:
            # Load ảnh thật
            path = self.img_paths[idx]
            label = self.targets[idx]
            
            try:
                img = Image.open(path).convert('RGB')
            except:
                # Fallback nếu ảnh lỗi
                img = Image.new('RGB', (224, 224))
            
            if self.transform:
                img = self.transform(img)
                
            return img, label, idx