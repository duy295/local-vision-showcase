## 📋 THAY ĐỔI CHI TIẾT VỀ loss.py

### ✅ Các cải thiện chính

#### 1. **Tự động phát hiện Dataset**
```python
dataset_name: 'cifar100', 'CUB200', 'ImageNetR'
```
- Tự động kiểm tra dataset hợp lệ
- Tự động tìm thư mục tương ứng trong `output_json/`
- In ra thông tin dataset đang sử dụng

#### 2. **Load CLIP Embeddings từ JSON Files**
Thay vì hardcode embeddings, giờ đọc từ file JSON:
- **Final Embeddings**: `{class_name}_final.json`
- **Global Embeddings**: `{class_name}_global.json`

#### 3. **Hỗ trợ Label Mapping**
Hai cách định nghĩa class names:
```python
# Cách 1: List (tự động convert thành dict theo index)
label_to_classname = ['apple', 'aquarium_fish', 'baby', ...]

# Cách 2: Dict (ánh xạ trực tiếp)
label_to_classname = {0: 'apple', 1: 'aquarium_fish', ...}
```

#### 4. **Caching Mechanism**
- Tránh load file JSON lặp lại
- Cache key: `(label_id, embedding_type)` 
- Tiết kiệm I/O và thời gian training

### 📝 Constructor mới

```python
StructureAwareClipLoss(
    output_json_path,       # Đường dẫn tới thư mục output_json
    dataset_name='cifar100', # Dataset: 'cifar100', 'CUB200', 'ImageNetR'
    label_to_classname=None, # List hoặc Dict ánh xạ label -> class name
    alpha=0.05,              # Threshold cho khác class
    alpha_soft=0.2,          # Threshold mềm cho semantic close
    beta=0.9,                # Min boundary cho same class
    device='cpu'             # GPU/CPU
)
```

### 🔄 Forward Method

Vẫn giữ nguyên logic, nhưng giờ:
1. Load embeddings động từ JSON files (cached)
2. Sử dụng **final embeddings** để tính CLIP similarity
3. Có sẵn **global embeddings** nếu cần dùng sau

```python
def forward(self, fuzzy_scores, feat1, feat2, label1, label2):
    # Load final embeddings từ JSON
    emb1_final = self._load_embedding(label1, type='final')
    emb2_final = self._load_embedding(label2, type='final')
    
    # Tính cosine similarity
    clip_sim_final = torch.sum(emb1_final * emb2_final, dim=1)
    
    # Phân loại độ khó và tính loss
    ...
```

### 📂 Cấu trúc thư mục JSON

```
output_json/
├── cifar100/
│   ├── apple_final.json
│   ├── apple_global.json
│   ├── aquarium_fish_final.json
│   ├── aquarium_fish_global.json
│   └── ...
├── CUB200/
│   ├── acadian_flycatcher_final.json
│   ├── acadian_flycatcher_global.json
│   └── ...
└── ImageNetR/
    └── ...
```

### 🛠️ Ví dụ sử dụng

```python
# CIFAR-100
loss_fn = StructureAwareClipLoss(
    output_json_path='output_json/',
    dataset_name='cifar100',
    label_to_classname=cifar100_classes,
    device='cuda'
)

# CUB-200
loss_fn = StructureAwareClipLoss(
    output_json_path='output_json/',
    dataset_name='CUB200',
    label_to_classname=cub200_classes,
    device='cuda'
)

# Training loop
loss = loss_fn(fuzzy_scores, feat1, feat2, label1, label2)
```

### ⚡ Output khi khởi tạo

```
🔍 Đang sử dụng dataset: CIFAR100
✓ Đã tìm thấy thư mục: output_json/cifar100
✓ Số lượng class: 100
```

### 🐛 Error Handling

- ✅ Kiểm tra dataset hợp lệ
- ✅ Kiểm tra thư mục tồn tại
- ✅ Kiểm tra file JSON có sẵn
- ✅ Kiểm tra label_to_classname không None
- ✅ Xử lý label mapping tự động

---

**Lợi ích:**
- ✨ Tự động phát hiện dataset
- 🚀 Caching giúp tăng tốc độ
- 📊 Load embeddings động từ file JSON
- 🔒 An toàn hơn với error checking
- 📚 Project structure rõ ràng hơn
