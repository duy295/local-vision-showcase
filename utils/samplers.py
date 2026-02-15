import numpy as np
import torch
import random
import os
import json
import glob
from torch.utils.data import Sampler

# ========================================================
# 1. HYBRID HARD RELATION SAMPLER (Bản chuẩn - Ép 200 Batches)
# ========================================================
class HybridHardRelationSampler(Sampler):
    """
    Sampler chuyên dụng cho Phase 2 & 3.
    - ÉP CỨNG số lượng batch mỗi epoch (mặc định 200).
    - Đảm bảo tỉ lệ cặp cùng loài (Positive) và khác loài (Negative).
    - Hỗ trợ Hard Negative Mining từ JSON.
    """
    def __init__(self, dataset, batch_size, pos_fraction=0.25, hard_neg_fraction=0.7, sim_matrix=None, num_batches=200):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos_fraction = pos_fraction
        self.hard_neg_fraction = hard_neg_fraction
        self.sim_matrix = sim_matrix or {}
        self.num_batches = num_batches  # <--- CHÌA KHÓA: Ép số lượng batch cố định

        # --- Bước 1: Trích xuất nhãn ---
        if hasattr(dataset, 'data') and isinstance(dataset.data, dict):
            labels_list = dataset.data['label'] # Cho custom dataset dạng dict
        elif hasattr(dataset, 'targets'):
            labels_list = dataset.targets       # Cho ImageFolder chuẩn
        else:
            print("⚠️ Sampler: Đang trích xuất nhãn thủ công...")
            labels_list = [dataset[i][1] for i in range(len(dataset))]

        # --- Bước 2: Gom index theo từng class ---
        self.label_to_indices = {}
        for idx, label in enumerate(labels_list):
            if torch.is_tensor(label): label = label.item()
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
        # --- Bước 3: Chuẩn bị danh sách class hợp lệ ---
        # Chỉ giữ class có >= 2 ảnh để tạo cặp Positive
        self.labels = [l for l in self.label_to_indices.keys() if len(self.label_to_indices[l]) >= 2]
        self.all_labels = list(self.label_to_indices.keys())

    def __iter__(self):
        # --- VÒNG LẶP VÔ TẬN GIẢ LẬP (Infinite Sampling) ---
        # Chạy đúng num_batches lần, không phụ thuộc vào độ dài dataset
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Tính toán số lượng cặp cần thiết cho batch này
            # Ví dụ: Batch 512 -> 256 cặp
            num_pairs = self.batch_size // 2
            
            # Tính số lượng cặp Positive và Negative dựa trên tỷ lệ
            num_pos = int(num_pairs * self.pos_fraction) # Ví dụ: 25%
            num_neg = num_pairs - num_pos                # Còn lại là Negative
            
            # -------------------------------------------------
            # A. TẠO CẶP POSITIVE (Cùng loài)
            # -------------------------------------------------
            for _ in range(num_pos):
                # 1. Chọn ngẫu nhiên 1 loài
                l = random.choice(self.labels)
                # 2. Chọn ngẫu nhiên 2 ảnh của loài đó
                idx1, idx2 = random.sample(self.label_to_indices[l], 2)
                batch_indices.extend([idx1, idx2])
            
            # -------------------------------------------------
            # B. TẠO CẶP NEGATIVE (Khác loài)
            # -------------------------------------------------
            for _ in range(num_neg):
                # 1. Chọn loài thứ nhất (Anchor)
                l1 = random.choice(self.all_labels)
                
                # 2. Chọn loài thứ hai (Negative)
                # Logic: Kiểm tra xem có dùng Hard Negative từ JSON không
                use_hard = (random.random() < self.hard_neg_fraction) and \
                           (l1 in self.sim_matrix) and \
                           (len(self.sim_matrix[l1]) > 0)
                
                if use_hard:
                    # Lấy từ danh sách "chim giống nhau" trong JSON
                    l2 = random.choice(self.sim_matrix[l1])
                else:
                    # Lấy Random (nhưng phải khác l1)
                    l2 = random.choice(self.all_labels)
                    while l2 == l1: # Retry nếu lỡ bốc trùng
                        l2 = random.choice(self.all_labels)
                
                # 3. Chọn ảnh đại diện cho mỗi loài
                idx1 = random.choice(self.label_to_indices[l1])
                idx2 = random.choice(self.label_to_indices[l2])
                batch_indices.extend([idx1, idx2])
            
            # Trả về batch hoàn chỉnh (list các index ảnh)
            yield batch_indices

    def __len__(self):
        # DataLoader sẽ hỏi hàm này để biết thanh progress bar dài bao nhiêu
        return self.num_batches

# ========================================================
# 2. HELPER: Load Hard Negatives từ JSON (Giữ nguyên)
# ========================================================
def load_hard_negatives_from_json(json_folder, dataset):
    """
    Ánh xạ tên loài từ JSON sang Index của Dataset.
    """
    name_to_idx = {}
    if not hasattr(dataset, 'classes'):
        # Fallback nếu dùng custom dataset
        # Giả sử dataset.classes là list tên loài
        try:
             # Cố gắng lấy classes từ label_to_indices nếu có thể, hoặc báo lỗi
             pass 
        except:
             print("⚠️ Dataset không có thuộc tính .classes")
             return {}
    else:
        for i, class_name in enumerate(dataset.classes):
            clean_name = class_name.split('.')[-1].lower().replace(' ', '_')
            name_to_idx[clean_name] = i

    sim_map = {}
    if not os.path.exists(json_folder):
        return sim_map
    
    json_files = glob.glob(os.path.join(json_folder, "*_final.json"))
    for f_path in json_files:
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                src_name = data.get('class_name', '').lower().replace(' ', '_')
                similar_list = data.get('similar_classes', [])
                
                if src_name in name_to_idx:
                    src_idx = name_to_idx[src_name]
                    hard_indices = []
                    for n in similar_list:
                        n_clean = n.lower().replace(' ', '_')
                        if n_clean in name_to_idx:
                            hard_indices.append(name_to_idx[n_clean])
                    
                    if hard_indices:
                        sim_map[src_idx] = hard_indices
        except:
            continue
    
    print(f"✅ Đã kết nối {len(sim_map)} lớp với danh sách Hard Negatives.")
    return sim_map

# ========================================================
# 3. CLASS SPECIFIC SAMPLER (Cập nhật hỗ trợ num_batches)
# ========================================================
class ClassSpecificBatchSampler(Sampler):
    """
    Sampler cho Phase 1. 
    Cập nhật: Cho phép ép num_batches=200 để đồng bộ với Phase 2, 3.
    """
    def __init__(self, labels, batch_size, num_batches=200):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.classes = np.unique(self.labels)
        self.indices_by_class = {c: np.where(self.labels == c)[0] for c in self.classes}
        
        # Nếu muốn tính tự động theo dataset thì dùng logic cũ, 
        # nhưng ở đây ta ưu tiên tham số num_batches truyền vào.
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            # Chọn ngẫu nhiên 1 class
            c = np.random.choice(self.classes)
            indices = self.indices_by_class[c]
            
            # Lấy mẫu có lặp lại (replace=True) để đảm bảo luôn đủ batch_size
            # kể cả khi class đó có ít ảnh hơn batch_size
            if len(indices) == 0: continue # Skip nếu class rỗng
            
            batch = np.random.choice(indices, self.batch_size, replace=(len(indices) < self.batch_size))
            yield batch.tolist()

    def __len__(self):
        return self.num_batches