import numpy as np
import torch
import random
import os
import json
import glob
from torch.utils.data import Sampler

# ========================================================
# 1. HYBRID HARD RELATION SAMPLER (Bản chuẩn nhất)
# ========================================================
class HybridHardRelationSampler(Sampler):
    """
    Chiến thuật lấy mẫu linh hoạt:
    - pos_fraction=0.25 sẽ tạo ra 8/32 ảnh cùng class (4 cặp pos, 12 cặp neg).
    - Hỗ trợ Hard Negative từ JSON để ép model học phân biệt loài giống nhau.
    """
    def __init__(self, dataset, batch_size, pos_fraction=0.25, hard_neg_fraction=0.7, sim_matrix=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos_fraction = pos_fraction
        self.hard_neg_fraction = hard_neg_fraction
        self.sim_matrix = sim_matrix or {}

        # --- Bước 1: Trích xuất nhãn nhanh (không load ảnh) ---
        if hasattr(dataset, 'data') and isinstance(dataset.data, dict):
            labels_list = dataset.data['label']
        elif hasattr(dataset, 'targets'):
            labels_list = dataset.targets
        else:
            # Chỉ gọi nếu dataset không có sẵn list nhãn
            print("⚠️ Sampler: Đang trích xuất nhãn thủ công (có thể hơi chậm)...")
            labels_list = [dataset[i][1] for i in range(len(dataset))]

        # --- Bước 2: Gom index theo từng class ---
        self.label_to_indices = {}
        for idx, label in enumerate(labels_list):
            if torch.is_tensor(label): label = label.item()
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
        # --- Bước 3: Lọc các class có thể tạo cặp ---
        # Chỉ giữ class có >= 2 ảnh mới tạo được cặp Positive
        self.labels = [l for l in self.label_to_indices.keys() if len(self.label_to_indices[l]) >= 2]
        self.all_labels = list(self.label_to_indices.keys()) # Dùng cho Negative sampling
        
        self.num_batches = len(dataset) // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_indices = []
            num_pairs = self.batch_size // 2
            
            # Tính toán số lượng cặp
            # Với batch_size=32, num_pos sẽ là 4 (nếu fraction=0.25)
            num_pos = int(num_pairs * self.pos_fraction)
            num_neg = num_pairs - num_pos
            
            # --- TẠO CẶP POSITIVE (Cùng loài) ---
            for _ in range(num_pos):
                l = random.choice(self.labels)
                idx1, idx2 = random.sample(self.label_to_indices[l], 2)
                batch_indices.extend([idx1, idx2])
            
            # --- TẠO CẶP NEGATIVE (Khác loài) ---
            for _ in range(num_neg):
                l1 = random.choice(self.all_labels)
                
                # Check xem có dùng Hard Negative từ JSON không
                is_hard = (random.random() < self.hard_neg_fraction) and \
                          (l1 in self.sim_matrix) and len(self.sim_matrix[l1]) > 0
                
                if is_hard:
                    l2 = random.choice(self.sim_matrix[l1])
                else:
                    # Chọn bừa một class khác l1
                    l2 = random.choice([l for l in self.all_labels if l != l1])
                
                idx1 = random.choice(self.label_to_indices[l1])
                idx2 = random.choice(self.label_to_indices[l2])
                batch_indices.extend([idx1, idx2])
            
            yield batch_indices

    def __len__(self):
        return self.num_batches

# ========================================================
# 2. HELPER: Load Hard Negatives từ JSON
# ========================================================
def load_hard_negatives_from_json(json_folder, dataset):
    """
    Ánh xạ tên loài từ JSON sang Index của Dataset.
    """
    name_to_idx = {}
    if not hasattr(dataset, 'classes'):
        print("⚠️ Dataset không có thuộc tính .classes")
        return {}

    for i, class_name in enumerate(dataset.classes):
        # Chuẩn hóa tên: "001.Black_footed_Albatross" -> "black_footed_albatross"
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
                    hard_indices = [name_to_idx[n.lower().replace(' ', '_')] 
                                   for n in similar_list 
                                   if n.lower().replace(' ', '_') in name_to_idx]
                    if hard_indices:
                        sim_map[src_idx] = hard_indices
        except:
            continue
    
    print(f"✅ Đã kết nối {len(sim_map)} lớp với danh sách Hard Negatives.")
    return sim_map

# ========================================================
# 3. CLASS SPECIFIC SAMPLER (Dùng cho Phase 1/Validation)
# ========================================================
class ClassSpecificBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.classes = np.unique(self.labels)
        self.indices_by_class = {c: np.where(self.labels == c)[0] for c in self.classes}
        
        total_batches = 0
        for c in self.classes:
            n = len(self.indices_by_class[c])
            if n > 1:
                total_batches += (n + batch_size - 1) // batch_size
        self.num_batches = total_batches

    def __iter__(self):
        shuffled_classes = np.random.permutation(self.classes)
        for c in shuffled_classes:
            indices = self.indices_by_class[c].copy()
            np.random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) > 1:
                    yield batch.tolist()

    def __len__(self):
        return self.num_batches