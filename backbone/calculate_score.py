import torch

def calculate_exemplar_priority(memory_bank, dataset_indices, labels):
    """
    Tính điểm ưu tiên (Score) cho từng ảnh để chọn Exemplar.
    Score = Tổng Similarity với các ảnh cùng class / Tổng số ảnh.
    
    memory_bank: Object ExemplarMemory
    dataset_indices: List index của toàn bộ tập dữ liệu
    labels: List nhãn tương ứng
    """
    priorities = {} # {global_index: score}
    
    unique_classes = torch.unique(torch.tensor(labels))
    
    for c in unique_classes:
        # Lấy index của tất cả ảnh thuộc class c
        c_idxs = [i for i, label in zip(dataset_indices, labels) if label == c]
        if len(c_idxs) == 0: continue
            
        c_idxs_tensor = torch.tensor(c_idxs, dtype=torch.long)
        
        # Lấy ma trận similarity [Nc, Nc] từ Memory Bank
        sub_mat = memory_bank.get_completed_matrix(c_idxs_tensor)
        
        # Xử lý các giá trị -1 (chưa so sánh)
        # Cách 1: Coi là 0. Cách 2: Bỏ qua. Ở đây ta coi là 0 để phạt ảnh ít được so sánh
        valid_mask = (sub_mat != -1.0).float()
        clean_mat = sub_mat * valid_mask # Chuyển -1 thành -số âm -> sai logic. Cần replace
        clean_mat = torch.where(sub_mat == -1.0, torch.tensor(0.0), sub_mat)
        
        # Tính tổng sim cho từng hàng (từng ảnh)
        sum_scores = torch.sum(clean_mat, dim=1)
        
        # Chuẩn hóa (chia cho số lượng ảnh cùng class)
        norm_scores = sum_scores / len(c_idxs)
        
        for local_i, global_i in enumerate(c_idxs):
            priorities[global_i] = norm_scores[local_i].item()
            
    return priorities