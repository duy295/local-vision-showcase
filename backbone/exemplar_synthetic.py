import torch
import numpy as np

class ExemplarMemory:
    def __init__(self, num_samples, device='cpu'):
        """
        Lưu trữ ma trận tương đồng kích thước N x N (N = tổng số ảnh train)
        Giá trị -1 nghĩa là chưa được so sánh.
        """
        self.num_samples = num_samples
        self.device = device
        # Dùng CPU memory để tiết kiệm VRAM, chỉ load batch cần thiết
        # Hoặc dùng sparse matrix nếu N quá lớn.
        # Ở đây giả sử N vừa phải (<50k)
        self.sim_matrix = torch.full((num_samples, num_samples), -1.0, device='cpu') 

    def update(self, indices_1, indices_2, scores):
        """
        Cập nhật điểm similarity cho batch hiện tại vào bộ nhớ chung.
        indices_1: [B] (chỉ số global của ảnh batch 1)
        indices_2: [B]
        scores: [B]
        """
        # Chuyển về CPU để update
        idx1 = indices_1.cpu()
        idx2 = indices_2.cpu()
        s = scores.detach().cpu()
        
        # Update ma trận đối xứng
        self.sim_matrix[idx1, idx2] = s
        self.sim_matrix[idx2, idx1] = s # Đối xứng

    def get_completed_matrix(self, class_indices):
        """
        Lấy ma trận con của các ảnh thuộc 1 class cụ thể
        """
        # class_indices: List các index của ảnh thuộc class C
        sub_matrix = self.sim_matrix[class_indices][:, class_indices]
        return sub_matrix