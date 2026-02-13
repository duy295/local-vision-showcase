# Hướng Dẫn Chạy Training với Dataset CUB-200

## Command Chính
```bash
py train\main.py --root "E:\DATASET-FSCIL\CUB_200_2011" --output_json_path "C:\Users\FPT SHOP\CODING PROBLEM\LLM via FSCIL\output_json" --sorted_json_path "E:\DATASET-FSCIL\CUB_200_2011\sorted_CUB200.json" --epochs 80 --batch_size 1024 --min_delta 1e-4 --lr 1e-4
```

## Giải Thích Các Arguments

| Argument | Giá Trị | Ý Nghĩa |
|----------|--------|--------|
| `--root` | `E:\DATASET-FSCIL\CUB_200_2011` | Đường dẫn tới thư mục images của CUB-200 dataset |
| `--output_json_path` | `C:\Users\FPT SHOP\CODING PROBLEM\LLM via FSCIL\output_json` | Thư mục chứa embeddings JSON (CUB200 subfolder) |
| `--sorted_json_path` | `E:\DATASET-FSCIL\CUB_200_2011\sorted_CUB200.json` | File JSON chứa rank của các ảnh |
| `--epochs` | `80` | Tổng số epochs training |
| `--batch_size` | `1024` | Số ảnh trong mỗi batch |
| `--min_delta` | `1e-4` | Ngưỡng cải thiện loss tối thiểu |
| `--lr` | `1e-4` | Learning rate của optimizer |

## Cấu Trúc Thư Mục Cần Có

```
E:\DATASET-FSCIL\CUB_200_2011\
├── images\                          # Ảnh dataset
│   ├── class_1\
│   ├── class_2\
│   └── ...
├── sorted_CUB200.json              # ← QUAN TRỌNG: File rank

C:\Users\FPT SHOP\CODING PROBLEM\LLM via FSCIL\
├── output_json\
│   └── CUB200\
│       ├── class_0_final.json
│       ├── class_0_global.json
│       ├── class_1_final.json
│       └── ...
```

## Phase Training (Mặc Định)

```
Phase 1: Structure Learning (Epoch 1-20)     - 20 epochs
Phase 2: Discrimination (Epoch 21-60)         - 40 epochs
Phase 3: Regularization (Epoch 61-80)         - 20 epochs
```

## Tùy Chỉnh Khác

### Chạy với 10 classes (để test nhanh)
```bash
py train\main.py --root "..." --output_json_path "..." --sorted_json_path "..." --num_classes 10 --epochs 20
```

### Thay đổi phase epochs
```bash
py train\main.py --root "..." --output_json_path "..." --sorted_json_path "..." --p1_epochs 15 --p2_epochs 35
```
- Phase 1: 15 epochs
- Phase 2: 35 epochs
- Phase 3: 80 - 50 = 30 epochs

### Thay đổi max_rank_diff (nếu có số ảnh/class khác)
```bash
py train\main.py --root "..." --output_json_path "..." --sorted_json_path "..." --max_rank_diff 60.0
```

### Early Stopping
```bash
py train\main.py --root "..." --output_json_path "..." --sorted_json_path "..." --patience 10
```
Dừng training nếu không cải thiện loss trong 10 epochs liên tiếp

## Output Weights

Weights sẽ được lưu tại:
```
weights/
├── backbone_full.pth
└── relation_full.pth
```

## Lưu Ý Quan Trọng

1. ✅ File `sorted_CUB200.json` **PHẢI** tồn tại, không lệnh sẽ stop
2. ✅ Folder `output_json/CUB200/` phải chứa embeddings JSON files
3. ✅ Dataset phải ở đúng structure (images subfolder)
4. ✅ GPU cần đủ memory (~8GB+ cho batch_size 1024)

## Monitoring Training

Khi chạy, bạn sẽ thấy:
```
🚀 Bắt đầu Training trên thiết bị: cuda
📊 Phase 1 (Structure Learning):    Epoch 1-20 (20 epochs)
📊 Phase 2 (Discrimination):         Epoch 21-60 (40 epochs)
📊 Phase 3 (Regularization):         Epoch 61-80 (20 epochs)

Epoch 1/80 | Phase 1: Structure Learning (Same Class Only)
   [Batch 20] Loss: 2.3456
   [Batch 40] Loss: 1.8234
   >>> End Epoch 1 - Avg Loss: 1.5678 - Time: 45.2s
   ✓ Loss improved to 1.5678 (epoch 1)
```

## Tổng Thời Gian Ước Tính

- **10 classes, 80 epochs**: ~2-3 giờ
- **200 classes, 80 epochs**: ~10-15 giờ (tùy GPU)

---
**Updated:** 2026-02-12
