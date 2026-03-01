# QUICK START

Tài liệu này mô tả quy trình train và chạy theo đúng chu trình 

## 1) Train backbone để predict score (fuzzy score)

Mục tiêu: train model backbone để dự đoán score cho ảnh mẫu.

Các bước:

1. Chuẩn bị dataset ảnh mẫu.
2. Chạy `train/main.py` với input là đường dẫn dataset ảnh và các tham số cần thiết.
3. Sau khi train xong, weights được lưu trong thư mục `weights/`.

Gợi ý lệnh (tham khảo):

```bash
python train/main.py --root <PATH_TO_IMAGES> --output_json_path <PATH_TO_OUTPUT_JSON> --sorted_json_path <PATH_TO_SORTED_JSON>
```

## 2) Train KAN (pipeline chuẩn)

Mục tiêu: tạo dữ liệu Ec và train KAN để lưu model `.pth`.

Chu trình:

1. Chạy `clip_topk_from_json_fixed.py` để CLIP dự đoán ảnh thuộc những class nào.
2. Chạy `predict_all.py` để sinh file Ec, tính score mỗi ảnh thuộc class đó bao nhiêu.
3. Chạy `kan_fusion_ranker.py` với input là các file Ec để train KAN và lưu model `.pth`.

## 3) Incremental class (few-shot)

Mục tiêu: thêm class mới (few-shot) dựa trên các class base đã có.

Chu trình:

1. Chạy `clip_topk_incre_class.py` để dự đoán ảnh few-shot thuộc các class nào trong base.
2. Chạy `Ec_synthesizer.py` để sinh Ec cho class few-shot dựa trên các class base, từ đó tạo Ec class mới.
3. Chạy `kan_fusion_ranker.py` để dự đoán sử dụng file `.pth` đã train ở bước 2.

---
