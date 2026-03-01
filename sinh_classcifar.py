import pickle

# Đường dẫn đến file meta trong thư mục cifar-100-python sau khi giải nén
meta_file = 'E:\\DATASET-FSCIL\\CIFAR 100\\images'

with open(meta_file, 'rb') as f:
    data = pickle.load(f, encoding='utf-8')
    # Lấy danh sách 100 lớp (fine_label_names)
    fine_label_names = data['fine_label_names']

# Thực hiện đánh số từ 001 đến 100 và thêm vào trước tên class
formatted_classes = [f"{(i+60):03d}.{name}" for i, name in enumerate(fine_label_names)]

# In thử 10 tên đầu tiên
for cls in formatted_classes[:10]:
    print(cls)