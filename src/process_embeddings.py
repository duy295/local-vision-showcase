import torch
import open_clip

# Load helper: tạo model CLIP, preprocess (transform) và tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_clip_model(model_name='ViT-B-32', pretrained='laion2b_s34b_b79k'):
    """Trả về: model (đã to(device), eval()), preprocess, tokenizer"""
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device)
    model.eval()
    return model, preprocess, tokenizer


def get_optimized_feature(json_data, model, tokenizer, device, preprocess=None):
    class_name = json_data['class_name']
    
    # --- 1. GLOBAL FEATURE (Tổng quan) ---
    # Thêm Template "A photo of..." để giúp CLIP hiểu đây là ảnh
    global_text = f"A photo of a {class_name}. {json_data['global_description']}"
    
    with torch.no_grad():
        global_tokens = tokenizer([global_text]).to(device)
        global_emb = model.encode_text(global_tokens)
        global_emb /= global_emb.norm(dim=-1, keepdim=True) # Chuẩn hóa

    # --- 2. LOCAL PARTS FEATURE (Chi tiết bộ phận) ---
    # Đây là "Tinh hoa": Gom feature của từng bộ phận lại
    part_embs_list = []
    if json_data.get('part_details'):
        for part_name, description in json_data['part_details'].items():
            # Template chuyên biệt cho bộ phận: "A close-up view of..."
            part_text = f"A close-up photo of the {part_name} of a {class_name}, described as {description}"
            tokens = tokenizer([part_text]).to(device)
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            part_embs_list.append(emb)
        
        # Stack lại và lấy trung bình (Mean Pooling) -> Ra 1 vector đại diện cho cấu trúc
        if part_embs_list:
            parts_tensor = torch.stack(part_embs_list).squeeze()
            local_emb = torch.mean(parts_tensor, dim=0, keepdim=True)
            local_emb /= local_emb.norm(dim=-1, keepdim=True)
        else:
            local_emb = torch.zeros_like(global_emb)
    else:
        local_emb = torch.zeros_like(global_emb)

    # --- 3. ATTRIBUTE FEATURE (Đặc điểm nhận dạng) ---
    attr_embs_list = []
    if json_data.get('discriminative_attributes'):
        for attr in json_data['discriminative_attributes']:
            attr_text = f"A photo of {class_name} with {attr}"
            tokens = tokenizer([attr_text]).to(device)
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            attr_embs_list.append(emb)
                
        if attr_embs_list:
            attr_tensor = torch.stack(attr_embs_list).squeeze()
            attr_emb = torch.mean(attr_tensor, dim=0, keepdim=True)
            attr_emb /= attr_emb.norm(dim=-1, keepdim=True)
        else:
            attr_emb = torch.zeros_like(global_emb)
    else:
        attr_emb = torch.zeros_like(global_emb)

    # --- 3.5. IMAGE FEATURE (nếu có ảnh) ---
    image_emb = torch.zeros_like(global_emb)
    if json_data.get('image_path'):
        from PIL import Image
        img = Image.open(json_data['image_path']).convert('RGB')
        if preprocess is None:
            raise ValueError("preprocess is required to compute image embeddings. Pass preprocess from load_clip_model.")
        image_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_emb = model.encode_image(image_input)
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

    # --- 3.6 Relational image
    relation_emb = []
    if json_data.get('part_details'):
        for part_name, description in json_data['spatial_relations'].items():
            # Template chuyên biệt cho bộ phận: "A close-up view of..."
            part_text = f"A close-up photo of a {class_name}, with the part {description}"
            tokens = tokenizer([part_text]).to(device)
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            relation_emb.append(emb)
        
        # Stack lại và lấy trung bình (Mean Pooling) -> Ra 1 vector đại diện cho cấu trúc
        if part_embs_list:
            parts_tensor = torch.stack(relation_emb).squeeze()
            relation_embs = torch.mean(parts_tensor, dim=0, keepdim=True)
            relation_embs /= local_emb.norm(dim=-1, keepdim=True)
        else:
            relation_embs = torch.zeros_like(global_emb)
    else:
        relation_embs = torch.zeros_like(global_emb)

    # --- 4. TỔNG HỢP (FEATURE FUSION) ---
    # Kết hợp các đặc trưng theo tỷ trọng
    # Sử dụng: Global (text): 40%, Image: 25%, Local: 20%, Attributes: 15%
    final_feature = 0.4 * global_emb + 0.25 * image_emb + 0.2 * local_emb + 0.15 * attr_emb + 0.15 * relation_embs
    final_feature /= final_feature.norm(dim=-1, keepdim=True)

    return final_feature, global_emb, local_emb, image_emb, relation_embs # Trả về cả thành phần để train loss riêng