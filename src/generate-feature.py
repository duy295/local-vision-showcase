import os
import sys
import json
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import yaml
from tqdm import tqdm
from Prompt.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from openai import OpenAI
# LƯU Ý ĐÂY HÀM SINH RA DESCIPTION CỦA ẢNH CỦA CLASS --> SINH PROMT ĐÓ
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_dataset_classes(json_path, dataset_name):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if dataset_name not in data:
        raise ValueError(f"Dataset '{dataset_name}' not found in {json_path}. Available: {list(data.keys())}")
    return data[dataset_name]

def generate_description(client, model, system_prompt, user_template, class_name):
    """Hàm gọi API có xử lý lỗi"""
    # Format prompt với tên class hiện tại
    user_prompt = user_template.format(CLASS_NAME=class_name)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"\n[Error] Generating '{class_name}': {str(e)}")
        return None

def main():
    # 1. Cấu hình Arguments
    parser = argparse.ArgumentParser(description="Generate Visual Descriptions using LLM")
    
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Tên key của dataset trong file json (VD: CUB-200)')
    parser.add_argument('--input_file', type=str, default='datasets.json', 
                        help='Đường dẫn file json chứa danh sách class')
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Đường dẫn file config yaml')
    parser.add_argument('--api_key', type=str, default=None, 
                        help='OpenAI API Key (nếu không set biến môi trường)')
    parser.add_argument('--dry_run', action='store_true', 
                        help='Chạy thử 1 class đầu tiên để test, không chạy hết')

    args = parser.parse_args()

    # 2. Load Config & Data
    cfg = load_config(args.config)
    classes = load_dataset_classes(args.input_file, args.dataset)
    
    # 3. Setup Client
    # Nếu có args.api_key (từ dòng lệnh) thì dùng, nếu không thì dùng key cứng này
    api_key = args.api_key or "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"
    
    if not api_key:
        raise ValueError("API Key is missing...")
    
    client = OpenAI(api_key=api_key)
    # 4. Tạo thư mục output riêng cho dataset này
    output_dir = os.path.join(cfg['paths']['output_base_dir'], args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f" Started generating for dataset: {args.dataset}")
    print(f" Output directory: {output_dir}")
    print(f" Total classes: {len(classes)}")

    if args.dry_run:
        print(" DRY RUN MODE: Chỉ chạy class đầu tiên để kiểm tra.")
        classes = classes[:1]

    # 5. Main Loop
    success_count = 0
    
    for cls_name in tqdm(classes, desc="Processing"):
        # Chuẩn hóa tên file (bỏ ký tự lạ)
        safe_name = cls_name.replace(" ", "_").replace("/", "-").lower()
        file_path = os.path.join(output_dir, f"{safe_name}.json")
        
        # Skip nếu file đã tồn tại (Resume capability)
        if os.path.exists(file_path) :
            print(f"Skipping {safe_name} (File exists)")
            success_count += 1
            continue
        
            
        # Tiền xử lý tên class cho đẹp khi đưa vào prompt (VD: Black_footed_Albatross -> Black footed Albatross)
        readable_name = cls_name.replace("_", " ")
        
        # Gọi hàm sinh
        result = generate_description(
            client=client,
            model=cfg['api']['model'],
            system_prompt=SYSTEM_PROMPT,
            user_template=USER_PROMPT_TEMPLATE,
            class_name=readable_name
        )
        
        if result:
            # Inject thêm metadata vào file kết quả để dễ trace ngược
            result['original_dataset_key'] = args.dataset
            result['original_class_key'] = cls_name
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            success_count += 1
        
        # Sleep nhẹ để tránh rate limit
        time.sleep(0.5)

    print(f"\nCompleted! Successfully generated {success_count}/{len(classes)} classes.")
    print(f"Results saved in 🍀🛤️🚂: {output_dir}")

if __name__ == "__main__":
    main()