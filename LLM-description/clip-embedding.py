import os
import json
import torch
try:
    import open_clip
except ImportError:
    open_clip = None
import argparse
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# --- 1. DEVICE ---
# Determine device early; model will be loaded in `main` (configurable via CLI)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preprocess = None
model = None
tokenizer = None

# --- 2. HÀM XỬ LÝ CHÍNH 
def get_optimized_feature(json_data, model, tokenizer, device, preprocess=None):
    class_name = json_data.get('class_name', 'object')

    # 1) Global text
    global_text = f"A photo of a {class_name}. {json_data.get('global_description','')}"
    with torch.no_grad():
        global_tokens = tokenizer([global_text]).to(device)
        global_emb = model.encode_text(global_tokens)
        global_emb = global_emb / global_emb.norm(dim=-1, keepdim=True)

    # 2) Local parts
    part_embs_list = []
    if json_data.get('part_details'):
        for part_name, description in json_data['part_details'].items():
            part_text = f"A close-up photo of the {part_name} of a {class_name}, described as {description}"
            tokens = tokenizer([part_text]).to(device)
            with torch.no_grad():
                emb = model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            part_embs_list.append(emb)
    
    if part_embs_list:
        parts_tensor = torch.stack(part_embs_list).squeeze(1) # Fix dimension khi stack
        local_emb = torch.mean(parts_tensor, dim=0, keepdim=True)
        local_emb = local_emb / local_emb.norm(dim=-1, keepdim=True)
    else:
        local_emb = torch.zeros_like(global_emb)

    # 3) Attributes
    attr_embs_list = []
    if json_data.get('discriminative_attributes'):
        for attr in json_data['discriminative_attributes']:
            attr_text = f"A photo of {class_name} with {attr}"
            tokens = tokenizer([attr_text]).to(device)
            with torch.no_grad():
                emb = model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            attr_embs_list.append(emb)
            
    if attr_embs_list:
        attr_tensor = torch.stack(attr_embs_list).squeeze(1)
        attr_emb = torch.mean(attr_tensor, dim=0, keepdim=True)
        attr_emb = attr_emb / attr_emb.norm(dim=-1, keepdim=True)
    else:
        attr_emb = torch.zeros_like(global_emb)

    # 3.5) Image (optional)
    image_emb = torch.zeros_like(global_emb)
    if json_data.get('image_path') and os.path.exists(json_data['image_path']):
        try:
            img = Image.open(json_data['image_path']).convert('RGB')
            if preprocess is None:
                 raise ValueError("preprocess required")
            image_input = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                image_emb = model.encode_image(image_input)
                image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        except:
            pass # Bỏ qua nếu lỗi ảnh

    # 3.6) Relational (Spatial Relations)
    relation_emb_list = []
    if json_data.get('spatial_relations'): 
        for description in json_data['spatial_relations']: 
            part_text = f"A close-up photo of a {class_name}, with the part {description}"
            tokens = tokenizer([part_text]).to(device)
            with torch.no_grad():
                emb = model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            relation_emb_list.append(emb)

    if relation_emb_list:
        rel_tensor = torch.stack(relation_emb_list).squeeze(1)
        relation_embs = torch.mean(rel_tensor, dim=0, keepdim=True)
        relation_embs = relation_embs / relation_embs.norm(dim=-1, keepdim=True)
    else:
        relation_embs = torch.zeros_like(global_emb)

    # 4) TỔNG HỢP (FEATURE FUSION)
    # Global (text): 30%, Image: 10%, Local: 20%, Attributes: 20%, Relation: 20%
    final_feature = 0.3 * global_emb + 0.1 * image_emb + 0.2 * local_emb + 0.2 * attr_emb + 0.2 * relation_embs
    final_feature = final_feature / final_feature.norm(dim=-1, keepdim=True)

    # Trả về các thành phần cần lưu
    return final_feature, global_emb, relation_embs

# --- 3. MAIN LOOP (Xử lý file/folder và lưu output) ---

def main(args):
    # Setup path
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if not input_root.exists():
        print("Input folder not found!")
        return

    # Determine which datasets (subfolders) to process
    if args.dataset:
        dataset_folder = input_root / args.dataset
        if not dataset_folder.exists():
            print(f"Dataset '{args.dataset}' not found under {input_root}")
            return
        subfolders = [dataset_folder]
    else:
        subfolders = sorted([f for f in input_root.iterdir() if f.is_dir()])

    for dataset_folder in subfolders:
        dataset_name = dataset_folder.name
        print(f"\nProcessing dataset: {dataset_name}")

        # Tạo folder output tương ứng
        save_path = output_root / dataset_name
        save_path.mkdir(parents=True, exist_ok=True)

        # Sắp xếp file JSON theo tên class
        json_files = sorted(list(dataset_folder.glob("*.json")))

        if not json_files:
            print(f"  No json files found in {dataset_folder}")
            continue

        total_before = len(json_files)
        # Dry run: limit to first class for quick test
        if args.dry_run:
            json_files = json_files[:1]
            print(f"  DRY RUN: only processing first file out of {total_before}")
        else:
            print(f"  Found {total_before} class files; processing all (resume skips existing outputs)")

        for json_file in tqdm(json_files, desc=f"  Encoding {dataset_name}"):
            try:
                # Đọc file JSON
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                base_name = json_file.stem
                out_final = save_path / f"{base_name}_final.pt"

                # Resume capability: skip if final embedding exists
                if out_final.exists():
                    print(f"  Skipping {base_name} (final embedding exists)")
                    continue

                # --- GỌI HÀM CỦA BẠN ---
                # Lấy feature từ hàm gốc
                final_feat, global_feat, relation_feat = get_optimized_feature(
                    data, model, tokenizer, device, preprocess
                )

                # --- LƯU OUTPUT (Squeeze để về shape [512]) ---
                # 1. Final
                torch.save(final_feat.squeeze(0).cpu(), out_final)
                
                # 2. Global
                torch.save(global_feat.squeeze(0).cpu(), save_path / f"{base_name}_global.pt")
                
                # 3. Relation
                torch.save(relation_feat.squeeze(0).cpu(), save_path / f"{base_name}_relation.pt")

            except Exception as e:
                print(f"  Error processing {json_file.name}: {e}")

    print("\n--- Done! Saved global, relation, and final embeddings (shape 512). ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="LLM-description", help="Folder chứa file JSON")
    parser.add_argument("--output_dir", type=str, default="LLM-embeddings", help="Folder lưu output")
    parser.add_argument("--dataset", type=str, default=None, help="(Optional) chỉ xử lý dataset con này (tên folder)")
    parser.add_argument("--dry_run", action="store_true", help="Chỉ xử lý 1 file đầu tiên để test giống generate-feature")
    parser.add_argument("--open_clip_model", type=str, default="ViT-B-32", help="OpenCLIP model name (e.g. ViT-B-32, ViT-B-16, ViT-L-14)")
    parser.add_argument("--open_clip_pretrained", type=str, default="laion2b_s34b_b79k", help="Pretrained weights or HF repo id (e.g. 'openai/clip-vit-base-patch16')")
    args = parser.parse_args()

    # Ensure open_clip is installed
    if open_clip is None:
        print("open_clip is not installed in this environment.")
        print("Install with: pip install open-clip-torch")
        print("Or install from source: pip install git+https://github.com/mlfoundations/open_clip.git")
        raise SystemExit(1)

    # Allow using a Hugging Face repo id as --open_clip_pretrained
    pretrained_arg = args.open_clip_pretrained
    if isinstance(pretrained_arg, str) and "/" in pretrained_arg:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("huggingface_hub is required to download from Hugging Face. Install: pip install huggingface-hub")
            raise SystemExit(1)
        print(f"Detected Hugging Face repo id '{pretrained_arg}', downloading repository (may be large)...")
        # On Windows, symlink creation may fail if not running as admin or dev mode.
        # To avoid OSError on symlink creation, disable symlinks for HF hub if needed.
        if os.name == 'nt' and os.environ.get('HF_HUB_DISABLE_SYMLINKS') != '1':
            print("Running on Windows — disabling HF hub symlinks to avoid permission errors (uses more disk space).")
            os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

        try:
            repo_dir = snapshot_download(repo_id=pretrained_arg)
        except OSError as e:
            # Retry with symlinks disabled if we hit Windows symlink privilege error
            if os.name == 'nt' and ('symlink' in str(e).lower() or 'privilege' in str(e).lower() or 'winerror' in str(e).lower()):
                print("Symlink creation failed due to privileges. Retrying download with HF_HUB_DISABLE_SYMLINKS=1...")
                os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'
                repo_dir = snapshot_download(repo_id=pretrained_arg)
            else:
                raise

        # Try to find a weight file inside the downloaded repo; prefer .pt/.pth/.safetensors/.bin
        import glob
        candidates = []
        for ext in ("*.pt", "*.pth", "*.safetensors", "*.bin"):
            candidates += glob.glob(os.path.join(repo_dir, "**", ext), recursive=True)
        if candidates:
            pretrained_arg = candidates[0]
            print(f"Found weight file: {pretrained_arg}")
        else:
            # fallback to using repo dir directly
            pretrained_arg = repo_dir
            print(f"No explicit weight found in repo, using repo directory: {repo_dir}")

    # Load open_clip model based on args so user can select recipe/weights
    try:
        print(f"Loading OpenCLIP model {args.open_clip_model} with pretrained '{pretrained_arg}' ...")
        model, _, preprocess = open_clip.create_model_and_transforms(args.open_clip_model, pretrained=pretrained_arg)
        tokenizer = open_clip.get_tokenizer(args.open_clip_model)
        model.to(device)
        model.eval()
    except Exception as e:
        print("Failed to load requested open_clip model/pretrained. Error:", e)
        print("Hints: ensure the pretrained identifier is valid (HF repo id or known checkpoint) and that dependencies are installed.")
        raise

    main(args)