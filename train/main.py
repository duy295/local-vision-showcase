import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import sys
import time
import json
import random
import numpy as np
from torch.utils.data import Sampler

# --- SETUP PATHS TRƯỚC KHI IMPORT CUSTOM MODULES ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- IMPORT CUSTOM MODULES ---
from backbone.feature_extract import HybridResNetBackbone
from backbone.relation_net import BilinearRelationNet
from backbone.loss import StructureAwareClipLoss
from utils.samplers import ClassSpecificBatchSampler, HybridHardRelationSampler, load_hard_negatives_from_json
from torchvision import transforms

# --- SCORE COMBINER NEURAL NETWORK ---
class ScoreCombinerNet(torch.nn.Module):
    """Kết hợp 3 scores từ các branch khác nhau bằng neural network.
    Học được non-linear relationships giữa 3 scores để sinh ra final score.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, hidden_dim),              # 3 scores -> hidden
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1)  # Output raw score (không sigmoid)
        )
    
    def forward(self, scores1, scores2, scores3):
        """
        Args:
            scores1: Similarity scores từ local features [batch_size]
            scores2: Similarity scores từ global features [batch_size]
            scores3: Similarity scores từ combined patches [batch_size]
        Returns:
            combined_score: Kết hợp 3 scores [0, 1] [batch_size]
        """
        # Stack 3 scores thành [batch_size, 3]
        combined_input = torch.stack([scores1, scores2, scores3], dim=1)
        # Đưa qua neural network
        output = self.net(combined_input)  # [batch_size, 1]
        # Clamp output về [0, 1] để match input range
        return torch.clamp(output.squeeze(-1), 0, 1)  # [batch_size]

# --- DATASET IMPORT HOẶC FALLBACK ---
try:
    from utils.data_loader import CUB200_First10, CUB200_Full
except ImportError:
    print("⚠️ Không tìm thấy utils.data_loader. Đang dùng Dummy Dataset để test luồng...")
    from torch.utils.data import Dataset
    class CUB200_First10(Dataset):
        def __init__(self, root, train=True, transform=None):
            # Giả lập 100 ảnh, 10 class
            self.data = {'label': torch.randint(0, 10, (100,)).tolist()} 
            self.transform = transform
            # QUAN TRỌNG: Phải có list classes để mapping
            self.classes = [f"class_{i}" for i in range(10)] 
            
        def __len__(self): return 100
        def __getitem__(self, idx):
            return torch.randn(3, 224, 224), self.data['label'][idx], idx
    
    class CUB200_Full(Dataset):
        def __init__(self, root, train=True, transform=None):
            # Giả lập 1000 ảnh, 200 classes
            self.data = {'label': torch.randint(0, 200, (1000,)).tolist()} 
            self.transform = transform
            self.classes = [f"class_{i}" for i in range(200)] 
            
        def __len__(self): return 1000
        def __getitem__(self, idx):
            return torch.randn(3, 224, 224), self.data['label'][idx], idx

# --- HÀM TẠO DUMMY JSON THEO CẤU TRÚC MỚI ---
def gen_dummy_clip_structure(base_path, dataset_name, class_names):
    """
    Tạo file JSON giả lập (dummy) cho các class có embeddings bị thiếu.
    Cấu trúc: base_path/dataset_name/{classname}_final.json
    """
    target_dir = os.path.join(base_path, dataset_name)
    os.makedirs(target_dir, exist_ok=True)
    
    missing_count = 0
    for cls in class_names:
        final_path = os.path.join(target_dir, f"{cls}_final.json")
        global_path = os.path.join(target_dir, f"{cls}_global.json")
        
        # Chỉ tạo nếu file thiếu
        if not os.path.exists(final_path):
            emb = torch.randn(512).tolist()
            with open(final_path, 'w') as f:
                json.dump(emb, f)
            missing_count += 1
        
        if not os.path.exists(global_path):
            emb = torch.randn(512).tolist()
            with open(global_path, 'w') as f:
                json.dump(emb, f)
    
    if missing_count > 0:
        print(f"✓ Tạo {missing_count} dummy CLIP file thiếu tại: {target_dir}")

def extract_class_names_from_files(output_json_path, dataset_name):
    """
    Extract class names từ file names trong folder output_json/dataset_name/
    (e.g., từ "acadian_flycatcher_final.json" hoặc "acadian_flycatcher_global.json" -> "acadian_flycatcher")
    """
    target_dir = os.path.join(output_json_path, dataset_name)
    if not os.path.exists(target_dir):
        return None
    
    try:
        files = os.listdir(target_dir)
        class_names_set = set()
        for f in files:
            if f.endswith('.json'):
                # Loại bỏ suffixes: _final.json, _global.json, _relation.json, vv
                # Giữ lại tên base
                cls_name = f.replace('.json', '')
                # Strip các suffix như _final, _global, _relation
                for suffix in ['_final', '_global', '_relation']:
                    if cls_name.endswith(suffix):
                        cls_name = cls_name[:-len(suffix)]
                        break
                if cls_name:
                    class_names_set.add(cls_name)
        
        if class_names_set:
            class_names_list = sorted(list(class_names_set))
            print(f"✓ Extracted {len(class_names_list)} class names từ {target_dir}")
            return class_names_list
    except Exception as e:
        print(f"⚠️ Lỗi extract class names từ files: {e}")
    
    return None

def normalize_class_name(class_name):
    """
    Normalize class name để match với format trong files JSON.
    Ví dụ: "070.Green_Violetear" -> "green_violetear"
    - Strip số đằng trước (format: XXX.)
    - Convert to lowercase
    - Replace spaces/hyphens với underscores
    """
    # Strip số đằng trước (e.g., "070.Green_Violetear" -> "Green_Violetear")
    if '.' in class_name:
        class_name = class_name.split('.', 1)[1]
    
    # Convert to lowercase
    class_name = class_name.lower()
    
    # Replace spaces và hyphens với underscores
    class_name = class_name.replace(' ', '_').replace('-', '_')
    
    return class_name

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Đường dẫn tới folder images dataset (e.g., E:\\DATASET-FSCIL\\CUB_200_2011\\images)')
    
    parser.add_argument('--output_json_path', type=str, required=True, help='Đường dẫn tới folder chứa embeddings JSON (e.g., C:\\...\\output_json)')
    
    parser.add_argument('--sorted_json_path', type=str, required=True, help='Đường dẫn tới file sorted_CUB200.json (e.g., E:\\DATASET-FSCIL\\CUB_200_2011\\sorted_CUB200.json)')
    
    parser.add_argument('--epochs', type=int, default=80, help='Tổng số epoch')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--patience', type=int, default=10, help='Số epoch không cải thiện loss để dừng sớm')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Giảm tối thiểu để coi là cải thiện')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    # Chọn số lượng classes
    parser.add_argument('--num_classes', type=int, default=200, help='Số lượng classes: 10 hoặc 200 (default: 200)')
    parser.add_argument('--test_10_classes', action='store_true', help='[DEPRECATED] Dùng --num_classes 10 thay vào đó')
    
    # Loss params
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--alpha_soft', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=0.9)
    
    parser.add_argument('--p1_epochs', type=int, default=5, help='Phase 1 epochs (default: 20)')
    parser.add_argument('--p2_epochs', type=int, default=20, help='Phase 2 epochs (default: 40)')
    
    # Max rank difference for normalization
    parser.add_argument('--max_rank_diff', type=float, default=60.0, help='Max rank difference cho normalization (estimate: ~số ảnh/class)')
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Handle deprecated --test_10_classes flag
    if args.test_10_classes:
        args.num_classes = 10
        print("⚠️ --test_10_classes is deprecated. Use --num_classes 10 instead.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Bắt đầu Training trên thiết bị: {device}")

    # Debug prints: confirm whether CUDA is available and which device is used
    print("Using CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("CUDA device:", torch.cuda.get_device_name(torch.cuda.current_device()))
        except Exception:
            pass
    
    dataset_name = 'CUB200' # Tên dataset chuẩn

    # 1. Data Setup
    transform = transforms.Compose([
    # Thay Resize + CenterCrop bằng RandomResizedCrop
    # Nó sẽ lấy một vùng ngẫu nhiên và scale lên 224x224
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    
    # Lật ảnh ngẫu nhiên (Con chim nhìn trái hay nhìn phải vẫn là con chim đó)
    transforms.RandomHorizontalFlip(p=0.5),
    
    # Thay đổi nhẹ độ sáng, tương phản để model không bị đánh lừa bởi điều kiện ánh sáng
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

    print("⏳ Đang load dữ liệu...")
    try:
        # Chọn dataset class dựa trên num_classes
        if args.num_classes == 10:
            print("   📌 Mode: Training 10 classes")
            train_set = CUB200_First10(args.root, train=True, transform=transform)
        else:
            print("   📌 Mode: Training tất cả 200 classes")
            train_set = CUB200_Full(args.root, train=True, transform=transform)
        
        # --- LẤY DANH SÁCH CLASS NAMES (Bắt buộc cho Loss mới) ---
        if hasattr(train_set, 'classes'):
            class_names = train_set.classes
        else:
            # Fallback nếu dataset không có thuộc tính classes
            # Quét label max để đoán số class
            max_label = 0
            for _, y, _ in train_set:
                if y > max_label: max_label = y
            class_names = [f"class_{i}" for i in range(max_label + 1)]
            print(f"⚠️ Dataset không có thuộc tính .classes, tự động tạo: {len(class_names)} class giả.")
        
        # Debug: In class names đầu tiên để kiểm tra format
        print(f"📋 Sample class names (from dataset): {class_names[:3] if len(class_names) >= 3 else class_names}")
        
        # Cố gắng extract class names từ files trong output_json nếu tồn tại
        # (để match với tên file thực tế nếu khác với dataset.classes)
        extracted_class_names = extract_class_names_from_files(args.output_json_path, dataset_name)
        if extracted_class_names:
            print(f"📁 Found {len(extracted_class_names)} classes từ output_json folder")
            class_names = extracted_class_names  # Use extracted names
        else:
            # Fallback: normalize class names từ dataset (strip số, convert to lowercase)
            print(f"📋 Normalizing class names (removing leading numbers, converting to lowercase)...")
            class_names = [normalize_class_name(cls) for cls in class_names]
            print(f"   Sample normalized names: {class_names[:3] if len(class_names) >= 3 else class_names}")

        # Lấy label list cho Sampler
        if hasattr(train_set, 'data') and isinstance(train_set.data, dict):
             train_labels = train_set.data['label'] if isinstance(train_set.data['label'], list) else train_set.data['label'].tolist()
        elif hasattr(train_set, 'targets'):
             train_labels = train_set.targets
        else:
            train_labels = [y for _, y, _ in train_set]
            
        print(f"✅ Đã load {len(train_set)} ảnh training.")
        print(f"📋 Số lượng Class: {len(class_names)}")
        
    except Exception as e:
        print(f"❌ Lỗi load data: {e}")
        return

    # 2. Load sorted per-class JSON (from explicit path)
    sorted_json_path = args.sorted_json_path
    basename_rank_map = {}
    if os.path.exists(sorted_json_path):
        try:
            with open(sorted_json_path, 'r', encoding='utf-8') as f:
                sorted_data = json.load(f)
            # sorted_data: { class_name: [ {path:..., distance:..., rank:...}, ... ], ... }
            for cls, items in sorted_data.items():
                for entry in items:
                    p = entry.get('path')
                    r = entry.get('rank')
                    if p is None or r is None: continue
                    b = os.path.basename(p)
                    try:
                        basename_rank_map[b] = float(r)
                    except Exception:
                        basename_rank_map[b] = 0.0
            print(f"✓ Loaded sorted ranks file with {len(basename_rank_map)} entries from {sorted_json_path}")
        except Exception as e:
            print(f"⚠️ Không thể đọc {sorted_json_path}: {e}")
            return
    else:
        print(f"❌ File sorted json không tìm thấy tại: {sorted_json_path}")
        return

    # 3. Model & Loss Setup
    print("🛠️ Khởi tạo Model & Loss...")
    backbone = HybridResNetBackbone().to(device)
    relation = BilinearRelationNet().to(device)
    score_combiner = ScoreCombinerNet(hidden_dim=64).to(device)  # ← ScoreCombinerNet
    
    for name, param in backbone.backbone.named_parameters():
        if "layer4" in name: # Mở khóa layer cuối cùng của ResNet
            param.requires_grad = True

# Cập nhật lại Optimizer (thêm linear_combiner vào)
    trainable_params = [
    {'params': filter(lambda p: p.requires_grad, backbone.parameters()), 'lr': args.lr * 0.1}, # Backbone chạy chậm
    {'params': relation.parameters(), 'lr': args.lr}, # RelationNet chạy tốc độ bình thường
    {'params': score_combiner.parameters(), 'lr': args.lr}  # ← ScoreCombinerNet cùng tốc độ relation
]

    optimizer = optim.Adam(trainable_params, betas=(0.9, 0.999), eps=1e-8)
    # Debug: verify models are on the expected device and show basic GPU memory usage
    try:
        print("Backbone device:", next(backbone.parameters()).device)
        print("Relation device:", next(relation.parameters()).device)
    except StopIteration:
        print("Warning: models have no parameters to check device.")
    if torch.cuda.is_available():
        try:
            print("GPU memory allocated (MB):", torch.cuda.memory_allocated()/1024**2)
            print("GPU memory reserved (MB):", torch.cuda.memory_reserved()/1024**2)
        except Exception:
            pass

    print("ResNet backbone:")
    print("  Trainable:",
      sum(p.numel() for p in backbone.backbone.parameters() if p.requires_grad))
    print("  Frozen:",
      sum(p.numel() for p in backbone.backbone.parameters() if not p.requires_grad))

    print("Projector:")
    print("  Trainable:",
      sum(p.numel() for p in backbone.projector.parameters() if p.requires_grad))
    trainable = sum(p.numel() for p in relation.parameters() if p.requires_grad)
    total = sum(p.numel() for p in relation.parameters())

    print(f"🔥 RelationNet trainable params: {trainable:,}")
    print(f"📦 RelationNet total params:     {total:,}")
    
    # Debug: Print ScoreCombinerNet status
    print(f"🧠 ScoreCombinerNet initialized (hidden_dim=64)")
    print(f"🧠 ScoreCombinerNet params: {sum(p.numel() for p in score_combiner.parameters()):,}")
    
    # --- KHỞI TẠO LOSS và TẠO DUMMY CLIPS NẾU THIẾU ---
    # Auto-generate dummy CLIP embeddings nếu files bị thiếu
    gen_dummy_clip_structure(args.output_json_path, dataset_name, class_names)
    loss_fn = StructureAwareClipLoss(
        output_json_path=args.output_json_path, # Đường dẫn thư mục gốc
        dataset_name=dataset_name,              # 'CUB200'
        label_to_classname=class_names,         # List tên class
        alpha=args.alpha,
        alpha_soft=args.alpha_soft,
        beta=args.beta,
        max_rank_diff=args.max_rank_diff,      # Từ argument
        device=device
    )
    # Chèn vào sau dòng: loss = loss_fn(scores, feat1, feat2, rank1, rank2, lbl1, lbl2
    #optimizer = optim.Adam(list(backbone.parameters()) + list(relation.parameters()), lr=args.lr)
    '''optimizer = optim.Adam(
    list(filter(lambda p: p.requires_grad, backbone.parameters())) +
    list(relation.parameters()),
    lr=args.lr
)'''

    # 4. Training Loop
    phase1_end = args.p1_epochs
    phase2_end = args.p1_epochs + args.p2_epochs
    phase3_epochs = args.epochs - phase2_end
    
    print("\n" + "="*50)
    print(">>> TRAINING PHASE DISTRIBUTION")
    print("="*50)
    print(f"📊 Phase 1 (Structure Learning):    Epoch 1-{phase1_end} ({phase1_end} epochs)")
    print(f"📊 Phase 2 (Discrimination):         Epoch {phase1_end+1}-{phase2_end} ({args.p2_epochs} epochs)")
    print(f"📊 Phase 3 (Regularization):         Epoch {phase2_end+1}-{args.epochs} ({phase3_epochs} epochs)")
    print("="*50)
    print(">>> START TRAINING 3-PHASE STRATEGY")
    print("="*50)
    
    # --- LOAD HARD NEGATIVES TỪ JSON (CHO PHASE 2-3) ---
    json_path = os.path.join(args.output_json_path, dataset_name)
    hard_sim_map = load_hard_negatives_from_json(json_path, train_set)
    print(f"🗺️ Hard Negative similarity map loaded: {len(hard_sim_map)} classes have neighbors")
    
    total_start_time = time.time()

    best_loss = float('inf')
    epochs_no_improve = 0
    patience = args.patience
    min_delta = args.min_delta
    sim_matrix = load_hard_negatives_from_json(json_path, train_set)
    for epoch in range(args.epochs):
        epoch_start = time.time()
        backbone.train()
        relation.train()
        score_combiner.train()  # ← Set score_combiner to training mode
        '''
        # --- LOGIC CHUYỂN ĐỔI PHASE ---
        if epoch < phase1_end:
            phase_name = "Phase 1: Structure Learning (Same Class Only)"
            sampler = ClassSpecificBatchSampler(train_labels, args.batch_size)
            loader = DataLoader(train_set, batch_sampler=sampler)
        elif epoch < phase2_end:
            phase_name = "Phase 2: Discrimination (Mixed Batch + CLIP)"
            loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        else:
            phase_name = "Phase 3: Regularization (Full Shuffle)"
            loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
            '''
        # --- LOGIC CHUYỂN ĐỔI PHASE (ĐÃ CẬP NHẬT SAMPLER) ---
        if epoch < phase1_end:
            phase_name = "Phase 1: Structure Learning (Same Class Only)"
            sampler = ClassSpecificBatchSampler(train_labels, args.batch_size)
            loader = DataLoader(train_set, batch_sampler=sampler)
        elif epoch < phase2_end:
            # Phase 2: Discrimination (25% same class + 75% random negatives)
            phase_name = "Phase 2: Discrimination (65% Same Class)"
            balanced_sampler_p2 = HybridHardRelationSampler(
                train_set,
                batch_size=args.batch_size,
                pos_fraction=0.65,        # 25% Cùng loài
                hard_neg_fraction=0.7,    # 70% Hard Negative (từ JSON)
                sim_matrix=sim_matrix,
                num_batches=200           # <--- ÉP CỨNG 200 BATCH
                )
            #loader = DataLoader(train_set, batch_sampler=balanced_sampler_p2, num_workers=4, pin_memory=True)
            loader = DataLoader(train_set, batch_sampler=balanced_sampler_p2, num_workers=4, pin_memory=True)
        else:
            # Phase 3: Regularization (15% same class + 85% random negatives)
            phase_name = "Phase 3: Regularization (50% Same Class)"
            balanced_sampler_p3 = HybridHardRelationSampler(
                 train_set,
                    batch_size=args.batch_size,
                    pos_fraction=0.5,        # 15% Cùng loài
                    hard_neg_fraction=0.5,    # Giảm Hard Negative xuống còn 30%
                    sim_matrix=sim_matrix,
                    num_batches=200           # <--- ÉP CỨNG 200 BATCH
                    )
            loader =DataLoader(train_set, batch_sampler=balanced_sampler_p3, num_workers=4, pin_memory=True)
        print(f"\nEpoch {epoch+1}/{args.epochs} | {phase_name}")
        print(f"   📦 DataLoader có {len(loader)} batches")
        sys.stdout.flush()
        
        epoch_loss = 0
        batch_count = 0
        
        # Pre-define rank lookup function outside loop for efficiency
        def _get_rank_for_index(i):
            try:
                if hasattr(train_set, 'samples') and len(train_set.samples) > int(i):
                    sample_path = train_set.samples[int(i)][0]
                    b = os.path.basename(sample_path)
                    if b in basename_rank_map:
                        return float(basename_rank_map[b])
            except Exception:
                pass
            # fallback to 0
            return 0.0
        
        for batch_idx, batch in enumerate(loader):
            # Support dataset that returns (img, label) or (img, label, idx)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                imgs, lbls, idxs = batch
            else:
                imgs, lbls = batch
                # Fallback: create dummy indices (ranks lookup will return default 0)
                idxs = torch.arange(imgs.size(0), device=device)

            imgs, lbls = imgs.to(device), lbls.to(device)
            idxs = idxs.to(device)
            if imgs.size(0) < 2: continue

            # Split Batch
            curr_bs = imgs.size(0)
            if curr_bs % 2 != 0:
                imgs = imgs[:-1]; lbls = lbls[:-1]; idxs = idxs[:-1]; curr_bs -= 1

# Lấy các ảnh ở vị trí chẵn (0, 2, 4...) làm tập 1
            img1 = imgs[0::2] 
            lbl1 = lbls[0::2]
            idx1 = idxs[0::2]

# Lấy các ảnh ở vị trí lẻ (1, 3, 5...) làm tập 2
            img2 = imgs[1::2]
            lbl2 = lbls[1::2]
            idx2 = idxs[1::2]
            # Kiểm tra xem có phải đang so sánh ảnh với chính nó không
            diff = (img1[0] - img2[0]).abs().sum()
            if diff < 1e-5:
                print("🚨 CẢNH BÁO: img1 và img2 giống hệt nhau! Kiểm tra lại Sampler/Indexing.")
            feat1, global_feat1, combined_patch_feat1 = backbone(img1)
            feat2, global_feat2, combined_patch_feat2 = backbone(img2)
            scores1=relation(feat1, feat2)
            scores2 = relation(global_feat1, global_feat2)
            score3 = relation(combined_patch_feat1, combined_patch_feat2)
            scores = score_combiner(scores1, scores2, score3)  # ← Kết hợp 3 scores bằng neural network
            # Build rank tensors by looking up idx -> rank from sorted json (default 0)
            rank_vals1 = [_get_rank_for_index(i) for i in idx1]
            rank_vals2 = [_get_rank_for_index(i) for i in idx2]
            rank1 = torch.tensor(rank_vals1, dtype=torch.float32, device=device)
            rank2 = torch.tensor(rank_vals2, dtype=torch.float32, device=device)

            # Call loss with rank inputs
            loss = loss_fn(scores, feat1, feat2, rank1, rank2, lbl1, lbl2)
            '''# --- ĐOẠN CHÈN THÊM ĐỂ SOI CA KHÓ ---
            if loss.item() > 0.4:
                with torch.no_grad():
                    label_a = lbl1[-1].item()
                    label_b = lbl2[-1].item()
                    final_score = scores[-1].item()
                    print(f"   [!] Ca khó: Class {label_a} vs Class {label_b}")
                    print(f"       Score: {final_score:.4f} | Loss Batch: {loss.item():.4f}")
            # --- KẾT THÚC ĐOẠN CHÈN ---'''
            if batch_idx % 50 == 0:
                with torch.no_grad():
                    print(f"\n--- DEBUG BATCH {batch_idx} ---")
                    print(f"  > Score Mean: {scores.mean().item():.4f}")
                    print(f"  > Score Min/Max: {scores.min().item():.4f} / {scores.max().item():.4f}")
        
        # Kiểm tra xem loss_fn có thực sự nhận diện được cùng class không
                    is_same = (lbl1 == lbl2).float()
                    print(f"  > Số cặp cùng class nhận diện được: {is_same.sum().item()}/{len(lbl1)}")
        
        # Quan trọng: Kiểm tra target thực tế mà loss đang hướng tới
        # (Bạn có thể copy logic tính target từ hàm loss sang đây để in ra)
                    print(f"  > Loss hiện tại: {loss.item():.6f}")
            # Debug: print loss details on first batch of each phase
            if batch_idx == 0:
                print(f"   [DEBUG Batch 0] img shape: {img1.shape}, feat shape: {feat1.shape}, scores shape: {scores.shape}, loss type: {type(loss)}, loss value: {loss}")
                print(f"   [DEBUG] feat1 min/max: {feat1.min():.4f}/{feat1.max():.4f}, scores min/max: {scores.min():.4f}/{scores.max():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if batch_count % 5 == 0:
                print(f"   [Batch {batch_count}/{len(loader)}] Loss: {loss.item():.4f}", flush=True)
            sys.stdout.flush()
        
        avg_loss = epoch_loss / max(batch_count, 1)
        print(f"   >>> End Epoch {epoch+1} - Avg Loss: {avg_loss:.4f} - Time: {(time.time() - epoch_start):.1f}s")

        # Early stopping check (based on training loss)
        if avg_loss + min_delta < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            best_epoch = epoch + 1
            print(f"   ✓ Loss improved to {best_loss:.6f} (epoch {best_epoch})")
        else:
            epochs_no_improve += 1
            print(f"   ⚠️ No improvement for {epochs_no_improve}/{patience} epochs")

        if epochs_no_improve >= patience:
            print(f"⏹️ Early stopping: no improvement in {patience} epochs. Stopping training.")
            break


    # 5. Save Model
    print("\n" + "="*50)
    print("💾 Đang lưu model weights...")
    os.makedirs('weights', exist_ok=True)
    
    suffix = "_10class" if args.test_10_classes else "_full"
    torch.save(backbone.state_dict(), f'weights/backbone{suffix}.pth')
    torch.save(relation.state_dict(), f'weights/relation{suffix}.pth')
    torch.save(score_combiner.state_dict(), f'weights/score_combiner{suffix}.pth')  # ← Lưu ScoreCombinerNet
    
    print(f"✅ Đã lưu model tại thư mục weights/ (suffix: {suffix})")
    print(f"⏱️ Tổng thời gian train: {(time.time() - total_start_time)/60:.1f} phút")

if __name__ == "__main__":
    main()