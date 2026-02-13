"""
Hàm inference để so sánh 2 ảnh bất kỳ và lấy concept-based similarity score
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import os
import sys

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backbone.feature_extract import HybridResNetBackbone
from backbone.relation_net import BilinearRelationNet

class ImageSimilarityComparator:
    def __init__(self, backbone_path, relation_path, concept_embeddings_dir=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Khởi tạo comparator để so sánh 2 ảnh
        
        Args:
            backbone_path: Path tới checkpoint của HybridResNetBackbone
            relation_path: Path tới checkpoint của BilinearRelationNet
            concept_embeddings_dir: Path tới thư mục chứa CLIP embeddings (output_json/CUB200, v.v.)
            device: cuda hoặc cpu
        """
        self.device = device
        
        # Load models
        self.backbone = HybridResNetBackbone(output_dim=512).to(device)
        self.relation = BilinearRelationNet(input_dim=512, hidden_dim=256).to(device)
        
        # Load checkpoints
        if os.path.exists(backbone_path):
            self.backbone.load_state_dict(torch.load(backbone_path, map_location=device))
            print(f"✓ Loaded backbone from {backbone_path}")
        else:
            print(f"⚠️ Backbone checkpoint không tìm thấy: {backbone_path}")
        
        if os.path.exists(relation_path):
            self.relation.load_state_dict(torch.load(relation_path, map_location=device))
            print(f"✓ Loaded relation net from {relation_path}")
        else:
            print(f"⚠️ Relation net checkpoint không tìm thấy: {relation_path}")
        
        # Set to eval mode
        self.backbone.eval()
        self.relation.eval()
        
        # Load concept embeddings (nếu có)
        self.concept_embeddings = {}
        if concept_embeddings_dir and os.path.exists(concept_embeddings_dir):
            self._load_concept_embeddings(concept_embeddings_dir)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])
        ])
    
    def _load_concept_embeddings(self, concept_dir):
        """Load CLIP concept embeddings từ JSON files"""
        if not os.path.exists(concept_dir):
            return
        
        try:
            for filename in os.listdir(concept_dir):
                if filename.endswith('_final.json'):
                    concept_name = filename.replace('_final.json', '')
                    json_path = os.path.join(concept_dir, filename)
                    with open(json_path, 'r') as f:
                        emb = torch.tensor(json.load(f), dtype=torch.float32, device=self.device)
                        emb = F.normalize(emb, p=2, dim=0)
                        self.concept_embeddings[concept_name] = emb
            
            print(f"✓ Loaded {len(self.concept_embeddings)} concept embeddings")
        except Exception as e:
            print(f"⚠️ Lỗi load concept embeddings: {e}")
    
    def load_image(self, image_path):
        """Load và preprocess ảnh"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            return img_tensor
        except Exception as e:
            print(f"❌ Lỗi load ảnh {image_path}: {e}")
            return None
    
    def extract_features(self, image_path_or_tensor):
        """Extract feature từ 1 ảnh"""
        with torch.no_grad():
            # Load ảnh nếu là path
            if isinstance(image_path_or_tensor, str):
                img_tensor = self.load_image(image_path_or_tensor)
                if img_tensor is None:
                    return None
            else:
                img_tensor = image_path_or_tensor
            
            # Extract features
            features = self.backbone.forward_single(img_tensor)  # [1, 512]
            features = F.normalize(features, p=2, dim=1)  # Normalize
            
            return features.squeeze(0)  # [512]
    
    def compare_images(self, image1_path, image2_path, 
                      use_visual_only=False, 
                      verbose=True):
        """
        So sánh 2 ảnh và lấy similarity score
        
        Args:
            image1_path: Path ảnh 1 (hoặc tensor)
            image2_path: Path ảnh 2 (hoặc tensor)
            use_visual_only: Nếu True, chỉ dùng visual similarity; 
                           Nếu False, kết hợp visual + concept
            verbose: In chi tiết scores
        
        Returns:
            dict với keys:
            - visual_score: Score từ BilinearRelationNet (0-1)
            - concept_score: Score từ concept embeddings (nếu có)
            - final_score: Score kết hợp (0-1)
        """
        # Extract features
        feat1 = self.extract_features(image1_path)
        feat2 = self.extract_features(image2_path)
        
        if feat1 is None or feat2 is None:
            return None
        
        # Visual similarity từ relation net
        with torch.no_grad():
            feat1_batch = feat1.unsqueeze(0)
            feat2_batch = feat2.unsqueeze(0)
            visual_score = self.relation(feat1_batch, feat2_batch).item()
            visual_score = max(0.0, min(1.0, visual_score))  # Clamp to [0, 1]
        
        # Concept similarity (nếu có embeddings)
        concept_score = None
        if not use_visual_only and len(self.concept_embeddings) > 0:
            concept_score = self._compute_concept_similarity(feat1, feat2)
        
        # Final score
        if concept_score is not None:
            # Kết hợp: 70% visual + 30% concept
            final_score = 0.7 * visual_score + 0.3 * concept_score
        else:
            final_score = visual_score
        
        result = {
            'visual_score': visual_score,
            'concept_score': concept_score,
            'final_score': final_score
        }
        
        if verbose:
            self._print_comparison_result(result)
        
        return result
    
    def _compute_concept_similarity(self, feat1, feat2):
        """
        Tính concept similarity bằng cách so sánh features với concept embeddings
        """
        if len(self.concept_embeddings) == 0:
            return None
        
        # Tính similarity với tất cả concept embeddings
        max_sim1 = 0.0
        max_sim2 = 0.0
        
        for concept_emb in self.concept_embeddings.values():
            sim1 = F.cosine_similarity(feat1.unsqueeze(0), concept_emb.unsqueeze(0)).item()
            sim2 = F.cosine_similarity(feat2.unsqueeze(0), concept_emb.unsqueeze(0)).item()
            
            max_sim1 = max(max_sim1, sim1)
            max_sim2 = max(max_sim2, sim2)
        
        # Concept score: nếu cả 2 ảnh gần tới cùng 1 concept → score cao
        concept_score = (max_sim1 + max_sim2) / 2.0  # Average proximity
        concept_score = (concept_score + 1.0) / 2.0  # Scale to [0, 1] từ [-1, 1]
        
        return concept_score
    
    def _print_comparison_result(self, result):
        """Pretty print comparison results"""
        print("\n" + "="*60)
        print("📊 IMAGE COMPARISON RESULTS")
        print("="*60)
        print(f"Visual Score        : {result['visual_score']:.4f} (BilinearRelationNet)")
        if result['concept_score'] is not None:
            print(f"Concept Score       : {result['concept_score']:.4f} (CLIP Embeddings)")
        print(f"{'─'*60}")
        print(f"FINAL SCORE         : {result['final_score']:.4f}")
        print("="*60 + "\n")


def demo_comparison(image1_path, image2_path, 
                   backbone_path='./weights/backbone_full.pth',
                   relation_path='./weights/relation_full.pth',
                   concept_dir='./output_json/CUB200'):
    """
    Quick demo function
    """
    print("🚀 Initializing comparator...")
    comparator = ImageSimilarityComparator(
        backbone_path=backbone_path,
        relation_path=relation_path,
        concept_embeddings_dir=concept_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\n📸 Comparing images:")
    print(f"   Image 1: {image1_path}")
    print(f"   Image 2: {image2_path}")
    
    result = comparator.compare_images(
        image1_path, 
        image2_path,
        use_visual_only=False,
        verbose=True
    )
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare 2 images using trained model")
    parser.add_argument('image1', type=str, help='Path to first image')
    parser.add_argument('image2', type=str, help='Path to second image')
    parser.add_argument('--backbone', type=str, default='./weights/backbone_full.pth',
                       help='Path to backbone checkpoint')
    parser.add_argument('--relation', type=str, default='./weights/relation_full.pth',
                       help='Path to relation net checkpoint')
    parser.add_argument('--concept-dir', type=str, default='./output_json/CUB200',
                       help='Path to concept embeddings directory')
    parser.add_argument('--visual-only', action='store_true',
                       help='Use only visual similarity, ignore concepts')
    
    args = parser.parse_args()
    
    demo_comparison(
        image1_path=args.image1,
        image2_path=args.image2,
        backbone_path=args.backbone,
        relation_path=args.relation,
        concept_dir=args.concept_dir
    )
