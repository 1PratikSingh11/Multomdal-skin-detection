import torch
from models import MultimodalSkinClassifier
from config import NUM_CLASSES

def run_smoke_test():
    print("Running Multimodal Architecture Smoke Tests...")
    batch_size = 4
    sample_img = torch.randn(batch_size, 3, 224, 224)
    meta_dim = 13
    sample_meta = torch.randn(batch_size, meta_dim)

    # 1. Test ResNet50 + Cross Attention Fusion (Ou et al.)
    print("Testing Model 1: ResNet50 + Inter/Intra-Attention Fusion...")
    model_attn = MultimodalSkinClassifier(meta_in_dim=meta_dim, num_classes=NUM_CLASSES, backbone="resnet50", fusion_type="attention", pretrained=False)
    out_attn = model_attn(sample_img, sample_meta)
    assert out_attn.shape == (batch_size, NUM_CLASSES), f"Shape mismatch: {out_attn.shape}"
    print(f"-> Success! Output shape: {out_attn.shape}")

    # 2. Test ResNet50 + Concat Late Fusion (Yap et al.)
    print("Testing Model 2: ResNet50 + Concat Late Fusion...")
    model_concat = MultimodalSkinClassifier(meta_in_dim=meta_dim, num_classes=NUM_CLASSES, backbone="resnet50", fusion_type="concat", pretrained=False)
    out_concat = model_concat(sample_img, sample_meta)
    assert out_concat.shape == (batch_size, NUM_CLASSES), f"Shape mismatch: {out_concat.shape}"
    print(f"-> Success! Output shape: {out_concat.shape}")

    # 3. Test ViT + Transformer Fusion
    print("Testing Model 3: ViT + Attention Fusion...")
    model_vit = MultimodalSkinClassifier(meta_in_dim=meta_dim, num_classes=NUM_CLASSES, backbone="vit", fusion_type="attention", pretrained=False)
    out_vit = model_vit(sample_img, sample_meta)
    assert out_vit.shape == (batch_size, NUM_CLASSES), f"Shape mismatch: {out_vit.shape}"
    print(f"-> Success! Output shape: {out_vit.shape}")

    # 4. Test Image Only Baseline
    print("Testing Model 4: Image Only Baseline...")
    model_img = MultimodalSkinClassifier(meta_in_dim=meta_dim, num_classes=NUM_CLASSES, backbone="resnet50", fusion_type="image_only", pretrained=False)
    out_img = model_img(sample_img)
    assert out_img.shape == (batch_size, NUM_CLASSES), f"Shape mismatch: {out_img.shape}"
    print(f"-> Success! Output shape: {out_img.shape}")

    print("\n[ALL TESTS PASSED SUCCESSFULLY!]")

if __name__ == "__main__":
    run_smoke_test()
