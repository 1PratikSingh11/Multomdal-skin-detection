import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# -------------------------------------------------------------
# 1. Vision Backbones
# -------------------------------------------------------------
class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-50 feature extractor as utilized in Yap et al. and Ou et al.
    Extracts 2048-dimensional deep feature representations.
    """
    def __init__(self, pretrained=True, feature_dim=2048):
        super().__init__()
        # Using weights default or None
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.feature_dim = feature_dim

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        return feat

class ViTFeatureExtractor(nn.Module):
    """
    Vision Transformer (ViT-B/16) backbone option for ViT + Transformer fusion architecture.
    """
    def __init__(self, pretrained=True, feature_dim=768):
        super().__init__()
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        vit = models.vit_b_16(weights=weights)
        self.conv_proj = vit.conv_proj
        self.class_token = vit.class_token
        self.encoder = vit.encoder
        self.heads = nn.Identity()
        self.feature_dim = feature_dim

    def forward(self, x):
        n = x.shape[0]
        x = self.conv_proj(x)
        x = x.reshape(n, self.feature_dim, -1).permute(0, 2, 1)
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        # return CLS token feature
        cls_feat = x[:, 0]
        return cls_feat

# -------------------------------------------------------------
# 2. Tabular/Metadata Multi-Layer Perceptron (MLP)
# -------------------------------------------------------------
class MetadataEncoder(nn.Module):
    """
    MLP encoder projecting heterogeneous patient metadata into embedding space.
    """
    def __init__(self, in_features, hidden_dim=256, out_dim=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------------------
# 3. Multimodal Attention Fusion (Ou et al. Frontiers in Surgery)
# -------------------------------------------------------------
class CrossAttentionBlock(nn.Module):
    """
    Inter-modality Cross-Attention:
    Computes Q from source modality, K from guiding modality, V from source modality.
    """
    def __init__(self, embed_dim=512, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, q_src, kv_guide):
        # inputs are (batch_size, embed_dim) -> unsqueeze to (batch_size, 1, embed_dim)
        q = q_src.unsqueeze(1)
        k = kv_guide.unsqueeze(1)
        v = q_src.unsqueeze(1) # As per Ou et al. paper formulation
        attn_out, _ = self.mha(query=q, key=k, value=v)
        out = self.norm(q + attn_out).squeeze(1)
        return out

class AttentionFusionModule(nn.Module):
    """
    Dual-way Intra-modality Self-Attention & Inter-modality Cross-Attention
    From Ou et al. (Frontiers in Surgery 2022).
    """
    def __init__(self, img_dim=2048, meta_dim=512, proj_dim=512, num_heads=4):
        super().__init__()
        # Linear projection to unified dimension
        self.img_proj = nn.Linear(img_dim, proj_dim)
        self.meta_proj = nn.Linear(meta_dim, proj_dim)

        # 1. Intra-modality self-attention
        self.img_self_attn = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, batch_first=True)
        self.meta_self_attn = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, batch_first=True)
        self.norm_img = nn.LayerNorm(proj_dim)
        self.norm_meta = nn.LayerNorm(proj_dim)

        # 2. Inter-modality cross-attention
        self.cross_img = CrossAttentionBlock(embed_dim=proj_dim, num_heads=num_heads)
        self.cross_meta = CrossAttentionBlock(embed_dim=proj_dim, num_heads=num_heads)

        # Final fused representation dimension
        self.fused_dim = proj_dim * 2

    def forward(self, f_img, f_meta):
        p_img = self.img_proj(f_img).unsqueeze(1)
        p_meta = self.meta_proj(f_meta).unsqueeze(1)

        # Intra-modality self-attention
        sa_img, _ = self.img_self_attn(p_img, p_img, p_img)
        sa_img = self.norm_img(p_img + sa_img).squeeze(1)

        sa_meta, _ = self.meta_self_attn(p_meta, p_meta, p_meta)
        sa_meta = self.norm_meta(p_meta + sa_meta).squeeze(1)

        # Inter-modality cross-attention
        # Metadata guides image, and image guides metadata
        cross_img_out = self.cross_img(sa_img, sa_meta)
        cross_meta_out = self.cross_meta(sa_meta, sa_img)

        # Final concatenation
        fused = torch.cat([cross_img_out, cross_meta_out], dim=1)
        return fused

# -------------------------------------------------------------
# 4. Late Fusion Baseline (Yap et al. Exp Dermatol 2018)
# -------------------------------------------------------------
class LateConcatFusion(nn.Module):
    """
    Concatenates visual features + metadata features followed by an embedding MLP.
    As proposed by Yap et al. (Experimental Dermatology 2018).
    """
    def __init__(self, img_dim=2048, meta_dim=512, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim + meta_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.fused_dim = hidden_dim

    def forward(self, f_img, f_meta):
        combined = torch.cat([f_img, f_meta], dim=1)
        return self.net(combined)

# -------------------------------------------------------------
# 5. Complete Multimodal Classification Network
# -------------------------------------------------------------
class MultimodalSkinClassifier(nn.Module):
    def __init__(self, 
                 meta_in_dim=13, 
                 num_classes=6, 
                 backbone="resnet50", 
                 fusion_type="attention", 
                 pretrained=True):
        super().__init__()
        self.fusion_type = fusion_type
        
        # 1. Vision Tower
        if backbone == "resnet50":
            self.vision_tower = ResNetFeatureExtractor(pretrained=pretrained)
            img_dim = 2048
        elif backbone == "vit":
            self.vision_tower = ViTFeatureExtractor(pretrained=pretrained)
            img_dim = 768
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # 2. Metadata Tower
        self.meta_tower = MetadataEncoder(in_features=meta_in_dim, hidden_dim=256, out_dim=512)
        meta_dim = 512

        # 3. Multimodal Fusion Module
        if fusion_type == "attention":
            self.fusion_module = AttentionFusionModule(img_dim=img_dim, meta_dim=meta_dim, proj_dim=512)
            fused_dim = self.fusion_module.fused_dim
        elif fusion_type == "concat":
            self.fusion_module = LateConcatFusion(img_dim=img_dim, meta_dim=meta_dim, hidden_dim=1024)
            fused_dim = self.fusion_module.fused_dim
        elif fusion_type == "image_only":
            self.fusion_module = None
            fused_dim = img_dim
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # 4. Disease Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fused_dim, num_classes)
        )

    def forward(self, img, meta=None):
        f_img = self.vision_tower(img)
        
        if self.fusion_type == "image_only" or meta is None:
            logits = self.classifier(f_img)
            return logits

        f_meta = self.meta_tower(meta)
        f_fused = self.fusion_module(f_img, f_meta)
        logits = self.classifier(f_fused)
        return logits
