import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

from config import PAD_UFES_DIR, CHECKPOINTS_DIR, NUM_CLASSES, CLASSES, DEFAULT_BATCH_SIZE
from dataset import MultimodalSkinDataset, get_image_transforms
from models import MultimodalSkinClassifier
from train import compute_metrics, evaluate

def train_high_accuracy_model(epochs=12, lr=3e-4, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training High-Accuracy Multimodal Model on {device} ===")

    csv_file = os.path.join(PAD_UFES_DIR, "pad_ufes_metadata.csv")
    img_dir = os.path.join(PAD_UFES_DIR, "images")
    df = pd.read_csv(csv_file)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(df, df['diagnostic']))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_ds = MultimodalSkinDataset(train_df, img_dir, transform=get_image_transforms('train'))
    val_ds = MultimodalSkinDataset(val_df, img_dir, transform=get_image_transforms('val'))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    meta_in_dim = train_ds[0][1].shape[0]

    # Initialize model using ResNet50 + Dual Attention Fusion with ImageNet transfer initialization
    model = MultimodalSkinClassifier(
        meta_in_dim=meta_in_dim,
        num_classes=NUM_CLASSES,
        backbone="resnet50",
        fusion_type="attention",
        pretrained=True # Transfer Learning
    ).to(device)

    # Cross entropy with label smoothing for improved generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_checkpoint = os.path.join(CHECKPOINTS_DIR, "resnet50_attention_fold1.pt")
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, metas, labels, _ in train_loader:
            imgs = imgs.to(device)
            metas = metas.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs, metas)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss = total_loss / len(train_loader.dataset)

        val_results, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {train_loss:.4f} | Val ACC: {val_results['acc']:.4f} | Val BACC: {val_results['bacc']:.4f} | Val AUC: {val_results['auc']:.4f} | Val MEL-AUC: {val_results['mel_auc']:.4f}")

        if val_results['acc'] > best_acc:
            best_acc = val_results['acc']
            torch.save(model.state_dict(), best_checkpoint)

    print(f"\n[OK] Training Complete! Best Validation Accuracy: {best_acc:.4f}")
    print(f"[OK] High-accuracy checkpoint saved to {best_checkpoint}")
    return best_checkpoint

if __name__ == "__main__":
    train_high_accuracy_model(epochs=10, lr=2e-4, batch_size=16)
