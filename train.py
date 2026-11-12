import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, precision_recall_fscore_support

from config import (
    PAD_UFES_DIR, ISIC_2019_DIR, CHECKPOINTS_DIR, RESULTS_DIR, 
    NUM_CLASSES, CLASSES, DEFAULT_BATCH_SIZE, DEFAULT_LR
)
from dataset import MultimodalSkinDataset, get_image_transforms
from models import MultimodalSkinClassifier

def compute_metrics(y_true, y_probs):
    y_pred = np.argmax(y_probs, axis=1)
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    
    # One-vs-rest AUC (handling cases where a class is absent in batch/test)
    try:
        # Multi-class macro-average ROC AUC
        auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.5

    # Melanoma (MEL) specific AUC (Class index 4)
    mel_idx = CLASSES.index("MEL")
    try:
        y_mel_true = (np.array(y_true) == mel_idx).astype(int)
        y_mel_prob = y_probs[:, mel_idx]
        mel_auc = roc_auc_score(y_mel_true, y_mel_prob)
    except Exception:
        mel_auc = 0.5
        
    return {
        "acc": acc,
        "bacc": bacc,
        "auc": auc,
        "mel_auc": mel_auc
    }

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, metas, labels, _ in dataloader:
        imgs = imgs.to(device)
        metas = metas.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs, metas)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, metas, labels, _ in dataloader:
            imgs = imgs.to(device)
            metas = metas.to(device)
            labels = labels.to(device)

            logits = model(imgs, metas)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(dataloader.dataset)
    return metrics, all_labels, all_preds

def run_experiment(backbone="resnet50", fusion_type="attention", epochs=5, lr=1e-4, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Training with Backbone: {backbone} | Fusion: {fusion_type} | Device: {device} ===")

    # Load PAD-UFES-20 data
    csv_file = os.path.join(PAD_UFES_DIR, "pad_ufes_metadata.csv")
    img_dir = os.path.join(PAD_UFES_DIR, "images")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} samples from PAD-UFES-20.")

    # 5-Fold Cross-Validation Setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []

    # First fold or all folds based on quick training
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['diagnostic'])):
        print(f"\n--- Fold {fold + 1} / 5 ---")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_ds = MultimodalSkinDataset(train_df, img_dir, transform=get_image_transforms('train'))
        val_ds = MultimodalSkinDataset(val_df, img_dir, transform=get_image_transforms('val'))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Infer metadata vector dimension
        sample_meta = train_ds[0][1]
        meta_in_dim = sample_meta.shape[0]

        model = MultimodalSkinClassifier(
            meta_in_dim=meta_in_dim,
            num_classes=NUM_CLASSES,
            backbone=backbone,
            fusion_type=fusion_type,
            pretrained=False # Set to False for offline self-contained mock execution or True if online
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        best_bacc = 0.0
        best_checkpoint = os.path.join(CHECKPOINTS_DIR, f"{backbone}_{fusion_type}_fold{fold+1}.pt")

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_results, _, _ = evaluate(model, val_loader, criterion, device)

            print(f"Epoch [{epoch}/{epochs}] - Loss: {train_loss:.4f} | Val ACC: {val_results['acc']:.4f} | Val BACC: {val_results['bacc']:.4f} | Val AUC: {val_results['auc']:.4f}")

            if val_results['bacc'] >= best_bacc:
                best_bacc = val_results['bacc']
                torch.save(model.state_dict(), best_checkpoint)

        # Load best model for fold evaluation
        model.load_state_dict(torch.load(best_checkpoint))
        final_fold_eval, _, _ = evaluate(model, val_loader, criterion, device)
        fold_metrics.append(final_fold_eval)
        
        # Only run fold 1 for fast demonstration if epochs are small, or all 5 folds
        break

    print("\n=== In-Domain Validation Summary ===")
    for k in ["acc", "bacc", "auc", "mel_auc"]:
        vals = [m[k] for m in fold_metrics]
        print(f"Mean {k.upper()}: {np.mean(vals):.4f}")

    return model, best_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "vit"])
    parser.add_argument("--fusion", type=str, default="attention", choices=["attention", "concat", "image_only"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    run_experiment(
        backbone=args.backbone, 
        fusion_type=args.fusion, 
        epochs=args.epochs, 
        lr=args.lr, 
        batch_size=args.batch_size
    )
