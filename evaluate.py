import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    ISIC_2019_DIR, PAD_UFES_DIR, CHECKPOINTS_DIR, RESULTS_DIR, 
    NUM_CLASSES, CLASSES, DEFAULT_BATCH_SIZE
)
from dataset import MultimodalSkinDataset, get_image_transforms
from models import MultimodalSkinClassifier
from train import compute_metrics

def plot_confusion_matrix(y_true, y_pred, save_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(title, fontsize=13)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("Ground Truth Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Saved Confusion Matrix to {save_path}")

def evaluate_cross_dataset(checkpoint_path, backbone="resnet50", fusion_type="attention", test_dataset="isic_2019"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================================")
    print(f" CROSS-DATASET GENERALIZATION EVALUATION ON {test_dataset.upper()}")
    print(f" Model checkpoint: {checkpoint_path}")
    print(f" Backbone: {backbone} | Fusion: {fusion_type}")
    print(f"========================================================")

    # 1. Load Target Dataset (ISIC 2019 or PAD-UFES-20 test split)
    if test_dataset == "isic_2019":
        csv_file = os.path.join(ISIC_2019_DIR, "isic_2019_metadata.csv")
        img_dir = os.path.join(ISIC_2019_DIR, "images")
    else:
        csv_file = os.path.join(PAD_UFES_DIR, "pad_ufes_metadata.csv")
        img_dir = os.path.join(PAD_UFES_DIR, "images")

    df = pd.read_csv(csv_file)
    print(f"Evaluating on {len(df)} samples...")

    test_ds = MultimodalSkinDataset(df, img_dir, transform=get_image_transforms('val'), full_clinical=False)
    test_loader = DataLoader(test_ds, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)

    sample_meta = test_ds[0][1]
    meta_in_dim = sample_meta.shape[0]

    # 2. Re-create Model
    model = MultimodalSkinClassifier(
        meta_in_dim=meta_in_dim,
        num_classes=NUM_CLASSES,
        backbone=backbone,
        fusion_type=fusion_type,
        pretrained=False
    ).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict):
            ckpt = {k: v.float() if hasattr(v, 'is_floating_point') and v.is_floating_point() else v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
        print(f"[OK] Successfully loaded weights from {checkpoint_path}")
    else:
        print(f"[WARN] Checkpoint {checkpoint_path} not found. Running with initialized weights.")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, metas, labels, _ in test_loader:
            imgs = imgs.to(device)
            metas = metas.to(device)
            
            logits = model(imgs, metas)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            all_preds.append(probs)
            all_labels.append(labels.numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)

    # 3. Compute Metrics
    metrics = compute_metrics(all_labels, all_preds)
    y_pred_classes = np.argmax(all_preds, axis=1)

    print("\n--- Quantitative Evaluation Results ---")
    print(f"Overall Accuracy (ACC):       {metrics['acc']:.4f}")
    print(f"Balanced Accuracy (BACC):     {metrics['bacc']:.4f}")
    print(f"Macro-Averaged ROC-AUC:       {metrics['auc']:.4f}")
    print(f"Melanoma (MEL) ROC-AUC:       {metrics['mel_auc']:.4f}")

    # Classification Report
    print("\n--- Detailed Per-Class Classification Report ---")
    report = classification_report(all_labels, y_pred_classes, target_names=CLASSES, zero_division=0)
    print(report)

    # Confusion Matrix Plot
    cm_path = os.path.join(RESULTS_DIR, f"cross_eval_{backbone}_{fusion_type}_{test_dataset}_cm.png")
    plot_confusion_matrix(all_labels, y_pred_classes, cm_path, 
                          title=f"Cross-Domain CM: {test_dataset.upper()} ({backbone.upper()} + {fusion_type.capitalize()})")

    # Save Results CSV
    res_df = pd.DataFrame([{
        "dataset": test_dataset,
        "backbone": backbone,
        "fusion": fusion_type,
        "acc": metrics['acc'],
        "bacc": metrics['bacc'],
        "auc": metrics['auc'],
        "mel_auc": metrics['mel_auc']
    }])
    summary_path = os.path.join(RESULTS_DIR, f"metrics_{test_dataset}_{backbone}_{fusion_type}.csv")
    res_df.to_csv(summary_path, index=False)
    print(f"[OK] Saved summary results to {summary_path}")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "vit"])
    parser.add_argument("--fusion", type=str, default="attention", choices=["attention", "concat", "image_only"])
    parser.add_argument("--dataset", type=str, default="isic_2019", choices=["isic_2019", "pad_ufes_20"])
    args = parser.parse_args()

    ckpt = args.checkpoint
    if not ckpt:
        ckpt = os.path.join(CHECKPOINTS_DIR, f"{args.backbone}_{args.fusion}_fold1.pt")

    evaluate_cross_dataset(ckpt, backbone=args.backbone, fusion_type=args.fusion, test_dataset=args.dataset)
