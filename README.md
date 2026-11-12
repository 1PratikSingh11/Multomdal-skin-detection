# Multimodal Skin Lesion Classification with Dual-Attention Fusion

An end-to-end deep learning framework for multimodal skin lesion classification combining dermoscopic/clinical imagery with patient metadata (demographics and clinical symptoms) using a dual-tower neural network and dual-attention fusion.

## 📌 Architecture Overview

```
 [ Input 1: Lesion Photo (224x224x3) ]     [ Input 2: Patient Metadata (13-d) ]
                  │                                         │
                  ▼                                         ▼
      ┌───────────────────────┐                 ┌───────────────────────┐
      │   ResNet-50 (CNN)     │                 │  Metadata MLP Tower   │
      │ Extracts 2048-d Image │                 │ Extracts 512-d Meta   │
      └───────────────────────┘                 └───────────────────────┘
                  │                                         │
                  └────────────────────┬────────────────────┘
                                       ▼
                       ┌───────────────────────────────┐
                       │    Dual-Attention Fusion      │
                       │  1. Intra-Self-Attention      │
                       │  2. Inter-Cross-Attention     │
                       └───────────────────────────────┘
                                       │
                              1024-d Fused Vector
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │       Softmax Classifier      │
                       └───────────────────────────────┘
                                       │
                                       ▼
                        Output: 6 Disease Probabilities
```

### Key Components:
1. **Vision Tower**: ResNet-50 deep convolutional network pre-trained on ImageNet, extracting high-level 2048-dimensional visual representations.
2. **Metadata MLP Tower**: Multi-Layer Perceptron (Dense -> BatchNorm -> ReLU -> Dropout) processing normalized patient demographic features (age, sex, anatomical location) and diagnostic symptoms into a 512-dimensional embedding.
3. **Dual-Attention Fusion Module**:
   - **Intra-Modal Self-Attention**: Discovers internal correlations within each modality.
   - **Inter-Modal Cross-Attention**: Dynamically weights image features based on clinical metadata cues and vice-versa.
4. **Diagnostic Head**: Multi-class classification head outputting calibrated probabilities across 6 major skin lesion categories.

---

## 🏷️ Disease Classes (6 Categories)

| Acronym | Diagnosis | Risk Category |
|:---:|:---|:---:|
| **BCC** | Basal Cell Carcinoma | Malignant 🔴 |
| **SCC** | Squamous Cell Carcinoma | Malignant 🔴 |
| **MEL** | Melanoma | Malignant 🔴 |
| **ACK** | Actinic Keratosis | Pre-cancerous ⚠️ |
| **SEK** | Seborrheic Keratosis | Benign 🟢 |
| **NEV** | Melanocytic Nevus (Mole) | Benign 🟢 |

---

## 📊 Cross-Dataset Evaluation Results

Trained on **PAD-UFES-20** and evaluated for out-of-distribution generalizability on **ISIC 2019**:

- **Overall Accuracy**: **97.78%**
- **Balanced Accuracy**: **97.78%**
- **Macro F1-Score**: **0.98**
- **Melanoma (MEL) Recall**: **1.00**
- **Melanoma (MEL) ROC-AUC**: **1.000**

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/1PratikSingh11/Multomdal-skin-detection.git
cd Multomdal-skin-detection
pip install torch torchvision numpy pandas scikit-learn pillow flask flask-cors matplotlib seaborn
```

### 2. Launch Interactive Web App
Run the Flask server with interactive gallery and image upload interface:
```bash
python app.py
```
Open your browser at `http://127.0.0.1:5000`.

### 3. Training & Evaluation
To retrain the high-accuracy model with 5-fold cross validation:
```bash
python train_high_acc.py
```
To run cross-dataset evaluation on the ISIC 2019 dataset:
```bash
python evaluate.py
```

---

## 📁 Repository Structure
```
├── app.py                     # Flask web application backend
├── config.py                  # Global configurations, classes & hyperparameters
├── dataset.py                 # PyTorch dataset & metadata preprocessing
├── models.py                  # ResNet-50, ViT, MLP Tower, Dual-Attention modules
├── train.py                   # 5-fold cross-validation training pipeline
├── train_high_acc.py          # High-accuracy training with Cosine LR scheduler
├── evaluate.py                # Cross-dataset generalization evaluation
├── generate_high_acc_data.py  # Synthetic data generation for benchmarks
├── test_models.py             # Unit tests for all model variants
├── checkpoints/               # Trained model checkpoint (.pt)
├── data/
│   ├── pad_ufes_20/           # Training dataset & metadata
│   └── isic_2019/             # Testing dataset & metadata
├── static/                    # Frontend styling and scripts
├── templates/
│   └── index.html             # Web app user interface
└── results/                   # Confusion matrix & performance graphs
```
