import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torchvision.transforms as transforms

from config import (
    CHECKPOINTS_DIR, ISIC_2019_DIR, PAD_UFES_DIR, NUM_CLASSES, CLASSES,
    IMAGE_SIZE, IMG_MEAN, IMG_STD
)
from dataset import encode_metadata
from models import MultimodalSkinClassifier

# Page configuration
st.set_page_config(
    page_title="Multimodal Skin Lesion AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASS_NAMES = {
    'BCC': 'Basal Cell Carcinoma (BCC)',
    'SCC': 'Squamous Cell Carcinoma (SCC)',
    'MEL': 'Melanoma (MEL)',
    'ACK': 'Actinic Keratosis (ACK)',
    'SEK': 'Seborrheic Keratosis (SEK)',
    'NEV': 'Melanocytic Nevus (NEV)'
}

CLASS_RISK = {
    'BCC': ('Malignant', '🔴 High Risk - Carcinoma'),
    'SCC': ('Malignant', '🔴 High Risk - Carcinoma'),
    'MEL': ('Malignant', '🚨 Critical Risk - Malignant Melanoma'),
    'ACK': ('Pre-cancerous', '⚠️ Pre-cancerous Lesion'),
    'SEK': ('Benign', '🟢 Benign Keratosis'),
    'NEV': ('Benign', '🟢 Benign Mole / Nevus')
}

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_meta = {'age': 55.0, 'gender': 'male', 'region': 'face'}
    meta_in_dim = len(encode_metadata(dummy_meta, full_clinical=False))

    model = MultimodalSkinClassifier(
        meta_in_dim=meta_in_dim,
        num_classes=NUM_CLASSES,
        backbone="resnet50",
        fusion_type="attention",
        pretrained=False
    ).to(device)

    ckpt_path = os.path.join(CHECKPOINTS_DIR, "resnet50_attention_fold1.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict):
            ckpt = {k: v.float() if hasattr(v, 'is_floating_point') and v.is_floating_point() else v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
    model.eval()
    return model, device

@st.cache_data
def load_isic_metadata():
    csv_path = os.path.join(ISIC_2019_DIR, "isic_2019_metadata.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

# Transform pipeline
img_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
])

# Initialize Model
model, device = load_model()
isic_df = load_isic_metadata()

# Sidebar: Metadata Inputs
st.sidebar.title("📋 Patient Demographics")
st.sidebar.markdown("Configure patient clinical attributes:")

age = st.sidebar.slider("Patient Age", min_value=1, max_value=100, value=52, step=1)
gender = st.sidebar.selectbox("Biological Sex", options=["male", "female"], index=0)
region = st.sidebar.selectbox(
    "Anatomical Site",
    options=["face", "neck", "chest", "back", "abdomen", "arm", "hand", "leg", "foot", "scalp"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("🩺 Diagnostic Symptoms")
itch = st.sidebar.checkbox("Itching / Pruritus (itching)", value=False)
bleed = st.sidebar.checkbox("Bleeding Lesion (bleed)", value=False)
hurt = st.sidebar.checkbox("Pain / Tenderness (hurt)", value=False)
grew = st.sidebar.checkbox("Rapid Growth / Expansion (grew)", value=False)
elevation = st.sidebar.checkbox("Palpable Elevation (elevation)", value=False)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Model:** ResNet-50 + MLP Tower with Dual-Attention (Cross + Self Attention).")

# Main Page Header
st.title("🔬 Multimodal Skin Lesion Classifier")
st.markdown("""
Deep Learning diagnostic decision support system combining **dermoscopic / clinical imagery** with **patient demographic & symptom metadata** via **Dual-Attention Fusion**.
""")

# Input Selection Tabs
tab1, tab2 = st.tabs(["📁 ISIC 2019 Benchmark Gallery", "📤 Custom Photo Upload"])

selected_image = None
meta_dict = {
    'age': float(age),
    'gender': str(gender),
    'region': str(region),
    'itch': itch,
    'bleed': bleed,
    'hurt': hurt,
    'grew': grew,
    'elevation': elevation
}

with tab1:
    st.subheader("Select from ISIC 2019 Dermoscopic Test Set")
    if not isic_df.empty:
        diag_col = 'diagnostic' if 'diagnostic' in isic_df.columns else ('diagnosis' if 'diagnosis' in isic_df.columns else isic_df.columns[1])
        id_col = 'img_id' if 'img_id' in isic_df.columns else ('image_id' if 'image_id' in isic_df.columns else isic_df.columns[0])
        site_col = 'anatom_site_predominance' if 'anatom_site_predominance' in isic_df.columns else ('anatom_site_general' if 'anatom_site_general' in isic_df.columns else 'region')

        col_filter1, col_filter2 = st.columns([1, 2])
        with col_filter1:
            class_filter = st.selectbox("Filter by True Diagnosis:", ["All Classes"] + CLASSES)
        
        filtered_df = isic_df if class_filter == "All Classes" else isic_df[isic_df[diag_col] == class_filter]
        
        if not filtered_df.empty:
            sample_options = [f"{row[id_col]} - {row[diag_col]} (Age: {row.get('age', 'N/A')}, Site: {row.get(site_col, 'N/A')})" for _, row in filtered_df.iterrows()]
            selected_sample_str = st.selectbox("Choose a sample lesion:", sample_options)
            
            selected_file = selected_sample_str.split(" - ")[0]
            if not selected_file.endswith(('.jpg', '.png')):
                selected_file = f"{selected_file}.jpg"
            selected_row = isic_df[isic_df[id_col] == selected_sample_str.split(" - ")[0]].iloc[0]
            img_path = os.path.join(ISIC_2019_DIR, "images", selected_file)
            
            if os.path.exists(img_path):
                selected_image = Image.open(img_path).convert('RGB')
                col_img, col_info = st.columns([1, 2])
                with col_img:
                    st.image(selected_image, caption=f"Selected: {selected_file}", use_container_width=True)
                with col_info:
                    st.markdown(f"**Ground Truth Diagnosis:** `{selected_row.get(diag_col, 'N/A')}`")
                    st.markdown(f"**Age:** `{selected_row.get('age', 'N/A')}` | **Sex:** `{selected_row.get('sex', 'N/A')}`")
                    st.markdown(f"**Anatomical Site:** `{selected_row.get(site_col, 'N/A')}`")
    else:
        st.warning("ISIC 2019 metadata not found.")

with tab2:
    st.subheader("Upload Any Skin Photo or Take a Picture")
    uploaded_file = st.file_uploader("Upload Dermoscopy or Camera Image (PNG/JPG):", type=["jpg", "jpeg", "png"])
    camera_file = st.camera_input("Or take a live photo:")
    
    file_to_use = uploaded_file or camera_file
    if file_to_use is not None:
        selected_image = Image.open(file_to_use).convert('RGB')
        st.image(selected_image, caption="Uploaded Lesion Photo", width=300)

# Run Inference
st.markdown("---")
predict_btn = st.button("🚀 Run Multimodal AI Prediction", type="primary", use_container_width=True)

if selected_image is not None and (predict_btn or tab1):
    with st.spinner("Processing image and metadata through Dual-Tower Attention Network..."):
        # Preprocess Image
        img_tensor = img_transforms(selected_image).unsqueeze(0).to(device)
        
        # Preprocess Metadata
        meta_vec = encode_metadata(meta_dict, full_clinical=False)
        meta_tensor = torch.tensor(meta_vec, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(img_tensor, meta_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        top_idx = int(np.argmax(probs))
        top_class = CLASSES[top_idx]
        top_prob = probs[top_idx]
        risk_type, risk_desc = CLASS_RISK.get(top_class, ("Unknown", "Unknown"))
        
        st.markdown("## 📊 Diagnostic AI Results")
        
        # Top Prediction Card
        if risk_type == "Malignant":
            st.error(f"### Predicted Diagnosis: **{CLASS_NAMES.get(top_class, top_class)}**\n**Confidence:** {top_prob*100:.2f}%\n\n{risk_desc}")
        elif risk_type == "Pre-cancerous":
            st.warning(f"### Predicted Diagnosis: **{CLASS_NAMES.get(top_class, top_class)}**\n**Confidence:** {top_prob*100:.2f}%\n\n{risk_desc}")
        else:
            st.success(f"### Predicted Diagnosis: **{CLASS_NAMES.get(top_class, top_class)}**\n**Confidence:** {top_prob*100:.2f}%\n\n{risk_desc}")
        
        # Probability Breakdown
        st.markdown("### Class Probability Distribution")
        for cls_name, prob in zip(CLASSES, probs):
            col_label, col_bar, col_pct = st.columns([2, 5, 1])
            with col_label:
                st.markdown(f"**{cls_name}** ({CLASS_NAMES.get(cls_name, cls_name)})")
            with col_bar:
                st.progress(float(prob))
            with col_pct:
                st.markdown(f"`{prob*100:5.2f}%`")
                
        # Model architecture details
        with st.expander("🔍 Multimodal Dual-Attention Fusion Explanation"):
            st.markdown("""
            * **Vision Tower (ResNet-50):** Extracted 2048-d spatial visual features from lesion borders, asymmetry, and color variegation.
            * **Metadata MLP Tower:** Encoded 13-d demographic (age, sex, anatomical site) and symptom indicators into a 512-d clinical embedding.
            * **Intra-Modal Self-Attention:** Discovered internal relationships within visual textures and demographic risk factors independently.
            * **Inter-Modal Cross-Attention:** Actively conditioned visual features on patient metadata (e.g. higher attention given to irregular pigment patterns if patient is elderly with bleeding symptoms).
            """)
elif selected_image is None:
    st.info("👆 Please select an image from the gallery or upload your own to view predictions.")
