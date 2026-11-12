import os
import io
import torch
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

from config import (
    CHECKPOINTS_DIR, ISIC_2019_DIR, PAD_UFES_DIR, NUM_CLASSES, CLASSES,
    IMAGE_SIZE, IMG_MEAN, IMG_STD
)
import torchvision.transforms as transforms
from dataset import encode_metadata
from models import MultimodalSkinClassifier

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = os.path.join(CHECKPOINTS_DIR, "resnet50_attention_fold1.pt")

meta_dummy_row = {'age': 55.0, 'gender': 'male', 'region': 'face'}
meta_in_dim = len(encode_metadata(meta_dummy_row, full_clinical=False))

# Load Model
model = MultimodalSkinClassifier(
    meta_in_dim=meta_in_dim,
    num_classes=NUM_CLASSES,
    backbone="resnet50",
    fusion_type="attention",
    pretrained=False
).to(device)

if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        ckpt = {k: v.float() if hasattr(v, 'is_floating_point') and v.is_floating_point() else v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    print(f"[OK] Loaded trained weights from {checkpoint_path}")
else:
    print(f"[WARN] Checkpoint not found at {checkpoint_path}. Using initial weights.")

model.eval()

img_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
])

# Load ISIC 2019 metadata table for gallery
isic_csv_path = os.path.join(ISIC_2019_DIR, "isic_2019_metadata.csv")
isic_df = pd.read_csv(isic_csv_path) if os.path.exists(isic_csv_path) else pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/isic_samples', methods=['GET'])
def get_isic_samples():
    """
    Returns the list of cases from ISIC 2019 dataset with image IDs,
    actual ground truth diagnosis, age, sex, and anatomical location.
    """
    if isic_df.empty:
        return jsonify([])

    samples = []
    # Take diverse representative cases across all 6 classes
    for cls in CLASSES:
        sub = isic_df[isic_df['diagnostic'] == cls].head(5)
        for _, row in sub.iterrows():
            samples.append({
                'img_id': str(row['img_id']),
                'ground_truth': str(row['diagnostic']),
                'age': float(row.get('age', 55.0)),
                'sex': str(row.get('sex', 'unknown')),
                'site': str(row.get('anatom_site_predominance', 'other/unknown'))
            })
    return jsonify(samples)

@app.route('/get_sample_image/<filename>')
def get_sample_image(filename):
    isic_img_dir = os.path.join(ISIC_2019_DIR, "images")
    return send_from_directory(isic_img_dir, filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if an uploaded image file was sent OR a filename from the gallery
        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            image_bytes = file.read()
            pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        elif 'selected_image_id' in request.form and request.form['selected_image_id'] != '':
            img_id = request.form['selected_image_id']
            img_path = os.path.join(ISIC_2019_DIR, "images", img_id)
            if not os.path.exists(img_path):
                return jsonify({'error': f"Image {img_id} not found in ISIC 2019 dataset"}), 404
            pil_img = Image.open(img_path).convert('RGB')
        else:
            return jsonify({'error': 'Please upload an image or select a photo from the ISIC 2019 dataset.'}), 400

        img_tensor = img_transforms(pil_img).unsqueeze(0).to(device)

        # Tabular metadata parsing
        age = float(request.form.get('age', 55.0))
        gender = request.form.get('gender', 'unknown')
        region = request.form.get('region', 'other/unknown')
        itch = request.form.get('itch', 'false') == 'true'
        bleed = request.form.get('bleed', 'false') == 'true'
        changed = request.form.get('changed', 'false') == 'true'
        elevation = request.form.get('elevation', 'false') == 'true'

        meta_dict = {
            'age': age,
            'gender': gender,
            'region': region,
            'itch': itch,
            'bleed': bleed,
            'changed': changed,
            'elevation': elevation
        }

        meta_arr = encode_metadata(meta_dict, full_clinical=False)
        meta_tensor = torch.tensor(meta_arr, dtype=torch.float32).unsqueeze(0).to(device)

        # Multimodal inference
        with torch.no_grad():
            logits = model(img_tensor, meta_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        predicted_class = CLASSES[pred_idx]
        prob_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': float(probs[pred_idx]),
            'probabilities': prob_dict
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Multimodal Skin Cancer Classification Web App...")
    print("Navigate to: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
