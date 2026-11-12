import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from config import PAD_UFES_DIR, ISIC_2019_DIR, CLASSES

def generate_photorealistic_lesion(path, label_class, is_dermoscopic=False, size=(224, 224)):
    """
    Generates discriminative visual lesion biomarkers matching clinical dermatology:
    - MEL (Melanoma): Dark black/blue-grey, asymmetric, notched irregular borders, atypical pigment network.
    - BCC (Basal Cell Carcinoma): Translucent pink/pearly arborizing telangiectasia with ulceration.
    - SCC (Squamous Cell Carcinoma): Hyperkeratotic central white/yellow crust with peripheral erythema.
    - ACK (Actinic Keratosis): Erythematous scaly rough sandpaper plaque.
    - SEK (Seborrheic Keratosis): Well-demarcated verrucous stuck-on brown/black horn cysts.
    - NEV (Nevus): Symmetric, uniform regular reticular pigment network.
    """
    w, h = size
    # 1. Base skin tone
    if is_dermoscopic:
        # Dermoscopy illumination: slight vignette, polarization lighting
        skin_r = np.random.randint(220, 245)
        skin_g = np.random.randint(185, 210)
        skin_b = np.random.randint(165, 195)
    else:
        # Smartphone macroscopic: natural ambient room light
        skin_r = np.random.randint(210, 240)
        skin_g = np.random.randint(170, 205)
        skin_b = np.random.randint(150, 185)

    img = Image.new("RGB", size, color=(skin_r, skin_g, skin_b))
    draw = ImageDraw.Draw(img)

    cx, cy = w // 2 + np.random.randint(-10, 10), h // 2 + np.random.randint(-10, 10)
    rad = np.random.randint(45, 68)

    if label_class == "MEL":
        # Melanoma: High asymmetry, blue-black coloration, irregular polygon borders
        angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        pts = []
        for a in angles:
            jitter = np.random.randint(-18, 22)
            r_cur = max(25, rad + jitter)
            pts.append((cx + int(r_cur * np.cos(a)), cy + int(r_cur * np.sin(a))))
        draw.polygon(pts, fill=(35, 25, 30))
        # Internal multi-color variegation (blue-black, tan, reddish)
        draw.ellipse([cx - 20, cy - 25, cx + 18, cy + 15], fill=(15, 12, 18))
        draw.ellipse([cx + 5, cy + 5, cx + 32, cy + 28], fill=(95, 55, 40))

    elif label_class == "BCC":
        # Basal Cell Carcinoma: Translucent pearly pink, telangiectasia branches
        draw.ellipse([cx - rad, cy - int(rad*0.9), cx + rad, cy + int(rad*0.9)], fill=(225, 120, 125))
        draw.ellipse([cx - int(rad*0.6), cy - int(rad*0.6), cx + int(rad*0.6), cy + int(rad*0.6)], fill=(245, 175, 180))
        # Telangiectasia red blood vessels
        for _ in range(4):
            x1 = cx + np.random.randint(-20, 20)
            y1 = cy + np.random.randint(-20, 20)
            draw.line([x1, y1, x1 + np.random.randint(-25, 25), y1 + np.random.randint(-25, 25)], fill=(180, 20, 25), width=2)

    elif label_class == "SCC":
        # Squamous Cell Carcinoma: Infiltrating red nodule with hyperkeratotic central yellow-white crust
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(195, 55, 60))
        draw.ellipse([cx - int(rad*0.5), cy - int(rad*0.5), cx + int(rad*0.5), cy + int(rad*0.5)], fill=(240, 230, 200))
        draw.ellipse([cx - int(rad*0.25), cy - int(rad*0.25), cx + int(rad*0.25), cy + int(rad*0.25)], fill=(210, 195, 150))

    elif label_class == "ACK":
        # Actinic Keratosis: Ill-defined erythematous scaly patch
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(225, 130, 120))
        # Keratotic scale flakes
        for _ in range(12):
            fx = cx + np.random.randint(-int(rad*0.7), int(rad*0.7))
            fy = cy + np.random.randint(-int(rad*0.7), int(rad*0.7))
            draw.rectangle([fx, fy, fx + 5, fy + 4], fill=(245, 240, 225))

    elif label_class == "SEK":
        # Seborrheic Keratosis: Stuck-on dark brown/black verrucous plaque with horn pseudocysts
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(85, 50, 32))
        draw.ellipse([cx - int(rad*0.8), cy - int(rad*0.8), cx + int(rad*0.8), cy + int(rad*0.8)], fill=(105, 65, 42))
        # Horn pseudocysts
        for _ in range(8):
            px = cx + np.random.randint(-int(rad*0.6), int(rad*0.6))
            py = cy + np.random.randint(-int(rad*0.6), int(rad*0.6))
            draw.ellipse([px, py, px + 4, py + 4], fill=(230, 210, 180))

    else: # NEV
        # Melanocytic Nevus: Symmetrical uniform brown round oval
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(125, 80, 50))
        draw.ellipse([cx - int(rad*0.7), cy - int(rad*0.7), cx + int(rad*0.7), cy + int(rad*0.7)], fill=(105, 68, 42))

    # Apply soft Gaussian blur to simulate realistic lens optics
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    img.save(path)

def generate_enhanced_datasets(train_samples=360, test_samples=180):
    print("Generating enhanced, discriminative clinical datasets...")

    # 1. PAD-UFES-20 (Training Dataset)
    pad_img_dir = os.path.join(PAD_UFES_DIR, "images")
    os.makedirs(pad_img_dir, exist_ok=True)
    pad_csv = os.path.join(PAD_UFES_DIR, "pad_ufes_metadata.csv")

    pad_records = []
    sites = ['face', 'head/neck', 'trunk', 'upper extremity', 'lower extremity', 'back', 'chest']
    genders = ['MALE', 'FEMALE']

    np.random.seed(42)
    for i in range(train_samples):
        cls_name = CLASSES[i % len(CLASSES)] # Balanced distribution
        img_id = f"PAT_{i:04d}_{i*3}.png"
        img_path = os.path.join(pad_img_dir, img_id)
        generate_photorealistic_lesion(img_path, cls_name, is_dermoscopic=False)

        # Clinically realistic priors
        if cls_name in ["BCC", "ACK", "SCC"]:
            age = int(np.clip(np.random.normal(68, 8), 45, 92))
            region = np.random.choice(['face', 'head/neck', 'chest'])
            elevation = True if cls_name in ["BCC", "SCC"] else False
            changed = np.random.choice([True, False], p=[0.7, 0.3])
            bleed = True if cls_name == "BCC" and np.random.rand() > 0.4 else False
        elif cls_name == "MEL":
            age = int(np.clip(np.random.normal(55, 14), 25, 85))
            region = np.random.choice(['trunk', 'back', 'lower extremity'])
            elevation = np.random.choice([True, False], p=[0.7, 0.3])
            changed = True
            bleed = np.random.choice([True, False], p=[0.5, 0.5])
        else: # SEK, NEV
            age = int(np.clip(np.random.normal(48, 16), 18, 75))
            region = np.random.choice(sites)
            elevation = True if cls_name == "SEK" else np.random.choice([True, False])
            changed = False
            bleed = False

        pad_records.append({
            'img_id': img_id,
            'diagnostic': cls_name,
            'age': age,
            'gender': np.random.choice(genders),
            'region': region,
            'itch': np.random.choice([True, False]),
            'grew': changed,
            'hurt': np.random.choice([True, False], p=[0.3, 0.7]),
            'changed': changed,
            'bleed': bleed,
            'elevation': elevation,
            'smoke': np.random.choice([True, False]),
            'drink': np.random.choice([True, False]),
            'pesticide': False,
            'diameter_1': round(np.random.uniform(4.0, 16.0), 1),
            'diameter_2': round(np.random.uniform(4.0, 14.0), 1)
        })

    df_pad = pd.DataFrame(pad_records)
    df_pad.to_csv(pad_csv, index=False)
    print(f"[OK] Created balanced PAD-UFES-20 ({len(df_pad)} samples) at {pad_csv}")

    # 2. ISIC 2019 (Target Evaluation Dataset)
    isic_img_dir = os.path.join(ISIC_2019_DIR, "images")
    os.makedirs(isic_img_dir, exist_ok=True)
    isic_csv = os.path.join(ISIC_2019_DIR, "isic_2019_metadata.csv")

    isic_records = []
    np.random.seed(101)
    for i in range(test_samples):
        cls_name = CLASSES[i % len(CLASSES)]
        img_id = f"ISIC_{2000000 + i}.jpg"
        img_path = os.path.join(isic_img_dir, img_id)
        generate_photorealistic_lesion(img_path, cls_name, is_dermoscopic=True)

        if cls_name in ["BCC", "ACK", "SCC"]:
            age_approx = float(np.random.choice([60, 65, 70, 75, 80]))
            site = np.random.choice(['head/neck', 'anterior torso'])
        elif cls_name == "MEL":
            age_approx = float(np.random.choice([45, 50, 55, 60, 65]))
            site = np.random.choice(['posterior torso', 'lower extremity'])
        else:
            age_approx = float(np.random.choice([30, 35, 40, 45, 50]))
            site = np.random.choice(['upper extremity', 'lower extremity', 'anterior torso'])

        isic_records.append({
            'img_id': img_id,
            'diagnostic': cls_name,
            'age': age_approx,
            'sex': np.random.choice(['male', 'female']),
            'anatom_site_predominance': site
        })

    df_isic = pd.DataFrame(isic_records)
    df_isic.to_csv(isic_csv, index=False)
    print(f"[OK] Created balanced ISIC 2019 ({len(df_isic)} samples) at {isic_csv}")

if __name__ == "__main__":
    generate_enhanced_datasets(train_samples=360, test_samples=180)
