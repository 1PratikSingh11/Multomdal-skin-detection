import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from config import PAD_UFES_DIR, ISIC_2019_DIR, CLASSES

def generate_synthetic_lesion_image(path, label_class, size=(224, 224)):
    """
    Generates a realistic-looking synthetic skin lesion photograph
    with pigmentation, border characteristics, and skin tone variations.
    """
    # Background skin color (randomized Fitzpatrick tones)
    base_tones = [
        (240, 215, 195), # Light
        (225, 190, 165), # Intermediate
        (190, 150, 120), # Olive
        (140, 100, 75)   # Brown
    ]
    skin_color = base_tones[np.random.randint(0, len(base_tones))]
    img = Image.new("RGB", size, color=skin_color)
    draw = ImageDraw.Draw(img)

    # Class-specific lesion styling
    w, h = size
    center_x, center_y = w // 2 + np.random.randint(-15, 15), h // 2 + np.random.randint(-15, 15)
    radius = np.random.randint(35, 65)

    if label_class == "MEL": # Melanoma: Dark, irregular border, multi-color
        lesion_color = (40 + np.random.randint(0, 20), 20 + np.random.randint(0, 20), 20 + np.random.randint(0, 20))
        # Draw irregular shape
        points = []
        for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            r = radius + np.random.randint(-12, 18)
            px = center_x + int(r * np.cos(angle))
            py = center_y + int(r * np.sin(angle))
            points.append((px, py))
        draw.polygon(points, fill=lesion_color)
        # Inner dark core
        draw.ellipse([center_x - 15, center_y - 15, center_x + 15, center_y + 15], fill=(20, 10, 10))

    elif label_class == "BCC": # Basal Cell Carcinoma: Translucent, pearly, telangiectasia/pinkish
        lesion_color = (210, 110, 110)
        draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius], fill=lesion_color)
        draw.ellipse([center_x - radius//2, center_y - radius//2, center_x + radius//2, center_y + radius//2], fill=(235, 150, 150))

    elif label_class == "SCC": # Squamous Cell Carcinoma: Erythematous, hyperkeratotic, crusting
        lesion_color = (180, 70, 70)
        draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius], fill=lesion_color)
        # Central keratotic crust
        draw.ellipse([center_x - radius//3, center_y - radius//3, center_x + radius//3, center_y + radius//3], fill=(230, 220, 190))

    elif label_class == "ACK": # Actinic Keratosis: Rough, scaly, reddish-pink
        lesion_color = (220, 140, 130)
        draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius], fill=lesion_color)

    elif label_class == "SEK": # Seborrheic Keratosis: Stuck-on, brown/black, verrucous
        lesion_color = (90, 60, 40)
        draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius], fill=lesion_color)

    else: # NEV: Nevus - regular, uniform pigment brown
        lesion_color = (110, 75, 50)
        draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius], fill=lesion_color)

    img.save(path)

def setup_mock_pad_ufes(num_samples=120):
    """
    Creates mock PAD-UFES-20 dataset (smartphone clinical images + 21 metadata features)
    for running immediate verification and baseline reproduction.
    """
    img_dir = os.path.join(PAD_UFES_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)
    csv_path = os.path.join(PAD_UFES_DIR, "pad_ufes_metadata.csv")

    records = []
    sites = ['face', 'head/neck', 'trunk', 'upper extremity', 'lower extremity', 'back', 'chest']
    genders = ['MALE', 'FEMALE']

    np.random.seed(42)
    for i in range(num_samples):
        img_id = f"PAT_{i:04d}_{i*3}.png"
        img_path = os.path.join(img_dir, img_id)
        
        # Clinical distribution
        cls_choice = np.random.choice(CLASSES, p=[0.30, 0.15, 0.20, 0.15, 0.08, 0.12])
        generate_synthetic_lesion_image(img_path, cls_choice)

        age = int(np.clip(np.random.normal(58, 16), 18, 90))
        gender = np.random.choice(genders)
        region = np.random.choice(sites)

        # Symptoms
        itch = np.random.choice([True, False], p=[0.4, 0.6])
        grew = np.random.choice([True, False], p=[0.5, 0.5])
        hurt = np.random.choice([True, False], p=[0.25, 0.75])
        changed = np.random.choice([True, False], p=[0.45, 0.55])
        bleed = np.random.choice([True, False], p=[0.3, 0.7])
        elevation = np.random.choice([True, False], p=[0.6, 0.4])

        records.append({
            'img_id': img_id,
            'diagnostic': cls_choice,
            'age': age,
            'gender': gender,
            'region': region,
            'itch': itch,
            'grew': grew,
            'hurt': hurt,
            'changed': changed,
            'bleed': bleed,
            'elevation': elevation,
            'smoke': np.random.choice([True, False]),
            'drink': np.random.choice([True, False]),
            'pesticide': np.random.choice([True, False]),
            'diameter_1': round(np.random.uniform(4.0, 18.0), 1),
            'diameter_2': round(np.random.uniform(3.0, 15.0), 1)
        })

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"[OK] Generated mock PAD-UFES-20 dataset with {len(df)} samples at {csv_path}")
    return df

def setup_mock_isic_2019(num_samples=60):
    """
    Creates mock ISIC 2019 dataset (dermoscopic images + demographic metadata)
    for out-of-distribution cross-dataset evaluation.
    """
    img_dir = os.path.join(ISIC_2019_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)
    meta_path = os.path.join(ISIC_2019_DIR, "isic_2019_metadata.csv")

    records = []
    sites = ['anterior torso', 'posterior torso', 'upper extremity', 'lower extremity', 'head/neck', 'palms/soles']
    genders = ['male', 'female']

    np.random.seed(101)
    for i in range(num_samples):
        img_id = f"ISIC_{2000000 + i}.jpg"
        img_path = os.path.join(img_dir, img_id)
        
        cls_choice = np.random.choice(CLASSES, p=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
        generate_synthetic_lesion_image(img_path, cls_choice)

        age_approx = float(np.random.choice(range(20, 85, 5)))
        sex = np.random.choice(genders)
        anatom_site = np.random.choice(sites)

        records.append({
            'img_id': img_id,
            'diagnostic': cls_choice,
            'age': age_approx,
            'sex': sex,
            'anatom_site_predominance': anatom_site
        })

    df = pd.DataFrame(records)
    df.to_csv(meta_path, index=False)
    print(f"[OK] Generated mock ISIC 2019 dataset with {len(df)} samples at {meta_path}")
    return df

if __name__ == "__main__":
    setup_mock_pad_ufes(num_samples=120)
    setup_mock_isic_2019(num_samples=60)
