import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from config import IMAGE_SIZE, IMG_MEAN, IMG_STD, CLASS_TO_IDX, STANDARDIZED_SITES, GENDERS

def get_image_transforms(split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
        ])

def clean_site_name(site_str):
    if not isinstance(site_str, str) or pd.isna(site_str):
        return 'other/unknown'
    s = site_str.lower().strip()
    if any(x in s for x in ['face', 'nose', 'ear', 'cheek', 'forehead', 'lip']):
        return 'face'
    if any(x in s for x in ['head', 'neck', 'scalp']):
        return 'head/neck'
    if any(x in s for x in ['back']):
        return 'back'
    if any(x in s for x in ['chest']):
        return 'chest'
    if any(x in s for x in ['abdomen', 'belly', 'torso', 'trunk']):
        return 'trunk'
    if any(x in s for x in ['arm', 'hand', 'shoulder', 'forearm', 'upper extremity']):
        return 'upper extremity'
    if any(x in s for x in ['leg', 'foot', 'thigh', 'knee', 'lower extremity']):
        return 'lower extremity'
    return 'other/unknown'

def clean_gender(gen_str):
    if not isinstance(gen_str, str) or pd.isna(gen_str):
        return 'unknown'
    g = gen_str.lower().strip()
    if g.startswith('m'):
        return 'male'
    elif g.startswith('f'):
        return 'female'
    return 'unknown'

def encode_metadata(row, age_mean=55.0, age_std=18.0, full_clinical=False):
    """
    Transforms tabular metadata into a fixed-length numerical tensor.
    Supports either standard cross-dataset metadata (Age, Gender, Site) or full PAD-UFES-20 metadata.
    """
    # 1. Age (standardized)
    try:
        raw_age = float(row.get('age', 55.0))
        if np.isnan(raw_age) or raw_age <= 0:
            raw_age = age_mean
    except Exception:
        raw_age = age_mean
    norm_age = (raw_age - age_mean) / (age_std if age_std > 0 else 1.0)
    
    # 2. Gender One-Hot
    gender = clean_gender(row.get('gender', row.get('sex', 'unknown')))
    gender_vec = [1.0 if gender == g else 0.0 for g in GENDERS]
    
    # 3. Anatomical Site One-Hot
    raw_site = row.get('region', row.get('anatom_site_predominance', row.get('anatom_site_special', 'other/unknown')))
    site = clean_site_name(raw_site)
    site_vec = [1.0 if site == s else 0.0 for s in STANDARDIZED_SITES]
    
    feature_list = [norm_age] + gender_vec + site_vec
    
    # Optional full clinical questionnaire attributes from PAD-UFES-20
    if full_clinical:
        clinical_binary_keys = ['itch', 'grew', 'hurt', 'changed', 'bleed', 'elevation', 'smoke', 'drink', 'pesticide']
        for k in clinical_binary_keys:
            val = row.get(k, False)
            if isinstance(val, (bool, np.bool_)):
                b_val = 1.0 if val else 0.0
            elif isinstance(val, (int, float)):
                b_val = 1.0 if val > 0 else 0.0
            elif isinstance(val, str):
                b_val = 1.0 if val.lower() in ['true', 'yes', 't', '1'] else 0.0
            else:
                b_val = 0.0
            feature_list.append(b_val)
            
        # Diameters
        for d in ['diameter_1', 'diameter_2']:
            try:
                dv = float(row.get(d, 0.0))
                if np.isnan(dv): dv = 0.0
            except:
                dv = 0.0
            feature_list.append(dv / 20.0) # scaled
            
    return np.array(feature_list, dtype=np.float32)

class MultimodalSkinDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, full_clinical=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.full_clinical = full_clinical

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load Image
        img_id = str(row['img_id'])
        if not (img_id.endswith('.png') or img_id.endswith('.jpg') or img_id.endswith('.jpeg')):
            # default search with extensions
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = os.path.join(self.img_dir, img_id + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            else:
                img_path = os.path.join(self.img_dir, img_id + '.png')
        else:
            img_path = os.path.join(self.img_dir, img_id)
            
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            # Fallback placeholder synthetic lesion image if file missing
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color=(200, 160, 150))
            
        if self.transform:
            image = self.transform(image)
            
        # Metadata encoding
        meta_features = encode_metadata(row, full_clinical=self.full_clinical)
        meta_tensor = torch.tensor(meta_features, dtype=torch.float32)
        
        # Label
        label_str = row['diagnostic']
        label = CLASS_TO_IDX.get(label_str, 0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return image, meta_tensor, label_tensor, img_id
