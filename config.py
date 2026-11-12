import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
PAD_UFES_DIR = os.path.join(DATA_DIR, "pad_ufes_20")
ISIC_2019_DIR = os.path.join(DATA_DIR, "isic_2019")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure required folders exist
for folder in [DATA_DIR, PAD_UFES_DIR, ISIC_2019_DIR, CHECKPOINTS_DIR, RESULTS_DIR]:
    os.makedirs(folder, exist_ok=True)

# 6 Shared Core Clinical Classes between PAD-UFES-20 and ISIC 2019:
# 1. BCC: Basal Cell Carcinoma
# 2. SCC: Squamous Cell Carcinoma
# 3. ACK: Actinic Keratosis (AK / AKIEC)
# 4. SEK: Seborrheic Keratosis (BKL)
# 5. MEL: Melanoma
# 6. NEV: Melanocytic Nevus
CLASSES = ["BCC", "SCC", "ACK", "SEK", "MEL", "NEV"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}

# Mappings for ISIC 2019 columns to unified classes
ISIC_MAP = {
    'MEL': 'MEL',
    'NV': 'NEV',
    'BCC': 'BCC',
    'AK': 'ACK',
    'BKL': 'SEK',
    'SCC': 'SCC',
    'VASC': None,  # Not in 6-class target set
    'DF': None     # Dermatofibroma (not in 6-class set)
}

# Standardized Metadata Features across datasets
# Both datasets share age, sex, and anatomical site
STANDARDIZED_SITES = [
    'head/neck', 'face', 'trunk', 'upper extremity', 
    'lower extremity', 'abdomen', 'back', 'chest', 'other/unknown'
]

GENDERS = ['male', 'female', 'unknown']

# Image config
IMAGE_SIZE = 224
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# Training defaults
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 15
DEVICE = "cuda" if False else "cpu"  # Checked dynamically in scripts
