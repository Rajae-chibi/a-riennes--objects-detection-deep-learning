import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

# Config
DATA_DIR = r"C:\Users\hhhh\Desktop\PFE SMI S6\Unetsatellite\test"
SAVE_DIR = r"C:\Users\hhhh\Desktop\PFE SMI S6\Unetsatellite\predictions1"
IMAGE_SIZE = (512, 512)
CLASSES = ['background', 'home', 'water', 'tree', 'sol']

# Créer dossier si besoin
os.makedirs(SAVE_DIR, exist_ok=True)

# Dataset
class SegmentationDataset(Dataset):
    def __init__(self, base_dir):
        self.image_dir = os.path.join(base_dir, 'images')
        self.images = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
        image = np.array(image) / 255.0
        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)
        return image, img_name.replace('.jpg', '.png')

# Charger modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet("efficientnet-b3", encoder_weights=None, in_channels=3, classes=len(CLASSES)).to(device)
model.load_state_dict(torch.load("unet_model.pth", map_location=device))
model.eval()

# Prédiction et sauvegarde
loader = DataLoader(SegmentationDataset(DATA_DIR), batch_size=1)

with torch.no_grad():
    for img, name in loader:
        img = img.to(device)
        output = model(img)
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

        save_path = os.path.join(SAVE_DIR, name[0])
        Image.fromarray(pred_mask.astype(np.uint8)).save(save_path)

print("Masques prédits enregistrés dans :", SAVE_DIR)
