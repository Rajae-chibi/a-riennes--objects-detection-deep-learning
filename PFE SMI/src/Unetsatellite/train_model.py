import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Configuration
IMAGE_SIZE = (512, 512)
CLASSES = ['background', 'home', 'water', 'tree', 'sol']
NUM_CLASSES = len(CLASSES)
DATA_DIR = r'C:\Users\hhhh\Desktop\segUNET\train'
EPOCHS = 60
BATCH_SIZE = 2
LEARNING_RATE = 1e-3

#  Augmentations avancées
train_transform = A.Compose([
    A.Resize(*IMAGE_SIZE),
    A.RandomCrop(480, 480),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(p=0.2),
    A.RandomShadow(p=0.2),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

#  Dataset
class SegmentationDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.image_dir = os.path.join(base_dir, "images")
        self.mask_dir = os.path.join(base_dir, "masks")
        self.images = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))

        image = np.array(Image.open(img_path).convert("RGB").resize(IMAGE_SIZE))
        mask = np.array(Image.open(mask_path).resize(IMAGE_SIZE, resample=Image.NEAREST))

        # Sécurité sur les valeurs
        mask[mask >= NUM_CLASSES] = 0

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].long()
        else:
            image = torch.tensor(image / 255.0, dtype=torch.float).permute(2, 0, 1)
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

#  Modèle avec efficientnet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(
    encoder_name='efficientnet-b3',
    encoder_weights='imagenet',
    in_channels=3,
    classes=NUM_CLASSES
).to(device)

#  Poids adaptés pour compenser les classes minoritaires
class_weights = torch.tensor([1.0, 0.8, 0.9, 2.0, 0.2]).to(device)

#  Optimisation
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

#  Chargement des données
train_loader = DataLoader(
    SegmentationDataset(DATA_DIR, transform=train_transform),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

#  Entraînement
best_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    print(f" Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_unet_model.pth")
        print(" Meilleur modèle sauvegardé.")

torch.save(model.state_dict(), "unet_model.pth")
print("Modèle final sauvegardé.")
