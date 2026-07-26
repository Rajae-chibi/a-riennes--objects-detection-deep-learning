import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import segmentation_models_pytorch as smp

#  Config
DATA_DIR = r'C:\Users\hhhh\Desktop\PFE SMI S6\Unetdrone\imgs'
IMAGE_SIZE = (212,212)
CLASSES = ['background', 'car', 'pavement', 'vegetation', 'water']
CLASS_COLORS = [
    [0, 0, 0],
    [128, 0, 0],
    [128, 128, 128],
    [0, 128, 0],
    [0, 0, 255]
]
#  Dataset
class SegmentationDataset(Dataset):
    def __init__(self, base_dir):
        self.image_dir = os.path.join(base_dir, 'images')
        self.mask_dir = os.path.join(base_dir, 'masks')
        self.images = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '_mask.png'))

        image = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
        mask = Image.open(mask_path).resize(IMAGE_SIZE, resample=Image.NEAREST)

        image = torch.tensor(np.array(image) / 255.0, dtype=torch.float).permute(2, 0, 1)
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        mask[mask >= len(CLASSES)] = 0  # sécurité

        return image, mask

#  Metrics
def dice_score(pred, target, num_classes):
    dice_scores = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice = (2. * intersection) / (union + 1e-6)
        dice_scores.append(dice.item())
    return dice_scores

def iou_score(pred, target, num_classes):
    iou_scores = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        iou = intersection / union if union > 0 else float('nan')
        iou_scores.append(iou)
    return iou_scores

#  Visualisation
def show_image_pred_mask(image, pred_mask, true_mask):
    image = image.permute(1, 2, 0).cpu().numpy()
    cmap = ListedColormap(np.array(CLASS_COLORS) / 255.0)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(image); plt.title("Image"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(pred_mask.cpu(), cmap=cmap); plt.title("Segmentation"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(true_mask.cpu(), cmap=cmap); plt.title("Masque Réel"); plt.axis("off")
    plt.tight_layout()
    plt.show()

#  Chargement du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet("efficientnet-b3", encoder_weights=None, in_channels=3, classes=len(CLASSES)).to(device)
model.load_state_dict(torch.load("best_unet_model.pth", map_location=device))
model.eval()

#  Prédiction + Évaluation
test_loader = DataLoader(SegmentationDataset(DATA_DIR), batch_size=1)

total_dice = np.zeros(len(CLASSES))
total_iou = np.zeros(len(CLASSES))
count = 0

with torch.no_grad():
    for img, mask in test_loader:
        img, mask = img.to(device), mask.to(device)
        output = model(img)
        pred = torch.argmax(output.squeeze(), dim=0)

        dice = dice_score(pred, mask.squeeze(), len(CLASSES))
        iou = iou_score(pred, mask.squeeze(), len(CLASSES))

        total_dice += np.array(dice)
        total_iou += np.array(iou)
        count += 1

        show_image_pred_mask(img.squeeze(), pred, mask.squeeze())
        print("Dice :", ["{:.3f}".format(d) for d in dice])
        print("IoU  :", ["{:.3f}".format(i) for i in iou])
        print("-" * 40)

#  Moyennes
avg_dice = total_dice / count
avg_iou = total_iou / count

print("Moyenne Dice par classe :", ["{:.3f}".format(d) for d in avg_dice])
print(" Moyenne IoU  par classe :", ["{:.3f}".format(i) for i in avg_iou])
