
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# ======== CONFIGURATION ========
MODEL_PATH = "maskrcnn_model.pth"
# IMAGE_PATH = "train/images/5_8_01_00pm_10_JPG.rf.0e256c74e1dd953096f74adae01c2b90.jpg" 
IMAGE_PATH ="outimg/frame104_jpg.rf.b907d8da5840a8fe98f264145a3ce11d.jpg"# <---  image hors dataset
CLASS_NAMES = ["__background__", "Car", "Pavement", "Vegetation", "Water"]
SCORE_THRESHOLD = 0.15

# Couleurs fixes pour chaque classe
CLASS_COLORS = {
    "Car": (255, 0, 0),
    "Pavement": (128, 128, 128),
    "Vegetation": (0, 255, 0),
    "Water": (0, 0, 255),
}

# ======== CHARGER LE MODELE ========
model = maskrcnn_resnet50_fpn(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ======== CHARGER IMAGE ========
image = Image.open(IMAGE_PATH).convert("RGB")
transform = T.ToTensor()
img_tensor = transform(image).unsqueeze(0)

# ======== PRÉDICTION ========
with torch.no_grad():
    output = model(img_tensor)[0]

# ======== FILTRAGE DES PRÉDICTIONS ========
scores = output["scores"]
keep = scores >= SCORE_THRESHOLD
output = {k: v[keep] for k, v in output.items()}

# ======== VISUALISATION ========
image_np = np.array(image)
mask_canvas = np.zeros_like(image_np)

for i in range(len(output["scores"])):
    label_idx = output["labels"][i].item()
    label = CLASS_NAMES[label_idx] if label_idx < len(CLASS_NAMES) else f"Class {label_idx}"
    mask = output["masks"][i, 0].numpy()
    mask = (mask > 0.5).astype(np.uint8)

    color = CLASS_COLORS.get(label, (255, 255, 0))

    for c in range(3):
        mask_canvas[:, :, c] = np.where(mask == 1, color[c], mask_canvas[:, :, c])

    # Affichage nom de classe centré
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(mask_canvas, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(mask_canvas, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

# Fusion des masques et image originale
final_img = cv2.addWeighted(image_np, 0.6, mask_canvas, 0.6, 0)

# ======== AFFICHAGE ========
plt.figure(figsize=(16, 8))

# Image originale
plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.axis("off")
plt.title("Image Originale")

# Image + Masque Prédit
plt.subplot(1, 2, 2)
plt.imshow(final_img)
plt.axis("off")
plt.title("Segmentation - Mask R-CNN")

plt.tight_layout()
plt.show()
