import numpy as np
from PIL import Image
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

IMAGE_SIZE = (212, 212)
CLASSES = ['background', 'car', 'pavement', ' vegetation', 'water']
CLASS_COLORS = [
    [0, 0, 0],
    [128, 0, 0],
    [128, 128, 128],
    [0, 128, 0],
    [0, 0, 255]
]

def show_prediction(image_tensor, pred_mask):
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    cmap = ListedColormap(np.array(CLASS_COLORS) / 255.0)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(image); plt.title("Image"); plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(pred_mask, cmap=cmap); plt.title("Masque Prédit"); plt.axis("off")
    plt.tight_layout()
    plt.show()
    print("Classes détectées :", torch.unique(pred_mask))

def predict_image(path):
    image = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    img_array = np.array(image) / 255.0
    img_tensor = torch.tensor(img_array, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)

    model = smp.Unet("efficientnet-b3", encoder_weights=None, in_channels=3, classes=len(CLASSES))
    model.load_state_dict(torch.load("best_unet_model.pth", map_location=torch.device("cpu")))
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu()
        show_prediction(img_tensor.squeeze(), pred_mask)


predict_image("outimg/5_8_09_00am_68_JPG_jpg.rf.0fdba5557dedc557d254737a0426800d.jpg")
