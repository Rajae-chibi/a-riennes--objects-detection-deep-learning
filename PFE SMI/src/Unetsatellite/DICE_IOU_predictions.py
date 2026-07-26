import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#  Dossiers
ground_truth_dir = r"C:\Users\hhhh\Desktop\PFE SMI S6\Unetsatellite\train\masks"
predicted_mask_dir = r"C:\Users\hhhh\Desktop\PFE SMI S6\Unetsatellite\predictions"

#  Classe à évaluer (change selon ton besoin)
class_id = 3  # 1=home, 2=water, 3=tree, 4=sol

#  Métriques
def dice_score(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-6)

def iou_score(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return intersection / (union + 1e-6)

#  Résultats
all_dice = []
all_iou = []
valid_cases = 0

#  Parcours
for fname in os.listdir(ground_truth_dir):
    if fname.endswith(".png"):
        gt_path = os.path.join(ground_truth_dir, fname)
        pred_path = os.path.join(predicted_mask_dir, fname)

        if not os.path.exists(pred_path):
            print(f" Manque prédiction pour : {fname}")
            continue

        gt = np.array(Image.open(gt_path).resize((512, 512), resample=Image.NEAREST))
        pred = np.array(Image.open(pred_path).resize((512, 512), resample=Image.NEAREST))


        gt_bin = (gt == class_id).astype(np.uint8)
        pred_bin = (pred == class_id).astype(np.uint8)

        # 🔎 Debug des valeurs uniques
        print(f"->{fname} | GT: {np.unique(gt)} | Pred: {np.unique(pred)}")

        # ❗ Ignorer si rien à comparer
        if gt_bin.sum() == 0 and pred_bin.sum() == 0:
            print(f" Ignoré (aucun pixel classe {class_id})")
            continue

        dice = dice_score(pred_bin, gt_bin)
        iou = iou_score(pred_bin, gt_bin)
        all_dice.append(dice)
        all_iou.append(iou)
        valid_cases += 1

        #  Affichage
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(gt_bin, cmap='Greens')
        axs[0].set_title("Masque Réel")
        axs[1].imshow(pred_bin, cmap='Blues')
        axs[1].set_title("Masque Prédit")
        axs[2].imshow(gt_bin, cmap='Greens', alpha=0.5)
        axs[2].imshow(pred_bin, cmap='Blues', alpha=0.5)
        axs[2].set_title("Superposition")
        for ax in axs: ax.axis("off")
        plt.suptitle(f"{fname} — Dice: {dice:.3f}, IoU: {iou:.3f}")
        plt.tight_layout()
        plt.show()

#  Moyennes
if valid_cases > 0:
    print(f"\n {valid_cases} images évaluées.")
    print(f"Dice moyen : {np.mean(all_dice):.4f}")
    print(f"IoU  moyen : {np.mean(all_iou):.4f}")
else:
    print(" Aucune image évaluée — vérifie les classes ou les noms de fichiers.")
