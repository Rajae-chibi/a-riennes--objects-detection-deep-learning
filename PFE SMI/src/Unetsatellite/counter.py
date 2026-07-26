import os
import numpy as np
from PIL import Image
from collections import Counter

# 🔧 Définir les classes
CLASSES = ['background', 'home', 'water', 'tree', 'sol']

def count_class_frequencies(mask_folder, class_names):
    counter = Counter()
    total_pixels = 0

    for fname in os.listdir(mask_folder):
        if fname.endswith(".png"):
            mask = np.array(Image.open(os.path.join(mask_folder, fname)))
            unique, counts = np.unique(mask, return_counts=True)
            counter.update(dict(zip(unique, counts)))
            total_pixels += mask.size

    print(" Répartition des classes dans les masques :")
    for cls_index in range(len(class_names)):
        count = counter.get(cls_index, 0)
        percent = (count / total_pixels) * 100 if total_pixels > 0 else 0
        print(f"  {class_names[cls_index]} ({cls_index}): {count} pixels -> {percent:.2f}%")


# 📁 Utilisation : remplace le chemin si besoin
count_class_frequencies(r"C:\Users\hhhh\Desktop\segUNET\train\masks", CLASSES)
