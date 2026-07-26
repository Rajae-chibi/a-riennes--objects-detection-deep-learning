
import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import json

# ======== CONFIGURATION ========
DATA_DIR = r"train\images"
ANNOTATIONS_FILE = r"train\_annotations.coco.json"
EPOCHS = 10
BATCH_SIZE = 2
NUM_CLASSES = 4 + 1  # 4 classes (sans Water-Rocks) + 1 background

# ======== DATASET COCO CUSTOM ========
class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

        # Filtrer les annotations qui ne sont PAS Water-Rocks (id 0)
        self.valid_cat_ids = [1, 2, 3, 4]  # garder uniquement Car, Pavement, Vegetation, Water

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.valid_cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        img = np.array(img)

        masks = []
        boxes = []
        labels = []
        for ann in anns:
            if ann['category_id'] not in self.valid_cat_ids:
                continue
            mask = self.coco.annToMask(ann)
            masks.append(mask)
            bbox = ann['bbox']
            boxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
            labels.append(self.valid_cat_ids.index(ann['category_id']) + 1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        if len(masks) == 0:
          return self.__getitem__((index + 1) % len(self))

        masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)


        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id])
        }

        if self.transforms:
            img = self.transforms(Image.fromarray(img))

        return img, target

    def __len__(self):
        return len(self.ids)

# ======== TRANSFORMS ========
transform = T.Compose([T.ToTensor()])

# ======== LOAD DATA ========
dataset = CocoDataset(DATA_DIR, ANNOTATIONS_FILE, transforms=transform)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# ======== LOAD MODEL ========
model = maskrcnn_resnet50_fpn(num_classes=NUM_CLASSES)
model.train()

# ======== OPTIMIZER ========
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# ======== TRAIN LOOP ========
for epoch in range(EPOCHS):
    for images, targets in data_loader:
        images = list(img for img in images)
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"[Epoch {epoch+1}] Loss: {losses.item():.4f}")

# ======== SAVE MODEL ========
torch.save(model.state_dict(), "maskrcnn_model.pth")
print(" Modèle enregistré sous 'maskrcnn_model.pth'")
