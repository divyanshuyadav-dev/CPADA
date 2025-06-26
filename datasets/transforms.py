import os
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import json

class PigFaceDataset(Dataset):
    def __init__(self, image_dir, annotation_path, img_size=544, max_boxes=50, transform=None):
        self.image_dir = image_dir
        self.img_size = img_size
        self.max_boxes = max_boxes
        self.transform = transform

        # Load COCO-style annotations
        with open(annotation_path) as f:
            data = json.load(f)

        self.images = data['images']
        self.annotations = data['annotations']
        self.categories = data['categories']

        # Build a lookup: image_id → [anns]
        self.image_id_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            self.image_id_to_anns.setdefault(img_id, []).append(ann)

        self.id_to_filename = {img['id']: img['file_name'] for img in self.images}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = image_info['id']
        file_name = image_info['file_name']
        img_path = os.path.join(self.image_dir, file_name)

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transform (resize, to_tensor, normalize)
        transform = self.transform or T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
        ])
        image = transform(image)

        # Get annotations for this image
        anns = self.image_id_to_anns.get(image_id, [])

        # Target tensor: [max_boxes, 5] → (x_center, y_center, w, h, class)
        target = torch.zeros((self.max_boxes, 5))
        for i, ann in enumerate(anns):
            if i >= self.max_boxes:
                break
            x, y, w, h = ann['bbox']  # COCO format: top-left x,y,w,h
            cx = (x + w / 2) / image_info['width']
            cy = (y + h / 2) / image_info['height']
            norm_w = w / image_info['width']
            norm_h = h / image_info['height']
            class_id = ann['category_id']
            target[i] = torch.tensor([cx, cy, norm_w, norm_h, class_id])

        return image, target