import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import your custom components
from models.cpada_yolo import CPADAYOLO
from datasets.transforms import PigFaceDataset
from losses.cpada_loss import cpada_loss

# === Step 1: Define Anchors ===
# These are YOLOv3-compatible anchor boxes (width, height)
anchors = [
    [(10,13), (16,30), (33,23)],     # for 52x52 scale
    [(30,61), (62,45), (59,119)],    # for 26x26 scale
    [(116,90), (156,198), (373,326)] # for 13x13 scale
]

# === Step 2: Training Setup ===
def train_model(
    img_size=544,
    batch_size=2,
    epochs=10,
    lr=1e-4,
    save_path="weights/cpada_yolo.pt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 2.1: Load dataset ===
    train_dataset = PigFaceDataset(
        image_dir="datasets/pigface/train/images",
        annotation_path="datasets/pigface/train/annotations/train.json",
        img_size=img_size
    )
    num_classes = train_dataset.class_count
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"[INFO] Loaded {len(train_dataset)} training samples")

    # === 2.2: Build model ===
    model = CPADAYOLO(num_classes=1).to(device)
    print("[INFO] Model initialized")

    # === 2.3: Define optimizer ===
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # === 2.4: Begin training loop ===
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)             # [B, 3, H, W]
            targets = targets.to(device)           # [B, max_boxes, 5]

            # === Forward pass ===
            preds, attn_data = model(images)

            # === Loss computation ===
            loss = cpada_loss(
                preds, targets, attn_data, anchors,
                # num_classes=num_classes,
                beta1=1.0, beta2=1.0, lambda1=1.0, lambda2=1.0,
                t1=1.0, t2=1.0
            )

            # === Backpropagation ===
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch+1}] Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")

        # === Optional: Save checkpoint ===
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"[INFO] Checkpoint saved to {save_path}")

    print("[DONE] Training complete.")

# === Entry Point ===
if __name__ == "__main__":
    train_model()