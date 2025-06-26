import torch
from torch.utils.data import DataLoader
from models.cpada_yolo import CPADAYOLO
from datasets.transforms import PigFaceDataset
from losses.cpada_loss import cpada_loss
from utils.metrics import compute_map  # Make sure you have a metric utility
from utils.visualizer import visualize_predictions

# === 1. Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = CPADAYOLO(num_classes=1).to(device)

# Load the checkpoint (after 1 epoch in this case)
checkpoint_path = "weights/cpada_yolo.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
print(f"[INFO] Loaded model from {checkpoint_path}")

# === 2. Dataset & DataLoader ===
test_dataset = PigFaceDataset(
    image_dir="datasets/pigface/val/images",
    annotation_path="datasets/pigface/val/annotations/val.json",
    img_size=544
)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

print(f"[INFO] Loaded {len(test_dataset)} validation samples")

anchors = [
    [(10,13), (16,30), (33,23)],     # for 52x52 scale
    [(30,61), (62,45), (59,119)],    # for 26x26 scale
    [(116,90), (156,198), (373,326)] # for 13x13 scale
]

# === 3. Evaluation Loop ===
model.eval()  # Set model to evaluation mode
total_loss = 0.0
total_preds = []
total_targets = []

with torch.no_grad():
    for i, (images, targets) in enumerate(test_loader):
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        preds, attn_data = model(images)

        # Compute loss
        loss = cpada_loss(preds, targets, attn_data, anchors, beta1=1.0, beta2=1.0)
        total_loss += loss.item()

        # Store predictions and targets for metrics calculation
        total_preds.append(preds)
        total_targets.append(targets)

        if i <= 3:
            img, trgt = test_dataset[i]
            predictions = total_preds[i]
            visualize_predictions(img, trgt, predictions)


        print(i)
        if i == 100:
            break

# Calculate average loss over the validation set
avg_loss = total_loss / len(test_loader)
print(f"[INFO] Average Loss on Validation Set: {avg_loss:.4f}")

# === 4. Compute mAP (Mean Average Precision) ===
# TODO: implement the map metrices
mAP = compute_map(total_preds, total_targets)
print(f"[INFO] mAP (mean Average Precision): {mAP:.4f}")

# === 5. (Optional) Visualize predictions ===
# TODO: implement visualiser

# for i in range(3):  # Show first 3 images
#     image, target = test_dataset[i]
#     pred = total_preds[i]
#     visualize_predictions(image, target, pred)  # Ensure this function is implemented