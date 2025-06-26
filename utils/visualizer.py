import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np

def draw_box(ax, box, label, color):
    """
    Draws one bounding box.
    box: [x1, y1, x2, y2]
    label: str (optional)
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    if label:
        ax.text(x1, y1, label, color=color, fontsize=8)

def visualize_predictions(image_tensor, targets, preds, obj_threshold=0.3):
    """
    image_tensor: [3, H, W]
    targets: [max_boxes, 5] → (cx, cy, w, h, cls)
    preds: list of [B, A, H, W, 5+C]
    """
    image = image_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    H, W = image.shape[:2]

    # --- Draw GT boxes ---
    for box in targets:
        if box[2] == 0:
            continue
        x1, y1, x2, y2 = xywh_to_xyxy(box[:4].tolist(), W, H)
        draw_box(ax, [x1, y1, x2, y2], label="GT", color="green")

    # --- Process predictions ---
    pred_tensor = preds[0]  # Use highest resolution scale
    B, A, h, w, C = pred_tensor.shape
    pred_tensor = pred_tensor.reshape(B, -1, C)[0]  # Take 1st image only

    for p in pred_tensor:
        # if p[4] < obj_threshold:
        #     continue
        cx, cy, bw, bh = p[:4]
        x1 = (cx - bw / 2) * W
        y1 = (cy - bh / 2) * H
        x2 = (cx + bw / 2) * W
        y2 = (cy + bh / 2) * H
        draw_box(ax, [x1, y1, x2, y2], label=f"Pred {p[4]:.2f}", color="red")

    plt.axis("off")
    plt.show()

def xywh_to_xyxy(box, img_w, img_h):
    """
    box: [cx, cy, w, h] in [0–1]
    Returns: [x1, y1, x2, y2] in image pixels
    """
    cx, cy, w, h = box
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]