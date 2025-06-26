import torch
import torch.nn.functional as F

def compute_yolo_loss(preds, targets, anchors, num_classes=1, ignore_thresh=0.5):
    """
    Computes YOLO loss across all 3 scales:
    - preds: list of predictions at 3 scales [B, 3, H, W, 5 + C]
    - targets: [B, max_boxes, 5] with values [x, y, w, h, class_id] (all normalized)
    - anchors: list of anchors for each scale: [ [(w, h), ...], ... ]
    """
    device = preds[0].device
    batch_size = preds[0].size(0)

    total_loc_loss = 0.0
    total_conf_loss = 0.0
    total_cls_loss = 0.0

    for scale_id, pred in enumerate(preds):
        B, A, H, W, _ = pred.shape
        stride = 1.0 / H  # each grid cell covers stride of image (normalized)

        # Get current anchors for this scale and normalize
        scale_anchors = torch.tensor(anchors[scale_id], device=device) / H  # shape [A, 2]

        # Split predictions
        pred_xy = torch.sigmoid(pred[..., 0:2])      # tx, ty → center offsets
        pred_wh = pred[..., 2:4]                     # tw, th → width/height log-space
        pred_conf = torch.sigmoid(pred[..., 4])      # objectness
        pred_cls = torch.sigmoid(pred[..., 5:])      # class confidence (for C=1, still needed)

        # Build grid for decoding
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).float().to(device)  # [H, W, 2]

        grid = grid.view(1, 1, H, W, 2)  # [1, 1, H, W, 2] to match pred

        # Decode box coordinates to normalized format
        box_xy = (pred_xy + grid) * stride                      # center x,y
        box_wh = torch.exp(pred_wh) * scale_anchors.view(1, A, 1, 1, 2)  # width, height
        pred_boxes = torch.cat([box_xy, box_wh], dim=-1)        # [B, A, H, W, 4]

        # ------------------- Assign Ground Truth ---------------------- #
        # Create target masks
        obj_mask = torch.zeros((B, A, H, W), dtype=torch.bool, device=device)
        tgt_xy = torch.zeros((B, A, H, W, 2), device=device)
        tgt_wh = torch.zeros((B, A, H, W, 2), device=device)
        tgt_cls = torch.zeros((B, A, H, W, num_classes), device=device)

        for b in range(B):
            for t in targets[b]:
                gx, gy, gw, gh, cls = t
                if gw == 0 or gh == 0:
                    continue

                gx_idx, gy_idx = int(gx * W), int(gy * H)
                if gx_idx >= W or gy_idx >= H:
                    continue

                gt_box = torch.tensor([gx, gy, gw, gh], device=device).unsqueeze(0)
                anchor_shapes = torch.cat([
                    torch.zeros((A, 2), device=device),
                    scale_anchors
                ], dim=1)  # [A, 4]

                gt_shapes = gt_box.repeat(A, 1)  # [A, 4]
                ious = bbox_iou(gt_shapes, anchor_shapes, format="xywh")

                best_a = torch.argmax(ious)

                obj_mask[b, best_a, gy_idx, gx_idx] = 1
                tgt_xy[b, best_a, gy_idx, gx_idx] = torch.tensor([gx * W - gx_idx, gy * H - gy_idx], device=device)
                tgt_wh[b, best_a, gy_idx, gx_idx] = torch.log(torch.tensor([gw * H, gh * H], device=device) / scale_anchors[best_a] + 1e-16)
                tgt_cls[b, best_a, gy_idx, gx_idx, int(cls)] = 1

        # ------------------ Compute Losses ----------------------------- #
        # Location: SmoothL1 on x,y,w,h where object exists
        loc_loss = F.smooth_l1_loss(pred_xy[obj_mask], tgt_xy[obj_mask], reduction='sum') + \
                   F.smooth_l1_loss(pred_wh[obj_mask], tgt_wh[obj_mask], reduction='sum')

        # Objectness: BCE
        conf_target = obj_mask.float()
        conf_loss = F.binary_cross_entropy(pred_conf, conf_target, reduction='sum')

        # Class: BCE (since binary classification)
        cls_loss = F.binary_cross_entropy(pred_cls[obj_mask], tgt_cls[obj_mask], reduction='sum')

        total_loc_loss += loc_loss
        total_conf_loss += conf_loss
        total_cls_loss += cls_loss

    return total_loc_loss + total_conf_loss + total_cls_loss


def bbox_iou(box1, box2, format="xywh"):
    """
    box1: [N, 4]
    box2: [M, 4]
    Returns: [N, M] IoU matrix
    """
    if format == "xywh":
        # Convert to (x1, y1, x2, y2)
        b1_x1 = box1[:, 0] - box1[:, 2] / 2
        b1_y1 = box1[:, 1] - box1[:, 3] / 2
        b1_x2 = box1[:, 0] + box1[:, 2] / 2
        b1_y2 = box1[:, 1] + box1[:, 3] / 2

        b2_x1 = box2[:, 0] - box2[:, 2] / 2
        b2_y1 = box2[:, 1] - box2[:, 3] / 2
        b2_x2 = box2[:, 0] + box2[:, 2] / 2
        b2_y2 = box2[:, 1] + box2[:, 3] / 2

    inter_x1 = torch.max(b1_x1[:, None], b2_x1)
    inter_y1 = torch.max(b1_y1[:, None], b2_y1)
    inter_x2 = torch.min(b1_x2[:, None], b2_x2)
    inter_y2 = torch.min(b1_y2[:, None], b2_y2)

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area[:, None] + b2_area - inter_area + 1e-6)
