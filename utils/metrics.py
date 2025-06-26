import torch
import numpy as np

def compute_map(predictions, targets, iou_threshold=0.5, obj_threshold=0.5):
    true_positives = 0
    false_positives = 0
    total_ground_truths = 0

    for pred_list, target in zip(predictions, targets):
        # Pick only the largest scale (e.g. 52x52, out1)
        pred = pred_list[0]  # shape: [B, A, H, W, 5+C]
        B, A, H, W, _ = pred.shape

        # Flatten all predictions across anchors and grid
        pred = pred.reshape(B, -1, pred.shape[-1])  
        pred = pred[0]  # Only first image in batch

        # Filter by objectness
        obj_scores = pred[:, 4]
        mask = obj_scores > obj_threshold
        pred = pred[mask]

        if pred.shape[0] == 0:
            continue

        pred_boxes = pred[:, :4]  # [cx, cy, w, h]
        pred_boxes = [xywh_to_xyxy(box.tolist()) for box in pred_boxes]

        # Clean target boxes
        target = target[0]  # Only first image in batch
        target = target[target[:, 2] > 0]  # Filter empty boxes
        target_boxes = [xywh_to_xyxy(box.tolist()[:4]) for box in target]

        matched_gt = []
        for p in pred_boxes:
            best_iou = 0
            best_match = -1
            for i, gt in enumerate(target_boxes):
                iou = calculate_iou(p, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_match = i
            if best_iou > iou_threshold:
                matched_gt.append(best_match)

        for matched in matched_gt:
            if matched != -1:
                true_positives += 1
            else:
                false_positives += 1
        total_ground_truths += len(target_boxes)

    if true_positives + false_positives == 0:
        return 0.0

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / total_ground_truths if total_ground_truths > 0 else 0.0
    mAP = precision * recall  # Simplified proxy

    return mAP

def xywh_to_xyxy(box):
    """
    Converts [cx, cy, w, h] → [x1, y1, x2, y2]
    Args:
        box: list or 1D tensor of length 4
    Returns:
        list of [x1, y1, x2, y2]
    """
    cx, cy, w, h = box
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two boxes
    Each box is [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

