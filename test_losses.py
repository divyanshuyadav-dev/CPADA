import torch
from losses.yolo_loss import compute_yolo_loss
from losses.agdl_loss import compute_agdl_loss
from losses.cpada_loss import cpada_loss

def test_yolo_loss():
    # Dummy predictions: [B, A, H, W, 6] * 3 scales
    B = 2
    num_classes = 1
    preds = [
        torch.randn(B, 3, 68, 68, 5 + num_classes),
        torch.randn(B, 3, 34, 34, 5 + num_classes),
        torch.randn(B, 3, 17, 17, 5 + num_classes),
    ]

    # Dummy targets: [B, max_boxes, 5]
    targets = torch.tensor([
        [[0.5, 0.5, 0.2, 0.3, 0]],
        [[0.4, 0.4, 0.1, 0.1, 0]],
    ])  # shape [2, 1, 5]

    anchors = [
        [(10,13), (16,30), (33,23)],
        [(30,61), (62,45), (59,119)],
        [(116,90), (156,198), (373,326)]
    ]

    loss = compute_yolo_loss(preds, targets, anchors)
    print("YOLO Loss:", loss.item())
    assert loss.requires_grad or loss >= 0

def test_agdl_loss():
    B = 2
    F_list = [torch.rand(B, 16, 68, 68), torch.rand(B, 16, 34, 34)]
    Z_list = [torch.rand(B, 64), torch.rand(B, 128)]

    agdl = compute_agdl_loss(F_list, Z_list)
    print("AGDL Loss:", agdl.item())
    assert agdl.requires_grad or agdl >= 0

def test_cpada_loss():
    B = 2
    preds = [
        torch.randn(B, 3, 68, 68, 6),
        torch.randn(B, 3, 34, 34, 6),
        torch.randn(B, 3, 17, 17, 6),
    ]
    targets = torch.tensor([
        [[0.5, 0.5, 0.2, 0.3, 0]],
        [[0.4, 0.4, 0.1, 0.1, 0]],
    ])  # shape [2, 1, 5]

    attn_data = {
        "F": [torch.rand(B, 16, 68, 68), torch.rand(B, 16, 34, 34)],
        "Z": [torch.rand(B, 64), torch.rand(B, 128)]
    }

    anchors = [
        [(10,13), (16,30), (33,23)],
        [(30,61), (62,45), (59,119)],
        [(116,90), (156,198), (373,326)]
    ]

    loss = cpada_loss(preds, targets, attn_data, anchors)
    print("Total CPADA Loss:", loss.item())
    assert loss.requires_grad or loss >= 0

if __name__ == "__main__":
    test_yolo_loss()
    test_agdl_loss()
    test_cpada_loss()
