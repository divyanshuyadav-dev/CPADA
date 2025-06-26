from losses.agdl_loss import compute_agdl_loss
from losses.yolo_loss import compute_yolo_loss


def cpada_loss(preds, targets, attn_data, anchors, beta1=1.0, beta2=1.0, lambda1=1.0, lambda2=1.0, t1=1.0, t2=1.0):
    """
    Combined loss used in CPADA:
    L_total = beta1 * AGDL + beta2 * (YOLO)
    """
    agdl = compute_agdl_loss(
        attn_data["F"], attn_data["Z"],
        lambda1=lambda1, lambda2=lambda2, t1=t1, t2=t2
    )

    yolo = compute_yolo_loss(preds, targets, anchors, num_classes=1)

    return beta1 * agdl + beta2 * yolo