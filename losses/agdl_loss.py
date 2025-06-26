import torch
import torch.nn.functional as F

def compute_agdl_loss(F_list, Z_list, lambda1=1.0, lambda2=1.0, t1=1.0, t2=1.0):
    """
    F_list: list of spatial attention maps, each of shape [B, G, H, W]
    Z_list: list of channel attention vectors, each of shape [B, C]
    Returns: scalar AGDL loss
    """
    B = F_list[0].shape[0]
    if B < 2:
        return torch.tensor(0.0, device=F_list[0].device)

    agdl_loss = 0.0
    count = 0

    for F_i in F_list:
        B, G, H, W = F_i.shape
        F_flat = F_i.view(B, -1)  # [B, G*H*W]
        for i in range(B):
            for j in range(i + 1, B):
                dist_m = F.pairwise_distance(F_flat[i], F_flat[j])
                agdl_loss += F.relu(t1 - dist_m) * lambda1
                count += 1

    for Z in Z_list:
        B, C = Z.shape
        for i in range(B):
            for j in range(i + 1, B):
                dist_p = F.pairwise_distance(Z[i], Z[j])
                agdl_loss += F.relu(t2 - dist_p) * lambda2
                count += 1

    agdl_loss *= 2.0 / (B * (B - 1))
    return agdl_loss
