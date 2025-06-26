import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpatialAttention(nn.Module):
    def __init__(self, num_groups=16):
        super().__init__()
        self.num_groups = num_groups
        self.conv = nn.Conv2d(2,1, kernel_size=1, padding=0, bias=False)
    
    def forward(self, X):
        B, C, H, W = X.size()
        self.num_groups = math.gcd(C, self.num_groups)
        # if C % self.num_groups != 0:
            # raise ValueError("Input channels must be divisible by group size.")
        
        #group split
        Q_i = X.view(B * self.num_groups, C // self.num_groups, H, W)

        AvgPool = torch.mean(Q_i, dim=1, keepdim=True)
        MaxPool, _ = torch.max(Q_i, dim=1, keepdim=True)
        Pool_concat =  torch.cat([AvgPool, MaxPool], dim=1)  # Shape: (B*g, 2, H, W)
        f_i = self.conv(Pool_concat)                      # Shape: (B*g, 1, H, W)
        F_i = torch.sigmoid(f_i)

        m_i = Q_i * F_i  # Element-wise spatial attention
        X_z = m_i.view(B, C, H, W)
        # TODO: check whether F_i is correct or m_i
        return X_z, F_i.view(B, self.num_groups, H, W)  # return attention for AGDL
    

class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        k = math.ceil(math.log2(channels))
        # k_odd = k+1 if k % 2 == 0 else k
        if k % 2 == 0: k += 1

        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=k//2)

    def forward(self, X):
        B, C, _, _ = X.size()
        A = self.avg_pool(X).view(B, 1, C)  # (B, 1, C)
        M = self.max_pool(X).view(B, 1, C)

        A_s = self.conv1d(A)
        M_s = self.conv1d(M)

        Channel_Attention_Vector = torch.sigmoid(A_s + M_s).view(B, C, 1, 1)

        Z = X * Channel_Attention_Vector
        return Z, Channel_Attention_Vector.view(B, C)