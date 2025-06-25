import torch
from CPADA import SpatialAttention, ChannelAttention 
# from models.attention import SpatialAttention, ChannelAttention

def test_spatial_attention():
    B, C, H, W = 2, 64, 32, 32
    x = torch.randn(B, C, H, W)

    sa = SpatialAttention(num_groups=16)
    out, att = sa(x)

    assert out.shape == (B, C, H, W)
    assert att.shape == (B, sa.num_groups, H, W)
    print("✅ Spatial Attention passed")

def test_channel_attention():
    B, C, H, W = 2, 64, 32, 32
    x = torch.randn(B, C, H, W)

    ca = ChannelAttention(C)
    out, att = ca(x)

    assert out.shape == (B, C, H, W)
    assert att.shape == (B, C)
    print("✅ Channel Attention passed")

if __name__ == "__main__":
    test_spatial_attention()
    test_channel_attention()