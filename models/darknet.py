import torch
import torch.nn as nn
from models.attention import SpatialAttention, ChannelAttention

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels // 2, k= 1, p=0)
        self.conv2 = ConvBlock(channels // 2, channels)

        self.spatial_att = SpatialAttention()
        self.channel_att = ChannelAttention(channels)


    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)

        # TODO:recheck the usage of F_i over m_i
        x_z, F_i = self.spatial_att(x) # the F_i is intended to use in AGDL loss
        z, cav = self.channel_att(x_z) # Z is used in pruning

        out = residual + z
        return out, F_i, cav
    
class DarknetBlock(nn.Module):
    def __init__(self, in_channels, num_residuals):
        super().__init__()
        self.downsample = ConvBlock(in_channels, in_channels * 2, k=3, s=2, p=1)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(in_channels * 2) for _ in range(num_residuals)]
        )

    def forward(self, x):
        x = self.downsample(x)
        F_list, Z_list = [], []

        for block in self.res_blocks:
            x, F_i, Z = block(x)
            F_list.append(F_i)
            Z_list.append(Z)

        return x, F_list, Z_list
    


class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = ConvBlock(3, 32)
        self.block1 = DarknetBlock(32, 1)
        self.block2 = DarknetBlock(64, 2)
        self.block3 = DarknetBlock(128, 8)
        self.block4 = DarknetBlock(256, 8)
        self.block5 = DarknetBlock(512, 4)

    def forward(self, x):
        attn_data = {"F": [], "Z": []}
        x = self.stage1(x)

        x, F, Z = self.block1(x); attn_data["F"] += F; attn_data["Z"] += Z
        x, F, Z = self.block2(x); attn_data["F"] += F; attn_data["Z"] += Z
        x, F, Z = self.block3(x); attn_data["F"] += F; attn_data["Z"] += Z
        x, F, Z = self.block4(x); attn_data["F"] += F; attn_data["Z"] += Z
        x, F, Z = self.block5(x); attn_data["F"] += F; attn_data["Z"] += Z

        return x, attn_data
    
if __name__ == "__main__":
    model = Darknet53()
    x = torch.randn(1,3,544,544)
    with torch.no_grad():
        out = model(x)
    print("Output shape ", out.shape)