import torch
import torch.nn as nn

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
        self.block = nn.Sequential(
            ConvBlock(channels, channels // 2, k = 1, p = 0),
            ConvBlock(channels // 2, channels)
        )

    def forward(self, x):
        return x  + self.block(x)
    
class DarknetBlock(nn.Module):
    def __init__(self, in_channels, num_residuals):
        super().__init__()
        layers = [ConvBlock(in_channels, in_channels * 2, k =3, s =2, p =1)]
        for _ in range(num_residuals):
            layers.append(ResidualBlock(in_channels * 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

    
class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(3,32),
            DarknetBlock(32, 1),
            DarknetBlock(64, 2),
            DarknetBlock(128, 8),
            DarknetBlock(256, 8),
            DarknetBlock(512, 4),
        )

    def forward(self, x):
        return self.layers(x)
    
if __name__ == "__main__":
    model = Darknet53()
    x = torch.randn(1,3,544,544)
    with torch.no_grad():
        out = model(x)
    print("Output shape ", out.shape)