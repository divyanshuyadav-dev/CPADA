import torch
import torch.nn as nn

class YOLODetectionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),

            nn.Conv2d(out_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),

            nn.Conv2d(out_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.layers(x)