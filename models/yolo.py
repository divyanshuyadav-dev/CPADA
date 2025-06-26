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

class YOLOHead(nn.Module):
    def __init__(self, num_classes=1, anchors=[(10,13), (16,30), (33,23)], num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_outputs = num_anchors * (5 + num_classes)

        # High → Low resolution backbone outputs
        # C3: 256, C4: 512, C5: 1024
        self.head3 = YOLODetectionBlock(1024, 512)
        self.out3 = nn.Conv2d(512, self.num_outputs, kernel_size=1)

        self.up2 = nn.Conv2d(512, 256, 1)
        self.head2 = YOLODetectionBlock(768, 256)
        self.out2 = nn.Conv2d(256, self.num_outputs, kernel_size=1)

        self.up1 = nn.Conv2d(256, 128, 1)
        self.head1 = YOLODetectionBlock(384, 128)
        self.out1 = nn.Conv2d(128, self.num_outputs, kernel_size=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def _reshape_output(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
        x = x.view(B, H, W, self.num_anchors, 5 + self.num_classes)
        x = x.permute(0, 3, 1, 2, 4)  # B, num_anchors, H, W, 5 + C
        return x


    def forward(self, C3, C4, C5):
        x = self.head3(C5)
        out3 = self._reshape_output(self.out3(x))

        x = self.upsample(self.up2(x))
        x = torch.cat([x, C4], dim=1)
        x = self.head2(x)
        out2 = self._reshape_output(self.out2(x))

        x = self.upsample(self.up1(x))
        x = torch.cat([x, C3], dim=1)
        x = self.head1(x)
        out1 = self._reshape_output(self.out1(x))

        return [out1, out2, out3]  # [52x52, 26x26, 13x13]