import torch
import torch.nn as nn
from models.darknet import Darknet53
from models.yolo import YOLOHead

class CPADAYOLO(nn.Module):
    def __init__(self, num_classes=1, num_anchors=3):
        super().__init__()
        self.backbone = Darknet53()
        self.head = YOLOHead(num_classes=num_classes, num_anchors=num_anchors)

    def forward(self, x):
        # Step 1: Backbone
        features, attn_data = self.backbone(x)  # features = [C3, C4, C5]

        # Step 2: YOLO Head
        predictions = self.head(*features)  # returns [out1, out2, out3]

        return predictions, attn_data