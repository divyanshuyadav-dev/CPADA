import torch
from models.yolo import YOLOHead

def test_yolo_head():
    model = YOLOHead(num_classes=1)
    C3 = torch.randn(2, 256, 52, 52)
    C4 = torch.randn(2, 512, 26, 26)
    C5 = torch.randn(2, 1024, 13, 13)

    out = model(C3, C4, C5)

    print("Output shapes:")
    for i, o in enumerate(out):
        print(f"  Scale {i+1}: {o.shape}")  # Expecting: [B, 3, H, W, 6]
        assert o.shape[1] == 3, "Anchor count should be 3"
        assert o.shape[-1] == 6, "Output should include 5 bbox params + 1 class"


if __name__ == "__main__":
    test_yolo_head()
