import torch
from models.cpada_yolo import CPADAYOLO

def test_cpada_model():
    model = CPADAYOLO(num_classes=1, num_anchors=3)
    x = torch.randn(2, 3, 544, 544)

    with torch.no_grad():
        preds, attn = model(x)

    print("Predictions:")
    for i, p in enumerate(preds):
        print(f"  Scale {i+1}: {p.shape}")  # should be [2, 3, H, W, 6]

    print(f"\nTotal spatial maps: {len(attn['F'])}")
    print(f"First F_i shape: {attn['F'][0].shape}")
    print(f"First Z shape: {attn['Z'][0].shape}")

if __name__ == "__main__":
    test_cpada_model()