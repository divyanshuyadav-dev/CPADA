import torch
from models.darknet import Darknet53

def test_darknet_with_attention():
    model = Darknet53()
    x = torch.randn(2, 3, 544, 544)

    with torch.no_grad():
        out, attn = model(x)

    print("Final output shape:", out.shape)
    print("Num spatial attention maps:", len(attn["F"]))
    print("Shape of first spatial map:", attn["F"][0].shape)
    print("Shape of first channel vector:", attn["Z"][0].shape)

if __name__ == "__main__":
    test_darknet_with_attention()
