from datasets.transforms import PigFaceDataset
from torch.utils.data import DataLoader

def test_loader():
    dataset = PigFaceDataset(
        image_dir="datasets/pigface/train/images",
        annotation_path="datasets/pigface/train/annotations/train.json"
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    for images, targets in loader:
        print("Image batch:", images.shape)       # [B, 3, 544, 544]
        print("Target batch:", targets.shape)     # [B, max_boxes, 5]
        break

if __name__ == "__main__":
    test_loader()