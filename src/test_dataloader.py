import torch
from torch.utils.data import DataLoader

from .camvid_dataset import CamVidDataset


TRAIN_IMAGE_DIR = "data/processed/train/images"
TRAIN_MASK_DIR = "data/processed/train/masks"


dataset = CamVidDataset(
    image_dir=TRAIN_IMAGE_DIR,
    mask_dir=TRAIN_MASK_DIR
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)


images, masks = next(iter(loader))


print("Dataset size:", len(dataset))
print("Number of batches:", len(loader))

print("Images shape:", images.shape)
print("Images dtype:", images.dtype)

print("Masks shape:", masks.shape)
print("Masks dtype:", masks.dtype)

print("Image min:", images.min().item())
print("Image max:", images.max().item())

print("Unique values in first mask:")
print(torch.unique(masks[0]).tolist())