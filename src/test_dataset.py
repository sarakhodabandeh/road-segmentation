from .camvid_dataset import CamVidDataset


dataset = CamVidDataset(
    image_dir="data/raw/701_StillsRaw_full",
    mask_dir="data/raw"
)

print("Dataset size:", len(dataset))

image, mask = dataset[0]

print("Image shape:", image.shape)
print("Image dtype:", image.dtype)

print("Mask shape:", mask.shape)
print("Mask dtype:", mask.dtype)

print("Unique mask classes:", mask.unique().tolist())