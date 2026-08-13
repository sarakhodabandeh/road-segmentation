import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .camvid_dataset import CamVidDataset
from .unet import UNet
from .metrics import (
    pixel_accuracy,
    mean_iou,
    dice_score
)


# --------------------------------
# Configuration
# --------------------------------

NUM_CLASSES = 11
BATCH_SIZE = 4

TEST_IMAGE_DIR = "data/processed/test/images"
TEST_MASK_DIR = "data/processed/test/masks"

MODEL_PATH = "checkpoints/best_model.pth"


# --------------------------------
# Device
# --------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)


# --------------------------------
# Test Dataset
# --------------------------------

test_dataset = CamVidDataset(
    image_dir=TEST_IMAGE_DIR,
    mask_dir=TEST_MASK_DIR
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test samples:", len(test_dataset))
print("Number of batches:", len(test_loader))


# --------------------------------
# Load Model
# --------------------------------

model = UNet(
    num_classes=NUM_CLASSES
).to(device)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Loaded checkpoint from epoch:",
    checkpoint["epoch"]
)

print(
    "Validation mIoU of checkpoint:",
    f"{checkpoint['val_miou']:.4f}"
)


# --------------------------------
# Loss
# --------------------------------

criterion = nn.CrossEntropyLoss(
    ignore_index=255
)


# --------------------------------
# Test
# --------------------------------

total_loss = 0.0
total_accuracy = 0.0
total_miou = 0.0
total_dice = 0.0

num_batches = 0


with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            masks
        )

        accuracy = pixel_accuracy(
            outputs,
            masks
        )

        miou = mean_iou(
            outputs,
            masks,
            num_classes=NUM_CLASSES
        )

        dice = dice_score(
            outputs,
            masks,
            num_classes=NUM_CLASSES
        )

        total_loss += loss.item()
        total_accuracy += accuracy
        total_miou += miou
        total_dice += dice

        num_batches += 1


# --------------------------------
# Final Results
# --------------------------------

test_loss = total_loss / num_batches
test_accuracy = total_accuracy / num_batches
test_miou = total_miou / num_batches
test_dice = total_dice / num_batches


print("\n" + "=" * 50)
print("FINAL TEST RESULTS")
print("=" * 50)

print(f"Test Loss:       {test_loss:.4f}")
print(f"Pixel Accuracy:  {test_accuracy:.4f}")
print(f"Mean IoU:        {test_miou:.4f}")
print(f"Dice Score:      {test_dice:.4f}")

print("=" * 50)