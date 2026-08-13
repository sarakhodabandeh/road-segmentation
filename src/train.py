import os

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
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3

TRAIN_IMAGE_DIR = "data/processed/train/images"
TRAIN_MASK_DIR = "data/processed/train/masks"

VAL_IMAGE_DIR = "data/processed/val/images"
VAL_MASK_DIR = "data/processed/val/masks"

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)


# --------------------------------
# Device
# --------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)


# --------------------------------
# Datasets
# --------------------------------

train_dataset = CamVidDataset(
    image_dir=TRAIN_IMAGE_DIR,
    mask_dir=TRAIN_MASK_DIR
)

val_dataset = CamVidDataset(
    image_dir=VAL_IMAGE_DIR,
    mask_dir=VAL_MASK_DIR
)


# --------------------------------
# DataLoaders
# --------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# --------------------------------
# Model
# --------------------------------

model = UNet(
    num_classes=NUM_CLASSES
).to(device)


# --------------------------------
# Loss
# --------------------------------

criterion = nn.CrossEntropyLoss(
    ignore_index=255
)


# --------------------------------
# Optimizer
# --------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------
# Checkpoint directory
# --------------------------------

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


best_miou = 0.0


# ========================================
# Training Loop
# ========================================

for epoch in range(NUM_EPOCHS):

    print("\n" + "=" * 50)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print("=" * 50)

    # --------------------------------
    # Training
    # --------------------------------

    model.train()

    running_train_loss = 0.0

    for batch_index, (images, masks) in enumerate(train_loader):

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        running_train_loss += loss.item()

        if (batch_index + 1) % 20 == 0:
            print(
                f"Batch {batch_index + 1}/{len(train_loader)} "
                f"- Loss: {loss.item():.4f}"
            )

    train_loss = (
        running_train_loss /
        len(train_loader)
    )


    # --------------------------------
    # Validation
    # --------------------------------

    model.eval()

    running_val_loss = 0.0

    total_accuracy = 0.0
    total_miou = 0.0
    total_dice = 0.0

    num_val_batches = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                masks
            )

            running_val_loss += loss.item()

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

            total_accuracy += accuracy
            total_miou += miou
            total_dice += dice

            num_val_batches += 1


    val_loss = (
        running_val_loss /
        num_val_batches
    )

    val_accuracy = (
        total_accuracy /
        num_val_batches
    )

    val_miou = (
        total_miou /
        num_val_batches
    )

    val_dice = (
        total_dice /
        num_val_batches
    )


    # --------------------------------
    # Results
    # --------------------------------

    print("\nResults:")
    print(f"Train Loss:       {train_loss:.4f}")
    print(f"Validation Loss:  {val_loss:.4f}")
    print(f"Pixel Accuracy:   {val_accuracy:.4f}")
    print(f"Mean IoU:          {val_miou:.4f}")
    print(f"Dice Score:        {val_dice:.4f}")


    # --------------------------------
    # Save best model
    # --------------------------------

    if val_miou > best_miou:

        best_miou = val_miou

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_miou": val_miou,
                "val_loss": val_loss,
            },
            BEST_MODEL_PATH
        )

        print(
            f"\n✓ New best model saved!"
        )

        print(
            f"Best mIoU: {best_miou:.4f}"
        )


print("\n" + "=" * 50)
print("Training completed!")
print("=" * 50)

print(
    f"Best Validation mIoU: {best_miou:.4f}"
)

print(
    f"Best model saved at: {BEST_MODEL_PATH}"
)