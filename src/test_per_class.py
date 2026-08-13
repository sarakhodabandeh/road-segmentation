import torch
from torch.utils.data import DataLoader

from .camvid_dataset import CamVidDataset
from .unet import UNet
from .camvid_classes import CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TEST_IMAGE_DIR = "data/processed/test/images"
TEST_MASK_DIR = "data/processed/test/masks"

CHECKPOINT_PATH = "checkpoints/best_model.pth"

BATCH_SIZE = 4


# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", DEVICE)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

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


# --------------------------------------------------
# Model
# --------------------------------------------------

model = UNet(
    num_classes=NUM_CLASSES
).to(DEVICE)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("Loaded checkpoint from epoch:", checkpoint["epoch"])


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

confusion_matrix = torch.zeros(
    NUM_CLASSES,
    NUM_CLASSES,
    dtype=torch.int64
)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        # Ignore Void pixels
        valid_pixels = masks != IGNORE_INDEX

        true_labels = masks[valid_pixels].cpu()
        predicted_labels = predictions[valid_pixels].cpu()

        # Convert pairs into indices
        indices = (
            true_labels * NUM_CLASSES
            + predicted_labels
        )

        counts = torch.bincount(
            indices,
            minlength=NUM_CLASSES * NUM_CLASSES
        )

        confusion_matrix += counts.reshape(
            NUM_CLASSES,
            NUM_CLASSES
        )


# --------------------------------------------------
# Per-Class IoU
# --------------------------------------------------

print()
print("=" * 60)
print("PER-CLASS IoU")
print("=" * 60)

ious = []

for class_id in range(NUM_CLASSES):

    true_positive = confusion_matrix[
        class_id,
        class_id
    ].item()

    false_positive = (
        confusion_matrix[:, class_id].sum().item()
        - true_positive
    )

    false_negative = (
        confusion_matrix[class_id, :].sum().item()
        - true_positive
    )

    denominator = (
        true_positive
        + false_positive
        + false_negative
    )

    if denominator == 0:
        iou = float("nan")
    else:
        iou = true_positive / denominator

    ious.append(iou)

    if iou != iou:
        print(
            f"{class_id:2d}. "
            f"{CLASS_NAMES[class_id]:12s} "
            f"IoU: N/A"
        )
    else:
        print(
            f"{class_id:2d}. "
            f"{CLASS_NAMES[class_id]:12s} "
            f"IoU: {iou:.4f}"
        )


# --------------------------------------------------
# Mean IoU
# --------------------------------------------------

valid_ious = [
    iou for iou in ious
    if iou == iou
]

mean_iou = sum(valid_ious) / len(valid_ious)


print("=" * 60)
print(f"Mean IoU: {mean_iou:.4f}")
print("=" * 60)