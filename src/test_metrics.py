import torch
from torch.utils.data import DataLoader

from .camvid_dataset import CamVidDataset
from .unet import UNet
from .camvid_classes import NUM_CLASSES, IGNORE_INDEX


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

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)


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
).to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
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


# --------------------------------------------------
# Metrics
# --------------------------------------------------

total_pixels = 0
correct_pixels = 0

intersection = torch.zeros(
    NUM_CLASSES,
    dtype=torch.float64
)

union = torch.zeros(
    NUM_CLASSES,
    dtype=torch.float64
)

dice_intersection = torch.zeros(
    NUM_CLASSES,
    dtype=torch.float64
)

dice_denominator = torch.zeros(
    NUM_CLASSES,
    dtype=torch.float64
)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        # Ignore Void pixels
        valid = masks != IGNORE_INDEX

        true = masks[valid]
        pred = predictions[valid]

        # ------------------------------------------
        # Pixel Accuracy
        # ------------------------------------------

        correct_pixels += (
            (pred == true).sum().item()
        )

        total_pixels += true.numel()


        # ------------------------------------------
        # Per-class statistics
        # ------------------------------------------

        for class_id in range(NUM_CLASSES):

            true_class = true == class_id
            pred_class = pred == class_id

            tp = (
                true_class & pred_class
            ).sum().item()

            fp = (
                (~true_class) & pred_class
            ).sum().item()

            fn = (
                true_class & (~pred_class)
            ).sum().item()

            intersection[class_id] += tp

            union[class_id] += (
                tp + fp + fn
            )

            dice_intersection[class_id] += tp

            dice_denominator[class_id] += (
                2 * tp + fp + fn
            )


# --------------------------------------------------
# Final Metrics
# --------------------------------------------------

pixel_accuracy = (
    correct_pixels / total_pixels
)

iou_per_class = (
    intersection / union.clamp(min=1)
)

dice_per_class = (
    2 * dice_intersection
    / dice_denominator.clamp(min=1)
)

mean_iou = iou_per_class.mean().item()

mean_dice = dice_per_class.mean().item()


# --------------------------------------------------
# Results
# --------------------------------------------------

print()
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    f"Pixel Accuracy: {pixel_accuracy:.4f}"
)

print(
    f"Mean IoU:       {mean_iou:.4f}"
)

print(
    f"Dice Score:     {mean_dice:.4f}"
)

print("=" * 60)