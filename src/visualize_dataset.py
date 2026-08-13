import matplotlib.pyplot as plt
import numpy as np
import torch

from .camvid_dataset import CamVidDataset
from .camvid_classes import CLASS_NAMES, IGNORE_INDEX


# -----------------------------
# Load dataset
# -----------------------------

dataset = CamVidDataset(
    image_dir="data/processed/train/images",
    mask_dir="data/processed/train/masks"
)


# Get one sample
image, mask = dataset[0]


# -----------------------------
# Convert image for display
# -----------------------------

image_display = image.permute(1, 2, 0).numpy()


# -----------------------------
# Create colored mask
# -----------------------------

# A simple visualization color for each class
colors = np.array([
    [128, 128, 128],  # Sky
    [128, 0, 0],      # Building
    [192, 192, 128],  # Pole
    [128, 64, 128],   # Road
    [0, 0, 192],      # Pavement
    [128, 128, 0],    # Tree
    [192, 128, 128],  # SignSymbol
    [64, 64, 128],    # Fence
    [64, 0, 128],     # Car
    [64, 64, 0],      # Pedestrian
    [0, 128, 192],    # Bicyclist
], dtype=np.uint8)


mask_numpy = mask.numpy()

colored_mask = np.zeros(
    (mask_numpy.shape[0], mask_numpy.shape[1], 3),
    dtype=np.uint8
)

for class_id in range(len(CLASS_NAMES)):
    colored_mask[mask_numpy == class_id] = colors[class_id]


# Ignore pixels remain black
colored_mask[mask_numpy == IGNORE_INDEX] = [0, 0, 0]


# -----------------------------
# Create overlay
# -----------------------------

overlay = image_display.copy()

valid_pixels = mask_numpy != IGNORE_INDEX

overlay[valid_pixels] = (
    0.5 * image_display[valid_pixels]
    + 0.5 * (colored_mask[valid_pixels] / 255.0)
)


# -----------------------------
# Display
# -----------------------------

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(image_display)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(colored_mask)
plt.title("Segmentation Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()