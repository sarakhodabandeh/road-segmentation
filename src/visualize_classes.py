import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .camvid_dataset import CamVidDataset
from .unet import UNet
from .camvid_classes import CLASS_NAMES, NUM_CLASSES


# --------------------------------
# Configuration
# --------------------------------

TEST_IMAGE_DIR = "data/processed/test/images"
TEST_MASK_DIR = "data/processed/test/masks"

MODEL_PATH = "checkpoints/best_model.pth"

OUTPUT_DIR = "outputs/class_visualizations"

NUM_SAMPLES = 5


# --------------------------------
# Unique display colors
# --------------------------------

CLASS_DISPLAY_COLORS = [
    "#87CEEB",  # 0 Sky
    "#8B4513",  # 1 Building
    "#FFD700",  # 2 Pole
    "#444444",  # 3 Road
    "#A9A9A9",  # 4 Pavement
    "#228B22",  # 5 Tree
    "#FF69B4",  # 6 SignSymbol
    "#D2691E",  # 7 Fence
    "#1E90FF",  # 8 Car
    "#FF0000",  # 9 Pedestrian
    "#800080",  # 10 Bicyclist
]

cmap = ListedColormap(CLASS_DISPLAY_COLORS)


# --------------------------------
# Device
# --------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)


# --------------------------------
# Dataset
# --------------------------------

dataset = CamVidDataset(
    image_dir=TEST_IMAGE_DIR,
    mask_dir=TEST_MASK_DIR
)

print("Test samples:", len(dataset))


# --------------------------------
# Model
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
    "Loaded model from epoch:",
    checkpoint["epoch"]
)


# --------------------------------
# Output directory
# --------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# --------------------------------
# Visualization
# --------------------------------

for index in range(NUM_SAMPLES):

    image, mask = dataset[index]

    # -----------------------------
    # Prediction
    # -----------------------------

    input_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(input_tensor)

        prediction = torch.argmax(
            output,
            dim=1
        )

    prediction = prediction.squeeze(0).cpu().numpy()
    mask = mask.numpy()

    # -----------------------------
    # Original image
    # -----------------------------

    image_np = image.permute(
        1, 2, 0
    ).numpy()

    image_np = np.clip(
        image_np,
        0,
        1
    )

    # -----------------------------
    # Ignore pixels
    # -----------------------------

    mask_display = mask.astype(float)
    prediction_display = prediction.astype(float)

    mask_display[mask == 255] = np.nan
    prediction_display[prediction == 255] = np.nan

    # -----------------------------
    # Figure
    # -----------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    # Original
    axes[0].imshow(image_np)

    axes[0].set_title(
        "Original Image",
        fontsize=14
    )

    axes[0].axis("off")


    # Ground Truth
    axes[1].imshow(
        mask_display,
        cmap=cmap,
        vmin=0,
        vmax=NUM_CLASSES - 1
    )

    axes[1].set_title(
        "Ground Truth",
        fontsize=14
    )

    axes[1].axis("off")


    # Prediction
    axes[2].imshow(
        prediction_display,
        cmap=cmap,
        vmin=0,
        vmax=NUM_CLASSES - 1
    )

    axes[2].set_title(
        "U-Net Prediction",
        fontsize=14
    )

    axes[2].axis("off")


    # Overlay
    axes[3].imshow(image_np)

    axes[3].imshow(
        prediction_display,
        cmap=cmap,
        vmin=0,
        vmax=NUM_CLASSES - 1,
        alpha=0.45
    )

    axes[3].set_title(
        "Prediction Overlay",
        fontsize=14
    )

    axes[3].axis("off")


    # --------------------------------
    # Legend
    # --------------------------------

    handles = []

    for class_id, class_name in enumerate(CLASS_NAMES):

        handle = plt.Line2D(
            [0],
            [0],
            marker="s",
            color="white",
            markerfacecolor=CLASS_DISPLAY_COLORS[class_id],
            markersize=10,
            label=f"{class_id}: {class_name}"
        )

        handles.append(handle)


    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.05),
        fontsize=9
    )


    plt.tight_layout()


    # --------------------------------
    # Save
    # --------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"sample_{index + 1}.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


    print(
        f"Saved: {output_path}"
    )


print("\nClass visualization completed!")