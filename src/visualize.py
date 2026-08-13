import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from .camvid_dataset import CamVidDataset
from .unet import UNet
from .camvid_classes import NUM_CLASSES


# --------------------------------
# Configuration
# --------------------------------

TEST_IMAGE_DIR = "data/processed/test/images"
TEST_MASK_DIR = "data/processed/test/masks"

MODEL_PATH = "checkpoints/best_model.pth"

OUTPUT_DIR = "outputs/visualizations"

NUM_SAMPLES = 5


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

    # Add batch dimension
    input_tensor = image.unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():

        output = model(input_tensor)

        prediction = torch.argmax(
            output,
            dim=1
        )

    prediction = prediction.squeeze(0).cpu().numpy()
    mask = mask.numpy()

    # Convert image from CHW to HWC
    image_np = image.permute(
        1, 2, 0
    ).numpy()

    # --------------------------------
    # Create figure
    # --------------------------------

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Ground truth
    axes[1].imshow(mask)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Prediction
    axes[2].imshow(prediction)
    axes[2].set_title("U-Net Prediction")
    axes[2].axis("off")

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"sample_{index + 1}.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


print("\nVisualization completed!")
