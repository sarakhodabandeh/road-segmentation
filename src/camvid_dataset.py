from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .camvid_classes import CLASS_COLORS, IGNORE_INDEX, IGNORE_COLOR


class CamVidDataset(Dataset):
    """
    PyTorch Dataset for the CamVid semantic segmentation dataset.
    """

    # PIL uses (width, height)
    TARGET_SIZE = (480, 360)

    def __init__(self, image_dir, mask_dir):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.image_paths = sorted(self.image_dir.glob("*.png"))
        self.mask_paths = sorted(self.mask_dir.glob("*_L.png"))

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

        if len(self.image_paths) != len(self.mask_paths):
            raise RuntimeError(
                f"Number of images ({len(self.image_paths)}) "
                f"does not match number of masks ({len(self.mask_paths)})"
            )

    def __len__(self):
        return len(self.image_paths)

    def rgb_to_class(self, mask):
        """
        Convert an RGB CamVid mask into a class-ID mask.

        Unknown / Void pixels become IGNORE_INDEX.
        """

        mask = np.array(mask)

        class_mask = np.full(
            mask.shape[:2],
            IGNORE_INDEX,
            dtype=np.uint8
        )

        for color, class_id in CLASS_COLORS.items():
            matches = np.all(mask == color, axis=-1)
            class_mask[matches] = class_id

        # Explicitly ignore Void pixels
        void_matches = np.all(mask == IGNORE_COLOR, axis=-1)
        class_mask[void_matches] = IGNORE_INDEX

        return class_mask

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        # Resize image with bilinear interpolation
        image = image.resize(
            self.TARGET_SIZE,
            Image.Resampling.BILINEAR
        )

        # Resize mask with NEAREST interpolation
        # This preserves the original RGB class colors.
        mask = mask.resize(
            self.TARGET_SIZE,
            Image.Resampling.NEAREST
        )

        # -----------------------------
        # Image → Tensor
        # -----------------------------

        image = np.array(image)

        image = torch.from_numpy(
            image
        ).permute(2, 0, 1).float()

        # Normalize from [0, 255] → [0, 1]
        image = image / 255.0

        # -----------------------------
        # Mask RGB → Class IDs
        # -----------------------------

        mask = self.rgb_to_class(mask)

        mask = torch.from_numpy(mask).long()

        return image, mask