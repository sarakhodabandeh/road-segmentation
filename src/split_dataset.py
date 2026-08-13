from pathlib import Path
import random
import shutil


# -----------------------------
# Configuration
# -----------------------------

SOURCE_IMAGE_DIR = Path("data/raw/701_StillsRaw_full")
SOURCE_MASK_DIR = Path("data/raw")

OUTPUT_DIR = Path("data/processed")

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# -----------------------------
# Create output directories
# -----------------------------

for split in ["train", "val", "test"]:
    (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / split / "masks").mkdir(parents=True, exist_ok=True)


# -----------------------------
# Find image-mask pairs
# -----------------------------

image_paths = sorted(SOURCE_IMAGE_DIR.glob("*.png"))

pairs = []

for image_path in image_paths:

    mask_path = SOURCE_MASK_DIR / f"{image_path.stem}_L.png"

    if mask_path.exists():
        pairs.append((image_path, mask_path))
    else:
        print(f"Warning: mask not found for {image_path.name}")


print("Total valid image-mask pairs:", len(pairs))


# -----------------------------
# Shuffle reproducibly
# -----------------------------

random.seed(SEED)
random.shuffle(pairs)


# -----------------------------
# Calculate split sizes
# -----------------------------

total = len(pairs)

train_size = int(total * TRAIN_RATIO)
val_size = int(total * VAL_RATIO)

train_pairs = pairs[:train_size]
val_pairs = pairs[train_size:train_size + val_size]
test_pairs = pairs[train_size + val_size:]


print("Train:", len(train_pairs))
print("Validation:", len(val_pairs))
print("Test:", len(test_pairs))


# -----------------------------
# Copy files
# -----------------------------

def copy_pairs(pairs, split):

    image_output = OUTPUT_DIR / split / "images"
    mask_output = OUTPUT_DIR / split / "masks"

    for image_path, mask_path in pairs:

        shutil.copy2(
            image_path,
            image_output / image_path.name
        )

        shutil.copy2(
            mask_path,
            mask_output / mask_path.name
        )


copy_pairs(train_pairs, "train")
copy_pairs(val_pairs, "val")
copy_pairs(test_pairs, "test")


print("\nDataset split completed successfully!")