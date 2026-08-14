# CamVid Semantic Segmentation with U-Net

A PyTorch-based computer vision project for **semantic segmentation of road scenes** using the **CamVid dataset** and a custom **U-Net architecture**.

The model performs pixel-level classification and assigns each image pixel to one of **11 semantic classes**, including road, sky, building, car, pedestrian, and bicyclist.

---

## Key Results

| Metric                       |     Result |
| ---------------------------- | ---------: |
| **Test Pixel Accuracy**      | **90.35%** |
| **Test Mean IoU**            | **57.18%** |
| **Test Dice Score**          | **68.24%** |
| **Best Validation Mean IoU** | **52.74%** |
| **Best Class — Road IoU**    | **94.29%** |
| **Training Images**          |    **490** |
| **Validation Images**        |    **105** |
| **Test Images**              |    **106** |
| **Semantic Classes**         |     **11** |
| **Training Epochs**          |     **20** |

The final model was evaluated on a held-out test set of 106 images. The best checkpoint was selected based on validation Mean IoU.

---

## Project Overview

Semantic segmentation is a computer vision task in which every pixel of an image is assigned a semantic class.

This project implements a complete semantic segmentation pipeline using a U-Net model trained on road-scene images from the CamVid dataset.

The project covers:

* Dataset preparation and organization
* Train/validation/test splitting
* RGB segmentation-mask processing
* Conversion of RGB masks into class IDs
* PyTorch Dataset and DataLoader implementation
* U-Net architecture implementation
* Training with Apple Silicon MPS acceleration
* Model evaluation using multiple metrics
* Per-class IoU analysis
* Qualitative prediction visualization
* Training-curve visualization

---

## Dataset

The project uses the **CamVid road-scene dataset**.

The dataset labels used in this project are grouped into 11 semantic classes:

| ID | Class      |
| -: | ---------- |
|  0 | Sky        |
|  1 | Building   |
|  2 | Pole       |
|  3 | Road       |
|  4 | Pavement   |
|  5 | Tree       |
|  6 | SignSymbol |
|  7 | Fence      |
|  8 | Car        |
|  9 | Pedestrian |
| 10 | Bicyclist  |

Pixels that do not correspond to a known class are assigned an **ignore index of `255`** and are excluded from the relevant training/evaluation calculations.

### Dataset Split

| Split      | Number of Images |
| ---------- | ---------------: |
| Training   |              490 |
| Validation |              105 |
| Test       |              106 |
| **Total**  |          **701** |

The test set is kept separate from training and validation and is used only for final evaluation.

---

## Preprocessing

All images and segmentation masks are resized to:

```text
Height: 360
Width: 480
```

### Input Images

The preprocessing pipeline:

1. Loads the image as RGB
2. Resizes the image to `480 × 360`
3. Converts it to a PyTorch tensor
4. Changes the tensor layout to `C × H × W`
5. Converts pixel values from `[0, 255]` to `[0, 1]`

The resulting image tensor has the shape:

```text
3 × 360 × 480
```

### Segmentation Masks

Segmentation masks are:

1. Loaded as RGB images
2. Resized using **nearest-neighbor interpolation**
3. Converted from RGB colors to integer class IDs
4. Assigned `255` for ignored/unknown pixels

Nearest-neighbor interpolation is used for masks to prevent interpolation from creating invalid class colors.

---

## Model

The project uses a custom **U-Net** architecture implemented with PyTorch.

U-Net is an encoder-decoder architecture designed for pixel-level image segmentation. The network combines downsampling and upsampling paths with skip connections to preserve spatial information.

### Model Input

```text
Batch × 3 × 360 × 480
```

### Model Output

```text
Batch × 11 × 360 × 480
```

The 11 output channels correspond to the 11 semantic classes.

For each pixel, the predicted class is obtained by selecting the class with the highest output score.

---

## Training

The model was trained for **20 epochs** using the **Apple Metal Performance Shaders (MPS)** backend available on Apple Silicon.

### Training Configuration

| Parameter          | Value     |
| ------------------ | --------- |
| Framework          | PyTorch   |
| Model              | U-Net     |
| Device             | Apple MPS |
| Epochs             | 20        |
| Batch Size         | 4         |
| Input Resolution   | 360 × 480 |
| Number of Classes  | 11        |
| Training Samples   | 490       |
| Validation Samples | 105       |

During training, the model was evaluated on the validation set after each epoch.

The checkpoint with the highest validation Mean IoU was saved as the best model.

### Best Validation Result

```text
Best Validation Mean IoU: 0.5274
```

The best checkpoint was obtained at **epoch 20**.

---

## Test Results

After training was completed, the best checkpoint was evaluated on the held-out test set containing **106 images**.

### Final Test Performance

| Metric         |      Score |
| -------------- | ---------: |
| Pixel Accuracy | **90.35%** |
| Mean IoU       | **57.18%** |
| Dice Score     | **68.24%** |

### Metric Interpretation

**Pixel Accuracy** measures the percentage of valid pixels that were classified correctly.

**Mean Intersection over Union (Mean IoU)** measures the average overlap between predicted and ground-truth regions across the semantic classes.

**Dice Score** measures the similarity between predicted and ground-truth segmentation regions.

Mean IoU is particularly useful for semantic segmentation because it provides a more balanced view of class-level segmentation performance than pixel accuracy alone.

---

## Per-Class Performance

The model was also evaluated separately for each semantic class.

| Class      |        IoU |
| ---------- | ---------: |
| Sky        | **92.20%** |
| Building   | **82.09%** |
| Pole       |     14.55% |
| Road       | **94.29%** |
| Pavement   | **78.50%** |
| Tree       | **74.96%** |
| SignSymbol |     32.10% |
| Fence      |     39.74% |
| Car        | **68.56%** |
| Pedestrian |     19.17% |
| Bicyclist  |     32.83% |

### Observations

The strongest performance was obtained for large and visually distinctive regions:

* Road — **94.29%**
* Sky — **92.20%**
* Building — **82.09%**
* Pavement — **78.50%**
* Tree — **74.96%**

The model had more difficulty with smaller or thin objects:

* Pole — **14.55%**
* Pedestrian — **19.17%**
* SignSymbol — **32.10%**
* Bicyclist — **32.83%**

This behavior is expected in semantic segmentation because small objects occupy fewer pixels and can be difficult to distinguish from surrounding regions.

---

## Visual Results

Qualitative predictions were generated using images from the test set.

The visualization pipeline produces comparisons between:

* Original image
* Ground-truth segmentation
* U-Net prediction
* Prediction overlay

The generated visualizations are stored in:

```text
outputs/visualizations/
```

Example files:

```text
sample_1.png
sample_2.png
sample_3.png
sample_4.png
sample_5.png
```

Additional class-color visualizations are stored in:

```text
outputs/class_visualizations/
```

---

## Training Curves

The training process is also visualized using the generated learning curves.

### Training Loss

```text
outputs/training_loss.png
```

### Validation Mean IoU

```text
outputs/validation_miou.png
```

These plots show how the model's training loss and validation segmentation performance changed throughout the 20 training epochs.

---

## Project Structure

```text
road-segmentation/
│
├── notebooks/
│   └── 01_explore_camvid.ipynb
│
├── outputs/
│   ├── class_visualizations/
│   │   ├── sample_1.png
│   │   ├── sample_2.png
│   │   ├── sample_3.png
│   │   ├── sample_4.png
│   │   └── sample_5.png
│   │
│   ├── visualizations/
│   │   ├── sample_1.png
│   │   ├── sample_2.png
│   │   ├── sample_3.png
│   │   ├── sample_4.png
│   │   └── sample_5.png
│   │
│   ├── training_loss.png
│   └── validation_miou.png
│
├── src/
│   ├── __init__.py
│   ├── camvid_classes.py
│   ├── camvid_dataset.py
│   ├── metrics.py
│   ├── plot_training.py
│   ├── split_dataset.py
│   ├── train.py
│   ├── unet.py
│   ├── visualize.py
│   ├── visualize_classes.py
│   ├── test.py
│   ├── test_classes.py
│   ├── test_dataloader.py
│   ├── test_dataset.py
│   ├── test_metrics.py
│   ├── test_model.py
│   └── test_per_class.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

The raw and processed dataset files and trained model checkpoints are excluded from version control.

---

## Installation

### 1. Clone the repository

```bash
git clone git@github.com:sarakhodabandeh/road-segmentation.git
cd road-segmentation
```

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
```

### 3. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

The CamVid dataset is not included in this repository.

After obtaining the dataset, organize the processed files according to the following structure:

```text
data/
├── raw/
│
└── processed/
    ├── train/
    │   ├── images/
    │   └── masks/
    │
    ├── val/
    │   ├── images/
    │   └── masks/
    │
    └── test/
        ├── images/
        └── masks/
```

The processed dataset should contain the following number of samples:

```text
Train:      490
Validation: 105
Test:       106
Total:      701
```

---

## Running the Project

### Test Dataset

```bash
python -m src.test_dataset
```

### Test DataLoader

```bash
python -m src.test_dataloader
```

### Test U-Net Model

```bash
python -m src.test_model
```

### Train the Model

```bash
python -m src.train
```

### Evaluate the Model

```bash
python -m src.test_metrics
```

### Evaluate Per-Class IoU

```bash
python -m src.test_per_class
```

### Generate Segmentation Visualizations

```bash
python -m src.visualize
```

### Generate Training Curves

```bash
python -m src.plot_training
```

---

## Future Improvements

Several improvements could be explored in future versions of the project:

* Data augmentation
* Class-weighted loss
* Dice loss or combined Cross-Entropy + Dice loss
* Learning-rate scheduling
* Transfer learning with a pretrained encoder
* More advanced segmentation architectures
* Higher-resolution training
* Improved handling of small objects
* Additional quantitative analysis
* More extensive qualitative evaluation

---

## Technologies

* **Python 3.11**
* **PyTorch**
* **NumPy**
* **Pillow**
* **Matplotlib**
* **Apple Metal Performance Shaders (MPS)**

---

## Author

### Sara Khodabandeh Yalabadi

**Computer Science**

Interested in **Artificial Intelligence, Deep Learning, and Computer Vision**.

This project was developed as a computer vision portfolio project focused on **semantic segmentation using deep learning**.
