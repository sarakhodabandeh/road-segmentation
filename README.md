# CamVid Semantic Segmentation with U-Net

A deep learning project for semantic segmentation of road scenes using the CamVid dataset and a U-Net architecture implemented with PyTorch.

The model classifies every pixel of an image into one of 11 semantic classes such as road, sky, building, car, pedestrian, and bicyclist.

---

## Project Overview

Semantic segmentation assigns a class label to every pixel in an image.

In this project, a U-Net model is trained to perform semantic segmentation on road-scene images from the CamVid dataset.

### Main objectives

- Build a complete semantic segmentation pipeline
- Preprocess and organize the CamVid dataset
- Convert RGB segmentation masks into class IDs
- Train a U-Net model using PyTorch
- Use Apple Silicon MPS acceleration
- Evaluate the model using multiple segmentation metrics
- Analyze performance for individual classes
- Visualize segmentation predictions

---

## Dataset

The project uses the CamVid road-scene dataset.

The original CamVid labels are grouped into 11 semantic classes:

| ID | Class |
|---:|---|
| 0 | Sky |
| 1 | Building |
| 2 | Pole |
| 3 | Road |
| 4 | Pavement |
| 5 | Tree |
| 6 | SignSymbol |
| 7 | Fence |
| 8 | Car |
| 9 | Pedestrian |
| 10 | Bicyclist |

Void / unknown pixels are ignored during training and evaluation using an ignore index of `255`.

### Dataset split

| Split | Images |
|---|---:|
| Training | 490 |
| Validation | 105 |
| Test | 106 |
| **Total** | **701** |

---

## Preprocessing

Images and masks are resized to:

```text
360 × 480
Images are:

converted to RGB
converted to PyTorch tensors
normalized from [0, 255] to [0, 1]

Segmentation masks are resized using nearest-neighbor interpolation to preserve class labels.

RGB mask colors are converted into integer class IDs.

Model

The project uses a U-Net architecture for semantic segmentation.

Input
3 × 360 × 480
Output
11 × 360 × 480

Each output channel corresponds to one semantic class.

The predicted class for each pixel is obtained using argmax over the class dimension.
Training

The model was trained for 20 epochs.

Training was performed using Apple's Metal Performance Shaders (MPS) backend.

Training configuration
Framework: PyTorch
Device: Apple MPS
Epochs: 20
Batch size: 4
Input size: 360 × 480
Number of classes: 11

The best model checkpoint was selected based on validation Mean IoU.

Best validation Mean IoU:

0.5274

The best checkpoint is stored at:

checkpoints/best_model.pth
Results

The final model was evaluated on the held-out test set containing 106 images.

Metric	Score
Pixel Accuracy	90.35%
Mean IoU	57.18%
Dice Score	68.24%
Interpretation

The model performs particularly well on large and visually distinct regions such as roads, sky, buildings, pavement, and trees.

Smaller objects such as poles, pedestrians, signs, and bicyclists remain more challenging.

Per-Class IoU
Class	IoU
Sky	92.20%
Building	82.09%
Pole	14.55%
Road	94.29%
Pavement	78.50%
Tree	74.96%
SignSymbol	32.10%
Fence	39.74%
Car	68.56%
Pedestrian	19.17%
Bicyclist	32.83%

The strongest class is Road, with an IoU of 94.29%.

The most challenging classes are Pole and Pedestrian, with IoUs of 14.55% and 19.17%, respectively.

Visualizations

The project includes qualitative segmentation results generated from the test set.

Generated visualizations are stored in:

outputs/visualizations/

Example files:

sample_1.png
sample_2.png
sample_3.png
sample_4.png
sample_5.png

Each visualization compares the input road scene with the segmentation result.

Training Curves

Training and validation loss:

outputs/training_loss.png

Validation Mean IoU:

outputs/validation_miou.png

These curves show the progression of model training over 20 epochs.

Project Structure
road-segmentation/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── train/
│       │   ├── images/
│       │   └── masks/
│       ├── val/
│       │   ├── images/
│       │   └── masks/
│       └── test/
│           ├── images/
│           └── masks/
│
├── checkpoints/
│   └── best_model.pth
│
├── outputs/
│   ├── training_loss.png
│   ├── validation_miou.png
│   └── visualizations/
│
├── src/
│   ├── camvid_classes.py
│   ├── camvid_dataset.py
│   ├── unet.py
│   ├── train.py
│   ├── metrics.py
│   ├── test_model.py
│   ├── test_dataset.py
│   ├── test_metrics.py
│   ├── test_per_class.py
│   ├── visualize.py
│   └── plot_training.py
│
└── README.md
Installation

Create a Python virtual environment:

python3.11 -m venv .venv

Activate it:

source .venv/bin/activate

Install the required packages:

pip install torch torchvision numpy pillow matplotlib
Running the Project
Test the dataset
python -m src.test_dataset
Test the DataLoader
python -m src.test_dataloader
Test the model
python -m src.test_model
Train the model
python -m src.train
Evaluate the model
python -m src.test_metrics
Calculate per-class IoU
python -m src.test_per_class
Generate visualizations
python -m src.visualize
Generate training curves
python -m src.plot_training
Future Improvements

Possible improvements include:

Data augmentation
Class-weighted loss
Dice loss or combined Cross-Entropy + Dice loss
A deeper or more efficient segmentation architecture
Transfer learning with a pretrained encoder
Improved handling of small objects
Higher-resolution training
Learning-rate scheduling
Additional qualitative analysis
Technologies
Python 3.11
PyTorch
NumPy
Pillow
Matplotlib
Apple Metal Performance Shaders (MPS)
Author

Developed as a computer vision and deep learning project focused on semantic segmentation.
