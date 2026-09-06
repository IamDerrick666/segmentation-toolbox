<div align="center">

# Segmentation Toolbox

**A lightweight PyTorch research template for 2D binary image segmentation experiments.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

**Segmentation Toolbox** is a compact training and experiment-management template developed for image segmentation research.

It integrates the main components of a typical segmentation workflow, including dataset loading, augmentation, K-fold cross-validation, model training, evaluation, checkpoint handling, prediction export, and experiment logging.

The toolbox is designed to simplify repeated segmentation experiments while keeping the model and dataset components easy to replace.

---

## Features

- **K-fold cross-validation** with `scikit-learn`
- **Data augmentation** with Albumentations
- **Mixed-precision training** with PyTorch AMP
- **Adam optimization**
- Optional **ReduceLROnPlateau** learning-rate scheduling
- **Checkpoint saving and loading**
- Automatic export of segmentation predictions
- CSV-based experiment logging
- Support for common segmentation metrics:
  - IoU
  - Dice
  - Sensitivity
  - Specificity
  - Precision
  - AUC
  - Accuracy

---

## Workflow

```text
Dataset
   │
   ▼
Data loading & augmentation
   │
   ▼
K-fold cross-validation
   │
   ▼
Model training
   │
   ▼
Validation & evaluation
   │
   ▼
Prediction export
   │
   ▼
Experiment logging
````

---

## Repository Structure

```text
segmentation-toolbox/
├── README.md
├── LICENSE
├── ToolBox_SEG.zip
└── ToolBox_SEG/
    ├── Project_Dataset/
    │   ├── Dataset/
    │   │   └── [dataset example]
    │   │
    │   └── Project (Reproduction only, please do not modify)/
    │       ├── train.py
    │       └── utils/
    │           ├── dataset.py
    │           └── utils.py
    │
    └── RecordData/
        └── [experiment outputs]
```

The current repository structure is retained for compatibility with the original implementation.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/IamDerrick666/segmentation-toolbox.git
cd segmentation-toolbox
```

### 2. Install dependencies

The current implementation relies on the following main packages:

```bash
pip install torch torchvision numpy scikit-learn albumentations pandas tqdm pillow
```

Package versions should be selected according to your local Python and CUDA environment.

---

## Dataset Preparation

The default dataset loader expects paired images and binary masks.

The current implementation assumes:

* images are stored as `.tif`
* masks are stored as `.png`
* image and mask pairs share the same base filename

A typical dataset layout is:

```text
Project_Dataset/
├── YOUR_DATASET/
│   ├── images/
│   │   ├── sample_001.tif
│   │   ├── sample_002.tif
│   │   └── ...
│   │
│   └── masks/
│       ├── sample_001.png
│       ├── sample_002.png
│       └── ...
│
└── Project (Reproduction only, please do not modify)/
```

Then update the dataset settings in `train.py`:

```python
DATASET = "YOUR_DATASET"

IMG_DIR = "../YOUR_DATASET/images"
MASK_DIR = "../YOUR_DATASET/masks"
```

---

## Add Your Model

Place your segmentation model under a local `models/` directory and update the corresponding import in `train.py`.

For example:

```python
from models.unet import UNet

MODEL = UNet().to(DEVICE)
```

The model should accept image tensors as input and return segmentation logits compatible with the selected loss function.

---

## Experiment Configuration

The main experimental settings are defined near the beginning of `train.py`.

```python
NUM_FOLD = 5
NUM_EPOCHS = 100

BATCH_SIZE = 4
NUM_WORKERS = 0

DEFAULT_LEARNING_RATE = 1e-4
DYNAMIC_LR = True

IMG_HEIGHT = 224
IMG_WIDTH = 224

SAVE_IMG = True
SAVE_MODEL = False
LOAD_MODEL = False

USE_AMP = True

DEVICE = "cuda:0"
```

These values can be adjusted according to the dataset, model, and available hardware.

---

## Training

Run the training script from the project directory:

```bash
cd "ToolBox_SEG/Project_Dataset/Project (Reproduction only, please do not modify)"
python train.py
```

By default, the framework performs **5-fold cross-validation**.

Each fold independently trains and evaluates a copy of the selected model.

---

## Evaluation

The current evaluation pipeline reports:

| Metric          | Supported |
| --------------- | :-------: |
| IoU             |     ✓     |
| Dice            |     ✓     |
| Sensitivity     |     ✓     |
| Specificity     |     ✓     |
| Precision       |     ✓     |
| AUC             |     ✓     |
| Accuracy        |     ✓     |
| Training Loss   |     ✓     |
| Validation Loss |     ✓     |

Evaluation results from each epoch are collected and exported to CSV.

---

## Experiment Outputs

Experiment results are stored under:

```text
ToolBox_SEG/RecordData/
```

For each model–dataset combination, the toolbox can generate:

```text
MODEL_DATASET/
├── MODEL_DATASET.csv
├── saved_images0/
├── saved_images1/
├── saved_images2/
├── saved_images3/
└── saved_images4/
```

The saved prediction folders correspond to the individual cross-validation folds.

Depending on the configuration, the toolbox can export:

* input images
* ground-truth masks
* predicted masks
* evaluation metrics
* training loss
* validation loss
* model checkpoints

---

## Current Scope

The current implementation is primarily intended for:

* **2D image segmentation**
* **binary segmentation**
* paired image-mask datasets
* research experiments and model benchmarking

This repository is a **research template rather than a packaged software library**.

Users may need to modify the dataset loader, loss function, evaluation metrics, or model interface for multi-class segmentation, volumetric imaging, or other data formats.

---

## Authors

**Yuquan Xu**
**Ronghui Feng**

---

## License

This project is released under the [MIT License](LICENSE).

````
