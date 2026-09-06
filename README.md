<div align="center">

# Segmentation Toolbox

**A lightweight PyTorch research template for 2D binary image segmentation experiments.**

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

## Overview

Segmentation Toolbox is a compact research template for training and comparing PyTorch models on paired 2D image-mask datasets. It keeps experiment configuration in a single training script and provides dataset loading, augmentation, cross-validation, evaluation, prediction export, checkpoint handling, and CSV logging.

The repository intentionally does not include a dataset or model implementation. Add these components for the experiment you want to run.

## Features

- K-fold cross-validation with scikit-learn
- Albumentations-based resizing, normalization, flips, and rotation augmentation
- PyTorch AMP mixed-precision training
- Adam optimizer
- Optional `ReduceLROnPlateau` learning-rate scheduler
- Checkpoint save/load support
- Prediction, input-image, and ground-truth-mask export
- CSV experiment logging
- IoU, Dice, sensitivity, specificity, precision, AUC, accuracy, training loss, and validation loss reporting

## Workflow

```text
Paired images and masks
        |
        v
Loading and augmentation
        |
        v
K-fold training and validation
        |
        v
Metrics, predictions, and CSV logs
```

## Repository Structure

```text
segmentation-toolbox/
├── README.md
├── LICENSE
├── .gitignore
├── train.py
├── utils/
│   ├── dataset.py
│   └── utils.py
├── models/
│   └── README.md
├── data/
│   └── README.md
└── outputs/
    └── .gitkeep
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/IamDerrick666/segmentation-toolbox.git
cd segmentation-toolbox
```

Install the libraries imported by the current code:

```bash
pip install torch torchvision numpy scikit-learn albumentations pandas tqdm pillow
```

Choose package builds and versions that match your Python, hardware, and CUDA environment.

## Dataset Preparation

Datasets are not distributed with this repository. The current loader expects paired 2D images and binary masks under `data/DATASET_NAME/`:

```text
data/
└── DATASET_NAME/
    ├── images/
    │   ├── sample_001.tif
    │   └── ...
    └── masks/
        ├── sample_001.png
        └── ...
```

Image-mask pairs must have matching base filenames. The implemented filename mapping replaces the `.tif` image suffix with `.png` for the mask. See [data/README.md](data/README.md) for the exact loader assumptions.

## Add Your Model

Place your segmentation model implementation under `models/`, then update the template import and model construction in `train.py`. For example:

```python
from models.model_name import model_name

MODEL = model_name().to(DEVICE)
```

The model must accept image tensors and return segmentation logits compatible with the configured `BCEWithLogitsLoss`. No model is bundled with the repository. See [models/README.md](models/README.md).

## Experiment Configuration

Edit the constants near the top of `train.py` before running an experiment:

```python
NUM_FOLD = 5
NUM_EPOCHS = 100
BATCH_SIZE = 4
NUM_WORKERS = 0
DEFAULT_LEARNING_RATE = 1e-4
DYNAMIC_LR = True
IMG_HEIGHT = 224
IMG_WIDTH = 224
DATASET = "DATASET_NAME"
IMG_DIR = "data/DATASET_NAME/images"
MASK_DIR = "data/DATASET_NAME/masks"
SAVE_IMG = True
SAVE_MODEL = False
LOAD_MODEL = False
USE_AMP = True
DEVICE = "cuda:0"
MODEL_NAME = "MODEL_NAME"
```

The current augmentation policy, fold settings, loss, optimizer, metrics, and experiment loop are research-code defaults; review them for your own study without assuming they are universally appropriate.

## Training

After adding a dataset and model, run from the repository root:

```bash
python train.py
```

The script trains one copied model per fold and evaluates it after every epoch. Training batches follow the configured `DROP_LAST` setting, while validation always uses `drop_last=False` so every validation sample is evaluated.

## Evaluation

The validation loop reports IoU, Dice, sensitivity, specificity, precision, AUC, and accuracy. Specificity is computed from validation-set true-negative and false-positive counts, and AUC is computed once from prediction scores aggregated across the full validation loader. If a validation target contains only one class, AUC is recorded as `NaN`. Each epoch also records the mean training loss and validation loss. All epoch records across folds are written to one CSV file after training completes.

## Experiment Outputs

For a configured model and dataset, the script creates:

```text
outputs/
└── MODEL_DATASET/
    ├── MODEL_DATASET.csv
    ├── saved_images0/
    ├── saved_images1/
    ├── saved_images2/
    ├── saved_images3/
    └── saved_images4/
```

With `SAVE_IMG = True`, each fold directory receives predicted masks, ground-truth masks, and input images. The result directory is created before training and the script exits if the same `MODEL_DATASET` directory already exists, preventing silent overwrite.

With `SAVE_MODEL = True`, checkpoints are saved in the repository root as `MODEL_DATASET_foldN.pth.tar` when validation loss improves. With `LOAD_MODEL = True`, the current code loads `checkpoint.pth.tar` from the repository root.

## Current Scope

This repository is a lightweight research template primarily intended for:

- 2D image segmentation
- Binary segmentation
- Paired image-mask datasets
- Research experiments and model benchmarking

It is not a packaged or production-oriented segmentation library. Multi-class segmentation, volumetric data, alternative filename mappings, and other task-specific behavior require deliberate changes to the current code.

## Authors

**Yuquan Xu**<br>
**Ronghui Feng**

Chengdu University

## License

This project is available under the [MIT License](LICENSE).
