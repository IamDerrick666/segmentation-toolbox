<div align="center">

# ToolBox_SEG

**A lightweight research toolbox for training and evaluating binary image segmentation models with PyTorch.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

**ToolBox_SEG** is a compact training and experiment-management template developed for image segmentation research. It brings together dataset loading, augmentation, cross-validation, model training, evaluation, checkpoint handling, prediction export, and experiment logging in a single workflow.

The toolbox is intended to make repeated segmentation experiments easier to organize while keeping the model and dataset components easy to replace.

## Key Features

- **K-fold cross-validation** using `sklearn.model_selection.KFold` (5 folds by default).
- **Data augmentation** with Albumentations, including resizing, flipping, rotation, normalization, and tensor conversion.
- **Mixed-precision training** through PyTorch AMP.
- **Flexible optimization** with Adam and an optional `ReduceLROnPlateau` learning-rate scheduler.
- **Segmentation evaluation** with IoU, Dice, sensitivity, specificity, precision, AUC, and accuracy.
- **Experiment logging** of evaluation metrics, training loss, and validation loss to CSV files.
- **Prediction export** for input images, ground-truth masks, and predicted masks.
- **Checkpoint support** for saving and loading model states.

## Workflow

```text
Dataset preparation
        ↓
Data loading & augmentation
        ↓
K-fold split
        ↓
Model training
        ↓
Validation & metric calculation
        ↓
Prediction export
        ↓
CSV experiment record
