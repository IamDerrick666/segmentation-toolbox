# Dataset Layout

Datasets are not distributed with this repository. Place each paired 2D image and binary-mask dataset under this directory:

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

The current `DatasetInit` loader reads every filename in `images/` and expects a corresponding mask with the same base filename. It specifically maps an image filename ending in `.tif` to a mask filename ending in `.png`. Images are converted to RGB, masks are converted to grayscale, and mask values above 128 are mapped to 1 while all other values are mapped to 0.

Keep only compatible image files in the `images/` directory, or adapt `utils/dataset.py` for a different filename or format convention.
