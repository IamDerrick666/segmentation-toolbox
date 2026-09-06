# Model Implementations

Place your PyTorch segmentation model implementation in this directory. No model is distributed with the repository.

For example:

```text
models/
├── model_name.py
└── ...
```

Then update the import and model construction in `train.py`:

```python
from models.model_name import model_name

MODEL = model_name().to(DEVICE)
```

The model must accept image tensors and return segmentation logits compatible with the configured `BCEWithLogitsLoss`.
