# Implementation Tutorial

Step-by-step guide for running experiments and reproducing results.

## Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- NVIDIA GPU optional but recommended for faster training

## Environment Setup

### Using uv (recommended)

```bash
git clone <repository-url>
cd waste-sorting-ai
pip install uv
uv sync
```

### Using pip

```bash
pip install -r requirements.txt
```

### GPU Support

For CUDA-enabled GPU:
```bash
pip install tensorflow[and-cuda]
```

Verify GPU detection:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

## Dataset Preparation

### Option 1: Automatic Download

```bash
uv run download_data.py
```

This downloads TrashNet and creates train/val/test splits automatically.

### Option 2: Manual Setup

1. Download TrashNet from https://github.com/garythung/trashnet
2. Extract archive
3. Organize into directory structure:

```
data/
├── train/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

Recommended split: 70% train, 15% validation, 15% test

### Verify Dataset

Check image counts:
```bash
find data/train -name "*.jpg" | wc -l
find data/val -name "*.jpg" | wc -l  
find data/test -name "*.jpg" | wc -l
```

Expected: ~1700 train, ~400 val, ~400 test

## Running Experiments

### Quick Test

For rapid prototyping (reduces training time):

Edit `model_comparison.py`:
```python
EPOCHS = 5
BATCH_SIZE = 16
```

Run:
```bash
uv run model_comparison.py
```

Completes in 10-20 minutes. Results will be suboptimal but useful for debugging.

### Full Training

Default configuration:
```python
EPOCHS = 20
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
```

Run:
```bash
uv run model_comparison.py
```

Expected duration: 1-4 hours depending on hardware.

### Training on Custom Data

```bash
export WASTE_DATA_ROOT=/path/to/custom/dataset
uv run model_comparison.py
```

Dataset must follow same directory structure as TrashNet.

## Understanding Output

### Console Output

Training progress for each model:
```
Training ResNet50...
Found 1769 images belonging to 6 classes.
Epoch 1/20
56/56 [==============================] - 45s 804ms/step - loss: 1.6234 - accuracy: 0.3891 - val_loss: 1.3456 - val_accuracy: 0.4987
...
```

Watch for:
- Decreasing loss
- Increasing accuracy
- Validation metrics tracking training (no large gap)

### Generated Files

`paper_outputs/` directory contains:

**table2_model_comparison.csv**:
```csv
Model,Accuracy,Precision,Recall,F1-Score
ResNet50,0.923,0.918,0.920,0.919
...
```

**figure2_confusion_matrices.png**: Three side-by-side confusion matrices showing classification patterns.

**figure3_xai_comparison_<model>.png**: XAI visualizations for best performing model.

## Configuration Options

### Adjusting Hyperparameters

In `model_comparison.py`:

```python
IMG_SIZE = (224, 224)    # Image dimensions
BATCH_SIZE = 32          # Samples per batch
EPOCHS = 20              # Training iterations
DATA_ROOT = "./data"     # Dataset location
OUTPUT_DIR = "paper_outputs"  # Results directory
```

### Data Augmentation

Modify in `load_data()` function:

```python
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,        # Degrees of rotation
    width_shift_range=0.05,   # Horizontal shift
    height_shift_range=0.05,  # Vertical shift
    horizontal_flip=True,     # Random horizontal flip
)
```

Additional augmentations:
```python
brightness_range=[0.8, 1.2],  # Brightness variation
zoom_range=0.1,                # Zoom range
shear_range=0.05,              # Shear transformation
```

### Model Selection

Edit `models` list in `main()`:

```python
models = ["ResNet50", "EfficientNetB0", "MobileNetV2"]
```

Available architectures:
- ResNet50: Deep residual network, highest accuracy
- EfficientNetB0: Balanced efficiency and accuracy
- MobileNetV2: Lightweight, suitable for mobile deployment

### XAI Parameters

In `generate_xai_figure()`:

```python
# Occlusion sensitivity
patch_size=20  # Size of occlusion patch
stride=10      # Step size for sliding window

# Integrated gradients
steps=50       # Number of interpolation steps

# LIME
num_samples=200   # Perturbations to generate
num_features=5    # Top features to highlight
```

## Troubleshooting

### Memory Errors

Reduce batch size:
```python
BATCH_SIZE = 8  # or even 4
```

### Slow Training

- Enable GPU if available
- Reduce image size (not recommended for accuracy)
- Use smaller model (MobileNetV2)
- Decrease augmentation complexity

### Poor Accuracy

- Verify data quality (check random samples)
- Increase epochs (try 30-50)
- Check for class imbalance
- Review data augmentation (may be too aggressive)

### Model Collapse

If one class dominates predictions, add class weights:

```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# In model.fit():
model.fit(..., class_weight=class_weight_dict)
```

## Advanced: Fine-tuning

After initial training, unfreeze base model for fine-tuning:

```python
# After first training phase
base.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train additional epochs
model.fit(train_gen, validation_data=val_gen, epochs=10)
```

This typically improves accuracy by 2-5%.

## Reproducibility

For reproducible results:

```python
import random
import numpy as np
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

Note: Complete reproducibility difficult on GPU due to non-deterministic operations.

## Citation

If using this code:

```bibtex
@misc{gagua2025wastesorting,
  author = {Gagua, Nika},
  title = {An Explainable AI based Decision Support System for Waste Sorting Systems},
  year = {2025},
  institution = {Kutaisi International University}
}
```

## Support

For issues:
- Check documentation in `/docs`
- Review error messages carefully
- Verify data integrity
- Contact: Nika.Gagua@kiu.edu.ge
