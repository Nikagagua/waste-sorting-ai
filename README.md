# Waste Sorting Decision Support System

Implementation of "An Explainable AI based Decision Support System for Waste Sorting Systems"

**Author**: Nika Gagua
**Institution**: Kutaisi International University
**Contact**: Nika.Gagua@kiu.edu.ge

## Overview

Comparative evaluation of CNN architectures (ResNet50, EfficientNetV2B0, MobileNetV2) for waste classification with explainability analysis using Grad-CAM and LIME.

## Installation

```bash
pip install uv
uv sync
```

## Dataset Preparation

### Option 1: Automatic Download

```bash
uv run download_data.py
```

### Option 2: Manual Setup

Download TrashNet dataset from https://github.com/garythung/trashnet and organize as:

```
data/
├── train/<class_name>/*.jpg
├── val/<class_name>/*.jpg
└── test/<class_name>/*.jpg
```

Classes: cardboard, glass, metal, paper, plastic, trash

### Custom Data Path

```bash
export WASTE_DATA_ROOT=/path/to/dataset
```

## Usage

```bash
uv run model_comparison.py
```

Training time: 1-4 hours depending on hardware.

## Output

Results are saved to `paper_outputs/`:

- `table2_model_comparison.csv` - Performance metrics
- `table2_model_comparison.tex` - LaTeX table
- `figure2_confusion_matrices.png` - Confusion matrices
- `figure3_xai_comparison.png` - XAI visualizations

## Configuration

Edit `model_comparison.py` to adjust:

```python
DATA_ROOT = os.environ.get("WASTE_DATA_ROOT", "./data")
EPOCHS = 20
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
```

## Performance

Expected metrics on TrashNet dataset:

| Model            | Accuracy | Precision | Recall | F1-Score |
| ---------------- | -------- | --------- | ------ | -------- |
| ResNet50         | ~0.92    | ~0.91     | ~0.91  | ~0.91    |
| EfficientNetV2B0 | ~0.94    | ~0.93     | ~0.93  | ~0.93    |
| MobileNetV2      | ~0.90    | ~0.89     | ~0.89  | ~0.89    |

_Results vary with train/test split randomization._

## Troubleshooting

| Issue           | Solution                                        |
| --------------- | ----------------------------------------------- |
| Out of memory   | Reduce `BATCH_SIZE` to 16 or 8                  |
| Missing modules | Run `uv sync`                                   |
| Data not found  | Run `download_data.py` or set `WASTE_DATA_ROOT` |

## Project Structure

```
waste-sorting-ai/
├── model_comparison.py    # Main training script
├── download_data.py       # Dataset preparation
├── pyproject.toml         # Dependencies
└── README.md              # Documentation
```

## Citation

If you use this code, please cite:

```
Gagua, N. (2025). An Explainable AI based Decision Support System
for Waste Sorting Systems. Kutaisi International University.
```
