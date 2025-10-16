# Waste Sorting Decision Support System - Implementation

**Paper**: "An Explainable AI based Decision Support System for Waste Sorting Systems"
**Code Author**: Nika Gagua
**Date**: October 2025

---

## What This Code Does

This script implements the complete experimental methodology described in Section 4 of our paper:

1. ✅ **Trains 3 CNN models**: ResNet50, EfficientNet-B0, MobileNetV2
2. ✅ **Evaluates performance**: Accuracy, Precision, Recall, F1-Score
3. ✅ **Generates paper figures**:
   - Table 2: Model Performance Comparison
   - Figure 2: Confusion Matrices (all 3 models)
   - Figure 3: XAI Visualizations (Grad-CAM + LIME)

---

## Quick Setup

### 1. Install Dependencies

```bash
pip install uv
uv sync
```

### 2. Prepare Data

Place TrashNet dataset in this structure:

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
└── test/
```

**Download from**: https://github.com/garythung/trashnet

**Alternative**: Set custom data path via environment variable:

```bash
export WASTE_DATA_ROOT=/path/to/your/data
```

### 3. Run Experiment

```bash
uv run model_comparison.py
```

**Runtime**: ~1 hour (with GPU) or ~3-4 hours (CPU only)

---

## Output Files

Results saved to `paper_outputs/`:

```
paper_outputs/
├── table2_model_comparison.csv    # For Table 2 in paper
├── table2_model_comparison.tex    # LaTeX version
├── figure2_confusion_matrices.png # For Figure 2
└── figure3_xai_comparison.png     # For Figure 3
```

---

## Configuration

The script uses `./data` by default. To change settings, edit `model_comparison.py`:

```python
DATA_ROOT = os.environ.get("WASTE_DATA_ROOT", "./data")  # Or set env variable
EPOCHS = 20                        # Training epochs
BATCH_SIZE = 32                    # Adjust for GPU memory
```

---

## Expected Results

Example output (actual values depend on data splits):

| Model          | Accuracy | Precision | Recall | F1-Score |
| -------------- | -------- | --------- | ------ | -------- |
| ResNet50       | ~0.92    | ~0.91     | ~0.91  | ~0.91    |
| EfficientNetB0 | ~0.94    | ~0.93     | ~0.93  | ~0.93    |
| MobileNetV2    | ~0.90    | ~0.89     | ~0.89  | ~0.89    |

---

## Troubleshooting

**Problem**: Out of memory
**Solution**: Reduce `BATCH_SIZE` to 16 or 8

**Problem**: No GPU detected
**Solution**: Code works on CPU (just slower)

**Problem**: Module not found
**Solution**: `uv sync`

---

## Next Steps for Paper

Once you run this and review results:

1. Insert Table 2 into Results section (Section 5.1)
2. Insert Figure 2 into Results section (Section 5.2)
3. Insert Figure 3 into XAI section (Section 5.3)
4. We can discuss results and write the discussion

---

## Files Included

```
waste-sorting-dss/
├── README.md                    # This file
├── model_comparison.py          # Main script
└── project.toml             # Python packages
```

---

## Contact

Nika Gagua
Nika.Gagua@kiu.edu.ge
Kutaisi International University

For questions or modifications, please let me know.
