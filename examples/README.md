# Examples

This folder contains example outputs and demo scripts.

## What to Put Here

### Example Outputs
- Sample confusion matrices (PNG)
- XAI visualizations (PNG)
- Performance metrics (CSV)
- Example predictions on test images

### Demo Scripts
- `quick_demo.py` - Test a single image
- `batch_inference.py` - Process multiple images
- `compare_xai.py` - Side-by-side XAI comparison

## Current Status

Right now your `paper_outputs/` folder has your actual results. Consider copying some here as examples:

```bash
# Copy your best results here
cp paper_outputs/figure2_confusion_matrices.png examples/
cp paper_outputs/figure3_xai_comparison_*.png examples/
cp paper_outputs/table2_model_comparison.csv examples/
```

## Creating a Quick Demo

Here's a simple script to test single images:

```python
# save as examples/test_single_image.py
import numpy as np
from PIL import Image
from tensorflow import keras

# Load model
model = keras.models.load_model('../paper_outputs/best_model.h5')

# Load and preprocess image
img = Image.open('test_image.jpg').resize((224, 224))
img_array = np.array(img) / 255.0
img_batch = np.expand_dims(img_array, 0)

# Predict
preds = model.predict(img_batch)[0]
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Show results
for cls, prob in zip(classes, preds):
    print(f"{cls}: {prob:.3f}")
```

## For Your GitHub

When publishing, include:
- 1-2 example confusion matrices
- 3-4 XAI visualizations showing good explanations
- Sample CSV with metrics
- A few test images with predictions

This helps people understand what your code produces without running it.
