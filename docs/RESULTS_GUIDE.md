# Results Interpretation Guide

Documentation for understanding model outputs and XAI visualizations.

## Confusion Matrix Interpretation

Confusion matrices show prediction patterns across all classes.

### Reading the Matrix

**Axes**:
- Y-axis (vertical): True labels
- X-axis (horizontal): Predicted labels

**Values**:
- Diagonal elements: Correct predictions
- Off-diagonal elements: Misclassifications
- Values normalized per row (sum to 1.0)

**Example**: If true label is "paper" and predicted "plastic" shows 0.31, this means 31% of paper samples were incorrectly classified as plastic.

### Expected Patterns

**Strong diagonal** (values > 0.80): Model has learned distinguishing features for each class.

**Weak diagonal** (values < 0.60): Model struggling to differentiate classes. Check for:
- Insufficient training
- Class imbalance  
- Poor quality data
- Classes too visually similar

**Single column dominance**: Model defaulting to one prediction. Indicates collapsed training, usually from extreme class imbalance.

### Common Confusions

Some misclassifications are expected due to visual similarity:

**Paper ↔ Cardboard**: Both materials are fibrous with similar color palettes. Main difference is corrugation, which may not always be visible in images.

**Glass ↔ Plastic**: Transparent containers share similar shapes (bottles, jars). Distinguishing features are subtle (reflections, rigidity markers).

**Metal ↔ Glass**: Reflective surfaces can appear similar under certain lighting conditions.

**Trash → Various classes**: "Trash" is a heterogeneous category containing items that don't fit other classes. High confusion is expected.

## XAI Method Comparison

Three explainability methods are used, each with different properties.

### Occlusion Sensitivity

**Mechanism**: Systematically blocks image patches and measures prediction drop.

**Visual output**: Heatmap showing importance of each region.
- Hot colors (red, yellow): Regions critical to prediction
- Cool colors (blue, purple): Less important regions

**Interpretation**:
- Should highlight object features, not background
- Blocky appearance due to patch-based masking
- Computationally expensive (tests many occlusions)

**Strengths**: Intuitive, model-agnostic  
**Weaknesses**: Slow, discrete patches

### Integrated Gradients

**Mechanism**: Accumulates gradients along path from baseline to input image.

**Visual output**: Pixel-level attribution map.
- Bright regions: High attribution
- Dark regions: Low attribution

**Interpretation**:
- Smoother than occlusion
- Shows fine-grained pixel importance
- Baseline typically set to black image or blurred version

**Strengths**: Theoretically grounded, smooth visualizations  
**Weaknesses**: Baseline choice affects results

### LIME (Local Interpretable Model-Agnostic Explanations)

**Mechanism**: Fits linear model locally using superpixel perturbations.

**Visual output**: Highlighted superpixels that influenced decision.
- Yellow outlines: Important regions
- Unmarked areas: Not relevant to this prediction

**Interpretation**:
- Groups pixels into semantically meaningful segments
- Shows which image parts contributed to classification
- Local explanation (specific to this instance)

**Strengths**: Human-interpretable segments  
**Weaknesses**: Superpixel quality varies, slower than gradient methods

## Analyzing XAI Outputs

### Signs of Good Explanations

Model focusing on relevant features:
- **Cardboard**: Corrugated texture, fold lines
- **Glass**: Transparency, smooth surface, bottle shape
- **Metal**: Metallic shine, cylindrical structure, can rim
- **Paper**: Fibrous texture, flat surface
- **Plastic**: Bottle shape, cap, smooth curves

### Signs of Problems

Model relying on spurious correlations:
- Focusing on background instead of object
- Highlighting text or logos
- Attention scattered randomly
- Different XAI methods showing completely different patterns

### Failure Case Analysis

When models misclassify, XAI reveals reasoning:

**Example**: Paper item (pen) classified as glass
- XAI shows focus on cylindrical shape
- Model learned "cylinder = bottle = glass"
- Understandable mistake based on shape alone

This type of failure is acceptable and shows model using geometric features. More concerning would be if model focused on irrelevant background elements.

## Performance Benchmarks

Expected ranges on TrashNet dataset:

| Model            | Accuracy | Per-Class F1 |
|------------------|----------|--------------|
| ResNet50         | 0.88-0.94| 0.80-0.92    |
| EfficientNetV2B0 | 0.90-0.96| 0.85-0.94    |
| MobileNetV2      | 0.85-0.92| 0.78-0.90    |

### Per-Class Difficulty

**Easier categories** (typically F1 > 0.85):
- Metal: Distinctive reflective properties
- Glass: Consistent transparency
- Cardboard: Unique corrugated texture

**Harder categories** (typically F1 < 0.80):
- Trash: Heterogeneous items
- Paper: Similar to cardboard
- Plastic: High visual variability

## Common Training Issues

### Class Imbalance

**Symptom**: One class dominates predictions in confusion matrix.

**Check**: Count samples per class in training set.

**Solution**: Apply class weights during training:
```python
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', 
                                classes=np.unique(y_train), 
                                y=y_train)
```

### Overfitting

**Symptom**: High training accuracy (>95%) but low test accuracy (<80%).

**Solution**: 
- Increase data augmentation
- Add dropout layers
- Reduce model complexity
- Early stopping

### Underfitting

**Symptom**: Both training and test accuracy low (<75%).

**Solution**:
- Train longer (more epochs)
- Increase model capacity
- Lower learning rate
- Check data quality

## Metrics Explanation

**Accuracy**: Overall fraction of correct predictions. Can be misleading with imbalanced classes.

**Precision**: Of predicted positives, how many are actually positive. Important when false positives are costly.

**Recall**: Of actual positives, how many were found. Important when false negatives are costly.

**F1-Score**: Harmonic mean of precision and recall. Useful for imbalanced datasets.

**Confusion Matrix**: Complete picture of per-class performance. Most informative metric.

## Visualization Best Practices

When creating figures for papers:
- Use high DPI (300) for publication quality
- Include color bars with proper labels
- Ensure text is readable
- Show multiple example images per class
- Include both correct and incorrect predictions

## Reproducibility Notes

Results depend on:
- Random train/test split
- Weight initialization
- Data augmentation random seed
- GPU vs CPU training (minor differences)

Expected variance: ±2-3% accuracy between runs.

Report mean and standard deviation from multiple runs for rigorous evaluation.
