# Project Overview

**Project**: Explainable AI for Waste Sorting  
**Author**: Nika Gagua  
**Institution**: Kutaisi International University  
**Date**: January 2025

## The Problem

Recycling facilities need automated sorting systems, but current AI solutions lack transparency. Plant managers won't trust a system that just says "this is plastic" without explaining why. Regulators require auditable decision-making. Workers need to understand when the system might be wrong.

This project addresses the black-box problem in AI-driven waste sorting by integrating explainability methods into the classification pipeline.

## Research Questions

1. How do different CNN architectures compare for waste classification?
2. Can explainability methods reveal what visual features drive predictions?
3. Do the models learn meaningful material properties or superficial patterns?
4. Where do failures occur and why?

## Methodology

### Dataset

TrashNet benchmark with 2,527 images across six categories:
- Cardboard (corrugated packaging)
- Glass (bottles, jars)
- Metal (cans, tins)
- Paper (documents, office waste)
- Plastic (bottles, containers)
- Trash (miscellaneous items)

Images split 70% training, 15% validation, 15% testing.

### Models

Three architectures representing different design philosophies:

**ResNet50** - Deep residual network with skip connections. Highest accuracy, most parameters, slowest inference.

**EfficientNetV2B0** - Optimized for efficiency-accuracy tradeoff. Balanced choice for production systems.

**MobileNetV2** - Lightweight architecture for mobile deployment. Fastest inference, slight accuracy reduction.

All models use transfer learning from ImageNet pre-trained weights.

### Explainability Methods

**Occlusion Sensitivity** - Systematically masks image regions and measures prediction impact. Shows which areas matter most through colored heatmaps.

**Integrated Gradients** - Accumulates gradients along interpolation path from baseline to input. Provides pixel-level attribution with theoretical guarantees.

**LIME** - Fits local linear model using superpixel perturbations. Segments image into meaningful regions and ranks importance.

### Training Procedure

- Adam optimizer, learning rate 1e-4
- Data augmentation (rotation, shifts, flips)
- Early stopping on validation loss
- Class weight balancing for imbalanced categories

## Expected Results

### Classification Performance

Anticipated accuracies based on TrashNet benchmarks:
- ResNet50: 88-94%
- EfficientNetV2B0: 90-96%  
- MobileNetV2: 85-92%

Per-class difficulty:
- **Easy**: Metal, Glass (distinctive properties)
- **Medium**: Cardboard, Plastic (moderate variation)
- **Hard**: Paper, Trash (high similarity or heterogeneity)

### Explainability Findings

Visual explanations should reveal:
- **Cardboard**: Focus on corrugated texture, fold lines
- **Glass**: Attention to transparency, smooth surfaces
- **Metal**: Highlighting metallic reflections, cylindrical shapes
- **Paper**: Recognition of fibrous patterns

Common confusions expected:
- Paper ↔ Cardboard (similar fibrous materials)
- Glass ↔ Plastic (transparent containers)
- Trash → Various (catch-all category)

## Key Contributions

### Technical

1. **Comparative analysis** of modern CNN architectures on standardized waste dataset
2. **Integration** of three complementary XAI methods in environmental domain
3. **Open-source implementation** with complete documentation for reproducibility
4. **Systematic evaluation** of explainability quality across methods

### Practical

1. **Decision support framework** combining accuracy with interpretability
2. **Visual explanations** suitable for non-technical stakeholders
3. **Failure analysis** revealing model reasoning on difficult cases
4. **Deployment considerations** for real-world recycling facilities

### Scientific

1. **Evidence** that models learn material-specific features, not just shapes
2. **Demonstration** of when different XAI methods agree vs. disagree
3. **Documentation** of failure modes and their underlying causes
4. **Foundation** for future work on transparent environmental AI

## Implementation

### Technology Stack

- Python 3.8+
- TensorFlow/Keras for neural networks
- scikit-learn for evaluation metrics
- LIME library for local explanations
- OpenCV for image processing
- Matplotlib/Seaborn for visualization

### Code Structure

```
waste-sorting-ai/
├── model_comparison.py      # Main training pipeline
├── download_data.py          # Dataset preparation
├── docs/
│   ├── UNDERSTANDING_RESULTS.md
│   └── TUTORIAL.md
└── paper_outputs/            # Generated visualizations
```

### Reproducibility

Complete environment specification in `pyproject.toml`. Installation takes 5 minutes, full training 1-4 hours depending on hardware.

```bash
pip install uv
uv sync
uv run model_comparison.py
```

## Current Status

### Completed Components

✅ Training pipeline with three architectures  
✅ Three XAI methods fully integrated  
✅ Automated visualization generation  
✅ Performance metrics and confusion matrices  
✅ Comprehensive documentation

### In Testing Phase

The training runs revealed class imbalance effects. Implementation now includes class weight balancing to address this. Retraining will produce final results for paper.

### Next Steps

1. Complete final training runs with balanced weights
2. Generate publication-quality figures
3. Document results in academic paper
4. Prepare presentation materials

## Observations

### Class Imbalance Impact

The "trash" category has significantly fewer samples (137 vs 400-600 for other classes). This causes models to under-represent trash in predictions. Solution implemented: sklearn's balanced class weights.

### Architecture Differences

Preliminary results show expected patterns:
- EfficientNet offers best accuracy-efficiency balance
- MobileNetV2 suitable for resource-constrained deployment
- ResNet50 most robust but computationally intensive

### XAI Agreement

When all three explanation methods highlight the same features, we can trust the model is using meaningful patterns. Disagreement suggests multiple reasoning paths or prediction uncertainty.

## Significance

### For Waste Management

Provides practical tool for recycling facilities requiring transparent automation. Visual explanations enable operator training and system debugging.

### For AI Research

Demonstrates XAI application in environmental domain. Shows how multiple methods provide complementary views of model behavior.

### For Policy

Addresses regulatory requirements for explainable automated systems. Provides audit trails supporting compliance verification.

## Limitations

**Dataset Scale**: TrashNet is relatively small (2,527 images). Real-world systems need larger, more diverse datasets.

**Controlled Conditions**: Clean backgrounds and single objects don't reflect cluttered conveyor belt scenarios.

**Category Coverage**: Missing important waste types (e-waste, organics, hazardous materials, textiles).

**Static Images**: Real deployment involves moving items, variable lighting, occlusion.

## Future Directions

### Short Term
- Expand to larger datasets (TACO, Waste-Net)
- Test on facility footage
- Add more waste categories
- Optimize for edge deployment

### Long Term
- Real-time video processing
- Multi-object detection
- Integration with robotic sorting
- Continuous learning from operator corrections

## Deliverables

1. **Code Repository** - Complete, documented, reproducible
2. **Research Paper** - Methodology, results, analysis
3. **Visualizations** - Confusion matrices, XAI examples
4. **Documentation** - Setup, usage, interpretation guides
5. **Presentation** - Overview for technical and non-technical audiences

## Questions for Discussion

1. Should we expand to more waste categories before publication?
2. Is the XAI analysis sufficient or add additional methods?
3. How to best present the class imbalance challenge in the paper?
4. What deployment scenarios should we emphasize?
5. Recommendations for real-world testing with industry partners?

## Contact

Nika Gagua  
Email: Nika.Gagua@kiu.edu.ge  
Repository: [To be added]

---

## For Review

Please examine:
1. `README.md` - Project overview
2. `TECHNICAL_NOTES.md` - Development findings
3. `docs/UNDERSTANDING_RESULTS.md` - Results interpretation
4. `paper/` - Research paper (to be added)
5. `paper_outputs/` - Generated visualizations
