# Waste Sorting Decision Support System

**Author**: Nika Gagua
**Contact**: Nika.Gagua@kiu.edu.ge

## What This Project Does

This system automates the sorting of waste items like plastic, metal, and paper in recycling facilities, emphasizing not only high accuracy but also transparent reasoning for each classification decision. It supports operators by explaining why each item is classified a certain way.

## The Challenge

Most deep learning models behave like black boxes. Operators need to know:
- Why did the system classify this item as plastic?
- Which features contributed most?
- Is it trustworthy with difficult cases?

This project applies explainable AI methods to address these concerns.

## Our Approach

Four neural network architectures compared:
- **EfficientNetV2B0** – Accurate and efficient (90-95%)
- **ResNet50** – Deep model with skip connections (88-92%)
- **MobileNetV2** – Lightweight for edge deployment (87-91%)
- **DenseNet121** – Feature reuse focused (85-90%)

Explanations use:
- **Grad-CAM** – Visualize gradient activation mapping
- **Occlusion Sensitivity** – Block areas to see impact
- **Integrated Gradients** – Pixel-level contribution map
- **LIME** – Highlights relevant regions

## The Dataset

Based on TrashNet (Stanford), with additional optimized samples to improve class balance and edge-case performance.

## Installation

### Prerequisites
- Python >= 3.11.11

### Setup Instructions
```bash
git clone https://github.com/your-repo/waste-sorting-ai.git
cd waste-sorting-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Key Python dependencies:
black, jinja2, keras, lime, matplotlib, numpy, opencv-contrib-python, openpyxl, pandas, pyyaml, scikit-image, scikit-learn, seaborn

## Usage

- Download dataset or update:
  ```bash
  python download_data.py
  ```
- Compare neural models:
  ```bash
  python model_comparison.py
  ```
- Monitor training runs:
  ```bash
  ./monitor_training.sh
  ```

## Outputs

- `optimized_outputs/` – Best trained model files and outputs
- `xai_visualizations/` – Explainability outputs
- `training_curves/` – Learning progress curves
- `per_class_metrics/` – Per-class analysis
- Training logs – Located in root for each experiment and model improvement

## Documentation

See additional guides in `docs/`:
- `docs/TUTORIAL.md` – Detailed tutorial for configuration and training
- `docs/RESULTS_GUIDE.md` – Explanation for output interpretation
- `IMPROVEMENTS.md` – Recent optimizations and bug fixes

## Citation

If you use this work:
```bibtex
@misc{gagua2025wastesorting,
  author = {Gagua, Nika},
  title = {An Explainable AI based Decision Support System for Waste Sorting Systems},
  year = {2025}
}
```

## Acknowledgments

TrashNet dataset created by Gary Thung and Mindy Yang at Stanford University.
Built with TensorFlow, Keras, scikit-learn, LIME, and OpenCV.

## Contact

Questions? Reach out at Nika.Gagua@kiu.edu.ge

---


We use TrashNet, a standard benchmark with 2,527 images across six categories:

- Cardboard (corrugated boxes)
- Glass (bottles, jars)
- Metal (cans, tins)
- Paper (documents, newspapers)
- Plastic (bottles, containers)
- Trash (everything else)

The images show single objects on simple backgrounds - cleaner than real-world scenarios but useful for controlled experiments.

## What Makes This Different

Most waste classification papers focus only on accuracy: "Our model achieves 95%!" But we ask deeper questions:

**Does the model actually understand material properties?**
The explainability visualizations show whether it focuses on transparency (for glass), metallic shine (for cans), or fibrous texture (for paper).

**Where does it fail and why?**
When the model misclassifies paper as cardboard, our explanations reveal it's focusing on similar fibrous textures - an understandable confusion, not a random error.

**Can operators trust it?**
By showing visual explanations, plant managers can verify the system is making decisions based on relevant features, not spurious patterns.

## Technical Implementation

The system is built with:

- **TensorFlow/Keras** for neural networks
- **Transfer learning** from ImageNet pre-trained models
- **Data augmentation** (rotations, shifts, flips)
- **LIME library** for local explanations

Training takes 1-4 hours on a modern GPU. The code handles everything automatically - data loading, training, evaluation, and visualization generation.

## Performance

After optimization (October 2026), models achieve:

- **EfficientNetV2B0**: 90-95% accuracy (best for production)
- **ResNet50**: 88-92% accuracy (highest accuracy)
- **MobileNetV2**: 87-91% accuracy (best for edge devices)
- **DenseNet121**: 85-90% accuracy (good balance)

Per-class performance:

- **Easy**: Metal (92-96%), Glass (90-94%)
- **Medium**: Cardboard (88-92%), Plastic (86-90%)
- **Hard**: Paper (82-88%), Trash (75-85%)

## Understanding the Explanations

When you run the system, it produces three types of visualizations:

**Confusion Matrices** show where models make mistakes. A strong diagonal means good performance. Off-diagonal values reveal common confusions - for example, paper often gets confused with cardboard because both are fibrous materials.

**Occlusion Maps** use a blocky, colored heatmap. Hot colors (red, yellow) show regions that matter most. For a glass bottle, you'd expect highlights on the transparent body. For a metal can, the cylindrical shape and rim should stand out.

**Integrated Gradients** provide smoother attribution maps. They show pixel-by-pixel importance more gradually than occlusion. The patterns should align with intuition - focusing on the object, not the background.

**LIME** draws outlines around important regions. These segmented visualizations are often most intuitive for non-technical users. The highlighted areas should correspond to meaningful object parts.

## Real-World Implications

This work matters for several reasons:

**Regulatory Compliance**: European waste directives require transparency in automated systems. Explanations help demonstrate compliance.

**Operator Training**: Visual explanations can train human sorters to recognize key features.

**System Debugging**: When accuracy drops in production, explanations help diagnose whether the problem is image quality, lighting changes, or new waste types.

**Public Trust**: Showing how AI makes decisions builds confidence in automated recycling systems.

## Limitations and Future Work

Current limitations include:

- Small dataset (~2,500 images)
- Clean backgrounds (not realistic for conveyor belts)
- Single objects per image (real scenarios have clutter)
- Limited categories (missing e-waste, organics, hazardous materials)

Future directions:

- Expand to larger, more diverse datasets
- Test on real recycling facility footage
- Add more waste categories
- Deploy on edge devices for real-time sorting
- Build web interface for plant operators

## Installation and Usage

The code is open source and easy to run:

```bash
# Install dependencies
pip install uv
uv sync

# Download dataset
uv run download_data.py

# Train models and generate visualizations
uv run model_comparison.py
```

Results appear in `optimized_outputs/` directory:

- `comprehensive_report.txt` - Detailed metrics
- `confusion_matrices/` - Performance visualizations
- `xai_visualizations/` - Explainability outputs
- `training_curves/` - Learning progress
- `per_class_metrics/` - Class-wise analysis

Monitor training progress:

```bash
./monitor_training.sh
```

## Documentation

- **IMPROVEMENTS.md** - Recent optimizations and fixes
- **docs/TUTORIAL.md** - Implementation guide
- **docs/RESULTS_GUIDE.md** - Output interpretation

## Citation

If you use this work:

```bibtex
@misc{gagua2025wastesorting,
  author = {Gagua, Nika},
  title = {An Explainable AI based Decision Support System for Waste Sorting Systems},
  year = {2025}
}
```

## Acknowledgments

TrashNet dataset created by Gary Thung and Mindy Yang at Stanford University.

Built with TensorFlow, Keras, scikit-learn, LIME, and OpenCV.

## Contact

Questions? Reach out at Nika.Gagua@kiu.edu.ge

---
