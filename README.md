# Waste Sorting Decision Support System

**Author**: Nika Gagua
**Institution**: Kutaisi International University
**Contact**: Nika.Gagua@kiu.edu.ge

## What This Project Does

Imagine a recycling facility processing thousands of items per hour. Workers manually sort waste into different bins - plastic here, metal there, paper over there. It's slow, tiring, and mistakes happen. What if a computer could do this automatically?

That's what this project tackles. But here's the catch: if the computer just says "this is plastic" without explaining why, plant managers won't trust it. So we built a system that not only classifies waste but shows its reasoning.

## The Challenge

Modern deep learning models are very accurate but operate as "black boxes." You feed in an image of a bottle, it says "plastic," but you don't know if it recognized the bottle shape, the transparency, or just memorized patterns from training data.

For recycling facilities, this opacity is a dealbreaker. Operators need to understand:

- Why did the system classify this item as plastic?
- Is it focusing on the right features?
- Can we trust it with difficult cases?

This project addresses these questions using explainable AI methods.

## Our Approach

We compare three popular neural network architectures:

- **ResNet50** - Deep network with skip connections
- **EfficientNetV2B0** - Balanced efficiency and accuracy
- **MobileNetV2** - Lightweight, runs on phones

Then we apply three explanation methods to visualize what the models "see":

- **Occlusion Sensitivity** - Block parts of the image, see what matters
- **Integrated Gradients** - Measure how each pixel contributes
- **LIME** - Highlight the most important regions

## The Dataset

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

## Expected Results

On TrashNet, these models typically achieve:

- ResNet50: ~92% accuracy
- EfficientNetV2B0: ~94% accuracy
- MobileNetV2: ~90% accuracy

Some classes are easier than others:

- **Easy**: Metal and glass (distinctive visual properties)
- **Medium**: Cardboard and plastic (more variation)
- **Hard**: Trash and paper (heterogeneous or similar to other classes)

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

Results appear in `paper_outputs/` directory:

- Performance metrics (CSV and LaTeX tables)
- Confusion matrices (PNG images)
- XAI visualizations (PNG images)

## Documentation

Complete documentation includes:

- **TUTORIAL.md** - Step-by-step implementation guide
- **RESULTS_GUIDE.md** - How to interpret outputs
- **NOTES.md** - Technical findings and observations

## Citation

If you use this work:

```bibtex
@misc{gagua2025wastesorting,
  author = {Gagua, Nika},
  title = {An Explainable AI based Decision Support System for Waste Sorting Systems},
  year = {2025},
  institution = {Kutaisi International University}
}
```

## Acknowledgments

TrashNet dataset created by Gary Thung and Mindy Yang at Stanford University.

Built with TensorFlow, Keras, scikit-learn, LIME, and OpenCV.

## Contact

Questions? Reach out at Nika.Gagua@kiu.edu.ge

---
