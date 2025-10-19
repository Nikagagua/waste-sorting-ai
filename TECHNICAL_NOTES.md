# Technical Observations

Documented findings from testing and development.

## The Class Imbalance Challenge

One recurring issue in waste classification is uneven data distribution. Some categories have many examples, others have few.

In TrashNet:
- Paper: 594 images
- Glass: 501 images  
- Plastic: 482 images
- Metal: 410 images
- Cardboard: 403 images
- Trash: 137 images

Notice trash has less than a quarter the samples of paper. This creates problems during training.

### Why Imbalance Matters

Neural networks learn by seeing many examples. When one class dominates, the model may discover a shortcut: "Just predict the common class most of the time and I'll be right often enough."

This leads to a collapsed model where everything gets classified as the majority class. The accuracy might look decent (if 70% of data is paper and you always predict paper, you're 70% accurate), but the model is useless.

### The Solution: Class Weights

We can tell the model "pay more attention to rare classes." Mathematically, we apply higher loss penalties when misclassifying underrepresented categories.

The sklearn library calculates balanced weights automatically. Adding these to training prevents the model from ignoring minority classes.

## Three Architectures, Three Philosophies

### ResNet50: Deep and Careful

ResNet uses "skip connections" - shortcuts that let information flow directly through the network. Think of it like a building with both stairs and elevators.

**Strengths**: Learns complex patterns, high accuracy, well-established  
**Weaknesses**: Slower training, more parameters, needs more memory

Good choice when accuracy is paramount and computational resources aren't constrained.

### EfficientNetV2B0: Balanced Approach

EfficientNet optimizes the tradeoff between depth, width, and resolution. It's like finding the sweet spot between a sports car and a truck.

**Strengths**: Best accuracy-to-efficiency ratio, modern architecture  
**Weaknesses**: Newer (less tested), can be sensitive to hyperparameters

Often the best choice for production systems where both accuracy and speed matter.

### MobileNetV2: Light and Fast

MobileNet uses "depthwise separable convolutions" - a trick that dramatically reduces computation. Like using a Swiss Army knife instead of a full toolbox.

**Strengths**: Fast inference, small model size, runs on phones  
**Weaknesses**: Slightly lower accuracy, less capacity for complex patterns

Perfect for edge devices, real-time applications, or when deploying to mobile platforms.

## XAI Methods: Different Lenses

### Why Multiple Methods?

Each explanation technique has blind spots. Using three methods provides triangulation - if they all agree, we can be more confident.

### Occlusion: The Brute Force Approach

This method is conceptually simple: block parts of the image and measure impact. It's like doing a grid search through the image.

**Advantages**:
- Intuitive to understand
- Model-agnostic (works with any architecture)
- Direct causality (blocking definitely affects prediction)

**Disadvantages**:
- Computationally expensive (many forward passes)
- Blocky artifacts (fixed patch size)
- May miss fine details between patch boundaries

### Integrated Gradients: The Mathematical Approach

This method has theoretical guarantees. It satisfies "axioms" that any good attribution should follow.

**Advantages**:
- Smooth, continuous visualizations
- Mathematically principled
- Captures fine details

**Disadvantages**:
- Requires differentiable models
- Baseline choice affects results
- Less intuitive to non-technical audiences

### LIME: The Local Approximation

LIME fits a simple model (linear regression) around each prediction. It's like drawing a straight line through a curved function - locally accurate even if globally wrong.

**Advantages**:
- Human-friendly segmentation
- Model-agnostic
- Provides feature importance rankings

**Disadvantages**:
- Slower than gradient methods
- Superpixel quality varies
- Local explanations don't show global patterns

## Interesting Failure Modes

### The Shape-Over-Material Trap

Models sometimes learn shape patterns when they should learn material properties. A cylindrical pen gets classified as glass because "cylinder = bottle = glass."

This reveals the model hasn't fully learned material characteristics. It's using geometry as a proxy, which works often but fails on edge cases.

### The Background Shortcut

If training data has consistent backgrounds per class (all glass photos on white backgrounds, all plastic on gray), models learn "white background = glass."

This fails immediately in real-world deployment. XAI visualizations reveal this by showing attention on backgrounds rather than objects.

### The Logo Memorization

Models might learn "Coca-Cola logo = plastic" instead of learning plastic material properties. Works for common brands but fails on generic items.

## Training Dynamics

### Learning Curves

Watching accuracy evolve during training tells a story:

**Early epochs** (1-5): Rapid improvement as model learns basic features  
**Middle epochs** (6-15): Steady progress, model refining understanding  
**Late epochs** (16-20): Diminishing returns, fine-tuning details

If accuracy jumps around wildly, learning rate is too high. If it plateaus early, learning rate might be too low or data is insufficient.

### Overfitting Signs

Gap between training and validation accuracy indicates overfitting:
- Small gap (<5%): Healthy
- Medium gap (5-15%): Mild overfitting, acceptable
- Large gap (>20%): Severe overfitting, model memorizing training data

Solutions include more data, stronger augmentation, regularization, or earlier stopping.

## Data Augmentation Philosophy

Augmentation creates artificial variations to teach robustness. But there's a balance.

### Appropriate Augmentations

**Rotation** (small angles): Real waste appears at various orientations on conveyor belts.

**Shifts and flips**: Items aren't always centered in camera view.

**Brightness variations**: Different lighting conditions throughout the day.

### Dangerous Augmentations

**Extreme rotation** (90+ degrees): Waste doesn't typically appear upside-down.

**Heavy distortion**: Unrealistic transformations may hurt more than help.

**Wrong color shifts**: Changing glass to red isn't realistic.

The key is augmenting within the distribution of real variations you expect to see.

## Explainability Insights

### When Explanations Agree

All three XAI methods highlighting the same features is strong evidence the model learned meaningful patterns. For example, if occlusion, gradients, and LIME all focus on corrugated texture for cardboard, we can trust that decision pathway.

### When Explanations Disagree

Different methods showing different features might indicate:
- Multiple valid reasoning paths
- Redundant features (model using any of several cues)
- Instability in predictions (low confidence)

### Explanation Quality Over Time

Early in training, XAI maps look random - scattered attention, no clear pattern. As training progresses, they become focused and coherent. This evolution mirrors the model's learning process.

## Computational Considerations

### Memory Bottlenecks

GPU memory limits batch size. Larger batches give more stable gradients but require more memory.

Typical limits:
- 8GB GPU: batch size 16-32
- 16GB GPU: batch size 32-64
- 24GB GPU: batch size 64+

Running out of memory? Reduce batch size first before changing model.

### Training Time

On modern hardware:
- MobileNetV2: ~30 minutes per 20 epochs
- EfficientNetV2B0: ~1 hour per 20 epochs
- ResNet50: ~1.5 hours per 20 epochs

XAI generation adds time:
- Occlusion: Slowest (~2 minutes per image)
- Integrated Gradients: Medium (~30 seconds per image)
- LIME: Fast (~15 seconds per image)

## Dataset Limitations

TrashNet is valuable but artificial. Real recycling scenarios differ:

**Missing complexity**: Single object per image vs. cluttered conveyor belts

**Clean backgrounds**: Simple settings vs. industrial environments

**Good lighting**: Controlled conditions vs. varying facility lighting

**Limited categories**: 6 classes vs. dozens of waste types in practice

**Static images**: Photos vs. moving items on belts

These gaps mean models trained on TrashNet need adaptation for real deployment.

## Lessons for Future Work

### Data Matters Most

Better data trumps better algorithms. Before trying fancy architectures, improve dataset quality:
- More samples per class
- More diverse examples
- Real-world conditions
- Better labeling

### Start Simple

Begin with MobileNetV2 for fast iteration. Only move to larger models if accuracy isn't sufficient. Often the bottleneck is data, not model capacity.

### Validate Explanations

Don't just generate XAI visualizations - actually look at them. Are they sensible? Do they match intuition? Nonsensical explanations reveal training problems early.

### Think About Deployment

Lab accuracy isn't everything. Consider:
- Inference speed requirements
- Hardware constraints  
- Interpretability needs
- Maintenance and updates

A 92% accurate model that runs in real-time and provides clear explanations often beats a 95% accurate black box that's slow and opaque.
