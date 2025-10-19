# Understanding the Results

This guide explains what the model outputs mean and how to interpret them.

## Reading Confusion Matrices

A confusion matrix is a grid showing prediction patterns. Think of it like a scoreboard for each waste category.

### The Layout

Rows represent the true category (what the item actually is). Columns show what the model predicted. Each cell contains a number between 0 and 1.

**The diagonal** (top-left to bottom-right) shows correct predictions. When you see 0.90 in the "glass" row and "glass" column, it means 90% of glass items were correctly identified as glass.

**Off-diagonal values** reveal mistakes. If the "paper" row shows 0.25 in the "cardboard" column, it means 25% of paper items were misclassified as cardboard.

### What Good Looks Like

A well-trained model has:
- Dark diagonal (values above 0.80)
- Light off-diagonal (most values near 0)
- Sensible confusions (similar materials getting mixed up)

For example, paper and cardboard confusion is reasonable - they're both fibrous materials. But if metal gets confused with paper frequently, something's wrong.

## Three Ways to Explain Decisions

The system uses three different visualization methods. Each has strengths.

### Occlusion Sensitivity: The Blocking Method

Imagine covering different parts of an image with your hand and watching the prediction change. That's what occlusion does systematically.

The result is a colored heatmap:
- **Red/yellow** = important regions (prediction drops when blocked)
- **Blue/purple** = less important (blocking doesn't affect prediction)

For a cardboard box, you'd expect red highlights on the corrugated texture and fold lines. For a metal can, the cylindrical body and metallic rim should light up.

If the heatmap shows random scattered colors or focuses on the background, the model isn't learning meaningful features.

### Integrated Gradients: The Accumulation Method

This method measures how important each pixel is by gradually building up from a blank image to the final image. Think of it like watching a photo develop and seeing which parts appear first.

The output is smoother than occlusion - less blocky, more flowing. Bright regions contributed most to the decision.

These maps tend to highlight edges and textures. A glass bottle should show concentrated brightness on transparent areas and the bottle outline.

### LIME: The Region Method

LIME segments the image into patches (called superpixels) and tests each one. It's like breaking a puzzle into pieces and seeing which pieces matter most.

The result shows yellow outlines around important regions. These visualizations are often easiest for non-technical viewers to understand because they follow natural object boundaries.

For a plastic bottle, LIME might outline the bottle body, cap, and label area - all relevant features for identifying plastic.

## Common Patterns in Results

### Material-Specific Features

Different materials have characteristic patterns:

**Glass** - Models learn to recognize:
- Transparency and translucency
- Smooth surfaces with reflections
- Typical bottle/jar shapes
- Uniform texture

**Metal** - Key features include:
- Metallic shine and reflections
- Cylindrical shapes (cans)
- Sharp edges
- Uniform material appearance

**Cardboard** - Distinguishing traits:
- Corrugated texture (wavy layers)
- Fold lines and creases
- Tan/brown coloring
- Fibrous surface

**Paper** - Recognition based on:
- Flat, smooth surfaces
- White or printed patterns
- Thin appearance
- Similar texture to cardboard (causing confusion)

**Plastic** - Identified by:
- Variety of colors
- Bottle shapes
- Smooth, molded surfaces
- Sometimes transparent like glass (causing confusion)

**Trash** - The difficult category:
- Heterogeneous items
- Doesn't fit other patterns
- Often misclassified as whatever it resembles

### Expected Confusions

Some mistakes make sense:

**Paper ↔ Cardboard** is the most common confusion. Both materials are cellulose-based with fibrous textures. The main difference is thickness and corrugation, which may not always be visible in photos.

**Glass ↔ Plastic** happens with transparent containers. Both can be bottle-shaped and clear. The model must learn subtle differences in reflections and surface properties.

**Trash → Various** is expected because "trash" is a catch-all category. An item might look plastic-like, metal-like, or paper-like while technically being trash.

### Problem Patterns

Some patterns indicate issues:

**One column lights up** across all rows means the model is predicting everything as one class. This happens when training collapses, usually from severe class imbalance.

**Random scatter** with no clear pattern suggests the model hasn't learned anything meaningful. It's essentially guessing.

**Background focus** in XAI visualizations means the model is cheating - learning shortcuts based on backgrounds rather than object features.

## Interpreting Performance Numbers

### Accuracy

The overall percentage of correct predictions. Sounds simple, but can be misleading.

If you have 100 images - 90 plastic and 10 glass - a model that always predicts "plastic" gets 90% accuracy while being completely useless for glass detection.

That's why we look at per-class metrics.

### Precision

Of all items predicted as plastic, how many actually are plastic?

High precision means few false alarms. Important when misclassifying items is costly (e.g., contaminating recyclables).

### Recall

Of all actual plastic items, how many did we correctly identify?

High recall means we're not missing much. Important when it's critical to catch everything (e.g., hazardous materials).

### F1-Score

Combines precision and recall into a single number. Useful for comparing models when you care about both false positives and false negatives.

## What Different Accuracies Mean

**Above 0.90** - Excellent performance. Model has learned strong features. Ready for consideration in real applications with human oversight.

**0.75 - 0.90** - Good performance. Some confusions but generally useful. Typical for categories with high visual variability.

**0.60 - 0.75** - Moderate performance. Model captures some patterns but struggles with difficult cases. May need more training data or better features.

**Below 0.60** - Poor performance. Model hasn't learned meaningful patterns. Could be data quality issues, insufficient training, or inherently difficult classification task.

## Real Examples

Let's walk through some actual cases:

### Case 1: Cardboard Correctly Classified

The model sees a cardboard box and predicts "cardboard" with 0.97 confidence.

**Occlusion map** highlights the horizontal fold line and corrugated texture.
**Integrated gradients** shows high attribution on the fold and edges.
**LIME** outlines the corrugated section.

**Interpretation**: The model is focusing on exactly what we'd want - the distinctive features of cardboard. This is trustworthy.

### Case 2: Paper Misclassified as Glass

A pen is labeled as paper but the model predicts "glass" with 0.31 confidence.

**Occlusion map** focuses on the cylindrical shape.
**Integrated gradients** highlights the smooth surface.
**LIME** outlines the entire pen body.

**Interpretation**: The model saw a cylindrical, smooth object and thought "bottle." It's wrong, but for understandable reasons. The shape resembles glass containers more than flat paper.

### Case 3: Trash Correctly Identified

A miscellaneous item labeled trash, predicted as "trash" with 0.40 confidence.

**Low confidence** is actually appropriate here - trash is heterogeneous. 
**XAI maps** show attention scattered across multiple regions.

**Interpretation**: The model knows this doesn't fit clean patterns. Low confidence reflects genuine ambiguity, which is honest behavior.

## When to Trust the Model

Trust the model when:
- High confidence (>0.80) on clear-cut cases
- XAI explanations align with human intuition
- Mistakes are understandable (similar materials)
- Performance consistent across different image conditions

Be cautious when:
- Focusing on irrelevant features (backgrounds, text)
- High confidence on ambiguous cases (overconfident)
- Random or scattered XAI patterns
- Large performance drops on new data

## Practical Takeaways

For researchers:
- Multiple XAI methods provide different views
- Agreement between methods increases confidence
- Confusion patterns reveal dataset biases
- Failure analysis guides improvements

For practitioners:
- Explanations enable system debugging
- Visual outputs communicate with non-technical staff
- Trustworthy when reasoning matches domain knowledge
- Valuable for training and quality control

For policymakers:
- Transparent decision-making supports compliance
- Audit trails for accountability
- Risk assessment through failure analysis
- Evidence-based deployment decisions
