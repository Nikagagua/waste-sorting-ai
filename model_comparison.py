"""
Enhanced Waste Sorting CNN Model with Comprehensive XAI Analysis
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import (
    ResNet50,
    EfficientNetV2B0,
    MobileNetV2,
    DenseNet121,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import Image
from lime import lime_image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from skimage.segmentation import mark_boundaries
import cv2
import json


# Configuration
DATA_ROOT = os.environ.get("WASTE_DATA_ROOT", "./data")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
OUTPUT_DIR = "enhanced_paper_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create subdirectories for organization
os.makedirs(f"{OUTPUT_DIR}/confusion_matrices", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/training_curves", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/xai_visualizations", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/per_class_metrics", exist_ok=True)


def load_data():
    """Load train, val, test generators with enhanced augmentation."""
    if not os.path.exists(DATA_ROOT):
        raise FileNotFoundError(f"\nData directory not found: {DATA_ROOT}")

    for split in ["train", "val", "test"]:
        split_path = os.path.join(DATA_ROOT, split)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"\nMissing {split} directory: {split_path}")

    # Enhanced augmentation based on state-of-the-art approaches
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,  # Increased from 15
        width_shift_range=0.1,  # Increased from 0.05
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,  # Added vertical flip
        zoom_range=0.1,  # Added zoom
        brightness_range=[0.8, 1.2],  # Added brightness variation
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train = train_datagen.flow_from_directory(
        os.path.join(DATA_ROOT, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )
    val = val_datagen.flow_from_directory(
        os.path.join(DATA_ROOT, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    test = val_datagen.flow_from_directory(
        os.path.join(DATA_ROOT, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return train, val, test, list(train.class_indices.keys())


def build_model(arch, num_classes, use_dropout=True, dropout_rate=0.3):
    """Build and compile model with improved architecture."""
    base_models = {
        "ResNet50": ResNet50,
        "EfficientNetB0": EfficientNetV2B0,
        "MobileNetV2": MobileNetV2,
        "DenseNet121": DenseNet121,  # Added DenseNet121 based on SOTA
    }

    try:
        base = base_models[arch](
            include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,)
        )
    except (ValueError, OSError):
        base = base_models[arch](
            include_top=False, weights=None, input_shape=IMG_SIZE + (3,)
        )

    # Fine-tuning: Unfreeze last few layers for better accuracy
    base.trainable = True
    # Freeze initial layers
    for layer in base.layers[:-20]:  # Unfreeze last 20 layers
        layer.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # Enhanced dense layers
    if use_dropout:
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation="relu")(x)  # Added dense layer
    if use_dropout:
        x = layers.Dropout(dropout_rate / 2)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    # Use different learning rates for different layers
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
        ],
    )
    return model


def train_and_evaluate(arch, train_gen, val_gen, test_gen):
    """Train model and return comprehensive metrics."""
    print(f"\n{'=' * 60}")
    print(f"Training {arch}...")
    print(f"{'=' * 60}")

    arch_key = arch
    model = build_model(arch_key, len(train_gen.class_indices))

    # Compute class weights to handle class imbalance
    class_weights_arr = compute_class_weight(
        "balanced", classes=np.unique(train_gen.classes), y=train_gen.classes
    )
    class_weights_dict = {i: w for i, w in enumerate(class_weights_arr)}
    print(f"Class weights: {class_weights_dict}")

    # Enhanced callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
        ),
        ModelCheckpoint(
            f"{OUTPUT_DIR}/{arch}_best_model.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set
    test_gen.reset()
    y_true = test_gen.classes
    y_pred_proba = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Calculate comprehensive metrics
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        target_names=list(train_gen.class_indices.keys()),
        digits=4,
    )
    cm = confusion_matrix(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(train_gen.class_indices))
    )

    print(f"\n{arch} Results:")
    print(f"Overall Accuracy: {report['accuracy']:.4f}")
    print(f"Macro Avg F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}")

    return {
        "model": model,
        "history": history,
        "accuracy": report["accuracy"],
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
        "cm": cm,
        "report": report,
        "y_pred": y_pred,
        "y_true": y_true,
        "y_pred_proba": y_pred_proba,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "per_class_support": support,
    }


def plot_training_curves(results_dict):
    """Generate individual training curves for each model."""
    for name, res in results_dict.items():
        history = res["history"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy plot
        axes[0].plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
        axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
        axes[0].set_title(
            f"{name} - Accuracy over Epochs", fontsize=14, fontweight="bold"
        )
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("Accuracy", fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # Loss plot
        axes[1].plot(history.history["loss"], label="Train Loss", linewidth=2)
        axes[1].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
        axes[1].set_title(f"{name} - Loss over Epochs", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("Loss", fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{OUTPUT_DIR}/training_curves/{name}_training_curves.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"Saved training curves for {name}")


def generate_comparison_table(results_dict, class_names):
    """Generate enhanced performance comparison table."""
    data = []
    for name, res in results_dict.items():
        data.append(
            {
                "Model": name,
                "Accuracy": f"{res['accuracy']:.4f}",
                "Precision": f"{res['weighted_avg']['precision']:.4f}",
                "Recall": f"{res['weighted_avg']['recall']:.4f}",
                "F1-Score": f"{res['weighted_avg']['f1-score']:.4f}",
                "Macro F1": f"{res['macro_avg']['f1-score']:.4f}",
            }
        )

    df = pd.DataFrame(data)
    df = df.sort_values("Accuracy", ascending=False)

    # Save to CSV
    df.to_csv(f"{OUTPUT_DIR}/table_model_comparison.csv", index=False)

    # Print to console
    print("\n" + "=" * 80)
    print("MODEL COMPARISON TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    # Save to LaTeX
    with open(f"{OUTPUT_DIR}/table_model_comparison.tex", "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{Performance Comparison of CNN Architectures on Waste Classification}\n"
        )
        f.write("\\label{tab:model_comparison}\n")
        f.write(df.to_latex(index=False, escape=False))
        f.write("\\end{table}\n")

    return df


def plot_individual_confusion_matrices(results_dict, class_names):
    """Generate individual confusion matrix for each model."""
    for name, res in results_dict.items():
        fig, ax = plt.subplots(figsize=(10, 8))

        # Normalize confusion matrix
        cm_norm = res["cm"].astype("float") / res["cm"].sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            ax=ax,
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Normalized Frequency"},
            vmin=0,
            vmax=1,
            square=True,
        )

        ax.set_title(
            f"{name} - Confusion Matrix\nAccuracy: {res['accuracy']:.4f}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            f"{OUTPUT_DIR}/confusion_matrices/{name}_confusion_matrix.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"Saved confusion matrix for {name}")


def plot_per_class_metrics(results_dict, class_names):
    """Generate per-class performance metrics for each model."""
    for name, res in results_dict.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        precision = res["per_class_precision"]
        recall = res["per_class_recall"]
        f1 = res["per_class_f1"]

        x = np.arange(len(class_names))
        width = 0.25

        # Precision
        axes[0].bar(
            x, precision, width, label="Precision", color="steelblue", alpha=0.8
        )
        axes[0].set_xlabel("Class", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Precision", fontsize=12, fontweight="bold")
        axes[0].set_title(
            f"{name} - Per-Class Precision", fontsize=13, fontweight="bold"
        )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(class_names, rotation=45, ha="right")
        axes[0].set_ylim([0, 1.05])
        axes[0].grid(axis="y", alpha=0.3)

        # Recall
        axes[1].bar(x, recall, width, label="Recall", color="forestgreen", alpha=0.8)
        axes[1].set_xlabel("Class", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Recall", fontsize=12, fontweight="bold")
        axes[1].set_title(f"{name} - Per-Class Recall", fontsize=13, fontweight="bold")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(class_names, rotation=45, ha="right")
        axes[1].set_ylim([0, 1.05])
        axes[1].grid(axis="y", alpha=0.3)

        # F1-Score
        axes[2].bar(x, f1, width, label="F1-Score", color="coral", alpha=0.8)
        axes[2].set_xlabel("Class", fontsize=12, fontweight="bold")
        axes[2].set_ylabel("F1-Score", fontsize=12, fontweight="bold")
        axes[2].set_title(
            f"{name} - Per-Class F1-Score", fontsize=13, fontweight="bold"
        )
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(class_names, rotation=45, ha="right")
        axes[2].set_ylim([0, 1.05])
        axes[2].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{OUTPUT_DIR}/per_class_metrics/{name}_per_class_metrics.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"Saved per-class metrics for {name}")


def grad_cam(model, img_array, pred_class_idx, layer_name=None):
    """Generate Grad-CAM heatmap."""
    # Find the last convolutional layer if not specified
    if layer_name is None:
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:  # Convolutional layer
                layer_name = layer.name
                break

    grad_model = Model(
        inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(img_array, 0))
        loss = predictions[:, pred_class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, IMG_SIZE)

    return heatmap


def occlusion_sensitivity(model, img_array, pred_class_idx, patch_size=20, stride=10):
    """Generate occlusion sensitivity heatmap."""
    height, width = IMG_SIZE
    heatmap = np.zeros((height, width))

    baseline_pred = model.predict(np.expand_dims(img_array, 0), verbose=0)[0][
        pred_class_idx
    ]

    for y in range(0, height - patch_size, stride):
        for x in range(0, width - patch_size, stride):
            occluded_img = img_array.copy()
            occluded_img[y : y + patch_size, x : x + patch_size] = 0.5

            occluded_pred = model.predict(np.expand_dims(occluded_img, 0), verbose=0)[
                0
            ][pred_class_idx]
            importance = baseline_pred - occluded_pred

            heatmap[y : y + patch_size, x : x + patch_size] += importance

    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


def integrated_gradients(model, img_array, pred_class_idx, baseline=None, steps=50):
    """Generate integrated gradients attribution map."""
    if baseline is None:
        baseline = np.zeros_like(img_array)

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    baseline_tensor = tf.convert_to_tensor(baseline, dtype=tf.float32)

    alphas = tf.linspace(0.0, 1.0, steps + 1)
    gradient_batches = []

    for alpha in alphas:
        interpolated = baseline_tensor + alpha * (img_tensor - baseline_tensor)
        interpolated = tf.expand_dims(interpolated, 0)

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            preds = model(interpolated)
            target_class = preds[:, pred_class_idx]

        grads = tape.gradient(target_class, interpolated)
        gradient_batches.append(grads[0])

    grads = tf.stack(gradient_batches)
    avg_grads = tf.reduce_mean(grads, axis=0)

    integrated_grads = (img_tensor - baseline_tensor) * avg_grads
    attribution = tf.reduce_sum(tf.abs(integrated_grads), axis=-1)

    attribution = attribution.numpy()
    if attribution.max() > 0:
        attribution = attribution / attribution.max()

    return attribution


def generate_xai_figure(model, test_path, class_names, model_name, num_samples=6):
    """Generate comprehensive XAI visualizations with individual saves."""
    sample_images = []

    # Select samples from different classes
    for cls in class_names[:num_samples]:
        cls_path = Path(test_path) / cls
        if cls_path.exists():
            imgs = sorted(list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.png")))
            if imgs:
                sample_images.append((str(imgs[0]), cls))

    if not sample_images:
        print("Warning: No sample images found for XAI visualization.")
        return

    # Create directory for this model's XAI visualizations
    model_xai_dir = f"{OUTPUT_DIR}/xai_visualizations/{model_name}"
    os.makedirs(model_xai_dir, exist_ok=True)

    for idx, (img_path, true_class) in enumerate(sample_images):
        img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        img_array = np.array(img) / 255.0

        pred = model.predict(np.expand_dims(img_array, 0), verbose=0)[0]
        pred_class_idx = np.argmax(pred)
        pred_class = class_names[pred_class_idx]
        confidence = np.max(pred)

        # Create individual figure for this sample
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        # Original Image
        axes[0].imshow(img)
        axes[0].set_title(
            f"Original\nTrue: {true_class}\nPred: {pred_class}\nConf: {confidence:.3f}",
            fontsize=10,
            fontweight="bold",
        )
        axes[0].axis("off")

        # Grad-CAM
        try:
            print(f"  Computing Grad-CAM for {true_class}...")
            gradcam_map = grad_cam(model, img_array, pred_class_idx)
            gradcam_resized = np.uint8(255 * gradcam_map)

            axes[1].imshow(img)
            axes[1].imshow(gradcam_resized, cmap="jet", alpha=0.5)
            axes[1].set_title("Grad-CAM", fontsize=10, fontweight="bold")
            axes[1].axis("off")
        except Exception as e:
            print(f"    Grad-CAM error: {str(e)}")
            axes[1].imshow(img)
            axes[1].set_title("Grad-CAM (error)", fontsize=10)
            axes[1].axis("off")

        # Occlusion Sensitivity
        try:
            print(f"  Computing Occlusion for {true_class}...")
            occlusion_map = occlusion_sensitivity(model, img_array, pred_class_idx)
            occlusion_resized = np.uint8(255 * occlusion_map)

            axes[2].imshow(img)
            axes[2].imshow(occlusion_resized, cmap="jet", alpha=0.5)
            axes[2].set_title("Occlusion", fontsize=10, fontweight="bold")
            axes[2].axis("off")
        except Exception as e:
            print(f"    Occlusion error: {str(e)}")
            axes[2].imshow(img)
            axes[2].set_title("Occlusion (error)", fontsize=10)
            axes[2].axis("off")

        # Integrated Gradients
        try:
            print(f"  Computing Integrated Gradients for {true_class}...")
            ig_map = integrated_gradients(model, img_array, pred_class_idx)
            ig_resized = np.uint8(255 * ig_map)

            axes[3].imshow(img)
            axes[3].imshow(ig_resized, cmap="hot", alpha=0.5)
            axes[3].set_title("Integrated Gradients", fontsize=10, fontweight="bold")
            axes[3].axis("off")
        except Exception as e:
            print(f"    Integrated Gradients error: {str(e)}")
            axes[3].imshow(img)
            axes[3].set_title("Int. Gradients (error)", fontsize=10)
            axes[3].axis("off")

        # LIME
        try:
            print(f"  Computing LIME for {true_class}...")
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                img_array.astype(np.float64),
                lambda x: model.predict(x, verbose=0),
                top_labels=1,
                num_samples=300,
                random_seed=42,
            )
            temp, mask = explanation.get_image_and_mask(
                pred_class_idx, positive_only=True, num_features=5, hide_rest=False
            )

            lime_vis = mark_boundaries(temp, mask)
            axes[4].imshow(lime_vis)
            axes[4].set_title("LIME", fontsize=10, fontweight="bold")
            axes[4].axis("off")
        except Exception as e:
            print(f"    LIME error: {str(e)}")
            axes[4].imshow(img)
            axes[4].set_title("LIME (error)", fontsize=10)
            axes[4].axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{model_xai_dir}/{true_class}_sample_{idx}_xai.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"  Saved XAI visualization for {true_class} (sample {idx})")


def save_detailed_report(results_dict, class_names):
    """Save comprehensive detailed report."""
    report_path = f"{OUTPUT_DIR}/detailed_report.txt"

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE WASTE SORTING MODEL EVALUATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for name, res in results_dict.items():
            f.write(f"\n{'=' * 80}\n")
            f.write(f"MODEL: {name}\n")
            f.write(f"{'=' * 80}\n\n")

            f.write(f"Overall Metrics:\n")
            f.write(
                f"  Accuracy: {res['accuracy']:.4f} ({res['accuracy'] * 100:.2f}%)\n"
            )
            f.write(f"  Macro Avg Precision: {res['macro_avg']['precision']:.4f}\n")
            f.write(f"  Macro Avg Recall: {res['macro_avg']['recall']:.4f}\n")
            f.write(f"  Macro Avg F1-Score: {res['macro_avg']['f1-score']:.4f}\n")
            f.write(
                f"  Weighted Avg Precision: {res['weighted_avg']['precision']:.4f}\n"
            )
            f.write(f"  Weighted Avg Recall: {res['weighted_avg']['recall']:.4f}\n")
            f.write(
                f"  Weighted Avg F1-Score: {res['weighted_avg']['f1-score']:.4f}\n\n"
            )

            f.write("Per-Class Performance:\n")
            f.write(
                f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n"
            )
            f.write("-" * 80 + "\n")

            for i, cls in enumerate(class_names):
                f.write(
                    f"{cls:<15} {res['per_class_precision'][i]:<12.4f} "
                    f"{res['per_class_recall'][i]:<12.4f} "
                    f"{res['per_class_f1'][i]:<12.4f} "
                    f"{int(res['per_class_support'][i]):<10}\n"
                )

            f.write("\n")

    print(f"\nDetailed report saved to: {report_path}")


def main():
    print("=" * 80)
    print("ENHANCED WASTE SORTING CNN MODEL WITH COMPREHENSIVE XAI ANALYSIS")
    print("=" * 80)

    print("\n[1/7] Loading data...")
    train_gen, val_gen, test_gen, class_names = load_data()
    print(f"Classes: {class_names}")
    print(
        f"Train: {train_gen.samples}, Val: {val_gen.samples}, Test: {test_gen.samples}"
    )
    print(f"Class distribution in training set:")
    for cls, count in zip(class_names, np.bincount(train_gen.classes)):
        print(f"  {cls}: {count} samples ({count / train_gen.samples * 100:.1f}%)")

    print("\n[2/7] Training models...")
    # Include DenseNet121 based on SOTA results
    models = ["ResNet50", "EfficientNetB0", "MobileNetV2", "DenseNet121"]
    results = {}

    for arch in models:
        results[arch] = train_and_evaluate(arch, train_gen, val_gen, test_gen)

    print("\n[3/7] Generating comparison table...")
    comparison_df = generate_comparison_table(results, class_names)

    print("\n[4/7] Generating training curves...")
    plot_training_curves(results)

    print("\n[5/7] Generating confusion matrices...")
    plot_individual_confusion_matrices(results, class_names)

    print("\n[6/7] Generating per-class metrics...")
    plot_per_class_metrics(results, class_names)

    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
    best_model_name = best_model[0]

    print(f"\n[7/7] Generating XAI visualizations for {best_model_name}...")
    generate_xai_figure(
        best_model[1]["model"],
        os.path.join(DATA_ROOT, "test"),
        class_names,
        best_model_name,
        num_samples=len(class_names),
    )

    print("\n[8/7] Saving detailed report...")
    save_detailed_report(results, class_names)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nBest model: {best_model_name}")
    print(
        f"  Accuracy: {best_model[1]['accuracy']:.4f} ({best_model[1]['accuracy'] * 100:.2f}%)"
    )
    print(f"  Macro F1: {best_model[1]['macro_avg']['f1-score']:.4f}")
    print(f"  Weighted F1: {best_model[1]['weighted_avg']['f1-score']:.4f}")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
