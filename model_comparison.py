"""
Waste Sorting CNN Model Comparison with XAI Analysis
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50, EfficientNetV2B0, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import Image
from lime import lime_image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from skimage.segmentation import mark_boundaries
import cv2


DATA_ROOT = os.environ.get("WASTE_DATA_ROOT", "./data")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
OUTPUT_DIR = "paper_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load train, val, test generators."""
    if not os.path.exists(DATA_ROOT):
        raise FileNotFoundError(f"\nData directory not found: {DATA_ROOT}")

    for split in ["train", "val", "test"]:
        split_path = os.path.join(DATA_ROOT, split)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"\nMissing {split} directory: {split_path}")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
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


def build_model(arch, num_classes):
    """Build and compile model."""
    base_models = {
        "ResNet50": ResNet50,
        "EfficientNetB0": EfficientNetV2B0,
        "MobileNetV2": MobileNetV2,
    }

    try:
        base = base_models[arch](
            include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,)
        )
    except (ValueError, OSError):
        base = base_models[arch](
            include_top=False, weights=None, input_shape=IMG_SIZE + (3,)
        )

    base.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(5e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate(arch, train_gen, val_gen, test_gen):
    """Train model and return metrics."""
    print(f"\nTraining {arch}...")
    arch_key = "EfficientNetB0" if arch == "EfficientNetV2B0" else arch
    model = build_model(arch_key, len(train_gen.class_indices))

    # Compute class weights to handle class imbalance
    class_weights_arr = compute_class_weight(
        "balanced", classes=np.unique(train_gen.classes), y=train_gen.classes
    )
    class_weights_dict = {i: w for i, w in enumerate(class_weights_arr)}

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights_dict,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1,
    )

    test_gen.reset()
    y_true = test_gen.classes
    y_pred = np.argmax(model.predict(test_gen, verbose=0), axis=1)

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        target_names=list(train_gen.class_indices.keys()),
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "model": model,
        "history": history,
        "accuracy": report["accuracy"],
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
        "cm": cm,
        "report": report,
    }


def generate_comparison_table(results_dict, class_names):
    """Generate performance comparison table."""
    data = []
    display_names = {
        "ResNet50": "ResNet50",
        "EfficientNetB0": "EfficientNetV2B0",
        "MobileNetV2": "MobileNetV2",
    }
    for name, res in results_dict.items():
        data.append(
            {
                "Model": display_names.get(name, name),
                "Accuracy": f"{res['accuracy']:.3f}",
                "Precision": f"{res['weighted_avg']['precision']:.3f}",
                "Recall": f"{res['weighted_avg']['recall']:.3f}",
                "F1-Score": f"{res['weighted_avg']['f1-score']:.3f}",
            }
        )

    df = pd.DataFrame(data)
    df.to_csv(f"{OUTPUT_DIR}/table2_model_comparison.csv", index=False)
    print(df.to_string(index=False))

    with open(f"{OUTPUT_DIR}/table2_model_comparison.tex", "w") as f:
        f.write(
            "\\begin{table}[h]\n\\centering\n\\caption{Performance Comparison of CNN Architectures}\n"
        )
        f.write("\\label{tab:model_comparison}\n")
        f.write(df.to_latex(index=False, escape=False))
        f.write("\\end{table}\n")

    return df


def plot_confusion_matrices(results_dict, class_names):
    """Generate confusion matrices visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    display_names = {
        "ResNet50": "ResNet50",
        "EfficientNetB0": "EfficientNetV2B0",
        "MobileNetV2": "MobileNetV2",
    }

    for ax, (name, res) in zip(axes, results_dict.items()):
        cm_norm = res["cm"].astype("float") / res["cm"].sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            ax=ax,
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Frequency"},
            vmin=0,
            vmax=1,
        )

        ax.set_title(
            f"{display_names.get(name, name)}\nAcc: {res['accuracy']:.3f}",
            fontweight="bold",
        )
        ax.set_ylabel("True Label" if ax == axes[0] else "")
        ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/figure2_confusion_matrices.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


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


def saliency_map(model, img_array, pred_class_idx):
    """Generate saliency map using gradients."""
    img_tensor = tf.convert_to_tensor(np.expand_dims(img_array, 0), dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor)
        target_class = preds[:, pred_class_idx]

    grads = tape.gradient(target_class, img_tensor)
    grads = tf.abs(grads)
    grads = tf.reduce_max(grads, axis=-1)
    grads = grads[0].numpy()

    if grads.max() > 0:
        grads = grads / grads.max()

    return grads


def generate_xai_figure(model, test_path, class_names, model_name):
    """Generate XAI visualizations using multiple methods."""

    sample_images = []
    for cls in class_names[:6]:
        cls_path = Path(test_path) / cls
        if cls_path.exists():
            imgs = sorted(list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.png")))
            if imgs:
                sample_images.append((str(imgs[0]), cls))

    if not sample_images:
        print("Warning: No sample images found for XAI visualization.")
        return

    fig, axes = plt.subplots(
        len(sample_images), 4, figsize=(16, 3 * len(sample_images))
    )
    if len(sample_images) == 1:
        axes = axes.reshape(1, -1)

    for i, (img_path, true_class) in enumerate(sample_images):
        img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        img_array = np.array(img) / 255.0

        pred = model.predict(np.expand_dims(img_array, 0), verbose=0)[0]
        pred_class_idx = np.argmax(pred)
        pred_class = class_names[pred_class_idx]
        confidence = np.max(pred)

        # Original Image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(
            f"Original\nTrue: {true_class}\nPred: {pred_class} ({confidence:.2f})"
        )
        axes[i, 0].axis("off")

        # Occlusion Sensitivity
        try:
            print(f"Computing Occlusion for {true_class}...")
            occlusion_map = occlusion_sensitivity(model, img_array, pred_class_idx)
            occlusion_resized = cv2.resize(occlusion_map, IMG_SIZE)
            occlusion_resized = np.uint8(255 * occlusion_resized)

            axes[i, 1].imshow(img)
            axes[i, 1].imshow(occlusion_resized, cmap="jet", alpha=0.5)
            axes[i, 1].set_title("Occlusion Sensitivity")
            axes[i, 1].axis("off")
        except Exception as e:
            print(f"Occlusion error for {true_class}: {str(e)}")
            axes[i, 1].imshow(img)
            axes[i, 1].set_title("Occlusion (error)")
            axes[i, 1].axis("off")

        # Integrated Gradients
        try:
            print(f"Computing Integrated Gradients for {true_class}...")
            ig_map = integrated_gradients(model, img_array, pred_class_idx)
            ig_resized = np.uint8(255 * ig_map)

            axes[i, 2].imshow(img)
            axes[i, 2].imshow(ig_resized, cmap="hot", alpha=0.5)
            axes[i, 2].set_title("Integrated Gradients")
            axes[i, 2].axis("off")
        except Exception as e:
            print(f"Integrated Gradients error for {true_class}: {str(e)}")
            axes[i, 2].imshow(img)
            axes[i, 2].set_title("Int. Gradients (error)")
            axes[i, 2].axis("off")

        # LIME
        try:
            print(f"Computing LIME for {true_class}...")
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                img_array.astype(np.float64),
                lambda x: model.predict(x, verbose=0),
                top_labels=1,
                num_samples=200,
            )
            temp, mask = explanation.get_image_and_mask(
                pred_class_idx, positive_only=True, num_features=5, hide_rest=False
            )

            lime_vis = mark_boundaries(temp, mask)
            axes[i, 3].imshow(lime_vis)
            axes[i, 3].set_title("LIME")
            axes[i, 3].axis("off")
        except Exception as e:
            print(f"LIME error for {true_class}: {str(e)}")
            axes[i, 3].imshow(img)
            axes[i, 3].set_title("LIME (error)")
            axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/figure3_xai_comparison_{model_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main():
    print("Loading data...")
    train_gen, val_gen, test_gen, class_names = load_data()
    print(f"Classes: {class_names}")
    print(
        f"Train: {train_gen.samples}, Val: {val_gen.samples}, Test: {test_gen.samples}"
    )

    print("\nTraining models...")
    models = ["ResNet50", "EfficientNetB0", "MobileNetV2"]
    results = {}

    for arch in models:
        results[arch] = train_and_evaluate(arch, train_gen, val_gen, test_gen)

    print("\nGenerating comparison table...")
    _ = generate_comparison_table(results, class_names)

    print("\nGenerating figures...")
    plot_confusion_matrices(results, class_names)

    best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
    best_model_name = best_model[0]
    display_model_name = (
        "EfficientNetV2B0" if best_model_name == "EfficientNetB0" else best_model_name
    )

    print(f"\nGenerating XAI visualization for {display_model_name}...")
    generate_xai_figure(
        best_model[1]["model"],
        os.path.join(DATA_ROOT, "test"),
        class_names,
        display_model_name,
    )

    print(
        f"\nBest model: {display_model_name} (accuracy: {best_model[1]['accuracy']:.3f})"
    )
    print(f"Output directory: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
