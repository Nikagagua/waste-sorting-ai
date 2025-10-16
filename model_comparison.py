"""
Waste Sorting CNN Model Comparison
Evaluates ResNet50, EfficientNetV2B0, and MobileNetV2 with XAI analysis
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


DATA_ROOT = os.environ.get("WASTE_DATA_ROOT", "./data")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
OUTPUT_DIR = "paper_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load train, val, test generators."""
    if not os.path.exists(DATA_ROOT):
        raise FileNotFoundError(
            f"\nData directory not found: {DATA_ROOT}\n\n"
            f"Please create the directory structure:\n"
            f"  {DATA_ROOT}/train/<class_name>/*.jpg\n"
            f"  {DATA_ROOT}/val/<class_name>/*.jpg\n"
            f"  {DATA_ROOT}/test/<class_name>/*.jpg\n\n"
            f"Or set the WASTE_DATA_ROOT environment variable to your data location."
        )

    for split in ["train", "val", "test"]:
        split_path = os.path.join(DATA_ROOT, split)
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"\nMissing {split} directory: {split_path}\n"
                f"Required structure: {DATA_ROOT}/{train, val, test}/<class_name>/*.jpg"
            )

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
        optimizer=keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate(arch, train_gen, val_gen, test_gen):
    """Train model and return metrics."""
    print(f"\nTraining {arch}...")
    model = build_model(arch, len(train_gen.class_indices))

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
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
    for name, res in results_dict.items():
        data.append(
            {
                "Model": name,
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

        ax.set_title(f"{name}\nAcc: {res['accuracy']:.3f}", fontweight="bold")
        ax.set_ylabel("True Label" if ax == axes[0] else "")
        ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/figure2_confusion_matrices.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def generate_xai_figure(model, test_path, class_names, model_name):
    """Generate XAI visualizations (Grad-CAM, LIME)."""

    def find_last_conv_layer(m):
        """Recursively find the last conv layer."""
        for layer in reversed(m.layers):
            if len(layer.output.shape) == 4:
                return layer
            if hasattr(layer, "layers"):
                found = find_last_conv_layer(layer)
                if found:
                    return found
        return None

    last_conv_layer = find_last_conv_layer(model)

    sample_images = []
    for cls in class_names[:6]:
        cls_path = Path(test_path) / cls
        if cls_path.exists():
            imgs = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.png"))
            if imgs:
                sample_images.append((str(imgs[0]), cls))

    fig, axes = plt.subplots(
        len(sample_images), 3, figsize=(12, 3 * len(sample_images))
    )
    if len(sample_images) == 1:
        axes = axes.reshape(1, -1)

    for i, (img_path, true_class) in enumerate(sample_images):
        img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        img_array = np.array(img) / 255.0

        pred = model.predict(np.expand_dims(img_array, 0), verbose=0)[0]
        pred_class = class_names[np.argmax(pred)]
        confidence = np.max(pred)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(
            f"Original\nTrue: {true_class}\nPred: {pred_class} ({confidence:.2f})"
        )
        axes[i, 0].axis("off")

        if last_conv_layer:
            try:
                grad_model = keras.models.Model(
                    [model.inputs], [last_conv_layer.output, model.output]
                )

                with tf.GradientTape() as tape:
                    conv_out, preds = grad_model(np.expand_dims(img_array, 0))
                    top_idx = tf.argmax(preds[0])
                    top_class = preds[:, top_idx]

                grads = tape.gradient(top_class, conv_out)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)
                heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
                heatmap = heatmap.numpy()

                heatmap_resized = np.array(
                    Image.fromarray((heatmap * 255).astype(np.uint8)).resize(img.size)
                )

                axes[i, 1].imshow(img)
                axes[i, 1].imshow(heatmap_resized, cmap="jet", alpha=0.4)
                axes[i, 1].set_title("Grad-CAM")
                axes[i, 1].axis("off")
            except Exception as e:
                axes[i, 1].text(0.5, 0.5, "Error", ha="center", va="center")
                axes[i, 1].axis("off")
        else:
            axes[i, 1].text(0.5, 0.5, "N/A", ha="center", va="center")
            axes[i, 1].axis("off")

        try:
            from skimage.segmentation import mark_boundaries

            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                (img_array * 255).astype(np.uint8),
                lambda x: model.predict(x / 255.0, verbose=0),
                top_labels=1,
                num_samples=300,
            )
            temp, mask = explanation.get_image_and_mask(
                np.argmax(pred), positive_only=True, num_features=5, hide_rest=False
            )

            lime_vis = mark_boundaries(temp / 255.0, mask)
            axes[i, 2].imshow(lime_vis)
            axes[i, 2].set_title("LIME")
            axes[i, 2].axis("off")
        except:
            axes[i, 2].text(0.5, 0.5, "Error", ha="center", va="center")
            axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/figure3_xai_comparison.png", dpi=300, bbox_inches="tight"
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
    comparison_df = generate_comparison_table(results, class_names)

    print("\nGenerating figures...")
    plot_confusion_matrices(results, class_names)

    best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
    generate_xai_figure(
        best_model[1]["model"],
        os.path.join(DATA_ROOT, "test"),
        class_names,
        best_model[0],
    )

    print(f"\nBest model: {best_model[0]} (accuracy: {best_model[1]['accuracy']:.3f})")
    print(f"Output directory: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
