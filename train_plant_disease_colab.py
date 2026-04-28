"""
Plant disease classifier training (Tomato + Potato) using PlantVillage.

Meets requirements:
- Dataset: PlantVillage (via tensorflow_datasets), only Tomato & Potato, 6 classes max, balanced per-class
- Preprocessing: 224x224 resize, normalized via MobileNetV2 preprocess_input, strong augmentation
- Model: MobileNetV2 pretrained (ImageNet) + GAP + Dense(128, ReLU) + Dropout(0.5) + Softmax
- Training: Adam lr=1e-4, EarlyStopping(val_loss), ReduceLROnPlateau, 10–15 epochs total
- Output: saves "plant_disease_model.h5" and "class_labels.json"

Colab usage:
1) Upload this file to Colab OR paste the contents into a cell.
2) Run it. It will download PlantVillage automatically via TFDS.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import classification_report, confusion_matrix


SEED = 42
IMG_SIZE = (224, 224)
LR_WARMUP = 1e-4
LR_FINETUNE = 1e-5

# Train 25–30 epochs total using warmup + fine-tuning.
EPOCHS_WARMUP = 5
EPOCHS_FINETUNE = 23  # total = 28

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# 6 classes (<=6), 2 crops only (Tomato, Potato).
SELECTED_CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]


def _set_seeds(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    try:
        # Improves determinism when available; safe to ignore in older TF.
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def _build_augmentation() -> tf.keras.Model:
    # Strong augmentation: rotation, zoom, horizontal flip, brightness, translation, contrast.
    layers: List[tf.keras.layers.Layer] = [
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(0.15, seed=SEED),
        tf.keras.layers.RandomZoom(0.2, seed=SEED),
        # width_shift_range / height_shift_range equivalent
        tf.keras.layers.RandomTranslation(height_factor=0.08, width_factor=0.08, seed=SEED),
    ]

    # RandomBrightness exists in recent TF versions; fallback to tf.image otherwise.
    try:
        layers.append(tf.keras.layers.RandomBrightness(0.2, value_range=(0.0, 255.0), seed=SEED))
    except Exception:
        layers.append(
            tf.keras.layers.Lambda(
                lambda x: tf.clip_by_value(tf.image.random_brightness(x, max_delta=0.2 * 255.0), 0.0, 255.0)
            )
        )

    try:
        layers.append(tf.keras.layers.RandomContrast(0.15, seed=SEED))
    except Exception:
        layers.append(
            tf.keras.layers.Lambda(
                lambda x: tf.clip_by_value(tf.image.random_contrast(x, lower=0.9, upper=1.1), 0.0, 255.0)
            )
        )

    return tf.keras.Sequential(layers, name="data_augmentation")


def _load_filtered_balanced_dataset(
    selected_class_names: List[str],
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Returns:
      train_ds, val_ds, class_names_in_order
    """
    ds, info = tfds.load("plant_village", split="train", with_info=True, as_supervised=True)
    all_class_names = list(info.features["label"].names)

    missing = [c for c in selected_class_names if c not in all_class_names]
    if missing:
        raise ValueError(
            "These requested classes were not found in TFDS 'plant_village' labels: "
            + ", ".join(missing)
        )

    selected_original_ids = [all_class_names.index(c) for c in selected_class_names]
    keys = tf.constant(selected_original_ids, dtype=tf.int64)
    vals = tf.constant(list(range(len(selected_original_ids))), dtype=tf.int64)  # remap to 0..K-1

    lookup = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys=keys, values=vals),
        default_value=tf.constant(-1, dtype=tf.int64),
    )

    def keep_and_remap(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        new_label = lookup.lookup(label)
        return image, new_label

    def is_selected(image: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        return lookup.lookup(label) >= 0

    filtered = ds.filter(is_selected).map(keep_and_remap, num_parallel_calls=AUTOTUNE)

    # Count per-class to balance (equal images per class).
    counts: Dict[int, int] = {i: 0 for i in range(len(selected_class_names))}
    for _, y in tfds.as_numpy(filtered):
        counts[int(y)] += 1

    min_count = min(counts.values())
    if min_count == 0:
        raise RuntimeError(f"At least one class has 0 samples after filtering. Counts: {counts}")

    print("Per-class counts (before balancing):")
    for i, name in enumerate(selected_class_names):
        print(f"  {i}: {name} -> {counts[i]}")
    print(f"Balancing to {min_count} images per class.")

    balanced = None
    for cls in range(len(selected_class_names)):
        cls_ds = filtered.filter(lambda x, y, c=cls: tf.equal(y, c)).take(min_count)
        balanced = cls_ds if balanced is None else balanced.concatenate(cls_ds)

    total = min_count * len(selected_class_names)
    balanced = balanced.shuffle(buffer_size=total, seed=SEED, reshuffle_each_iteration=False)

    train_size = int(0.8 * total)
    train_ds = balanced.take(train_size)
    val_ds = balanced.skip(train_size)

    return train_ds, val_ds, selected_class_names


def _preprocess_pipeline(
    ds: tf.data.Dataset,
    *,
    augment: tf.keras.Model | None,
    training: bool,
) -> tf.data.Dataset:
    def preprocess(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.image.resize(image, IMG_SIZE, method="bilinear")
        image = tf.cast(image, tf.float32)  # keep 0..255 range for brightness augmentation
        if training and augment is not None:
            image = augment(image, training=True)
        # Normalization: MobileNetV2 preprocessing maps to [-1, 1].
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def _build_model(num_classes: int) -> Tuple[tf.keras.Model, tf.keras.Model]:
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="plant_disease_mobilenetv2")
    return model, base


def _unfreeze_top_layers_for_finetune(base: tf.keras.Model, *, trainable_layers: int = 30) -> None:
    base.trainable = True
    if trainable_layers <= 0:
        return

    for layer in base.layers[:-trainable_layers]:
        layer.trainable = False
    for layer in base.layers[-trainable_layers:]:
        # Keep BatchNorm frozen for more stable fine-tuning on smaller datasets.
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True


def main() -> None:
    _set_seeds(SEED)

    print("Loading PlantVillage via TFDS and preparing a balanced subset...")
    train_raw, val_raw, class_names = _load_filtered_balanced_dataset(SELECTED_CLASS_NAMES)

    augment = _build_augmentation()
    train_ds = _preprocess_pipeline(train_raw, augment=augment, training=True)
    val_ds = _preprocess_pipeline(val_raw, augment=None, training=False)

    num_classes = len(class_names)
    model, base = _build_model(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_WARMUP),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    checkpoint_path = "plant_disease_model.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\nTraining (warmup: frozen base)...")
    history_warmup = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_WARMUP,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nFine-tuning: unfreezing top layers of MobileNetV2...")
    _unfreeze_top_layers_for_finetune(base, trainable_layers=40)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_FINETUNE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_WARMUP + EPOCHS_FINETUNE,
        initial_epoch=history_warmup.epoch[-1] + 1 if history_warmup.epoch else 0,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate the BEST saved model (by val_accuracy).
    best_model = tf.keras.models.load_model(checkpoint_path)
    val_loss, val_acc = best_model.evaluate(val_ds, verbose=0)
    print(f"\nFinal validation accuracy: {val_acc * 100:.2f}%")

    # Save class labels (model file already saved via checkpoint).
    with open("class_labels.json", "w", encoding="utf-8") as f:
        json.dump({"classes": class_names}, f, indent=2)

    # Confusion matrix + classification report on validation set.
    y_true: List[int] = []
    y_pred: List[int] = []
    for x_batch, y_batch in val_ds:
        probs = best_model.predict(x_batch, verbose=0)
        y_true.extend(y_batch.numpy().tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print('Model training complete')

    # Extra: print last epoch accuracies as requested (training + validation).
    # (These are also shown during fit; this ensures they appear at the end too.)
    last_hist = history_ft.history if history_ft and history_ft.history else history_warmup.history
    if "accuracy" in last_hist and "val_accuracy" in last_hist:
        print(f"Last epoch train accuracy: {last_hist['accuracy'][-1] * 100:.2f}%")
        print(f"Last epoch val accuracy:   {last_hist['val_accuracy'][-1] * 100:.2f}%")


if __name__ == "__main__":
    # Reduce TF log noise for readability (Colab).
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()

