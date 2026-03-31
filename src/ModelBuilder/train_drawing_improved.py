

import os, sys, shutil, math, random
import numpy as np

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "spiral_data")
CLEAN_DIR  = os.path.join(BASE_DIR, "spiral_data_improved")
MODEL_OUT  = os.path.join(BASE_DIR, "..", "..", "keras_model.h5")
IMG_SIZE   = 224
BATCH      = 16
EPOCHS_P1  = 30   # frozen base
EPOCHS_P2  = 20   # fine-tune
SEED       = 42

random.seed(SEED)
np.random.seed(SEED)

def build_dataset():
    
    for cls in ["Healthy", "Parkinson"]:
        d = os.path.join(CLEAN_DIR, cls)
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    mapping = {"healthy": "Healthy", "parkinson": "Parkinson"}
    counts  = {"Healthy": 0, "Parkinson": 0}

    for drawing_type in ["spiral", "wave"]:
        for split in ["training", "testing"]:
            for src_cls, dst_cls in mapping.items():
                src = os.path.join(DATA_DIR, drawing_type, split, src_cls)
                if os.path.isdir(src):
                    for f in os.listdir(src):
                        if f.lower().endswith((".png", ".jpg", ".jpeg")):
                            dest_name = f"{drawing_type}_{split}_{f}"
                            shutil.copy2(
                                os.path.join(src, f),
                                os.path.join(CLEAN_DIR, dst_cls, dest_name)
                            )
                            counts[dst_cls] += 1

    print("\n[Dataset Summary]")
    print(f"  Healthy  : {counts['Healthy']} images")
    print(f"  Parkinson: {counts['Parkinson']} images")
    print(f"  Total    : {sum(counts.values())} images in 2 classes")
    return counts

def train(counts):
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import classification_report, confusion_matrix

    tf.random.set_seed(SEED)

    print("\n" + "="*60)
    print("ParkiSense - Improved Drawing Model Trainer")
    print(f"  Backbone   : EfficientNetB0")
    print(f"  Dataset    : {sum(counts.values())} real images (spiral + wave)")
    print(f"  Image size : {IMG_SIZE}x{IMG_SIZE}")
    print("="*60)

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode="nearest",
        validation_split=0.2
    )
    val_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_ds = train_gen.flow_from_directory(
        CLEAN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        class_mode="binary",
        subset="training",
        seed=SEED,
        shuffle=True
    )
    val_ds = val_gen.flow_from_directory(
        CLEAN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        class_mode="binary",
        subset="validation",
        seed=SEED,
        shuffle=False
    )

    class_names = list(train_ds.class_indices.keys())
    print(f"\n  Classes  : {train_ds.class_indices}")
    print(f"  Train    : {train_ds.samples} samples")
    print(f"  Val      : {val_ds.samples} samples")

    print("\n[Building EfficientNetB0 model...]")
    base = EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)

    print(f"\n[Phase 1] Training head ({EPOCHS_P1} epochs, frozen base)...")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=8, restore_best_weights=True, monitor="val_accuracy", verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=4, monitor="val_loss", verbose=1, min_lr=1e-6
        ),
    ]

    n_healthy   = counts["Healthy"]
    n_parkinson = counts["Parkinson"]
    total       = n_healthy + n_parkinson
    class_weight = {
        0: total / (2.0 * n_healthy),
        1: total / (2.0 * n_parkinson)
    }
    print(f"  Class weights: {class_weight}")

    model.fit(
        train_ds,
        epochs=EPOCHS_P1,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    print(f"\n[Phase 2] Fine-tuning top 40 layers ({EPOCHS_P2} epochs)...")
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        epochs=EPOCHS_P2,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    print("\n[Evaluation]")
    val_ds.reset()
    y_pred_proba = model.predict(val_ds, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    y_true = val_ds.classes[:len(y_pred)]

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\nFinal Validation Accuracy: {val_acc*100:.2f}%")

    out_path = os.path.abspath(MODEL_OUT)
    if os.path.exists(out_path):
        bak = out_path.replace(".h5", ".h5.bak_improved")
        shutil.copy(out_path, bak)
        print(f"\nPrevious model backed up to: {bak}")
    model.save(out_path)
    print(f"New improved model saved to : {out_path}")

    labels_path = os.path.join(BASE_DIR, "..", "..", "labels.txt")
    with open(labels_path, "w") as f:
        for cls_name, idx in sorted(train_ds.class_indices.items(), key=lambda x: x[1]):
            f.write(f"{idx} {cls_name}\n")
    print("labels.txt updated.")

    print("\n" + "="*60)
    print(f"TRAINING COMPLETE — Accuracy: {val_acc*100:.2f}%")
    print("="*60)

if __name__ == "__main__":
    print("="*60)
    print("Building improved dataset (spiral + wave, all splits)...")
    print("="*60)
    counts = build_dataset()
    train(counts)
