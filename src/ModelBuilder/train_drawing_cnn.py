"""
ParkiSense - Drawing CNN Trainer
=================================
Trains a MobileNetV2-based image classifier on Parkinson's spiral drawings.

Strategy:
  1. Try to use the vikasukani Kaggle dataset if available in ./spiral_data/
  2. Otherwise, generate synthetic spiral images (healthy=smooth, parkinson=shaky)
  3. Train MobileNetV2 with transfer learning
  4. Save model to ../../keras_model.h5

Usage:
    python train_drawing_cnn.py
"""

import os, sys, math, random, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR      = "./spiral_data_clean"  # Real Kaggle spiral images (Healthy/ + Parkinson/ only)
MODEL_OUT     = "../../keras_model.h5"
IMG_SIZE      = 224
BATCH         = 16
EPOCHS        = 20
SYNTH_COUNT   = 300   # images per class for synthetic generation
SEED          = 42
random.seed(SEED)
np.random.seed(SEED)

# ─── STEP 1: Generate synthetic data if real dataset not found ────────────────
def draw_spiral(draw, cx, cy, radius, loops, noise=0.0, color=0):
    """Draw an Archimedean spiral with optional noise (tremor simulation)."""
    points = []
    total_angle = loops * 2 * math.pi
    steps = 600
    for i in range(steps):
        angle = (i / steps) * total_angle
        r = (i / steps) * radius
        nx = random.gauss(0, noise)
        ny = random.gauss(0, noise)
        x = cx + (r + nx) * math.cos(angle)
        y = cy + (r + ny) * math.sin(angle)
        points.append((x, y))
    for i in range(len(points) - 1):
        draw.line([points[i], points[i+1]], fill=color, width=2)

def generate_synthetic_dataset(out_dir, count_per_class=SYNTH_COUNT):
    """Generate synthetic spiral images for Healthy and Parkinson classes."""
    for cls, noise in [("Healthy", 0.8), ("Parkinson", 6.5)]:
        cls_dir = os.path.join(out_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        print(f"  Generating {count_per_class} '{cls}' images (noise={noise})...")
        for i in range(count_per_class):
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            # Vary spiral parameters slightly for diversity
            loops = random.uniform(3.5, 4.5)
            radius = random.uniform(80, 95)
            extra_noise = noise * random.uniform(0.8, 1.3)
            draw_spiral(draw, IMG_SIZE//2, IMG_SIZE//2, radius, loops, extra_noise)
            # Add slight blur for realism
            img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
            img.save(os.path.join(cls_dir, f"{cls}_{i:04d}.png"))
    print(f"  Synthetic dataset ready at: {out_dir}")

def find_or_create_dataset():
    """Return dataset dir with Healthy/ and Parkinson/ subdirs."""
    # Check if kaggle dataset already extracted here
    healthy_dir  = os.path.join(DATA_DIR, "Healthy")
    parkinson_dir = os.path.join(DATA_DIR, "Parkinson")
    if os.path.isdir(healthy_dir) and os.path.isdir(parkinson_dir):
        h_count = len([f for f in os.listdir(healthy_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        p_count = len([f for f in os.listdir(parkinson_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        if h_count >= 10 and p_count >= 10:
            print(f"  Found real dataset: {h_count} Healthy, {p_count} Parkinson images.")
            return DATA_DIR
    # Try common kaggle extraction patterns
    for sub in ["spiral/training", "spiral", "dataset/spiral/training"]:
        alt = os.path.join(DATA_DIR, sub)
        if os.path.isdir(os.path.join(alt, "healthy")) or os.path.isdir(os.path.join(alt, "parkinson")):
            merged = os.path.join(DATA_DIR, "merged")
            os.makedirs(os.path.join(merged, "Healthy"), exist_ok=True)
            os.makedirs(os.path.join(merged, "Parkinson"), exist_ok=True)
            for src_name, dst_name in [("healthy","Healthy"),("parkinson","Parkinson")]:
                src = os.path.join(alt, src_name)
                if os.path.isdir(src):
                    for f in os.listdir(src):
                        shutil.copy(os.path.join(src, f), os.path.join(merged, dst_name, f))
            return merged
    # Fallback: generate synthetic
    print("\n  [INFO] Real dataset not found. Generating synthetic spiral images...")
    synth_dir = os.path.join(DATA_DIR, "synthetic")
    generate_synthetic_dataset(synth_dir, SYNTH_COUNT)
    return synth_dir

# ─── STEP 2: Train CNN ────────────────────────────────────────────────────────
def train():
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import confusion_matrix, classification_report

    tf.random.set_seed(SEED)
    print("\n" + "="*60)
    print("ParkiSense - Drawing Model Trainer")
    print("="*60)

    # ── Dataset ──
    print("\n[1/5] Preparing dataset...")
    dataset_dir = find_or_create_dataset()

    # ── Data generators ──
    print("\n[2/5] Building data pipeline...")
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )
    val_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_ds = train_gen.flow_from_directory(
        dataset_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        class_mode='binary',
        subset='training',
        seed=SEED
    )
    val_ds = val_gen.flow_from_directory(
        dataset_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        class_mode='binary',
        subset='validation',
        seed=SEED
    )

    class_names = list(train_ds.class_indices.keys())
    print(f"  Classes: {train_ds.class_indices}")
    print(f"  Training samples: {train_ds.samples}")
    print(f"  Validation samples: {val_ds.samples}")

    # ── Model ──
    print("\n[3/5] Building MobileNetV2 model...")
    base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    base.trainable = False  # Freeze base for transfer learning

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # ── Train (frozen base) ──
    print("\n[4/5] Training... (Phase 1 - top layers)")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss')
    ]
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    # ── Fine-tune top layers of base ──
    print("\n  Fine-tuning top 30 layers of MobileNetV2...")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    # ── Evaluate ──
    print("\n[5/5] Evaluating model...")
    val_ds.reset()
    y_pred = (model.predict(val_ds) > 0.5).astype(int).flatten()
    y_true = val_ds.classes[:len(y_pred)]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\nFinal Validation Accuracy: {val_acc*100:.2f}%")

    # ── Save ──
    out_path = os.path.abspath(MODEL_OUT)
    # Backup existing model
    if os.path.exists(out_path):
        bak = out_path.replace('.h5', '.h5.bak2')
        shutil.copy(out_path, bak)
        print(f"\nExisting model backed up to: {bak}")
    model.save(out_path)
    print(f"New model saved to:          {out_path}")

    # Save class indices for reference
    with open("../../labels.txt", "w") as f:
        for cls_name, idx in sorted(train_ds.class_indices.items(), key=lambda x: x[1]):
            f.write(f"{idx} {cls_name}\n")
    print("labels.txt updated.")

    print("\n" + "="*60)
    print(f"Training complete! Accuracy: {val_acc*100:.2f}%")
    print("="*60)

if __name__ == "__main__":
    train()
