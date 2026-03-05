"""
ParkiSense – Drawing Model Trainer (Feature-Based)
====================================================
Uses HOG + LBP image features + SVM/Random Forest classifier.
This approach works significantly better than raw CNN on small datasets
(51-102 images per class) and is the method used in published Parkinson's
spiral research (Zham et al., 2017; Pereira et al., 2018).

Saves:
  - drawing_model.pkl   : trained sklearn model
  - drawing_scaler.pkl  : feature scaler
  - labels.txt          : class labels
"""

import os, sys, joblib, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "spiral_data"
MODEL_OUT  = BASE_DIR / ".." / ".." / "drawing_model.pkl"
SCALER_OUT = BASE_DIR / ".." / ".." / "drawing_scaler.pkl"
LABELS_OUT = BASE_DIR / ".." / ".." / "labels.txt"

IMG_SIZE = 128  # resize to 128x128
SEED     = 42


def extract_features(img_gray_arr):
    """Extract HOG + LBP features from a grayscale image array."""
    from skimage.feature import hog, local_binary_pattern
    from skimage.transform import resize

    # Resize to standard size
    img = resize(img_gray_arr, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)

    # HOG features — captures shape/edge orientation
    hog_feats = hog(
        img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )

    # LBP features — captures texture/micro-patterns
    lbp = local_binary_pattern(img, P=24, R=3, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)

    return np.concatenate([hog_feats, lbp_hist])


def load_dataset():
    """Load ALL spiral + wave images from both training and testing splits."""
    from PIL import Image

    X, y = [], []
    class_map = {'healthy': 0, 'parkinson': 1}
    counts = {0: 0, 1: 0}

    for draw_type in ['spiral', 'wave']:
        for split in ['training', 'testing']:
            for cls_name, cls_idx in class_map.items():
                folder = DATA_DIR / draw_type / split / cls_name
                if not folder.is_dir():
                    continue
                for fp in folder.iterdir():
                    if fp.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
                        continue
                    try:
                        img = Image.open(fp).convert('L')
                        arr = np.asarray(img, dtype=np.float32) / 255.0
                        feats = extract_features(arr)
                        X.append(feats)
                        y.append(cls_idx)
                        counts[cls_idx] += 1
                    except Exception as e:
                        print(f"  Skipping {fp.name}: {e}")

    print(f"\n  Healthy  : {counts[0]} images")
    print(f"  Parkinson: {counts[1]} images")
    print(f"  Total    : {sum(counts.values())} images")
    return np.array(X), np.array(y)


def train():
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib

    print("=" * 60)
    print("ParkiSense – Feature-Based Drawing Model Trainer")
    print("  Method: HOG + LBP features → SVM / Random Forest")
    print("=" * 60)

    print("\n[1/4] Loading and extracting features...")
    X, y = load_dataset()
    print(f"  Feature vector size: {X.shape[1]}")

    print("\n[2/4] Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Try multiple classifiers via cross-validation ──────────────────────────
    print("\n[3/4] Evaluating classifiers (5-fold cross-validation)...")
    candidates = {
        'SVM (RBF)'    : SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=SEED),
        'SVM (Linear)' : SVC(kernel='linear', C=1.0, probability=True, random_state=SEED),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=SEED),
        'Gradient Boost': GradientBoostingClassifier(n_estimators=100, random_state=SEED),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_name, best_score, best_clf = None, 0.0, None

    for name, clf in candidates.items():
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_acc = scores.mean() * 100
        std_acc  = scores.std() * 100
        print(f"  {name:20s}: {mean_acc:.1f}% ± {std_acc:.1f}%")
        if mean_acc > best_score:
            best_score, best_name, best_clf = mean_acc, name, clf

    print(f"\n  Best: {best_name} ({best_score:.1f}%)")

    # ── Train best model on ALL data ──────────────────────────────────────────
    print("\n[4/4] Training best model on full dataset...")
    best_clf.fit(X_scaled, y)

    # Classification report on training data (for reference)
    y_pred = best_clf.predict(X_scaled)
    print("\nTraining Classification Report:")
    print(classification_report(y, y_pred, target_names=['Healthy', 'Parkinson']))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    # ── Save model, scaler, labels ────────────────────────────────────────────
    model_path  = MODEL_OUT.resolve()
    scaler_path = SCALER_OUT.resolve()
    labels_path = LABELS_OUT.resolve()

    joblib.dump(best_clf, model_path)
    joblib.dump(scaler,   scaler_path)

    with open(labels_path, 'w') as f:
        f.write("0 Healthy\n1 Parkinson\n")

    print(f"\n  Model  saved: {model_path}")
    print(f"  Scaler saved: {scaler_path}")
    print(f"  Labels saved: {labels_path}")
    print(f"\n{'='*60}")
    print(f"DONE — Cross-validation accuracy: {best_score:.1f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    try:
        import skimage
    except ImportError:
        print("Installing scikit-image...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-image'])

    train()
