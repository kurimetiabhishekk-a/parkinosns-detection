"""
Parkinson's Disease Detection - Train with UCI Dataset
UCI Dataset: 196 samples, 22 voice features, target: 'status' (1=Parkinson, 0=Healthy)
Features match what RecognitionLib.py extracts (jitter, shimmer, HNR, etc.)
Saves model to: ../../trainedModel.sav  (replaces the old model)
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# ============================================================================
# 1. LOAD UCI DATASET
# ============================================================================
print("=" * 60)
print("LOADING UCI PARKINSON'S DATASET")
print("=" * 60)

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "DataSet", "uci_parkinsons.csv")
model_save_path = os.path.join(script_dir, "..", "trainedModel.sav")

df = pd.read_csv(dataset_path)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nClass distribution (status: 1=Parkinson, 0=Healthy):")
print(df['status'].value_counts())
print(f"\nParkinson's rate: {df['status'].mean()*100:.1f}%")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 60)
print("FEATURE SELECTION")
print("=" * 60)

# Drop 'name' (patient identifier, not a feature)
# Keep all 22 voice features; target is 'status'
DROP_COLS = ['name']
LABEL_COL = 'status'

X = df.drop(columns=DROP_COLS + [LABEL_COL])
y = df[LABEL_COL]

print(f"Features used ({X.shape[1]}): {list(X.columns)}")
print(f"Samples: {len(X)}")

# ============================================================================
# 3. PREPROCESSING
# ============================================================================
print("\n" + "=" * 60)
print("PREPROCESSING")
print("=" * 60)

# MinMaxScaler works well for SVM and matches original project style
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples    : {len(X_test)}")

# ============================================================================
# 4. TRAIN MULTIPLE MODELS AND PICK BEST
# ============================================================================
print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

models = {
    "SVM (RBF)": SVC(kernel='rbf', C=10, gamma=0.1, probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc  = model.score(X_test,  y_test)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    results[name] = {
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }
    print(f"\n{name}:")
    print(f"  Train Accuracy : {train_acc*100:.2f}%")
    print(f"  Test  Accuracy : {test_acc*100:.2f}%")
    print(f"  CV Accuracy    : {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# ============================================================================
# 5. PICK BEST MODEL (by CV accuracy then test accuracy)
# ============================================================================
print("\n" + "=" * 60)
print("MODEL SELECTION")
print("=" * 60)

best_name = max(results, key=lambda n: (results[n]["cv_mean"], results[n]["test_acc"]))
best = results[best_name]
best_model = best["model"]

print(f"Best Model      : {best_name}")
print(f"Test  Accuracy  : {best['test_acc']*100:.2f}%")
print(f"CV    Accuracy  : {best['cv_mean']*100:.2f}%")

y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ============================================================================
# 6. SAVE MODEL + SCALER
# ============================================================================
print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

# Bundle model + scaler + feature names so RecognitionLib can use it
model_bundle = {
    "model": best_model,
    "scaler": scaler,
    "features": list(X.columns),
    "model_name": best_name,
    "test_accuracy": round(best['test_acc'] * 100, 2),
    "cv_accuracy": round(best['cv_mean'] * 100, 2),
    "dataset": "UCI Parkinson's Dataset (196 samples, 22 features)",
}

joblib.dump(model_bundle, model_save_path)
print(f"Model bundle saved to: {model_save_path}")
print(f"\nBundle contents: model + scaler + feature list")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print(f"  Model    : {best_name}")
print(f"  Accuracy : {best['test_acc']*100:.2f}% (test), {best['cv_mean']*100:.2f}% (CV)")
print(f"  Saved to : {os.path.abspath(model_save_path)}")
print("=" * 60)
