

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Loading dataset...")
parkinson = pd.read_csv("DataSet/dataset.csv")
print(f"Dataset shape: {parkinson.shape}")
print(f"\nFirst few rows:")
print(parkinson.head())

print(f"\n{'='*60}")
print("DATASET INFORMATION")
print(f"{'='*60}")
print(f"Total samples: {len(parkinson)}")
print(f"Total features: {len(parkinson.columns) - 1}")
print(f"\nClass distribution:")
print(parkinson['class'].value_counts())
print(f"\nMissing values: {parkinson.isnull().sum().sum()}")

print(f"\n{'='*60}")
print("DATA PREPROCESSING")
print(f"{'='*60}")

X = parkinson.drop(['class', 'id', 'gender'], axis=1)
y = parkinson['class']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

if X.isnull().sum().sum() > 0:
    print("\nHandling missing values...")
    X = X.fillna(X.mean())
else:
    print("\nNo missing values found!")

print(f"\n{'='*60}")
print("TRAIN-TEST SPLIT")
print(f"{'='*60}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

print(f"\n{'='*60}")
print("FEATURE SCALING")
print(f"{'='*60}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler")

print(f"\n{'='*60}")
print("LOGISTIC REGRESSION MODEL")
print(f"{'='*60}")

clf_lr = LogisticRegression(max_iter=1000, random_state=1)
clf_lr.fit(X_train_scaled, y_train)

train_score_lr = clf_lr.score(X_train_scaled, y_train)
test_score_lr = clf_lr.score(X_test_scaled, y_test)

print(f"Training accuracy: {train_score_lr:.4f}")
print(f"Test accuracy: {test_score_lr:.4f}")

scores_lr = cross_val_score(clf_lr, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {scores_lr}")
print(f"Mean CV accuracy: {scores_lr.mean():.4f} (+/- {scores_lr.std():.4f})")

print(f"\n{'='*60}")
print("RANDOM FOREST MODEL")
print(f"{'='*60}")

clf_rf = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1)
clf_rf.fit(X_train, y_train)  # Random Forest doesn't need scaling

train_score_rf = clf_rf.score(X_train, y_train)
test_score_rf = clf_rf.score(X_test, y_test)

print(f"Training accuracy: {train_score_rf:.4f}")
print(f"Test accuracy: {test_score_rf:.4f}")

scores_rf = cross_val_score(clf_rf, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {scores_rf}")
print(f"Mean CV accuracy: {scores_rf.mean():.4f} (+/- {scores_rf.std():.4f})")

print(f"\n{'='*60}")
print("TOP 20 MOST IMPORTANT FEATURES")
print(f"{'='*60}")

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': clf_rf.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20))

print(f"\n{'='*60}")
print("SAVING MODEL")
print(f"{'='*60}")

if test_score_rf >= test_score_lr:
    best_model = clf_rf
    best_model_name = "Random Forest"
    model_filename = "trainedModel_RF.sav"
else:
    best_model = clf_lr
    best_model_name = "Logistic Regression"
    model_filename = "trainedModel_LR.sav"

joblib.dump(best_model, model_filename)
print(f"Best model ({best_model_name}) saved as: {model_filename}")

if best_model_name == "Logistic Regression":
    joblib.dump(scaler, "scaler.sav")
    print("Scaler saved as: scaler.sav")

print(f"\n{'='*60}")
print("TRAINING SUMMARY")
print(f"{'='*60}")
print(f"Dataset: 756 samples, 754 features")
print(f"\nLogistic Regression:")
print(f"  - Test Accuracy: {test_score_lr:.4f}")
print(f"  - CV Accuracy: {scores_lr.mean():.4f}")
print(f"\nRandom Forest:")
print(f"  - Test Accuracy: {test_score_rf:.4f}")
print(f"  - CV Accuracy: {scores_rf.mean():.4f}")
print(f"\nBest Model: {best_model_name}")
print(f"Saved as: {model_filename}")
print(f"\n{'='*60}")
