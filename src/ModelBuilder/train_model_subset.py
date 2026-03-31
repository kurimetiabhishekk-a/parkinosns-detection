
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

selected_features = [
    'locPctJitter', 'locAbsJitter', 'rapJitter', 'ppq5Jitter', 
    'locShimmer', 'locDbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer',
    'meanHarmToNoiseHarmonicity', 'meanNoiseToHarmHarmonicity'
]

print("Loading dataset...")
try:
    df = pd.read_csv("DataSet/dataset.csv")
    print(f"Dataset shape: {df.shape}")

    for feat in selected_features:
        if feat not in df.columns:
            print(f"WARNING: Feature {feat} not found in dataset!")
    
    X = df[selected_features]
    y = df['class']

    X = X.fillna(X.mean())
    
    print(f"Training on {len(selected_features)} features: {selected_features}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)
    print(f"Test Accuracy: {score:.4f}")

    joblib.dump(clf, "trainedModel_subset.sav")
    print("Model saved to trainedModel_subset.sav")
    
except Exception as e:
    print(f"Error: {e}")
