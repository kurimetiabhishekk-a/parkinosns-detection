
import joblib
import pandas as pd
import numpy as np

PATH = "src/trainedModel.sav"
try:
    clf = joblib.load(PATH)
    print("Model loaded successfully.")
    print("Type:", type(clf))
    if isinstance(clf, dict):
        print("Keys:", clf.keys())
        if "features" in clf:
            print("Features expected by bundle:", clf["features"])
    elif hasattr(clf, "feature_names_in_"):
        print("Features expected (feature_names_in_):", clf.feature_names_in_)
    elif hasattr(clf, "n_features_in_"):
        print("Number of features:", clf.n_features_in_)
except Exception as e:
    print("Error loading model:", e)
