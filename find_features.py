
import pandas as pd

try:
    df = pd.read_csv("src/ModelBuilder/DataSet/dataset.csv", nrows=1)
    with open("found_features.txt", "w") as f:
        f.write("Columns matching keywords:\n")
        for col in df.columns:
            if any(key in col for key in ["Jitter", "Shimmer", "HNR", "Harmonicity", "NHR", "RAP", "PPQ", "APQ"]):
                f.write(f"{col}\n")
    print("Features written to found_features.txt")
except Exception as e:
    print(f"Error: {e}")
