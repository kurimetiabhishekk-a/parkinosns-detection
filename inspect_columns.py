
import pandas as pd

try:
    df = pd.read_csv("src/ModelBuilder/DataSet/dataset.csv", nrows=1)
    print("Columns in dataset.csv:")
    for col in df.columns:
        if "Jitter" in col or "Shimmer" in col or "HNR" in col or "Harmonicity" in col:
            print(col)
except Exception as e:
    print(f"Error: {e}")
