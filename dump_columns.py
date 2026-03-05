
import pandas as pd

try:
    df = pd.read_csv("src/ModelBuilder/DataSet/dataset.csv", nrows=1)
    with open("dataset_columns.txt", "w") as f:
        for col in df.columns:
            f.write(f"{col}\n")
    print("Columns written to dataset_columns.txt")
except Exception as e:
    print(f"Error: {e}")
