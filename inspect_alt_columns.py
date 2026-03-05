
import pandas as pd

try:
    df = pd.read_csv("src/ModelBuilder/archive_extracted/parkinson_disease.csv", nrows=1)
    print("Columns in parkinson_disease.csv:")
    print(df.columns.tolist())
except Exception as e:
    print(f"Error: {e}")
