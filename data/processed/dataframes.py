# Re-import necessary libraries after reset
import pandas as pd

# Define the file path
file_path = "data/processed/yellow_tripdata_2024-04.csv"

# Read only the first 10 rows
df_sample = pd.read_csv(file_path, nrows=10)

df_sample.to_csv("data/processed/tenrows.csv", index=False)