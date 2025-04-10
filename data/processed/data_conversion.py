import pandas as pd
df = pd.read_parquet('data/raw/yellow_tripdata_2024-04.parquet')
df.to_csv('data/processed/yellow_tripdata_2024-04.csv', index=False)
