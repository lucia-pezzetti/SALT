import pandas as pd
df = pd.read_parquet('yellow_tripdata_2024-04.parquet')
df.to_csv('yellow_tripdata_2024-04.csv', index=False)
