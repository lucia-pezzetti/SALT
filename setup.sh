#!/bin/bash

set -e  # Exit immediately on error

echo "Setting up environment..."

# Create environment and install Python dependencies
echo "Installing Python dependencies..."
conda create -y -n nyc-taxi python=3.13.2
eval "$(conda shell.bash hook)"
conda activate nyc-taxi

pip install --upgrade pip
pip install -r requirements.txt

# Create folder structure
echo "Creating folder structure..."
mkdir -p data/raw
mkdir -p data/processed

# Download Yellow Taxi data (Parquet)
YEAR=2024
MONTH=04
FILE_NAME="yellow_tripdata_${YEAR}-${MONTH}.parquet"
CSV_NAME="yellow_tripdata_${YEAR}-${MONTH}.csv"

echo "Downloading Yellow Taxi trip data for ${YEAR}-${MONTH}..."
wget -O data/raw/$FILE_NAME https://d37ci6vzurychx.cloudfront.net/trip-data/$FILE_NAME

# Convert to CSV
echo "Converting Parquet to CSV..."
$(which python) - <<EOF
import pandas as pd
df = pd.read_parquet("data/raw/$FILE_NAME")
df.to_csv("data/processed/$CSV_NAME", index=False)
EOF

# Download taxi zones shapefile
echo "Downloading NYC Taxi Zones shapefile..."
wget -O taxi_zones.zip https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip
unzip -o taxi_zones.zip -d data/processed/
rm taxi_zones.zip

# Download taxi zone lookup CSV
echo "Downloading taxi_zone_lookup.csv..."
wget -O data/processed/taxi_zone_lookup.csv "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"

echo "Setup complete!"
