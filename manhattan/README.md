# Manhattan experiments
This repository contains the code for the south manhattan experiments of the paper **A Separation Principle for Multi-Agent Reinforcement Learning**.

## Quickstart

### 1. Clone the repository

```bash
git clone <repository-url>
cd ride-sharing-simulator
```

### 2. Create the environment
To install the required dependencies for an NVIDIA GPU machine:

```bash
conda create -n ride-sharing python=3.10
conda activate ride-sharing
pip install -r requirements.txt
```

After the setup is complete, activate the environment manually:

```bash
conda activate ride-sharing
```

## Dataset
The default Manhattan Q-learning run uses:

- NYC TLC taxi zone shapefile, expected at `data/processed/taxi_zones.shp`
- OpenStreetMap Manhattan road network, downloaded automatically with OSMnx

The full TLC data source is [NYC TLC Trip Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

## Code Pipeline

The current repository implements the Manhattan discrete tabular Q-learning pipeline used for the experiments in the paper. The canonical launcher is:

```bash
bash src/run_main.sh
```

The launcher activates the Conda environment configured by `CONDA_ENV` (default: `ride-sharing`), and runs `main.py` with the discrete Q-learning flags used for the experiment.

By default, the launcher expects an NVIDIA CUDA backend through JAX:

```bash
JAX_PLATFORMS=cuda
```

If CUDA is not visible to JAX, the script exits before training starts. For a CPU-only fallback, override the platform explicitly:

Useful overrides:

```bash
WANDB_MODE=online bash src/run_main.sh
JAX_PLATFORMS=cpu bash src/run_main.sh
CONDA_ENV=ride-sharing bash src/run_main.sh
```

## Manhattan Area Selection

The Manhattan experiment area is controlled by `--manhattan_area`.

Available presets:

```bash
--manhattan_area south_manhattan
--manhattan_area small_manhattan_area
```

`small_manhattan_area` selects:

```text
Upper East Side North
Yorkville West
Upper East Side South
Lenox Hill West
```

You can also pass explicit zone names:

```bash
python src/main.py --discrete --manhattan_area "Upper East Side North" "Yorkville West" "Upper East Side South" "Lenox Hill West"
```


### NYC TLC / NYC Open Data

The Manhattan experiments use the NYC Taxi & Limousine Commission taxi zone shapefile, expected at:

```text
data/processed/taxi_zones.shp
```

Source pages:

- [NYC TLC Trip Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- [NYC Taxi Zones on NYC Open Data](https://data.cityofnewyork.us/Transportation/NYC-Taxi-Zones/8meu-9t5y)
- [NYC Open Data FAQ](https://opendata.cityofnewyork.us/faq/)

NYC Open Data datasets are made available for public use, but they are provided as-is and without warranties as to accuracy, completeness, or fitness for a particular use. Users should consult the current NYC Open Data and TLC source pages for the authoritative terms, metadata, and documentation.

### OpenStreetMap road network

The Manhattan road network is downloaded automatically with OSMnx from OpenStreetMap.

OpenStreetMap data is © OpenStreetMap contributors and is available under the Open Data Commons Open Database License, ODbL.

Source pages:

- [OpenStreetMap Copyright and License](https://www.openstreetmap.org/copyright)
- [OpenStreetMap Foundation Attribution Guidelines](https://osmfoundation.org/wiki/Licence/Attribution_Guidelines)

Any public outputs that use OpenStreetMap data should provide appropriate OpenStreetMap attribution. If this project distributes modified or derived OpenStreetMap databases, those outputs may also need to be distributed under the ODbL.
