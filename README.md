# A Separation Principle for Multi-Agent Reinforcement Learning
This repository contains the code for the paper [A Separation Principle for Multi-Agent Reinforcement Learning]().

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/lucia-pezzetti/ride-sharing-simulator.git
cd ride-sharing-simulator
```

### 2. Run the setup script
To install the required dependencies and automatically download and convert the data:

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
The full TLC dataset is publicly available at [NYC TLC Trip Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

## Discrete Q-Learning Pipeline

The current repository is polished around the Manhattan discrete tabular Q-learning pipeline used for the experiments. The canonical launcher is:

```bash
bash src/run_main.sh
```

The launcher moves into `src/`, activates the Conda environment configured by `CONDA_ENV` (default: `ride-sharing`), prints the active JAX backend, and runs `main.py` with the discrete Q-learning flags used for the experiment.

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

`small_manhattan_area` select:

```text
Upper East Side North
Yorkville West
Upper East Side South
Lenox Hill West
```

You can also pass explicit zone names:

```bash
python src/main.py --discrete --manhattan_area "Upper East Side North" "Yorkville West" "Upper East Side South" "Lenox Hill West"
``
