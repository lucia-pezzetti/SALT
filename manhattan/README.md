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

## Dispatch baselines (isolating the OT layer)

SALT dispatches taxis in two stages: an optimal-transport (Hungarian) assignment
of taxis to pickups, followed by a learned single-taxi routing policy. To measure
how much of the benefit comes from the assignment layer itself, pass
`--eval_assignment_baselines`. At final evaluation this reports two extra
baselines that reuse the **same learned routing policy** but replace SALT's OT
assignment:

1. **Random assignment** — taxis are matched to pickups by a random permutation.
2. **Myopic nominal-shortest-path assignment** — Hungarian matching on the static
   precomputed shortest-path distance matrix (ignoring congestion and
   time-dependence), i.e. the cheap "nominal shortest-path" assignment resolved
   at the start of the episode.

Comparing these against SALT (OT on congestion/time-aware Q-value estimates +
the same routing) isolates the value of the OT assignment layer. Example:

```bash
python src/main.py --discrete --eval_assignment_baselines \
  --manhattan_area south_manhattan --num_agents 10
```

The per-agent and mean continuous travel times for both baselines are printed
alongside the SALT and shortest-path results (and logged to W&B when enabled).

## Static, deterministic dynamic, and SALT routing

The Manhattan environment has two sources of non-stationarity: deterministic
time variation from traffic-signal phases, and stochastic per-edge travel-time
noise. This distinction matters when interpreting shortest-path baselines.

1. **Static shortest path** uses one offline distance matrix computed from the
   nominal congested edge travel times. It assigns taxis to pickups once at
   `t=0` and then follows the nominal shortest path. This is cheap and
   reproducible, but it ignores both signal phase and stochastic delays.
2. **Deterministic dynamic shortest path** removes the stochastic component
   while keeping the known time-varying signal dynamics. For a fixed finite
   horizon and the same time discretization used by Q-learning, the exact
   single-taxi problem can be solved in polynomial time by augmenting the graph
   with time, i.e. states `(node, time_index)`, and running dynamic programming
   or a shortest-path algorithm on the time-expanded graph. This gives the
   optimal policy for the deterministic model, not for the original stochastic
   environment.
3. **SALT** learns time-aware expected returns under the stochastic environment
   and uses those returns inside the optimal-transport assignment. It therefore
   targets the deployed decision problem rather than the deterministic
   relaxation.

Including the deterministic dynamic shortest-path solution can be useful as a
diagnostic or oracle baseline: it isolates the value of time awareness by
showing how much performance is available from exploiting the known signal
schedule alone. It should not replace the main stochastic comparison, however.
If the baseline is evaluated with the stochastic noise switched off, it solves a
different problem; if it is allowed to optimize against realized noise, it
becomes clairvoyant. Even with noise removed, the time-expanded solve can be much
larger than the static baseline, and it does not address the stochastic
robustness that SALT is designed to learn. For that reason, it is best reported
as a deterministic-oracle ablation, placed between the static shortest-path
baseline and SALT in the comparison.

The existing `--eval_reassignment_baselines` option is a practical
receding-horizon shortest-path comparison: it periodically re-solves the
agent-pickup assignment from the agents' current nodes, but the routing policy
still uses offline nominal shortest-path distances. It is therefore stronger
than the static assign-once baseline, but it is not the full deterministic
dynamic shortest-path oracle described above.

When a Q-learning trajectory reaches the training horizon without completing,
the last update uses the nominal shortest-path travel time remaining from the
post-step node to the assigned pickup as a continuation value. In reward units,
the terminal target is
`last_step_reward + gamma * (-remaining_travel_seconds / 60)`. This avoids a
fixed timeout penalty whose size is unrelated to how far the agent remains from
its target.

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
