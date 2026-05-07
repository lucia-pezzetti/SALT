# Grid-world experiments

This repository contains the code for the grid-world experiments of the paper `A Separation Principle for Multi-Agent Reinforcement Learning`.

## Requirements

Install dependencies with:

```setup
pip install -r requirements.txt
pip install torch
```

Optional logging:

```setup
pip install wandb
```

Run commands from the repository root with:

```setup
export PYTHONPATH=.
```

## Training

To train and evaluate Separation-PPO and MAPPO on one gridworld setting:

```train
PYTHONPATH=. python scripts/run_comparison.py \
  --sep_method ppo \
  --grid_h 20 \
  --grid_w 20 \
  --horizon 40 \
  --agents 8 \
  --p 0.1 \
  --noise_mode individual \
  --eval_start_mode bottom_left \
  --eval_target_region full_grid \
  --sep_ppo_batches 500 \
  --sep_ppo_batch_eps 64 \
  --mappo_batches 500 \
  --mappo_batch_eps 64 \
  --eval_seeds 20 \
  --outdir runs/comparison
```

To run the square-grid scalability sweep:

```train
PYTHONPATH=. python scripts/sweep_quadratic_grid_agents.py \
  --grid_w_values 20 \
  --agents_values 2 4 6 8 16 32 64 128\
  --constant_target_density \
  --p 0.1 \
  --eval_seeds 50 \
  --target_interactions 2000000000 \
  --outdir runs/scalability
```

For a short smoke test, reduce the training batches, target interactions, and evaluation seeds.

## Evaluation

Evaluation runs automatically after training. The scripts report:

- terminal optimal-transport cost
- target coverage
- reach rate
- mean time to reach
- mean time to reach including unreached agents
- total cost

Outputs are written under the selected `--outdir`. The main result file is:

```eval
runs/comparison/<run_id>/comparison.json
```

Per-noise-kind metrics are saved in files such as:

```eval
runs/comparison/<run_id>/individual/sep_ppo_metrics.json
runs/comparison/<run_id>/individual/mappo_metrics.json
```

## Pre-trained Models

No pre-trained models are required. Models are trained from scratch by the provided scripts.

## Results

The scripts save aggregate metrics and per-seed results as JSON files. Generated results, plots, logs, and checkpoints are not tracked by Git.

| File | Description |
| --- | --- |
| `comparison.json` | Aggregate results for all evaluated noise kinds. |
| `<noise>/sep_ppo_metrics.json` | Separation-PPO evaluation metrics. |
| `<noise>/mappo_metrics.json` | MAPPO evaluation metrics. |
| `<noise>/eval_trajectories_viz.json` | Trajectory data for visualization. |

## Contributing

This repository is provided for reproducibility. Please keep generated outputs, checkpoints, and logs out of version control.
