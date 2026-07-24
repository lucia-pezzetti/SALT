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
  --outdir runs/comparison
```

To run the square-grid scalability sweep:

```train
PYTHONPATH=. python scripts/sweep_quadratic_grid_agents.py \
  --agents_values 2 4 6 8 16 32 64 128\
  --constant_target_density \
  --p 0.1 \
  --eval_seeds 50 \
  --target_interactions 2000000000 \
  --outdir runs/scalability
```

For a short smoke test, reduce the training batches, target interactions, and evaluation seeds.

## Baselines

`run_comparison.py` always trains Separation (SALT) and MAPPO. Additional
cooperative-MARL baselines are opt-in:

- `--run_ippo` — Independent PPO
- `--run_qmix` — QMIX
- `--run_vdn` — Value Decomposition Networks
- `--run_mfq` — **Mean-Field Q-learning** (Yang et al., 2018)

The mean-field baseline (`--run_mfq`) conditions a shared per-agent Q-network on
the population **mean action** and is the natural point of comparison for the
large-population, homogeneous-fleet regime that motivates SALT. It uses a global
mean field (the neighbourhood is all currently-active agents), a Boltzmann
policy with a mean-action fixed point, and a mean-field soft-value Bellman
target; the team objective is decomposed into per-agent rewards (the terminal
optimal-transport cost is split via the same Hungarian assignment), so it is
trained on the same environment and scored with the same metrics as the other
baselines. Example:

```train
PYTHONPATH=. python scripts/run_comparison.py \
  --sep_method ppo \
  --run_mfq --mfq_batches 5000 --mfq_batch_eps 64 \
  --outdir runs/comparison
```

To include it in the square-grid scalability sweep, pass `--run_mfq` to the
launcher (its training budget is scaled to `--target_interactions` like MAPPO):

```train
PYTHONPATH=. python scripts/sweep_quadratic_grid_agents.py \
  --agents_values 2 4 6 8 16 32 64 128 \
  --constant_target_density --p 0.1 --run_mfq \
  --outdir runs/scalability
```

## Evaluation

Evaluation runs automatically after training. The scripts report:

- terminal optimal-transport cost
- reach rate
- mean time to reach
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

## Results

The scripts save aggregate metrics and per-seed results as JSON files. Generated results, plots, logs, and checkpoints are not tracked by Git.
