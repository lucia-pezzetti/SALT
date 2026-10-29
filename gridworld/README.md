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

The scalability launcher uses threshold-only early stopping by default: a
method stops after reaching at least `0.99` for `10` consecutive batches. While
that target is enabled, a flat curve below `0.99` cannot trigger the older
plateau rule. Override these settings with `--early_stop_reach_target` and
`--early_stop_reach_window`; pass `--early_stop_reach_target 0` to restore the
plateau fallback.

For a short smoke test, reduce the training batches, target interactions, and evaluation seeds.

## Origin-destination frequency transfer

The dedicated SALT-Q experiment trains with 20 agents on a fixed `30x30` grid,
samples starts and targets uniformly over the full grid, records post-matching
origin-destination visitation counts, and evaluates fixed initial assignments at
fleet sizes `1,5,10,20,50`:

```bash
for seed in 0 1 2 3 4; do
  PYTHONPATH=. python scripts/run_pair_frequency_transfer.py \
    --seed "$seed" \
    --train_episodes 1000000 \
    --alpha_end 0.05 \
    --alpha_decay_episodes 800000 \
    --action_exploration_decay_episodes 500000 \
    --matching_exploration_decay_episodes 800000 \
    --outdir runs/pair_frequency_transfer_1m
done
```

This extended schedule uses `1000000 x 20 x 100 = 2e9` nominal interactions.
Alpha decays from `0.25` to `0.05` over 800k episodes, action exploration over
500k, and random matching over 800k. The original defaults remain 112k episodes
and `2.24e8` nominal interactions for comparison with the main grid-world
budget. Evaluation uses the same `--eval_seed` for every training seed and
samples nested start/target sets across fleet sizes.
Each initial Q-based Hungarian assignment remains fixed during its diagnostic
rollout, so completion and censored travel time are attributable to an exact
training-frequency pair. Training logs report both the most recent episode and
the moving-average reach rate; `training_metrics.json` also records how often
the final greedy policy reduces Manhattan distance.

For a smoke run:

```bash
PYTHONPATH=. python scripts/run_pair_frequency_transfer.py \
  --grid_size 6 --horizon 12 --train_agents 4 \
  --eval_agents 1,2,4,6 --train_episodes 20 --eval_episodes 10 \
  --outdir runs/pair_frequency_smoke
```

Aggregate completed seeds into the paper-facing main and diagnostic appendix
figures (PNG and PDF), plus metric and distance-controlled seed-range CSVs,
with:

```bash
PYTHONPATH=. python scripts/plot_pair_frequency_transfer.py \
  --input_root runs/pair_frequency_transfer_1m
```

The main figure uses the stable training-frequency groups `0-5`, `6-20`, and
`21+`. It shows the realized evaluation mass, the per-agent completion rate by
fleet size, and per-agent completion against the fraction of under-covered OD
assignments in each episode. By default, an under-covered nontrivial pair has at most 20 training
assignments; change this with `--low_visit_threshold`. Episode-level points with
fewer than 20 trials per training seed are omitted. The appendix reports excess
censored travel time and distance-controlled completion at the training fleet
size; cells with fewer than 100 evaluation trials per seed are masked by
default. Shaded bands and error bars span the minimum to maximum across the five
training seeds.

Each seed directory contains the Q-table, compressed training counters,
per-fleet pair aggregates and compact episode-level records, and JSON summaries.
`--mode train` and `--mode eval` can be used in separate cluster jobs;
evaluation then loads the saved Q-table and visitation archive.

To repeat the identical protocol while training with five agents, change only
the training fleet and output directory:

```bash
for seed in 0 1 2 3 4; do
  PYTHONPATH=. python scripts/run_pair_frequency_transfer.py \
    --seed "$seed" \
    --train_agents 5 \
    --train_episodes 1000000 \
    --alpha_end 0.05 \
    --alpha_decay_episodes 800000 \
    --action_exploration_decay_episodes 500000 \
    --matching_exploration_decay_episodes 800000 \
    --outdir runs/pair_frequency_transfer_n5_1m
done
```

Then generate the same result bundle with:

```bash
PYTHONPATH=. python scripts/plot_pair_frequency_transfer.py \
  --input_root runs/pair_frequency_transfer_n5_1m
```

Training now also writes `seed*/training_history.npz`. It contains the raw
per-episode completion, total and per-agent cost/reward, actual and nominal
cumulative transition counts, learning-rate alpha, action epsilon, and matching
epsilon. Plot completion against both episode and cumulative active transitions
when comparing training speed across fleet sizes: the former measures wall-clock
training progress, whereas the latter controls for the fact that an N=20 episode
can generate four times as many agent transitions as an N=5 episode.

## Baselines

`run_comparison.py` always trains Separation (SALT) and MAPPO. Additional
cooperative-MARL baselines are opt-in:

- `--run_ippo` — Independent PPO
- `--run_happo` — **Heterogeneous-Agent PPO** (Kuba et al., 2022)
- `--run_qmix` — QMIX
- `--run_vdn` — Value Decomposition Networks
- `--run_mfq` — **Mean-Field Q-learning** (Yang et al., 2018)

HAPPO (`--run_happo`) is a CTDE actor-critic that, unlike MAPPO, uses
non-shared per-agent actors updated **sequentially** with the heterogeneous-agent
advantage decomposition (each agent's clipped surrogate is scaled by the
cumulative product of the previously-updated agents' policy ratios). It shares a
centralized critic and the same environment, reward decomposition and metrics as
the other baselines, so HAPPO-vs-MAPPO isolates the effect of the sequential
trust-region update.

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

## Clariden GPU runs

All neural methods accept `--device auto`, `--device cpu`, or `--device cuda`.
The environment simulation and tabular Separation Q-learning remain on the CPU;
neural inference and optimizer updates run on the selected PyTorch device.

Sync the code from a laptop without copying generated runs:

```bash
rsync -av --exclude runs/ --exclude __pycache__/ \
  gridworld/ clariden:'$SCRATCH/SALT/gridworld/'
```

Build and import the container in the same interactive allocation:

```bash
srun -A aa004 -p debug -N1 --time=01:30:00 --pty bash
cd "$SCRATCH/SALT/gridworld"
bash scripts/build_clariden_image.sh
exit
```

The resulting image is
`$SCRATCH/images/salt-gridworld-0.1.0.sqsh`, matching `clariden.toml`.

Run a one-GPU smoke test before submitting the full experiment:

```bash
cd "$SCRATCH/SALT/gridworld"
P_VALUES=0 GRIDWORLD_NOISE_MODE=individual GRIDWORLD_TRAIN_SEED=0 \
srun -A aa004 -p debug -N1 --ntasks=1 --gpus-per-task=1 \
  --cpus-per-task=72 --time=00:20:00 \
  --network=disable_rdzv_get \
  --environment=./clariden.toml \
  bash scripts/run_comparison_rank.sh \
  --sep_method ppo --sep_ppo_batches 2 --sep_ppo_batch_eps 4 \
  --mappo_batches 2 --mappo_batch_eps 4 \
  --happo_batches 2 --happo_batch_eps 4 --eval_seeds 2
```

Submit the core Separation/MAPPO/HAPPO comparison for one training seed:

```bash
bash scripts/submit_clariden_comparison.sh \
  --seeds "0" \
  --max-concurrent-nodes 3 \
  -- --sep_method ppo --eval_seeds 50
```

The three array elements cover individual, local, and global noise. Each array
element occupies one node and runs the four default probabilities concurrently,
one process per GH200 GPU. HAPPO is always enabled by the rank launcher.

To run independent training seeds, extend `--seeds`. The concurrency cap limits
how many exclusive nodes can run simultaneously:

```bash
bash scripts/submit_clariden_comparison.sh \
  --seeds "0 1 2 3" \
  --max-concurrent-nodes 6 \
  -- --sep_method ppo --eval_seeds 50
```

To include every implemented optional baseline and every Separation backend:

```bash
bash scripts/submit_clariden_comparison.sh \
  --seeds "0" --max-concurrent-nodes 3 \
  -- --sep_method q,ddqn,ppo,a2c \
  --run_ippo --run_qmix --run_vdn --run_mfq --run_mfq_local \
  --eval_seeds 50
```

That profile is substantially longer. Benchmark one `(p, noise, seed)` shard on
the debug partition before relying on the 12-hour normal-partition limit. The
launcher also supports `--partition low --time 24:00:00` when project `aa004`
has access to the low partition.

Monitor jobs and inspect output files with:

```bash
squeue -u "$USER" -o "%.12i %.18j %.10T %.10M %.10L %.8D %R"
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,Timelimit,ExitCode
find runs/clariden -name comparison.json -print
```

Outputs are isolated by submission, training seed, noise mode, and probability:

```text
runs/clariden/comparison_<jobid>/seed0/individual/p0.0_<timestamp>/comparison.json
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


### Transport cost of the assignment at evaluation time

During evaluation the SALT rollouts solve the optimal-transport (Hungarian)
assignment at `t = 0` and at every `--rematch_every` steps. The transport cost
is controlled by `--sep_eval_assignment`:

- `learned` (default): the learned cost-to-go estimate `\hat j(s, z)` of the
  SALT learner, i.e. `min_a Q(s, z, a)` for tabular SALT-Q, `-max_a Q(s, z, a)`
  for SALT-DDQN and `-V(s, z, t)` from the saved critic for SALT-PPO/SALT-A2C
  (this is the deployment step of Algorithm 1 in the paper);
- `distance`: the nominal Manhattan distance between agent and target.

PPO/A2C checkpoints written before the critic was persisted
(`critic_state_dict` missing) can only be evaluated with
`--sep_eval_assignment distance`; retrain to obtain the critic.

To re-evaluate saved SALT checkpoints instead of training, pass
`--q_path_<noise>`, `--q_ddqn_path_<noise>`, `--sep_policy_path_<noise>` or
`--sep_a2c_policy_path_<noise>` (`<noise>` ∈ `individual`, `local`, `global`;
the files are written to `<outdir>/<noise>/`). A path that is given but does
not exist aborts the run rather than silently training from scratch.

## Results

The scripts save aggregate metrics and per-seed results as JSON files. Generated results, plots, logs, and checkpoints are not tracked by Git.
