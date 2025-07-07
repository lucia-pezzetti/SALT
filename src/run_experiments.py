# plot_simple_env.py

import subprocess
import re
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

# 1) How many layers you want to sweep over
LAYERS_TO_TEST = [6, 7, 8, 9, 10]  # adjust as needed

# 2) How many offsets your simple env supports (0 through N-1)
OFFSETS = [0.0, 50.0, 100.0, 150.0]  # seconds into the cycle

# 3) Regex patterns to pull out the two numbers we need from main’s stdout
_rlrl_matching = re.compile(r"RL matching \+ RL policy avg time: ([0-9]+\.[0-9]+)")
_rlsp_matching   = re.compile(r"RL matching \+ SP policy avg time: ([0-9]+\.[0-9]+)")
_sprl_matching   = re.compile(r"SP matching \+ RL policy avg time: ([0-9]+\.[0-9]+)")
_spsp_policy     = re.compile(r"SP matching \+ SP policy avg time: ([0-9]+\.[0-9]+)")

def run_main_for(layer_count, offset):
    """
    Calls your main script with a particular --num_layers and --offset,
    captures stdout, and returns (matching_time, policy_time) as floats.
    """
    print(f"Running main() for layers={layer_count}, offset={offset}...")
    if layer_count == 3:
        epochs = "10_000"  # shorter for 3 layers
    elif layer_count == 4:
        epochs = "15_000"
    elif layer_count == 5:
        epochs = "30_000"
    elif layer_count == 6:
        epochs = "50_000"
    elif layer_count == 7:
        epochs = "100_000"
    elif layer_count == 8:
        epochs = "150_000"
    elif layer_count == 9:
        epochs = "200_000"
    elif layer_count == 10:
        epochs = "250_000"
    cmd = [
        "python", "main.py",
        "--env_type",    "simple",
        "--num_layers",  str(layer_count),
        # "--cycle_length", "200",
        "--offset",      str(offset),
        "--epochs", epochs,
        # you can pass through any other flags you need, e.g.
        "--num_agents", "5",
        # "--max_steps",  "128",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    out = completed.stdout
    print(out)

    rlrl = _rlrl_matching.search(out)
    rlsp = _rlsp_matching.search(out)
    sprl = _sprl_matching.search(out)
    spsp = _spsp_policy.search(out)
    if not (rlrl and rlsp and sprl and spsp):
        raise RuntimeError(f"Failed to parse output for layers={layer_count}, offset={offset}\n\n{out}")

    return float(rlrl.group(1)), float(rlsp.group(1)), float(sprl.group(1)), float(spsp.group(1))


def evaluate_over_layers(layers, offsets):
    """
    For each layer in layers:
      - run main() on every offset in offsets
      - collect matching & policy times
      - return two lists of averaged metrics
    """
    avg_rlrl = []
    avg_rlsp  = []
    avg_sprl  = []
    avg_spsp  = []

    for L in layers:
        rlrl_vals = []
        rlsp_vals = []
        sprl_vals = []
        spsp_vals = []
        for o in offsets:
            rlrl, rlsp, sprl, spsp = run_main_for(L, o)
            rlrl_vals.append(rlrl)
            rlsp_vals.append(rlsp)
            sprl_vals.append(sprl)
            spsp_vals.append(spsp)
        avg_rlrl.append(mean(rlrl_vals))
        avg_rlsp.append(mean(rlsp_vals))
        avg_sprl.append(mean(sprl_vals))
        avg_spsp.append(mean(spsp_vals))

    return np.array(avg_rlrl), np.array(avg_rlsp), np.array(avg_sprl), np.array(avg_spsp)


def plot_results(layers, rlrl, rlsp, sprl, spsp):
    plt.figure(figsize=(8,5))
    plt.plot(layers, rlrl, marker='o', label='RL matching + RL policy')
    plt.plot(layers, rlsp,   marker='s', label='RL matching + SP policy')
    plt.plot(layers, sprl,   marker='^', label='SP matching + RL policy')
    plt.plot(layers, spsp,   marker='x', label='SP matching + SP policy')
    plt.xlabel("Number of Layers in Simple Grid")
    plt.ylabel("Average Travel Time")
    plt.title("Simple-Env: Matching vs Policy Evaluation\n(averaged over offsets)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("simple_env_matching_vs_policy.pdf")


if __name__ == "__main__":
    rlrl_curve, rlsp_curve, sprl_curve, spsp_curve = evaluate_over_layers(LAYERS_TO_TEST, OFFSETS)
    plot_results(LAYERS_TO_TEST, rlrl_curve, rlsp_curve, sprl_curve, spsp_curve)
