# plot_simple_env.py

import subprocess
import re
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

# 1) How many layers you want to sweep over
LAYERS_TO_TEST = [9]  # adjust as needed

# 2) How many offsets your simple env supports (0 through N-1)
OFFSETS = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0]  # seconds into the cycle

# 3) Regex patterns to pull out the two numbers we need from main’s stdout
_rlrl_matching = re.compile(r"RL matching \+ RL policy avg time: ([0-9]+\.[0-9]+)")
_spsp_policy     = re.compile(r"SP matching \+ SP policy avg time: ([0-9]+\.[0-9]+)")
_baseline = re.compile(r"Brute-force matching \+ SP policy avg time: ([0-9]+\.[0-9]+)")  # Placeholder for brute-force baseline

def run_main_for(layer_count, offset):
    """
    Calls your main script with a particular --num_layers and --offset,
    captures stdout, and returns (matching_time, policy_time) as floats.
    """
    print(f"Running main() for layers={layer_count}, offset={offset}...")
    if layer_count == 3:
        epochs = "5_000"  # shorter for 3 layers
    elif layer_count == 4:
        epochs = "10_000"
    elif layer_count == 5:
        epochs = "15_000"
    elif layer_count == 6:
        epochs = "30_000"
    elif layer_count == 7:
        epochs = "40_000"
    elif layer_count == 8:
        epochs = "50_000"
    elif layer_count == 9:
        epochs = "60_000"
    elif layer_count == 10:
        epochs = "70_000"
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
    spsp = _spsp_policy.search(out)
    bfbf = _baseline.search(out)  # Placeholder for brute-force baseline
    if not (rlrl and spsp and bfbf):
        raise RuntimeError(f"Failed to parse output for layers={layer_count}, offset={offset}\n\n{out}")

    return float(rlrl.group(1)), float(spsp.group(1)), float(bfbf.group(1))  # Return matching and policy times as floats


def evaluate_over_layers(layers, offsets):
    """
    For each layer in layers:
      - run main() on every offset in offsets
      - collect matching & policy times
      - return two lists of averaged metrics
    """
    avg_rlrl = []
    avg_spsp  = []
    avg_bfbf = []  # Placeholder for brute-force baseline, if needed

    for L in layers:
        rlrl_vals = []
        spsp_vals = []
        bfbf_vals = []
        for o in offsets:
            rlrl, spsp, bfbf = run_main_for(L, o)
            rlrl_vals.append(rlrl)
            spsp_vals.append(spsp)
            bfbf_vals.append(bfbf)
        avg_rlrl.append(mean(rlrl_vals))
        avg_spsp.append(mean(spsp_vals))
        avg_bfbf.append(mean(bfbf_vals))

    return np.array(avg_rlrl), np.array(avg_spsp), np.array(avg_bfbf)


def plot_results(layers, rlrl, spsp, bfbf):
    plt.figure(figsize=(8,5))
    plt.plot(layers, rlrl, marker='o', label='RL matching + RL policy')
    plt.plot(layers, spsp, marker='x', label='SP matching + SP policy')
    plt.plot(layers, bfbf, marker='s', label='baseline', linestyle='--')
    plt.xlabel("Number of Layers in Simple Grid")
    plt.ylabel("Average Travel Time")
    plt.title("Simple-Env: Matching vs Policy Evaluation\n(averaged over offsets)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("simple_env_matching_vs_policy.pdf")


if __name__ == "__main__":
    rlrl_curve, spsp_curve, bfbf_curve = evaluate_over_layers(LAYERS_TO_TEST, OFFSETS)
    plot_results(LAYERS_TO_TEST, rlrl_curve, spsp_curve, bfbf_curve)
