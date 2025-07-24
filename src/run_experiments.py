# plot_simple_env.py

import subprocess
import re
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

# 1) How many layers you want to sweep over
LAYERS_TO_TEST = [10]  # adjust as needed

WIDTHS = [3, 4, 5, 6, 7, 8, 9, 10]  # width of each layer

# 2) How many offsets your simple env supports (0 through N-1)
OFFSETS = [0.0, 40.0, 80.0, 120.0, 160.0]  # seconds into the cycle

# 3) Regex patterns to pull out the two numbers we need from main’s stdout
_rlrl_matching = re.compile(r"RL matching \+ RL policy avg time: ([0-9]+\.[0-9]+)")
_spsp_policy     = re.compile(r"SP matching \+ SP policy avg time: ([0-9]+\.[0-9]+)")
_baseline = re.compile(r"Brute-force matching \+ SP policy avg time: ([0-9]+\.[0-9]+)")  # Placeholder for brute-force baseline

def run_main_for(layer_count, width, offset):
    """
    Calls your main script with a particular --num_layers and --offset,
    captures stdout, and returns (matching_time, policy_time) as floats.
    """
    print(f"Running main() for layers={layer_count}, width={width}, offset={offset}...")
    if layer_count == 3:
        epochs = 5000
    elif layer_count == 4:
        epochs = 10000
    elif layer_count == 5:
        epochs = 15000
    elif layer_count == 6:
        epochs = 20000
    elif layer_count == 7:
        epochs = 25000
    elif layer_count == 8:
        epochs = 30000
    elif layer_count == 9:
        epochs = 40000
    elif layer_count == 10:
        epochs = 50000

    if width == 5:
        epochs = epochs*1.5
    elif width == 6:
        epochs = epochs*1.5
    elif width == 7:
        epochs = epochs*2
    elif width == 8:
        epochs = epochs*2
    elif width == 9:
        epochs = epochs*2.5
    elif width == 10:
        epochs = epochs*3
    cmd = [
        "python", "main.py",
        "--env_type",    "simple",
        "--num_layers",  str(layer_count),
        # "--cycle_length", "200",
        "--layer_width", str(width),
        "--no_congestion", "True", 
        "--offset",      str(offset),
        "--epochs", str(int(epochs)),
        # you can pass through any other flags you need, e.g.
        "--num_agents", "5",
        "--model", "dqn",
        # "--max_steps",  "128",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    out = completed.stdout
    print(out)

    rlrl = _rlrl_matching.search(out)
    spsp = _spsp_policy.search(out)
    bfbf = _baseline.search(out)  # Placeholder for brute-force baseline
    if not (rlrl and spsp and bfbf):
        raise RuntimeError(f"Failed to parse output for layers={layer_count}, width={width}, offset={offset}\n\n{out}")

    return float(rlrl.group(1)), float(spsp.group(1)), float(bfbf.group(1))  # Return matching and policy times as floats


def evaluate_over_layers(layers, widths, offsets):
    nL = len(layers)
    nW = len(widths)

    # pre‐allocate
    avg_rlrl = np.empty((nL, nW))
    avg_spsp = np.empty((nL, nW))
    avg_bfbf = np.empty((nL, nW))

    for i, L in enumerate(layers):
        for j, w in enumerate(widths):
            # gather metrics for all offsets
            rlrl_vals = []
            spsp_vals = []
            bfbf_vals = []
            for o in offsets:
                rlrl, spsp, bfbf = run_main_for(L, w, o)
                rlrl_vals.append(rlrl)
                spsp_vals.append(spsp)
                bfbf_vals.append(bfbf)

            # store the per‐(L,w) means
            avg_rlrl[i, j] = mean(rlrl_vals)
            avg_spsp[i, j] = mean(spsp_vals)
            avg_bfbf[i, j] = mean(bfbf_vals)

    return avg_rlrl, avg_spsp, avg_bfbf


def plot_normalized_boxplots(layers, widths, rlrl_arr, spsp_arr, bfbf_arr):
    for i, L in enumerate(layers):
        # RL/RL normalized
        normalized_rlrl = rlrl_arr[i] / bfbf_arr[i]  # shape (W, O)
        plt.figure()
        plt.boxplot([normalized_rlrl[j] for j in range(normalized_rlrl.shape[0])])
        plt.xticks(np.arange(1, len(widths) + 1), widths)
        plt.xlabel("Width")
        plt.ylabel("Normalized RL/RL Time")
        plt.title(f"Layer {L}: RL/RL normalized over brute force")
        plt.show()

        # SP/SP normalized
        normalized_spsp = spsp_arr[i] / bfbf_arr[i]  # shape (W, O)
        plt.figure()
        plt.boxplot([normalized_spsp[j] for j in range(normalized_spsp.shape[0])])
        plt.xticks(np.arange(1, len(widths) + 1), widths)
        plt.xlabel("Width")
        plt.ylabel("Normalized SP/SP Time")
        plt.title(f"Layer {L}: SP/SP normalized over brute force")
        plt.show()

if __name__ == "__main__":
    rlrl_curve, spsp_curve, bfbf_curve = evaluate_over_layers(LAYERS_TO_TEST, WIDTHS, OFFSETS)
    plot_normalized_boxplots(LAYERS_TO_TEST, WIDTHS, rlrl_curve, spsp_curve, bfbf_curve)
