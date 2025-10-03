# plot_simple_env.py

import subprocess
import re
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import argparse

# 1) How many layers you want to sweep over
LAYERS_TO_TEST = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # adjust as needed

WIDTHS = [3]  # width of each layer

# 2) How many offsets your simple env supports (0 through N-1)
OFFSETS = [0.0, 40.0, 80.0, 120.0, 160.0]  # seconds into the cycle

# 3) Regex patterns to pull out the two numbers we need from main's stdout
_value_policy = re.compile(r"Value Policy\s+avg time: ([0-9]+\.[0-9]+)")
_sp_policy = re.compile(r"SP Policy\s+avg time: ([0-9]+\.[0-9]+)")
_baseline = re.compile(r"Brute-force matching \+ SP policy avg time: ([0-9]+\.[0-9]+)")  # Placeholder for brute-force baseline

def run_main_for(layer_count, width, offset, save_brute_force=True):
    """
    Calls your main script with a particular --num_layers and --offset,
    captures stdout, and returns (value_policy_time, sp_policy_time) as floats.
    If save_brute_force is False, brute force results are not expected/parsed.
    """
    print(f"Running main() for layers={layer_count}, width={width}, offset={offset}...")
    if layer_count <= 40:
        epochs = 2500
    else:
        epochs = 5000
    # if layer_count == 3:
    #     epochs = 5000
    # elif layer_count == 4:
    #     epochs = 5000
    # elif layer_count == 5:
    #     epochs = 10000
    # elif layer_count == 6:
    #     epochs = 10000
    # elif layer_count == 7:
    #     epochs = 15000
    # elif layer_count == 8:
    #     epochs = 15000
    # elif layer_count == 9:
    #     epochs = 20000
    # elif layer_count == 10:
    #     epochs = 20000
    # elif layer_count == 11:
    #     epochs = 20000
    # elif layer_count == 12:
    #     epochs = 20000
    # elif layer_count == 13:
    #     epochs = 20000
    # elif layer_count == 14:
    #     epochs = 20000
    # elif layer_count == 15:
    #     epochs = 20000
    # elif layer_count == 16:
    #     epochs = 20000
    # elif layer_count == 17:
    #     epochs = 20000
    # elif layer_count == 18:
    #     epochs = 20000
    # elif layer_count == 19:
    #     epochs = 20000
    # elif layer_count == 20:
    #     epochs = 20000
    # else:
    #     epochs = 30000

    # if width == 5:
    #     epochs = epochs*1.5
    # elif width == 6:
    #     epochs = epochs*1.5
    # elif width == 7:
    #     epochs = epochs*2
    # elif width == 8:
    #     epochs = epochs*2
    # elif width == 9:
    #     epochs = epochs*2.5
    # elif width == 10:
    #     epochs = epochs*3
    cmd = [
        "python", "optimized_main_v2.py",
        "--env_type",    "simple",
        "--num_layers",  str(layer_count),
        # "--cycle_length", "200",
        "--layer_width", str(width),
        "--no_congestion", "True", 
        "--offset",      str(offset),
        "--epochs", str(int(epochs)),
        # you can pass through any other flags you need, e.g.
        "--num_agents", "5"
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    out = completed.stdout
    print(out)

    value_policy = _value_policy.search(out)
    sp_policy = _sp_policy.search(out)
    
    if save_brute_force:
        bfbf = _baseline.search(out)  # Placeholder for brute-force baseline
        if not (value_policy and sp_policy and bfbf):
            raise RuntimeError(f"Failed to parse output for layers={layer_count}, width={width}, offset={offset}\n\n{out}")
        return float(value_policy.group(1)), float(sp_policy.group(1)), float(bfbf.group(1))  # Return matching and policy times as floats
    else:
        if not (value_policy and sp_policy):
            raise RuntimeError(f"Failed to parse output for layers={layer_count}, width={width}, offset={offset}\n\n{out}")
        return float(value_policy.group(1)), float(sp_policy.group(1)), None  # Return matching and policy times, None for brute force


def evaluate_over_layers(layers, widths, offsets, save_brute_force=True):
    nL = len(layers)
    nW = len(widths)

    # pre‐allocate
    avg_value_policy = np.empty((nL, nW))
    avg_sp_policy = np.empty((nL, nW))
    avg_bfbf = np.empty((nL, nW)) if save_brute_force else None

    for i, L in enumerate(layers):
        for j, w in enumerate(widths):
            # gather metrics for all offsets
            value_policy_vals = []
            sp_policy_vals = []
            bfbf_vals = [] if save_brute_force else None
            for o in offsets:
                value_policy, sp_policy, bfbf = run_main_for(L, w, o, save_brute_force)
                value_policy_vals.append(value_policy)
                sp_policy_vals.append(sp_policy)
                if save_brute_force and bfbf is not None:
                    bfbf_vals.append(bfbf)

            # store the per‐(L,w) means
            avg_value_policy[i, j] = mean(value_policy_vals)
            avg_sp_policy[i, j] = mean(sp_policy_vals)
            if save_brute_force and bfbf_vals:
                avg_bfbf[i, j] = mean(bfbf_vals)

    return avg_value_policy, avg_sp_policy, avg_bfbf


def plot_normalized_boxplots(layers, widths, value_policy_arr, sp_policy_arr, bfbf_arr=None):
    # Create one plot per width, with depth (layers) on x-axis
    for j, width in enumerate(widths):
        plt.figure(figsize=(10, 6))
        
        if bfbf_arr is not None:
            # Normalized data
            normalized_value_policy = value_policy_arr[:, j] / bfbf_arr[:, j]  # shape (L,)
            normalized_sp_policy = sp_policy_arr[:, j] / bfbf_arr[:, j]  # shape (L,)
            
            # Create positions for the boxplots (side by side for each layer)
            positions_value = np.arange(1, len(layers) + 1) - 0.2
            positions_sp = np.arange(1, len(layers) + 1) + 0.2
            
            # Create the boxplots
            bp1 = plt.boxplot([normalized_value_policy[i] for i in range(len(layers))], 
                             positions=positions_value, widths=0.3, patch_artist=True)
            bp2 = plt.boxplot([normalized_sp_policy[i] for i in range(len(layers))], 
                             positions=positions_sp, widths=0.3, patch_artist=True)
            
            # Color the boxplots
            for patch in bp1['boxes']:
                patch.set_facecolor('lightblue')
            for patch in bp2['boxes']:
                patch.set_facecolor('lightcoral')
            
            plt.xlabel("Depth (Layers)")
            plt.ylabel("Normalized Time")
            plt.title(f"Width {width}: Value Policy vs SP Policy (Normalized over Brute Force)")
            plt.xticks(np.arange(1, len(layers) + 1), layers)
            plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Value Policy', 'SP Policy'])
            plt.grid(True, alpha=0.3)
            plt.savefig(f"width_{width}_normalized_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            # Raw data when brute force baseline is not available
            value_policy_data = [value_policy_arr[i, j] for i in range(len(layers))]
            sp_policy_data = [sp_policy_arr[i, j] for i in range(len(layers))]
            
            # Create positions for the boxplots (side by side for each layer)
            positions_value = np.arange(1, len(layers) + 1) - 0.2
            positions_sp = np.arange(1, len(layers) + 1) + 0.2
            
            # Create the boxplots
            bp1 = plt.boxplot(value_policy_data, positions=positions_value, widths=0.3, 
                             patch_artist=True)
            bp2 = plt.boxplot(sp_policy_data, positions=positions_sp, widths=0.3, 
                             patch_artist=True)
            
            # Color the boxplots
            for patch in bp1['boxes']:
                patch.set_facecolor('lightblue')
            for patch in bp2['boxes']:
                patch.set_facecolor('lightcoral')
            
            plt.xlabel("Depth (Layers)")
            plt.ylabel("Time")
            plt.title(f"Width {width}: Value Policy vs SP Policy Times")
            plt.xticks(np.arange(1, len(layers) + 1), layers)
            plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Value Policy', 'SP Policy'])
            plt.grid(True, alpha=0.3)
            plt.savefig(f"width_{width}_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments with optional brute force baseline')
    parser.add_argument('--save_brute_force', action='store_true', default=False,
                        help='Save and use brute force baseline results (default: False)')
    parser.add_argument('--no_brute_force', action='store_true', default=False,
                        help='Skip brute force baseline (overrides --save_brute_force)')
    
    args = parser.parse_args()
    
    # Determine if we should save brute force results
    save_brute_force = args.save_brute_force and not args.no_brute_force
    
    print(f"Running experiments with brute force baseline: {save_brute_force}")
    
    value_policy_curve, sp_policy_curve, bfbf_curve = evaluate_over_layers(LAYERS_TO_TEST, WIDTHS, OFFSETS, save_brute_force)
    plot_normalized_boxplots(LAYERS_TO_TEST, WIDTHS, value_policy_curve, sp_policy_curve, bfbf_curve)
