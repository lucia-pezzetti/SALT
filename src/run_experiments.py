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
_sp_policy = re.compile(r"SP avg time: ([0-9]+\.[0-9]+)")
_ppo_policy = re.compile(r"PPO avg time: ([0-9]+\.[0-9]+)")
_baseline = re.compile(r"Brute-force matching \+ SP policy avg time: ([0-9]+\.[0-9]+)")  # Placeholder for brute-force baseline

def run_main_for(layer_count, width, offset, num_agents=1, save_brute_force=True, parse_ppo=False, quiet=False):
    """
    Calls your main script with a particular --num_layers and --offset,
    captures stdout, and returns (value_policy_time, sp_policy_time) as floats.
    If save_brute_force is False, brute force results are not expected/parsed.
    """
    if not quiet:
        print(f"Running main() for layers={layer_count}, width={width}, offset={offset}...")
    if layer_count <= 40:
        epochs = 1000
    else:
        epochs = 2500
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
        "python3", "main.py",
        "--env_type",    "simple",
        "--num_layers",  str(layer_count),
        "--config", "ppo_config.json",
        "--cycle_length", "200",
        "--layer_width", str(width),
        "--no_congestion", "True", 
        "--offset",      str(offset),
        "--epochs", str(int(epochs)),
        # you can pass through any other flags you need, e.g.
        "--num_agents", str(num_agents)
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    out = completed.stdout
    if not quiet:
        print(out)

    value_policy = _value_policy.search(out)
    sp_policy = _sp_policy.search(out)
    ppo_policy = _ppo_policy.search(out) if parse_ppo else None
    
    if parse_ppo:
        # PPO mode: parse PPO and SP only
        if not (ppo_policy and sp_policy):
            raise RuntimeError(f"Failed to parse PPO/SP output for layers={layer_count}, width={width}, offset={offset}\n\n{out}")
        return float(ppo_policy.group(1)), float(sp_policy.group(1)), None
    else:
        if save_brute_force:
            bfbf = _baseline.search(out)  # Placeholder for brute-force baseline
            if not (value_policy and sp_policy and bfbf):
                raise RuntimeError(f"Failed to parse output for layers={layer_count}, width={width}, offset={offset}\n\n{out}")
            return float(value_policy.group(1)), float(sp_policy.group(1)), float(bfbf.group(1))
        else:
            if not (value_policy and sp_policy):
                raise RuntimeError(f"Failed to parse output for layers={layer_count}, width={width}, offset={offset}\n\n{out}")
                
            return float(value_policy.group(1)), float(sp_policy.group(1)), None


def evaluate_over_layers(layers, widths, offsets, num_agents=1, save_brute_force=True, parse_ppo=False, quiet=False):
    nL = len(layers)
    nW = len(widths)

    # pre‐allocate
    avg_value_policy = np.empty((nL, nW)) if not parse_ppo else None
    avg_sp_policy = np.empty((nL, nW))
    avg_ppo_policy = np.empty((nL, nW)) if parse_ppo else None
    avg_bfbf = np.empty((nL, nW)) if (save_brute_force and not parse_ppo) else None

    for i, L in enumerate(layers):
        for j, w in enumerate(widths):
            # gather metrics for all offsets
            value_policy_vals = [] if not parse_ppo else None
            sp_policy_vals = []
            ppo_policy_vals = [] if parse_ppo else None
            bfbf_vals = [] if (save_brute_force and not parse_ppo) else None
            for o in offsets:
                vp, sp, bf = run_main_for(L, w, o, num_agents=num_agents, save_brute_force=save_brute_force, parse_ppo=parse_ppo, quiet=quiet)
                if parse_ppo:
                    ppo_policy_vals.append(vp)
                else:
                    value_policy_vals.append(vp)
                sp_policy_vals.append(sp)
                if bfbf_vals is not None and bf is not None:
                    bfbf_vals.append(bf)

            # store the per‐(L,w) means
            if parse_ppo:
                avg_ppo_policy[i, j] = mean(ppo_policy_vals)
            else:
                avg_value_policy[i, j] = mean(value_policy_vals)
            avg_sp_policy[i, j] = mean(sp_policy_vals)
            if bfbf_vals:
                avg_bfbf[i, j] = mean(bfbf_vals)

    return (avg_ppo_policy if parse_ppo else avg_value_policy), avg_sp_policy, avg_bfbf


def plot_normalized_boxplots(layers, widths, value_policy_arr, sp_policy_arr, bfbf_arr=None, parse_ppo=False):
    # Create one plot per width, with depth (layers) on x-axis
    for j, width in enumerate(widths):
        plt.figure(figsize=(10, 6))
        
        if bfbf_arr is not None and not parse_ppo:
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
            if parse_ppo:
                ppo_policy_data = [value_policy_arr[i, j] for i in range(len(layers))]
            else:
                value_policy_data = [value_policy_arr[i, j] for i in range(len(layers))]
            sp_policy_data = [sp_policy_arr[i, j] for i in range(len(layers))]
            
            # Create positions for the boxplots (side by side for each layer)
            positions_value = np.arange(1, len(layers) + 1) - 0.2
            positions_sp = np.arange(1, len(layers) + 1) + 0.2
            
            # Create the boxplots
            if parse_ppo:
                bp1 = plt.boxplot(ppo_policy_data, positions=positions_value, widths=0.3, 
                                 patch_artist=True)
            else:
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
            if parse_ppo:
                plt.title(f"Width {width}: PPO vs SP Policy Times")
            else:
                plt.title(f"Width {width}: Value Policy vs SP Policy Times")
            plt.xticks(np.arange(1, len(layers) + 1), layers)
            if parse_ppo:
                plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['PPO', 'SP Policy'])
            else:
                plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Value Policy', 'SP Policy'])
            plt.grid(True, alpha=0.3)
            if parse_ppo:
                plt.savefig(f"width_{width}_ppo_vs_sp.png", dpi=300, bbox_inches='tight')
            else:
                plt.savefig(f"width_{width}_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments with optional PPO parsing and quiet mode')
    parser.add_argument('--save_brute_force', action='store_true', default=False,
                        help='Save and use brute force baseline results (default: False). Ignored in PPO mode.')
    parser.add_argument('--no_brute_force', action='store_true', default=False,
                        help='Skip brute force baseline (overrides --save_brute_force).')
    parser.add_argument('--ppo', action='store_true', default=False,
                        help='Parse PPO evaluation (expects lines like "PPO avg time:").')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Suppress stdout prints from subprocess and progress logs.')
    parser.add_argument('--num_agents', type=int, default=1,
                        help='Number of agents to run inside each main.py invocation (default: 1).')
    
    args = parser.parse_args()
    
    # Determine if we should save brute force results
    save_brute_force = (args.save_brute_force and not args.no_brute_force) and (not args.ppo)
    
    if not args.quiet:
        print(f"Running experiments | PPO mode: {args.ppo} | brute force baseline: {save_brute_force}")
    
    value_or_ppo_curve, sp_policy_curve, bfbf_curve = evaluate_over_layers(
        LAYERS_TO_TEST, WIDTHS, OFFSETS, num_agents=args.num_agents, save_brute_force=save_brute_force, parse_ppo=args.ppo, quiet=args.quiet
    )
    plot_normalized_boxplots(LAYERS_TO_TEST, WIDTHS, value_or_ppo_curve, sp_policy_curve, bfbf_curve, parse_ppo=args.ppo)
