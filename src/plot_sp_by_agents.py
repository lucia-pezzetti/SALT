#!/usr/bin/env python3
"""
Plot shortest path evaluation results by number of agents.
"""

import re
import matplotlib.pyplot as plt
import numpy as np

def parse_results_file(filename):
    """Parse the results file and extract metrics for each agent count."""
    with open(filename, 'r') as f:
        content = f.read()
    
    # Pattern to match evaluation sections
    pattern = r'=== Shortest Path Evaluation with (\d+) agents ===.*?'
    pattern += r'=== Shortest Path Baseline \(Continuous - real time\) ===.*?'
    pattern += r'  Avg reward: ([\d.]+) ± ([\d.]+).*?'
    pattern += r'  Avg steps: ([\d.]+) ± ([\d.]+).*?'
    pattern += r'  Completion rate: ([\d.]+)%.*?'
    pattern += r'  Avg time: ([\d.]+) ± ([\d.]+).*?'
    pattern += r'=== Shortest Path Baseline \(Discrete - dt=[\d.]+.*?\) ===.*?'
    pattern += r'  Avg reward: ([\d.]+) ± ([\d.]+).*?'
    pattern += r'  Avg steps: ([\d.]+) ± ([\d.]+).*?'
    pattern += r'  Completion rate: ([\d.]+)%.*?'
    pattern += r'  Avg time: ([\d.]+) ± ([\d.]+)'
    
    matches = re.finditer(pattern, content, re.DOTALL)
    
    results = []
    for match in matches:
        num_agents = int(match.group(1))
        cont_reward_mean = float(match.group(2))
        cont_reward_std = float(match.group(3))
        cont_steps_mean = float(match.group(4))
        cont_steps_std = float(match.group(5))
        cont_completion = float(match.group(6))
        cont_time_mean = float(match.group(7))
        cont_time_std = float(match.group(8))
        
        disc_reward_mean = float(match.group(9))
        disc_reward_std = float(match.group(10))
        disc_steps_mean = float(match.group(11))
        disc_steps_std = float(match.group(12))
        disc_completion = float(match.group(13))
        disc_time_mean = float(match.group(14))
        disc_time_std = float(match.group(15))
        
        results.append({
            'num_agents': num_agents,
            'continuous': {
                'reward_mean': cont_reward_mean,
                'reward_std': cont_reward_std,
                'steps_mean': cont_steps_mean,
                'steps_std': cont_steps_std,
                'completion': cont_completion,
                'time_mean': cont_time_mean,
                'time_std': cont_time_std,
            },
            'discrete': {
                'reward_mean': disc_reward_mean,
                'reward_std': disc_reward_std,
                'steps_mean': disc_steps_mean,
                'steps_std': disc_steps_std,
                'completion': disc_completion,
                'time_mean': disc_time_mean,
                'time_std': disc_time_std,
            }
        })
    
    return sorted(results, key=lambda x: x['num_agents'])

def plot_results(results, output_file='sp_by_agents_plot.png'):
    """Create plots showing metrics vs number of agents."""
    num_agents = [r['num_agents'] for r in results]
    
    cont_time_mean = [r['continuous']['time_mean'] for r in results]
    cont_time_std = [r['continuous']['time_std'] for r in results]
    disc_time_mean = [r['discrete']['time_mean'] for r in results]
    disc_time_std = [r['discrete']['time_std'] for r in results]
    
    cont_steps_mean = [r['continuous']['steps_mean'] for r in results]
    cont_steps_std = [r['continuous']['steps_std'] for r in results]
    disc_steps_mean = [r['discrete']['steps_mean'] for r in results]
    disc_steps_std = [r['discrete']['steps_std'] for r in results]
    
    cont_reward_mean = [r['continuous']['reward_mean'] for r in results]
    cont_reward_std = [r['continuous']['reward_std'] for r in results]
    disc_reward_mean = [r['discrete']['reward_mean'] for r in results]
    disc_reward_std = [r['discrete']['reward_std'] for r in results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Shortest Path Performance by Number of Agents', fontsize=16, fontweight='bold')
    
    # Plot 1: Average Time
    ax1 = axes[0, 0]
    ax1.errorbar(num_agents, cont_time_mean, yerr=cont_time_std, 
                 marker='o', linestyle='-', label='Continuous', linewidth=2, markersize=8, capsize=5)
    ax1.errorbar(num_agents, disc_time_mean, yerr=disc_time_std, 
                 marker='s', linestyle='--', label='Discrete', linewidth=2, markersize=8, capsize=5)
    ax1.set_xlabel('Number of Agents', fontsize=12)
    ax1.set_ylabel('Average Time (seconds)', fontsize=12)
    ax1.set_title('Average Travel Time', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot 2: Average Steps
    ax2 = axes[0, 1]
    ax2.errorbar(num_agents, cont_steps_mean, yerr=cont_steps_std, 
                 marker='o', linestyle='-', label='Continuous', linewidth=2, markersize=8, capsize=5)
    ax2.errorbar(num_agents, disc_steps_mean, yerr=disc_steps_std, 
                 marker='s', linestyle='--', label='Discrete', linewidth=2, markersize=8, capsize=5)
    ax2.set_xlabel('Number of Agents', fontsize=12)
    ax2.set_ylabel('Average Steps', fontsize=12)
    ax2.set_title('Average Number of Steps', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # Plot 3: Average Reward
    ax3 = axes[1, 0]
    ax3.errorbar(num_agents, cont_reward_mean, yerr=cont_reward_std, 
                 marker='o', linestyle='-', label='Continuous', linewidth=2, markersize=8, capsize=5)
    ax3.errorbar(num_agents, disc_reward_mean, yerr=disc_reward_std, 
                 marker='s', linestyle='--', label='Discrete', linewidth=2, markersize=8, capsize=5)
    ax3.set_xlabel('Number of Agents', fontsize=12)
    ax3.set_ylabel('Average Reward', fontsize=12)
    ax3.set_title('Average Reward', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary table
    table_data = []
    headers = ['Agents', 'Time', 'Steps', 'Reward']
    for r in results:
        table_data.append([
            f"{r['num_agents']}",
            f"{r['continuous']['time_mean']:.1f}±{r['continuous']['time_std']:.1f}",
            f"{r['continuous']['steps_mean']:.1f}±{r['continuous']['steps_std']:.1f}",
            f"{r['continuous']['reward_mean']:.2f}±{r['continuous']['reward_std']:.2f}"
        ])
    
    table = ax4.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.2, 0.3, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Summary Statistics (Continuous)', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    return fig

if __name__ == '__main__':
    import sys
    
    input_file = 'eval_sp_by_agents_results.txt'
    output_file = 'sp_by_agents_plot.png'
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print(f"Parsing results from {input_file}...")
    results = parse_results_file(input_file)
    
    if not results:
        print("No results found! Check the file format.")
        sys.exit(1)
    
    print(f"Found results for {len(results)} agent counts:")
    for r in results:
        print(f"  {r['num_agents']} agents: time={r['continuous']['time_mean']:.1f}±{r['continuous']['time_std']:.1f}s")
    
    print(f"\nCreating plots...")
    plot_results(results, output_file)
    print("Done!")
