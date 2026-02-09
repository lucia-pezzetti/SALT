#!/usr/bin/env python3
"""
Plot the difference between Q-learning and shortest path average times
as a function of shortest path average times.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import glob

def parse_log_file(log_file_path):
    """Parse a log file and extract Q-learning and shortest path average times."""
    ql_times_cont = []
    ql_times_disc = []
    sp_times_cont = []
    sp_times_disc = []
    
    # Check if file is actually a text file (not binary like PNG, etc.)
    log_file_path = Path(log_file_path)
    if log_file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.pdf', '.svg']:
        print(f"Warning: Skipping non-text file: {log_file_path}")
        return {
            'ql_continuous': [],
            'ql_discrete': [],
            'sp_continuous': [],
            'sp_discrete': [],
        }
    
    # Try to detect binary files by reading first few bytes
    try:
        with open(log_file_path, 'rb') as f:
            first_bytes = f.read(8)
            # PNG files start with \x89PNG\r\n\x1a\n
            # JPEG files start with \xff\xd8\xff
            # GIF files start with GIF89a or GIF87a
            if first_bytes.startswith(b'\x89PNG') or first_bytes.startswith(b'\xff\xd8') or first_bytes.startswith(b'GIF'):
                print(f"Warning: Skipping binary file (image): {log_file_path}")
                return {
                    'ql_continuous': [],
                    'ql_discrete': [],
                    'sp_continuous': [],
                    'sp_discrete': [],
                }
    except Exception as e:
        print(f"Warning: Could not check file type for {log_file_path}: {e}")
    
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Pattern for Q-learning avg time (continuous)
    pattern_ql_cont = r'Q-learning avg time \(continuous\):\s*([\d.]+)'
    matches = re.findall(pattern_ql_cont, content)
    ql_times_cont = [float(m) for m in matches]
    
    # Pattern for Q-learning avg time (discrete)
    pattern_ql_disc = r'Q-learning avg time \(discrete\):\s*([\d.]+)'
    matches = re.findall(pattern_ql_disc, content)
    ql_times_disc = [float(m) for m in matches]
    
    # Pattern for SP avg time (continuous)
    pattern_sp_cont = r'SP \(continuous\) avg time:\s*([\d.]+)'
    matches = re.findall(pattern_sp_cont, content)
    sp_times_cont = [float(m) for m in matches]
    
    # Pattern for SP avg time (discrete)
    pattern_sp_disc = r'SP \(discrete\) avg time:\s*([\d.]+)'
    matches = re.findall(pattern_sp_disc, content)
    sp_times_disc = [float(m) for m in matches]
    
    return {
        'ql_continuous': ql_times_cont,
        'ql_discrete': ql_times_disc,
        'sp_continuous': sp_times_cont,
        'sp_discrete': sp_times_disc,
    }

def plot_differences(log_files, output_path=None, use_discrete=True, use_continuous=True):
    """Plot differences between Q-learning and shortest path times."""
    all_ql_cont = []
    all_ql_disc = []
    all_sp_cont = []
    all_sp_disc = []
    
    # Parse all log files
    for log_file in log_files:
        data = parse_log_file(log_file)
        all_ql_cont.extend(data['ql_continuous'])
        all_ql_disc.extend(data['ql_discrete'])
        all_sp_cont.extend(data['sp_continuous'])
        all_sp_disc.extend(data['sp_discrete'])
    
    # Create figure with subplots
    num_plots = sum([use_continuous, use_discrete])
    if num_plots == 0:
        print("Error: At least one of --use_continuous or --use_discrete must be True")
        return
    
    fig, axes = plt.subplots(1, num_plots, figsize=(12, 5) if num_plots > 1 else (6, 5))
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot continuous comparison
    if use_continuous and len(all_ql_cont) > 0 and len(all_sp_cont) > 0:
        # Match lengths (take minimum)
        min_len = min(len(all_ql_cont), len(all_sp_cont))
        ql_cont = np.array(all_ql_cont[:min_len])
        sp_cont = np.array(all_sp_cont[:min_len])
        
        differences_cont = ql_cont - sp_cont
        
        ax = axes[plot_idx]
        ax.scatter(sp_cont, differences_cont, alpha=0.6, s=50)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1, label='Equal performance')
        ax.set_xlabel('Shortest Path Average Time (s)', fontsize=12)
        ax.set_ylabel('Q-learning - Shortest Path Time (s)', fontsize=12)
        ax.set_title('Continuous Evaluation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add statistics text
        mean_diff = np.mean(differences_cont)
        std_diff = np.std(differences_cont)
        textstr = f'Mean diff: {mean_diff:.2f} s\nStd diff: {std_diff:.2f} s\nN: {len(differences_cont)}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plot_idx += 1
    
    # Plot discrete comparison
    if use_discrete and len(all_ql_disc) > 0 and len(all_sp_disc) > 0:
        # Match lengths (take minimum)
        min_len = min(len(all_ql_disc), len(all_sp_disc))
        ql_disc = np.array(all_ql_disc[:min_len])
        sp_disc = np.array(all_sp_disc[:min_len])
        
        differences_disc = ql_disc - sp_disc
        
        ax = axes[plot_idx]
        ax.scatter(sp_disc, differences_disc, alpha=0.6, s=50, color='green')
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1, label='Equal performance')
        ax.set_xlabel('Shortest Path Average Time (s)', fontsize=12)
        ax.set_ylabel('Q-learning - Shortest Path Time (s)', fontsize=12)
        ax.set_title('Discrete Evaluation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add statistics text
        mean_diff = np.mean(differences_disc)
        std_diff = np.std(differences_disc)
        textstr = f'Mean diff: {mean_diff:.2f} s\nStd diff: {std_diff:.2f} s\nN: {len(differences_disc)}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plot_idx += 1
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot Q-learning vs shortest path time differences')
    parser.add_argument('--log_files', type=str, nargs='+', required=True,
                        help='Path(s) to log file(s) to parse')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory containing log files (alternative to --log_files)')
    parser.add_argument('--pattern', type=str, default='*.txt',
                        help='Pattern to match log files in directory (default: *.txt)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for plot (default: show interactively)')
    parser.add_argument('--use_continuous', action='store_true', default=True,
                        help='Plot continuous evaluation results (default: True)')
    parser.add_argument('--use_discrete', action='store_true', default=True,
                        help='Plot discrete evaluation results (default: True)')
    
    args = parser.parse_args()
    
    # Collect log files
    log_files = []
    if args.log_files:
        log_files.extend(args.log_files)
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_files.extend(glob.glob(str(log_dir / args.pattern)))
    
    # Filter out non-text files (images, etc.)
    text_extensions = {'.txt', '.log', '.out', '.dat'}
    log_files = [f for f in log_files if Path(f).suffix.lower() in text_extensions or Path(f).suffix == '']
    
    if not log_files:
        print("Error: No log files found. Please provide --log_files or --log_dir")
        print("Note: Only text files (.txt, .log, .out, .dat) are processed. Image files are automatically skipped.")
        return
    
    print(f"Parsing {len(log_files)} log file(s)...")
    for log_file in log_files:
        print(f"  - {log_file}")
    
    plot_differences(log_files, args.output, args.use_discrete, args.use_continuous)

if __name__ == '__main__':
    main()
