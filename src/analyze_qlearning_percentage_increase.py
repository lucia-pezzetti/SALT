#!/usr/bin/env python3
"""
Calculate percentage increase of Q-learning average times over shortest path average times.
"""

import re
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
    
    with open(log_file_path, 'r') as f:
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

def calculate_percentage_increase(ql_times, sp_times):
    """Calculate percentage increase: (QL - SP) / SP * 100"""
    if len(ql_times) == 0 or len(sp_times) == 0:
        return None
    
    # Match lengths (take minimum)
    min_len = min(len(ql_times), len(sp_times))
    ql = np.array(ql_times[:min_len])
    sp = np.array(sp_times[:min_len])
    
    # Avoid division by zero
    valid_mask = sp > 0
    if not np.any(valid_mask):
        return None
    
    ql_valid = ql[valid_mask]
    sp_valid = sp[valid_mask]
    
    # Percentage increase: (QL - SP) / SP * 100
    pct_increase = ((ql_valid - sp_valid) / sp_valid) * 100
    
    return {
        'mean': float(np.mean(pct_increase)),
        'std': float(np.std(pct_increase)),
        'median': float(np.median(pct_increase)),
        'min': float(np.min(pct_increase)),
        'max': float(np.max(pct_increase)),
        'count': len(pct_increase),
        'raw_values': pct_increase.tolist(),
    }

def main():
    parser = argparse.ArgumentParser(description='Calculate Q-learning percentage increase over shortest path')
    parser.add_argument('--log_files', type=str, nargs='+', required=True,
                        help='Path(s) to log file(s) to parse')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory containing log files (alternative to --log_files)')
    parser.add_argument('--pattern', type=str, default='*.txt',
                        help='Pattern to match log files in directory (default: *.txt)')
    parser.add_argument('--use_continuous', action='store_true', default=True,
                        help='Use continuous evaluation results (default: True)')
    parser.add_argument('--use_discrete', action='store_true', default=False,
                        help='Use discrete evaluation results (default: False)')
    
    args = parser.parse_args()
    
    # Collect log files
    log_files = []
    if args.log_files:
        log_files.extend(args.log_files)
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_files.extend(glob.glob(str(log_dir / args.pattern)))
    
    if not log_files:
        print("Error: No log files found. Please provide --log_files or --log_dir")
        return
    
    print(f"Parsing {len(log_files)} log file(s)...")
    for log_file in log_files:
        print(f"  - {log_file}")
    
    # Parse all log files
    all_ql_cont = []
    all_ql_disc = []
    all_sp_cont = []
    all_sp_disc = []
    
    for log_file in log_files:
        data = parse_log_file(log_file)
        all_ql_cont.extend(data['ql_continuous'])
        all_ql_disc.extend(data['ql_discrete'])
        all_sp_cont.extend(data['sp_continuous'])
        all_sp_disc.extend(data['sp_discrete'])
    
    print("\n" + "="*80)
    print("PERCENTAGE INCREASE ANALYSIS")
    print("="*80)
    
    # Continuous evaluation
    if args.use_continuous and len(all_ql_cont) > 0 and len(all_sp_cont) > 0:
        print("\n📊 Continuous Evaluation:")
        print("-" * 80)
        stats = calculate_percentage_increase(all_ql_cont, all_sp_cont)
        if stats:
            print(f"  Mean percentage increase: {stats['mean']:.2f}%")
            print(f"  Median percentage increase: {stats['median']:.2f}%")
            print(f"  Std deviation: {stats['std']:.2f}%")
            print(f"  Min: {stats['min']:.2f}%")
            print(f"  Max: {stats['max']:.2f}%")
            print(f"  Sample size: {stats['count']}")
            
            # Count how many are better/worse
            better = np.sum(np.array(stats['raw_values']) < 0)
            worse = np.sum(np.array(stats['raw_values']) > 0)
            equal = np.sum(np.array(stats['raw_values']) == 0)
            print(f"  Q-learning better (negative %): {better} ({better/stats['count']*100:.1f}%)")
            print(f"  Q-learning worse (positive %): {worse} ({worse/stats['count']*100:.1f}%)")
            print(f"  Equal: {equal} ({equal/stats['count']*100:.1f}%)")
    
    # Discrete evaluation
    if args.use_discrete and len(all_ql_disc) > 0 and len(all_sp_disc) > 0:
        print("\n📊 Discrete Evaluation:")
        print("-" * 80)
        stats = calculate_percentage_increase(all_ql_disc, all_sp_disc)
        if stats:
            print(f"  Mean percentage increase: {stats['mean']:.2f}%")
            print(f"  Median percentage increase: {stats['median']:.2f}%")
            print(f"  Std deviation: {stats['std']:.2f}%")
            print(f"  Min: {stats['min']:.2f}%")
            print(f"  Max: {stats['max']:.2f}%")
            print(f"  Sample size: {stats['count']}")
            
            # Count how many are better/worse
            better = np.sum(np.array(stats['raw_values']) < 0)
            worse = np.sum(np.array(stats['raw_values']) > 0)
            equal = np.sum(np.array(stats['raw_values']) == 0)
            print(f"  Q-learning better (negative %): {better} ({better/stats['count']*100:.1f}%)")
            print(f"  Q-learning worse (positive %): {worse} ({worse/stats['count']*100:.1f}%)")
            print(f"  Equal: {equal} ({equal/stats['count']*100:.1f}%)")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
