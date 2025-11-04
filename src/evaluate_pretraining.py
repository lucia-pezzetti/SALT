#!/usr/bin/env python3
"""
Script to evaluate pretraining effectiveness by comparing pretrained vs non-pretrained runs.

This script analyzes W&B logs or local log files to determine:
1. Initial performance (after pretraining vs. random initialization)
2. Learning speed (how fast performance improves)
3. Final performance (ultimate performance after training)
4. Comparison to shortest path baseline
5. Training efficiency (episodes needed to reach target performance)
"""

import argparse
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_wandb_log(log_file: str) -> Dict:
    """Parse a W&B log file to extract metrics."""
    metrics = {}
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Parse key metrics directly from content
    # Format: "wandb:             metric_name value" or "metric_name value" in summary section
    metric_patterns = {
        'final_eval_reward': r'(?:wandb:\s+)?final/final_eval_reward\s+([-\d.]+)',
        'final_avg_return': r'(?:wandb:\s+)?final/final_avg_return\s+([-\d.]+)',
        'final_eval_steps': r'(?:wandb:\s+)?final/final_eval_steps\s+([-\d.]+)',
        'final_completion_rate': r'(?:wandb:\s+)?final/final_completion_rate\s+([-\d.]+)',
        'final_performance_ratio': r'(?:wandb:\s+)?final/final_performance_ratio\s+([-\d.]+)',
        'final_improvement_over_sp': r'(?:wandb:\s+)?final/improvement_over_sp\s+([-\d.]+)',
        'final_total_episodes': r'(?:wandb:\s+)?final/total_episodes\s+(\d+)',
        'final_total_epochs': r'(?:wandb:\s+)?final/total_epochs\s+(\d+)',
        'evaluation_avg_reward': r'(?:wandb:\s+)?evaluation/avg_reward\s+([-\d.]+)',
        'evaluation_avg_steps': r'(?:wandb:\s+)?evaluation/avg_steps\s+([-\d.]+)',
        'evaluation_completion_rate': r'(?:wandb:\s+)?evaluation/completion_rate\s+([-\d.]+)',
        'evaluation_improvement_over_sp': r'(?:wandb:\s+)?evaluation/improvement_over_sp\s+([-\d.]+)',
        'sp_baseline_reward': r'(?:wandb:\s+)?evaluation/sp_baseline_reward\s+([-\d.]+)',
        'sp_baseline_steps': r'(?:wandb:\s+)?evaluation/sp_baseline_steps\s+([-\d.]+)',
        'ppo_avg_time': r'(?:wandb:\s+)?final_eval/ppo_avg_time\s+([-\d.]+)',
        'sp_avg_time': r'(?:wandb:\s+)?final_eval/sp_avg_time\s+([-\d.]+)',
        'ppo_vs_sp_ratio': r'(?:wandb:\s+)?final_eval/ppo_vs_sp_ratio\s+([-\d.]+)',
        'ppo_improvement': r'(?:wandb:\s+)?final_eval/ppo_improvement\s+([-\d.]+)',
    }
    
    for key, pattern in metric_patterns.items():
        match = re.search(pattern, content)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except ValueError:
                metrics[key] = None
    
    return metrics


def extract_training_curve_from_log(log_file: str) -> Dict[str, List]:
    """Extract training curve data from log file if available."""
    # This is a placeholder - actual implementation would parse epoch-by-epoch metrics
    # For now, we'll work with summary metrics
    return {}


def compare_runs(pretrained_log: str, non_pretrained_log: str) -> Dict:
    """Compare pretrained vs non-pretrained runs."""
    pretrained_metrics = parse_wandb_log(pretrained_log)
    non_pretrained_metrics = parse_wandb_log(non_pretrained_log)
    
    comparison = {
        'pretrained': pretrained_metrics,
        'non_pretrained': non_pretrained_metrics,
        'improvements': {}
    }
    
    # Calculate improvements for key metrics
    key_metrics = [
        'final_eval_reward',
        'final_avg_return',
        'final_eval_steps',
        'final_completion_rate',
        'final_improvement_over_sp',
        'ppo_vs_sp_ratio',
    ]
    
    for metric in key_metrics:
        if metric in pretrained_metrics and metric in non_pretrained_metrics:
            pretrained_val = pretrained_metrics[metric]
            non_pretrained_val = non_pretrained_metrics[metric]
            
            if pretrained_val is not None and non_pretrained_val is not None:
                if 'steps' in metric or 'ratio' in metric:
                    # Lower is better
                    improvement = ((non_pretrained_val - pretrained_val) / non_pretrained_val) * 100
                elif 'reward' in metric or 'return' in metric:
                    # Higher is better (more negative = worse since rewards are negative)
                    improvement = ((pretrained_val - non_pretrained_val) / abs(non_pretrained_val)) * 100 if non_pretrained_val != 0 else 0
                elif 'improvement_over_sp' in metric:
                    # Higher is better
                    improvement = ((pretrained_val - non_pretrained_val) / abs(non_pretrained_val)) * 100 if non_pretrained_val != 0 else 0
                else:
                    improvement = 0
                
                comparison['improvements'][metric] = {
                    'pretrained': pretrained_val,
                    'non_pretrained': non_pretrained_val,
                    'improvement_pct': improvement,
                    'better': improvement > 0 if 'steps' not in metric and 'ratio' not in metric else improvement < 0
                }
    
    return comparison


def generate_report(comparison: Dict, output_file: Optional[str] = None):
    """Generate a human-readable report on pretraining effectiveness."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PRETRAINING EFFECTIVENESS ANALYSIS")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Key metrics comparison
    report_lines.append("KEY METRICS COMPARISON:")
    report_lines.append("-" * 80)
    
    metrics_labels = {
        'final_eval_reward': 'Final Evaluation Reward',
        'final_avg_return': 'Final Average Return',
        'final_eval_steps': 'Final Evaluation Steps (lower is better)',
        'final_completion_rate': 'Final Completion Rate',
        'final_improvement_over_sp': 'Improvement over Shortest Path',
        'ppo_vs_sp_ratio': 'PPO/SP Time Ratio (lower is better)',
    }
    
    for metric, label in metrics_labels.items():
        if metric in comparison['improvements']:
            imp = comparison['improvements'][metric]
            report_lines.append(f"\n{label}:")
            report_lines.append(f"  Pretrained:     {imp['pretrained']:.4f}")
            report_lines.append(f"  Non-pretrained: {imp['non_pretrained']:.4f}")
            report_lines.append(f"  Improvement:    {imp['improvement_pct']:+.2f}%")
            report_lines.append(f"  Better:         {'✓ YES' if imp['better'] else '✗ NO'}")
    
    # Overall assessment
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("OVERALL ASSESSMENT")
    report_lines.append("=" * 80)
    
    # Count wins
    wins = sum(1 for imp in comparison['improvements'].values() if imp['better'])
    total = len(comparison['improvements'])
    
    if wins > total * 0.6:
        verdict = "✓ PRETRAINING WAS EFFECTIVE"
        explanation = f"Pretraining improved {wins}/{total} key metrics."
    elif wins < total * 0.4:
        verdict = "✗ PRETRAINING WAS NOT EFFECTIVE"
        explanation = f"Pretraining improved only {wins}/{total} key metrics."
    else:
        verdict = "? PRETRAINING EFFECTIVENESS UNCLEAR"
        explanation = f"Pretraining improved {wins}/{total} key metrics. Manual inspection recommended."
    
    report_lines.append(verdict)
    report_lines.append(explanation)
    report_lines.append("")
    
    # Specific recommendations
    report_lines.append("RECOMMENDATIONS:")
    report_lines.append("-" * 80)
    
    if 'final_eval_steps' in comparison['improvements']:
        steps_imp = comparison['improvements']['final_eval_steps']
        if steps_imp['improvement_pct'] > 10:
            report_lines.append("• Pretraining significantly reduced steps needed - GOOD")
        elif steps_imp['improvement_pct'] < -10:
            report_lines.append("• Pretraining increased steps needed - BAD, consider adjusting pretraining")
        else:
            report_lines.append("• Pretraining had minimal effect on steps - consider more pretraining steps")
    
    if 'final_improvement_over_sp' in comparison['improvements']:
        sp_imp = comparison['improvements']['final_improvement_over_sp']
        if sp_imp['improvement_pct'] > 5:
            report_lines.append("• Pretraining improved performance vs shortest path - GOOD")
        elif sp_imp['improvement_pct'] < -5:
            report_lines.append("• Pretraining worsened performance vs shortest path - BAD")
    
    if 'ppo_vs_sp_ratio' in comparison['improvements']:
        ratio_imp = comparison['improvements']['ppo_vs_sp_ratio']
        if ratio_imp['pretrained'] < 2.0:
            report_lines.append("• Pretrained model performs close to shortest path - EXCELLENT")
        elif ratio_imp['pretrained'] < 3.0:
            report_lines.append("• Pretrained model is reasonable compared to shortest path")
        else:
            report_lines.append("• Pretrained model is far from shortest path - consider more training")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Report saved to {output_file}")
    
    print(report_text)
    
    return report_text


def create_comparison_plot(comparison: Dict, output_file: Optional[str] = None):
    """Create a visualization comparing pretrained vs non-pretrained performance."""
    metrics_to_plot = [
        ('final_eval_reward', 'Final Eval Reward\n(higher is better)'),
        ('final_eval_steps', 'Final Eval Steps\n(lower is better)'),
        ('final_improvement_over_sp', 'Improvement over SP\n(higher is better)'),
        ('ppo_vs_sp_ratio', 'PPO/SP Ratio\n(lower is better)'),
    ]
    
    available_metrics = [m for m, _ in metrics_to_plot if m in comparison['improvements']]
    
    if not available_metrics:
        print("No metrics available for plotting")
        return
    
    fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 5))
    if len(available_metrics) == 1:
        axes = [axes]
    
    pretrained_vals = []
    non_pretrained_vals = []
    labels = []
    
    for i, (metric, label) in enumerate([m for m in metrics_to_plot if m[0] in available_metrics]):
        imp = comparison['improvements'][metric]
        pretrained_vals.append(imp['pretrained'])
        non_pretrained_vals.append(imp['non_pretrained'])
        labels.append(label.split('\n')[0])
        
        axes[i].bar(['Pretrained', 'Non-pretrained'], 
                   [imp['pretrained'], imp['non_pretrained']],
                   color=['#2ecc71', '#e74c3c'], alpha=0.7)
        axes[i].set_title(label)
        axes[i].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate pretraining effectiveness by comparing logs'
    )
    parser.add_argument(
        '--pretrained_log',
        type=str,
        required=True,
        help='Path to log file from pretrained run'
    )
    parser.add_argument(
        '--non_pretrained_log',
        type=str,
        required=True,
        help='Path to log file from non-pretrained run'
    )
    parser.add_argument(
        '--output_report',
        type=str,
        default=None,
        help='Output file for text report (default: print to stdout)'
    )
    parser.add_argument(
        '--output_plot',
        type=str,
        default=None,
        help='Output file for comparison plot'
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    if not Path(args.pretrained_log).exists():
        print(f"Error: Pretrained log file not found: {args.pretrained_log}")
        return
    
    if not Path(args.non_pretrained_log).exists():
        print(f"Error: Non-pretrained log file not found: {args.non_pretrained_log}")
        return
    
    # Compare runs
    print("Parsing log files...")
    comparison = compare_runs(args.pretrained_log, args.non_pretrained_log)
    
    # Generate report
    print("\nGenerating analysis report...")
    generate_report(comparison, args.output_report)
    
    # Create plot if requested
    if args.output_plot:
        print("\nGenerating comparison plot...")
        create_comparison_plot(comparison, args.output_plot)


if __name__ == '__main__':
    main()

