#!/usr/bin/env python3
"""
Parse Q-learning debug logs and produce boxplots comparing average times.
"""

import argparse
import ast
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _extract_numeric_values(line: str, pattern: str) -> List[float]:
    """Extract one scalar or a list of numeric values from a metric line."""
    match = re.search(pattern, line)
    if not match:
        return []

    raw = match.group(1).strip()
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
        if isinstance(parsed, list):
            return [float(v) for v in parsed]
        return []

    try:
        return [float(raw)]
    except ValueError:
        return []


def extract_times(filepath: Path) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Extract Q-learning continuous, Q-learning discrete, SP continuous, and SP discrete avg times."""
    q_cont_times: List[float] = []
    q_disc_times: List[float] = []
    sp_cont_times: List[float] = []
    sp_disc_times: List[float] = []

    with filepath.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            # New format can contain scalars or lists: "...: 123.4" or "...: [123.4, ...]"
            q_cont_vals = _extract_numeric_values(
                line, r"Q-learning avg time \(continuous\):\s*(\[[^\]]*\]|[-+eE.\d]+)"
            )
            if q_cont_vals:
                q_cont_times.extend(q_cont_vals)
                continue

            q_disc_vals = _extract_numeric_values(
                line, r"Q-learning avg time \(discrete\):\s*(\[[^\]]*\]|[-+eE.\d]+)"
            )
            if q_disc_vals:
                q_disc_times.extend(q_disc_vals)
                continue

            # Fallback to old format: "Q-learning avg time:" (assume continuous for backward compatibility)
            q_vals = _extract_numeric_values(
                line, r"Q-learning avg time:\s*(\[[^\]]*\]|[-+eE.\d]+)"
            )
            if q_vals:
                q_cont_times.extend(q_vals)
                continue

            sp_cont_vals = _extract_numeric_values(
                line, r"SP \(continuous\) avg time:\s*(\[[^\]]*\]|[-+eE.\d]+)"
            )
            if sp_cont_vals:
                sp_cont_times.extend(sp_cont_vals)
                continue

            sp_disc_vals = _extract_numeric_values(
                line, r"SP \(discrete\) avg time:\s*(\[[^\]]*\]|[-+eE.\d]+)"
            )
            if sp_disc_vals:
                sp_disc_times.extend(sp_disc_vals)
                continue

            # Fallback for logs that only print mean-over-agents lines.
            q_cont_mean_vals = _extract_numeric_values(
                line, r"Q-learning mean over agents \(continuous\):\s*(\[[^\]]*\]|[-+eE.\d]+)"
            )
            if q_cont_mean_vals:
                q_cont_times.extend(q_cont_mean_vals)
                continue

            q_disc_mean_vals = _extract_numeric_values(
                line, r"Q-learning mean over agents \(discrete\):\s*(\[[^\]]*\]|[-+eE.\d]+)"
            )
            if q_disc_mean_vals:
                q_disc_times.extend(q_disc_mean_vals)
                continue

            sp_cont_mean_vals = _extract_numeric_values(
                line, r"SP mean over agents \(continuous\):\s*(\[[^\]]*\]|[-+eE.\d]+)"
            )
            if sp_cont_mean_vals:
                sp_cont_times.extend(sp_cont_mean_vals)
                continue

            sp_disc_mean_vals = _extract_numeric_values(
                line, r"SP mean over agents \(discrete\):\s*(\[[^\]]*\]|[-+eE.\d]+)"
            )
            if sp_disc_mean_vals:
                sp_disc_times.extend(sp_disc_mean_vals)
                continue

    if not q_cont_times and not q_disc_times:
        raise ValueError(f"No Q-learning avg times found in {filepath}")
    if not sp_disc_times and not sp_cont_times:
        raise ValueError(f"No SP avg times found in {filepath}")

    return q_cont_times, q_disc_times, sp_cont_times, sp_disc_times


def create_boxplot(
    q_cont_times: List[float],
    q_disc_times: List[float],
    sp_cont_times: List[float],
    sp_disc_times: List[float],
    output_path: Path,
) -> None:
    """Create and save a boxplot comparing the extracted times."""
    data = []
    labels = []
    colors = []

    # Filter out zero values from each series
    q_cont_filtered = [t for t in q_cont_times if t > 0]
    q_disc_filtered = [t for t in q_disc_times if t > 0]
    sp_cont_filtered = [t for t in sp_cont_times if t > 0]
    sp_disc_filtered = [t for t in sp_disc_times if t > 0]

    # Add Q-learning continuous if available
    if q_cont_filtered:
        data.append(q_cont_filtered)
        labels.append("Q-learning (continuous)")
        colors.append("#2ecc71")  # Green

    # Add Q-learning discrete if available
    if q_disc_filtered:
        data.append(q_disc_filtered)
        labels.append("Q-learning (discrete)")
        colors.append("#27ae60")  # Darker green

    # Add SP continuous if available
    if sp_cont_filtered:
        data.append(sp_cont_filtered)
        labels.append("SP (continuous)")
        colors.append("#3498db")  # Blue

    # Add SP discrete if available
    if sp_disc_filtered:
        data.append(sp_disc_filtered)
        labels.append("SP (discrete)")
        colors.append("#e74c3c")  # Red

    if not data:
        raise ValueError("No data to plot")

    fig, ax = plt.subplots(figsize=(12, 6))
    box = ax.boxplot(data, patch_artist=True, tick_labels=labels, showmeans=True, meanline=True)

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.setp(box["whiskers"], color="black")
    plt.setp(box["caps"], color="black")
    plt.setp(box["medians"], visible=False)  # Hide median line
    plt.setp(box["means"], color="black", linewidth=2)  # Show mean as line (same style as median)
    plt.setp(box["fliers"], markeredgecolor="black")

    ax.set_ylabel("Average time (s)")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")

    stats_lines = []
    for label, series in zip(labels, data):
        stats_lines.append(
            f"{label}: mean={np.mean(series):.2f}, median={np.median(series):.2f}"
        )
    ax.text(
        0.02,
        0.98,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved boxplot to {output_path}")


def create_scatter_plot(
    q_cont_times: List[float],
    sp_cont_times: List[float],
    output_path: Path,
    labels: List[str] = None,
) -> None:
    """Create a scatter plot comparing Q-learning (continuous) vs SP (continuous) times."""
    # Filter out zero values (pairwise) and keep corresponding labels
    if labels:
        filtered_data = [(q, s, l) for q, s, l in zip(q_cont_times, sp_cont_times, labels) if q > 0 and s > 0]
        q_vals = [d[0] for d in filtered_data]
        sp_vals = [d[1] for d in filtered_data]
        filtered_labels = [d[2] for d in filtered_data]
    else:
        filtered_data = [(q, s) for q, s in zip(q_cont_times, sp_cont_times) if q > 0 and s > 0]
        q_vals = [d[0] for d in filtered_data]
        sp_vals = [d[1] for d in filtered_data]
        filtered_labels = None
    
    if not q_vals:
        raise ValueError("No non-zero pairs found after filtering")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create scatter plot
    if filtered_labels and len(set(filtered_labels)) > 1:
        # If we have labels (e.g., from different files), use different colors
        unique_labels = sorted(list(set(filtered_labels)))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            indices = [j for j, l in enumerate(filtered_labels) if l == label]
            ax.scatter([q_vals[j] for j in indices], [sp_vals[j] for j in indices],
                      label=label, alpha=0.6, s=50, color=colors[i], zorder=2)
    else:
        ax.scatter(q_vals, sp_vals, alpha=0.6, s=50, color='#2ecc71', zorder=2)
    
    # Add diagonal line (y = x) for reference
    max_val = max(max(q_vals), max(sp_vals))
    min_val = min(min(q_vals), min(sp_vals))
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='y=x (equal performance)', alpha=0.7, zorder=1)
    
    ax.set_xlabel('Q-learning (continuous) avg time (s)', fontsize=12)
    ax.set_ylabel('SP (continuous) avg time (s)', fontsize=12)
    ax.set_title('Q-learning vs Shortest Path (Continuous)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Add statistics text
    mean_q = np.mean(q_vals)
    mean_sp = np.mean(sp_vals)
    improvement = ((mean_sp - mean_q) / mean_sp) * 100 if mean_sp > 0 else 0
    stats_text = f"Mean Q-learning: {mean_q:.2f}s\nMean SP: {mean_sp:.2f}s\nQ-learning improvement: {improvement:.1f}%"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved scatter plot to {output_path}")


def create_improvement_histogram(
    q_cont_times: List[float],
    sp_cont_times: List[float],
    output_path: Path,
    labels: List[str] = None,
) -> None:
    """Create a histogram of Q-learning improvement over SP (Q-learning - SP)."""
    # Filter out zero values (pairwise) and keep corresponding labels
    if labels:
        filtered_data = [(q, s, l) for q, s, l in zip(q_cont_times, sp_cont_times, labels) if q > 0 and s > 0]
        q_vals = [d[0] for d in filtered_data]
        sp_vals = [d[1] for d in filtered_data]
        filtered_labels = [d[2] for d in filtered_data]
    else:
        filtered_data = [(q, s) for q, s in zip(q_cont_times, sp_cont_times) if q > 0 and s > 0]
        q_vals = [d[0] for d in filtered_data]
        sp_vals = [d[1] for d in filtered_data]
        filtered_labels = None
    
    if not q_vals:
        raise ValueError("No non-zero pairs found after filtering")
    
    # Calculate improvement (negative = Q-learning is better/faster)
    improvements = [q - s for q, s in zip(q_vals, sp_vals)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    if filtered_labels and len(set(filtered_labels)) > 1:
        # If we have labels (e.g., from different files), use different colors
        unique_labels = sorted(list(set(filtered_labels)))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        # Create histogram for each label
        bins = np.linspace(min(improvements), max(improvements), 30)
        for i, label in enumerate(unique_labels):
            indices = [j for j, l in enumerate(filtered_labels) if l == label]
            label_improvements = [improvements[j] for j in indices]
            ax.hist(label_improvements, bins=bins, alpha=0.6, label=label, 
                   color=colors[i], edgecolor='black', linewidth=0.5)
        ax.legend()
    else:
        ax.hist(improvements, bins=30, alpha=0.7, color='#2ecc71', 
               edgecolor='black', linewidth=0.5)
    
    # Add vertical line at zero
    ax.axvline(0, color='red', linestyle='--', linewidth=2, 
              label='Equal performance (Q-learning = SP)', zorder=10)
    
    ax.set_xlabel('Q-learning improvement (Q-learning - SP) time (s)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Q-learning Improvement over Shortest Path (Continuous)', 
                fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Add statistics text
    mean_improvement = np.mean(improvements)
    median_improvement = np.median(improvements)
    better_count = sum(1 for imp in improvements if imp < 0)
    better_pct = (better_count / len(improvements)) * 100
    
    stats_text = (f"Mean improvement: {mean_improvement:.2f}s\n"
                 f"Median improvement: {median_improvement:.2f}s\n"
                 f"Q-learning better: {better_count}/{len(improvements)} ({better_pct:.1f}%)\n"
                 f"Note: Negative values = Q-learning is faster")
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved improvement histogram to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot boxplots from one or more Q-learning debug logs.")
    parser.add_argument(
        "log_paths",
        type=Path,
        nargs="+",
        help="Path(s) to debug log file(s). If multiple, series are aggregated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <log_stem>_boxplot.png in same directory).",
    )
    parser.add_argument(
        "--normalize_to",
        choices=["sp_cont", "sp_disc", "q_disc"],
        default=None,
        help="If set, plot normalized times. Options: 'sp_cont' or 'sp_disc' normalizes "
             "Q-learning (continuous) by SP baseline; 'q_disc' normalizes Q-learning (continuous) "
             "by Q-learning (discrete).",
    )
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="Create a scatter plot comparing Q-learning (continuous) vs SP (continuous) times.",
    )
    parser.add_argument(
        "--improvement",
        action="store_true",
        help="Create a histogram of Q-learning improvement over SP (Q-learning - SP times).",
    )
    args = parser.parse_args()

    all_q_cont, all_q_disc, all_spc, all_spd = [], [], [], []
    file_labels = []  # Track which file each data point comes from
    for log_path in args.log_paths:
        if not log_path.exists():
            raise FileNotFoundError(f"{log_path} does not exist")
        q_cont_times, q_disc_times, sp_cont_times, sp_disc_times = extract_times(log_path)
        all_q_cont.extend(q_cont_times)
        all_q_disc.extend(q_disc_times)
        all_spc.extend(sp_cont_times)
        all_spd.extend(sp_disc_times)
        # Create labels for scatter plot (extract offset or use filename)
        if len(args.log_paths) > 1:
            # Extract offset from filename if possible
            offset_match = re.search(r'offset([\d.]+)', str(log_path))
            if offset_match:
                label = f"offset {offset_match.group(1)}"
            else:
                label = log_path.stem
            file_labels.extend([label] * len(q_cont_times))
        else:
            file_labels.extend([None] * len(q_cont_times))

    # Default output: if multiple files, append '_combined_boxplot'
    if args.output is None:
        if len(args.log_paths) == 1:
            if args.scatter:
                output_path = args.log_paths[0].with_name(f"{args.log_paths[0].stem}_scatter.png")
            elif args.improvement:
                output_path = args.log_paths[0].with_name(f"{args.log_paths[0].stem}_improvement.png")
            else:
                output_path = args.log_paths[0].with_name(f"{args.log_paths[0].stem}_boxplot.png")
        else:
            first = args.log_paths[0]
            if args.scatter:
                output_path = first.with_name(f"{first.stem}_combined_scatter.png")
            elif args.improvement:
                output_path = first.with_name(f"{first.stem}_combined_improvement.png")
            else:
                output_path = first.with_name(f"{first.stem}_combined_boxplot.png")
    else:
        output_path = args.output

    # Scatter plot mode: compare Q-learning vs SP
    if args.scatter:
        if not all_q_cont or not all_spc:
            raise ValueError("Need both Q-learning (continuous) and SP (continuous) times for scatter plot")
        if len(all_q_cont) != len(all_spc):
            raise ValueError(f"Mismatch: {len(all_q_cont)} Q-cont values vs {len(all_spc)} SP-cont values")
        
        labels = file_labels if len(args.log_paths) > 1 else None
        create_scatter_plot(all_q_cont, all_spc, output_path, labels)
        return

    # Improvement histogram mode: Q-learning - SP
    if args.improvement:
        if not all_q_cont or not all_spc:
            raise ValueError("Need both Q-learning (continuous) and SP (continuous) times for improvement histogram")
        if len(all_q_cont) != len(all_spc):
            raise ValueError(f"Mismatch: {len(all_q_cont)} Q-cont values vs {len(all_spc)} SP-cont values")
        
        labels = file_labels if len(args.log_paths) > 1 else None
        create_improvement_histogram(all_q_cont, all_spc, output_path, labels)
        return

    # Normalized mode: plot normalized times
    if args.normalize_to is not None:
        import numpy as np  # local import to keep dependency explicit

        if args.normalize_to == "sp_cont":
            if not all_spc:
                raise ValueError("No SP (continuous) times found to normalize against")
            if not all_q_cont:
                raise ValueError("No Q-learning (continuous) times found")
            if len(all_q_cont) != len(all_spc):
                raise ValueError(f"Mismatch: {len(all_q_cont)} Q-cont values vs {len(all_spc)} SP-cont values")
            base_label = "SP (continuous)"
            # Pairwise normalization: each Q-cont / corresponding SP-cont, filter out zeros
            norm_q = [q / s for q, s in zip(all_q_cont, all_spc) if q > 0 and s > 0]
            if not norm_q:
                raise ValueError("No non-zero pairs found after filtering")
            data = [norm_q]
            labels = [f"Q-learning (continuous) / {base_label}"]
            colors = ["#2ecc71"]
        elif args.normalize_to == "sp_disc":
            if not all_spd:
                raise ValueError("No SP (discrete) times found to normalize against")
            if not all_q_cont:
                raise ValueError("No Q-learning (continuous) times found")
            if len(all_q_cont) != len(all_spd):
                raise ValueError(f"Mismatch: {len(all_q_cont)} Q-cont values vs {len(all_spd)} SP-disc values")
            base_label = "SP (discrete)"
            # Pairwise normalization: each Q-cont / corresponding SP-disc, filter out zeros
            norm_q = [q / s for q, s in zip(all_q_cont, all_spd) if q > 0 and s > 0]
            if not norm_q:
                raise ValueError("No non-zero pairs found after filtering")
            data = [norm_q]
            labels = [f"Q-learning (continuous) / {base_label}"]
            colors = ["#2ecc71"]
        else:  # q_disc
            if not all_q_disc:
                raise ValueError("No Q-learning (discrete) times found to normalize against")
            if not all_q_cont:
                raise ValueError("No Q-learning (continuous) times found")
            if len(all_q_cont) != len(all_q_disc):
                raise ValueError(f"Mismatch: {len(all_q_cont)} Q-cont values vs {len(all_q_disc)} Q-disc values")
            base_label = "Q-learning (discrete)"
            # Pairwise normalization: each Q-cont / corresponding Q-disc, filter out zeros
            norm_q = [q / s for q, s in zip(all_q_cont, all_q_disc) if q > 0 and s > 0]
            if not norm_q:
                raise ValueError("No non-zero pairs found after filtering")
            data = [norm_q]
            labels = [f"Q-learning (continuous) / {base_label}"]
            colors = ["#2ecc71"]

        fig, ax = plt.subplots(figsize=(6, 5))
        box = ax.boxplot(data, patch_artist=True, tick_labels=labels, showmeans=True, meanline=True)
        box["boxes"][0].set_facecolor(colors[0])
        box["boxes"][0].set_alpha(0.7)
        plt.setp(box["whiskers"], color="black")
        plt.setp(box["caps"], color="black")
        plt.setp(box["medians"], visible=False)  # Hide median line
        plt.setp(box["means"], color="black", linewidth=2)  # Show mean as line (same style as median)

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("Normalized time (ratio to baseline, pairwise)")
        ax.grid(axis="y", alpha=0.3)

        mean_val = float(np.mean(norm_q))
        median_val = float(np.median(norm_q))
        ax.text(
            0.02,
            0.98,
            f"mean={mean_val:.3f}\nmedian={median_val:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
        )

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved normalized boxplot to {output_path}")
        return

    # Default: 4-series boxplot (Q-continuous, Q-discrete, SP-continuous, SP-discrete)
    create_boxplot(all_q_cont, all_q_disc, all_spc, all_spd, output_path)


if __name__ == "__main__":
    main()





