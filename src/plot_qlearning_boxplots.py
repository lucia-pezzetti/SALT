#!/usr/bin/env python3
"""
Parse Q-learning debug logs and produce boxplots comparing average times.
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def extract_times(filepath: Path) -> Tuple[List[float], List[float], List[float]]:
    """Extract Q-learning, SP continuous, and SP discrete avg times."""
    q_times: List[float] = []
    sp_cont_times: List[float] = []
    sp_disc_times: List[float] = []

    with filepath.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            q_match = re.search(r"Q-learning avg time:\s*([\d.]+)", line)
            if q_match:
                q_times.append(float(q_match.group(1)))
                continue

            sp_cont_match = re.search(r"SP \(continuous\) avg time:\s*([\d.]+)", line)
            if sp_cont_match:
                sp_cont_times.append(float(sp_cont_match.group(1)))
                continue

            sp_disc_match = re.search(r"SP \(discrete\) avg time:\s*([\d.]+)", line)
            if sp_disc_match:
                sp_disc_times.append(float(sp_disc_match.group(1)))
                continue

    if not q_times:
        raise ValueError(f"No Q-learning avg times found in {filepath}")
    if not sp_disc_times and not sp_cont_times:
        raise ValueError(f"No SP avg times found in {filepath}")

    return q_times, sp_cont_times, sp_disc_times


def create_boxplot(
    q_times: List[float],
    sp_cont_times: List[float],
    sp_disc_times: List[float],
    output_path: Path,
) -> None:
    """Create and save a boxplot comparing the extracted times."""
    data = [q_times]
    labels = ["Q-learning"]
    colors = ["#2ecc71"]

    if sp_cont_times:
        data.append(sp_cont_times)
        labels.append("SP (continuous)")
        colors.append("#3498db")

    if sp_disc_times:
        data.append(sp_disc_times)
        labels.append("SP (discrete)")
        colors.append("#e74c3c")

    fig, ax = plt.subplots(figsize=(9, 6))
    box = ax.boxplot(data, patch_artist=True, tick_labels=labels)

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.setp(box["whiskers"], color="black")
    plt.setp(box["caps"], color="black")
    plt.setp(box["medians"], color="black")
    plt.setp(box["fliers"], markeredgecolor="black")

    ax.set_ylabel("Average time (s)")
    ax.grid(axis="y", alpha=0.3)

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
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved boxplot to {output_path}")


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
        choices=["sp_cont", "sp_disc"],
        default=None,
        help="If set, plot only Q-learning times normalized by the mean SP baseline "
             "(continuous or discrete).",
    )
    args = parser.parse_args()

    all_q, all_spc, all_spd = [], [], []
    for log_path in args.log_paths:
        if not log_path.exists():
            raise FileNotFoundError(f"{log_path} does not exist")
        q_times, sp_cont_times, sp_disc_times = extract_times(log_path)
        all_q.extend(q_times)
        all_spc.extend(sp_cont_times)
        all_spd.extend(sp_disc_times)

    # Default output: if multiple files, append '_combined_boxplot'
    if args.output is None:
        if len(args.log_paths) == 1:
            output_path = args.log_paths[0].with_name(f"{args.log_paths[0].stem}_boxplot.png")
        else:
            first = args.log_paths[0]
            output_path = first.with_name(f"{first.stem}_combined_boxplot.png")
    else:
        output_path = args.output

    # Normalized mode: plot only Q-learning times normalized by chosen SP baseline
    if args.normalize_to is not None:
        import numpy as np  # local import to keep dependency explicit

        if args.normalize_to == "sp_cont":
            if not all_spc:
                raise ValueError("No SP (continuous) times found to normalize against")
            base_mean = float(np.mean(all_spc))
            base_label = "SP (continuous)"
        else:  # sp_disc
            if not all_spd:
                raise ValueError("No SP (discrete) times found to normalize against")
            base_mean = float(np.mean(all_spd))
            base_label = "SP (discrete)"

        norm_q = [t / base_mean for t in all_q]

        fig, ax = plt.subplots(figsize=(6, 5))
        box = ax.boxplot([norm_q], patch_artist=True, tick_labels=[f"Q-learning / {base_label}"])
        box["boxes"][0].set_facecolor("#2ecc71")
        box["boxes"][0].set_alpha(0.7)
        plt.setp(box["whiskers"], color="black")
        plt.setp(box["caps"], color="black")
        plt.setp(box["medians"], color="black")

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("Normalized time (ratio to baseline mean)")
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

    # Default: 3-series boxplot (Q, SP continuous, SP discrete)
    create_boxplot(all_q, all_spc, all_spd, output_path)


if __name__ == "__main__":
    main()





