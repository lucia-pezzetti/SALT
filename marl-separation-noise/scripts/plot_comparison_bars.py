#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _parse_p_from_run_dir(run_dir: str) -> float:
    m = re.search(r"/p([0-9.]+)_", run_dir)
    if not m:
        raise ValueError(f"Cannot parse p from run dir: {run_dir}")
    return float(m.group(1))


def _load_records(
    run_dirs: List[str],
    noise_kinds: List[str],
    algos: List[str],
) -> Dict[Tuple[str, str], Dict[str, List[float]]]:
    data = {
        (n, a): {"p": [], "reach_mean": [], "reach_std": [], "ot_mean": [], "ot_std": []}
        for n in noise_kinds
        for a in algos
    }

    for rd in run_dirs:
        p = _parse_p_from_run_dir(rd)
        comparison_path = os.path.join(rd, "comparison.json")
        with open(comparison_path, "r", encoding="utf-8") as f:
            runs = json.load(f)
        by_noise = {r["noise"]: r for r in runs}

        for n in noise_kinds:
            r = by_noise[n]
            mapping = {
                "Separation[q]": r.get("sep_agg_by_method", {}).get("q"),
                "Separation[ppo]": r.get("sep_agg_by_method", {}).get("ppo"),
                "MAPPO[relative_targets]": r.get("mappo_agg_by_mode", {}).get("relative_targets"),
                "IPPO[relative_targets]": r.get("ippo_agg_by_mode", {}).get("relative_targets"),
                "QMIX[relative_targets]": r.get("qmix_agg_by_mode", {}).get("relative_targets"),
                "VDN[relative_targets]": r.get("vdn_agg_by_mode", {}).get("relative_targets"),
            }
            for a, agg in mapping.items():
                if agg is None:
                    continue
                data[(n, a)]["p"].append(p)
                data[(n, a)]["reach_mean"].append(100.0 * agg["reach_rate_mean"])
                data[(n, a)]["reach_std"].append(100.0 * agg["reach_rate_std"])
                data[(n, a)]["ot_mean"].append(agg["terminal_ot_cost_mean"])
                data[(n, a)]["ot_std"].append(agg["terminal_ot_cost_std"])

    # Sort each series by p
    for k in data:
        idx = np.argsort(data[k]["p"])
        for field in ["p", "reach_mean", "reach_std", "ot_mean", "ot_std"]:
            arr = np.asarray(data[k][field], dtype=float)
            data[k][field] = arr[idx].tolist()
    return data


def _load_time_box_data(
    run_dirs: List[str],
    noise_kinds: List[str],
    algos: List[str],
) -> Dict[Tuple[str, str], Dict[float, List[float]]]:
    """Load per-seed values for mean_time_to_reach_including_unreached."""
    out: Dict[Tuple[str, str], Dict[float, List[float]]] = {
        (n, a): {} for n in noise_kinds for a in algos
    }

    for rd in run_dirs:
        p = _parse_p_from_run_dir(rd)
        comparison_path = os.path.join(rd, "comparison.json")
        for n in noise_kinds:
            def _extract_seed_vals(seed_items):
                if not seed_items:
                    return []
                vals = []
                for item in seed_items:
                    v = item.get("mean_time_to_reach_including_unreached")
                    if v is not None:
                        vals.append(float(v))
                return vals

            kind_dir = os.path.join(rd, n)
            def _load_all_seeds(fname: str):
                path = os.path.join(kind_dir, fname)
                if not os.path.exists(path):
                    return []
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                return payload.get("all_seeds", [])

            sep_q_vals = _extract_seed_vals(_load_all_seeds("sep_q_metrics.json"))
            sep_ppo_vals = _extract_seed_vals(_load_all_seeds("sep_ppo_metrics.json"))
            mappo_vals = _extract_seed_vals(_load_all_seeds("mappo_relative_targets_metrics.json"))
            ippo_vals = _extract_seed_vals(_load_all_seeds("ippo_relative_targets_metrics.json"))
            qmix_vals = _extract_seed_vals(_load_all_seeds("qmix_relative_targets_metrics.json"))
            vdn_vals = _extract_seed_vals(_load_all_seeds("vdn_relative_targets_metrics.json"))

            mapping = {
                "Separation[q]": sep_q_vals,
                "Separation[ppo]": sep_ppo_vals,
                "MAPPO[relative_targets]": mappo_vals,
                "IPPO[relative_targets]": ippo_vals,
                "QMIX[relative_targets]": qmix_vals,
                "VDN[relative_targets]": vdn_vals,
            }
            for a, vals in mapping.items():
                if not vals:
                    continue
                out[(n, a)].setdefault(p, [])
                out[(n, a)][p].extend(vals)

    return out


def _plot_grouped_bars(
    data: Dict[Tuple[str, str], Dict[str, List[float]]],
    noise_kinds: List[str],
    algos: List[str],
    colors: Dict[str, str],
    metric: str,
    p_vals: List[float],
):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=False)
    p_labels = [f"{p:.2f}" for p in p_vals]
    x = np.arange(len(p_labels))
    width = 0.18

    for c, n in enumerate(noise_kinds):
        ax = axes[c]
        shown = [a for a in algos if len(data[(n, a)]["p"]) > 0]
        for i, a in enumerate(shown):
            offs = (i - (len(shown) - 1) / 2) * width
            if metric == "reach":
                y_src = np.asarray(data[(n, a)]["reach_mean"], dtype=float)
                e_src = np.asarray(data[(n, a)]["reach_std"], dtype=float)
                ylabel = "Reach rate (%)"
                super_title = "Reach rate vs noise level (grouped bars)"
            else:
                y_src = np.asarray(data[(n, a)]["ot_mean"], dtype=float)
                e_src = np.asarray(data[(n, a)]["ot_std"], dtype=float)
                ylabel = "Terminal OT cost"
                super_title = "Terminal OT cost vs noise level (grouped bars)"
            p_src = np.asarray(data[(n, a)]["p"], dtype=float)
            val_by_p = {
                float(pp): (float(vv), float(ee))
                for pp, vv, ee in zip(p_src, y_src, e_src)
            }
            x_pos: List[float] = []
            y: List[float] = []
            e: List[float] = []
            for j, p in enumerate(p_vals):
                if p not in val_by_p:
                    continue
                vv, ee = val_by_p[p]
                x_pos.append(float(x[j]) + offs)
                y.append(vv)
                e.append(ee)
            if not y:
                continue

            ax.bar(
                np.asarray(x_pos, dtype=float),
                y,
                width,
                label=a,
                color=colors[a],
                alpha=0.9,
                yerr=e,
                capsize=3,
                error_kw={"elinewidth": 1, "alpha": 0.8},
            )

        ax.set_title(f"{n.capitalize()} noise", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(p_labels)
        ax.set_xlabel("Noise probability p")
        if c == 0:
            ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    fig.suptitle(super_title, fontsize=13, y=0.98)
    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    return fig


def _plot_grouped_boxplots(
    box_data: Dict[Tuple[str, str], Dict[float, List[float]]],
    noise_kinds: List[str],
    algos: List[str],
    colors: Dict[str, str],
    p_vals: List[float],
):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=False)
    p_labels = [f"{p:.2f}" for p in p_vals]
    x = np.arange(len(p_vals))
    width = 0.18

    for c, n in enumerate(noise_kinds):
        ax = axes[c]
        shown = [a for a in algos if any(box_data[(n, a)].get(p, []) for p in p_vals)]
        for i, a in enumerate(shown):
            offs = (i - (len(shown) - 1) / 2) * width
            for j, p in enumerate(p_vals):
                vals = box_data[(n, a)].get(p, [])
                if not vals:
                    continue
                bp = ax.boxplot(
                    [vals],
                    positions=[x[j] + offs],
                    widths=width * 0.9,
                    patch_artist=True,
                    showfliers=False,
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(colors[a])
                    patch.set_alpha(0.6)
                for median in bp["medians"]:
                    median.set_color("black")
                    median.set_linewidth(1.5)

        ax.set_title(f"{n.capitalize()} noise", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(p_labels)
        ax.set_xlabel("Noise probability p")
        if c == 0:
            ax.set_ylabel("Mean time incl. unreached")
        ax.grid(axis="y", alpha=0.3)

    handles = []
    labels = []
    for a in algos:
        if any(any(box_data[(n, a)].values()) for n in noise_kinds):
            handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=colors[a], alpha=0.6))
            labels.append(a)
    fig.suptitle("Mean time to reach (including unreached) vs noise level (boxplots)", fontsize=13, y=0.98)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    return fig


def main():
    ap = argparse.ArgumentParser(description="Plot grouped-bar comparison charts from comparison runs.")
    ap.add_argument(
        "--run_dirs",
        type=str,
        required=True,
        help="Comma-separated run directories containing comparison.json",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="/home/luciapezzetti/Separation-Principle-MARL/marl-separation-noise/runs/comparison",
        help="Output directory for plots",
    )
    args = ap.parse_args()

    run_dirs = [r.strip() for r in args.run_dirs.split(",") if r.strip()]
    p_vals = sorted({_parse_p_from_run_dir(rd) for rd in run_dirs})
    if not p_vals:
        raise ValueError("No valid run directories provided.")
    noise_kinds = ["individual", "local", "global"]
    algos = [
        "Separation[q]",
        "Separation[ppo]",
        "MAPPO[relative_targets]",
        "IPPO[relative_targets]",
        "QMIX[relative_targets]",
        "VDN[relative_targets]",
    ]
    colors = {
        "Separation[q]": "#1f77b4",
        "Separation[ppo]": "#2ca02c",
        "MAPPO[relative_targets]": "#d62728",
        "IPPO[relative_targets]": "#ff7f0e",
        "QMIX[relative_targets]": "#8c564b",
        "VDN[relative_targets]": "#9467bd",
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    data = _load_records(run_dirs, noise_kinds, algos)
    box_data = _load_time_box_data(run_dirs, noise_kinds, algos)

    os.makedirs(args.outdir, exist_ok=True)
    fig_reach = _plot_grouped_bars(
        data, noise_kinds, algos, colors, metric="reach", p_vals=p_vals
    )
    fig_ot = _plot_grouped_bars(
        data, noise_kinds, algos, colors, metric="ot", p_vals=p_vals
    )
    fig_time_box = _plot_grouped_boxplots(
        box_data, noise_kinds, algos, colors, p_vals=p_vals
    )

    reach_png = os.path.join(args.outdir, "performance_vs_p_reach_bars.png")
    reach_pdf = os.path.join(args.outdir, "performance_vs_p_reach_bars.pdf")
    ot_png = os.path.join(args.outdir, "performance_vs_p_ot_bars.png")
    ot_pdf = os.path.join(args.outdir, "performance_vs_p_ot_bars.pdf")
    time_box_png = os.path.join(args.outdir, "performance_vs_p_time_incl_unreached_boxplots.png")
    time_box_pdf = os.path.join(args.outdir, "performance_vs_p_time_incl_unreached_boxplots.pdf")

    fig_reach.savefig(reach_png, dpi=220, bbox_inches="tight")
    fig_reach.savefig(reach_pdf, bbox_inches="tight")
    fig_ot.savefig(ot_png, dpi=220, bbox_inches="tight")
    fig_ot.savefig(ot_pdf, bbox_inches="tight")
    fig_time_box.savefig(time_box_png, dpi=220, bbox_inches="tight")
    fig_time_box.savefig(time_box_pdf, bbox_inches="tight")

    print(reach_png)
    print(reach_pdf)
    print(ot_png)
    print(ot_pdf)
    print(time_box_png)
    print(time_box_pdf)


if __name__ == "__main__":
    main()

