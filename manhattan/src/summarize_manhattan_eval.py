#!/usr/bin/env python3
"""Summarize Manhattan four-way evaluation logs across seeds and instances."""

import argparse
import glob
import math
import re
import statistics
from pathlib import Path
from typing import Optional


LINE_RE = re.compile(
    r"cont=(?P<cont>[0-9.]+)\s+disc=(?P<disc>[0-9.]+)\s+"
    r"completion_cont=(?P<completion_cont>[0-9.]+)%\s+"
    r"completion_disc=(?P<completion_disc>[0-9.]+)%"
)


def method_from_line(line: str) -> Optional[str]:
    if "SALT (learned Q + OT every timestep)" in line:
        return "SALT"
    if "SALT static-assignment" in line:
        return "SALT static"
    if "Shortest-path static" in line:
        return "SP static"
    match = re.search(r"\(K=(\d+)\) Shortest-path \+ reassign", line)
    if match:
        return f"SP reassign-{match.group(1)}"
    return None


def mean_and_se(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, math.nan
    return mean, statistics.stdev(values) / math.sqrt(len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="Log paths or glob patterns")
    args = parser.parse_args()

    paths: list[Path] = []
    for value in args.logs:
        matches = [Path(path) for path in glob.glob(value)]
        paths.extend(matches or ([Path(value)] if Path(value).exists() else []))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("No evaluation logs matched.")

    values: dict[str, dict[str, list[float]]] = {}
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            method = method_from_line(line)
            match = LINE_RE.search(line)
            if method is None or match is None:
                continue
            result = values.setdefault(
                method,
                {"time": [], "completion": []},
            )
            result["time"].append(float(match.group("cont")))
            result["completion"].append(float(match.group("completion_cont")))

    if not values:
        raise SystemExit(
            "No comparison lines with completion metrics found. "
            "Use logs produced by the updated evaluator."
        )

    order = ["SALT", "SALT static", "SP static", "SP reassign-5", "SP reassign-10"]
    print(f"Parsed {len(paths)} log(s). Continuous evaluation, mean over instances.")
    print(f"{'Method':<16} {'N':>6} {'Time mean':>12} {'Time SE':>10} {'Completion':>12} {'Comp. SE':>10}")
    print("-" * 72)
    for method in order + sorted(set(values) - set(order)):
        if method not in values:
            continue
        times = values[method]["time"]
        completions = values[method]["completion"]
        time_mean, time_se = mean_and_se(times)
        completion_mean, completion_se = mean_and_se(completions)
        print(
            f"{method:<16} {len(times):>6d} {time_mean:>12.2f} {time_se:>10.2f} "
            f"{completion_mean:>11.2f}% {completion_se:>9.2f}%"
        )


if __name__ == "__main__":
    main()
