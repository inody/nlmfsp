#!/usr/bin/env python3
"""Plot time comparisons across n_agents sweeps for analytic vs learned runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlmfsp.plot_style import apply_paper_style

apply_paper_style()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("outputs/experiments"),
        help="Root directory containing experiment outputs.",
    )
    parser.add_argument(
        "--scenario-prefix-analytic",
        type=str,
        default="analytical_gmm_agents_n",
        help="Prefix for analytical scenarios (default: analytical_gmm_agents_n). "
        "If scenarios are suffixed (e.g., _analytic), the suffix will be appended automatically.",
    )
    parser.add_argument(
        "--scenario-prefix-learned",
        type=str,
        default="learned_gmm_agents_n",
        help="Prefix for learned scenarios (default: learned_gmm_agents_n). "
        "If scenarios are suffixed (e.g., _learned), the suffix will be appended automatically.",
    )
    parser.add_argument(
        "--agent-min",
        type=int,
        default=10,
        help="Minimum n_agents to consider (default: 10).",
    )
    parser.add_argument(
        "--agent-max",
        type=int,
        default=200,
        help="Maximum n_agents to consider (default: 200).",
    )
    parser.add_argument(
        "--agent-step",
        type=int,
        default=10,
        help="Step for n_agents sweep (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/comparison_plots/gmm_agents_sweep.png"),
        help="Path to save the plot.",
    )
    parser.add_argument(
        "--title-label",
        type=str,
        default="",
        help="Optional label appended to the figure title.",
    )
    return parser.parse_args()


def find_latest_metrics(exp_dir: Path) -> Optional[Dict]:
    if not exp_dir.exists():
        return None
    exp_runs = sorted(exp_dir.glob("exp_*"), key=lambda p: p.stat().st_mtime)
    if not exp_runs:
        return None
    metrics_path = exp_runs[-1] / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open() as f:
        return json.load(f)


def parse_iteration_times_from_log(exp_run: Path) -> List[float]:
    """Parse per-iteration times from the latest training log."""
    logs_dir = exp_run / "logs"
    if not logs_dir.exists():
        return []
    logs = sorted(logs_dir.glob("training_*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return []
    try:
        text = logs[-1].read_text().splitlines()
        times: List[float] = []
        for line in text:
            if "Time:" in line:
                # Expect pattern: "Time: XXXs"
                parts = line.split("Time:")
                if len(parts) >= 2:
                    val = parts[1].strip().rstrip("s")
                    try:
                        times.append(float(val))
                    except ValueError:
                        continue
        return times
    except Exception:
        return []


def collect_time_stats(
    root: Path, prefix: str, suffix: Optional[str], agents: List[int]
) -> Tuple[List[int], List[float], List[float]]:
    """Collect per-iteration time mean/std for each agent count."""
    xs: List[int] = []
    means: List[float] = []
    stds: List[float] = []
    for n in agents:
        name = f"{prefix}{n}"
        if suffix:
            name = f"{name}{suffix}"
        scenario_dir = root / name
        metrics = find_latest_metrics(scenario_dir)
        times = metrics.get("time_per_iteration", []) if metrics else []
        if not times and metrics:
            timing = metrics.get("timing", {})
            config = metrics.get("config", {})
            total_wall = timing.get("total_wall_time_sec")
            n_outer = config.get("n_outer_iterations")
            if total_wall is not None and n_outer:
                times = [float(total_wall) / float(n_outer)]
        if not times:
            exp_runs = sorted(
                scenario_dir.glob("exp_*"), key=lambda p: p.stat().st_mtime
            )
            if exp_runs:
                times = parse_iteration_times_from_log(exp_runs[-1])
        if not times:
            continue
        arr = np.array(times, dtype=float)
        xs.append(n)
        means.append(float(arr.mean()))
        stds.append(float(arr.std()))
    return xs, means, stds


def main() -> None:
    args = parse_args()
    agents = list(range(args.agent_min, args.agent_max + 1, args.agent_step))
    palette = plt.get_cmap("tab10").colors

    # Per-iteration time with std bands (if available)
    x_a_t, m_a_t, s_a_t = collect_time_stats(
        args.experiments_root, args.scenario_prefix_analytic, "_analytic", agents
    )
    x_b_t, m_b_t, s_b_t = collect_time_stats(
        args.experiments_root, args.scenario_prefix_learned, "_learned", agents
    )
    if x_a_t or x_b_t:
        out_time = args.output.with_name(
            args.output.stem + "_time_per_iter" + args.output.suffix
        )
        plt.figure(figsize=(6.2, 4.6))
        if x_a_t:
            x = np.array(x_a_t)
            m = np.array(m_a_t)
            s = np.array(s_a_t)
            color_a = palette[0]
            plt.plot(x, m, "^--", label="Analytic", color=color_a)
            plt.fill_between(
                x, m - s, m + s, alpha=0.2, color=color_a
            )
        if x_b_t:
            x = np.array(x_b_t)
            m = np.array(m_b_t)
            s = np.array(s_b_t)
            color_b = palette[1]
            plt.plot(x, m, "o-", label="Learned", color=color_b)
            plt.fill_between(
                x, m - s, m + s, alpha=0.2, color=color_b
            )
        plt.xlabel(r"$N$")
        plt.ylabel("Time per iteration (s)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_time.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_time, dpi=150)
        out_time_pdf = out_time.with_suffix(".pdf")
        plt.savefig(out_time_pdf)
        print(f"Saved per-iteration time plot to {out_time} and {out_time_pdf}")


if __name__ == "__main__":
    main()
