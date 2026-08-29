#!/usr/bin/env python3
"""Plot pedestrian-visual V-neck agent scaling and analytical loss vs wall time."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlmfsp.plot_style import apply_paper_style

apply_paper_style()

ANALYTICAL_KEYS = (
    "loss_ipf_forward_analytical",
    "loss_td_forward_analytical",
    "loss_fk_forward_analytical",
    "loss_ipf_backward_analytical",
    "loss_td_backward_analytical",
    "loss_fk_backward_analytical",
)
AGENT_RE = re.compile(
    r"vneck_pedestrian_visual_agents_n(\d+)_(analytical_cpu_int|learned_int100)"
)


@dataclass(frozen=True)
class TimingPoint:
    backend: str
    n_agents: int
    exp_dir: Path
    time_per_iteration: float
    total_wall_time: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-root", type=Path, default=Path("outputs/experiments")
    )
    parser.add_argument(
        "--exact-main-dir",
        type=Path,
        default=Path(
            "outputs/experiments/vneck_pedestrian_visual_main_analytical_cpu_int_seed300"
        ),
    )
    parser.add_argument(
        "--nn-main-dir",
        type=Path,
        default=Path(
            "outputs/experiments/vneck_pedestrian_visual_strength_nn_normalized_w1_seed300"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/comparison_plots/vneck_pedestrian_visual_timing_seed300"),
    )
    return parser.parse_args()


def _latest_exp_with(root: Path, filename: str) -> Path:
    candidates = [exp for exp in sorted(root.glob("exp_*")) if (exp / filename).is_file()]
    if not candidates:
        raise FileNotFoundError(f"No {filename} found under {root}")
    return candidates[-1]


def _latest_numbered_checkpoint(exp_dir: Path) -> Path:
    checkpoints = sorted(
        (exp_dir / "checkpoints").glob("checkpoint_[0-9][0-9][0-9][0-9].pt")
    )
    if not checkpoints:
        raise FileNotFoundError(f"No numbered checkpoints under {exp_dir}")
    return checkpoints[-1]


def _load_checkpoint_metrics(root: Path) -> tuple[Path, dict]:
    exp_dirs = sorted(root.glob("exp_*"))
    for exp_dir in reversed(exp_dirs):
        try:
            checkpoint = _latest_numbered_checkpoint(exp_dir)
        except FileNotFoundError:
            continue
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            return exp_dir, metrics
    raise FileNotFoundError(f"No checkpoint metrics found under {root}")


def _analytical_curve(metrics: dict) -> np.ndarray:
    keys = ANALYTICAL_KEYS + (
        "loss_boundary_forward_analytical",
        "loss_boundary_backward_analytical",
    )
    length = min(
        [len(metrics.get(key, [])) for key in keys]
        + [len(metrics.get("time_per_iteration", []))]
    )
    curve = np.zeros(length, dtype=float)
    for key in keys:
        curve += np.asarray(metrics[key][:length], dtype=float)
    return curve


def discover_agent_points(root: Path) -> list[TimingPoint]:
    points: list[TimingPoint] = []
    for scenario in sorted(root.glob("vneck_pedestrian_visual_agents_n*")):
        match = AGENT_RE.fullmatch(scenario.name)
        if match is None:
            continue
        n_agents = int(match.group(1))
        backend = "Analytical" if match.group(2) == "analytical_cpu_int" else "Learned"
        try:
            exp_dir = _latest_exp_with(scenario, "metrics.json")
        except FileNotFoundError:
            continue
        metrics = json.loads((exp_dir / "metrics.json").read_text())
        total = float(metrics["timing"]["total_wall_time_sec"])
        n_outer = int(metrics["config"]["n_outer_iterations"])
        points.append(
            TimingPoint(
                backend=backend,
                n_agents=n_agents,
                exp_dir=exp_dir,
                time_per_iteration=total / n_outer,
                total_wall_time=total,
            )
        )
    return points


def plot_agents(points: list[TimingPoint], output_dir: Path, *, max_n: int = 900) -> None:
    points = [point for point in points if point.n_agents <= max_n]
    by_backend = {
        backend: sorted(
            [point for point in points if point.backend == backend],
            key=lambda point: point.n_agents,
        )
        for backend in ("Analytical", "Learned")
    }
    shared = sorted(
        {point.n_agents for point in by_backend["Analytical"]}
        & {point.n_agents for point in by_backend["Learned"]}
    )
    lookup = {
        backend: {point.n_agents: point for point in backend_points}
        for backend, backend_points in by_backend.items()
    }

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(
        shared,
        [lookup["Analytical"][n].time_per_iteration for n in shared],
        "^--",
        color="tab:blue",
        label="Analytical",
    )
    ax.plot(
        shared,
        [lookup["Learned"][n].time_per_iteration for n in shared],
        "o-",
        color="tab:orange",
        label="Learned",
    )
    ax.set_xlabel(r"$N$")
    ax.set_ylabel("Time per iteration (s)")
    ax.set_xticks(shared)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"time_per_iteration_vs_agents.{suffix}", dpi=240)
    plt.close(fig)

    with (output_dir / "time_per_iteration_vs_agents.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["backend", "n_agents", "time_per_iteration_sec", "total_wall_time_sec", "exp_dir"]
        )
        for backend in ("Analytical", "Learned"):
            for n_agents in shared:
                point = lookup[backend][n_agents]
                writer.writerow(
                    [
                        backend,
                        n_agents,
                        point.time_per_iteration,
                        point.total_wall_time,
                        point.exp_dir,
                    ]
                )


def plot_loss_vs_time(
    exact_root: Path, nn_root: Path, output_dir: Path
) -> None:
    exact_exp, exact_metrics = _load_checkpoint_metrics(exact_root)
    nn_exp, nn_metrics = _load_checkpoint_metrics(nn_root)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    rows: list[tuple[str, int, float, float]] = []
    for label, metrics, color, style in (
        ("Analytical", exact_metrics, "tab:blue", "--"),
        ("Learned", nn_metrics, "tab:orange", "-"),
    ):
        curve = _analytical_curve(metrics)
        times = np.cumsum(np.asarray(metrics["time_per_iteration"][: len(curve)], dtype=float))
        ax.plot(times, curve, linestyle=style, color=color, linewidth=2, label=label)
        rows.extend(
            (label, iteration, float(time_value), float(loss_value))
            for iteration, (time_value, loss_value) in enumerate(zip(times, curve), start=1)
        )
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Total Loss (Forward + Backward)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"analytical_loss_vs_wall_time.{suffix}", dpi=240)
    plt.close(fig)

    with (output_dir / "analytical_loss_vs_wall_time.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["backend", "iteration", "elapsed_time_sec", "total_analytical_loss"])
        writer.writerows(rows)

    (output_dir / "loss_vs_wall_time_sources.txt").write_text(
        f"Analytical: {exact_exp}\nLearned: {nn_exp}\n"
    )


def plot_interaction_loss(nn_root: Path, output_dir: Path) -> None:
    _exp_dir, metrics = _load_checkpoint_metrics(nn_root)
    values = np.asarray(metrics.get("loss_interaction_cost", []), dtype=float)
    if values.size == 0:
        raise ValueError(f"No interaction cost loss history found under {nn_root}")
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(
        np.arange(1, len(values) + 1),
        values,
        color=plt.get_cmap("tab10").colors[1],
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Interaction Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"interaction_loss_vs_iter.{suffix}", dpi=240)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points = discover_agent_points(args.experiments_root)
    plot_agents(points, args.output_dir, max_n=900)
    plot_loss_vs_time(args.exact_main_dir, args.nn_main_dir, args.output_dir)
    plot_interaction_loss(args.nn_main_dir, args.output_dir)
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
