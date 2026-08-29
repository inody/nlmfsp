#!/usr/bin/env python3
"""Plot available V-neck pedestrian-visual interaction-strength runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlmfsp.costs import PedestrianVisualCost
from nlmfsp.plot_style import apply_paper_style

from plot_vneck_interaction_strength_proximity_compare import (
    _resolve_device,
    sample_forward_trajectories,
)

apply_paper_style()

ANALYTICAL_KEYS = (
    "loss_ipf_forward_analytical",
    "loss_td_forward_analytical",
    "loss_fk_forward_analytical",
    "loss_ipf_backward_analytical",
    "loss_td_backward_analytical",
    "loss_fk_backward_analytical",
)


@dataclass(frozen=True)
class Run:
    backend: str
    weight: float
    range_scale: float
    interaction_steps: int
    exp_dir: Path
    latest_checkpoint: Path
    best_checkpoint: Path
    best_iteration: int
    best_analytical_loss: float
    completed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--weights", type=float, nargs="*", default=[0.5, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--num-trajectories", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path(
            "outputs/experiments/analytical_vneck_navigation/exp_20260510_120517"
        ),
        help="No-interaction V-neck experiment used as the common w=0 baseline.",
    )
    parser.add_argument(
        "--experiments-root", type=Path, default=Path("outputs/experiments")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/comparison_plots/vneck_pedestrian_visual_strength_seed300_available"
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _numbered_checkpoints(exp_dir: Path) -> list[Path]:
    return sorted((exp_dir / "checkpoints").glob("checkpoint_[0-9][0-9][0-9][0-9].pt"))


def _analytical_total(metrics: dict) -> float:
    return sum(float(metrics.get(key, 0.0)) for key in ANALYTICAL_KEYS)


def _checkpoint_iteration(checkpoint: dict, path: Path) -> int:
    iteration = checkpoint.get("iteration")
    if iteration is not None:
        return int(iteration)
    return int(path.stem.rsplit("_", 1)[-1])


def _select_checkpoints(exp_dir: Path) -> tuple[Path, Path, int, float]:
    numbered = _numbered_checkpoints(exp_dir)
    if not numbered:
        raise FileNotFoundError(f"No numbered checkpoints in {exp_dir}")

    latest = numbered[-1]
    best_path = latest
    best_iteration = -1
    best_total = float("inf")
    for path in numbered:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        total = _analytical_total(payload.get("metrics_analytical", {}))
        if total < best_total:
            best_path = path
            best_iteration = _checkpoint_iteration(payload, path)
            best_total = total
    return latest, best_path, best_iteration, best_total


def discover_runs(
    experiments_root: Path, seed: int, expected_weights: list[float]
) -> list[Run]:
    expected = {float(weight) for weight in expected_weights}
    candidates: dict[tuple[str, float], Run] = {}
    pattern = f"vneck_pedestrian_visual_strength_*_seed{seed}/exp_*"
    for exp_dir in sorted(experiments_root.glob(pattern)):
        config_path = exp_dir / "config.json"
        if not config_path.is_file():
            continue
        payload = _load_json(config_path)
        merged = payload.get("config_merged", payload.get("config", {}))
        if merged.get("cost_type") != "pedestrian_visual":
            continue
        backend = str(merged.get("interaction_backend"))
        if backend not in {"exact", "nn"}:
            continue
        weight = float(merged.get("int_cost_weight", float("nan")))
        if weight not in expected:
            continue
        try:
            latest, best, best_iteration, best_loss = _select_checkpoints(exp_dir)
        except FileNotFoundError:
            continue
        run = Run(
            backend=backend,
            weight=weight,
            range_scale=float(merged.get("int_cost_sigma", 1.0)),
            interaction_steps=int(merged.get("n_interaction_steps", 0)),
            exp_dir=exp_dir,
            latest_checkpoint=latest,
            best_checkpoint=best,
            best_iteration=best_iteration,
            best_analytical_loss=best_loss,
            completed=(exp_dir / "metrics.json").is_file(),
        )
        key = (backend, weight)
        current = candidates.get(key)
        if current is None or (
            run.completed,
            run.latest_checkpoint.name,
            run.exp_dir.name,
        ) > (
            current.completed,
            current.latest_checkpoint.name,
            current.exp_dir.name,
        ):
            candidates[key] = run
    return sorted(candidates.values(), key=lambda run: (run.backend, run.weight))


def pedestrian_visual_affinity(
    xs: torch.Tensor, *, range_scale: float, device: torch.device
) -> torch.Tensor:
    """Return the unweighted directed-pair kernel average at each time."""
    cost = PedestrianVisualCost(range_scale=range_scale, weight=1.0)
    cost.set_chunk_sizes(query_chunk_size=200, ref_chunk_size=200)
    xs_device = xs.to(device)
    with torch.no_grad():
        values = cost.evaluate_batched(xs_device, xs_device, time_block_size=10)
    return values.mean(dim=1).squeeze(-1).cpu()


def _latest_metrics(run: Run) -> dict:
    payload = torch.load(run.latest_checkpoint, map_location="cpu", weights_only=False)
    return payload.get("metrics", {})


def _analytical_history(metrics: dict) -> np.ndarray:
    available = [metrics.get(key, []) for key in ANALYTICAL_KEYS]
    length = min((len(values) for values in available), default=0)
    if length == 0:
        return np.empty(0)
    total = np.zeros(length, dtype=float)
    for values in available:
        total += np.asarray(values[:length], dtype=float)
    return np.minimum.accumulate(total)


def _colors(weights: list[float]) -> dict[float, object]:
    cmap = plt.get_cmap("viridis")
    return {
        weight: cmap(index / max(1, len(weights) - 1))
        for index, weight in enumerate(sorted(weights))
    }


def plot_response(
    runs: list[Run],
    response: dict[tuple[str, float], tuple[torch.Tensor, torch.Tensor]],
    output_dir: Path,
) -> None:
    weights = sorted({run.weight for run in runs} | {0.0})
    colors = _colors(weights)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for axis, backend in zip(axes, ("exact", "nn")):
        if ("baseline", 0.0) in response:
            time_grid, affinity = response[("baseline", 0.0)]
            axis.plot(
                time_grid.numpy(),
                affinity.numpy(),
                color=colors[0.0],
                linewidth=2,
                label="baseline",
            )
        for run in runs:
            if run.backend != backend:
                continue
            time_grid, affinity = response[(run.backend, run.weight)]
            axis.plot(
                time_grid.numpy(),
                affinity.numpy(),
                color=colors[run.weight],
                linewidth=2,
                label=rf"$w={run.weight:g}$",
            )
        axis.set_xlabel(r"$t$")
        axis.set_ylabel(r"$A_k(t)$")
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(output_dir / f"pedestrian_visual_affinity_vs_time{suffix}", dpi=240)
    plt.close(fig)

    for backend in ("exact", "nn"):
        fig, axis = plt.subplots(figsize=(5.4, 4.0))
        if ("baseline", 0.0) in response:
            time_grid, affinity = response[("baseline", 0.0)]
            axis.plot(
                time_grid.numpy(),
                affinity.numpy(),
                color=colors[0.0],
                linewidth=2,
                label="baseline",
            )
        for run in runs:
            if run.backend != backend:
                continue
            time_grid, affinity = response[(run.backend, run.weight)]
            axis.plot(
                time_grid.numpy(),
                affinity.numpy(),
                color=colors[run.weight],
                linewidth=2,
                label=rf"$w={run.weight:g}$",
            )
        axis.set_xlabel(r"$t$")
        axis.set_ylabel(r"$A_k(t)$")
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        for suffix in (".png", ".pdf"):
            fig.savefig(output_dir / f"{backend}_timeseries{suffix}", dpi=240)
        plt.close(fig)


def plot_losses(runs: list[Run], output_dir: Path) -> None:
    weights = sorted({run.weight for run in runs})
    colors = _colors(weights)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for axis, backend in zip(axes, ("exact", "nn")):
        for run in runs:
            if run.backend != backend:
                continue
            history = _analytical_history(_latest_metrics(run))
            if history.size == 0:
                continue
            axis.semilogy(
                np.arange(1, len(history) + 1),
                history,
                color=colors[run.weight],
                linewidth=2,
                label=rf"$w={run.weight:g}$",
            )
        axis.set_xlabel("Outer iteration")
        axis.set_ylabel("Best-so-far analytical loss")
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(output_dir / f"analytical_loss_vs_iteration{suffix}", dpi=240)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.4, 4.0))
    for run in runs:
        if run.backend != "nn":
            continue
        history = np.asarray(_latest_metrics(run).get("loss_interaction_cost", []), dtype=float)
        if history.size == 0:
            continue
        axis.semilogy(
            np.arange(1, len(history) + 1),
            history,
            color=colors[run.weight],
            linewidth=2,
            label=rf"$w={run.weight:g}$",
        )
    axis.set_xlabel("Outer iteration")
    axis.set_ylabel("Interaction learning loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(output_dir / f"interaction_loss_vs_iteration{suffix}", dpi=240)
    plt.close(fig)


def write_summary(runs: list[Run], output_dir: Path) -> None:
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "backend",
                "weight",
                "completed",
                "latest_iteration",
                "best_analytical_iteration",
                "best_analytical_loss",
                "interaction_steps",
                "experiment_dir",
            ]
        )
        for run in runs:
            writer.writerow(
                [
                    run.backend,
                    run.weight,
                    run.completed,
                    _checkpoint_iteration(
                        torch.load(run.latest_checkpoint, map_location="cpu", weights_only=False),
                        run.latest_checkpoint,
                    ),
                    run.best_iteration,
                    run.best_analytical_loss,
                    run.interaction_steps,
                    run.exp_dir,
                ]
            )


def write_response(
    runs: list[Run],
    response: dict[tuple[str, float], tuple[torch.Tensor, torch.Tensor]],
    output_dir: Path,
) -> None:
    with (output_dir / "timeseries.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["backend", "weight", "time", "Ak"])
        if ("baseline", 0.0) in response:
            time_grid, affinity = response[("baseline", 0.0)]
            for time_value, affinity_value in zip(time_grid.tolist(), affinity.tolist()):
                writer.writerow(["baseline", 0.0, float(time_value), float(affinity_value)])
        for run in runs:
            time_grid, affinity = response[(run.backend, run.weight)]
            for time_value, affinity_value in zip(time_grid.tolist(), affinity.tolist()):
                writer.writerow(
                    [run.backend, run.weight, float(time_value), float(affinity_value)]
                )


def main() -> None:
    args = parse_args()
    runs = discover_runs(args.experiments_root, args.seed, args.weights)
    if not runs:
        raise SystemExit("No matching V-neck pedestrian-visual runs found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device, "auto")

    response: dict[tuple[str, float], tuple[torch.Tensor, torch.Tensor]] = {}
    if args.baseline_dir.is_dir():
        latest, best, best_iteration, best_loss = _select_checkpoints(args.baseline_dir)
        print(
            f"baseline w=0: best analytical checkpoint {best.name} "
            f"(iteration={best_iteration}, loss={best_loss:.6g})"
        )
        time_grid, xs = sample_forward_trajectories(
            args.baseline_dir,
            checkpoint_arg=best.name,
            num_trajectories=args.num_trajectories,
            device_override=str(device),
        )
        response[("baseline", 0.0)] = (
            time_grid,
            pedestrian_visual_affinity(xs, range_scale=1.0, device=device),
        )
    for run in runs:
        print(
            f"{run.backend:5s} w={run.weight:g}: best analytical checkpoint "
            f"{run.best_checkpoint.name} (loss={run.best_analytical_loss:.6g})"
        )
        time_grid, xs = sample_forward_trajectories(
            run.exp_dir,
            checkpoint_arg=run.best_checkpoint.name,
            num_trajectories=args.num_trajectories,
            device_override=str(device),
        )
        response[(run.backend, run.weight)] = (
            time_grid,
            pedestrian_visual_affinity(
                xs, range_scale=run.range_scale, device=device
            ),
        )

    plot_response(runs, response, args.output_dir)
    plot_losses(runs, args.output_dir)
    write_summary(runs, args.output_dir)
    write_response(runs, response, args.output_dir)
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
