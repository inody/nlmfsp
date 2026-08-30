#!/usr/bin/env python3
"""
Compare V-neck Gaussian-kernel proximity across interaction strengths for exact vs NN.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_vneck_interaction_proximity import (
    _best_analytical_checkpoint,
    _resolve_device,
    gaussian_kernel_affinity_series,
)
from plot_experiment_trajectories import (
    _load_checkpoint_interaction_modules,
    load_experiment_config,
    resolve_checkpoint,
)

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlmfsp.fbsde import FBSDESolver
from nlmfsp.interaction import create_interactions
from nlmfsp.plot_style import apply_paper_style
from nlmfsp.problem_setup import setup_problem_artifacts
from nlmfsp.rng_utils import seed_offset, torch_seed_context

apply_paper_style()


DEFAULT_WEIGHTS = (0.0, 0.25, 0.5, 1.0)


@dataclass(frozen=True)
class RunSpec:
    backend: str
    weight: float
    sigma: float
    exp_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional seed list to aggregate. If omitted, the single --seed "
            "run is reproduced."
        ),
    )
    parser.add_argument("--n-agents", type=int, default=500)
    parser.add_argument(
        "--exact-root",
        type=Path,
        default=Path("outputs/experiments/analytical_vneck_navigation_gpu_dt02"),
    )
    parser.add_argument(
        "--nn-root",
        type=Path,
        default=Path("outputs/experiments/vneck_gaussian_conv_mean_interaction_strength_nn_gpu_dt02"),
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="*",
        default=list(DEFAULT_WEIGHTS),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best-analytical",
        help="Checkpoint filename or special value 'best-analytical'.",
    )
    parser.add_argument("--num-trajectories", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/comparison_plots/vneck_gaussian_conv_mean_interaction_strength_exact_vs_nn"),
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="vneck_gaussian_conv_mean_interaction_strength_exact_vs_nn_seed200_n500",
    )
    return parser.parse_args()


def _weight_key(weight: float) -> str:
    if math.isclose(weight, round(weight), rel_tol=0.0, abs_tol=1e-9):
        return str(int(round(weight)))
    return f"{weight:g}"


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _run_weight(config_merged: dict) -> Optional[float]:
    if config_merged.get("cost_type") == "zero":
        return 0.0
    if config_merged.get("cost_type") == "gaussian_conv_mean":
        weight = config_merged.get("int_cost_weight")
        return None if weight is None else float(weight)
    return None


def _discover_root(root: Path, backend: str, seed: int, n_agents: int, weights: list[float]) -> Dict[float, RunSpec]:
    runs: Dict[float, RunSpec] = {}
    for exp in sorted(root.glob("exp_*")):
        cfg = exp / "config.json"
        if not cfg.exists():
            continue
        data = _load_json(cfg)
        merged = data.get("config_merged", data.get("config", {}))
        if int(merged.get("seed", -1)) != seed:
            continue
        if int(merged.get("n_agents", -1)) != n_agents:
            continue
        if float(merged.get("dt", -1.0)) != 0.02:
            continue
        weight = _run_weight(merged)
        if weight is None or weight not in weights:
            continue
        sigma_raw = merged.get("int_cost_sigma")
        sigma = 1.0 if sigma_raw is None else float(sigma_raw)
        # prefer runs with metrics present
        score = ((exp / "metrics.json").exists(), exp.name)
        current = runs.get(weight)
        if current is None:
            runs[weight] = RunSpec(backend=backend, weight=weight, sigma=sigma, exp_dir=exp)
        else:
            current_score = ((current.exp_dir / "metrics.json").exists(), current.exp_dir.name)
            if score > current_score:
                runs[weight] = RunSpec(backend=backend, weight=weight, sigma=sigma, exp_dir=exp)
    return runs


def sample_forward_trajectories(
    exp_dir: Path,
    *,
    checkpoint_arg: Optional[str],
    num_trajectories: int,
    device_override: Optional[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    config, cli_args, _ = load_experiment_config(exp_dir)
    if checkpoint_arg == "best-analytical":
        checkpoint_path = _best_analytical_checkpoint(exp_dir)
    else:
        checkpoint_path = resolve_checkpoint(exp_dir, checkpoint_arg)

    cli_problem_config = getattr(cli_args, "problem_config", None)
    candidate = None
    if cli_problem_config:
        candidate = (PROJECT_ROOT / cli_problem_config).resolve()
    if candidate is not None and candidate.is_file():
        config.problem_config_path = str(candidate)
    else:
        basename = Path(str(config.problem_config_path)).name
        local_candidate = PROJECT_ROOT / "configs" / "problems" / basename
        if local_candidate.is_file():
            config.problem_config_path = str(local_candidate)

    default_device = getattr(cli_args, "device", None) or str(config.device) or "auto"
    device = _resolve_device(device_override, default_device)
    config.device = device
    cli_args.device = str(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    (
        _mu_0,
        _mu_1,
        sample_rho0,
        _sample_rho1,
        _mu_0_components,
        _mu_1_components,
        _obstacle_config,
        _xlim,
        _ylim,
        base_drift_fn,
        base_cost_fn,
        _log_p0,
        _log_p1,
    ) = setup_problem_artifacts(config)

    drift_net, cost_net, *_ = create_interactions(config)
    solver = FBSDESolver(
        config=config,
        drift_net=drift_net,
        cost_net=cost_net,
        base_drift_fn=base_drift_fn,
        base_cost_fn=base_cost_fn,
    )

    solver_state = checkpoint["solver_state"]
    solver.value_net.load_state_dict(solver_state["value_net"])
    solver.value_hat_net.load_state_dict(solver_state["value_hat_net"])
    solver.grad_net.load_state_dict(solver_state["grad_net"])
    solver.grad_hat_net.load_state_dict(solver_state["grad_hat_net"])
    _load_checkpoint_interaction_modules(solver, solver_state, config)

    sampling_seed = getattr(config, "visualization_seed", None)
    if sampling_seed is None:
        sampling_seed = int(getattr(config, "seed", 123))

    with torch_seed_context(seed_offset(sampling_seed, 0)):
        x0 = sample_rho0(max(1, int(num_trajectories)))

    with torch.no_grad():
        xs_forward, _, _ = solver.sample_forward_trajectory(
            x0, seed=seed_offset(sampling_seed, 2)
        )

    return config.time_grid.detach().cpu(), xs_forward.detach().cpu()


def _plot_summary(
    output_path: Path,
    *,
    seed: int,
    series_exact: Dict[float, torch.Tensor],
    series_nn: Dict[float, torch.Tensor],
    time_grid: torch.Tensor,
) -> None:
    weights = sorted(set(series_exact) & set(series_nn))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    cmap = plt.get_cmap("viridis")
    colors = {w: cmap(i / max(1, len(weights) - 1)) for i, w in enumerate(weights)}
    baseline_exact = series_exact[0.0]
    baseline_nn = series_nn[0.0]

    for w in weights:
        label = "baseline" if math.isclose(w, 0.0) else f"w={w:g}"
        axes[0, 0].plot(time_grid.numpy(), series_exact[w].numpy(), label=label, color=colors[w], linewidth=2)
        axes[1, 0].plot(time_grid.numpy(), series_nn[w].numpy(), label=label, color=colors[w], linewidth=2)

    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$A_k(t)$")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    x = np.arange(len(weights))
    tick_labels = ["0" if math.isclose(w, 0.0) else f"{w:g}" for w in weights]
    exact_gap_avg = np.array([float((series_exact[w] - baseline_exact).mean().item()) for w in weights])
    nn_gap_avg = np.array([float((series_nn[w] - baseline_nn).mean().item()) for w in weights])
    exact_gap_final = np.array([float((series_exact[w][-1] - baseline_exact[-1]).item()) for w in weights])
    nn_gap_final = np.array([float((series_nn[w][-1] - baseline_nn[-1]).item()) for w in weights])
    width = 0.38

    axes[0, 1].axhline(0.0, color="black", linewidth=1)
    axes[0, 1].bar(x - width / 2, exact_gap_avg, width=width, label="Exact", color="tab:blue")
    axes[0, 1].bar(x + width / 2, nn_gap_avg, width=width, label="NN", color="tab:orange")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(tick_labels)
    axes[0, 1].set_xlabel("interaction weight")
    axes[0, 1].set_ylabel("affinity gap")
    axes[0, 1].grid(True, axis="y", alpha=0.3)
    axes[0, 1].legend()

    axes[1, 1].axhline(0.0, color="black", linewidth=1)
    axes[1, 1].bar(x - width / 2, exact_gap_final, width=width, label="Exact", color="tab:blue")
    axes[1, 1].bar(x + width / 2, nn_gap_final, width=width, label="NN", color="tab:orange")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(tick_labels)
    axes[1, 1].set_xlabel("interaction weight")
    axes[1, 1].set_ylabel("affinity gap")
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"V-neck gaussian_conv_mean: exact vs NN by interaction strength (seed={seed}, N=500)",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_single_timeseries(
    output_path: Path,
    *,
    title: str,
    time_grid: torch.Tensor,
    series: Dict[float, list[torch.Tensor]],
) -> None:
    weights = sorted(series)
    cmap = plt.get_cmap("viridis")
    colors = {w: cmap(i / max(1, len(weights) - 1)) for i, w in enumerate(weights)}

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for w in weights:
        label = "baseline" if math.isclose(w, 0.0) else f"$w={w:g}$"
        stacked = torch.stack(series[w], dim=0)
        mean = stacked.mean(dim=0)
        ax.plot(time_grid.numpy(), mean.numpy(), label=label, color=colors[w], linewidth=2)
        if stacked.shape[0] > 1:
            std = stacked.std(dim=0, unbiased=False)
            ax.fill_between(
                time_grid.numpy(),
                (mean - std).numpy(),
                (mean + std).numpy(),
                color=colors[w],
                alpha=0.15,
                linewidth=0,
            )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$A_k(t)$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_gap_bar(
    output_path: Path,
    *,
    title: str,
    weights: list[float],
    exact_values: np.ndarray,
    nn_values: np.ndarray,
) -> None:
    x = np.arange(len(weights))
    width = 0.38
    tick_labels = ["0" if math.isclose(w, 0.0) else f"{w:g}" for w in weights]

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.bar(x - width / 2, exact_values, width=width, label="Exact", color="tab:blue")
    ax.bar(x + width / 2, nn_values, width=width, label="NN", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("interaction weight")
    ax.set_ylabel("affinity gap")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_csv(
    output_path: Path,
    *,
    series_exact: Dict[float, list[torch.Tensor]],
    series_nn: Dict[float, list[torch.Tensor]],
    time_grid: torch.Tensor,
    exact_seed_series: Dict[int, Dict[float, torch.Tensor]],
    nn_seed_series: Dict[int, Dict[float, torch.Tensor]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["backend", "seed", "weight", "time", "mean_affinity"])
        for backend, seed_series in [("exact", exact_seed_series), ("nn", nn_seed_series)]:
            for seed in sorted(seed_series):
                for w in sorted(seed_series[seed]):
                    for t, a in zip(time_grid.tolist(), seed_series[seed][w].tolist()):
                        writer.writerow([backend, seed, w, float(t), float(a)])


def _write_summary_csv(
    output_path: Path,
    *,
    series_exact: Dict[float, list[torch.Tensor]],
    series_nn: Dict[float, list[torch.Tensor]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "weight",
            "exact_n_seeds",
            "nn_n_seeds",
            "exact_time_average_mean",
            "exact_time_average_std",
            "nn_time_average_mean",
            "nn_time_average_std",
            "exact_final_mean",
            "exact_final_std",
            "nn_final_mean",
            "nn_final_std",
        ])
        for w in sorted(set(series_exact) & set(series_nn)):
            exact_stack = torch.stack(series_exact[w], dim=0)
            nn_stack = torch.stack(series_nn[w], dim=0)
            exact_time_avg = exact_stack.mean(dim=1)
            nn_time_avg = nn_stack.mean(dim=1)
            exact_final = exact_stack[:, -1]
            nn_final = nn_stack[:, -1]
            writer.writerow([
                w,
                exact_stack.shape[0],
                nn_stack.shape[0],
                float(exact_time_avg.mean().item()),
                float(exact_time_avg.std(unbiased=False).item()) if exact_stack.shape[0] > 1 else 0.0,
                float(nn_time_avg.mean().item()),
                float(nn_time_avg.std(unbiased=False).item()) if nn_stack.shape[0] > 1 else 0.0,
                float(exact_final.mean().item()),
                float(exact_final.std(unbiased=False).item()) if exact_stack.shape[0] > 1 else 0.0,
                float(nn_final.mean().item()),
                float(nn_final.std(unbiased=False).item()) if nn_stack.shape[0] > 1 else 0.0,
            ])


def main() -> None:
    args = parse_args()
    weights = [float(w) for w in args.weights]
    seeds = args.seeds if args.seeds is not None and len(args.seeds) > 0 else [args.seed]
    print("Seeds:", seeds)
    time_grid = None
    series_exact: Dict[float, list[torch.Tensor]] = {}
    series_nn: Dict[float, list[torch.Tensor]] = {}
    exact_seed_series: Dict[int, Dict[float, torch.Tensor]] = {}
    nn_seed_series: Dict[int, Dict[float, torch.Tensor]] = {}

    for seed in seeds:
        exact_runs = _discover_root(args.exact_root, "exact", seed, args.n_agents, weights)
        nn_runs = _discover_root(args.nn_root, "nn", seed, args.n_agents, weights)
        shared_weights = sorted(set(exact_runs) & set(nn_runs))
        if not shared_weights:
            print(f"Skipping seed {seed}: no shared weights found.")
            continue
        print(f"Seed {seed} shared weights: {shared_weights}")
        exact_seed_series[seed] = {}
        nn_seed_series[seed] = {}

        for w in shared_weights:
            print(f"Sampling exact seed={seed} weight={w:g} from {exact_runs[w].exp_dir.name}")
            tg, xs = sample_forward_trajectories(
                exact_runs[w].exp_dir,
                checkpoint_arg=args.checkpoint,
                num_trajectories=args.num_trajectories,
                device_override=args.device,
            )
            time_grid = tg
            exact_series = gaussian_kernel_affinity_series(xs, sigma=exact_runs[w].sigma)
            exact_seed_series[seed][w] = exact_series
            series_exact.setdefault(w, []).append(exact_series)

            print(f"Sampling nn seed={seed} weight={w:g} from {nn_runs[w].exp_dir.name}")
            tg, xs = sample_forward_trajectories(
                nn_runs[w].exp_dir,
                checkpoint_arg=args.checkpoint,
                num_trajectories=args.num_trajectories,
                device_override=args.device,
            )
            time_grid = tg
            nn_series = gaussian_kernel_affinity_series(xs, sigma=nn_runs[w].sigma)
            nn_seed_series[seed][w] = nn_series
            series_nn.setdefault(w, []).append(nn_series)

    assert time_grid is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = sorted(set(series_exact) & set(series_nn))
    _plot_single_timeseries(
        args.output_dir / f"{args.prefix}_exact_timeseries.png",
        title="Exact",
        time_grid=time_grid,
        series=series_exact,
    )
    _plot_single_timeseries(
        args.output_dir / f"{args.prefix}_nn_timeseries.png",
        title="NN",
        time_grid=time_grid,
        series=series_nn,
    )
    _write_csv(
        args.output_dir / f"{args.prefix}_timeseries.csv",
        series_exact=series_exact,
        series_nn=series_nn,
        time_grid=time_grid,
        exact_seed_series=exact_seed_series,
        nn_seed_series=nn_seed_series,
    )
    _write_summary_csv(
        args.output_dir / f"{args.prefix}_summary.csv",
        series_exact=series_exact,
        series_nn=series_nn,
    )


if __name__ == "__main__":
    main()
