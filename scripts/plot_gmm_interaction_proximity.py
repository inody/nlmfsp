#!/usr/bin/env python3
"""
Compute and visualize a Gaussian-kernel proximity metric for GMM interaction sweeps.

For the Gaussian mean-field drift

    f(x) = w E[exp(-||x-X||^2 / (2 sigma^2)) (X - x)],

define the scalar proximity score

    A(t) = (1 / (N (N-1))) sum_{i != j} exp(-||x_i(t)-x_j(t)||^2 / (2 sigma^2)).

This is the natural kernel overlap associated with the interaction term. The
drift is proportional to the spatial gradient of this kernel potential, so
larger A(t) means particles are more concentrated on the interaction scale.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import torch

from plot_experiment_trajectories import (
    _load_checkpoint_interaction_modules,
    load_experiment_config,
    resolve_checkpoint,
)

# Ensure local package imports work when this script is run directly.
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlmfsp.cli_helpers import resolve_device
from nlmfsp.fbsde import FBSDESolver
from nlmfsp.interaction import create_interactions
from nlmfsp.plot_style import apply_paper_style
from nlmfsp.problem_setup import setup_problem_artifacts
from nlmfsp.rng_utils import seed_offset, torch_seed_context

apply_paper_style()


DEFAULT_WEIGHTS = (0.0, 1.0, 2.0, 4.0, 8.0)


@dataclass(frozen=True)
class SweepRun:
    seed: int
    weight: float
    sigma: float
    exp_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Gaussian-kernel proximity for GMM interaction sweeps."
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("outputs/experiments/analytical_gmm_navigation"),
        help="Directory containing analytical GMM runs.",
    )
    parser.add_argument(
        "--problem-config",
        type=str,
        default="configs/problems/gmm_navigation_gaussian.json",
        help="Problem config path stored in cli_args.problem_config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/comparison_plots/gmm_interaction_proximity"),
        help="Directory to save plots and CSV summaries.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit seed list. Default: auto-discover complete seeds.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="*",
        default=list(DEFAULT_WEIGHTS),
        help="Weights to include. Use 0 for the no-interaction baseline.",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=1000,
        help="Number of forward trajectories to sample per run.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint override; default uses each run's latest checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Sampling device override (cpu, cuda, mps, auto).",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Include seeds even if some requested weights are missing.",
    )
    return parser.parse_args()


def _weight_key(weight: float) -> str:
    if math.isclose(weight, round(weight), rel_tol=0.0, abs_tol=1e-9):
        return str(int(round(weight)))
    return f"{weight:g}"


def _ordered_weight_map(weights: Iterable[float]) -> list[tuple[str, float]]:
    return [(f"weight_{_weight_key(weight)}", float(weight)) for weight in weights]


def discover_runs(
    experiments_root: Path,
    problem_config: str,
    weights: Iterable[float],
) -> Dict[int, Dict[str, SweepRun]]:
    expected = {f"weight_{_weight_key(weight)}": float(weight) for weight in weights}
    grouped: Dict[int, Dict[str, SweepRun]] = {}

    for exp_dir in sorted(experiments_root.glob("exp_*")):
        config_path = exp_dir / "config.json"
        if not config_path.is_file():
            continue

        payload = json.loads(config_path.read_text())
        cli_args = payload.get("cli_args", {})
        if cli_args.get("problem_config") != problem_config:
            continue
        if cli_args.get("cost_type") != "zero":
            continue

        seed = cli_args.get("seed")
        if seed is None:
            continue

        drift_type = cli_args.get("drift_type")
        sigma = float(cli_args.get("int_drift_sigma", 2.0))
        if drift_type == "zero":
            label = "weight_0"
            weight = 0.0
        elif drift_type == "gaussian":
            raw_weight = cli_args.get("int_drift_weight")
            if raw_weight is None:
                continue
            weight = float(raw_weight)
            label = f"weight_{_weight_key(weight)}"
        else:
            continue

        if label not in expected:
            continue

        grouped.setdefault(int(seed), {})[label] = SweepRun(
            seed=int(seed),
            weight=weight,
            sigma=sigma,
            exp_dir=exp_dir,
        )

    return grouped


def _resolve_device(device_choice: Optional[str], default_choice: str) -> torch.device:
    device_str = resolve_device(device_choice or default_choice, print_info=False)
    if device_str == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA device but none is available; falling back to CPU.")
        device_str = "cpu"
    elif device_str == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not has_mps:
            print("Requested MPS device but it is unavailable; falling back to CPU.")
            device_str = "cpu"
    return torch.device(device_str)


def sample_forward_trajectories(
    run: SweepRun,
    *,
    checkpoint_arg: Optional[str],
    num_trajectories: int,
    device_override: Optional[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    config, cli_args, _ = load_experiment_config(run.exp_dir)
    checkpoint_path = resolve_checkpoint(run.exp_dir, checkpoint_arg)

    cli_problem_config = getattr(cli_args, "problem_config", None)
    if cli_problem_config:
        candidate = (PROJECT_ROOT / cli_problem_config).resolve()
        if candidate.is_file():
            config.problem_config_path = str(candidate)

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


def gaussian_kernel_affinity_series(
    xs: torch.Tensor,
    *,
    sigma: float,
    query_chunk_size: int = 200,
    ref_chunk_size: int = 200,
) -> torch.Tensor:
    n_steps, n_agents, _ = xs.shape
    if n_agents < 2:
        raise ValueError("Need at least two agents to compute pairwise affinity.")

    denom = float(n_agents * (n_agents - 1))
    out = torch.zeros(n_steps, dtype=torch.float64)

    for t in range(n_steps):
        x_t = xs[t]
        total = 0.0
        for q_start in range(0, n_agents, query_chunk_size):
            q_end = min(q_start + query_chunk_size, n_agents)
            xq = x_t[q_start:q_end]
            for r_start in range(0, n_agents, ref_chunk_size):
                r_end = min(r_start + ref_chunk_size, n_agents)
                xr = x_t[r_start:r_end]
                diff = xr.unsqueeze(0) - xq.unsqueeze(1)
                dist_sq = (diff**2).sum(dim=2)
                kernel = torch.exp(-dist_sq / (2.0 * sigma**2))

                overlap_start = max(q_start, r_start)
                overlap_end = min(q_end, r_end)
                if overlap_start < overlap_end:
                    diag_q = torch.arange(
                        overlap_start - q_start,
                        overlap_end - q_start,
                        dtype=torch.long,
                    )
                    diag_r = torch.arange(
                        overlap_start - r_start,
                        overlap_end - r_start,
                        dtype=torch.long,
                    )
                    kernel[diag_q, diag_r] = 0.0

                total += float(kernel.sum().item())

        out[t] = total / denom

    return out


def _plot_seed_timeseries(
    seed: int,
    time_grid: torch.Tensor,
    per_weight: Dict[float, torch.Tensor],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for weight in sorted(per_weight):
        series = per_weight[weight]
        label = "baseline" if math.isclose(weight, 0.0) else rf"$w={weight:g}$"
        ax.plot(time_grid.numpy(), series.numpy(), linewidth=2, label=label)

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$A(t)$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_aggregate_summary(
    time_grid: torch.Tensor,
    seed_series: Dict[int, Dict[float, torch.Tensor]],
    output_dir: Path,
) -> None:
    weights = sorted({weight for per_seed in seed_series.values() for weight in per_seed})
    fig, (ax_time, ax_bar) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.7, 1.0]}
    )

    summary_rows = []
    for weight in weights:
        stacked = torch.stack(
            [seed_series[seed][weight] for seed in sorted(seed_series) if weight in seed_series[seed]],
            dim=0,
        )
        mean_series = stacked.mean(dim=0)
        std_series = stacked.std(dim=0, unbiased=False) if stacked.shape[0] > 1 else None
        label = "baseline" if math.isclose(weight, 0.0) else rf"$w={weight:g}$"
        ax_time.plot(time_grid.numpy(), mean_series.numpy(), linewidth=2, label=label)
        if std_series is not None:
            ax_time.fill_between(
                time_grid.numpy(),
                (mean_series - std_series).numpy(),
                (mean_series + std_series).numpy(),
                alpha=0.15,
            )

        time_avg = mean_series.mean().item()
        final_val = mean_series[-1].item()
        if math.isclose(weight, 0.0):
            excess_values = [0.0 for _seed in seed_series]
        else:
            excess_values = []
            for seed in sorted(seed_series):
                baseline = seed_series[seed].get(0.0)
                current = seed_series[seed].get(weight)
                if baseline is None or current is None:
                    continue
                excess_values.append(float((current - baseline).mean().item()))
        excess_mean = sum(excess_values) / len(excess_values)
        if len(excess_values) > 1:
            variance = sum((value - excess_mean) ** 2 for value in excess_values) / len(excess_values)
            excess_std = math.sqrt(variance)
        else:
            excess_std = 0.0
        summary_rows.append((weight, time_avg, final_val, excess_mean, excess_std))

    ax_time.set_title("Affinity")
    ax_time.set_xlabel(r"$t$")
    ax_time.set_ylabel(r"$A(t)$")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend()

    bar_positions = list(range(len(summary_rows)))
    ax_bar.bar(
        bar_positions,
        [row[3] for row in summary_rows],
        color="#4C72B0",
        alpha=0.85,
        yerr=[row[4] for row in summary_rows],
        capsize=4,
    )
    ax_bar.set_xticks(bar_positions)
    ax_bar.set_xticklabels(
        ["baseline" if math.isclose(row[0], 0.0) else f"w={row[0]:g}" for row in summary_rows],
        rotation=20,
    )
    ax_bar.set_title("Affinity gap")
    ax_bar.set_ylabel("Time-averaged gap")
    ax_bar.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "gmm_interaction_proximity_summary.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "gmm_interaction_proximity_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_csv(
    output_path: Path,
    *,
    time_grid: torch.Tensor,
    seed_series: Dict[int, Dict[float, torch.Tensor]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "weight", "time", "mean_affinity"])
        for seed in sorted(seed_series):
            for weight in sorted(seed_series[seed]):
                series = seed_series[seed][weight]
                for time_value, affinity in zip(time_grid.tolist(), series.tolist()):
                    writer.writerow([seed, weight, time_value, affinity])


def _write_summary_csv(
    output_path: Path,
    *,
    seed_series: Dict[int, Dict[float, torch.Tensor]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed",
                "weight",
                "time_average",
                "final_value",
                "peak_value",
                "excess_time_average_vs_baseline",
                "excess_final_value_vs_baseline",
            ]
        )
        for seed in sorted(seed_series):
            baseline = seed_series[seed].get(0.0)
            for weight in sorted(seed_series[seed]):
                series = seed_series[seed][weight]
                if baseline is None:
                    excess_time_avg = 0.0
                    excess_final = 0.0
                else:
                    excess_time_avg = float((series - baseline).mean().item())
                    excess_final = float((series[-1] - baseline[-1]).item())
                writer.writerow(
                    [
                        seed,
                        weight,
                        float(series.mean().item()),
                        float(series[-1].item()),
                        float(series.max().item()),
                        excess_time_avg,
                        excess_final,
                    ]
                )


def main() -> None:
    args = parse_args()
    grouped = discover_runs(args.experiments_root, args.problem_config, args.weights)
    ordered_weights = _ordered_weight_map(args.weights)

    requested_seeds = set(args.seeds) if args.seeds else None
    selected_seeds: list[int] = []
    for seed in sorted(grouped):
        if requested_seeds is not None and seed not in requested_seeds:
            continue
        if not args.allow_incomplete and any(
            label not in grouped[seed] for label, _weight in ordered_weights
        ):
            continue
        selected_seeds.append(seed)

    if not selected_seeds:
        raise SystemExit("No matching seeds found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_series: Dict[int, Dict[float, torch.Tensor]] = {}
    reference_time_grid: Optional[torch.Tensor] = None

    for seed in selected_seeds:
        print(f"Processing seed {seed}")
        per_weight: Dict[float, torch.Tensor] = {}
        for label, weight in ordered_weights:
            run = grouped[seed].get(label)
            if run is None:
                continue
            print(f"  sampling weight={weight:g} from {run.exp_dir.name}")
            time_grid, xs_forward = sample_forward_trajectories(
                run,
                checkpoint_arg=args.checkpoint,
                num_trajectories=args.num_trajectories,
                device_override=args.device,
            )
            series = gaussian_kernel_affinity_series(
                xs_forward,
                sigma=run.sigma,
                query_chunk_size=200,
                ref_chunk_size=200,
            )
            per_weight[weight] = series
            if reference_time_grid is None:
                reference_time_grid = time_grid

        if not per_weight:
            continue

        seed_series[seed] = per_weight
        seed_dir = args.output_dir / f"seed_{seed}"
        _plot_seed_timeseries(
            seed=seed,
            time_grid=reference_time_grid,
            per_weight=per_weight,
            output_path=seed_dir / f"gmm_interaction_proximity_seed_{seed}.png",
        )

    if not seed_series or reference_time_grid is None:
        raise SystemExit("No proximity series were computed.")

    _plot_aggregate_summary(reference_time_grid, seed_series, args.output_dir)
    _write_csv(
        args.output_dir / "gmm_interaction_proximity_timeseries.csv",
        time_grid=reference_time_grid,
        seed_series=seed_series,
    )
    _write_summary_csv(
        args.output_dir / "gmm_interaction_proximity_summary.csv",
        seed_series=seed_series,
    )

    print(
        f"Saved plots and CSV summaries to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
