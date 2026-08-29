#!/usr/bin/env python3
"""Plot GMM affinity responses across interaction strengths for exact and NN runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch

from plot_experiment_trajectories import (
    _load_checkpoint_interaction_modules,
    load_experiment_config,
    resolve_checkpoint,
)
from plot_gmm_interaction_proximity import _resolve_device, gaussian_kernel_affinity_series

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


DEFAULT_WEIGHTS = (0.0, 1.0, 2.0, 4.0, 8.0)


@dataclass(frozen=True)
class RunSpec:
    backend: str
    seed: int
    weight: float
    sigma: float
    exp_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Legacy shorthand: use the same seed for exact and NN unless overridden.",
    )
    parser.add_argument(
        "--exact-seed",
        type=int,
        default=200,
        help="Seed for exact runs.",
    )
    parser.add_argument(
        "--nn-seed",
        type=int,
        default=201,
        help="Seed for NN runs.",
    )
    parser.add_argument(
        "--exact-root",
        type=Path,
        default=Path("outputs/experiments/analytical_gmm_navigation"),
    )
    parser.add_argument(
        "--nn-root",
        type=Path,
        default=Path("outputs/experiments/gmm_interaction_strength_nn"),
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
    parser.add_argument(
        "--min-best-iteration",
        type=int,
        default=75,
        help=(
            "Minimum numbered checkpoint iteration considered when "
            "--checkpoint=best-analytical."
        ),
    )
    parser.add_argument("--num-trajectories", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("latex/R05/figs/gmm_interaction_strength_compare"),
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="gmm_interaction_strength",
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
    drift_type = config_merged.get("drift_type")
    if drift_type == "zero":
        return 0.0
    if drift_type == "gaussian":
        weight = config_merged.get("int_drift_weight")
        return None if weight is None else float(weight)
    return None


def _discover_root(
    root: Path,
    backend: str,
    seed: int,
    weights: list[float],
) -> Dict[float, RunSpec]:
    runs: Dict[float, RunSpec] = {}
    for exp in sorted(root.glob("exp_*")):
        cfg = exp / "config.json"
        if not cfg.exists():
            continue
        if not (exp / "metrics.json").exists():
            continue
        data = _load_json(cfg)
        merged = data.get("config_merged", data.get("config", {}))
        run_seed = int(merged.get("seed", -1))
        if run_seed != int(seed):
            continue
        if merged.get("cost_type") != "zero":
            continue
        weight = _run_weight(merged)
        if weight is None or weight not in weights:
            continue
        sigma_raw = merged.get("int_drift_sigma")
        sigma = 2.0 if sigma_raw is None else float(sigma_raw)
        score = ((exp / "metrics.json").exists(), exp.name)
        current = runs.get(weight)
        if current is None:
            runs[weight] = RunSpec(
                backend=backend,
                seed=run_seed,
                weight=weight,
                sigma=sigma,
                exp_dir=exp,
            )
        else:
            current_score = ((current.exp_dir / "metrics.json").exists(), current.exp_dir.name)
            if score > current_score:
                runs[weight] = RunSpec(
                    backend=backend,
                    seed=run_seed,
                    weight=weight,
                    sigma=sigma,
                    exp_dir=exp,
                )
    return runs


def _checkpoint_iteration(path: Path) -> int | None:
    suffix = path.stem.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def _best_analytical_checkpoint(exp_dir: Path, *, min_iteration: int = 1) -> Path:
    numbered = sorted((exp_dir / "checkpoints").glob("checkpoint_[0-9][0-9][0-9][0-9].pt"))
    candidates = [
        path
        for path in numbered
        if (_checkpoint_iteration(path) or 0) >= min_iteration
    ]
    if not candidates:
        raise ValueError(
            f"No numbered checkpoints with iteration >= {min_iteration} in {exp_dir}"
        )

    analytical_keys = [
        "loss_ipf_forward_analytical",
        "loss_td_forward_analytical",
        "loss_fk_forward_analytical",
        "loss_ipf_backward_analytical",
        "loss_td_backward_analytical",
        "loss_fk_backward_analytical",
    ]
    best_path: Path | None = None
    best_total = float("inf")
    for checkpoint_path in candidates:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metrics_analytical = checkpoint.get("metrics_analytical", {})
        total = sum(float(metrics_analytical.get(key, 0.0)) for key in analytical_keys)
        if total < best_total:
            best_total = total
            best_path = checkpoint_path

    if best_path is None:
        raise ValueError(f"Could not select a best analytical checkpoint in {exp_dir}")
    return best_path


def sample_forward_trajectories(
    exp_dir: Path,
    *,
    checkpoint_arg: Optional[str],
    min_best_iteration: int,
    num_trajectories: int,
    device_override: Optional[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    config, cli_args, _ = load_experiment_config(exp_dir)
    if checkpoint_arg == "best-analytical":
        checkpoint_path = _best_analytical_checkpoint(
            exp_dir,
            min_iteration=min_best_iteration,
        )
    else:
        checkpoint_path = resolve_checkpoint(exp_dir, checkpoint_arg)

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


def _plot_panel(
    output_path: Path,
    *,
    title: str,
    series_by_weight: Dict[float, torch.Tensor],
    time_grid: torch.Tensor,
) -> list[dict[str, float | int | str]]:
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    cmap = plt.get_cmap("viridis")
    weights = sorted(series_by_weight)
    colors = {w: cmap(i / max(1, len(weights) - 1)) for i, w in enumerate(weights)}
    rows: list[dict[str, float | int | str]] = []
    for w in weights:
        series = series_by_weight[w]
        label = "baseline" if math.isclose(w, 0.0) else rf"$w={w:g}$"
        ax.plot(time_grid.numpy(), series.numpy(), label=label, color=colors[w], linewidth=2)
        rows.append(
            {
                "weight": w,
                "A0": float(series[0]),
                "AT": float(series[-1]),
                "Amax": float(series.max()),
            }
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
    return rows


def _write_summary(
    output_path: Path,
    *,
    exact_rows: list[dict[str, float | int | str]],
    nn_rows: list[dict[str, float | int | str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["backend", "weight", "A0", "AT", "Amax"],
        )
        writer.writeheader()
        for backend, rows in (("exact", exact_rows), ("nn", nn_rows)):
            for row in rows:
                writer.writerow({"backend": backend, **row})


def main() -> None:
    args = parse_args()
    weights = sorted(set(float(w) for w in args.weights))
    exact_seed = int(args.seed if args.seed is not None else args.exact_seed)
    nn_seed = int(args.seed if args.seed is not None else args.nn_seed)

    exact_runs = _discover_root(args.exact_root, "exact", exact_seed, weights)
    nn_runs = _discover_root(args.nn_root, "nn", nn_seed, weights)

    if not exact_runs and not nn_runs:
        raise SystemExit("No matching runs found.")

    exact_rows: list[dict[str, float | int | str]] = []
    nn_rows: list[dict[str, float | int | str]] = []
    if exact_runs:
        exact_series: Dict[float, torch.Tensor] = {}
        time_grid_ref = None
        for weight, run in sorted(exact_runs.items()):
            time_grid, xs_forward = sample_forward_trajectories(
                run.exp_dir,
                checkpoint_arg=args.checkpoint,
                min_best_iteration=args.min_best_iteration,
                num_trajectories=args.num_trajectories,
                device_override=args.device,
            )
            exact_series[weight] = gaussian_kernel_affinity_series(xs_forward, sigma=run.sigma)
            time_grid_ref = time_grid
        exact_rows = _plot_panel(
            args.output_dir / f"{args.prefix}_exact_timeseries.png",
            title="Exact",
            series_by_weight=exact_series,
            time_grid=time_grid_ref,
        )

    if nn_runs:
        nn_series: Dict[float, torch.Tensor] = {}
        time_grid_ref = None
        for weight, run in sorted(nn_runs.items()):
            time_grid, xs_forward = sample_forward_trajectories(
                run.exp_dir,
                checkpoint_arg=args.checkpoint,
                min_best_iteration=args.min_best_iteration,
                num_trajectories=args.num_trajectories,
                device_override=args.device,
            )
            nn_series[weight] = gaussian_kernel_affinity_series(xs_forward, sigma=run.sigma)
            time_grid_ref = time_grid
        nn_rows = _plot_panel(
            args.output_dir / f"{args.prefix}_nn_timeseries.png",
            title="NN",
            series_by_weight=nn_series,
            time_grid=time_grid_ref,
        )
    _write_summary(
        args.output_dir / f"{args.prefix}_summary.csv",
        exact_rows=exact_rows,
        nn_rows=nn_rows,
    )


if __name__ == "__main__":
    main()
