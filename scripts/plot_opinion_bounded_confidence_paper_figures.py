#!/usr/bin/env python3
"""Generate R05 figures for the smooth bounded-confidence opinion experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nlmfsp.cli_helpers import resolve_device
from nlmfsp.fbsde import FBSDESolver
from nlmfsp.interaction import create_interactions
from nlmfsp.plot_style import apply_paper_style
from nlmfsp.problem_setup import setup_problem_artifacts
from nlmfsp.rng_utils import seed_offset, torch_seed_context

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_experiment_trajectories import (  # noqa: E402
    _load_checkpoint_interaction_modules,
    load_experiment_config,
    resolve_checkpoint,
)


ANALYTICAL_DIR = Path(
    "outputs/experiments/"
    "analytical_opinion_1k_bounded_confidence_cost_off_n400_dt001_sigma1_clip1/"
    "exp_20260524_163204"
)
LEARNED_DIR = Path(
    "outputs/experiments/"
    "learned_opinion_1k_bounded_confidence_cost_off_n400_dt001_sigma1_int100_clip1_detachdiv/"
    "exp_20260524_192349"
)
RB_DIR = Path(
    "outputs/experiments/"
    "rb100_opinion_1k_bounded_confidence_cost_off_n400_dt001_sigma1_clip1/"
    "exp_20260524_163226"
)
OUTPUT_DIR = Path("latex/R05/figs/opinion_bounded_confidence_n400")
PCA_AXIS_LIMIT = 6.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytical-dir", type=Path, default=ANALYTICAL_DIR)
    parser.add_argument("--learned-dir", type=Path, default=LEARNED_DIR)
    parser.add_argument("--rb-dir", type=Path, default=RB_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--num-trajectories", type=int, default=320)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--learned-max-iteration",
        type=int,
        default=None,
        help="If set, truncate the learned run to this outer iteration for plots and summaries.",
    )
    return parser.parse_args()


def latest_checkpoint(exp_dir: Path) -> Path:
    checkpoints = sorted(exp_dir.glob("checkpoints/checkpoint_*.pt"))
    numbered = [
        p
        for p in checkpoints
        if p.stem.rsplit("_", 1)[-1].isdigit()
    ]
    if not numbered:
        raise FileNotFoundError(f"No numbered checkpoints found in {exp_dir}")
    return numbered[-1]


def load_checkpoint_metrics(exp_dir: Path) -> tuple[dict[str, Any], dict[str, float]]:
    checkpoint = torch.load(latest_checkpoint(exp_dir), map_location="cpu", weights_only=False)
    metrics = checkpoint["metrics"]
    config = checkpoint["config"]
    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    lambdas = {
        "lambda_ipf": float(config_dict.get("lambda_ipf", 1.0)),
        "lambda_td": float(config_dict.get("lambda_td", 1.0)),
        "lambda_fk": float(config_dict.get("lambda_fk", 1.0)),
        "lambda_boundary": float(config_dict.get("lambda_boundary", 0.0)),
    }
    return metrics, lambdas


def weighted_total(metrics: dict[str, Any], lambdas: dict[str, float], suffix: str) -> np.ndarray:
    keys = [
        f"loss_ipf_{suffix}",
        f"loss_td_{suffix}",
        f"loss_fk_{suffix}",
        f"loss_boundary_{suffix}",
    ]
    length = min(len(metrics.get(k, [])) for k in keys)
    out = np.empty(length, dtype=float)
    for i in range(length):
        out[i] = (
            lambdas["lambda_ipf"] * metrics[keys[0]][i]
            + lambdas["lambda_td"] * metrics[keys[1]][i]
            + lambdas["lambda_fk"] * metrics[keys[2]][i]
            + lambdas["lambda_boundary"] * metrics[keys[3]][i]
        )
    return out


def analytical_total_series(exp_dir: Path, max_iteration: int | None = None) -> dict[str, np.ndarray]:
    metrics, lambdas = load_checkpoint_metrics(exp_dir)
    forward = weighted_total(metrics, lambdas, "forward_analytical")
    backward = weighted_total(metrics, lambdas, "backward_analytical")
    length = min(len(forward), len(backward), len(metrics["time_per_iteration"]))
    if max_iteration is not None:
        length = min(length, max_iteration)
    return {
        "iteration": np.arange(1, length + 1),
        "time": np.cumsum(np.asarray(metrics["time_per_iteration"][:length], dtype=float)),
        "total": forward[:length] + backward[:length],
    }


def best_iteration(exp_dir: Path, max_iteration: int | None = None) -> int:
    series = analytical_total_series(exp_dir, max_iteration=max_iteration)
    return int(series["iteration"][np.argmin(series["total"])])


def load_solver_and_sample(
    exp_dir: Path,
    iteration: int,
    *,
    n_samples: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    config, cli_args, _ = load_experiment_config(exp_dir)
    config.device = device
    cli_args.device = str(device)
    checkpoint_path = resolve_checkpoint(exp_dir, str(iteration))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    (
        _mu_0,
        _mu_1,
        sample_rho0,
        sample_rho1,
        *_rest,
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

    state = checkpoint["solver_state"]
    solver.value_net.load_state_dict(state["value_net"])
    solver.value_hat_net.load_state_dict(state["value_hat_net"])
    solver.grad_net.load_state_dict(state["grad_net"])
    solver.grad_hat_net.load_state_dict(state["grad_hat_net"])
    _load_checkpoint_interaction_modules(solver, state, config)

    with torch_seed_context(seed_offset(seed, 0)):
        x0 = sample_rho0(n_samples)
    with torch_seed_context(seed_offset(seed, 1)):
        x1 = sample_rho1(n_samples)

    with torch.no_grad():
        xs_forward, _, _ = solver.sample_forward_trajectory(
            x0, seed=seed_offset(seed, 2)
        )
        xs_backward, _, _ = solver.sample_backward_trajectory(
            x1, seed=seed_offset(seed, 3)
        )
    return xs_forward.cpu(), xs_backward.cpu(), config.time_grid.cpu()


def project_joint(
    trajectories: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    n_components: int = 2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    terminal_forward = torch.cat([xs_f[-1] for xs_f, _ in trajectories], dim=0)
    mean = terminal_forward.mean(dim=0, keepdim=True)
    _, _, vt = torch.linalg.svd(terminal_forward - mean, full_matrices=False)
    basis = vt[:n_components].T

    projected = []
    for xs_f, xs_b in trajectories:
        shape_f = xs_f.shape[:-1]
        shape_b = xs_b.shape[:-1]
        proj_f = ((xs_f.reshape(-1, xs_f.shape[-1]) - mean) @ basis).reshape(
            *shape_f, n_components
        )
        proj_b = ((xs_b.reshape(-1, xs_b.shape[-1]) - mean) @ basis).reshape(
            *shape_b, n_components
        )
        projected.append((proj_f.numpy(), proj_b.numpy()))
    return projected


def plot_snapshot_comparison(
    analytical: tuple[torch.Tensor, torch.Tensor],
    learned: tuple[torch.Tensor, torch.Tensor],
    time_grid: torch.Tensor,
    output_dir: Path,
) -> None:
    projected = project_joint([analytical, learned])
    steps = np.linspace(0, analytical[0].shape[0] - 1, 5).astype(int)
    titles = [
        r"$t=0$",
        r"$t=0.25T$",
        r"$t=0.50T$",
        r"$t=0.75T$",
        r"$t=T$",
    ]
    rows = [
        ("Analytical forward", projected[0][0], "#1f77b4"),
        ("Analytical backward", projected[0][1], "#ff7f0e"),
        ("Learned forward", projected[1][0], "#1f77b4"),
        ("Learned backward", projected[1][1], "#ff7f0e"),
    ]
    xlim = (-PCA_AXIS_LIMIT, PCA_AXIS_LIMIT)
    ylim = (-PCA_AXIS_LIMIT, PCA_AXIS_LIMIT)

    fig, axes = plt.subplots(
        len(rows), len(steps), figsize=(13.0, 9.5), constrained_layout=True
    )
    for row_idx, (row_label, arr, color) in enumerate(rows):
        for col_idx, step in enumerate(steps):
            ax = axes[row_idx, col_idx]
            ax.scatter(
                arr[step, :, 0],
                arr[step, :, 1],
                s=8.0,
                alpha=0.48,
                color=color,
                linewidths=0,
                rasterized=True,
            )
            if row_idx == 0:
                ax.set_title(titles[col_idx], fontweight="normal")
            if col_idx == 0:
                ax.set_ylabel(row_label)
            ax.set_aspect("equal")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.18, linewidth=0.6)
            ax.tick_params(axis="both", which="major")
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"snapshot_pca_comparison.{ext}", dpi=200)
    plt.close(fig)


def plot_total_loss_vs_time(
    analytical_dir: Path,
    learned_dir: Path,
    output_dir: Path,
    *,
    learned_max_iteration: int | None = None,
) -> None:
    palette = plt.get_cmap("tab10").colors
    series_a = analytical_total_series(analytical_dir)
    series_l = analytical_total_series(learned_dir, max_iteration=learned_max_iteration)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(
        series_a["time"],
        series_a["total"],
        "--",
        label="Analytic",
        color=palette[0],
    )
    ax.plot(
        series_l["time"],
        series_l["total"],
        "-",
        label="Learned",
        color=palette[1],
    )
    ax.set_yscale("log")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Total Loss (Forward + Backward)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"analytical_loss_vs_time.{ext}", dpi=200)
    plt.close(fig)


def plot_interaction_loss(
    learned_dir: Path,
    output_dir: Path,
    *,
    learned_max_iteration: int | None = None,
) -> None:
    metrics, _lambdas = load_checkpoint_metrics(learned_dir)
    length = len(metrics["loss_interaction_drift"])
    if learned_max_iteration is not None:
        length = min(length, learned_max_iteration)
    iterations = np.arange(1, length + 1)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(
        iterations,
        metrics["loss_interaction_drift"][:length],
        color=plt.get_cmap("tab10").colors[1],
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Interaction Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"interaction_loss_vs_iter.{ext}", dpi=200)
    plt.close(fig)


def write_summary(args: argparse.Namespace, output_dir: Path) -> None:
    learned_metrics = analytical_total_series(
        args.learned_dir,
        max_iteration=args.learned_max_iteration,
    )
    rb_metrics = analytical_total_series(args.rb_dir)
    with (args.learned_dir / "metrics.json").open() as f:
        learned_eval = json.load(f)
    with (args.rb_dir / "metrics.json").open() as f:
        rb_eval = json.load(f)

    def mean_drift_error(payload: dict[str, Any]) -> float:
        errors = payload["interaction_errors"]
        return float(
            np.mean(
                [
                    errors["forward_drift"]["relative_l2"],
                    errors["backward_drift"]["relative_l2"],
                ]
            )
        )

    payload = {
        "Learned surrogate": {
            "time_per_iter_sec": float(np.mean(np.diff(np.r_[0, learned_metrics["time"]]))),
            "best_analytical_loss": float(np.min(learned_metrics["total"])),
            "best_iter": int(learned_metrics["iteration"][np.argmin(learned_metrics["total"])]),
            "latest_iter": int(learned_metrics["iteration"][-1]),
            "interaction_error": mean_drift_error(learned_eval),
        },
        "Random batch": {
            "time_per_iter_sec": float(np.mean(np.diff(np.r_[0, rb_metrics["time"]]))),
            "best_analytical_loss": float(np.min(rb_metrics["total"])),
            "best_iter": int(rb_metrics["iteration"][np.argmin(rb_metrics["total"])]),
            "latest_iter": int(rb_metrics["iteration"][-1]),
            "interaction_error": mean_drift_error(rb_eval),
        },
    }
    with (output_dir / "summary_learned_rb.json").open("w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    apply_paper_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(resolve_device(args.device, print_info=False))

    analytical_best = best_iteration(args.analytical_dir)
    learned_best = best_iteration(
        args.learned_dir,
        max_iteration=args.learned_max_iteration,
    )
    xs_a_f, xs_a_b, time_grid = load_solver_and_sample(
        args.analytical_dir,
        analytical_best,
        n_samples=args.num_trajectories,
        seed=args.seed,
        device=device,
    )
    xs_l_f, xs_l_b, _ = load_solver_and_sample(
        args.learned_dir,
        learned_best,
        n_samples=args.num_trajectories,
        seed=args.seed,
        device=device,
    )

    plot_snapshot_comparison(
        (xs_a_f, xs_a_b),
        (xs_l_f, xs_l_b),
        time_grid,
        args.output_dir,
    )
    plot_total_loss_vs_time(
        args.analytical_dir,
        args.learned_dir,
        args.output_dir,
        learned_max_iteration=args.learned_max_iteration,
    )
    plot_interaction_loss(
        args.learned_dir,
        args.output_dir,
        learned_max_iteration=args.learned_max_iteration,
    )
    write_summary(args, args.output_dir)
    print(f"Saved figures to {args.output_dir}")
    print(f"Analytical trajectory checkpoint: {analytical_best}")
    print(f"Learned trajectory checkpoint: {learned_best}")


if __name__ == "__main__":
    main()
