#!/usr/bin/env python3
"""Plot training comparisons for Option A (analytic) vs Option B (learned).

The script loads the latest checkpoint from each experiment directory,
extracts analytic re-evaluated losses, computes derived totals, and produces
comparison plots for losses vs iteration/time and timing distributions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure local package modules are importable when deserialising checkpoints.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    src_path = str(SRC_ROOT)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

from nlmfsp.plot_style import apply_paper_style

apply_paper_style()

# Color palette (tab10): Analytic -> index 0, Learned -> index 1
PALETTE = plt.get_cmap("tab10").colors
COLOR_ANALYTIC = PALETTE[0]
COLOR_LEARNED = PALETTE[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analytic-dir",
        type=Path,
        required=True,
        help="Experiment directory for Option A (analytic interactions).",
    )
    parser.add_argument(
        "--learned-dir",
        type=Path,
        required=True,
        help="Experiment directory for Option B (learned interactions).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save comparison plots (default: ./comparison_plots).",
    )
    parser.add_argument(
        "--figure-format",
        type=str,
        default="png",
        help="Image format for saved figures (default: png).",
    )
    parser.add_argument(
        "--max-checkpoint",
        type=str,
        default=None,
        help=(
            "Optional maximum checkpoint iteration to load (inclusive). "
            "Accepts values like 60, 'checkpoint_0060', or 'checkpoint_0060.pt'."
        ),
    )
    args = parser.parse_args()
    args.max_checkpoint = _parse_max_checkpoint_arg(args.max_checkpoint)
    return args


def _parse_max_checkpoint_arg(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.lower() in {"final", "latest"}:
        return None
    match = re.search(r"(\d+)", stripped)
    if match:
        return int(match.group(1))
    raise ValueError(f"Invalid max checkpoint specification: {value}")


def _checkpoint_iteration(path: Path) -> Optional[int]:
    stem = path.stem
    if "_" in stem:
        suffix = stem.split("_", 1)[1]
    else:
        suffix = stem
    lowered = suffix.lower()
    if lowered in {"final", "latest"}:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            iteration = checkpoint.get("iteration")
            if isinstance(iteration, (int, float)):
                return int(iteration)
        except Exception:
            return None
        return None
    try:
        return int(suffix)
    except ValueError:
        return None


def find_latest_checkpoint(exp_dir: Path, max_iteration: Optional[int] = None) -> Path:
    checkpoints_dir = exp_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory under {exp_dir}")

    numbered = sorted(checkpoints_dir.glob("checkpoint_*.pt"))
    if not numbered:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoints_dir}")
    if max_iteration is None:
        stage_final = checkpoints_dir / "checkpoint_final.pt"
        if stage_final.exists():
            return stage_final
        return numbered[-1]

    best_path: Optional[Path] = None
    best_iteration = -math.inf
    for path in numbered:
        iteration = _checkpoint_iteration(path)
        if iteration is None:
            continue
        if iteration <= max_iteration and iteration > best_iteration:
            best_path = path
            best_iteration = iteration

    if best_path is not None:
        return best_path

    raise FileNotFoundError(
        f"No checkpoint with iteration <= {max_iteration} found in {checkpoints_dir}"
    )


def load_checkpoint_metrics(
    exp_dir: Path, max_iteration: Optional[int] = None
) -> Tuple[Dict[str, List[float]], Dict[str, float], Dict[str, Any]]:
    checkpoint_path = find_latest_checkpoint(exp_dir, max_iteration=max_iteration)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    metrics = checkpoint.get("metrics", {})
    if not metrics:
        raise ValueError(f"No metrics found in checkpoint {checkpoint_path}")

    # Load config for lambda values either from checkpoint or config.json
    config = checkpoint.get("config")
    if config is None:
        config_path = exp_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config in checkpoint or {config_path}")
        with open(config_path) as f:
            config = json.load(f)["config"]
    else:
        config = config.to_dict() if hasattr(config, "to_dict") else dict(config)

    lambdas = {
        "lambda_ipf": config.get("lambda_ipf", 1.0),
        "lambda_td": config.get("lambda_td", 1.0),
        "lambda_fk": config.get("lambda_fk", 1.0),
        "lambda_boundary": config.get("lambda_boundary", 0.0),
    }
    info = {
        "checkpoint_path": str(checkpoint_path),
        "learn_interaction_drift": bool(config.get("learn_interaction_drift", False)),
        "learn_interaction_cost": bool(config.get("learn_interaction_cost", False)),
    }
    return metrics, lambdas, info


def require_series(metrics: Dict[str, List[float]], keys: List[str], label: str) -> None:
    missing = [k for k in keys if k not in metrics or not metrics[k]]
    if missing:
        raise ValueError(
            f"Missing required metrics {missing} in {label}. "
            "These checkpoints may have been produced before analytical re-evaluation "
            "was enabled by default."
        )


def ensure_analytical_metrics(metrics: Dict[str, List[float]]) -> Dict[str, List[float]]:
    analytic_keys = [
        "loss_ipf_forward_analytical",
        "loss_td_forward_analytical",
        "loss_fk_forward_analytical",
        "loss_boundary_forward_analytical",
        "loss_ipf_backward_analytical",
        "loss_td_backward_analytical",
        "loss_fk_backward_analytical",
        "loss_boundary_backward_analytical",
    ]
    if all(k in metrics and metrics[k] for k in analytic_keys):
        return metrics

    print(
        "  Warning: analytical loss metrics missing; falling back to training losses."
        " Consider retraining with a build that records analytical evaluations."
    )
    metrics = dict(metrics)  # shallow copy
    forward_components = ["ipf", "td", "fk", "boundary"]
    for comp in forward_components:
        src = f"loss_{comp}_forward"
        dst = f"{src}_analytical"
        if src in metrics:
            metrics[dst] = list(metrics[src])
    for comp in forward_components:
        src = f"loss_{comp}_backward"
        dst = f"{src}_analytical"
        if src in metrics:
            metrics[dst] = list(metrics[src])
    return metrics


def compute_total_loss(
    metrics: Dict[str, List[float]], lambdas: Dict[str, float], suffix: str
) -> np.ndarray:
    """Compute weighted total losses for either forward/backward analytic metrics."""
    keys = [
        f"loss_ipf_{suffix}",
        f"loss_td_{suffix}",
        f"loss_fk_{suffix}",
        f"loss_boundary_{suffix}",
    ]
    length = min(len(metrics.get(k, [])) for k in keys)
    values = []
    for i in range(length):
        total = (
            lambdas["lambda_ipf"] * metrics[keys[0]][i]
            + lambdas["lambda_td"] * metrics[keys[1]][i]
            + lambdas["lambda_fk"] * metrics[keys[2]][i]
            + lambdas["lambda_boundary"] * metrics[keys[3]][i]
        )
        values.append(total)
    return np.array(values)


def prepare_series(
    metrics: Dict[str, List[float]], lambdas: Dict[str, float]
) -> Dict[str, np.ndarray]:
    metrics = ensure_analytical_metrics(metrics)
    require_series(
        metrics,
        [
            "loss_ipf_forward_analytical",
            "loss_td_forward_analytical",
            "loss_fk_forward_analytical",
            "loss_boundary_forward_analytical",
            "loss_ipf_backward_analytical",
            "loss_td_backward_analytical",
            "loss_fk_backward_analytical",
            "loss_boundary_backward_analytical",
            "time_per_iteration",
        ],
        label="metrics_analytical",
    )

    series = {
        "iterations": np.arange(1, len(metrics.get("loss_ipf_forward", [])) + 1),
        "time": np.cumsum(metrics.get("time_per_iteration", [])),
    }

    for key in metrics:
        values = metrics.get(key, [])
        series[key] = np.array(values)

    series["total_forward_analytical"] = compute_total_loss(
        metrics, lambdas, "forward_analytical"
    )
    series["total_backward_analytical"] = compute_total_loss(
        metrics, lambdas, "backward_analytical"
    )
    series["total_combined_analytical"] = (
        series["total_forward_analytical"] + series["total_backward_analytical"]
    )

    return series


def plot_losses_vs_iteration(series_a, series_b, output_dir: Path, suffix: str, fmt: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    components = ["ipf", "td", "fk", "boundary"]
    for ax, comp in zip(axes, components):
        key = f"loss_{comp}_forward_analytical"
        len_a = min(len(series_a["iterations"]), len(series_a[key]))
        len_b = min(len(series_b["iterations"]), len(series_b[key]))
        xa = series_a["iterations"][:len_a]
        ya = series_a[key][:len_a]
        xb = series_b["iterations"][:len_b]
        yb = series_b[key][:len_b]
        ax.plot(xa, ya, "--", label="Analytic", color=COLOR_ANALYTIC)
        ax.plot(xb, yb, "-", label="Learned", color=COLOR_LEARNED)
        title_comp = "Boundary" if comp == "boundary" else comp.upper()
        ax.set_title(title_comp)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    png_path = output_dir / f"loss_forward_vs_iter{suffix}.png"
    pdf_path = output_dir / f"loss_forward_vs_iter{suffix}.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, comp in zip(axes, components):
        key = f"loss_{comp}_backward_analytical"
        len_a = min(len(series_a["iterations"]), len(series_a[key]))
        len_b = min(len(series_b["iterations"]), len(series_b[key]))
        xa = series_a["iterations"][:len_a]
        ya = series_a[key][:len_a]
        xb = series_b["iterations"][:len_b]
        yb = series_b[key][:len_b]
        ax.plot(xa, ya, "--", label="Analytic", color=COLOR_ANALYTIC)
        ax.plot(xb, yb, "-", label="Learned", color=COLOR_LEARNED)
        title_comp = "Boundary" if comp == "boundary" else comp.upper()
        ax.set_title(title_comp)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    png_path = output_dir / f"loss_backward_vs_iter{suffix}.png"
    pdf_path = output_dir / f"loss_backward_vs_iter{suffix}.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)


def plot_total_vs_iteration(series_a, series_b, output_dir: Path, suffix: str, fmt: str) -> None:
    plt.figure(figsize=(6.2, 4.6))
    len_a = min(len(series_a["iterations"]), len(series_a["total_combined_analytical"]))
    len_b = min(len(series_b["iterations"]), len(series_b["total_combined_analytical"]))
    xa = series_a["iterations"][:len_a]
    xb = series_b["iterations"][:len_b]
    plt.plot(
        xa,
        series_a["total_combined_analytical"][:len_a],
        "--",
        label="Analytic",
        color=COLOR_ANALYTIC,
    )
    plt.plot(
        xb,
        series_b["total_combined_analytical"][:len_b],
        "-",
        label="Learned",
        color=COLOR_LEARNED,
    )
    plt.xlabel("Iteration")
    plt.ylabel("Total Loss (Forward + Backward)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    png_path = output_dir / f"total_loss_vs_iter{suffix}.png"
    pdf_path = output_dir / f"total_loss_vs_iter{suffix}.pdf"
    plt.savefig(png_path, dpi=150)
    plt.savefig(pdf_path)
    plt.close()


def plot_losses_vs_time(series_a, series_b, output_dir: Path, suffix: str, fmt: str) -> None:
    components = ["ipf", "td", "fk", "boundary"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, comp in zip(axes, components):
        key = f"loss_{comp}_forward_analytical"
        len_a = min(len(series_a["time"]), len(series_a[key]))
        len_b = min(len(series_b["time"]), len(series_b[key]))
        xa = series_a["time"][:len_a]
        ya = series_a[key][:len_a]
        xb = series_b["time"][:len_b]
        yb = series_b[key][:len_b]
        ax.plot(xa, ya, "--", label="Analytic", color=COLOR_ANALYTIC)
        ax.plot(xb, yb, "-", label="Learned", color=COLOR_LEARNED)
        title_comp = "Boundary" if comp == "boundary" else comp.upper()
        ax.set_title(title_comp)
        ax.set_xlabel("Elapsed Time (s)")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    png_path = output_dir / f"loss_forward_vs_time{suffix}.png"
    pdf_path = output_dir / f"loss_forward_vs_time{suffix}.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, comp in zip(axes, components):
        key = f"loss_{comp}_backward_analytical"
        len_a = min(len(series_a["time"]), len(series_a[key]))
        len_b = min(len(series_b["time"]), len(series_b[key]))
        xa = series_a["time"][:len_a]
        ya = series_a[key][:len_a]
        xb = series_b["time"][:len_b]
        yb = series_b[key][:len_b]
        ax.plot(xa, ya, "--", label="Analytic", color=COLOR_ANALYTIC)
        ax.plot(xb, yb, "-", label="Learned", color=COLOR_LEARNED)
        title_comp = "Boundary" if comp == "boundary" else comp.upper()
        ax.set_title(title_comp)
        ax.set_xlabel("Elapsed Time (s)")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    png_path = output_dir / f"loss_backward_vs_time{suffix}.png"
    pdf_path = output_dir / f"loss_backward_vs_time{suffix}.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)


def plot_total_vs_time(series_a, series_b, output_dir: Path, suffix: str, fmt: str) -> None:
    plt.figure(figsize=(6.2, 4.6))
    len_a = min(len(series_a["time"]), len(series_a["total_combined_analytical"]))
    len_b = min(len(series_b["time"]), len(series_b["total_combined_analytical"]))
    xa = series_a["time"][:len_a]
    xb = series_b["time"][:len_b]
    plt.plot(
        xa,
        series_a["total_combined_analytical"][:len_a],
        "--",
        label="Analytic",
        color=COLOR_ANALYTIC,
    )
    plt.plot(
        xb,
        series_b["total_combined_analytical"][:len_b],
        "-",
        label="Learned",
        color=COLOR_LEARNED,
    )
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Total Loss (Forward + Backward)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    png_path = output_dir / f"total_loss_vs_time{suffix}.png"
    pdf_path = output_dir / f"total_loss_vs_time{suffix}.pdf"
    plt.savefig(png_path, dpi=150)
    plt.savefig(pdf_path)
    plt.close()


def plot_time_violin(series_a, series_b, output_dir: Path, suffix: str, fmt: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.violinplot(
        [series_a["time_per_iteration"], series_b["time_per_iteration"]],
        showmeans=True,
        showextrema=True,
    )
    plt.xticks([1, 2], ["Analytic", "Learned"])
    plt.ylabel("Iteration Time (s)")
    plt.tight_layout()
    png_path = output_dir / f"iteration_time_violin{suffix}.png"
    pdf_path = output_dir / f"iteration_time_violin{suffix}.pdf"
    plt.savefig(png_path, dpi=150)
    plt.savefig(pdf_path)
    plt.close()


def plot_interaction_losses(
    series_a,
    series_b,
    info_a: Dict[str, Any],
    info_b: Dict[str, Any],
    output_dir: Path,
    suffix: str,
    fmt: str,
) -> None:
    show_drift_a = bool(info_a.get("learn_interaction_drift"))
    show_drift_b = bool(info_b.get("learn_interaction_drift"))
    show_cost_a = bool(info_a.get("learn_interaction_cost"))
    show_cost_b = bool(info_b.get("learn_interaction_cost"))

    if not (show_drift_a or show_drift_b or show_cost_a or show_cost_b):
        return

    plt.figure(figsize=(6.2, 4.6))
    legend_entries = 0
    if show_drift_a and "loss_interaction_drift" in series_a:
        len_a = min(len(series_a["iterations"]), len(series_a["loss_interaction_drift"]))
        xa = series_a["iterations"][:len_a]
        ya = series_a["loss_interaction_drift"][:len_a]
        plt.plot(
            xa,
            ya,
            "--",
            label="Drift (Analytic)",
            color=COLOR_ANALYTIC,
        )
        legend_entries += 1
    if show_drift_b and "loss_interaction_drift" in series_b:
        len_b = min(len(series_b["iterations"]), len(series_b["loss_interaction_drift"]))
        xb = series_b["iterations"][:len_b]
        yb = series_b["loss_interaction_drift"][:len_b]
        plt.plot(
            xb,
            yb,
            "-",
            label="Drift",
            color=COLOR_LEARNED,
        )
        legend_entries += 1
    if show_cost_a and "loss_interaction_cost" in series_a:
        len_a = min(len(series_a["iterations"]), len(series_a["loss_interaction_cost"]))
        xa = series_a["iterations"][:len_a]
        ya = series_a["loss_interaction_cost"][:len_a]
        plt.plot(
            xa,
            ya,
            "--",
            label="Cost (Analytic)",
            color=COLOR_ANALYTIC,
        )
        legend_entries += 1
    if show_cost_b and "loss_interaction_cost" in series_b:
        len_b = min(len(series_b["iterations"]), len(series_b["loss_interaction_cost"]))
        xb = series_b["iterations"][:len_b]
        yb = series_b["loss_interaction_cost"][:len_b]
        plt.plot(
            xb,
            yb,
            "-",
            label="Cost",
            color=COLOR_LEARNED,
        )
        legend_entries += 1
    plt.xlabel("Iteration")
    plt.ylabel("Interaction Loss")
    plt.grid(True, alpha=0.3)
    if legend_entries > 1:
        plt.legend()
    plt.tight_layout()
    png_path = output_dir / f"interaction_loss_vs_iter{suffix}.png"
    pdf_path = output_dir / f"interaction_loss_vs_iter{suffix}.pdf"
    plt.savefig(png_path, dpi=150)
    plt.savefig(pdf_path)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or Path.cwd() / "comparison_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_a, lambdas_a, info_a = load_checkpoint_metrics(
        args.analytic_dir, max_iteration=args.max_checkpoint
    )
    metrics_b, lambdas_b, info_b = load_checkpoint_metrics(
        args.learned_dir, max_iteration=args.max_checkpoint
    )

    series_a = prepare_series(metrics_a, lambdas_a)
    series_b = prepare_series(metrics_b, lambdas_b)

    suffix = ""
    plot_losses_vs_iteration(series_a, series_b, output_dir, suffix, args.figure_format)
    plot_total_vs_iteration(series_a, series_b, output_dir, suffix, args.figure_format)
    plot_losses_vs_time(series_a, series_b, output_dir, suffix, args.figure_format)
    plot_total_vs_time(series_a, series_b, output_dir, suffix, args.figure_format)
    plot_time_violin(series_a, series_b, output_dir, suffix, args.figure_format)
    plot_interaction_losses(
        series_a,
        series_b,
        info_a,
        info_b,
        output_dir,
        suffix,
        args.figure_format,
    )

    print("Analytic checkpoint:", info_a["checkpoint_path"])
    print("Learned checkpoint :", info_b["checkpoint_path"])
    print("Plots saved under  :", output_dir)


if __name__ == "__main__":
    main()
