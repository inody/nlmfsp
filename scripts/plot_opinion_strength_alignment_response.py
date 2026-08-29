#!/usr/bin/env python3
"""Plot Gaussian-kernel proximity A(t) for bounded-confidence strength sweep."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_DIR = REPO_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from nlmfsp.cli_helpers import resolve_device
from nlmfsp.plot_style import apply_paper_style
from plot_opinion_bounded_confidence_paper_figures import (
    analytical_total_series,
    best_iteration,
    load_solver_and_sample,
)


@dataclass(frozen=True)
class ExperimentSpec:
    label: str
    weight: float
    exp_dir: Path
    min_iteration: int | None = None
    max_iteration: int | None = None
    seed: int | None = None


EXACT_EXPERIMENTS = [
    ExperimentSpec(
        "No interaction",
        0.0,
        Path("outputs/experiments/opinion_bounded_confidence_no_interaction/exp_20260525_063803"),
    ),
    ExperimentSpec(
        r"$w=0.25$",
        0.25,
        Path(
            "outputs/experiments/"
            "opinion_bounded_confidence_weight_sweep_exact_w0p25_seed42/"
            "exp_20260525_063832"
        ),
    ),
    ExperimentSpec(
        r"$w=0.5$",
        0.5,
        Path(
            "outputs/experiments/"
            "opinion_bounded_confidence_weight_sweep_exact_w0p5_seed42/"
            "exp_20260525_190451"
        ),
        min_iteration=20,
    ),
    ExperimentSpec(
        r"$w=1.0$",
        1.0,
        Path(
            "outputs/experiments/"
            "analytical_opinion_1k_bounded_confidence_cost_off_n400_dt001_sigma1_clip1/"
            "exp_20260524_163204"
        ),
    ),
    ExperimentSpec(
        r"$w=1.5$",
        1.5,
        Path(
            "outputs/experiments/"
            "opinion_bounded_confidence_weight_sweep_exact_w1p5_seed42/"
            "exp_20260527_160945"
        ),
    ),
    ExperimentSpec(
        r"$w=2.0$",
        2.0,
        Path(
            "outputs/experiments/"
            "opinion_bounded_confidence_weight_sweep_exact_w2p0_seed42/"
            "exp_20260526_074326"
        ),
    ),
]

LEARNED_EXPERIMENTS = [
    ExperimentSpec(
        "No interaction",
        0.0,
        Path("outputs/experiments/opinion_bounded_confidence_no_interaction/exp_20260525_063803"),
    ),
    ExperimentSpec(
        r"$w=0.25$",
        0.25,
        Path(
            "outputs/experiments/"
            "opinion_bounded_confidence_weight_sweep_nn_w0p25_seed42/"
            "exp_20260525_063855"
        ),
    ),
    ExperimentSpec(
        r"$w=0.5$",
        0.5,
        Path(
            "outputs/experiments/"
            "opinion_bounded_confidence_weight_sweep_nn_w0p5_seed42/"
            "exp_20260525_092031"
        ),
    ),
    ExperimentSpec(
        r"$w=1.0$",
        1.0,
        Path(
            "outputs/experiments/"
            "learned_opinion_1k_bounded_confidence_cost_off_n400_dt001_sigma1_int100_clip1_detachdiv/"
            "exp_20260524_192349"
        ),
        max_iteration=45,
    ),
    ExperimentSpec(
        r"$w=1.5$",
        1.5,
        Path(
            "outputs/experiments/"
            "opinion_bounded_confidence_weight_sweep_nn_w1p5_seed42/"
            "exp_20260528_033606"
        ),
    ),
    ExperimentSpec(
        r"$w=2.0$",
        2.0,
        Path(
            "outputs/experiments/"
            "opinion_bounded_confidence_weight_sweep_nn_w2p0_seed42/"
            "exp_20260525_120213"
        ),
    ),
]

OUTPUT_DIR = Path("latex/R05/figs/opinion_bounded_confidence_strength_sweep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--num-trajectories", type=int, default=400)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Kernel scale for A(t). Default: int_drift_sigma from the first run.",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=1.5,
        help="If set, only include runs with interaction weight <= this value.",
    )
    parser.add_argument(
        "--aggregate-seeds",
        action="store_true",
        help=(
            "Discover matching local weight-sweep runs and plot mean/std over "
            "available seeds. Default keeps the original hand-picked runs."
        ),
    )
    parser.add_argument(
        "--exact-run-prefix",
        default="opinion_bounded_confidence_weight_sweep_exact",
        help="Run-name prefix for exact weight-sweep experiments.",
    )
    parser.add_argument(
        "--learned-run-prefix",
        default="opinion_bounded_confidence_weight_sweep_nn",
        help="Run-name prefix for learned weight-sweep experiments.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional seed list for --aggregate-seeds.",
    )
    return parser.parse_args()


def load_interaction_sigma(exp_dir: Path) -> float:
    with (exp_dir / "config.json").open() as f:
        payload = json.load(f)
    config = payload.get("config", payload)
    return float(config.get("int_drift_sigma", 1.0))


def gaussian_proximity_series(
    xs: torch.Tensor,
    *,
    sigma: float,
) -> np.ndarray:
    """Compute A(t) = mean_{i != j} exp(-|x_i-x_j|^2/(2 sigma^2))."""
    values = []
    n_steps, batch, _dim = xs.shape
    if batch < 2:
        raise ValueError("Need at least two trajectories to compute A(t).")
    for step in range(n_steps):
        x = xs[step]
        dist_sq = torch.cdist(x, x, p=2.0).square()
        affinity = torch.exp(-dist_sq / (2.0 * sigma**2))
        total = affinity.sum() - torch.diagonal(affinity).sum()
        values.append(float((total / (batch * (batch - 1))).cpu()))
    return np.asarray(values, dtype=float)


def select_best_iteration(exp: ExperimentSpec) -> int | None:
    if exp.min_iteration is None:
        return best_iteration(exp.exp_dir, max_iteration=exp.max_iteration)

    series = analytical_total_series(exp.exp_dir, max_iteration=exp.max_iteration)
    valid = series["iteration"] >= exp.min_iteration
    if not np.any(valid):
        print(
            f"Skipping {exp.label} in {exp.exp_dir}: "
            f"no iterations >= {exp.min_iteration}.",
            file=sys.stderr,
        )
        return None
    iterations = series["iteration"][valid]
    totals = series["total"][valid]
    return int(iterations[np.argmin(totals)])


def _load_config(exp_dir: Path) -> dict:
    with (exp_dir / "config.json").open() as f:
        payload = json.load(f)
    return payload.get("config", payload)


def _is_complete_exp(exp_dir: Path) -> bool:
    return (exp_dir / "metrics.json").exists() and bool(
        list((exp_dir / "checkpoints").glob("checkpoint_[0-9][0-9][0-9][0-9].pt"))
    )


def _latest_complete_exp(run_dir: Path) -> Path | None:
    candidates = []
    for exp_dir in sorted(run_dir.glob("exp_*")):
        if not (exp_dir / "config.json").exists() or not _is_complete_exp(exp_dir):
            continue
        candidates.append(exp_dir)
    return candidates[-1] if candidates else None


def discover_aggregate_experiments(
    *,
    seeds: set[int] | None,
    exact_run_prefix: str,
    learned_run_prefix: str,
) -> tuple[list[ExperimentSpec], list[ExperimentSpec]]:
    exact_specs: list[ExperimentSpec] = []
    learned_specs: list[ExperimentSpec] = []

    for run_prefix, target in (
        (exact_run_prefix, exact_specs),
        (learned_run_prefix, learned_specs),
    ):
        pattern = f"{run_prefix}_w*_seed*"
        for run_dir in sorted(Path("outputs/experiments").glob(pattern)):
            exp_dir = _latest_complete_exp(run_dir)
            if exp_dir is None:
                continue
            cfg = _load_config(exp_dir)
            if int(cfg.get("n_agents", -1)) != 400:
                continue
            seed = int(cfg.get("seed", -1))
            if seeds is not None and seed not in seeds:
                continue
            weight = float(cfg.get("int_drift_weight", 0.0))
            target.append(
                ExperimentSpec(
                    "No interaction" if math.isclose(weight, 0.0) else rf"$w={weight:g}$",
                    weight,
                    exp_dir,
                    seed=seed,
                )
            )

    return exact_specs, learned_specs


def plot_experiment_group(
    *,
    experiments: list[ExperimentSpec],
    output_dir: Path,
    output_stem: str,
    kernel_sigma: float,
    num_trajectories: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    summary_rows = []
    cmap = plt.get_cmap("viridis")
    sorted_experiments = sorted(experiments, key=lambda item: item.weight)
    plotted_weights = sorted({exp.weight for exp in sorted_experiments})
    colors = {
        weight: cmap(i / max(1, len(plotted_weights) - 1))
        for i, weight in enumerate(plotted_weights)
    }

    by_weight: dict[float, list[ExperimentSpec]] = {}
    for exp in sorted_experiments:
        by_weight.setdefault(exp.weight, []).append(exp)

    for weight in sorted(by_weight):
        proximities = []
        iterations = []
        seeds = []
        times = None
        for exp in by_weight[weight]:
            iteration = select_best_iteration(exp)
            if iteration is None:
                continue
            xs_forward, _xs_backward, time_grid = load_solver_and_sample(
                exp.exp_dir,
                iteration,
                n_samples=num_trajectories,
                seed=seed,
                device=device,
            )
            proximity = gaussian_proximity_series(
                xs_forward,
                sigma=kernel_sigma,
            )
            times = time_grid[: len(proximity)].numpy()
            proximities.append(proximity)
            iterations.append(iteration)
            if exp.seed is not None:
                seeds.append(exp.seed)

        if not proximities or times is None:
            continue

        stacked = np.stack(proximities, axis=0)
        mean_proximity = stacked.mean(axis=0)
        std_proximity = stacked.std(axis=0) if stacked.shape[0] > 1 else None
        label = "baseline" if math.isclose(weight, 0.0) else rf"$w={weight:g}$"
        ax.plot(
            times,
            mean_proximity,
            label=label,
            color=colors[weight],
            linewidth=2.0,
        )
        if std_proximity is not None:
            ax.fill_between(
                times,
                mean_proximity - std_proximity,
                mean_proximity + std_proximity,
                color=colors[weight],
                alpha=0.15,
                linewidth=0,
            )
        summary_rows.append(
            {
                "label": label,
                "weight": weight,
                "iteration": ";".join(str(i) for i in iterations),
                "n_seeds": len(proximities),
                "seeds": ";".join(str(s) for s in sorted(set(seeds))),
                "A0": float(mean_proximity[0]),
                "AT": float(mean_proximity[-1]),
                "Amax": float(np.max(mean_proximity)),
            }
        )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$A_k(t)$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{output_stem}.{ext}", dpi=200)
    plt.close(fig)
    return summary_rows


def main() -> None:
    args = parse_args()
    apply_paper_style()
    device = torch.device(resolve_device(args.device, print_info=False))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exact_experiments = EXACT_EXPERIMENTS
    learned_experiments = LEARNED_EXPERIMENTS
    if args.aggregate_seeds:
        seed_filter = set(args.seeds) if args.seeds else None
        exact_experiments, learned_experiments = discover_aggregate_experiments(
            seeds=seed_filter,
            exact_run_prefix=args.exact_run_prefix,
            learned_run_prefix=args.learned_run_prefix,
        )
    if args.max_weight is not None:
        exact_experiments = [
            exp for exp in exact_experiments if exp.weight <= args.max_weight
        ]
        learned_experiments = [
            exp for exp in learned_experiments if exp.weight <= args.max_weight
        ]
    sigma_source = next(
        (exp.exp_dir for exp in [*exact_experiments, *learned_experiments]),
        EXACT_EXPERIMENTS[0].exp_dir,
    )
    kernel_sigma = args.sigma if args.sigma is not None else load_interaction_sigma(sigma_source)

    exact_rows = plot_experiment_group(
        experiments=exact_experiments,
        output_dir=args.output_dir,
        output_stem="proximity_response_exact_vs_time",
        kernel_sigma=kernel_sigma,
        num_trajectories=args.num_trajectories,
        seed=args.seed,
        device=device,
    )
    learned_rows = plot_experiment_group(
        experiments=learned_experiments,
        output_dir=args.output_dir,
        output_stem="proximity_response_vs_time",
        kernel_sigma=kernel_sigma,
        num_trajectories=args.num_trajectories,
        seed=args.seed,
        device=device,
    )

    with (args.output_dir / "proximity_response_summary.csv").open("w") as f:
        f.write("backend,label,weight,iteration,n_seeds,seeds,sigma,A0,AT,Amax\n")
        for backend, rows in (("exact", exact_rows), ("learned", learned_rows)):
            for row in rows:
                f.write(
                    f"{backend},{row['label']},{row['weight']:g},{row['iteration']},"
                    f"{row.get('n_seeds', 1)},{row.get('seeds', '')},{kernel_sigma:.8f},"
                    f"{row['A0']:.8f},{row['AT']:.8f},{row['Amax']:.8f}\n"
                )

    print(f"Saved proximity response to {args.output_dir}")
    print(f"Kernel sigma: {kernel_sigma:g}")
    for backend, rows in (("Exact", exact_rows), ("Learned", learned_rows)):
        for row in rows:
            print(
                f"{backend} {row['label']}: iter={row['iteration']}, "
                f"A(0)={row['A0']:.4f}, A(T)={row['AT']:.4f}, max={row['Amax']:.4f}"
            )


if __name__ == "__main__":
    main()
