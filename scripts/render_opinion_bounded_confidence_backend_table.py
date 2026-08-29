#!/usr/bin/env python3
"""Render the paired-seed opinion NN/RB backend comparison table."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44, 45])
    parser.add_argument(
        "--learned-run-template",
        default=(
            "learned_opinion_1k_bounded_confidence_cost_off_n400_"
            "dt001_sigma1_int100_clip1_detachdiv_seed{seed}"
        ),
        help="Run directory template for learned runs; use {seed}.",
    )
    parser.add_argument(
        "--random-batch-run-template",
        default="rb80_opinion_bounded_confidence_n400_gpu_seed{seed}",
        help="Run directory template for random-batch runs; use {seed}.",
    )
    parser.add_argument("--interaction-steps", type=int, default=100)
    parser.add_argument("--random-batch-size", type=int, default=80)
    parser.add_argument("--n-outer-iterations", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/comparison_plots/"
            "opinion_bounded_confidence_backend_gpu_seed43_45"
        ),
    )
    return parser.parse_args()


def backend_plot_module():
    path = PROJECT_ROOT / "scripts" / "plot_backend_comparison.py"
    spec = importlib.util.spec_from_file_location("plot_backend_comparison", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def latest_complete_metrics(root: Path) -> Path:
    matches = [
        path
        for path in root.glob("exp_*/metrics.json")
        if (path.parent / "checkpoints" / "checkpoint_final.pt").exists()
    ]
    if not matches:
        raise FileNotFoundError(f"No complete run in {root}")
    return sorted(matches)[-1]


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def main() -> None:
    args = parse_args()
    plot_backend = backend_plot_module()
    experiments = Path("outputs/experiments")
    metric_paths: list[Path] = []
    for seed in args.seeds:
        metric_paths.append(
            latest_complete_metrics(
                experiments / args.learned_run_template.format(seed=seed)
            )
        )
        metric_paths.append(
            latest_complete_metrics(
                experiments / args.random_batch_run_template.format(seed=seed)
            )
        )

    rows = plot_backend._collect_rows(metric_paths, loss_stat="best-so-far")
    run_rows: list[dict[str, Any]] = []
    for row in rows:
        cfg = row["config_blob"].get("config_merged", {})
        method = (
            "Random batch"
            if row["label"].startswith("RB")
            else "Learned surrogate"
        )
        run_rows.append(
            {
                "method": method,
                "seed": row["seed"],
                "time_per_iter_sec": row["wall"] / int(cfg["n_outer_iterations"]),
                "best_analytical_loss": row["total_loss"],
                "interaction_error": row["interaction_error"],
                "run": str(row["path"].parent),
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for method in ("Learned surrogate", "Random batch"):
        selected = [row for row in run_rows if row["method"] == method]
        summary: dict[str, Any] = {"method": method, "n_runs": len(selected)}
        for key in (
            "time_per_iter_sec",
            "best_analytical_loss",
            "interaction_error",
        ):
            mean, std = mean_std([float(row[key]) for row in selected])
            summary[f"{key}_mean"] = mean
            summary[f"{key}_std"] = std
        summary_rows.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "backend_comparison_runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)
    with (args.output_dir / "backend_comparison_summary.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (args.output_dir / "backend_comparison_summary.json").write_text(
        json.dumps(summary_rows, indent=2) + "\n"
    )

    by_method = {row["method"]: row for row in summary_rows}
    n_runs = min(int(row["n_runs"]) for row in summary_rows)

    def value(method: str, key: str, digits: int) -> str:
        row = by_method[method]
        return (
            f"${row[f'{key}_mean']:.{digits}f} "
            f"\\pm {row[f'{key}_std']:.{digits}f}$"
        )

    tex = rf"""\begin{{table}}
  \centering
  \begin{{tabular}}{{lccc}}
    \hline
    Method & Time/iter. (s) & Analytical loss & Interaction error \\
    \hline
    Learned surrogate ($M_{{\mathrm{{int}}}}={args.interaction_steps}$) & {value("Learned surrogate", "time_per_iter_sec", 2)} & {value("Learned surrogate", "best_analytical_loss", 2)} & {value("Learned surrogate", "interaction_error", 3)} \\
    Random batch ($M_{{\mathrm{{RB}}}}={args.random_batch_size}$) & {value("Random batch", "time_per_iter_sec", 2)} & {value("Random batch", "best_analytical_loss", 2)} & {value("Random batch", "interaction_error", 3)} \\
    \hline
  \end{{tabular}}
  \caption{{Opinion dynamics backend comparison at $N=400$, $d=10$, and $\Delta t=0.01$. All columns are averaged over {n_runs} paired independent runs and reported as mean $\pm$ standard deviation. Time per iteration is total training time divided by {args.n_outer_iterations} outer iterations. Analytical loss is the best checkpoint-evaluated value of $\mathcal{{L}}_{{\mathrm{{tot}}}}$, computed with the analytical drift. Interaction error is the mean relative $L^2$ error of the forward and backward drift interaction.}}
  \label{{tab:opinion_backend_comparison}}
\end{{table}}
"""
    (args.output_dir / "backend_comparison_table.tex").write_text(tex)
    print(f"Saved backend comparison artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
