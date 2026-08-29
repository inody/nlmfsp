#!/usr/bin/env python3
"""Render the GPU-only directional V-neck NN/RB backend comparison table."""

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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/comparison_plots/"
            "vneck_pedestrian_visual_backend_gpu_seed310_312"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[310, 311, 312])
    parser.add_argument("--n-agents", type=int, default=1000)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def config(exp: Path) -> dict[str, Any]:
    payload = load_json(exp / "config.json")
    return payload.get("config_merged", payload.get("config", {}))


def is_complete(exp: Path) -> bool:
    return (exp / "metrics.json").exists() and (
        exp / "checkpoints" / "checkpoint_final.pt"
    ).exists()


def latest_complete_run(scenario: Path, seed: int, n_agents: int) -> Path | None:
    matches: list[Path] = []
    for config_path in scenario.glob("exp_*/config.json"):
        exp = config_path.parent
        cfg = config(exp)
        if int(cfg.get("seed", -1)) != seed:
            continue
        if int(cfg.get("n_agents", -1)) != n_agents:
            continue
        if cfg.get("cost_type") != "pedestrian_visual":
            continue
        if cfg.get("evaluate_interaction_cpu") is not False:
            continue
        if is_complete(exp):
            matches.append(exp)
    return sorted(matches)[-1] if matches else None


def backend_plot_module():
    path = PROJECT_ROOT / "scripts" / "plot_backend_comparison.py"
    spec = importlib.util.spec_from_file_location("plot_backend_comparison", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def main() -> None:
    args = parse_args()
    plot_backend = backend_plot_module()
    experiments = Path("outputs/experiments")
    scenarios = {
        "Learned surrogate": lambda seed: experiments
        / f"vneck_pedestrian_visual_backend_nn_h64_int200_gpu_seed{seed}",
        "Random batch": lambda seed: experiments
        / f"vneck_pedestrian_visual_backend_rb128_gpu_seed{seed}",
    }

    metric_paths: list[Path] = []
    for method, scenario_for_seed in scenarios.items():
        for seed in args.seeds:
            exp = latest_complete_run(scenario_for_seed(seed), seed, args.n_agents)
            if exp is not None:
                metric_paths.append(exp / "metrics.json")
            else:
                print(f"Missing complete run: {method}, seed={seed}")

    rows = plot_backend._collect_rows(metric_paths, loss_stat="best-so-far")
    if not rows:
        raise SystemExit("No complete GPU-only backend runs found.")

    run_rows: list[dict[str, Any]] = []
    for row in rows:
        method = (
            "Random batch"
            if row["label"].startswith("RB")
            else "Learned surrogate"
        )
        cfg = row["config_blob"].get("config_merged", {})
        run_rows.append(
            {
                "method": method,
                "seed": row["seed"],
                "time_per_iter_sec": row["wall"]
                / int(cfg["n_outer_iterations"]),
                "best_analytical_loss": row["total_loss"],
                "interaction_error": row["interaction_error"],
                "run": str(row["path"].parent),
            }
        )

    seeds_by_method = {
        method: {int(row["seed"]) for row in run_rows if row["method"] == method}
        for method in scenarios
    }
    paired_seeds = sorted(set.intersection(*seeds_by_method.values()))
    if not paired_seeds:
        raise SystemExit("No seed has both a complete learned-surrogate and random-batch run.")
    paired_rows = [row for row in run_rows if int(row["seed"]) in paired_seeds]

    summary_rows: list[dict[str, Any]] = []
    for method in ("Learned surrogate", "Random batch"):
        selected = [row for row in paired_rows if row["method"] == method]
        if not selected:
            continue
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

    def value(method: str, key: str, digits: int) -> str:
        row = by_method[method]
        return (
            f"${row[f'{key}_mean']:.{digits}f} "
            f"\\pm {row[f'{key}_std']:.{digits}f}$"
        )

    n_pairs = len(paired_seeds)
    seed_text = ", ".join(str(seed) for seed in paired_seeds)
    tex = rf"""\begin{{table}}
  \centering
  \begin{{tabular}}{{lccc}}
    \hline
    Method & Time/iter. (s) & Analytical loss & Interaction error \\
    \hline
    Learned surrogate ($M_{{\mathrm{{int}}}}=200$) & {value("Learned surrogate", "time_per_iter_sec", 2)} & {value("Learned surrogate", "best_analytical_loss", 2)} & {value("Learned surrogate", "interaction_error", 3)} \\
    Random batch ($M_{{\mathrm{{RB}}}}=128$) & {value("Random batch", "time_per_iter_sec", 2)} & {value("Random batch", "best_analytical_loss", 2)} & {value("Random batch", "interaction_error", 3)} \\
    \hline
  \end{{tabular}}
  \caption{{Directional V-neck GPU-only backend comparison at $N={args.n_agents}$ and $\Delta t=0.01$, using the {n_pairs} currently complete paired seed(s) ({seed_text}). Time per iteration is total training time divided by 100 outer iterations. Analytical loss is the best checkpoint-evaluated value of $\mathcal{{L}}_{{\mathrm{{tot}}}}$, computed with analytical interactions. Interaction error is the mean relative $L^2$ error of the forward and backward cost interaction.}}
  \label{{tab:vneck_backend_comparison}}
\end{{table}}
"""
    (args.output_dir / "backend_comparison_table.tex").write_text(tex)
    print(f"Saved backend comparison artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
