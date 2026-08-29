#!/usr/bin/env python3
"""Plot N-vs-time for bounded-confidence opinion agents sweep."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nlmfsp.plot_style import apply_paper_style


OUTPUT_DIR = Path("latex/R05/figs/opinion_bounded_confidence_agents_sweep")


@dataclass(frozen=True)
class TimingPoint:
    backend: str
    n_agents: int
    seed: int
    exp_dir: Path
    time_per_iter_sec: float
    total_wall_time_sec: float
    n_outer_iterations: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-prefix",
        default="opinion_bounded_confidence_agents_sweep",
        help="Experiment run-name prefix before _exact/_nn.",
    )
    parser.add_argument("--query-chunk-size", type=int, default=None)
    parser.add_argument("--ref-chunk-size", type=int, default=None)
    parser.add_argument("--time-block-size", type=int, default=None)
    parser.add_argument(
        "--backend",
        choices=("both", "exact", "nn"),
        default="both",
        help="Plot both backends or one backend without requiring paired N values.",
    )
    return parser.parse_args()


def load_point(
    exp_dir: Path,
    backend: str,
    n_agents: int,
    seed: int,
    args: argparse.Namespace,
) -> TimingPoint | None:
    metrics_path = exp_dir / "metrics.json"
    config_path = exp_dir / "config.json"
    if not metrics_path.is_file() or not config_path.is_file():
        return None

    metrics = json.loads(metrics_path.read_text())
    config_payload = json.loads(config_path.read_text())
    config = config_payload.get("config", {})
    expected_chunks = (
        ("interaction_query_chunk_size", args.query_chunk_size),
        ("interaction_ref_chunk_size", args.ref_chunk_size),
        ("interaction_time_block_size", args.time_block_size),
    )
    if any(
        expected is not None and config.get(key) != expected
        for key, expected in expected_chunks
    ):
        return None
    timing = metrics.get("timing", {})
    total = timing.get("total_wall_time_sec")
    if total is None:
        return None

    n_outer = int(config.get("n_outer_iterations") or 1)
    return TimingPoint(
        backend=backend,
        n_agents=int(config.get("n_agents", n_agents)),
        seed=seed,
        exp_dir=exp_dir,
        time_per_iter_sec=float(total) / max(1, n_outer),
        total_wall_time_sec=float(total),
        n_outer_iterations=n_outer,
    )


def discover_points(args: argparse.Namespace) -> list[TimingPoint]:
    candidates: dict[tuple[str, int], TimingPoint] = {}
    run_re = re.compile(rf"{re.escape(args.run_prefix)}_(exact|nn)_n(\d+)_seed(\d+)")
    for run_dir in args.experiments_root.glob(f"{args.run_prefix}_*_n*_seed*"):
        match = run_re.fullmatch(run_dir.name)
        if match is None:
            continue
        backend, n_text, seed_text = match.groups()
        n_agents = int(n_text)
        seed = int(seed_text)
        if seed != args.seed or n_agents > args.max_n:
            continue
        for exp_dir in sorted(run_dir.glob("exp_*")):
            point = load_point(exp_dir, backend, n_agents, seed, args)
            if point is None:
                continue
            key = (backend, n_agents)
            current = candidates.get(key)
            if current is None or exp_dir.name > current.exp_dir.name:
                candidates[key] = point
    return sorted(candidates.values(), key=lambda p: (p.backend, p.n_agents))


def main() -> None:
    args = parse_args()
    apply_paper_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    points = discover_points(args)
    exact = [p for p in points if p.backend == "exact"]
    nn = [p for p in points if p.backend == "nn"]
    if args.backend == "both":
        selected_n = sorted({p.n_agents for p in exact} & {p.n_agents for p in nn})
        if not selected_n:
            raise SystemExit("No shared exact/NN n_agents values found.")
    else:
        selected = exact if args.backend == "exact" else nn
        selected_n = sorted({p.n_agents for p in selected})
        if not selected_n:
            raise SystemExit(f"No {args.backend} n_agents values found.")

    by_backend = {
        "Analytic": {p.n_agents: p for p in exact},
        "Learned": {p.n_agents: p for p in nn},
    }

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    if args.backend in ("both", "exact"):
        ax.plot(
            selected_n,
            [by_backend["Analytic"][n].time_per_iter_sec for n in selected_n],
            "^--",
            label="Analytic",
            color="tab:blue",
        )
    if args.backend in ("both", "nn"):
        ax.plot(
            selected_n,
            [by_backend["Learned"][n].time_per_iter_sec for n in selected_n],
            "o-",
            label="Learned",
            color="tab:orange",
        )
    ax.set_xlabel(r"$N$")
    ax.set_ylabel("Time per iteration (s)")
    ax.set_xticks(selected_n)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / f"time_per_iteration_vs_agents.{ext}", dpi=200)
    plt.close(fig)

    with (args.output_dir / "time_per_iteration_vs_agents.csv").open("w") as f:
        f.write("backend,n_agents,seed,exp_dir,n_outer_iterations,total_wall_time_sec,time_per_iter_sec\n")
        backend_rows = (("Analytic", "exact"), ("Learned", "nn"))
        if args.backend == "exact":
            backend_rows = (("Analytic", "exact"),)
        elif args.backend == "nn":
            backend_rows = (("Learned", "nn"),)
        for backend_name, backend_key in backend_rows:
            for n in selected_n:
                p = by_backend[backend_name][n]
                f.write(
                    f"{backend_key},{n},{p.seed},{p.exp_dir},"
                    f"{p.n_outer_iterations},{p.total_wall_time_sec:.8f},{p.time_per_iter_sec:.8f}\n"
                )

    print(f"Saved N sweep timing plot to {args.output_dir}")
    print("Selected N:", selected_n)
    for n in selected_n:
        if args.backend == "both":
            exact_t = by_backend["Analytic"][n].time_per_iter_sec
            learned_t = by_backend["Learned"][n].time_per_iter_sec
            print(
                f"N={n}: analytic={exact_t:.2f}s, learned={learned_t:.2f}s, "
                f"speedup={exact_t / learned_t:.2f}x"
            )
        else:
            name = "Analytic" if args.backend == "exact" else "Learned"
            print(f"N={n}: {name.lower()}={by_backend[name][n].time_per_iter_sec:.2f}s")


if __name__ == "__main__":
    main()
