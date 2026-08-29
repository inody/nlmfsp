#!/usr/bin/env python3
"""Plot best/full analytical loss decay from experiment checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        type=Path,
        help="Experiment directories (exp_*) to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Analytical total loss decay",
        help="Figure title.",
    )
    return parser.parse_args()


def _load_checkpoint_metrics(exp_dir: Path) -> Tuple[str, Dict[str, List[float]]]:
    checkpoint_path = exp_dir / "checkpoints" / "checkpoint_final.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint_final.pt not found in {exp_dir}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"metrics missing in {checkpoint_path}")

    config = checkpoint.get("config")
    method = getattr(config, "interaction_backend", None)
    if method == "exact":
        label = "Exact"
    elif method == "random_batch":
        label = f"RB M={getattr(config, 'random_batch_size', '?')}"
    elif method == "nn":
        hidden = getattr(config, "interaction_hidden_dims", None)
        h = hidden[0] if hidden else "?"
        steps = getattr(config, "n_interaction_steps", "?")
        label = f"NN H={h} S={steps}"
    else:
        label = exp_dir.name

    return label, metrics


def _analytical_total_curve(metrics: Dict[str, List[float]]) -> List[float]:
    keys = [
        "loss_ipf_forward_analytical",
        "loss_td_forward_analytical",
        "loss_fk_forward_analytical",
        "loss_ipf_backward_analytical",
        "loss_td_backward_analytical",
        "loss_fk_backward_analytical",
    ]
    for key in keys:
        if key not in metrics or not isinstance(metrics[key], list) or not metrics[key]:
            raise ValueError(f"Missing analytical loss history: {key}")

    n = len(metrics[keys[0]])
    for key in keys[1:]:
        if len(metrics[key]) != n:
            raise ValueError(f"Inconsistent analytical history length for {key}")

    return [sum(float(metrics[key][idx]) for key in keys) for idx in range(n)]


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for exp_dir in args.experiment_dirs:
        label, metrics = _load_checkpoint_metrics(exp_dir)
        curve = _analytical_total_curve(metrics)
        ax.plot(range(len(curve)), curve, label=label, linewidth=2)

    ax.set_title(args.title)
    ax.set_xlabel("Checkpoint index")
    ax.set_ylabel("Analytical total loss (IPF+TD+FK)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
