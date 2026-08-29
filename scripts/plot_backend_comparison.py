#!/usr/bin/env python3
"""Plot end-to-end comparison for exact / random-batch / NN backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlmfsp.plot_style import apply_paper_style

apply_paper_style()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help=(
            "Experiment scenario directories, specific exp_* directories, or metrics.json files. "
            "For scenario directories, the latest run for each backend/param combination is used "
            "unless --aggregate-seeds is enabled, in which case all exp_*/metrics.json files are used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/comparison_plots/backend_comparison"),
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="backend_comparison",
        help="Filename prefix for saved figures.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom figure title prefix.",
    )
    parser.add_argument(
        "--aggregate-seeds",
        action="store_true",
        help="Aggregate runs with the same label and plot mean/std across seeds.",
    )
    parser.add_argument(
        "--loss-checkpoint-source",
        choices=("final", "numbered", "summary"),
        default="final",
        help=(
            "Which checkpoint to use when reading analytical loss histories. "
            "'final' uses checkpoint_final.pt; 'numbered' prefers the latest "
            "checkpoint_XXXX.pt and falls back to checkpoint_final.pt; "
            "'summary' uses metrics.json analytical_losses directly."
        ),
    )
    parser.add_argument(
        "--loss-checkpoint-name",
        type=str,
        default=None,
        help=(
            "Optional explicit checkpoint filename inside each run's checkpoints/ directory, "
            "e.g. 'checkpoint_0030.pt' or 'checkpoint_0050.pt'. "
            "When provided, this overrides --loss-checkpoint-source for loss-history loading."
        ),
    )
    parser.add_argument(
        "--loss-stat",
        choices=("summary-final", "last-eval", "last10-mean", "best-so-far"),
        default="best-so-far",
        help=(
            "How to aggregate analytical total loss. "
            "'summary-final' uses metrics.json analytical_losses directly; "
            "'last-eval' uses the last checkpoint-evaluated analytical total loss "
            "stored in the selected checkpoint file; "
            "'last10-mean' uses the mean of the last 10 checkpoint evaluations; "
            "'best-so-far' uses the minimum checkpoint-evaluated analytical total loss."
        ),
    )
    parser.add_argument(
        "--include-labels",
        nargs="+",
        default=None,
        help=(
            "Optional exact labels to keep after collection, e.g. "
            "'Exact' 'RB M=64' 'NN H=128 S=100'."
        ),
    )
    parser.add_argument(
        "--loss-log-scale",
        action="store_true",
        help="Plot analytical loss axes on a log scale.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _companion_config(metrics_path: Path) -> Dict[str, Any]:
    config_path = metrics_path.with_name("config.json")
    if not config_path.exists():
        return {}
    return _load_json(config_path)


def _analytical_total_loss_from_summary(summary: Dict[str, Any]) -> float:
    analytical = summary.get("analytical_losses", {})
    if analytical:
        return float(
            analytical.get("loss_ipf_forward_analytical", 0.0)
            + analytical.get("loss_td_forward_analytical", 0.0)
            + analytical.get("loss_fk_forward_analytical", 0.0)
            + analytical.get("loss_ipf_backward_analytical", 0.0)
            + analytical.get("loss_td_backward_analytical", 0.0)
            + analytical.get("loss_fk_backward_analytical", 0.0)
        )
    return float(
        summary.get("final_total_losses", {}).get("bidirectional_ipf_td_fk", 0.0) or 0.0
    )


def _analytical_loss_checkpoint_path(
    metrics_path: Path,
    checkpoint_source: str,
    checkpoint_name: str | None = None,
) -> Path | None:
    if checkpoint_source == "summary":
        return None
    checkpoints_dir = metrics_path.with_name("checkpoints")
    if checkpoint_name:
        explicit = checkpoints_dir / checkpoint_name
        return explicit if explicit.exists() else None
    final_path = checkpoints_dir / "checkpoint_final.pt"
    if checkpoint_source == "final":
        return final_path if final_path.exists() else None

    numbered = sorted(checkpoints_dir.glob("checkpoint_[0-9][0-9][0-9][0-9].pt"))
    if numbered:
        return numbered[-1]
    if final_path.exists():
        return final_path
    return None


def _tail_mean_analytical_total_loss(
    metrics_path: Path,
    summary: Dict[str, Any],
    *,
    window: int = 10,
    checkpoint_source: str = "final",
    checkpoint_name: str | None = None,
) -> float:
    checkpoint_path = _analytical_loss_checkpoint_path(
        metrics_path,
        checkpoint_source,
        checkpoint_name,
    )
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metrics = checkpoint.get("metrics", {})
        analytical_keys = [
            "loss_ipf_forward_analytical",
            "loss_td_forward_analytical",
            "loss_fk_forward_analytical",
            "loss_ipf_backward_analytical",
            "loss_td_backward_analytical",
            "loss_fk_backward_analytical",
        ]
        if all(isinstance(metrics.get(key), list) and metrics.get(key) for key in analytical_keys):
            totals = [
                sum(float(metrics[key][idx]) for key in analytical_keys)
                for idx in range(len(metrics["loss_ipf_forward_analytical"]))
            ]
            if totals:
                tail = totals[-min(window, len(totals)) :]
                return float(sum(tail) / len(tail))
    return _analytical_total_loss_from_summary(summary)


def _last_analytical_total_loss(
    metrics_path: Path,
    summary: Dict[str, Any],
    *,
    checkpoint_source: str = "final",
    checkpoint_name: str | None = None,
) -> float:
    checkpoint_path = _analytical_loss_checkpoint_path(
        metrics_path,
        checkpoint_source,
        checkpoint_name,
    )
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metrics = checkpoint.get("metrics", {})
        analytical_keys = [
            "loss_ipf_forward_analytical",
            "loss_td_forward_analytical",
            "loss_fk_forward_analytical",
            "loss_ipf_backward_analytical",
            "loss_td_backward_analytical",
            "loss_fk_backward_analytical",
        ]
        if all(isinstance(metrics.get(key), list) and metrics.get(key) for key in analytical_keys):
            idx = len(metrics["loss_ipf_forward_analytical"]) - 1
            return float(sum(float(metrics[key][idx]) for key in analytical_keys))
    return _analytical_total_loss_from_summary(summary)


def _best_analytical_total_loss(
    metrics_path: Path,
    summary: Dict[str, Any],
    *,
    checkpoint_source: str = "final",
    checkpoint_name: str | None = None,
) -> float:
    checkpoint_path = _analytical_loss_checkpoint_path(
        metrics_path,
        checkpoint_source,
        checkpoint_name,
    )
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metrics = checkpoint.get("metrics", {})
        analytical_keys = [
            "loss_ipf_forward_analytical",
            "loss_td_forward_analytical",
            "loss_fk_forward_analytical",
            "loss_ipf_backward_analytical",
            "loss_td_backward_analytical",
            "loss_fk_backward_analytical",
        ]
        if all(isinstance(metrics.get(key), list) and metrics.get(key) for key in analytical_keys):
            totals = [
                sum(float(metrics[key][idx]) for key in analytical_keys)
                for idx in range(len(metrics["loss_ipf_forward_analytical"]))
            ]
            if totals:
                return float(min(totals))
    return _analytical_total_loss_from_summary(summary)


def _analytical_total_loss(
    metrics_path: Path,
    summary: Dict[str, Any],
    *,
    checkpoint_source: str = "final",
    loss_stat: str = "best-so-far",
    checkpoint_name: str | None = None,
) -> float:
    if loss_stat == "summary-final":
        return _analytical_total_loss_from_summary(summary)
    if loss_stat == "last-eval":
        return _last_analytical_total_loss(
            metrics_path,
            summary,
            checkpoint_source=checkpoint_source,
            checkpoint_name=checkpoint_name,
        )
    if loss_stat == "best-so-far":
        return _best_analytical_total_loss(
            metrics_path,
            summary,
            checkpoint_source=checkpoint_source,
            checkpoint_name=checkpoint_name,
        )
    return _tail_mean_analytical_total_loss(
        metrics_path,
        summary,
        checkpoint_source=checkpoint_source,
        checkpoint_name=checkpoint_name,
    )


def _loss_axis_title(loss_stat: str) -> str:
    if loss_stat == "summary-final":
        return "Analytical total loss (final)"
    if loss_stat == "last-eval":
        return "Analytical total loss (selected checkpoint)"
    if loss_stat == "best-so-far":
        return "Analytical total loss (best so far)"
    return "Analytical total loss (last 10 mean)"


def _interaction_steps(summary: Dict[str, Any], config_blob: Dict[str, Any]) -> int | None:
    config_steps = summary.get("config", {}).get("n_interaction_steps")
    if config_steps is not None:
        return int(config_steps)
    cli_steps = config_blob.get("cli_args", {}).get("interaction_steps")
    if cli_steps is not None:
        return int(cli_steps)
    merged_steps = config_blob.get("config_merged", {}).get("n_interaction_steps")
    if merged_steps is not None:
        return int(merged_steps)
    config_steps = config_blob.get("config", {}).get("n_interaction_steps")
    if config_steps is not None:
        return int(config_steps)
    return None


def _label_from_summary(summary: Dict[str, Any]) -> str:
    method = summary.get("method", {})
    backend = method.get("interaction_backend", summary.get("config", {}).get("interaction_backend"))
    param = method.get("method_param")
    if backend == "exact":
        return "Exact"
    if backend == "random_batch":
        return f"RB {param}" if param else "RB"
    if backend == "nn":
        return f"NN {param}" if param else "NN"
    return str(backend)


def _label_for_row(summary: Dict[str, Any], config_blob: Dict[str, Any]) -> str:
    method = summary.get("method", {})
    backend = method.get("interaction_backend", summary.get("config", {}).get("interaction_backend"))
    param = method.get("method_param")
    if backend == "exact":
        return "Exact"
    if backend == "random_batch":
        return f"RB {param}" if param else "RB"
    if backend == "nn":
        steps = _interaction_steps(summary, config_blob)
        if param and steps is not None:
            return f"NN {param} S={steps}"
        if param:
            return f"NN {param}"
        return "NN"
    return str(backend)


def _backend_from_summary(summary: Dict[str, Any]) -> str | None:
    method = summary.get("method", {})
    backend = method.get("interaction_backend", summary.get("config", {}).get("interaction_backend"))
    if backend in {"exact", "random_batch", "nn"}:
        return str(backend)
    return None


def _sort_key(summary: Dict[str, Any], config_blob: Dict[str, Any]) -> tuple[int, int, int]:
    method = summary.get("method", {})
    backend = method.get("interaction_backend", summary.get("config", {}).get("interaction_backend"))
    param = method.get("method_param") or ""
    order = {"exact": 0, "random_batch": 1, "nn": 2}.get(str(backend), 99)
    digits = "".join(ch for ch in str(param) if ch.isdigit())
    numeric = int(digits) if digits else 0
    steps = _interaction_steps(summary, config_blob) or 0
    return order, numeric, steps


def _variant_key(summary: Dict[str, Any], config_blob: Dict[str, Any]) -> Tuple[str, str, int | None]:
    method = summary.get("method", {})
    backend = _backend_from_summary(summary)
    param = str(method.get("method_param"))
    steps = _interaction_steps(summary, config_blob) if backend == "nn" else None
    return backend or "", param, steps


def _row_bar_colors(rows: List[Dict[str, Any]]) -> List[Any]:
    colors: List[Any] = [None] * len(rows)

    def _numeric_variant(row: Dict[str, Any]) -> int:
        label = str(row.get("label", ""))
        digits = "".join(ch for ch in label if ch.isdigit())
        return int(digits) if digits else 0

    exact_idx = [idx for idx, row in enumerate(rows) if _backend_from_summary(row["summary"]) == "exact"]
    rb_idx = [idx for idx, row in enumerate(rows) if _backend_from_summary(row["summary"]) == "random_batch"]
    nn_idx = [idx for idx, row in enumerate(rows) if _backend_from_summary(row["summary"]) == "nn"]

    for idx in exact_idx:
        colors[idx] = "tab:blue"

    if rb_idx:
        rb_sorted = sorted(rb_idx, key=lambda idx: _numeric_variant(rows[idx]))
        rb_cmap = plt.cm.Greens(np.linspace(0.45, 0.85, len(rb_sorted)))
        for idx, color in zip(rb_sorted, rb_cmap):
            colors[idx] = color

    if nn_idx:
        nn_sorted = sorted(
            nn_idx,
            key=lambda idx: (_interaction_steps(rows[idx]["summary"], rows[idx]["config_blob"]) or 0),
        )
        nn_cmap = plt.cm.Oranges(np.linspace(0.45, 0.85, len(nn_sorted)))
        for idx, color in zip(nn_sorted, nn_cmap):
            colors[idx] = color

    return colors


def _latest_metrics_for_scenario_dir(path: Path) -> List[Path]:
    metric_paths = sorted(path.glob("exp_*/metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    latest: Dict[tuple[str, str, int | None], Path] = {}
    for metrics_path in metric_paths:
        summary = _load_json(metrics_path)
        backend = _backend_from_summary(summary)
        if backend is None:
            continue
        config_blob = _companion_config(metrics_path)
        key = _variant_key(summary, config_blob)
        if key not in latest:
            latest[key] = metrics_path
    return list(latest.values())


def _all_metrics_for_scenario_dir(path: Path) -> List[Path]:
    return sorted(path.glob("exp_*/metrics.json"))


def _resolve_metric_paths(inputs: Iterable[Path], *, aggregate_seeds: bool = False) -> List[Path]:
    resolved: List[Path] = []
    for raw in inputs:
        path = raw.expanduser()
        if path.is_file():
            if path.name == "metrics.json":
                resolved.append(path)
            continue
        if path.is_dir():
            if path.name.startswith("exp_"):
                metrics_path = path / "metrics.json"
                if metrics_path.exists():
                    resolved.append(metrics_path)
            else:
                if aggregate_seeds:
                    resolved.extend(_all_metrics_for_scenario_dir(path))
                else:
                    resolved.extend(_latest_metrics_for_scenario_dir(path))
    unique = []
    seen = set()
    for path in resolved:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _collect_rows(
    metric_paths: Iterable[Path],
    *,
    checkpoint_source: str = "final",
    loss_stat: str = "best-so-far",
    checkpoint_name: str | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in metric_paths:
        summary = _load_json(path)
        config_blob = _companion_config(path)
        interaction_error, interaction_error_kind = _interaction_error_metric(
            path, summary, config_blob
        )
        backend = _backend_from_summary(summary)
        if backend is None:
            continue
        timing = summary.get("timing", {})
        method = summary.get("method", {})
        param = method.get("method_param")
        n_agents = summary.get("config", {}).get("n_agents")
        if backend == "random_batch" and param == f"M={n_agents}":
            continue
        if "interaction_errors" not in summary or "final_total_losses" not in summary:
            continue
        rows.append(
            {
                "path": path,
                "summary": summary,
                "config_blob": config_blob,
                "label": _label_for_row(summary, config_blob),
                "seed": _seed_value(summary, config_blob, "seed"),
                "training_seed": _seed_value(summary, config_blob, "training_seed"),
                "validation_seed": _seed_value(summary, config_blob, "validation_seed"),
                "visualization_seed": _seed_value(summary, config_blob, "visualization_seed"),
                "wall": float(timing.get("total_wall_time_sec", 0.0) or 0.0),
                "drift": float(timing.get("drift_step_time_sec_mean", 0.0) or 0.0),
                "interaction_train": float(timing.get("interaction_training_time_sec", 0.0) or 0.0),
                "nn_inference": float(timing.get("nn_inference_time_sec_total", 0.0) or 0.0),
                "rb_sampling": float(timing.get("random_batch_sampling_time_sec_total", 0.0) or 0.0),
                "rb_eval": float(timing.get("random_batch_eval_time_sec_total", 0.0) or 0.0),
                "interaction_steps": _interaction_steps(summary, config_blob),
                "mean_error": float(summary.get("mean_error", 0.0) or 0.0),
                "std_error": float(summary.get("std_error", 0.0) or 0.0),
                "total_loss": _analytical_total_loss(
                    path,
                    summary,
                    checkpoint_source=checkpoint_source,
                    loss_stat=loss_stat,
                    checkpoint_name=checkpoint_name,
                ),
                "loss_stat": loss_stat,
                "interaction_error": interaction_error,
                "interaction_error_kind": interaction_error_kind,
                "sort_key": _sort_key(summary, config_blob),
            }
        )
    rows.sort(key=lambda row: row["sort_key"])
    return rows


def _scenario_name(metrics_path: Path, config_blob: Dict[str, Any]) -> str:
    scenario = config_blob.get("scenario_name")
    if scenario:
        return str(scenario)
    return metrics_path.parent.parent.name


def _interaction_error_metric(
    metrics_path: Path,
    summary: Dict[str, Any],
    config_blob: Dict[str, Any],
) -> tuple[float, str]:
    interaction_errors = summary.get("interaction_errors", {})
    scenario = _scenario_name(metrics_path, config_blob).lower()

    if "vneck" in scenario:
        return (
            float(
                (
                    interaction_errors.get("forward_cost", {}).get("relative_l2", 0.0)
                    + interaction_errors.get("backward_cost", {}).get("relative_l2", 0.0)
                )
                / 2.0
            ),
            "cost",
        )

    return (
        float(
            (
                interaction_errors.get("forward_drift", {}).get("relative_l2", 0.0)
                + interaction_errors.get("backward_drift", {}).get("relative_l2", 0.0)
            )
            / 2.0
        ),
        "drift",
    )


def _seed_value(
    summary: Dict[str, Any],
    config_blob: Dict[str, Any],
    key: str,
) -> Optional[int]:
    for container in (
        config_blob.get("cli_args", {}),
        config_blob.get("config_merged", {}),
        config_blob.get("config", {}),
        summary.get("config", {}),
    ):
        value = container.get(key)
        if value is not None:
            return int(value)
    return None


def _row_seed_sort_key(row: Dict[str, Any]) -> tuple[int, int, int, int]:
    def _norm(value: Any) -> int:
        return -1 if value is None else int(value)

    return (
        _norm(row.get("training_seed")),
        _norm(row.get("validation_seed")),
        _norm(row.get("visualization_seed")),
        _norm(row.get("seed")),
    )


def _aggregate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)

    metrics = [
        "wall",
        "interaction_train",
        "nn_inference",
        "rb_sampling",
        "rb_eval",
        "total_loss",
        "interaction_error",
    ]
    agg_rows: List[Dict[str, Any]] = []
    for label, group in grouped.items():
        group = sorted(group, key=_row_seed_sort_key)
        base = dict(group[0])
        base["seed_runs"] = group
        for metric in metrics:
            values = np.array([float(row.get(metric, 0.0) or 0.0) for row in group], dtype=float)
            base[metric] = float(values.mean())
            base[f"{metric}_std"] = float(values.std(ddof=0))
        agg_rows.append(base)
    agg_rows.sort(key=lambda row: row["sort_key"])
    return agg_rows


def _interaction_error_axis_title(rows: List[Dict[str, Any]]) -> str:
    kinds = {str(row.get("interaction_error_kind", "drift")) for row in rows}
    if len(kinds) == 1:
        kind = next(iter(kinds))
        if kind == "cost":
            return "Cost interaction error"
        if kind == "drift":
            return "Drift interaction error"
    return "Interaction error"


def _filter_rows_by_label(
    rows: List[Dict[str, Any]],
    labels: Iterable[str] | None,
) -> List[Dict[str, Any]]:
    if labels is None:
        return rows
    allowed = set(labels)
    return [row for row in rows if row["label"] in allowed]


def _set_log_scale_if_positive(ax: plt.Axes, values: np.ndarray) -> None:
    finite = values[np.isfinite(values)]
    if finite.size and np.all(finite > 0.0):
        ax.set_yscale("log")


def _plot_summary(
    rows: List[Dict[str, Any]],
    output_path: Path,
    title_prefix: str,
    *,
    loss_log_scale: bool = False,
) -> None:
    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    colors = _row_bar_colors(rows)
    wall = np.array([row["wall"] for row in rows], dtype=float)
    interaction_error = np.array([row["interaction_error"] for row in rows], dtype=float)
    total_loss = np.array([row["total_loss"] for row in rows], dtype=float)
    wall_std = np.array([row.get("wall_std", 0.0) for row in rows], dtype=float)
    total_loss_std = np.array([row.get("total_loss_std", 0.0) for row in rows], dtype=float)
    interaction_error_std = np.array([row.get("interaction_error_std", 0.0) for row in rows], dtype=float)
    loss_title = _loss_axis_title(rows[0].get("loss_stat", "last10-mean"))
    interaction_title = _interaction_error_axis_title(rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    axes[0].bar(x, wall, color=colors)
    axes[0].set_title("Wall time")
    axes[0].set_ylabel("seconds")
    axes[0].errorbar(x, wall, yerr=wall_std, fmt="none", ecolor="black", capsize=3)

    axes[1].bar(x, total_loss, label="final total loss", color=colors)
    axes[1].set_title("Analytical loss")
    axes[1].set_ylabel("loss")
    if loss_log_scale:
        _set_log_scale_if_positive(axes[1], total_loss)
    axes[1].errorbar(
        x,
        total_loss,
        yerr=total_loss_std,
        fmt="none",
        ecolor="black",
        capsize=3,
    )

    axes[2].bar(x, interaction_error, color=colors)
    axes[2].set_title("Interaction error")
    axes[2].set_ylabel("rel. L2")
    axes[2].errorbar(
        x,
        interaction_error,
        yerr=interaction_error_std,
        fmt="none",
        ecolor="black",
        capsize=3,
    )

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_tradeoff(
    rows: List[Dict[str, Any]],
    output_path: Path,
    title_prefix: str,
    *,
    loss_log_scale: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.get_cmap("tab10").colors
    loss_title = _loss_axis_title(rows[0].get("loss_stat", "last10-mean"))
    interaction_title = _interaction_error_axis_title(rows)

    for idx, row in enumerate(rows):
        label = row["label"]
        color = colors[idx % len(colors)]
        seed_runs = row.get("seed_runs")
        if seed_runs:
            axes[0].scatter(
                [r["wall"] for r in seed_runs],
                [r["total_loss"] for r in seed_runs],
                s=35,
                color=color,
                alpha=0.35,
            )
            axes[1].scatter(
                [r["wall"] for r in seed_runs],
                [r["interaction_error"] for r in seed_runs],
                s=35,
                color=color,
                alpha=0.35,
            )
        axes[0].scatter(row["wall"], row["total_loss"], s=110, color=color, label=label, edgecolors="black", linewidths=0.5)
        axes[1].scatter(row["wall"], row["interaction_error"], s=110, color=color, label=label, edgecolors="black", linewidths=0.5)

    axes[0].set_title(f"Wall time vs {loss_title.lower()}")
    axes[0].set_xlabel("Total wall time (s)")
    axes[0].set_ylabel(loss_title)
    if loss_log_scale:
        _set_log_scale_if_positive(
            axes[0], np.array([row["total_loss"] for row in rows], dtype=float)
        )
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title(f"Wall time vs {interaction_title.lower()}")
    axes[1].set_xlabel("Total wall time (s)")
    axes[1].set_ylabel(f"{interaction_title} rel. L2")
    axes[1].grid(True, alpha=0.3)

    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f"{title_prefix}: Speed-accuracy tradeoff")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_nn_steps(rows: List[Dict[str, Any]], output_path: Path, title_prefix: str) -> None:
    nn_rows = [row for row in rows if row["summary"].get("method", {}).get("interaction_backend") == "nn"]
    if not nn_rows:
        return

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in nn_rows:
        key = str(row["summary"].get("method", {}).get("method_param"))
        grouped.setdefault(key, []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"H=64": "tab:blue", "H=128": "tab:orange", "H=256": "tab:green"}
    loss_title = _loss_axis_title(rows[0].get("loss_stat", "last10-mean"))
    interaction_title = _interaction_error_axis_title(nn_rows)

    for h_label, h_rows in sorted(grouped.items()):
        h_rows.sort(key=lambda r: r["interaction_steps"] or 0)
        steps = [row["interaction_steps"] for row in h_rows]
        total_loss = [row["total_loss"] for row in h_rows]
        interaction_error = [row["interaction_error"] for row in h_rows]
        color = colors.get(h_label, None)
        axes[0].plot(steps, total_loss, marker="o", label=h_label, color=color)
        axes[1].plot(steps, interaction_error, marker="o", label=h_label, color=color)

    axes[0].set_title(f"NN steps vs {loss_title.lower()}")
    axes[0].set_xlabel("Interaction steps")
    axes[0].set_ylabel(loss_title)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title(f"NN steps vs {interaction_title.lower()}")
    axes[1].set_xlabel("Interaction steps")
    axes[1].set_ylabel(f"{interaction_title} rel. L2")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.suptitle(f"{title_prefix}: NN step sweep")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_rb_breakdown(rows: List[Dict[str, Any]], output_path: Path, title_prefix: str) -> None:
    rb_rows = [row for row in rows if row["label"].startswith("RB")]
    if not rb_rows:
        return
    labels = [row["label"] for row in rb_rows]
    x = np.arange(len(rb_rows))
    sampling = np.array([row["rb_sampling"] for row in rb_rows], dtype=float)
    eval_ = np.array([row["rb_eval"] for row in rb_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, sampling, label="sampling", color="tab:purple")
    ax.bar(x, eval_, bottom=sampling, label="eval", color="tab:cyan")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("seconds")
    ax.set_title(f"{title_prefix}: RB timing breakdown")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metric_paths = _resolve_metric_paths(args.inputs, aggregate_seeds=args.aggregate_seeds)
    if not metric_paths:
        raise SystemExit("No metrics.json files found from the provided inputs.")

    rows = _collect_rows(
        metric_paths,
        checkpoint_source=args.loss_checkpoint_source,
        loss_stat=args.loss_stat,
        checkpoint_name=args.loss_checkpoint_name,
    )
    if not rows:
        raise SystemExit("No valid runs found in the provided inputs.")
    if args.aggregate_seeds:
        rows = _aggregate_rows(rows)
    rows = _filter_rows_by_label(rows, args.include_labels)
    if not rows:
        raise SystemExit("No rows left after applying --include-labels.")

    n_agents = rows[0]["summary"].get("config", {}).get("n_agents")
    title_prefix = args.title or f"Backend comparison (N={n_agents})"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _plot_summary(
        rows,
        args.output_dir / f"{args.prefix}_summary.png",
        title_prefix,
        loss_log_scale=args.loss_log_scale,
    )
    _plot_tradeoff(
        rows,
        args.output_dir / f"{args.prefix}_tradeoff.png",
        title_prefix,
        loss_log_scale=args.loss_log_scale,
    )
    _plot_nn_steps(rows, args.output_dir / f"{args.prefix}_nn_steps.png", title_prefix)

    print("Included runs:")
    for row in rows:
        seed_desc = (
            f"seed={row.get('seed')}, "
            f"training={row.get('training_seed')}, "
            f"validation={row.get('validation_seed')}, "
            f"visualization={row.get('visualization_seed')}"
        )
        if row.get("seed_runs"):
            print(
                f"  - {row['label']}: {row['path']} "
                f"(n={len(row['seed_runs'])}, first {seed_desc})"
            )
        else:
            print(f"  - {row['label']}: {row['path']} ({seed_desc})")
    print(f"Saved figures under {args.output_dir}")


if __name__ == "__main__":
    main()
