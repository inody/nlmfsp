"""Tests for seed aggregation path resolution in plot_backend_comparison."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path


def _load_plot_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "plot_backend_comparison.py"
    )
    spec = importlib.util.spec_from_file_location("plot_backend_comparison", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_run(exp_dir: Path, *, backend: str, method_param: str, seed: int) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "method": {
            "interaction_backend": backend,
            "method_param": method_param,
        },
        "config": {
            "interaction_backend": backend,
            "n_agents": 1000,
        },
        "interaction_errors": {
            "forward_drift": {"relative_l2": 0.0},
            "backward_drift": {"relative_l2": 0.0},
        },
        "final_total_losses": {"bidirectional_ipf_td_fk": 0.0},
    }
    config = {
        "cli_args": {
            "seed": seed,
        }
    }
    (exp_dir / "metrics.json").write_text(json.dumps(metrics))
    (exp_dir / "config.json").write_text(json.dumps(config))


def test_resolve_metric_paths_expands_all_runs_for_seed_aggregation(tmp_path):
    module = _load_plot_module()
    scenario_dir = tmp_path / "analytical_gmm_navigation"
    _write_run(scenario_dir / "exp_001", backend="random_batch", method_param="M=100", seed=101)
    _write_run(scenario_dir / "exp_002", backend="random_batch", method_param="M=100", seed=102)

    resolved = module._resolve_metric_paths([scenario_dir], aggregate_seeds=True)

    assert [path.parent.name for path in resolved] == ["exp_001", "exp_002"]


def test_resolve_metric_paths_keeps_latest_variant_only_without_seed_aggregation(tmp_path):
    module = _load_plot_module()
    scenario_dir = tmp_path / "analytical_gmm_navigation"
    _write_run(scenario_dir / "exp_001", backend="random_batch", method_param="M=100", seed=101)
    _write_run(scenario_dir / "exp_002", backend="random_batch", method_param="M=100", seed=102)

    resolved = module._resolve_metric_paths([scenario_dir], aggregate_seeds=False)

    assert len(resolved) == 1
    assert resolved[0].parent.name == "exp_002"


def test_collect_rows_reads_phase_seeds_from_config_blob(tmp_path):
    module = _load_plot_module()
    exp_dir = tmp_path / "learned_gmm_navigation_h64" / "exp_001"
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "method": {
            "interaction_backend": "nn",
            "method_param": "H=64",
        },
        "config": {
            "interaction_backend": "nn",
            "n_agents": 1000,
        },
        "interaction_errors": {
            "forward_drift": {"relative_l2": 0.0},
            "backward_drift": {"relative_l2": 0.0},
        },
        "final_total_losses": {"bidirectional_ipf_td_fk": 0.0},
        "timing": {},
    }
    config = {
        "cli_args": {
            "seed": 1,
            "training_seed": 101,
            "validation_seed": 202,
            "visualization_seed": 303,
        }
    }
    (exp_dir / "metrics.json").write_text(json.dumps(metrics))
    (exp_dir / "config.json").write_text(json.dumps(config))

    rows = module._collect_rows([exp_dir / "metrics.json"])

    assert len(rows) == 1
    row = rows[0]
    assert row["seed"] == 1
    assert row["training_seed"] == 101
    assert row["validation_seed"] == 202
    assert row["visualization_seed"] == 303


def test_collect_rows_uses_cost_error_for_vneck(tmp_path):
    module = _load_plot_module()
    exp_dir = tmp_path / "learned_vneck_navigation" / "exp_001"
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "method": {
            "interaction_backend": "nn",
            "method_param": "H=128",
        },
        "config": {
            "interaction_backend": "nn",
            "n_agents": 1000,
        },
        "interaction_errors": {
            "forward_drift": {"relative_l2": 0.0},
            "backward_drift": {"relative_l2": 0.0},
            "forward_cost": {"relative_l2": 0.2},
            "backward_cost": {"relative_l2": 0.4},
        },
        "final_total_losses": {"bidirectional_ipf_td_fk": 0.0},
        "timing": {},
    }
    config = {
        "scenario_name": "learned_vneck_navigation",
        "cli_args": {
            "seed": 1,
        }
    }
    (exp_dir / "metrics.json").write_text(json.dumps(metrics))
    (exp_dir / "config.json").write_text(json.dumps(config))

    rows = module._collect_rows([exp_dir / "metrics.json"])

    assert len(rows) == 1
    row = rows[0]
    assert math.isclose(row["interaction_error"], 0.3)
    assert row["interaction_error_kind"] == "cost"
