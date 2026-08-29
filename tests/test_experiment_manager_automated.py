"""Tests for ExperimentManager.save_config() with automated to_dict()."""

import json
import tempfile
from pathlib import Path

import pytest

from nlmfsp.config import Config
from nlmfsp.experiment_manager import ExperimentManager


def test_save_config_automated():
    """Test that save_config works with automated to_dict()."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # 一時的な実験ルートを設定
        config = Config(
            dim=2,
            T=1.0,
            dt=0.1,
            n_agents=128,
            learning_rate_value=1e-3,
            learning_rate_grad=1e-3,
            experiment_root=temp_dir,
            problem_config_path="configs/paper/gmm/main_exact.json",
        )
        
        manager = ExperimentManager(config, scenario_name="test_scenario")
        
        # 設定保存
        manager.save_config(config)
        
        # 保存されたファイルを検証
        config_path = manager.exp_dir / "config.json"
        assert config_path.exists()
        
        with open(config_path) as f:
            saved_data = json.load(f)
        
        # 構造を確認
        assert "experiment_id" in saved_data
        assert "scenario_name" in saved_data
        assert "interaction_backend" in saved_data
        assert "method" in saved_data
        assert "method_group" in saved_data
        assert "timestamp" in saved_data
        assert "config" in saved_data
        
        # 設定値を確認
        assert saved_data["config"]["dim"] == 2
        assert saved_data["config"]["T"] == 1.0
        assert saved_data["config"]["n_agents"] == 128
        assert saved_data["interaction_backend"] == "exact"
        assert saved_data["method"] == "exact"
        assert saved_data["method_group"] == "analytical"
        assert (
            saved_data["config"]["problem_config_path"]
            == "configs/paper/gmm/main_exact.json"
        )
        
        # 計算済みフィールドが除外されているか
        assert "n_time_steps" not in saved_data["config"]
        
        # deviceが文字列に変換されているか
        assert "device" in saved_data["config"]
        assert isinstance(saved_data["config"]["device"], str)


def test_experiment_manager_requires_explicit_scenario_name():
    """Experiment outputs should not silently fall back to a generic scenario bucket."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = Config(
            dim=2,
            T=1.0,
            dt=0.1,
            n_agents=128,
            learning_rate_value=1e-3,
            learning_rate_grad=1e-3,
            experiment_root=temp_dir,
        )

        with pytest.raises(ValueError, match="scenario_name is required"):
            ExperimentManager(config)


def test_save_config_with_cli_args():
    """Test save_config with CLI arguments."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = Config(
            dim=3,
            T=2.0,
            n_agents=64,
            learning_rate_value=1e-3,
            learning_rate_grad=1e-3,
            experiment_root=temp_dir,
        )
        
        manager = ExperimentManager(config, scenario_name="test_cli")
        
        # 擬似CLI引数
        import argparse
        cli_args = argparse.Namespace()
        cli_args.learn_interaction_drift = True
        cli_args.learn_interaction_cost = True
        cli_args.interaction_steps = 500
        cli_args._parser = None  # 非シリアライズ可能オブジェクト
        
        # 設定保存
        manager.save_config(config, cli_args)
        
        # 保存されたファイルを検証
        config_path = manager.exp_dir / "config.json"
        with open(config_path) as f:
            saved_data = json.load(f)
        
        # CLI引数が保存されているか
        assert "cli_args" in saved_data
        assert saved_data["cli_args"]["learn_interaction_drift"] is True
        assert saved_data["cli_args"]["learn_interaction_cost"] is True
        assert saved_data["cli_args"]["interaction_steps"] == 500
        assert "_parser" not in saved_data["cli_args"]  # 除外されているか


def test_save_config_completeness():
    """Test that save_config includes all config fields."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # 全てのパラメータをカスタマイズ
        config = Config(
            dim=2,
            T=1.0,
            dt=0.05,
            n_agents=256,
            n_outer_iterations=100,
            n_inner_steps=50,
            learning_rate_value=1e-3,
            learning_rate_grad=1e-3,
            sigma=0.5,
            lambda_ipf=1.0,
            lambda_td=0.5,
            lambda_fk=0.1,
            lambda_boundary=0.0,
            td_horizon=3,
            td_bootstrap_from_target=True,
            ema_decay=0.99,
            ema_start_step=1000,
            ema_update_interval=10,
            lr_decay_gamma=0.999,
            grad_clip=1.0,
            hidden_dims=(64, 64),
            activation="tanh",
            experiment_root=temp_dir,
        )
        
        manager = ExperimentManager(config, scenario_name="test_complete")
        manager.save_config(config)
        
        # 保存された設定を検証
        config_path = manager.exp_dir / "config.json"
        with open(config_path) as f:
            saved_data = json.load(f)
        
        saved_config = saved_data["config"]
        
        # 主要なパラメータが全て保存されているか
        assert saved_config["dim"] == 2
        assert saved_config["T"] == 1.0
        assert saved_config["dt"] == 0.05
        assert saved_config["n_agents"] == 256
        assert saved_config["lambda_ipf"] == 1.0
        assert saved_config["td_horizon"] == 3
        assert saved_config["hidden_dims"] == [64, 64]
        assert saved_config["activation"] == "tanh"


def test_save_config_roundtrip():
    """Test that a saved config can be loaded back."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_config = Config(
            dim=2,
            T=1.0,
            n_agents=128,
            learning_rate_value=1e-3,
            learning_rate_grad=1e-3,
            experiment_root=temp_dir,
        )
        
        manager = ExperimentManager(original_config, scenario_name="test_compat")
        manager.save_config(original_config)
        
        # 保存された設定を読み込んで新しいConfigを作成
        config_path = manager.exp_dir / "config.json"
        with open(config_path) as f:
            saved_data = json.load(f)
        
        # from_existingで読み込めるか
        loaded_manager = ExperimentManager.from_existing(manager.exp_dir)
        
        # 設定が一致しているか
        assert loaded_manager.config.dim == original_config.dim
        assert loaded_manager.config.T == original_config.T
        assert loaded_manager.config.n_agents == original_config.n_agents
        assert loaded_manager.interaction_backend == "exact"
        assert loaded_manager.method == "exact"


def test_save_config_random_batch_writes_canonical_backend_metadata():
    """Random-batch runs should be identified canonically in top-level metadata."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = Config(
            dim=2,
            T=1.0,
            n_agents=128,
            random_batch_size=32,
            interaction_backend="random_batch",
            learning_rate_value=1e-3,
            learning_rate_grad=1e-3,
            experiment_root=temp_dir,
        )

        manager = ExperimentManager(config, scenario_name="test_random_batch")
        manager.save_config(config)

        config_path = manager.exp_dir / "config.json"
        with open(config_path) as f:
            saved_data = json.load(f)

        assert saved_data["interaction_backend"] == "random_batch"
        assert saved_data["method"] == "random_batch"
        assert saved_data["method_group"] == "analytical"


def test_from_existing_infers_backend_from_legacy_metadata():
    """Older config payloads without top-level interaction_backend remain loadable."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_config = Config(
            dim=2,
            T=1.0,
            n_agents=128,
            learning_rate_value=1e-3,
            learning_rate_grad=1e-3,
            learning_rate_interaction=1e-3,
            learn_interaction_drift=True,
            interaction_backend="nn",
            experiment_root=temp_dir,
        )

        manager = ExperimentManager(original_config, scenario_name="test_legacy_nn")
        manager.save_config(original_config)

        config_path = manager.exp_dir / "config.json"
        with open(config_path) as f:
            saved_data = json.load(f)

        saved_data.pop("interaction_backend", None)
        saved_data.pop("method_group", None)
        saved_data["method"] = "learned"
        saved_data["config"].pop("interaction_backend", None)

        with open(config_path, "w") as f:
            json.dump(saved_data, f, indent=2)

        loaded_manager = ExperimentManager.from_existing(manager.exp_dir)

        assert loaded_manager.interaction_backend == "nn"
        assert loaded_manager.method == "nn"
        assert loaded_manager.method_group == "learned"
        assert loaded_manager.config.interaction_backend == "nn"


def test_save_config_records_run_provenance():
    """Every run must record what produced it: revision, machine, versions.

    The paper's provenance had to be reconstructed from timestamps because none
    of this was saved. ``git_dirty`` is part of the contract: a commit hash does
    not identify the code if the working tree had uncommitted changes.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        config = Config(dim=2, T=1.0, dt=0.1, n_agents=8, experiment_root=temp_dir)
        manager = ExperimentManager(config, scenario_name="provenance_test")
        manager.save_config(config)

        saved = json.loads((manager.exp_dir / "config.json").read_text())
        provenance = saved["provenance"]

        for field in ("hostname", "platform", "python_version",
                      "git_commit", "git_dirty", "torch_version"):
            assert field in provenance, f"provenance is missing {field!r}"

        assert provenance["hostname"], "hostname was not recorded"
        # git_dirty is a tristate: True, False, or None when git is unavailable.
        assert provenance["git_dirty"] in (True, False, None)
        if provenance["git_commit"] is not None:
            assert len(provenance["git_commit"]) == 40, "expected a full commit hash"
