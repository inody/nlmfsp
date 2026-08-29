"""Tests for Config.to_dict() method."""

import pytest
from nlmfsp.config import Config


def test_config_to_dict_basic():
    """Test Config.to_dict() method with basic parameters."""
    config = Config(dim=2, T=1.0, dt=0.1, n_agents=64)
    config_dict = config.to_dict()
    
    # 基本的なフィールドが含まれているか
    assert "dim" in config_dict
    assert "T" in config_dict
    assert "dt" in config_dict
    assert "n_agents" in config_dict
    
    # 値が正しいか
    assert config_dict["dim"] == 2
    assert config_dict["T"] == 1.0
    assert config_dict["dt"] == 0.1
    assert config_dict["n_agents"] == 64


def test_config_to_dict_excludes_computed_fields():
    """Test that to_dict() excludes computed fields."""
    config = Config(dim=2, T=1.0, dt=0.1, n_agents=64)
    config_dict = config.to_dict()
    
    # 計算済みフィールドが除外されているか
    assert "n_time_steps" not in config_dict


def test_config_to_dict_completeness():
    """Test that to_dict() includes all relevant fields."""
    config = Config(n_agents=64)
    config_dict = config.to_dict()
    
    # 主要なフィールドが含まれているか
    expected_fields = {
        "dim", "T", "dt", "n_agents",
        "n_outer_iterations", "n_inner_steps", "learning_rate_value",
        "learning_rate_grad", "learning_rate_interaction",
        "sigma", "lambda_ipf", "lambda_td", "lambda_fk", "lambda_boundary",
        "td_horizon", "td_bootstrap_from_target",
        "ema_decay", "ema_start_step", "ema_update_interval",
        "lr_decay_gamma", "grad_clip", "hidden_dims", "activation",
        # その他のフィールド
    }
    
    # 少なくとも主要なフィールドが含まれていることを確認
    actual_fields = set(config_dict.keys())
    assert expected_fields.issubset(actual_fields), f"Missing fields: {expected_fields - actual_fields}"


def test_config_to_dict_with_custom_values():
    """Test to_dict() with custom parameter values."""
    config = Config(
        dim=3,
        T=2.0,
        dt=0.05,
        n_agents=128,
        lambda_ipf=1.5,
        lambda_td=0.8,
    )
    
    config_dict = config.to_dict()
    
    assert config_dict["dim"] == 3
    assert config_dict["T"] == 2.0
    assert config_dict["dt"] == 0.05
    assert config_dict["n_agents"] == 128
    assert config_dict["lambda_ipf"] == 1.5
    assert config_dict["lambda_td"] == 0.8


def test_config_to_dict_serializable():
    """Test that to_dict() returns JSON-serializable values."""
    config = Config(n_agents=64)
    config_dict = config.to_dict()
    
    import json
    
    # JSONシリアライズ可能か確認
    try:
        json.dumps(config_dict)
    except (TypeError, ValueError) as e:
        pytest.fail(f"Config.to_dict() returned non-serializable values: {e}")


def test_config_requires_explicit_n_agents():
    """Config should fail fast when n_agents is omitted."""
    with pytest.raises(ValueError, match="n_agents must be specified explicitly"):
        Config()


def test_config_rejects_exact_backend_with_learn_flags():
    with pytest.raises(ValueError, match="interaction_backend='exact'"):
        Config(
            n_agents=64,
            interaction_backend="exact",
            learn_interaction_drift=True,
        )


def test_config_requires_learn_flags_for_nn_backend():
    with pytest.raises(ValueError, match="interaction_backend='nn'"):
        Config(
            n_agents=64,
            interaction_backend="nn",
            learning_rate_interaction=1e-3,
        )


def test_config_rejects_random_batch_backend_with_learn_flags():
    with pytest.raises(ValueError, match="interaction_backend='random_batch'"):
        Config(
            n_agents=64,
            interaction_backend="random_batch",
            random_batch_size=16,
            learn_interaction_drift=True,
        )


def test_config_requires_random_batch_size_for_random_batch_backend():
    with pytest.raises(ValueError, match="requires random_batch_size"):
        Config(
            n_agents=64,
            interaction_backend="random_batch",
        )


def test_config_phase_seeds_fall_back_to_seed():
    config = Config(n_agents=64, seed=123)

    assert config.training_seed == 123
    assert config.validation_seed == 123
    assert config.visualization_seed == 123


def test_config_phase_seeds_allow_explicit_override():
    config = Config(
        n_agents=64,
        seed=123,
        training_seed=1,
        validation_seed=2,
        visualization_seed=3,
    )

    assert config.training_seed == 1
    assert config.validation_seed == 2
    assert config.visualization_seed == 3
