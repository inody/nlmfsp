import argparse
from unittest.mock import patch

import pytest
import torch

from nlmfsp import (
    apply_problem_config_overrides,
    build_config,
    get_default_device,
    resolve_device,
)
from nlmfsp.analytical_interactions import (
    AnalyticalMeanReversionDrift,
    AnalyticalQuadraticDeviationCost,
)
from nlmfsp.interaction import TimedInteractionModule, create_interactions
from nlmfsp.networks import InteractionNet


def make_args(**overrides):
    base = {
        "dt": 0.02,
        "n_agents": 1000,
        "n_outer_iterations": 24,
        "n_inner_steps": 50,
        "T": 1.0,
        "lr_value": 1e-4,
        "lr_grad": 1e-4,
        "lr_interaction": None,
        "lambda_ipf": 1.0,
        "lambda_td": 1.0,
        "lambda_fk": 0.05,
        "lambda_boundary": 0.0,
        "td_horizon": 1,
        "td_bootstrap_from_target": False,
        "ema_decay": 0.0,
        "ema_start_step": 0,
        "ema_update_interval": 1,
        "sigma": 0.7,
        "hidden_dims": [64, 64],
        "activation": "tanh",
        "lr_decay": 0.997,
        "grad_clip": 0.35,
        "log_interval": 1,
        "checkpoint_interval": 8,
        "seed": 42,
        "device": "cpu",
        "interaction_steps": 0,
        "interaction_query_chunk_size": None,
        "interaction_ref_chunk_size": None,
        "interaction_time_block_size": None,
        "interaction_hidden_dims": None,
        "interaction_t_hidden_dims": None,
        "interaction_activation": None,
        "interaction_backend": None,
        "learn_interaction_drift": None,
        "learn_interaction_cost": None,
        "random_batch_size": None,
        "time_input_mode": None,
        "use_time_index_normalization": None,
        "n_agents_test": None,
        "evaluate_interaction_cpu": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_config_defaults_time_horizon():
    args = make_args()
    config = build_config(args)
    assert config.T == pytest.approx(1.0)
    assert config.n_agents_test == config.n_agents


def test_build_config_uses_time_horizon_override():
    args = make_args(T=3.5)
    config = build_config(args)
    assert config.T == pytest.approx(3.5)


def test_build_config_prefers_new_inner_step_parameter():
    args = make_args(n_inner_steps=5)
    config = build_config(args)
    assert config.n_inner_steps == 5


def test_build_config_with_none_inner_steps():
    """Test that build_config resolves explicit None via Config defaults."""
    args = make_args(n_inner_steps=None)
    config = build_config(args)
    assert config.n_inner_steps == 1000


def test_build_config_defaults_n_inner_steps_consistency():
    """Test that build_config resolves omitted args via Config defaults."""
    # Create args without n_inner_steps attribute (simulates non-CLI usage)
    args = make_args()
    # Remove n_inner_steps to test getattr fallback
    if hasattr(args, "n_inner_steps"):
        delattr(args, "n_inner_steps")

    config = build_config(args)
    # Should match Config default source-of-truth
    assert config.n_inner_steps == 1000, (
        f"Expected 1000 (Config default), got {config.n_inner_steps}"
    )


def test_build_config_requires_explicit_n_agents():
    args = make_args(n_agents=None)
    with pytest.raises(ValueError, match="n_agents must be specified explicitly"):
        build_config(args)


def test_build_config_uses_config_default_lr_when_cli_value_is_none():
    args = make_args(lr_value=None, lr_grad=None)
    config = build_config(args)
    assert config.learning_rate_value == pytest.approx(1e-4)
    assert config.learning_rate_grad == pytest.approx(1e-4)


def test_build_config_uses_config_default_hidden_dims_when_none():
    args = make_args(hidden_dims=None)
    config = build_config(args)
    assert config.hidden_dims == (64, 64, 64)


def test_build_config_propagates_time_input_mode():
    args = make_args(time_input_mode="normalized", use_time_index_normalization=False)
    config = build_config(args)
    assert config.time_input_mode == "normalized"
    assert config.use_time_index_normalization is True


def test_build_config_legacy_time_index_flag_enables_normalized_mode():
    args = make_args(time_input_mode=None, use_time_index_normalization=True)
    config = build_config(args)
    assert config.time_input_mode == "normalized"


def test_build_config_requires_lr_interaction_when_learning_enabled():
    args = make_args(
        learn_interaction_drift=True,
        learn_interaction_cost=False,
        lr_interaction=None,
    )
    with pytest.raises(
        ValueError,
        match="learning_rate_interaction must be provided when interaction learning is enabled.",
    ):
        build_config(args)


def test_build_config_defaults_to_analytical_cycle():
    args = make_args()
    config = build_config(args)
    assert config.learn_interaction is False

    assert config.n_interaction_steps == 0
    assert config.evaluate_interaction_cpu is False


def test_build_config_enables_learned_cycle():
    args = make_args(
        learn_interaction_drift=True,
        learn_interaction_cost=True,
        interaction_steps=123,
        lr_interaction=5e-4,
        n_agents_test=512,
    )
    config = build_config(args)
    assert config.learn_interaction is True
    assert config.n_agents_test == 512

    assert config.n_interaction_steps == 123
    assert config.learning_rate_interaction == pytest.approx(5e-4)
    assert config.interaction_hidden_dims is None
    assert config.evaluate_interaction_cpu is False
    assert config.interaction_backend == "nn"


def test_build_config_explicit_random_batch_backend():
    args = make_args(
        interaction_backend="random_batch",
        random_batch_size=32,
        learn_interaction_drift=False,
        learn_interaction_cost=False,
    )
    config = build_config(args)
    assert config.interaction_backend == "random_batch"
    assert config.random_batch_size == 32
    assert config.learn_interaction is False


def test_build_config_propagates_interaction_activation():
    args = make_args(
        learn_interaction_drift=True,
        lr_interaction=5e-4,
        interaction_activation="silu",
    )
    config = build_config(args)
    assert config.interaction_activation == "silu"


def test_build_config_interaction_chunk_sizes():
    args = make_args(
        interaction_query_chunk_size=33,
        interaction_ref_chunk_size=77,
        interaction_time_block_size=5,
    )
    config = build_config(args)
    assert config.interaction_query_chunk_size == 33
    assert config.interaction_ref_chunk_size == 77
    assert config.interaction_time_block_size == 5


def test_build_config_split_interaction_flags():
    args = make_args(
        learn_interaction_drift=True,
        learn_interaction_cost=False,
        interaction_steps=10,
        n_agents_test=64,
        lr_interaction=5e-4,
    )
    config = build_config(args)
    assert config.learn_interaction is True
    assert config.learn_interaction_drift is True
    assert config.learn_interaction_cost is False
    # interaction steps remain since drift learning enabled
    assert config.n_interaction_steps == 10
    assert config.n_agents_test == 64
    assert config.evaluate_interaction_cpu is False


def test_build_config_evaluate_interaction_cpu_flag():
    args = make_args(evaluate_interaction_cpu=True)
    config = build_config(args)
    assert config.evaluate_interaction_cpu is True


def test_build_config_custom_interaction_network_dims():
    args = make_args(
        learn_interaction_drift=True,
        learn_interaction_cost=True,
        interaction_hidden_dims=[96, 64],
        interaction_t_hidden_dims=[32, 32],
        lr_interaction=5e-4,
    )
    config = build_config(args)
    assert config.interaction_hidden_dims == (96, 64)
    assert config.interaction_t_hidden_dims == (32, 32)


def test_get_default_device():
    """Test that get_default_device returns a valid device string."""
    device = get_default_device()
    assert device in ["cuda", "mps", "cpu"]


def test_resolve_device_explicit():
    """Test resolve_device with explicit device choices."""
    # Test explicit choices (no auto-detection)
    assert resolve_device("cpu", print_info=False) == "cpu"
    assert resolve_device("cuda", print_info=False) == "cuda"
    assert resolve_device("mps", print_info=False) == "mps"


def test_resolve_device_auto():
    """Test resolve_device with auto-detection."""
    resolved = resolve_device("auto", print_info=False)
    # Should resolve to one of the available devices
    assert resolved in ["cuda", "mps", "cpu"]

    # Auto should match get_default_device
    expected = get_default_device()
    assert resolved == expected


def test_resolve_device_print_info_no_error():
    """Test that resolve_device with print_info=True doesn't raise errors."""
    # Test that printing doesn't cause errors (even if CUDA unavailable)
    try:
        result = resolve_device("auto", print_info=True)
        assert result in ["cuda", "mps", "cpu"]
    except Exception as e:
        pytest.fail(f"resolve_device raised unexpected exception: {e}")

    # Test with explicit CUDA (should not error even if unavailable)
    try:
        result = resolve_device("cuda", print_info=True)
        assert result == "cuda"
    except Exception as e:
        pytest.fail(f"resolve_device with cuda raised unexpected exception: {e}")


def test_resolve_device_cuda_mapping():
    """Test CUDA device name printing when available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping CUDA mapping test")

    # Should print CUDA device name without error
    resolved = resolve_device("cuda", print_info=True)
    assert resolved == "cuda"


def test_apply_problem_config_overrides_basic():
    """Test basic problem config override functionality."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr-value", type=float, default=1e-3)
    parser.add_argument("--lr-grad", type=float, default=1e-3)
    parser.add_argument("--n-agents", dest="n_agents", type=int, default=128)
    parser.add_argument("--n_outer_iterations", type=int, default=24)

    args = parser.parse_args([])

    problem_cfg = {
        "training": {
            "lr_value": 5e-4,
            "lr_grad": 4e-4,
            "n_agents": 256,
            "interaction_hidden_dims": [128, 64],
        }
    }

    with pytest.raises(ValueError, match="interaction_hidden_dims"):
        apply_problem_config_overrides(args, problem_cfg, parser)


def test_apply_problem_config_overrides_cli_priority():
    """Test that explicit CLI values take priority over config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr-value", type=float, default=1e-3)
    parser.add_argument("--lr-grad", type=float, default=1e-3)
    parser.add_argument("--n-agents", dest="n_agents", type=int, default=128)

    # User explicitly sets lr-value via CLI
    args = parser.parse_args(["--lr-value", "2e-3"])

    problem_cfg = {
        "training": {
            "lr_value": 5e-4,  # Should be ignored (CLI priority)
            "lr_grad": 4e-4,
            "n_agents": 256,  # Should apply (CLI default)
        }
    }

    args = apply_problem_config_overrides(args, problem_cfg, parser)

    # CLI explicit value should be preserved
    assert args.lr_value == pytest.approx(2e-3)
    # Config value should apply when CLI uses default
    assert args.lr_grad == pytest.approx(4e-4)
    # Config value should apply when CLI uses default
    assert args.n_agents == 256


def test_apply_problem_config_overrides_td_bootstrap_from_target():
    """Test td_bootstrap_from_target parameter handling."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--td-bootstrap-from-target",
        dest="td_bootstrap_from_target",
        action="store_true",
        default=False,
    )

    args = parser.parse_args([])

    problem_cfg = {
        "training": {
            "td_bootstrap_from_target": True,  # Should set directly
        }
    }

    args = apply_problem_config_overrides(args, problem_cfg, parser)

    # td_bootstrap_from_target should be set directly (no inversion)
    assert args.td_bootstrap_from_target is True


def test_apply_problem_config_overrides_special_handler():
    """Test special handler for T (time horizon)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=float, default=1.0)

    args = parser.parse_args([])

    problem_cfg = {
        "training": {
            "T": 2.5,
        }
    }

    args = apply_problem_config_overrides(args, problem_cfg, parser)

    assert args.T == pytest.approx(2.5)


def test_apply_problem_config_overrides_unmatched_keys():
    """Test that unrecognized keys are rejected."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr-value", type=float, default=1e-3)
    parser.add_argument(
        "--interaction-activation",
        dest="interaction_activation",
        type=str,
        default=None,
    )

    args = parser.parse_args([])

    problem_cfg = {
        "training": {
            "lr_value": 5e-4,
            "unknown_param_1": 123,
            "another_unknown": "value",
            "interaction_activation": "silu",
        }
    }

    with pytest.raises(ValueError, match="unknown_param_1"):
        apply_problem_config_overrides(args, problem_cfg, parser)


def test_apply_problem_config_overrides_empty_training():
    """Test handling of empty or missing training section."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr-value", type=float, default=1e-3)

    args = parser.parse_args([])

    # No training section
    problem_cfg = {}
    args = apply_problem_config_overrides(args, problem_cfg, parser)
    assert args.lr_value == pytest.approx(1e-3)  # Unchanged

    # Empty training section
    problem_cfg = {"training": {}}
    args = apply_problem_config_overrides(args, problem_cfg, parser)
    assert args.lr_value == pytest.approx(1e-3)  # Unchanged


def test_apply_problem_config_overrides_rejects_hyphenated_key():
    """Training keys are strict snake_case and reject hyphenated forms."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interaction-activation",
        dest="interaction_activation",
        type=str,
        default=None,
    )

    args = parser.parse_args([])

    problem_cfg = {
        "training": {
            "interaction-activation": "gelu",
        }
    }

    with pytest.raises(ValueError, match="interaction-activation"):
        apply_problem_config_overrides(args, problem_cfg, parser)


def test_apply_problem_config_overrides_rejects_removed_training_mode():
    """training_mode is no longer accepted in problem configs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr-value", type=float, default=1e-3)
    args = parser.parse_args([])

    problem_cfg = {"training": {"training_mode": "alternating"}}
    with pytest.raises(ValueError, match="Training config key 'training_mode'"):
        apply_problem_config_overrides(args, problem_cfg, parser)


def test_create_interaction_modules_label_weights_once():
    """Ensure Option B label factories respect configured interaction weights."""

    args = make_args(
        learn_interaction_drift=True,
        learn_interaction_cost=True,
        interaction_steps=5,
        lr_interaction=5e-4,
        drift_type="mean_reversion",
        cost_type="quadratic_deviation",
        int_drift_weight=0.75,
        int_cost_weight=0.4,
        int_drift_sigma=1.0,
        int_cost_sigma=1.0,
        base_drift_type="zero",
        base_cost_type="zero",
        hidden_dims=[64, 64],
    )
    config = build_config(args)
    (
        drift_module,
        cost_module,
        f_interaction_fn,
        F_interaction_fn,
        analytical_drift_module,
        analytical_cost_module,
    ) = create_interactions(config)

    xs = torch.tensor([[0.5, -0.5], [-0.25, 0.75]], dtype=torch.float32)
    empirical = xs.clone()

    ts = torch.zeros(xs.shape[0], dtype=torch.float32)
    drift_labels = f_interaction_fn(xs, empirical, ts)
    cost_labels = F_interaction_fn(xs, empirical)

    drift_ref = AnalyticalMeanReversionDrift(weight=args.int_drift_weight)
    cost_ref = AnalyticalQuadraticDeviationCost(weight=args.int_cost_weight)

    expected_drift = drift_ref.drift.evaluate(xs, empirical)
    expected_cost = cost_ref.cost.evaluate(xs, empirical)

    assert torch.allclose(drift_labels, expected_drift)
    assert torch.allclose(cost_labels, expected_cost)


def test_create_interactions_cost_only_learning():
    """Cost-only learning should keep analytical drift and learn cost network."""

    args = make_args(
        learn_interaction_drift=False,
        learn_interaction_cost=True,
        interaction_steps=3,
        lr_interaction=1e-4,
        drift_type="mean_reversion",
        cost_type="quadratic_deviation",
        int_drift_weight=0.5,
        int_cost_weight=0.25,
        int_drift_sigma=1.0,
        int_cost_sigma=1.0,
        base_drift_type="zero",
        base_cost_type="zero",
        hidden_dims=[64, 64],
    )
    config = build_config(args)
    assert config.learn_interaction is True
    assert config.learn_interaction_drift is False
    assert config.learn_interaction_cost is True

    (
        drift_module,
        cost_module,
        f_fn,
        F_fn,
        analytical_drift_module,
        analytical_cost_module,
    ) = create_interactions(config)

    assert isinstance(drift_module, AnalyticalMeanReversionDrift)
    assert isinstance(cost_module, TimedInteractionModule)
    assert isinstance(cost_module.net, InteractionNet)
    assert f_fn is None
    assert F_fn is not None


def test_create_interactions_random_batch_backend():
    args = make_args(
        interaction_backend="random_batch",
        random_batch_size=16,
        drift_type="mean_reversion",
        cost_type="quadratic_deviation",
        int_drift_weight=0.5,
        int_cost_weight=0.25,
        learn_interaction_drift=False,
        learn_interaction_cost=False,
    )
    config = build_config(args)
    (
        drift_module,
        cost_module,
        f_fn,
        F_fn,
        _,
        _,
    ) = create_interactions(config)

    assert drift_module.__class__.__name__ == "RandomBatchDrift"
    assert cost_module.__class__.__name__ == "RandomBatchCost"
    assert f_fn is None
    assert F_fn is None

    xs = torch.randn(20, 2)
    ts = torch.zeros(20)
    drift_out = drift_module(xs, ts)
    cost_out = cost_module(xs, ts)
    assert drift_out.shape == xs.shape
    assert cost_out.shape == (20, 1)
