"""Tests for neural-network time input preprocessing."""

import pytest
import torch

from src.nlmfsp.config import Config, migrate_legacy_config_dict
from src.nlmfsp.networks import (
    ToyGradientNet,
    ToyInteractionNet,
    ToyValueNet,
    _process_time_input,
)


def _config(**kwargs) -> Config:
    params = dict(
        dim=2,
        T=2.0,
        dt=0.5,
        n_agents=4,
        hidden_dims=(16, 16),
        t_hidden_dims=(16, 16),
        interaction_hidden_dims=(16, 16),
        interaction_t_hidden_dims=(16, 16),
        activation="silu",
        use_time_embedding=False,
        time_index_interval=100.0,
    )
    params.update(kwargs)
    return Config(**params)


def test_raw_time_input_preserves_physical_time():
    config = _config(time_input_mode="raw", use_time_index_normalization=True)
    t = torch.tensor([0.0, 1.0, 2.0])

    processed = _process_time_input(t, config)

    assert config.use_time_index_normalization is False
    assert torch.allclose(processed.squeeze(-1), t)


def test_normalized_time_input_maps_terminal_time_to_interval():
    config = _config(time_input_mode="normalized", time_index_interval=50.0)
    t = torch.tensor([0.0, 1.0, 2.0])

    processed = _process_time_input(t, config)

    assert config.use_time_index_normalization is True
    assert torch.allclose(processed.squeeze(-1), torch.tensor([0.0, 25.0, 50.0]))


def test_invalid_time_input_mode_is_rejected():
    with pytest.raises(ValueError, match="time_input_mode"):
        _config(time_input_mode="index")


def test_legacy_config_dict_defaults_missing_time_input_mode_to_raw():
    migrated = migrate_legacy_config_dict(
        {
            "dim": 2,
            "T": 2.0,
            "dt": 0.5,
            "n_agents": 4,
            "use_time_index_normalization": True,
            "n_time_steps": 4,
        }
    )

    config = Config(**migrated)

    assert "n_time_steps" not in migrated
    assert config.time_input_mode == "raw"
    assert config.use_time_index_normalization is False


def _assert_normalized_matches_raw_with_preprocessed_time(net_cls, *args):
    raw_config = _config(time_input_mode="raw")
    normalized_config = _config(time_input_mode="normalized")
    raw_net = net_cls(raw_config, *args)
    normalized_net = net_cls(normalized_config, *args)
    raw_net.load_state_dict(normalized_net.state_dict())

    x = torch.randn(6, 2)
    t_physical = torch.linspace(0.0, normalized_config.T, x.shape[0])
    t_normalized = t_physical / normalized_config.T * normalized_config.time_index_interval

    out_normalized = normalized_net(x, t_physical)
    out_raw = raw_net(x, t_normalized)

    assert torch.allclose(out_normalized, out_raw, atol=1e-6)


def test_toy_value_net_uses_central_time_preprocessing():
    _assert_normalized_matches_raw_with_preprocessed_time(ToyValueNet)


def test_toy_gradient_net_uses_central_time_preprocessing():
    _assert_normalized_matches_raw_with_preprocessed_time(ToyGradientNet)


def test_toy_interaction_net_uses_central_time_preprocessing():
    _assert_normalized_matches_raw_with_preprocessed_time(ToyInteractionNet, 2, "silu")

