"""Tests for neural networks."""

import torch

from nlmfsp.config import Config as _Config
from nlmfsp.networks import InteractionNet, ValueNet, GradientNet


def Config(**kwargs):
    kwargs.setdefault("n_agents", 4)
    return _Config(**kwargs)


def test_interaction_net():
    """Test InteractionNet."""
    config = Config(dim=3, hidden_dims=(32, 32))

    # Test f_θ (output_dim = dim)
    f_net = InteractionNet(config, output_dim=config.dim)
    x = torch.randn(10, config.dim)
    t = torch.tensor([0.5] * 10)

    output = f_net(x, t)
    assert output.shape == (10, config.dim)

    # Test F_φ (output_dim = 1)
    F_net = InteractionNet(config, output_dim=1)
    output = F_net(x, t)
    assert output.shape == (10, 1)

    # Test with scalar time
    t_scalar = torch.tensor(0.5)
    output = f_net(x, t_scalar)
    assert output.shape == (10, config.dim)

    print("✓ InteractionNet test passed")


def test_value_net():
    """Test ValueNet."""
    config = Config(dim=3, hidden_dims=(32, 32))
    net = ValueNet(config)

    x = torch.randn(10, config.dim)
    t = torch.tensor([0.5] * 10)

    output = net(x, t)
    assert output.shape == (10, 1)

    # Test with scalar time
    t_scalar = torch.tensor(0.5)
    output = net(x, t_scalar)
    assert output.shape == (10, 1)

    print("✓ ValueNet test passed")


def test_gradient_net():
    """Test GradientNet."""
    config = Config(dim=3, hidden_dims=(32, 32))
    net = GradientNet(config)

    x = torch.randn(10, config.dim)
    t = torch.tensor([0.5] * 10)

    output = net(x, t)
    assert output.shape == (10, config.dim)

    # Test with scalar time
    t_scalar = torch.tensor(0.5)
    output = net(x, t_scalar)
    assert output.shape == (10, config.dim)

    print("✓ GradientNet test passed")


def test_gradient_computation():
    """Test gradient computation through networks."""
    config = Config(dim=2, hidden_dims=(16,))
    net = ValueNet(config)

    x = torch.randn(5, config.dim, requires_grad=True)
    t = torch.tensor([0.5] * 5)

    y = net(x, t)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape

    print("✓ Gradient computation test passed")


if __name__ == "__main__":
    print("\n=== Testing Neural Networks ===\n")
    test_interaction_net()
    test_value_net()
    test_gradient_net()
    test_gradient_computation()
    print("\n=== All network tests passed! ===\n")
