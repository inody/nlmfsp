"""Tests for networks with time embedding support."""

import torch

from src.nlmfsp.config import Config as _Config
from src.nlmfsp.networks import GradientNet, ValueNet


def Config(**kwargs):
    kwargs.setdefault("n_agents", 4)
    return _Config(**kwargs)


def test_value_net_without_time_embedding():
    """Test ValueNet with time embedding disabled."""
    config = Config(
        dim=2,
        hidden_dims=(64, 64, 64),
        activation="relu",
        use_time_embedding=False,
    )

    net = ValueNet(config)

    # Test forward pass
    x = torch.randn(10, 2)
    t = torch.rand(10)

    y = net(x, t)

    assert y.shape == (10, 1), f"Expected shape (10, 1), got {y.shape}"


def test_value_net_with_time_embedding():
    """Test ValueNet with time embedding enabled."""
    config = Config(
        dim=2,
        hidden_dims=(128, 128, 128, 128),
        activation="silu",
        use_time_embedding=True,
        time_embed_dim=128,
    )

    net = ValueNet(config)

    # Test forward pass
    x = torch.randn(10, 2)
    t = torch.rand(10)

    y = net(x, t)

    assert y.shape == (10, 1), f"Expected shape (10, 1), got {y.shape}"


def test_gradient_net_without_time_embedding():
    """Test GradientNet without embedding."""
    config = Config(
        dim=2,
        hidden_dims=(64, 64, 64),
        activation="relu",
        use_time_embedding=False,
    )

    net = GradientNet(config)

    # Test forward pass
    x = torch.randn(10, 2)
    t = torch.rand(10)

    z = net(x, t)

    assert z.shape == (10, 2), f"Expected shape (10, 2), got {z.shape}"


def test_gradient_net_with_time_embedding():
    """Test GradientNet with time embedding enabled."""
    config = Config(
        dim=2,
        hidden_dims=(128, 128, 128, 128),
        activation="silu",
        use_time_embedding=True,
        time_embed_dim=128,
    )

    net = GradientNet(config)

    # Test forward pass
    x = torch.randn(10, 2)
    t = torch.rand(10)

    z = net(x, t)

    assert z.shape == (10, 2), f"Expected shape (10, 2), got {z.shape}"


def test_time_embedding_influence_value_net():
    """Test that time embedding actually affects ValueNet output."""
    config = Config(
        dim=2,
        hidden_dims=(128, 128),
        activation="silu",
        use_time_embedding=True,
        time_embed_dim=128,
    )

    net = ValueNet(config)

    x = torch.randn(1, 2)
    t1 = torch.tensor([0.0])
    t2 = torch.tensor([1.0])

    y1 = net(x, t1)
    y2 = net(x, t2)

    # Different times should produce different outputs
    assert not torch.allclose(y1, y2), (
        "Different times should produce different outputs"
    )


def test_time_embedding_influence_gradient_net():
    """Test that time embedding actually affects GradientNet output."""
    config = Config(
        dim=2,
        hidden_dims=(128, 128),
        activation="silu",
        use_time_embedding=True,
        time_embed_dim=128,
    )

    net = GradientNet(config)

    x = torch.randn(1, 2)
    t1 = torch.tensor([0.0])
    t2 = torch.tensor([1.0])

    z1 = net(x, t1)
    z2 = net(x, t2)

    # Different times should produce different outputs
    assert not torch.allclose(z1, z2), (
        "Different times should produce different outputs"
    )


def test_value_net_scalar_time():
    """Test ValueNet with scalar time input."""
    config = Config(
        dim=2,
        hidden_dims=(64, 64),
        activation="relu",
        use_time_embedding=False,
    )

    net = ValueNet(config)

    x = torch.randn(5, 2)
    t = torch.tensor(0.5)  # Scalar time

    y = net(x, t)

    assert y.shape == (5, 1), f"Expected shape (5, 1), got {y.shape}"


def test_gradient_net_scalar_time():
    """Test GradientNet with scalar time input."""
    config = Config(
        dim=2,
        hidden_dims=(64, 64),
        activation="relu",
        use_time_embedding=False,
    )

    net = GradientNet(config)

    x = torch.randn(5, 2)
    t = torch.tensor(0.5)  # Scalar time

    z = net(x, t)

    assert z.shape == (5, 2), f"Expected shape (5, 2), got {z.shape}"


def test_value_net_different_embed_dims():
    """Test ValueNet with different embedding dimensions."""
    for embed_dim in [32, 64, 128, 256]:
        config = Config(
            dim=2,
            hidden_dims=(64, 64),
            activation="silu",
            use_time_embedding=True,
            time_embed_dim=embed_dim,
        )

        net = ValueNet(config)

        x = torch.randn(3, 2)
        t = torch.rand(3)

        y = net(x, t)

        assert y.shape == (3, 1), f"Expected shape (3, 1), got {y.shape}"


def test_gradient_net_different_embed_dims():
    """Test GradientNet with different embedding dimensions."""
    for embed_dim in [32, 64, 128, 256]:
        config = Config(
            dim=2,
            hidden_dims=(64, 64),
            activation="silu",
            use_time_embedding=True,
            time_embed_dim=embed_dim,
        )

        net = GradientNet(config)

        x = torch.randn(3, 2)
        t = torch.rand(3)

        z = net(x, t)

        assert z.shape == (3, 2), f"Expected shape (3, 2), got {z.shape}"


def test_value_net_gradient_flow():
    """Test that gradients flow through ValueNet with embedding."""
    config = Config(
        dim=2,
        hidden_dims=(64, 64),
        activation="silu",
        use_time_embedding=True,
        time_embed_dim=128,
    )

    net = ValueNet(config)

    x = torch.randn(5, 2, requires_grad=True)
    t = torch.rand(5, requires_grad=True)

    y = net(x, t)
    loss = y.sum()
    loss.backward()

    # Gradients should exist
    assert x.grad is not None, "Gradient w.r.t. x should exist"
    assert t.grad is not None, "Gradient w.r.t. t should exist"
    assert x.grad.abs().sum() > 0, "Gradient w.r.t. x should be non-zero"
    assert t.grad.abs().sum() > 0, "Gradient w.r.t. t should be non-zero"


def test_gradient_net_gradient_flow():
    """Test that gradients flow through GradientNet with embedding."""
    config = Config(
        dim=2,
        hidden_dims=(64, 64),
        activation="silu",
        use_time_embedding=True,
        time_embed_dim=128,
    )

    net = GradientNet(config)

    x = torch.randn(5, 2, requires_grad=True)
    t = torch.rand(5, requires_grad=True)

    z = net(x, t)
    loss = z.sum()
    loss.backward()

    # Gradients should exist
    assert x.grad is not None, "Gradient w.r.t. x should exist"
    assert t.grad is not None, "Gradient w.r.t. t should exist"
    assert x.grad.abs().sum() > 0, "Gradient w.r.t. x should be non-zero"
    assert t.grad.abs().sum() > 0, "Gradient w.r.t. t should be non-zero"


def test_deepgsb_config():
    """Test networks with DeepGSB-style configuration."""
    config = Config(
        dim=2,
        hidden_dims=(128, 128, 128, 128, 128, 128, 128, 128),  # 8 layers
        activation="silu",
        use_time_embedding=True,
        time_embed_dim=128,
    )

    value_net = ValueNet(config)
    grad_net = GradientNet(config)

    x = torch.randn(10, 2)
    t = torch.rand(10)

    y = value_net(x, t)
    z = grad_net(x, t)

    assert y.shape == (10, 1)
    assert z.shape == (10, 2)


def test_network_parameter_count():
    """Test parameter count for different configurations."""

    # Simple MLP (existing behavior)
    config_simple = Config(
        dim=2,
        hidden_dims=(64, 64, 64),
        activation="relu",
        use_time_embedding=False,
    )

    # DeepGSB style
    config_deepgsb = Config(
        dim=2,
        hidden_dims=(128, 128, 128, 128, 128, 128, 128, 128),
        activation="silu",
        use_time_embedding=True,
        time_embed_dim=128,
    )

    value_net_simple = ValueNet(config_simple)
    value_net_deepgsb = ValueNet(config_deepgsb)

    params_simple = sum(
        p.numel() for p in value_net_simple.parameters() if p.requires_grad
    )
    params_deepgsb = sum(
        p.numel() for p in value_net_deepgsb.parameters() if p.requires_grad
    )

    # DeepGSB should have significantly more parameters
    assert params_deepgsb > params_simple, (
        f"DeepGSB ({params_deepgsb}) should have more parameters "
        f"than simple MLP ({params_simple})"
    )

    print(f"Simple MLP parameters: {params_simple}")
    print(f"DeepGSB parameters: {params_deepgsb}")
