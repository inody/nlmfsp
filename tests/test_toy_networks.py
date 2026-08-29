"""Test script to verify ToyValueNet and ToyGradientNet parameter counts match DeepGSB."""

import torch
from src.nlmfsp.config import Config as _Config
from src.nlmfsp.networks import ToyGradientNet, ToyValueNet


def Config(**kwargs):
    kwargs.setdefault("n_agents", 4)
    return _Config(**kwargs)


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_toy_networks():
    """Test ToyValueNet and ToyGradientNet parameter counts."""

    # DeepGSB GMM navigation config
    config = Config(
        dim=2,
        hidden_dims=(128, 128, 128),
        t_hidden_dims=(128, 128),
        activation="silu",
        use_time_embedding=True,
        time_embed_dim=128,
    )

    # Create networks
    toy_value_net = ToyValueNet(config)
    toy_grad_net = ToyGradientNet(config)

    # Count parameters
    value_params = count_parameters(toy_value_net)
    grad_params = count_parameters(toy_grad_net)
    total_params = value_params + grad_params

    print("=" * 60)
    print("Toy Networks (DeepGSB ToyY/ToyZ architecture)")
    print("=" * 60)
    print(f"ToyValueNet parameters:    {value_params:,}")
    print(f"ToyGradientNet parameters: {grad_params:,}")
    print(f"Total parameters:          {total_params:,}")
    print()

    # Expected DeepGSB parameter counts (from reference implementation)
    expected_value = 115_969
    expected_grad = 83_202
    expected_total = 199_171

    print("DeepGSB reference (ToyY + ToyZ):")
    print(f"ToyY parameters:           {expected_value:,}")
    print(f"ToyZ parameters:           {expected_grad:,}")
    print(f"Total parameters:          {expected_total:,}")
    print()

    # Check if they match
    value_match = value_params == expected_value
    grad_match = grad_params == expected_grad
    total_match = total_params == expected_total

    print("Match status:")
    print(f"ToyValueNet:   {'✓' if value_match else '✗'} ({value_params - expected_value:+,} difference)")
    print(f"ToyGradientNet: {'✓' if grad_match else '✗'} ({grad_params - expected_grad:+,} difference)")
    print(f"Total:         {'✓' if total_match else '✗'} ({total_params - expected_total:+,} difference)")
    print()

    # Test forward pass
    print("=" * 60)
    print("Forward pass test")
    print("=" * 60)

    batch_size = 4
    x = torch.randn(batch_size, 2)
    t = torch.rand(batch_size)

    value_out = toy_value_net(x, t)
    grad_out = toy_grad_net(x, t)

    print(f"Input x shape:        {x.shape}")
    print(f"Input t shape:        {t.shape}")
    print(f"ToyValueNet output:   {value_out.shape} (expected: ({batch_size}, 1))")
    print(f"ToyGradientNet output: {grad_out.shape} (expected: ({batch_size}, 2))")

    if value_out.shape == (batch_size, 1) and grad_out.shape == (batch_size, 2):
        print("✓ Forward pass successful!")
    else:
        print("✗ Forward pass failed!")

    print()
    return total_match


if __name__ == "__main__":
    success = test_toy_networks()
    if success:
        print("✓ All tests passed! ToyValueNet and ToyGradientNet match DeepGSB architecture.")
    else:
        print("✗ Parameter counts do not match DeepGSB. Check network implementation.")
