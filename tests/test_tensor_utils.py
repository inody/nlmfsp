"""Tests for utility functions."""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nlmfsp import tensor_utils  # noqa: E402


def test_flatten_dim01_basic():
    """Test basic flatten operation."""
    x = torch.randn(4, 5, 2)  # (batch, time, dim)
    x_flat = tensor_utils.flatten_dim01(x)

    assert x_flat.shape == (20, 2)  # batch * time, dim

    # Check data integrity
    assert torch.allclose(x[0, 0], x_flat[0])
    assert torch.allclose(x[0, 1], x_flat[1])
    assert torch.allclose(x[1, 0], x_flat[5])


def test_unflatten_dim01_basic():
    """Test basic unflatten operation."""
    x_flat = torch.randn(20, 2)  # (batch*time, dim)
    x = tensor_utils.unflatten_dim01(x_flat, [4, 5])

    assert x.shape == (4, 5, 2)  # batch, time, dim

    # Check data integrity
    assert torch.allclose(x_flat[0], x[0, 0])
    assert torch.allclose(x_flat[1], x[0, 1])
    assert torch.allclose(x_flat[5], x[1, 0])


def test_flatten_unflatten_roundtrip():
    """Test that flatten/unflatten are inverses."""
    x_original = torch.randn(4, 5, 2, 3)  # Multiple trailing dimensions

    # Flatten and unflatten
    x_flat = tensor_utils.flatten_dim01(x_original)
    x_restored = tensor_utils.unflatten_dim01(x_flat, [4, 5])

    assert x_restored.shape == x_original.shape
    assert torch.allclose(x_restored, x_original)


def test_reverse_cumsum():
    """Test reverse cumulative sum."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    result = tensor_utils.reverse_cumsum(x, dim=1)

    # Expected: cumsum from right to left
    # Row 0: [1+2+3=6, 2+3=5, 3=3]
    # Row 1: [4+5+6=15, 5+6=11, 6=6]
    expected = torch.tensor([[6.0, 5.0, 3.0], [15.0, 11.0, 6.0]])

    assert torch.allclose(result, expected)


def test_reverse_cumsum_vs_cumsum():
    """Test relationship between forward and reverse cumsum."""
    x = torch.randn(3, 10)

    forward = torch.cumsum(x, dim=1)
    reverse = tensor_utils.reverse_cumsum(x, dim=1)

    # Last element should be same
    assert torch.allclose(forward[:, -1], reverse[:, 0])

    # Reverse should equal flipped cumsum of flipped input
    expected = torch.flip(torch.cumsum(torch.flip(x, dims=[1]), dim=1), dims=[1])
    assert torch.allclose(reverse, expected)


def test_batch_trajectory_forward():
    """Test trajectory batching for forward processing."""
    xs = torch.randn(51, 128, 2)  # (T+1, batch, dim)
    ts = torch.linspace(0, 1, 51)

    xs_flat, ts_flat = tensor_utils.batch_trajectory_forward(xs, ts, include_boundary=False)

    # Should exclude boundary -> 50 steps
    assert xs_flat.shape == (50 * 128, 2)  # (T*B, dim)
    assert ts_flat.shape == (50 * 128,)

    # Check time repetition
    assert torch.allclose(ts_flat[:128], ts[0].expand(128))
    assert torch.allclose(ts_flat[128:256], ts[1].expand(128))


def test_batch_trajectory_forward_with_boundary():
    """Test trajectory batching including boundary."""
    xs = torch.randn(51, 128, 2)
    ts = torch.linspace(0, 1, 51)

    xs_flat, ts_flat = tensor_utils.batch_trajectory_forward(xs, ts, include_boundary=True)

    # Should include boundary -> 51 steps
    assert xs_flat.shape == (51 * 128, 2)
    assert ts_flat.shape == (51 * 128,)


def test_unflatten_trajectory():
    """Test trajectory unflattening."""
    x_flat = torch.randn(6400, 2)  # 50 * 128

    x_unflat = tensor_utils.unflatten_trajectory(x_flat, batch_size=128, n_steps=50)

    assert x_unflat.shape == (128, 50, 2)  # (B, T, dim)

    # Check data integrity
    assert torch.allclose(x_flat[:50], x_unflat[0].reshape(-1, 2))


@pytest.mark.parametrize(
    "batch_size,n_steps,dim",
    [
        (1, 10, 2),
        (16, 50, 2),
        (128, 50, 4),
        (256, 100, 8),
    ],
)
def test_flatten_unflatten_various_shapes(batch_size, n_steps, dim):
    """Test flatten/unflatten with various shapes."""
    x = torch.randn(batch_size, n_steps, dim)

    x_flat = tensor_utils.flatten_dim01(x)
    assert x_flat.shape == (batch_size * n_steps, dim)

    x_restored = tensor_utils.unflatten_dim01(x_flat, [batch_size, n_steps])
    assert x_restored.shape == x.shape
    assert torch.allclose(x_restored, x)


def test_flatten_with_extra_dimensions():
    """Test flatten with more than 3 dimensions."""
    x = torch.randn(4, 5, 2, 3, 7)  # (d0, d1, d2, d3, d4)

    x_flat = tensor_utils.flatten_dim01(x)
    assert x_flat.shape == (20, 2, 3, 7)  # (d0*d1, d2, d3, d4)

    x_restored = tensor_utils.unflatten_dim01(x_flat, [4, 5])
    assert x_restored.shape == x.shape
    assert torch.allclose(x_restored, x)


def test_reverse_cumsum_gradient_flow():
    """Test that gradients flow through reverse_cumsum."""
    x = torch.randn(3, 10, requires_grad=True)

    result = tensor_utils.reverse_cumsum(x, dim=1)
    loss = result.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
