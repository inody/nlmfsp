"""Tests for timestep embedding function."""

import pytest
import torch

from src.nlmfsp.networks import timestep_embedding


def test_timestep_embedding_shape():
    """Test embedding output shape."""
    t = torch.tensor([0.0, 0.5, 1.0])
    emb = timestep_embedding(t, dim=128)

    assert emb.shape == (3, 128), f"Expected shape (3, 128), got {emb.shape}"


def test_timestep_embedding_even_dim():
    """Test that odd dimensions raise an error."""
    t = torch.tensor([0.0, 0.5, 1.0])

    with pytest.raises(ValueError, match="Embedding dimension must be even"):
        timestep_embedding(t, dim=127)


def test_timestep_embedding_uniqueness():
    """Test that different timesteps produce different embeddings."""
    t1 = torch.tensor([0.0, 0.5, 1.0])
    t2 = torch.tensor([0.1, 0.6, 1.1])

    emb1 = timestep_embedding(t1, dim=128)
    emb2 = timestep_embedding(t2, dim=128)

    # Embeddings should be different
    assert not torch.allclose(emb1, emb2), (
        "Different timesteps should produce different embeddings"
    )


def test_timestep_embedding_smoothness():
    """Test that embeddings change smoothly with time."""
    t = torch.linspace(0, 1, 100)
    emb = timestep_embedding(t, dim=128)

    # Consecutive embeddings should be similar
    diff = torch.norm(emb[1:] - emb[:-1], dim=-1)

    # Maximum difference should be reasonable (not too large)
    assert diff.max() < 0.5, f"Embeddings change too abruptly: max_diff={diff.max()}"


def test_timestep_embedding_batch_dimension():
    """Test embedding with 2D input."""
    t_1d = torch.tensor([0.0, 0.5, 1.0])
    t_2d = t_1d.unsqueeze(-1)  # (3, 1)

    emb_1d = timestep_embedding(t_1d, dim=128)
    emb_2d = timestep_embedding(t_2d, dim=128)

    assert torch.allclose(emb_1d, emb_2d), (
        "1D and 2D inputs should produce same embeddings"
    )


def test_timestep_embedding_zero():
    """Test embedding at t=0."""
    t = torch.tensor([0.0])
    emb = timestep_embedding(t, dim=128)

    assert emb.shape == (1, 128)
    # At t=0, cos terms should be 1, sin terms should be 0
    # First half is cos, second half is sin
    assert torch.allclose(emb[0, :64], torch.ones(64)), "cos(0) should be 1"
    assert torch.allclose(emb[0, 64:], torch.zeros(64), atol=1e-6), "sin(0) should be 0"


def test_timestep_embedding_device():
    """Test that embedding is on the correct device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        t = torch.tensor([0.0, 0.5, 1.0], device=device)
        emb = timestep_embedding(t, dim=128)

        assert emb.device == device, "Embedding should be on same device as input"


def test_timestep_embedding_different_dims():
    """Test embedding with different dimensions."""
    t = torch.tensor([0.5])

    for dim in [32, 64, 128, 256]:
        emb = timestep_embedding(t, dim=dim)
        assert emb.shape == (
            1,
            dim,
        ), f"Expected shape (1, {dim}), got {emb.shape}"


def test_timestep_embedding_negative_time():
    """Test embedding with negative time values."""
    t = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    emb = timestep_embedding(t, dim=128)

    assert emb.shape == (5, 128)
    # All embeddings should be different
    for i in range(5):
        for j in range(i + 1, 5):
            assert not torch.allclose(emb[i], emb[j]), (
                f"Embeddings at indices {i} and {j} should be different"
            )


def test_timestep_embedding_large_values():
    """Test embedding with large time values."""
    t = torch.tensor([0.0, 10.0, 100.0, 1000.0])
    emb = timestep_embedding(t, dim=128)

    assert emb.shape == (4, 128)
    # Should not contain NaN or Inf
    assert torch.isfinite(emb).all(), "Embedding should not contain NaN or Inf"


def test_timestep_embedding_gradient_flow():
    """Test that gradients flow through embedding."""
    t = torch.tensor([0.5], requires_grad=True)
    emb = timestep_embedding(t, dim=128)

    # Sum all values and backpropagate
    loss = emb.sum()
    loss.backward()

    # Gradient should exist and be non-zero
    assert t.grad is not None, "Gradient should exist"
    assert t.grad.abs() > 0, "Gradient should be non-zero"
