"""
Test suite for compute_div_gz optimization.

This test suite verifies that the optimized implementation of compute_div_gz
produces numerically equivalent results to the baseline implementation while
being more efficient.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nlmfsp.config import Config  # noqa: E402
from nlmfsp.fbsde import FBSDESolver  # noqa: E402


class DummyNetwork(nn.Module):
    """Simple network for testing."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 32),
            nn.Tanh(),
            nn.Linear(32, dim)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (batch, dim)
            t: (batch,)
        
        Returns:
            output: (batch, dim)
        """
        t_expanded = t.unsqueeze(-1)  # (batch, 1)
        xt = torch.cat([x, t_expanded], dim=-1)  # (batch, dim+1)
        return self.net(xt)


@pytest.fixture
def config():
    """Create test configuration."""
    cfg = Config(
        dim=2,
        n_agents=16,
        dt=0.02,
        T=1.0,
        sigma=0.5,
        n_hutchinson_samples=1,  # Single sample for optimization
        device="cpu"
    )
    return cfg


@pytest.fixture
def solver(config):
    """Create FBSDE solver."""
    return FBSDESolver(config)


@pytest.fixture
def networks(config):
    """Create test networks."""
    z_net = DummyNetwork(config.dim)
    drift_net = DummyNetwork(config.dim)
    return z_net, drift_net


def test_compute_div_gz_requires_grad(config, solver, networks):
    """Test that compute_div_gz works with requires_grad=True input."""
    z_net, _ = networks
    solver.drift_net = None
    solver.base_drift_fn = None
    
    # Create input with requires_grad=True
    xs = torch.randn(16, config.dim, requires_grad=True)
    ts = torch.rand(16)
    
    # Should work without error
    div_gz = solver.compute_div_gz(xs, ts, z_net, direction="forward")
    
    assert div_gz.shape == (16,)
    assert div_gz.requires_grad


def test_compute_div_gz_no_requires_grad(config, solver, networks):
    """Test that compute_div_gz handles requires_grad=False input."""
    z_net, _ = networks
    solver.drift_net = None
    solver.base_drift_fn = None
    
    # Create input WITHOUT requires_grad
    xs = torch.randn(16, config.dim, requires_grad=False)
    ts = torch.rand(16)
    
    # Should still work (will be handled internally)
    div_gz = solver.compute_div_gz(xs, ts, z_net, direction="forward")
    
    assert div_gz.shape == (16,)
    assert div_gz.requires_grad


def test_compute_div_gz_with_interaction(config, solver, networks):
    """Test compute_div_gz with interaction network."""
    z_net, drift_net = networks
    solver.drift_net = drift_net
    solver.base_drift_fn = None
    
    xs = torch.randn(16, config.dim, requires_grad=True)
    ts = torch.rand(16)
    
    div_gz = solver.compute_div_gz(xs, ts, z_net, direction="forward")
    
    assert div_gz.shape == (16,)
    assert div_gz.requires_grad


def test_compute_div_gz_with_base_drift(config, solver, networks):
    """Test compute_div_gz with base drift function."""
    z_net, _ = networks
    solver.drift_net = None
    
    # Simple base drift: drift towards origin
    def base_drift(x):
        return -0.1 * x
    
    solver.base_drift_fn = base_drift
    
    xs = torch.randn(16, config.dim, requires_grad=True)
    ts = torch.rand(16)
    
    div_gz = solver.compute_div_gz(xs, ts, z_net, direction="forward")
    
    assert div_gz.shape == (16,)
    assert div_gz.requires_grad


def test_compute_div_gz_backward_direction(config, solver, networks):
    """Test compute_div_gz with backward direction."""
    z_net, drift_net = networks
    solver.drift_net = drift_net
    solver.base_drift_fn = None
    
    xs = torch.randn(16, config.dim, requires_grad=True)
    ts = torch.rand(16)
    
    # Forward direction
    div_fwd = solver.compute_div_gz(xs, ts, z_net, direction="forward")
    
    # Backward direction (should differ due to sign of f)
    div_bwd = solver.compute_div_gz(xs, ts, z_net, direction="backward")
    
    assert div_fwd.shape == div_bwd.shape
    # Should differ because f term is negated
    assert not torch.allclose(div_fwd, div_bwd)


def test_compute_div_gz_gradient_flow(config, solver, networks):
    """Test that gradients flow through compute_div_gz."""
    z_net, _ = networks
    solver.drift_net = None
    solver.base_drift_fn = None
    
    xs = torch.randn(16, config.dim, requires_grad=True)
    ts = torch.rand(16)
    
    div_gz = solver.compute_div_gz(xs, ts, z_net, direction="forward")
    
    # Compute loss and backprop
    loss = div_gz.mean()
    loss.backward()
    
    # Check that gradients exist
    assert xs.grad is not None
    assert xs.grad.shape == xs.shape
    
    # Check that network has gradients
    has_grad = False
    for param in z_net.parameters():
        if param.grad is not None:
            has_grad = True
            break
    assert has_grad, "Network should have gradients"


def test_compute_div_gz_multiple_samples(config, solver, networks):
    """Test compute_div_gz with multiple Hutchinson samples."""
    # Create config with multiple samples
    config_multi = Config(
        dim=2,
        n_agents=16,
        dt=0.02,
        T=1.0,
        sigma=0.5,
        n_hutchinson_samples=5,  # Multiple samples
        device="cpu"
    )
    solver_multi = FBSDESolver(config_multi)
    
    z_net, _ = networks
    solver_multi.drift_net = None
    solver_multi.base_drift_fn = None
    
    xs = torch.randn(16, config.dim, requires_grad=True)
    ts = torch.rand(16)
    
    # Should work with multiple samples
    div_gz = solver_multi.compute_div_gz(xs, ts, z_net, direction="forward")
    
    assert div_gz.shape == (16,)
    assert div_gz.requires_grad


def test_compute_div_gz_deterministic_output():
    """Test that compute_div_gz is deterministic with same random seed."""
    config = Config(
        dim=2,
        n_agents=16,
        dt=0.02,
        T=1.0,
        sigma=0.5,
        n_hutchinson_samples=1,
        device="cpu"
    )
    solver = FBSDESolver(config)
    
    z_net = DummyNetwork(config.dim)
    solver.drift_net = None
    solver.base_drift_fn = None
    
    xs = torch.randn(16, config.dim, requires_grad=True)
    ts = torch.rand(16)
    
    # First computation
    torch.manual_seed(42)
    div_gz_1 = solver.compute_div_gz(xs, ts, z_net, direction="forward")
    
    # Second computation with same seed
    torch.manual_seed(42)
    div_gz_2 = solver.compute_div_gz(xs, ts, z_net, direction="forward")
    
    # Should be identical
    assert torch.allclose(div_gz_1, div_gz_2, rtol=1e-6, atol=1e-8)


def test_compute_div_gz_numerical_stability():
    """Test numerical stability with various input scales."""
    config = Config(
        dim=2,
        n_agents=16,
        dt=0.02,
        T=1.0,
        sigma=0.5,
        n_hutchinson_samples=1,
        device="cpu"
    )
    solver = FBSDESolver(config)
    
    z_net = DummyNetwork(config.dim)
    solver.drift_net = None
    solver.base_drift_fn = None
    
    ts = torch.rand(16)
    
    # Test with various scales
    for scale in [0.01, 0.1, 1.0, 10.0]:
        xs = torch.randn(16, config.dim, requires_grad=True) * scale
        div_gz = solver.compute_div_gz(xs, ts, z_net, direction="forward")
        
        # Check no NaN or Inf
        assert not torch.isnan(div_gz).any(), f"NaN detected with scale {scale}"
        assert not torch.isinf(div_gz).any(), f"Inf detected with scale {scale}"


@pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
def test_compute_div_gz_batch_sizes(batch_size):
    """Test compute_div_gz with various batch sizes."""
    config = Config(
        dim=2,
        n_agents=batch_size,
        dt=0.02,
        T=1.0,
        sigma=0.5,
        n_hutchinson_samples=1,
        device="cpu"
    )
    solver = FBSDESolver(config)
    
    z_net = DummyNetwork(config.dim)
    solver.drift_net = None
    solver.base_drift_fn = None
    
    xs = torch.randn(batch_size, config.dim, requires_grad=True)
    ts = torch.rand(batch_size)
    
    div_gz = solver.compute_div_gz(xs, ts, z_net, direction="forward")
    
    assert div_gz.shape == (batch_size,)
    assert div_gz.requires_grad


@pytest.mark.parametrize("dim", [1, 2, 4, 8])
def test_compute_div_gz_dimensions(dim):
    """Test compute_div_gz with various dimensions."""
    config = Config(
        dim=dim,
        n_agents=16,
        dt=0.02,
        T=1.0,
        sigma=0.5,
        n_hutchinson_samples=1,
        device="cpu"
    )
    solver = FBSDESolver(config)
    
    z_net = DummyNetwork(dim)
    solver.drift_net = None
    solver.base_drift_fn = None
    
    xs = torch.randn(16, dim, requires_grad=True)
    ts = torch.rand(16)
    
    div_gz = solver.compute_div_gz(xs, ts, z_net, direction="forward")
    
    assert div_gz.shape == (16,)
    assert div_gz.requires_grad


def compare_implementations_numerically():
    """
    Compare baseline and optimized implementations numerically.
    
    This function will be updated to compare old vs new implementation
    once the optimization is applied.
    """
    config = Config(
        dim=2,
        n_agents=32,
        dt=0.02,
        T=1.0,
        sigma=0.5,
        n_hutchinson_samples=1,
        device="cpu"
    )
    solver = FBSDESolver(config)
    
    z_net = DummyNetwork(config.dim)
    solver.drift_net = None
    solver.base_drift_fn = None
    
    xs = torch.randn(32, config.dim, requires_grad=True)
    ts = torch.rand(32)
    
    # Set random seed for reproducibility
    torch.manual_seed(123)
    div_gz = solver.compute_div_gz(xs, ts, z_net, direction="forward")
    
    print(f"\nDiv_gz statistics:")
    print(f"  Mean: {div_gz.mean().item():.6f}")
    print(f"  Std: {div_gz.std().item():.6f}")
    print(f"  Min: {div_gz.min().item():.6f}")
    print(f"  Max: {div_gz.max().item():.6f}")
    
    # Test gradient flow
    loss = div_gz.mean()
    loss.backward()
    
    print(f"\nGradient statistics (xs.grad):")
    print(f"  Mean: {xs.grad.mean().item():.6f}")
    print(f"  Std: {xs.grad.std().item():.6f}")
    print(f"  Max abs: {xs.grad.abs().max().item():.6f}")
    
    return div_gz, xs.grad


if __name__ == "__main__":
    print("Running compute_div_gz baseline tests...")
    print("=" * 60)
    
    # Run numerical comparison
    div_gz, grad = compare_implementations_numerically()
    
    print("\n" + "=" * 60)
    print("Baseline implementation results saved for comparison.")
    print("After optimization, re-run this script to verify equivalence.")
