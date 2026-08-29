"""Tests for TD loss computation on the current time-batch implementation."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nlmfsp.config import Config  # noqa: E402
from nlmfsp.fbsde import FBSDESolver  # noqa: E402


class DummyNetwork(nn.Module):
    """Simple network for testing."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 32), nn.Tanh(), nn.Linear(32, dim if dim > 1 else 1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_expanded = t.unsqueeze(-1)
        xt = torch.cat([x, t_expanded], dim=-1)
        return self.net(xt)


@pytest.fixture
def config():
    """Create test configuration."""
    return Config(
        dim=2,
        n_agents=16,
        dt=0.02,
        T=1.0,
        sigma=0.5,
        n_hutchinson_samples=1,
        td_horizon=1,
        device="cpu",
    )


@pytest.fixture
def solver(config):
    """Create FBSDE solver."""
    solver = FBSDESolver(config)
    solver.drift_net = None
    solver.cost_net = None
    solver.base_drift_fn = None
    solver.base_cost_fn = None
    return solver


@pytest.fixture
def trajectory_data(config):
    """Create sample trajectory data."""
    n_steps = config.n_time_steps
    batch_size = config.n_agents
    dim = config.dim

    xs = torch.randn(n_steps + 1, batch_size, dim)
    ts = config.time_grid
    dws = torch.randn(n_steps, batch_size, dim) * 0.1

    return xs, ts, dws


def test_td_single_step_forward(solver, trajectory_data):
    xs, ts, dws = trajectory_data
    torch.manual_seed(42)
    loss = solver.compute_td_loss(xs, ts, dws, direction="forward")
    assert loss.ndim == 0
    assert loss.requires_grad
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_td_single_step_backward(solver, trajectory_data):
    xs, ts, dws = trajectory_data
    torch.manual_seed(42)
    loss = solver.compute_td_loss(xs, ts, dws, direction="backward")
    assert loss.ndim == 0
    assert loss.requires_grad
    assert not torch.isnan(loss)


def test_td_gradient_flow(solver, trajectory_data):
    xs, ts, dws = trajectory_data
    loss = solver.compute_td_loss(xs, ts, dws, direction="forward")
    loss.backward()
    assert any(param.grad is not None for param in solver.value_net.parameters())


def test_td_numerical_stability(solver):
    n_steps = solver.config.n_time_steps
    batch_size = solver.config.n_agents
    dim = solver.config.dim
    ts = solver.config.time_grid

    for scale in [0.01, 0.1, 1.0, 10.0]:
        xs = torch.randn(n_steps + 1, batch_size, dim) * scale
        dws = torch.randn(n_steps, batch_size, dim) * scale * 0.1
        loss = solver.compute_td_loss(xs, ts, dws, direction="forward")
        assert not torch.isnan(loss), f"NaN with scale {scale}"
        assert not torch.isinf(loss), f"Inf with scale {scale}"


def test_td_deterministic(solver, trajectory_data):
    xs, ts, dws = trajectory_data
    torch.manual_seed(123)
    loss1 = solver.compute_td_loss(xs, ts, dws, direction="forward")
    torch.manual_seed(123)
    loss2 = solver.compute_td_loss(xs, ts, dws, direction="forward")
    assert torch.allclose(loss1, loss2, rtol=1e-6, atol=1e-8)


def test_td_multi_step_horizon_5(solver, trajectory_data):
    solver.config.td_horizon = 5
    xs, ts, dws = trajectory_data
    torch.manual_seed(42)
    loss = solver.compute_td_loss(xs, ts, dws, direction="forward")
    assert loss.ndim == 0
    assert loss.requires_grad
    assert not torch.isnan(loss)


def test_td_multi_step_horizon_10(solver, trajectory_data):
    solver.config.td_horizon = 10
    xs, ts, dws = trajectory_data
    torch.manual_seed(42)
    loss = solver.compute_td_loss(xs, ts, dws, direction="forward")
    assert loss.ndim == 0
    assert loss.requires_grad
    assert not torch.isnan(loss)


def test_td_with_interaction(config):
    solver = FBSDESolver(config)
    solver.drift_net = DummyNetwork(config.dim)
    solver.cost_net = None
    solver.base_drift_fn = None
    solver.base_cost_fn = None

    n_steps = config.n_time_steps
    batch_size = config.n_agents
    dim = config.dim
    xs = torch.randn(n_steps + 1, batch_size, dim)
    ts = config.time_grid
    dws = torch.randn(n_steps, batch_size, dim) * 0.1

    loss = solver.compute_td_loss(xs, ts, dws, direction="forward")
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_td_with_cost(config):
    class CostNetwork(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 1, 16),
                nn.Tanh(),
                nn.Linear(16, 1),
            )

        def forward(self, x, t):
            t_expanded = t.unsqueeze(-1)
            xt = torch.cat([x, t_expanded], dim=-1)
            return self.net(xt)

    solver = FBSDESolver(config)
    solver.drift_net = None
    solver.cost_net = CostNetwork(config.dim)
    solver.base_drift_fn = None
    solver.base_cost_fn = None

    n_steps = config.n_time_steps
    batch_size = config.n_agents
    dim = config.dim
    xs = torch.randn(n_steps + 1, batch_size, dim)
    ts = config.time_grid
    dws = torch.randn(n_steps, batch_size, dim) * 0.1

    loss = solver.compute_td_loss(xs, ts, dws, direction="forward")
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_td_with_base_cost(config, solver, trajectory_data):
    def base_cost(x, t):
        return 0.1 * (x**2).sum(dim=-1)

    solver.base_cost_fn = base_cost
    xs, ts, dws = trajectory_data
    loss = solver.compute_td_loss(xs, ts, dws, direction="forward")
    assert loss.ndim == 0
    assert not torch.isnan(loss)


@pytest.mark.parametrize(
    "batch_size,n_steps",
    [
        (1, 10),
        (8, 20),
        (32, 50),
        (128, 50),
    ],
)
def test_td_various_sizes(batch_size, n_steps):
    config = Config(
        dim=2,
        n_agents=batch_size,
        dt=1.0 / n_steps,
        T=1.0,
        sigma=0.5,
        n_hutchinson_samples=1,
        td_horizon=1,
        device="cpu",
    )

    solver = FBSDESolver(config)
    solver.drift_net = None
    solver.cost_net = None
    solver.base_drift_fn = None
    solver.base_cost_fn = None

    xs = torch.randn(n_steps + 1, batch_size, config.dim)
    ts = config.time_grid
    dws = torch.randn(n_steps, batch_size, config.dim) * 0.1
    loss = solver.compute_td_loss(xs, ts, dws, direction="forward")
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_td_return_aux_matches_network_output_shape(solver, trajectory_data):
    xs, ts, dws = trajectory_data
    loss, z_aux = solver.compute_td_loss(
        xs, ts, dws, direction="forward", return_aux=True
    )
    expected_shape = (
        solver.config.n_time_steps,
        solver.config.n_agents,
        solver.config.dim,
    )
    assert loss.ndim == 0
    assert z_aux.shape == expected_shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
