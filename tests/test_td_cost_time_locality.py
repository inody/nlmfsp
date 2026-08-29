import torch
import torch.nn as nn

from nlmfsp.config import Config
from nlmfsp.fbsde import FBSDESolver


class RecordingAnalyticalCost(nn.Module):
    """Parameter-free cost module that records batch sizes."""

    def __init__(self):
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(x.shape[0]))
        return torch.zeros(x.shape[0], 1, device=x.device)


class RecordingLearnedCost(nn.Module):
    """Trainable cost module that records batch sizes."""

    def __init__(self, dim: int):
        super().__init__()
        self.batch_sizes: list[int] = []
        self.proj = nn.Linear(dim + 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(x.shape[0]))
        xt = torch.cat([x, t.unsqueeze(-1)], dim=-1)
        return self.proj(xt)


def _make_data(config: Config):
    n_steps = config.n_time_steps
    batch = config.n_agents or 1
    xs = torch.randn(n_steps + 1, batch, config.dim, device=config.device)
    ts = config.time_grid
    dws = torch.randn(n_steps, batch, config.dim, device=config.device) * 0.1
    return xs, ts, dws


def test_td_analytical_cost_is_evaluated_per_time_slice():
    config = Config(dim=2, T=0.2, dt=0.1, n_agents=3, device="cpu")
    solver = FBSDESolver(config)
    solver.drift_net = None
    solver.base_drift_fn = None
    solver.base_cost_fn = None
    analytical_cost = RecordingAnalyticalCost()
    solver.cost_net = analytical_cost

    xs, ts, dws = _make_data(config)
    _ = solver.compute_td_loss(xs, ts, dws, direction="forward")

    assert len(analytical_cost.batch_sizes) == config.n_time_steps
    assert all(sz == config.n_agents for sz in analytical_cost.batch_sizes)


def test_td_learned_cost_remains_flattened_batch_evaluation():
    config = Config(dim=2, T=0.2, dt=0.1, n_agents=3, device="cpu")
    solver = FBSDESolver(config)
    solver.drift_net = None
    solver.base_drift_fn = None
    solver.base_cost_fn = None
    learned_cost = RecordingLearnedCost(config.dim)
    solver.cost_net = learned_cost

    xs, ts, dws = _make_data(config)
    _ = solver.compute_td_loss(xs, ts, dws, direction="forward")

    expected = config.n_time_steps * (config.n_agents or 1)
    assert learned_cost.batch_sizes
    assert learned_cost.batch_sizes[0] == expected


def test_td_analytical_cost_override_ignores_stale_trainable_state():
    config = Config(dim=2, T=0.2, dt=0.1, n_agents=3, device="cpu")
    learned_cost = RecordingLearnedCost(config.dim)
    solver = FBSDESolver(config, cost_net=learned_cost)
    solver.drift_net = None
    solver.base_drift_fn = None
    solver.base_cost_fn = None

    analytical_cost = RecordingAnalyticalCost()
    solver.cost_net = analytical_cost

    xs, ts, dws = _make_data(config)
    _ = solver.compute_td_loss(xs, ts, dws, direction="forward")

    assert len(analytical_cost.batch_sizes) == config.n_time_steps
    assert all(sz == config.n_agents for sz in analytical_cost.batch_sizes)
