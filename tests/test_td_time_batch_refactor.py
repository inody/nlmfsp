import pytest
import torch
import torch.nn as nn

from nlmfsp.config import Config
from nlmfsp.fbsde import FBSDESolver


class RecordingAnalyticalCost(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(x.shape[0]))
        return torch.zeros(x.shape[0], 1, device=x.device)


def _make_data(config: Config):
    n_steps = config.n_time_steps
    xs = torch.randn(n_steps + 1, config.n_agents, config.dim, device=config.device)
    ts = config.time_grid
    dws = torch.randn(n_steps, config.n_agents, config.dim, device=config.device) * 0.1
    return xs, ts, dws


def test_td_uses_time_batch_aux_override():
    config = Config(dim=2, T=0.2, dt=0.1, n_agents=3, device="cpu")
    solver = FBSDESolver(config)
    solver.drift_net = None
    solver.cost_net = None
    solver.base_drift_fn = None
    solver.base_cost_fn = None

    xs, ts, dws = _make_data(config)

    _, z_aux_steps = solver.compute_td_loss(
        xs, ts, dws, direction="forward", return_aux=True
    )
    loss_steps = solver.compute_td_loss(
        xs, ts, dws, direction="forward", zs_aux_override=z_aux_steps
    )

    assert torch.isfinite(loss_steps)


def test_td_rejects_flattened_aux_override():
    config = Config(dim=2, T=0.2, dt=0.1, n_agents=3, device="cpu")
    solver = FBSDESolver(config)
    solver.drift_net = None
    solver.cost_net = None
    solver.base_drift_fn = None
    solver.base_cost_fn = None

    xs, ts, dws = _make_data(config)
    _, z_aux_steps = solver.compute_td_loss(
        xs, ts, dws, direction="forward", return_aux=True
    )
    z_aux_flat = z_aux_steps.reshape(-1, config.dim)

    with pytest.raises(ValueError, match="time-batch shape"):
        solver.compute_td_loss(
            xs, ts, dws, direction="forward", zs_aux_override=z_aux_flat
        )


def test_td_analytical_cost_cpu_eval_keeps_time_local_batches():
    config = Config(
        dim=2,
        T=0.2,
        dt=0.1,
        n_agents=3,
        device="cpu",
        evaluate_interaction_cpu=True,
    )
    solver = FBSDESolver(config)
    solver.drift_net = None
    solver.base_drift_fn = None
    solver.base_cost_fn = None
    solver.cost_net = RecordingAnalyticalCost()

    xs, ts, dws = _make_data(config)
    loss = solver.compute_td_loss(xs, ts, dws, direction="forward")

    assert torch.isfinite(loss)
    assert solver.cost_net.batch_sizes == [config.n_agents] * config.n_time_steps
