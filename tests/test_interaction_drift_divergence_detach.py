import torch
from torch import nn

from nlmfsp.config import Config
from nlmfsp.fbsde import FBSDESolver


class ZeroValue(nn.Module):
    def forward(self, x, t):
        return x.new_zeros((x.shape[0], 1))


class ZeroGrad(nn.Module):
    def forward(self, x, t):
        return x * 0.0


class QuadraticDrift(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x, t):
        return self.scale * x.square()


def _make_solver(*, detach_interaction_drift_divergence: bool) -> FBSDESolver:
    config = Config(
        dim=1,
        T=0.2,
        dt=0.1,
        n_agents=4,
        sigma=1.0,
        seed=123,
        detach_interaction_drift_divergence=detach_interaction_drift_divergence,
    )
    return FBSDESolver(
        config,
        value_net=ZeroValue(),
        value_hat_net=ZeroValue(),
        grad_net=ZeroGrad(),
        grad_hat_net=ZeroGrad(),
        drift_net=QuadraticDrift(),
    )


def test_detach_interaction_drift_divergence_removes_learned_drift_jacobian_from_td():
    xs_traj = torch.tensor(
        [
            [[0.5], [1.0], [1.5], [2.0]],
            [[0.6], [1.1], [1.6], [2.1]],
            [[0.7], [1.2], [1.7], [2.2]],
        ],
        dtype=torch.float32,
    )
    ts_traj = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float32)
    dws_traj = torch.zeros((2, 4, 1), dtype=torch.float32)

    active_solver = _make_solver(detach_interaction_drift_divergence=False)
    detached_solver = _make_solver(detach_interaction_drift_divergence=True)

    active_loss = active_solver.compute_td_loss(
        xs_traj,
        ts_traj,
        dws_traj,
        direction="forward",
        trajectory_direction="forward",
    )
    detached_loss = detached_solver.compute_td_loss(
        xs_traj,
        ts_traj,
        dws_traj,
        direction="forward",
        trajectory_direction="forward",
    )

    assert active_loss.item() > 0.0
    assert detached_loss.item() == 0.0
