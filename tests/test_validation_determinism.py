import torch

from nlmfsp.config import Config
from nlmfsp.fbsde import FBSDESolver
from nlmfsp.validation import evaluate_transport_quality
from nlmfsp.visualization import sample_visualization_trajectories


class _DummyTrainer:
    def __init__(self, solver):
        self.solver = solver


def _sample_rho0(n: int) -> torch.Tensor:
    return torch.randn(n, 2)


def _sample_rho1(n: int) -> torch.Tensor:
    return torch.randn(n, 2) + 1.0


def test_evaluate_transport_quality_is_deterministic_with_seed():
    config = Config(
        dim=2,
        T=0.5,
        dt=0.25,
        n_agents=8,
        n_agents_test=8,
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
        device="cpu",
        seed=7,
    )
    solver = FBSDESolver(config)

    stats1 = evaluate_transport_quality(
        solver,
        _sample_rho0,
        _sample_rho1,
        config,
        n_test=8,
        seed=202,
    )
    stats2 = evaluate_transport_quality(
        solver,
        _sample_rho0,
        _sample_rho1,
        config,
        n_test=8,
        seed=202,
    )

    assert stats1 == stats2


def test_sample_visualization_trajectories_is_deterministic_with_seed():
    config = Config(
        dim=2,
        T=0.5,
        dt=0.25,
        n_agents=8,
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
        device="cpu",
        seed=7,
    )
    trainer = _DummyTrainer(FBSDESolver(config))

    xs_fwd_1, xs_bwd_1 = sample_visualization_trajectories(
        trainer,
        _sample_rho0,
        _sample_rho1,
        num_vis=8,
        seed=303,
    )
    xs_fwd_2, xs_bwd_2 = sample_visualization_trajectories(
        trainer,
        _sample_rho0,
        _sample_rho1,
        num_vis=8,
        seed=303,
    )

    assert torch.allclose(xs_fwd_1, xs_fwd_2)
    assert torch.allclose(xs_bwd_1, xs_bwd_2)
