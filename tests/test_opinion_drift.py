import torch

from nlmfsp.config import Config
from nlmfsp.drifts import (
    OpinionMeanFieldDrift,
    build_opinion_f_mul,
    build_opinion_xi_schedule,
)
from nlmfsp.interaction import create_interactions


def test_opinion_drift_forward_no_nan():
    dim = 4
    n_time_steps = 10
    n_grid_points = n_time_steps + 1
    xi = build_opinion_xi_schedule(
        n_steps=n_grid_points, dim=dim, seed=0, dt=0.1
    )
    f_mul = build_opinion_f_mul(n_steps=n_grid_points, T=1.0)
    drift = OpinionMeanFieldDrift(
        xi_schedule=xi,
        f_mul=f_mul,
        dt=0.1,
        T=1.0,
        polarize_strength=1.0,
    )

    # Single time index
    x = torch.randn(16, dim)
    t = torch.zeros(16)
    out = drift(x, t)
    assert torch.isfinite(out).all()
    assert out.shape == x.shape

    # Multiple time indices
    t_steps = torch.linspace(0, 1.0, n_grid_points)
    t_idx = t_steps.repeat_interleave(2)[: x.shape[0]]
    out2 = drift(x, t_idx)
    assert torch.isfinite(out2).all()
    assert out2.shape == x.shape


def test_opinion_drift_uses_l2_normalized_polarization():
    xi = torch.tensor([[1.0, 0.0]])
    f_mul = torch.ones(1)
    drift = OpinionMeanFieldDrift(
        xi_schedule=xi,
        f_mul=f_mul,
        dt=1.0,
        T=0.0,
        polarize_strength=1.0,
    )
    x = torch.tensor([[2.0, 0.0], [1.0, 0.0]])
    t = torch.zeros(2)

    out = drift(x, t)

    expected = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    torch.testing.assert_close(out, expected)


def test_opinion_drift_batched_matches_time_slice_loop():
    torch.manual_seed(321)
    dim = 4
    n_time_steps = 4
    n_grid_points = n_time_steps + 1
    xi = build_opinion_xi_schedule(
        n_steps=n_grid_points, dim=dim, seed=123, dt=0.1
    )
    f_mul = build_opinion_f_mul(n_steps=n_grid_points, T=0.4)
    drift = OpinionMeanFieldDrift(
        xi_schedule=xi,
        f_mul=f_mul,
        dt=0.1,
        T=0.4,
        polarize_strength=1.0,
    )
    x = torch.randn(n_grid_points, 6, dim)
    time_grid = torch.linspace(0.0, 0.4, n_grid_points)

    expected = torch.stack(
        [
            drift.evaluate(x_t, x_t, t.expand(x_t.shape[0]))
            for x_t, t in zip(x, time_grid)
        ],
        dim=0,
    )
    actual = drift.evaluate_batched(x, x, time_grid=time_grid, time_block_size=2)

    torch.testing.assert_close(actual, expected)


def test_build_opinion_xi_schedule_preserves_global_rng_state():
    torch.manual_seed(123)
    state_before = torch.random.get_rng_state()

    _ = build_opinion_xi_schedule(n_steps=10, dim=4, seed=999)

    state_after = torch.random.get_rng_state()
    assert torch.equal(state_before, state_after)


def test_create_interactions_opinion_schedule_matches_solver_time_grid():
    config = Config(
        dim=3,
        T=0.4,
        dt=0.1,
        n_agents=4,
        drift_type="opinion",
        cost_type="zero",
        device="cpu",
    )

    drift, _, _, _, _, _ = create_interactions(config)

    assert drift.xi_schedule.shape[0] == config.n_time_steps + 1
    assert drift.f_mul.shape[0] == config.n_time_steps + 1
    idx = drift._time_to_index(config.time_grid)
    torch.testing.assert_close(idx, torch.arange(config.n_time_steps + 1))
    assert drift.f_mul[-2] > drift.f_mul[-1]
