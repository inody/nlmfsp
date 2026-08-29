"""Basic tests for Trainer class.

These are smoke tests to verify that the Trainer can be initialized
and basic methods run without errors.
"""

from pathlib import Path

import pytest
import torch
import torch.distributions as dist

import nlmfsp.trainer as trainer_module
from nlmfsp.analytical_interactions import AnalyticalZeroCost, AnalyticalZeroDrift
from nlmfsp.config import Config
from nlmfsp.fbsde import FBSDESolver
from nlmfsp.interaction import create_interactions
from nlmfsp.rng_utils import seed_offset
from nlmfsp.trainer import Trainer


def test_trainer_initialization():
    """Test that Trainer can be initialized."""
    config = Config(
        dim=2,
        T=1.0,
        dt=0.2,  # T/dt = 5 steps
        n_agents=16,
        n_outer_iterations=2,
        verbose=False,
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
    )

    solver = FBSDESolver(config)

    # Define simple log-prob functions
    def log_p0(x):
        return -0.5 * (x**2).sum(dim=-1)

    def log_p1(x):
        return -0.5 * ((x - 1.0) ** 2).sum(dim=-1)

    trainer = Trainer(config, solver, log_p0, log_p1)

    # Check that optimizers are initialized
    assert trainer.opt_forward is not None
    assert trainer.opt_backward is not None

    # Check that schedulers are initialized
    assert trainer.scheduler_forward is not None
    assert trainer.scheduler_backward is not None

    # Check that metrics dict is initialized
    assert "loss_ipf_forward" in trainer.metrics
    assert "loss_td_backward" in trainer.metrics
    assert "loss_ipf_backward" in trainer.metrics
    assert "loss_td_forward" in trainer.metrics
    assert "loss_boundary_forward" in trainer.metrics
    assert "loss_boundary_backward" in trainer.metrics
    assert "loss_ipf_forward_analytical" in trainer.metrics
    assert "loss_ipf_backward_analytical" in trainer.metrics

    print("✓ Trainer initialization test passed")


def test_backward_drift_step():
    """Test backward drift step runs without error."""
    config = Config(
        dim=2,
        T=0.5,
        dt=0.25,  # T/dt = 2 steps
        n_agents=8,
        verbose=False,
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
    )

    solver = FBSDESolver(config)

    def log_p0(x):
        return -0.5 * (x**2).sum(dim=-1)

    def log_p1(x):
        return -0.5 * ((x - 1.0) ** 2).sum(dim=-1)

    trainer = Trainer(config, solver, log_p0, log_p1)

    # Define sampling function
    rho_0 = dist.Normal(0, 1)

    def sample_rho0(n):
        return rho_0.sample((n, config.dim))

    # Run one backward drift step
    loss_ipf, loss_td, loss_fk, loss_boundary = trainer.backward_drift_step(sample_rho0, iteration_idx=0)

    # Check losses are finite
    assert torch.isfinite(torch.tensor(loss_ipf))
    assert torch.isfinite(torch.tensor(loss_td))
    assert torch.isfinite(torch.tensor(loss_fk))
    assert torch.isfinite(torch.tensor(loss_boundary))

    print(
        f"✓ Backward drift step test passed (L_IPF={loss_ipf:.4f}, "
        f"L_TD={loss_td:.4f}, "
        f"L_FK={loss_fk:.4f}, L_B={loss_boundary:.4f})"
    )


def test_forward_drift_step():
    """Test forward drift step runs without error."""
    config = Config(
        dim=2,
        T=0.5,
        dt=0.25,  # T/dt = 2 steps
        n_agents=8,
        verbose=False,
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
    )

    solver = FBSDESolver(config)

    def log_p0(x):
        return -0.5 * (x**2).sum(dim=-1)

    def log_p1(x):
        return -0.5 * ((x - 1.0) ** 2).sum(dim=-1)

    trainer = Trainer(config, solver, log_p0, log_p1)

    # Define sampling function
    rho_1 = dist.Normal(1, 1)

    def sample_rho1(n):
        return rho_1.sample((n, config.dim))

    # Run one forward drift step
    loss_ipf, loss_td, loss_fk, loss_boundary = trainer.forward_drift_step(sample_rho1, iteration_idx=0)

    # Check losses are finite
    assert torch.isfinite(torch.tensor(loss_ipf))
    assert torch.isfinite(torch.tensor(loss_td))
    assert torch.isfinite(torch.tensor(loss_fk))
    assert torch.isfinite(torch.tensor(loss_boundary))

    print(
        f"✓ Forward drift step test passed (L_IPF={loss_ipf:.4f}, "
        f"L_TD={loss_td:.4f}, "
        f"L_FK={loss_fk:.4f}, L_B={loss_boundary:.4f})"
    )


def test_forward_drift_step_multistep_td():
    """Test forward drift step with multi-step TD horizon."""
    config = Config(
        dim=2,
        T=0.5,
        dt=0.25,  # T/dt = 2 steps
        n_agents=8,
        td_horizon=3,
        verbose=False,
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
    )

    solver = FBSDESolver(config)

    def log_p0(x):
        return -0.5 * (x**2).sum(dim=-1)

    def log_p1(x):
        return -0.5 * ((x - 1.0) ** 2).sum(dim=-1)

    trainer = Trainer(config, solver, log_p0, log_p1)

    rho_1 = dist.Normal(1, 1)

    def sample_rho1(n):
        return rho_1.sample((n, config.dim))

    loss_ipf, loss_td, loss_fk, loss_boundary = trainer.forward_drift_step(sample_rho1, iteration_idx=0)

    assert torch.isfinite(torch.tensor(loss_ipf))
    assert torch.isfinite(torch.tensor(loss_td))
    assert torch.isfinite(torch.tensor(loss_fk))
    assert torch.isfinite(torch.tensor(loss_boundary))

    print(
        f"✓ Forward drift multi-step TD test passed "
        f"(L_TD={loss_td:.4f}, horizon={config.td_horizon})"
    )


def test_forward_drift_step_with_ema():
    """Ensure EMA targets can be updated and used during TD computation."""
    config = Config(
        dim=2,
        T=0.5,
        dt=0.25,  # T/dt = 2 steps
        n_agents=8,
        ema_decay=0.9,
        ema_start_step=0,
        ema_update_interval=1,
        verbose=False,
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
    )

    solver = FBSDESolver(config)

    def log_p0(x):
        return -0.5 * (x**2).sum(dim=-1)

    def log_p1(x):
        return -0.5 * ((x - 1.0) ** 2).sum(dim=-1)

    trainer = Trainer(config, solver, log_p0, log_p1)
    rho_1 = dist.Normal(1, 1)

    def sample_rho1(n):
        return rho_1.sample((n, config.dim))

    loss_ipf, loss_td, loss_fk, loss_boundary = trainer.forward_drift_step(sample_rho1, iteration_idx=0)

    assert torch.isfinite(torch.tensor(loss_ipf))
    assert torch.isfinite(torch.tensor(loss_td))
    assert torch.isfinite(torch.tensor(loss_fk))
    assert torch.isfinite(torch.tensor(loss_boundary))
    assert trainer.use_ema
    assert trainer.ema_active

    print(
        f"✓ Forward drift EMA test passed (L_TD={loss_td:.4f})"
    )


def test_trainer_raises_when_ema_requested_without_torch_ema(monkeypatch):
    """EMA configuration should fail fast if torch_ema is unavailable."""
    monkeypatch.setattr(trainer_module, "ExponentialMovingAverage", None)
    config = Config(
        dim=2,
        T=0.5,
        dt=0.25,
        n_agents=8,
        ema_decay=0.9,
        ema_start_step=0,
        ema_update_interval=1,
        verbose=False,
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
    )
    solver = FBSDESolver(config)

    def log_p0(x):
        return -0.5 * (x**2).sum(dim=-1)

    def log_p1(x):
        return -0.5 * ((x - 1.0) ** 2).sum(dim=-1)

    with pytest.raises(RuntimeError, match="torch_ema is not installed"):
        Trainer(config, solver, log_p0, log_p1)


def test_save_checkpoint_records_analytical_metrics(tmp_path):
    """Ensure analytical evaluation metrics are captured on checkpoint."""

    output_dir = tmp_path / "outputs"
    log_dir = tmp_path / "logs"
    ckpt_dir = tmp_path / "ckpts"
    config = Config(
        dim=2,
        T=0.5,
        dt=0.25,
        n_agents=8,
        verbose=False,
        output_dir=str(output_dir),
        log_dir=str(log_dir),
        checkpoint_dir=str(ckpt_dir),
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
    )

    solver = FBSDESolver(config)

    def log_p0(x):
        return -0.5 * (x**2).sum(dim=-1)

    def log_p1(x):
        return -0.5 * ((x - 1.0) ** 2).sum(dim=-1)

    analytical_drift = AnalyticalZeroDrift().to(config.device)
    analytical_cost = AnalyticalZeroCost().to(config.device)

    trainer = Trainer(
        config,
        solver,
        log_p0,
        log_p1,
        analytical_drift_module=analytical_drift,
        analytical_cost_module=analytical_cost,
    )

    def sample_rho0(n):
        return torch.randn(n, config.dim, device=config.device)

    def sample_rho1(n):
        return torch.randn(n, config.dim, device=config.device)

    trainer._sample_rho0_fn = sample_rho0
    trainer._sample_rho1_fn = sample_rho1

    trainer.save_checkpoint(1)

    assert trainer.metrics["loss_ipf_forward_analytical"], "Analytical metrics missing"

    ckpt_path = Path(config.checkpoint_dir) / "checkpoint_0001.pt"
    assert ckpt_path.exists()
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    assert "metrics_analytical" in checkpoint
    assert "loss_ipf_forward_analytical" in checkpoint["metrics_analytical"]


def test_save_checkpoint_with_analytical_modules_requires_sampling_functions(tmp_path):
    """Analytical checkpoint evaluation should fail fast on incomplete setup."""
    output_dir = tmp_path / "outputs"
    log_dir = tmp_path / "logs"
    ckpt_dir = tmp_path / "ckpts"
    config = Config(
        dim=2,
        T=0.5,
        dt=0.25,
        n_agents=8,
        verbose=False,
        output_dir=str(output_dir),
        log_dir=str(log_dir),
        checkpoint_dir=str(ckpt_dir),
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
    )

    solver = FBSDESolver(config)

    def log_p0(x):
        return -0.5 * (x**2).sum(dim=-1)

    def log_p1(x):
        return -0.5 * ((x - 1.0) ** 2).sum(dim=-1)

    trainer = Trainer(
        config,
        solver,
        log_p0,
        log_p1,
        analytical_drift_module=AnalyticalZeroDrift().to(config.device),
        analytical_cost_module=AnalyticalZeroCost().to(config.device),
    )

    with pytest.raises(ValueError, match="sample_rho0_fn and sample_rho1_fn"):
        trainer.save_checkpoint(1)


def test_training_loop_runs():
    """Test that a short training loop completes."""
    config = Config(
        dim=2,
        T=0.5,
        dt=0.25,  # T/dt = 2 steps
        n_agents=8,
        n_outer_iterations=2,
        verbose=False,
        log_interval=1,
        checkpoint_interval=10,  # Don't save checkpoints in this test
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
    )

    solver = FBSDESolver(config)

    # Define distributions
    rho_0 = dist.Normal(0, 1)
    rho_1 = dist.Normal(1, 1)

    def log_p0(x):
        return rho_0.log_prob(x).sum(dim=-1)

    def log_p1(x):
        return rho_1.log_prob(x).sum(dim=-1)

    def sample_rho0(n):
        return rho_0.sample((n, config.dim))

    def sample_rho1(n):
        return rho_1.sample((n, config.dim))

    trainer = Trainer(config, solver, log_p0, log_p1)

    # Run training
    metrics = trainer.train(sample_rho0, sample_rho1)

    # Check that metrics were recorded
    assert len(metrics["loss_ipf_forward"]) == config.n_outer_iterations
    assert len(metrics["loss_td_backward"]) == config.n_outer_iterations
    assert len(metrics["loss_ipf_backward"]) == config.n_outer_iterations
    assert len(metrics["loss_td_forward"]) == config.n_outer_iterations
    assert len(metrics["loss_boundary_backward"]) == config.n_outer_iterations
    assert len(metrics["loss_boundary_forward"]) == config.n_outer_iterations
    assert len(metrics["time_per_iteration"]) == config.n_outer_iterations

    # Check all losses are finite
    for loss in metrics["loss_ipf_forward"]:
        assert torch.isfinite(torch.tensor(loss))
    for loss in metrics["loss_boundary_forward"]:
        assert torch.isfinite(torch.tensor(loss))
    for loss in metrics["loss_boundary_backward"]:
        assert torch.isfinite(torch.tensor(loss))

    print(
        f"✓ Training loop test passed (final L_IPF={metrics['loss_ipf_forward'][-1]:.4f})"
    )


def test_training_loop_runs_with_learned_interactions(tmp_path):
    """Ensure Trainer.train executes the learned interaction cycle."""
    config = Config(
        dim=2,
        T=0.5,
        dt=0.25,
        n_agents=8,
        n_outer_iterations=1,
        n_inner_steps=1,
        n_interaction_steps=1,
        interaction_backend="nn",
        learn_interaction_drift=True,
        learn_interaction_cost=True,
        verbose=False,
        log_interval=1,
        checkpoint_interval=10,
        output_dir=str(tmp_path / "outputs"),
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpts"),
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
        learning_rate_interaction=1e-3,
        drift_type="mean_reversion",
        cost_type="quadratic_deviation",
        int_drift_weight=0.5,
        int_cost_weight=0.25,
        device="cpu",
    )

    (
        drift_net,
        cost_net,
        f_interaction_fn,
        F_interaction_fn,
        analytical_drift,
        analytical_cost,
    ) = create_interactions(config)
    solver = FBSDESolver(config, drift_net=drift_net, cost_net=cost_net)

    rho_0 = dist.Normal(0, 1)
    rho_1 = dist.Normal(1, 1)

    def log_p0(x):
        return rho_0.log_prob(x).sum(dim=-1)

    def log_p1(x):
        return rho_1.log_prob(x).sum(dim=-1)

    def sample_rho0(n):
        return rho_0.sample((n, config.dim))

    def sample_rho1(n):
        return rho_1.sample((n, config.dim))

    trainer = Trainer(
        config,
        solver,
        log_p0,
        log_p1,
        f_interaction_fn=f_interaction_fn,
        F_interaction_fn=F_interaction_fn,
        analytical_drift_module=analytical_drift,
        analytical_cost_module=analytical_cost,
    )

    assert trainer.interaction_learner is not None

    drift_params_before = [p.detach().clone() for p in solver.drift_net.parameters()]
    cost_params_before = [p.detach().clone() for p in solver.cost_net.parameters()]

    metrics = trainer.train(sample_rho0, sample_rho1)

    assert len(metrics["loss_interaction_drift"]) == config.n_outer_iterations
    assert len(metrics["loss_interaction_cost"]) == config.n_outer_iterations
    assert torch.isfinite(torch.tensor(metrics["loss_interaction_drift"][0]))
    assert torch.isfinite(torch.tensor(metrics["loss_interaction_cost"][0]))

    drift_changed = any(
        not torch.allclose(before, after.detach())
        for before, after in zip(drift_params_before, solver.drift_net.parameters())
    )
    cost_changed = any(
        not torch.allclose(before, after.detach())
        for before, after in zip(cost_params_before, solver.cost_net.parameters())
    )
    assert drift_changed
    assert cost_changed


def test_backward_interaction_learning_uses_physical_time_grid(tmp_path, monkeypatch):
    config = Config(
        dim=2,
        T=0.5,
        dt=0.25,
        n_agents=8,
        n_outer_iterations=1,
        n_inner_steps=1,
        n_interaction_steps=1,
        interaction_backend="nn",
        learn_interaction_drift=True,
        verbose=False,
        log_interval=1,
        checkpoint_interval=10,
        output_dir=str(tmp_path / "outputs"),
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpts"),
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
        learning_rate_interaction=1e-3,
        drift_type="mean_reversion",
        int_drift_weight=0.5,
        device="cpu",
    )

    (
        drift_net,
        cost_net,
        f_interaction_fn,
        _F_interaction_fn,
        analytical_drift,
        analytical_cost,
    ) = create_interactions(config)
    solver = FBSDESolver(config, drift_net=drift_net, cost_net=cost_net)

    rho_0 = dist.Normal(0, 1)

    def log_p0(x):
        return rho_0.log_prob(x).sum(dim=-1)

    def log_p1(x):
        return rho_0.log_prob(x).sum(dim=-1)

    def sample_rho0(n):
        return rho_0.sample((n, config.dim))

    trainer = Trainer(
        config,
        solver,
        log_p0,
        log_p1,
        f_interaction_fn=f_interaction_fn,
        analytical_drift_module=analytical_drift,
        analytical_cost_module=analytical_cost,
    )

    captured = {}

    def fake_train(xs, ts, n_steps):
        captured["xs_shape"] = tuple(xs.shape)
        captured["ts"] = ts.detach().clone()
        captured["n_steps"] = n_steps
        return {"f_loss": [0.0], "F_loss": []}

    monkeypatch.setattr(trainer.interaction_learner, "train", fake_train)

    trainer.interaction_learning_step(sample_rho0, mode="backward")

    assert captured["xs_shape"][0] == config.n_time_steps + 1
    assert captured["n_steps"] == config.n_interaction_steps
    assert torch.allclose(captured["ts"], config.time_grid)
    assert not torch.allclose(captured["ts"], torch.flip(config.time_grid, dims=[0]))


def test_training_loop_runs_with_random_batch_backend(tmp_path):
    """Random-batch backend should use the normal training loop without interaction learning."""
    config = Config(
        dim=2,
        T=0.5,
        dt=0.25,
        n_agents=8,
        n_outer_iterations=1,
        n_inner_steps=1,
        interaction_backend="random_batch",
        random_batch_size=4,
        verbose=False,
        log_interval=1,
        checkpoint_interval=10,
        output_dir=str(tmp_path / "outputs"),
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpts"),
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
        drift_type="mean_reversion",
        cost_type="quadratic_deviation",
        int_drift_weight=0.5,
        int_cost_weight=0.25,
        device="cpu",
    )

    (
        drift_net,
        cost_net,
        _f_interaction_fn,
        _F_interaction_fn,
        analytical_drift,
        analytical_cost,
    ) = create_interactions(config)
    solver = FBSDESolver(config, drift_net=drift_net, cost_net=cost_net)

    rho_0 = dist.Normal(0, 1)
    rho_1 = dist.Normal(1, 1)

    def log_p0(x):
        return rho_0.log_prob(x).sum(dim=-1)

    def log_p1(x):
        return rho_1.log_prob(x).sum(dim=-1)

    def sample_rho0(n):
        return rho_0.sample((n, config.dim))

    def sample_rho1(n):
        return rho_1.sample((n, config.dim))

    trainer = Trainer(
        config,
        solver,
        log_p0,
        log_p1,
        analytical_drift_module=analytical_drift,
        analytical_cost_module=analytical_cost,
    )

    assert trainer.interaction_learner is None

    metrics = trainer.train(sample_rho0, sample_rho1)

    assert len(metrics["time_drift_per_iteration"]) == 1
    assert metrics["time_drift_total"] >= 0.0
    assert metrics["time_interaction_total"] == 0.0
    assert metrics["time_interaction_per_iteration"] == [0.0]
    assert metrics["loss_interaction_drift"] == [0.0]
    assert metrics["loss_interaction_cost"] == [0.0]


def test_training_loop_uses_correct_policy_boundary_pairing_for_interaction_learning(
    tmp_path, monkeypatch
):
    """Step 2 should use rho0+forward; Step 4 should use rho1+backward."""
    config = Config(
        dim=2,
        T=0.5,
        dt=0.25,
        n_agents=8,
        n_outer_iterations=1,
        n_inner_steps=1,
        interaction_backend="nn",
        learn_interaction_drift=True,
        n_interaction_steps=1,
        verbose=False,
        log_interval=1,
        checkpoint_interval=10,
        output_dir=str(tmp_path / "outputs"),
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpts"),
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
        learning_rate_interaction=1e-3,
        drift_type="gaussian",
        int_drift_weight=0.5,
        int_drift_sigma=1.0,
        cost_type="zero",
        device="cpu",
    )

    (
        drift_net,
        cost_net,
        f_interaction_fn,
        F_interaction_fn,
        analytical_drift,
        analytical_cost,
    ) = create_interactions(config)
    solver = FBSDESolver(config, drift_net=drift_net, cost_net=cost_net)

    rho0 = torch.tensor([-1.0, -1.0])
    rho1 = torch.tensor([2.0, 2.0])

    def log_p0(x):
        return -(x - rho0).pow(2).sum(dim=-1)

    def log_p1(x):
        return -(x - rho1).pow(2).sum(dim=-1)

    def sample_rho0(n):
        return rho0.expand(n, -1).clone()

    def sample_rho1(n):
        return rho1.expand(n, -1).clone()

    trainer = Trainer(
        config,
        solver,
        log_p0,
        log_p1,
        f_interaction_fn=f_interaction_fn,
        F_interaction_fn=F_interaction_fn,
        analytical_drift_module=analytical_drift,
        analytical_cost_module=analytical_cost,
    )

    calls = []

    def fake_interaction_learning_step(sample_fn, mode="forward", *, iteration_idx=0):
        sample = sample_fn(1)
        calls.append((mode, tuple(sample[0].tolist())))
        return 0.0, 0.0

    monkeypatch.setattr(
        trainer, "interaction_learning_step", fake_interaction_learning_step
    )

    trainer.train(sample_rho0, sample_rho1)

    assert calls == [
        ("forward", tuple(rho0.tolist())),
        ("backward", tuple(rho1.tolist())),
    ]


def test_training_loop_passes_rollout_seed_to_solver(tmp_path, monkeypatch):
    """Training drift steps should pass derived rollout seeds into the solver."""
    config = Config(
        dim=2,
        T=0.5,
        dt=0.25,
        n_agents=8,
        n_outer_iterations=1,
        n_inner_steps=1,
        verbose=False,
        log_interval=1,
        checkpoint_interval=10,
        output_dir=str(tmp_path / "outputs"),
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpts"),
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
        drift_type="mean_reversion",
        cost_type="quadratic_deviation",
        int_drift_weight=0.5,
        int_cost_weight=0.25,
        device="cpu",
        seed=123,
        training_seed=123,
    )

    (
        drift_net,
        cost_net,
        _f_interaction_fn,
        _F_interaction_fn,
        analytical_drift,
        analytical_cost,
    ) = create_interactions(config)
    solver = FBSDESolver(config, drift_net=drift_net, cost_net=cost_net)

    rho_0 = dist.Normal(0, 1)
    rho_1 = dist.Normal(1, 1)

    def log_p0(x):
        return rho_0.log_prob(x).sum(dim=-1)

    def log_p1(x):
        return rho_1.log_prob(x).sum(dim=-1)

    def sample_rho0(n):
        return rho_0.sample((n, config.dim))

    def sample_rho1(n):
        return rho_1.sample((n, config.dim))

    trainer = Trainer(
        config,
        solver,
        log_p0,
        log_p1,
        analytical_drift_module=analytical_drift,
        analytical_cost_module=analytical_cost,
    )

    captured = []
    n_steps = config.n_time_steps

    def fake_forward(x0, seed=None):
        captured.append(("forward", seed))
        xs = x0.unsqueeze(0).repeat(n_steps + 1, 1, 1)
        ts = config.time_grid
        dws = torch.zeros(n_steps, x0.shape[0], x0.shape[1], device=x0.device)
        return xs, ts, dws

    def fake_backward(x1, seed=None):
        captured.append(("backward", seed))
        xs = x1.unsqueeze(0).repeat(n_steps + 1, 1, 1)
        ts = torch.flip(config.time_grid, dims=[0])
        dws = torch.zeros(n_steps, x1.shape[0], x1.shape[1], device=x1.device)
        return xs, ts, dws

    monkeypatch.setattr(trainer.solver, "sample_forward_trajectory", fake_forward)
    monkeypatch.setattr(trainer.solver, "sample_backward_trajectory", fake_backward)

    trainer.train(sample_rho0, sample_rho1)

    assert captured[:2] == [
        ("forward", seed_offset(trainer._make_training_seed(0, 0, 0), 1)),
        ("backward", seed_offset(trainer._make_training_seed(1, 0, 0), 1)),
    ]


def test_interaction_learning_passes_rollout_seed_to_solver(tmp_path, monkeypatch):
    """Interaction learning should pass derived rollout seeds into the solver."""
    config = Config(
        dim=2,
        T=0.5,
        dt=0.25,
        n_agents=8,
        interaction_backend="nn",
        learn_interaction_drift=True,
        n_interaction_steps=1,
        verbose=False,
        log_interval=1,
        checkpoint_interval=10,
        output_dir=str(tmp_path / "outputs"),
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpts"),
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
        learning_rate_interaction=1e-3,
        drift_type="gaussian",
        int_drift_weight=0.5,
        int_drift_sigma=1.0,
        cost_type="zero",
        device="cpu",
        seed=123,
        training_seed=123,
    )

    (
        drift_net,
        cost_net,
        f_interaction_fn,
        F_interaction_fn,
        analytical_drift,
        analytical_cost,
    ) = create_interactions(config)
    solver = FBSDESolver(config, drift_net=drift_net, cost_net=cost_net)

    rho0 = torch.tensor([-1.0, -1.0])

    def log_p0(x):
        return -(x - rho0).pow(2).sum(dim=-1)

    def log_p1(x):
        return -(x - rho0).pow(2).sum(dim=-1)

    def sample_rho0(n):
        return rho0.expand(n, -1).clone()

    trainer = Trainer(
        config,
        solver,
        log_p0,
        log_p1,
        f_interaction_fn=f_interaction_fn,
        F_interaction_fn=F_interaction_fn,
        analytical_drift_module=analytical_drift,
        analytical_cost_module=analytical_cost,
    )

    captured = []
    n_steps = config.n_time_steps

    def fake_forward(x0, seed=None):
        captured.append(seed)
        xs = x0.unsqueeze(0).repeat(n_steps + 1, 1, 1)
        ts = config.time_grid
        dws = torch.zeros(n_steps, x0.shape[0], x0.shape[1], device=x0.device)
        return xs, ts, dws

    monkeypatch.setattr(trainer.solver, "sample_forward_trajectory", fake_forward)
    monkeypatch.setattr(
        trainer.interaction_learner,
        "train",
        lambda xs, ts, n_steps: {"f_loss": [0.0], "F_loss": []},
    )

    trainer.interaction_learning_step(sample_rho0, mode="forward", iteration_idx=7)

    expected_seed = seed_offset(trainer._make_training_seed(3, 1, 7), 1)
    assert captured == [expected_seed]


def test_training_loop_is_deterministic_with_training_seed(tmp_path):
    def build_trainer(run_dir):
        config = Config(
            dim=2,
            T=0.5,
            dt=0.25,
            n_agents=8,
            n_agents_test=8,
            n_outer_iterations=1,
            n_inner_steps=1,
            verbose=False,
            log_interval=1,
            checkpoint_interval=10,
            output_dir=str(run_dir / "outputs"),
            log_dir=str(run_dir / "logs"),
            checkpoint_dir=str(run_dir / "ckpts"),
            learning_rate_value=1e-3,
            learning_rate_grad=1e-3,
            drift_type="mean_reversion",
            cost_type="quadratic_deviation",
            int_drift_weight=0.5,
            int_cost_weight=0.25,
            device="cpu",
            seed=123,
            training_seed=123,
            validation_seed=456,
            visualization_seed=789,
        )

        (
            drift_net,
            cost_net,
            _f_interaction_fn,
            _F_interaction_fn,
            analytical_drift,
            analytical_cost,
        ) = create_interactions(config)
        solver = FBSDESolver(config, drift_net=drift_net, cost_net=cost_net)

        rho_0 = dist.Normal(0, 1)
        rho_1 = dist.Normal(1, 1)

        def log_p0(x):
            return rho_0.log_prob(x).sum(dim=-1)

        def log_p1(x):
            return rho_1.log_prob(x).sum(dim=-1)

        def sample_rho0(n):
            return rho_0.sample((n, config.dim))

        def sample_rho1(n):
            return rho_1.sample((n, config.dim))

        trainer = Trainer(
            config,
            solver,
            log_p0,
            log_p1,
            analytical_drift_module=analytical_drift,
            analytical_cost_module=analytical_cost,
        )
        metrics = trainer.train(sample_rho0, sample_rho1)
        return metrics

    metrics1 = build_trainer(tmp_path / "run1")
    metrics2 = build_trainer(tmp_path / "run2")

    keys_to_match = [
        "loss_ipf_forward",
        "loss_ipf_backward",
        "loss_td_forward",
        "loss_td_backward",
        "loss_fk_forward",
        "loss_fk_backward",
        "loss_boundary_forward",
        "loss_boundary_backward",
        "loss_ipf_forward_analytical",
        "loss_ipf_backward_analytical",
        "loss_td_forward_analytical",
        "loss_td_backward_analytical",
        "loss_fk_forward_analytical",
        "loss_fk_backward_analytical",
    ]

    for key in keys_to_match:
        assert metrics1[key] == metrics2[key], key


def test_trainer_raises_when_learned_drift_labels_are_missing():
    config = Config(
        dim=2,
        T=0.5,
        dt=0.25,
        n_agents=8,
        interaction_backend="nn",
        learn_interaction_drift=True,
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
        learning_rate_interaction=1e-3,
        device="cpu",
    )
    solver = FBSDESolver(
        config,
        drift_net=torch.nn.Linear(config.dim + 1, config.dim),
    )

    def log_p0(x):
        return -0.5 * (x**2).sum(dim=-1)

    def log_p1(x):
        return -0.5 * ((x - 1.0) ** 2).sum(dim=-1)

    with pytest.raises(ValueError, match="f_interaction_fn"):
        Trainer(config, solver, log_p0, log_p1)


def test_trainer_raises_when_learned_cost_labels_are_missing():
    config = Config(
        dim=2,
        T=0.5,
        dt=0.25,
        n_agents=8,
        interaction_backend="nn",
        learn_interaction_cost=True,
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
        learning_rate_interaction=1e-3,
        device="cpu",
    )
    solver = FBSDESolver(
        config,
        cost_net=torch.nn.Linear(config.dim + 1, 1),
    )

    def log_p0(x):
        return -0.5 * (x**2).sum(dim=-1)

    def log_p1(x):
        return -0.5 * ((x - 1.0) ** 2).sum(dim=-1)

    with pytest.raises(ValueError, match="F_interaction_fn"):
        Trainer(config, solver, log_p0, log_p1)


def test_checkpoint_save_load():
    """Test checkpoint save and load."""
    import os
    import tempfile

    config = Config(
        dim=2,
        T=0.5,
        dt=0.25,  # T/dt = 2 steps
        n_agents=8,
        verbose=False,
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
    )

    # Use temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        config.checkpoint_dir = tmpdir

        solver = FBSDESolver(config)

        def log_p0(x):
            return -0.5 * (x**2).sum(dim=-1)

        def log_p1(x):
            return -0.5 * ((x - 1.0) ** 2).sum(dim=-1)

        trainer = Trainer(config, solver, log_p0, log_p1)

        # Save a checkpoint
        trainer.save_checkpoint(iteration=1)

        # Check file was created
        checkpoint_path = os.path.join(tmpdir, "checkpoint_0001.pt")
        assert os.path.exists(checkpoint_path)

        # Create a new trainer and load checkpoint
        solver2 = FBSDESolver(config)
        trainer2 = Trainer(config, solver2, log_p0, log_p1)

        iteration = trainer2.load_checkpoint(checkpoint_path)
        assert iteration == 1

    print("✓ Checkpoint save/load test passed")


def test_checkpoint_load_rejects_invalid_time_total(tmp_path):
    """Checkpoint loading should reject malformed time_total instead of repairing it."""
    config = Config(
        dim=2,
        T=0.5,
        dt=0.25,
        n_agents=8,
        verbose=False,
        checkpoint_dir=str(tmp_path),
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
    )
    solver = FBSDESolver(config)

    def log_p0(x):
        return -0.5 * (x**2).sum(dim=-1)

    def log_p1(x):
        return -0.5 * ((x - 1.0) ** 2).sum(dim=-1)

    trainer = Trainer(config, solver, log_p0, log_p1)
    trainer.save_checkpoint(iteration=1)

    checkpoint_path = Path(config.checkpoint_dir) / "checkpoint_0001.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["metrics"]["time_total"] = ["bad"]
    torch.save(checkpoint, checkpoint_path)

    solver2 = FBSDESolver(config)
    trainer2 = Trainer(config, solver2, log_p0, log_p1)

    with pytest.raises(ValueError, match="time_total must be a scalar"):
        trainer2.load_checkpoint(str(checkpoint_path))


if __name__ == "__main__":
    print("\n=== Testing Trainer (Basic) ===\n")

    test_trainer_initialization()
    test_backward_drift_step()
    test_forward_drift_step()
    test_forward_drift_step_multistep_td()
    test_forward_drift_step_with_ema()
    test_training_loop_runs()
    test_checkpoint_save_load()

    print("\n=== All Trainer basic tests passed! ===\n")
