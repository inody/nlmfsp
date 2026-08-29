"""Tests for FBSDE solver components."""

import torch
import torch.nn as nn

from nlmfsp.config import Config
from nlmfsp.fbsde import FBSDESolver
from nlmfsp.interaction import create_interactions
from nlmfsp.networks import ValueNet, GradientNet, InteractionNet


class _IdentityGrad(nn.Module):
    def forward(self, x, t):
        return x


class _RandomConsumingZeroDrift(nn.Module):
    def forward(self, x, t):
        _ = torch.randperm(x.shape[0], device=x.device)
        return torch.zeros_like(x)


def test_fbsde_initialization():
    """Test FBSDESolver initialization."""
    config = Config(dim=2, T=1.0, dt=0.1, n_agents=16)

    # Test automatic network creation
    solver = FBSDESolver(config)

    assert solver.value_net is not None
    assert solver.value_hat_net is not None
    assert solver.grad_net is not None
    assert solver.grad_hat_net is not None
    assert solver.config == config

    print("✓ FBSDESolver initialization test passed")


def test_fbsde_with_custom_networks():
    """Test FBSDESolver with custom networks."""
    config = Config(dim=2, T=1.0, dt=0.1, n_agents=16)

    # Create a custom config with different hidden dimensions
    custom_config = Config(dim=2, T=1.0, dt=0.1, n_agents=16, hidden_dims=(32, 32))

    # Create custom networks
    value_net = ValueNet(custom_config)
    value_hat_net = ValueNet(custom_config)
    grad_net = GradientNet(custom_config)
    grad_hat_net = GradientNet(custom_config)

    solver = FBSDESolver(
        config,
        value_net=value_net,
        value_hat_net=value_hat_net,
        grad_net=grad_net,
        grad_hat_net=grad_hat_net,
    )

    assert solver.value_net is value_net
    assert solver.grad_net is grad_net

    print("✓ FBSDESolver with custom networks test passed")


def test_fbsde_with_interaction_terms():
    """Test FBSDESolver with interaction and cost networks."""
    config = Config(dim=2, T=1.0, dt=0.1, n_agents=16)

    # Create interaction networks
    drift_net = InteractionNet(config, output_dim=config.dim)
    cost_net = InteractionNet(config, output_dim=1)

    # Base drift function
    base_drift_fn = lambda x: -0.1 * x

    solver = FBSDESolver(
        config,
        drift_net=drift_net,
        cost_net=cost_net,
        base_drift_fn=base_drift_fn,
    )

    assert solver.drift_net is drift_net
    assert solver.cost_net is cost_net
    assert solver.base_drift_fn is not None

    print("✓ FBSDESolver with interaction terms test passed")


def test_compute_div_gz_forward():
    """Test divergence computation (forward direction)."""
    config = Config(dim=2, T=1.0, dt=0.1, n_agents=8, sigma=1.0, seed=42)
    solver = FBSDESolver(config)

    # Create test data
    xs = torch.randn(8, 2, requires_grad=True)
    ts = torch.tensor([0.5] * 8)

    # Compute divergence
    div_gz = solver.compute_div_gz(xs, ts, solver.grad_net, direction="forward")

    assert div_gz.shape == (8,)
    assert not torch.isnan(div_gz).any()
    assert not torch.isinf(div_gz).any()

    print("✓ compute_div_gz (forward) test passed")


def test_compute_div_gz_backward():
    """Test divergence computation (backward direction)."""
    config = Config(dim=2, T=1.0, dt=0.1, n_agents=8, sigma=1.0, seed=42)
    solver = FBSDESolver(config)

    # Create test data
    xs = torch.randn(8, 2, requires_grad=True)
    ts = torch.tensor([0.5] * 8)

    # Compute divergence
    div_gz = solver.compute_div_gz(xs, ts, solver.grad_hat_net, direction="backward")

    assert div_gz.shape == (8,)
    assert not torch.isnan(div_gz).any()
    assert not torch.isinf(div_gz).any()

    print("✓ compute_div_gz (backward) test passed")


def test_compute_div_gz_with_interaction():
    """Test divergence computation with interaction terms."""
    config = Config(dim=2, T=1.0, dt=0.1, n_agents=8, sigma=1.0, seed=42)

    # Create interaction network
    drift_net = InteractionNet(config, output_dim=config.dim)
    base_drift_fn = lambda x: -0.1 * x

    solver = FBSDESolver(
        config,
        drift_net=drift_net,
        base_drift_fn=base_drift_fn,
    )

    # Create test data
    xs = torch.randn(8, 2, requires_grad=True)
    ts = torch.tensor([0.5] * 8)

    # Compute divergence
    div_gz = solver.compute_div_gz(xs, ts, solver.grad_net, direction="forward")

    assert div_gz.shape == (8,)
    assert not torch.isnan(div_gz).any()

    print("✓ compute_div_gz with interaction test passed")


def test_stochastic_seed_scope_decouples_interaction_rng_from_hutchinson():
    """Interaction RNG consumption should not perturb Hutchinson draws."""
    config = Config(dim=2, T=1.0, dt=0.1, n_agents=8, sigma=1.0, seed=42)
    grad_net = _IdentityGrad()
    solver_exact = FBSDESolver(config, grad_net=grad_net, grad_hat_net=grad_net)
    solver_rb = FBSDESolver(
        config,
        grad_net=grad_net,
        grad_hat_net=grad_net,
        drift_net=_RandomConsumingZeroDrift(),
    )

    xs = torch.randn(8, 2, requires_grad=True)
    ts = torch.tensor([0.5] * 8)

    with solver_exact.stochastic_seed_scope(123):
        div_exact = solver_exact.compute_div_gz(
            xs, ts, solver_exact.grad_net, direction="forward"
        )
    with solver_rb.stochastic_seed_scope(123):
        div_rb = solver_rb.compute_div_gz(xs, ts, solver_rb.grad_net, direction="forward")

    assert torch.allclose(div_exact, div_rb)


def test_hutchinson_estimator_properties():
    """Test statistical properties of Hutchinson's estimator.

    For a linear function drift = Ax, the divergence should be exactly Tr(A).
    We test if the Hutchinson estimator approximates this well.
    """
    config = Config(dim=3, n_agents=100, sigma=1.0, seed=42)
    solver = FBSDESolver(config)

    # Create a simple linear network (approximate identity)
    # For testing, we'll use a small network and check consistency
    xs = torch.randn(100, 3, requires_grad=True)
    ts = torch.tensor([0.5] * 100)

    # Compute divergence multiple times and check variance
    divs = []
    for _ in range(10):
        div_gz = solver.compute_div_gz(xs, ts, solver.grad_net, direction="forward")
        divs.append(div_gz.mean().item())

    # Check that results are consistent (low variance)
    divs_tensor = torch.tensor(divs)
    mean = divs_tensor.mean()
    std = divs_tensor.std()

    # Standard deviation should be reasonable (not too large)
    assert std < 10.0, f"Hutchinson estimator has too high variance: {std:.3f}"

    print(f"✓ Hutchinson estimator properties test passed (mean={mean:.3f}, std={std:.3f})")


def test_sample_forward_trajectory():
    """Test forward SDE trajectory sampling."""
    config = Config(dim=2, T=1.0, dt=0.1, n_agents=16, sigma=1.0, seed=42)
    solver = FBSDESolver(config)

    # Initial states
    x0 = torch.randn(16, 2)

    # Sample trajectory
    xs, ts, dws = solver.sample_forward_trajectory(x0, seed=42)

    # Check shapes
    n_steps = int(config.T / config.dt)
    assert xs.shape == (n_steps + 1, 16, 2)
    assert ts.shape == (n_steps + 1,)
    assert dws.shape == (n_steps, 16, 2)

    # Check initial condition
    assert torch.allclose(xs[0], x0, atol=1e-6)

    # Check time grid
    assert torch.allclose(ts[0], torch.tensor(0.0))
    assert torch.allclose(ts[-1], torch.tensor(config.T))

    # Check no NaN/Inf
    assert not torch.isnan(xs).any()
    assert not torch.isinf(xs).any()

    print("✓ sample_forward_trajectory test passed")


def test_sample_backward_trajectory():
    """Test backward SDE trajectory sampling."""
    config = Config(dim=2, T=1.0, dt=0.1, n_agents=16, sigma=1.0, seed=42)
    solver = FBSDESolver(config)

    # Terminal states
    x1 = torch.randn(16, 2)

    # Sample trajectory
    xs, ts, dws = solver.sample_backward_trajectory(x1, seed=42)

    # Check shapes
    n_steps = int(config.T / config.dt)
    assert xs.shape == (n_steps + 1, 16, 2)
    assert ts.shape == (n_steps + 1,)
    assert dws.shape == (n_steps, 16, 2)

    # Check terminal condition (X̄_T = x1 stored at final index)
    assert torch.allclose(xs[-1], x1, atol=1e-6)

    # Check no NaN/Inf
    assert not torch.isnan(xs).any()
    assert not torch.isinf(xs).any()

    print("✓ sample_backward_trajectory test passed")


def test_trajectory_with_interaction():
    """Test trajectory sampling with interaction terms."""
    config = Config(dim=2, T=1.0, dt=0.1, n_agents=16, sigma=0.5, seed=42)

    # Create interaction networks
    drift_net = InteractionNet(config, output_dim=config.dim)
    cost_net = InteractionNet(config, output_dim=1)
    base_drift_fn = lambda x: -0.2 * x  # Mean reversion

    solver = FBSDESolver(
        config,
        drift_net=drift_net,
        cost_net=cost_net,
        base_drift_fn=base_drift_fn,
    )

    # Sample forward trajectory
    x0 = torch.randn(16, 2)
    xs_fwd, _, _ = solver.sample_forward_trajectory(x0, seed=42)

    # Sample backward trajectory
    x1 = torch.randn(16, 2)
    xs_bwd, _, _ = solver.sample_backward_trajectory(x1, seed=43)

    # Check shapes and validity
    n_steps = int(config.T / config.dt)
    assert xs_fwd.shape == (n_steps + 1, 16, 2)
    assert xs_bwd.shape == (n_steps + 1, 16, 2)
    assert not torch.isnan(xs_fwd).any()
    assert not torch.isnan(xs_bwd).any()

    print("✓ Trajectory with interaction test passed")


def test_trajectory_reproducibility():
    """Test that trajectories are reproducible with same seed."""
    config = Config(dim=2, T=1.0, dt=0.1, n_agents=8, sigma=1.0)
    solver = FBSDESolver(config)

    x0 = torch.randn(8, 2)

    # Sample twice with same seed
    xs1, _, _ = solver.sample_forward_trajectory(x0, seed=123)
    xs2, _, _ = solver.sample_forward_trajectory(x0, seed=123)

    # Should be identical
    assert torch.allclose(xs1, xs2, atol=1e-6)

    # Sample with different seed
    xs3, _, _ = solver.sample_forward_trajectory(x0, seed=456)

    # Should be different
    assert not torch.allclose(xs1, xs3, atol=1e-3)

    print("✓ Trajectory reproducibility test passed")


def test_gradient_flow_through_div_gz():
    """Test that gradients flow through divergence computation."""
    config = Config(dim=2, T=1.0, dt=0.1, n_agents=4, sigma=1.0, seed=42)
    solver = FBSDESolver(config)

    # Get network parameters
    params = list(solver.grad_net.parameters())
    initial_params = [p.clone().detach() for p in params]

    # Create test data
    xs = torch.randn(4, 2, requires_grad=True)
    ts = torch.tensor([0.5] * 4)

    # Compute divergence and loss
    div_gz = solver.compute_div_gz(xs, ts, solver.grad_net, direction="forward")
    loss = div_gz.mean()

    # Backward pass
    loss.backward()

    # Check that gradients exist for network parameters
    has_grad = False
    for p in params:
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients computed for network parameters"

    print("✓ Gradient flow through div_gz test passed")


def test_compute_ipf_loss():
    """Test L_IPF computation."""
    config = Config(dim=2, T=1.0, dt=0.1, n_agents=16, sigma=1.0, seed=42)
    solver = FBSDESolver(config)

    # Sample a forward trajectory
    x0 = torch.randn(16, 2)
    xs_fwd, ts_fwd, _ = solver.sample_forward_trajectory(x0, seed=42)

    # Define simple log-probability functions
    def log_p0(x):
        return -0.5 * (x**2).sum(dim=-1)

    def log_p1(x):
        return -0.5 * ((x - 1.0)**2).sum(dim=-1)

    # Compute loss
    loss = solver.compute_ipf_loss(xs_fwd, ts_fwd, log_p0, log_p1)

    # Check loss properties
    assert loss.shape == ()  # Scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    print(f"✓ compute_ipf_loss test passed (loss={loss.item():.3f})")


def test_compute_td_loss():
    """Test L_TD computation."""
    config = Config(dim=2, T=1.0, dt=0.1, n_agents=16, sigma=1.0, seed=42)

    cost_net = InteractionNet(config, output_dim=1)
    solver = FBSDESolver(config, cost_net=cost_net)

    # Sample a backward trajectory
    x1 = torch.randn(16, 2)
    xs_bwd, ts_bwd, dws_bwd = solver.sample_backward_trajectory(x1, seed=42)

    # Compute loss
    loss = solver.compute_td_loss(xs_bwd, ts_bwd, dws_bwd, direction="forward")

    # Check loss properties
    assert loss.shape == ()  # Scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() >= 0  # MSE loss is non-negative

    print(f"✓ compute_td_loss test passed (loss={loss.item():.3f})")


def test_compute_td_loss_backward_direction():
    """TD loss should also work when updating backward networks."""
    config = Config(dim=2, T=1.0, dt=0.1, n_agents=16, sigma=1.0, seed=123)
    solver = FBSDESolver(config)

    x0 = torch.randn(16, 2)
    xs_fwd, ts_fwd, dws_fwd = solver.sample_forward_trajectory(x0, seed=123)

    loss = solver.compute_td_loss(xs_fwd, ts_fwd, dws_fwd, direction="backward")

    assert loss.shape == ()
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    print(f"✓ compute_td_loss (backward direction) test passed (loss={loss.item():.3f})")


def test_compute_fk_loss():
    """Test L_FK computation."""
    config = Config(dim=2, T=1.0, dt=0.1, n_agents=16, sigma=1.0, seed=42)
    solver = FBSDESolver(config)

    # Sample a backward trajectory
    x1 = torch.randn(16, 2)
    xs_bwd, ts_bwd, _ = solver.sample_backward_trajectory(x1, seed=42)

    # Compute loss
    loss = solver.compute_fk_loss(xs_bwd, ts_bwd, direction="forward")

    # Check loss properties
    assert loss.shape == ()  # Scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() >= 0  # MSE loss is non-negative

    print(f"✓ compute_fk_loss test passed (loss={loss.item():.3f})")


def test_compute_fk_loss_backward_direction():
    """FK loss should support backward network updates."""
    config = Config(dim=2, T=1.0, dt=0.1, n_agents=16, sigma=1.0, seed=321)
    solver = FBSDESolver(config)

    x0 = torch.randn(16, 2)
    xs_fwd, ts_fwd, _ = solver.sample_forward_trajectory(x0, seed=321)

    loss = solver.compute_fk_loss(xs_fwd, ts_fwd, direction="backward")

    assert loss.shape == ()
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    print(f"✓ compute_fk_loss (backward direction) test passed (loss={loss.item():.3f})")


def test_loss_gradient_flow():
    """Test that gradients flow through all loss functions."""
    config = Config(dim=2, T=0.5, dt=0.1, n_agents=8, sigma=1.0, seed=42)

    cost_net = InteractionNet(config, output_dim=1)
    solver = FBSDESolver(config, cost_net=cost_net)

    # Sample trajectories
    x0 = torch.randn(8, 2)
    x1 = torch.randn(8, 2)
    xs_fwd, ts_fwd, _ = solver.sample_forward_trajectory(x0, seed=42)
    xs_bwd, ts_bwd, dws_bwd = solver.sample_backward_trajectory(x1, seed=43)

    solver.zero_grad()

    # Test IPF loss gradients
    loss_ipf = solver.compute_ipf_loss(xs_fwd, ts_fwd)
    loss_ipf.backward()

    # Check gradients exist
    has_grad_ipf = any(
        (param.grad is not None) and param.grad.detach().abs().sum() > 0
        for param in solver.grad_net.parameters()
    )
    assert has_grad_ipf, "IPF loss produced no gradient for grad_net parameters"

    solver.zero_grad()

    solver.zero_grad()

    # Test IPF loss gradients (backward direction)
    loss_ipf_back = solver.compute_ipf_loss(xs_fwd, ts_fwd, direction="backward")
    loss_ipf_back.backward()

    has_grad_ipf_hat = any(
        (param.grad is not None) and param.grad.detach().abs().sum() > 0
        for param in solver.grad_hat_net.parameters()
    )
    assert has_grad_ipf_hat, "IPF loss (backward) produced no gradient for grad_hat_net parameters"

    solver.zero_grad()

    # Test TD loss gradients
    loss_td = solver.compute_td_loss(xs_bwd, ts_bwd, dws_bwd, direction="forward")
    loss_td.backward()

    has_grad_td = any(
        (param.grad is not None) and param.grad.detach().abs().sum() > 0
        for param in solver.value_net.parameters()
    )
    assert has_grad_td, "TD loss produced no gradient for value_net parameters"

    solver.zero_grad()

    # Test FK loss gradients
    loss_fk = solver.compute_fk_loss(xs_bwd, ts_bwd, direction="forward")
    loss_fk.backward()

    has_grad_fk_value = any(
        (param.grad is not None) and param.grad.detach().abs().sum() > 0
        for param in solver.value_net.parameters()
    )
    has_grad_fk_grad = any(
        (param.grad is not None) and param.grad.detach().abs().sum() > 0
        for param in solver.grad_net.parameters()
    )
    assert has_grad_fk_value, "FK loss produced no gradient for value_net parameters"
    assert has_grad_fk_grad, "FK loss produced no gradient for grad_net parameters"

    solver.zero_grad()

    # Backward networks: use forward trajectories
    xs_fwd_b, ts_fwd_b, dws_fwd_b = solver.sample_forward_trajectory(x0, seed=99)

    loss_td_back = solver.compute_td_loss(xs_fwd_b, ts_fwd_b, dws_fwd_b, direction="backward")
    loss_td_back.backward()

    has_grad_td_hat = any(
        (param.grad is not None) and param.grad.detach().abs().sum() > 0
        for param in solver.value_hat_net.parameters()
    )
    assert has_grad_td_hat, "TD loss (backward) produced no gradient for value_hat_net parameters"

    solver.zero_grad()

    loss_fk_back = solver.compute_fk_loss(xs_fwd_b, ts_fwd_b, direction="backward")
    loss_fk_back.backward()

    has_grad_fk_hat_grad = any(
        (param.grad is not None) and param.grad.detach().abs().sum() > 0
        for param in solver.grad_hat_net.parameters()
    )
    assert has_grad_fk_hat_grad, "FK loss (backward) produced no gradient for grad_hat_net parameters"

    print("✓ Loss gradient flow test passed")


class _ZeroNet(nn.Module):
    """Simple zero-output network for deterministic trajectory tests."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x.new_zeros((x.shape[0], self.output_dim))


def _build_exact_and_random_batch_solvers():
    """Create exact and random-batch solvers that share deterministic gradients."""
    exact_config = Config(
        dim=2,
        T=1.0,
        dt=0.1,
        n_agents=16,
        sigma=1.0,
        seed=42,
        interaction_backend="exact",
        drift_type="gaussian",
        int_drift_weight=1.0,
        int_drift_sigma=1.0,
    )
    rb_config = Config(
        dim=2,
        T=1.0,
        dt=0.1,
        n_agents=16,
        sigma=1.0,
        seed=42,
        interaction_backend="random_batch",
        random_batch_size=4,
        drift_type="gaussian",
        int_drift_weight=1.0,
        int_drift_sigma=1.0,
    )

    exact_drift, _, _, _, _, _ = create_interactions(exact_config)
    rb_drift, _, _, _, _, _ = create_interactions(rb_config)

    exact_solver = FBSDESolver(
        exact_config,
        grad_net=_ZeroNet(exact_config.dim),
        grad_hat_net=_ZeroNet(exact_config.dim),
        drift_net=exact_drift,
    )
    rb_solver = FBSDESolver(
        rb_config,
        grad_net=_ZeroNet(rb_config.dim),
        grad_hat_net=_ZeroNet(rb_config.dim),
        drift_net=rb_drift,
    )
    return exact_solver, rb_solver


def test_random_batch_forward_keeps_brownian_noise_aligned():
    """RB subset RNG should not change forward-trajectory Brownian increments."""
    exact_solver, rb_solver = _build_exact_and_random_batch_solvers()
    x0 = torch.randn(16, 2)

    _, _, exact_dws = exact_solver.sample_forward_trajectory(x0, seed=123)
    _, _, rb_dws = rb_solver.sample_forward_trajectory(x0, seed=123)

    assert torch.allclose(exact_dws, rb_dws)


def test_random_batch_backward_keeps_brownian_noise_aligned():
    """RB subset RNG should not change backward-trajectory Brownian increments."""
    exact_solver, rb_solver = _build_exact_and_random_batch_solvers()
    x1 = torch.randn(16, 2)

    _, _, exact_dws = exact_solver.sample_backward_trajectory(x1, seed=456)
    _, _, rb_dws = rb_solver.sample_backward_trajectory(x1, seed=456)

    assert torch.allclose(exact_dws, rb_dws)


def test_exact_timing_includes_rollout_direct_calls():
    """Exact timing stats should include direct drift calls during rollout."""
    exact_solver, _ = _build_exact_and_random_batch_solvers()
    x0 = torch.randn(16, 2)
    x1 = torch.randn(16, 2)

    start = exact_solver.timing_stats()
    exact_solver.sample_forward_trajectory(x0, seed=123)
    mid = exact_solver.timing_stats()
    exact_solver.sample_backward_trajectory(x1, seed=456)
    end = exact_solver.timing_stats()

    n_steps = exact_solver.config.n_time_steps
    assert mid["exact_drift_call_count"] - start["exact_drift_call_count"] == n_steps
    assert end["exact_drift_call_count"] - mid["exact_drift_call_count"] == n_steps
    assert mid["exact_drift_eval_time_total"] > start["exact_drift_eval_time_total"]
    assert end["exact_drift_eval_time_total"] > mid["exact_drift_eval_time_total"]


if __name__ == "__main__":
    print("\n=== Testing FBSDE Solver Components ===\n")

    # Basic initialization tests
    test_fbsde_initialization()
    test_fbsde_with_custom_networks()
    test_fbsde_with_interaction_terms()

    # Divergence computation tests
    test_compute_div_gz_forward()
    test_compute_div_gz_backward()
    test_compute_div_gz_with_interaction()
    test_hutchinson_estimator_properties()

    # Trajectory sampling tests
    test_sample_forward_trajectory()
    test_sample_backward_trajectory()
    test_trajectory_with_interaction()
    test_trajectory_reproducibility()

    # Gradient tests
    test_gradient_flow_through_div_gz()
    test_compute_ipf_loss()
    test_compute_td_loss()
    test_compute_td_loss_backward_direction()
    test_compute_fk_loss()
    test_compute_fk_loss_backward_direction()
    test_loss_gradient_flow()

    print("\n=== All FBSDE tests passed! ===\n")
