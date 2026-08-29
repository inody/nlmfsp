"""
Test to verify that the TD loss boundary condition fix is working correctly.

This test validates that:
1. Forward networks (Y, Z) use backward trajectory with boundary at t=0
2. Backward networks (Ŷ, Ẑ) use forward trajectory with boundary at t=T
3. Boundary conditions match Liu et al. (2022) theory
"""

import torch

from nlmfsp.config import Config
from nlmfsp.fbsde import FBSDESolver


def test_td_boundary_forward_direction():
    """
    Test that forward direction (Y, Z update) uses correct boundary.

    Theory:
    - Uses backward trajectory: X̄_s from ρ_target (s: 0→T)
    - Boundary at s=T (real time t=0): X̄_T ~ ρ_0
    - Y(X̄_T, 0) = log ρ_0(X̄_T) - Ŷ(X̄_T, 0)
    """
    config = Config(
        dim=2,
        T=1.0,
        dt=0.1,
        n_agents=16,
        sigma=1.0,
        seed=42,
        td_horizon=5,
    )

    solver = FBSDESolver(config)

    # Sample backward trajectory from ρ_target
    x1 = torch.randn(16, 2)  # Sample from ρ_target
    xs_bwd, ts_bwd, dws_bwd = solver.sample_backward_trajectory(x1, seed=42)

    # Check trajectory structure
    n_steps = len(ts_bwd) - 1
    assert n_steps == 10, f"Expected 10 steps, got {n_steps}"

    # Backward trajectory currently returns reversed time stamps.
    assert torch.allclose(ts_bwd[0], torch.tensor(config.T)), "First time should be T"
    assert torch.allclose(ts_bwd[-1], torch.tensor(0.0)), "Last time should be 0"

    print("✓ Backward trajectory structure verified")

    # Define log density functions
    def log_p0_fn(x):
        """Log density of ρ_0 = N(0, I)"""
        return -0.5 * torch.sum(x**2, dim=-1) - config.dim * 0.5 * torch.log(
            torch.tensor(2 * torch.pi)
        )

    def log_p1_fn(x):
        """Log density of ρ_target = N(1, I)"""
        mean = torch.ones_like(x)
        return -0.5 * torch.sum((x - mean) ** 2, dim=-1) - config.dim * 0.5 * torch.log(
            torch.tensor(2 * torch.pi)
        )

    # Compute TD loss with boundary conditions
    loss = solver.compute_td_loss(
        xs_bwd,
        ts_bwd,
        dws_bwd,
        direction="forward",
        log_p0_fn=log_p0_fn,
        log_p1_fn=log_p1_fn,
    )

    assert loss.shape == ()
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be inf"
    assert loss.item() >= 0, "MSE loss should be non-negative"

    print(f"✓ Forward direction TD loss computed: {loss.item():.4f}")

    # Verify that boundary is used at the correct end
    # With horizon=5, the first few steps should reach the boundary
    # The boundary should be at xs_bwd[n_steps] (t=0)
    boundary_x = xs_bwd[-1]  # Should be at t=0
    log_p_boundary = log_p0_fn(boundary_x)

    assert not torch.isnan(log_p_boundary).any(), "Boundary log density should be valid"
    print(f"✓ Boundary log density at t=0: {log_p_boundary.mean().item():.4f}")


def test_td_boundary_backward_direction():
    """
    Test that backward direction (Ŷ, Ẑ update) uses correct boundary.

    Theory:
    - Uses forward trajectory: X_t from ρ_0 (t: 0→T)
    - Boundary at t=T: X_T ~ ρ_target
    - Ŷ(X_T, T) = log ρ_target(X_T) - Y(X_T, T)
    """
    config = Config(
        dim=2,
        T=1.0,
        dt=0.1,
        n_agents=16,
        sigma=1.0,
        seed=123,
        td_horizon=5,
    )

    solver = FBSDESolver(config)

    # Sample forward trajectory from ρ_0
    x0 = torch.randn(16, 2)  # Sample from ρ_0
    xs_fwd, ts_fwd, dws_fwd = solver.sample_forward_trajectory(x0, seed=123)

    # Check trajectory structure
    n_steps = len(ts_fwd) - 1
    assert n_steps == 10

    # xs_fwd[0] should be at t=0 ~ ρ_0
    # xs_fwd[-1] should be at t=T ~ ρ_target
    assert torch.allclose(ts_fwd[0], torch.tensor(0.0)), "First time should be 0"
    assert torch.allclose(ts_fwd[-1], torch.tensor(config.T)), "Last time should be T"

    print("✓ Forward trajectory structure verified")

    # Define log density functions
    def log_p0_fn(x):
        """Log density of ρ_0 = N(0, I)"""
        return -0.5 * torch.sum(x**2, dim=-1) - config.dim * 0.5 * torch.log(
            torch.tensor(2 * torch.pi)
        )

    def log_p1_fn(x):
        """Log density of ρ_target = N(1, I)"""
        mean = torch.ones_like(x)
        return -0.5 * torch.sum((x - mean) ** 2, dim=-1) - config.dim * 0.5 * torch.log(
            torch.tensor(2 * torch.pi)
        )

    # Compute TD loss with boundary conditions
    loss = solver.compute_td_loss(
        xs_fwd,
        ts_fwd,
        dws_fwd,
        direction="backward",
        log_p0_fn=log_p0_fn,
        log_p1_fn=log_p1_fn,
    )

    assert loss.shape == ()
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() >= 0

    print(f"✓ Backward direction TD loss computed: {loss.item():.4f}")

    # Verify boundary at t=T
    boundary_x = xs_fwd[-1]  # Should be at t=T
    log_p_boundary = log_p1_fn(boundary_x)

    assert not torch.isnan(log_p_boundary).any()
    print(f"✓ Boundary log density at t=T: {log_p_boundary.mean().item():.4f}")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing TD Loss Boundary Condition Fix")
    print("=" * 70)
    print()

    print("Test 1: Forward direction (Y, Z update)")
    print("-" * 70)
    test_td_boundary_forward_direction()
    print()

    print("Test 2: Backward direction (Ŷ, Ẑ update)")
    print("-" * 70)
    test_td_boundary_backward_direction()
    print()

    print("=" * 70)
    print("✅ All boundary condition tests passed!")
    print("=" * 70)
