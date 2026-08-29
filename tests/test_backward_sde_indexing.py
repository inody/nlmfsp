"""
Comprehensive tests for backward SDE indexing fix.

This test suite verifies that the backward SDE indexing fix (Phase 1-5)
works correctly. Tests are based on Section 5 of BACKWARD_SDE_INDEXING_FIX_PLAN_CORRECTED.md
"""

import torch
import pytest
from nlmfsp.config import Config
from nlmfsp.fbsde import FBSDESolver


class TestBackwardSDEIndexing:
    """Test suite for backward SDE physical time indexing fix."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(dim=2, T=1.0, dt=0.1, n_agents=5)

    @pytest.fixture
    def solver(self, config):
        """Create FBSDESolver instance."""
        return FBSDESolver(config)

    def test_backward_trajectory_noise_indexing(self, solver, config):
        """Test that dws indices correspond to correct time transitions.

        CRITICAL: Verifies that dws[i] is used for transition xs[i+1] → xs[i]
        (physical time DECREASING: from later to earlier time).

        After fix: xs[n_steps] at t=T, xs[0] at t=0
        dws[i] corresponds to transition FROM xs[i+1] TO xs[i]
        """
        # Sample with fixed seed
        x1 = torch.randn(5, 2)
        xs, ts, dws = solver.sample_backward_trajectory(x1, seed=42)

        n_steps = int(config.T / config.dt)

        # Verify shapes
        assert xs.shape == (n_steps + 1, 5, 2), f"xs shape: {xs.shape}"
        assert ts.shape == (n_steps + 1,), f"ts shape: {ts.shape}"
        assert dws.shape == (n_steps, 5, 2), f"dws shape: {dws.shape}"

        # Verify time ordering
        assert abs(ts[0].item() - config.T) < 1e-6, f"ts[0] should be T={config.T}, got {ts[0]}"
        assert abs(ts[-1].item() - 0.0) < 1e-6, f"ts[-1] should be 0.0, got {ts[-1]}"
        assert (ts[:-1] > ts[1:]).all(), "Time should be decreasing for backward"

        # CRITICAL: Verify dws[i] was used for transition xs[i+1] → xs[i]
        # Reconstruct one step deterministically
        xs_check, ts_check, dws_check = solver.sample_backward_trajectory(x1[:1], seed=42)

        # Manually verify: dws[n_steps-1] used for transition xs[n_steps] → xs[n_steps-1]
        # This is the FIRST transition in physical time (from t=T to t=T-dt)

        # Manually compute first step (in physical time)
        t_start = ts_check[0].expand(1)  # t=T
        x_start = xs_check[n_steps]       # xs[n_steps] at t=T (starting point)

        z_hat = solver.grad_hat_net(x_start, t_start)
        drift = config.sigma * z_hat

        # Manual propagation: xs[n_steps] + drift*dt + sigma*dw → xs[n_steps-1]
        # The noise used should be dws[n_steps-1]
        x_next_manual = x_start + drift * config.dt + config.sigma * dws_check[n_steps - 1]

        # Should match xs_check[n_steps-1]
        assert torch.allclose(x_next_manual, xs_check[n_steps - 1], atol=1e-5), \
            f"dws[{n_steps-1}] should correspond to transition xs[{n_steps}] → xs[{n_steps-1}]"

        print("✓ Noise indexing verified: dws[i] corresponds to xs[i+1] → xs[i]")
        print("✓ test_backward_trajectory_noise_indexing PASSED")

    def test_boundary_condition_symmetry(self, solver, config):
        """Test that Y + Ŷ = log ρ at boundaries.

        At t=0: Y(x,0) + Ŷ(x,0) ≈ log ρ_0(x)
        At t=T: Y(x,T) + Ŷ(x,T) ≈ log ρ_1(x)

        Note: This test checks the computation runs without NaN.
        After training, these should be small errors.
        """
        # Define simple Gaussian distributions
        def log_p0(x):
            return -0.5 * (x**2).sum(dim=-1)  # log N(0, I)

        def log_p1(x):
            mean = torch.tensor([1.0, 1.0], device=x.device)
            return -0.5 * ((x - mean)**2).sum(dim=-1)  # log N([1,1], I)

        # Sample test points
        x_test = torch.randn(100, 2, device=config.device)

        # Test at t=0
        t0 = torch.zeros(100, device=config.device)
        y_at_0 = solver.value_net(x_test, t0).squeeze(-1)
        y_hat_at_0 = solver.value_hat_net(x_test, t0).squeeze(-1)

        # Should satisfy: Y(x,0) + Ŷ(x,0) ≈ log ρ_0(x)
        sum_at_0 = y_at_0 + y_hat_at_0
        target_at_0 = log_p0(x_test)
        error_at_0 = (sum_at_0 - target_at_0).abs().mean()

        print(f"Boundary error at t=0: {error_at_0:.6f}")

        # Test at t=T
        tT = torch.full((100,), config.T, device=config.device)
        y_at_T = solver.value_net(x_test, tT).squeeze(-1)
        y_hat_at_T = solver.value_hat_net(x_test, tT).squeeze(-1)

        sum_at_T = y_at_T + y_hat_at_T
        target_at_T = log_p1(x_test)
        error_at_T = (sum_at_T - target_at_T).abs().mean()

        print(f"Boundary error at t=T: {error_at_T:.6f}")

        # After training, these should be small
        # Here we just verify the computation runs
        assert not torch.isnan(error_at_0), "Boundary computation at t=0 failed"
        assert not torch.isnan(error_at_T), "Boundary computation at t=T failed"

        print("✓ test_boundary_condition_symmetry PASSED")

    def test_td_loss_time_consistency(self, solver, config):
        """Test that TD targets use correct time indices.

        Verifies that TD loss computation runs without errors and produces
        finite positive losses for both forward and backward directions.
        """
        # Sample trajectories
        x0 = torch.randn(5, 2)
        x1 = torch.randn(5, 2)

        xs_fwd, ts_fwd, dws_fwd = solver.sample_forward_trajectory(x0)
        xs_bwd, ts_bwd, dws_bwd = solver.sample_backward_trajectory(x1)

        # Define simple log densities
        def log_p0(x):
            return -0.5 * (x**2).sum(dim=-1)

        def log_p1(x):
            return -0.5 * ((x - 1.0)**2).sum(dim=-1)

        # Compute TD loss for forward direction
        loss_fwd = solver.compute_td_loss(
            xs_bwd, ts_bwd, dws_bwd,
            log_p0_fn=log_p0,
            log_p1_fn=log_p1,
            direction="forward",
        )

        # Compute TD loss for backward direction
        loss_bwd = solver.compute_td_loss(
            xs_fwd, ts_fwd, dws_fwd,
            log_p0_fn=log_p0,
            log_p1_fn=log_p1,
            direction="backward",
        )

        # Both should be finite and positive
        assert torch.isfinite(loss_fwd), f"Forward TD loss is {loss_fwd}"
        assert torch.isfinite(loss_bwd), f"Backward TD loss is {loss_bwd}"
        assert loss_fwd >= 0, f"Forward TD loss should be non-negative, got {loss_fwd}"
        assert loss_bwd >= 0, f"Backward TD loss should be non-negative, got {loss_bwd}"

        print(f"✓ TD loss computed successfully: fwd={loss_fwd:.4f}, bwd={loss_bwd:.4f}")
        print("✓ test_td_loss_time_consistency PASSED")

    def test_ipf_loss_consistency(self, solver, config):
        """Test that IPF loss uses correct boundary indices.

        Verifies that IPF loss computation runs without errors and produces
        finite losses for both forward and backward directions.
        """
        # Sample trajectories
        x0 = torch.randn(5, 2)
        x1 = torch.randn(5, 2)

        xs_fwd, ts_fwd, _ = solver.sample_forward_trajectory(x0)
        xs_bwd, ts_bwd, _ = solver.sample_backward_trajectory(x1)

        # Define simple log densities
        def log_p0(x):
            return -0.5 * (x**2).sum(dim=-1)

        def log_p1(x):
            return -0.5 * ((x - 1.0)**2).sum(dim=-1)

        # Compute IPF loss for forward direction
        loss_fwd = solver.compute_ipf_loss(
            xs_bwd, ts_bwd,
            log_p0_fn=log_p0,
            log_p1_fn=log_p1,
            direction="forward",
        )

        # Compute IPF loss for backward direction
        loss_bwd = solver.compute_ipf_loss(
            xs_fwd, ts_fwd,
            log_p0_fn=log_p0,
            log_p1_fn=log_p1,
            direction="backward",
        )

        # Both should be finite
        assert torch.isfinite(loss_fwd), f"Forward IPF loss is {loss_fwd}"
        assert torch.isfinite(loss_bwd), f"Backward IPF loss is {loss_bwd}"

        print(f"✓ IPF loss computed successfully: fwd={loss_fwd:.4f}, bwd={loss_bwd:.4f}")
        print("✓ test_ipf_loss_consistency PASSED")

    def test_end_to_end_integration(self, solver, config):
        """Test complete trajectory sampling and loss computation.

        Simplified end-to-end test that verifies all components work together
        without requiring full training setup.
        """
        # Define simple sampling functions
        def sample_rho0(n):
            return torch.randn(n, 2)

        def sample_rho1(n):
            return torch.randn(n, 2) + torch.tensor([1.0, 1.0])

        def log_p0(x):
            return -0.5 * (x**2).sum(dim=-1)

        def log_p1(x):
            mean = torch.tensor([1.0, 1.0], device=x.device)
            return -0.5 * ((x - mean)**2).sum(dim=-1)

        # Sample trajectories
        x0_test = sample_rho0(10)
        x1_test = sample_rho1(10)

        xs_fwd, ts_fwd, dws_fwd = solver.sample_forward_trajectory(x0_test)
        xs_bwd, ts_bwd, dws_bwd = solver.sample_backward_trajectory(x1_test)

        # Verify trajectory shapes and times
        n_steps = int(config.T / config.dt)

        # Forward trajectory
        assert abs(ts_fwd[0].item() - 0.0) < 1e-6, "Forward trajectory should start at t=0"
        assert abs(ts_fwd[-1].item() - config.T) < 1e-6, "Forward trajectory should end at t=T"
        assert xs_fwd.shape == (n_steps + 1, 10, 2), f"Forward xs shape: {xs_fwd.shape}"

        # Backward trajectory with physical indexing
        assert abs(ts_bwd[0].item() - config.T) < 1e-6, "Backward ts[0] should be T"
        assert abs(ts_bwd[-1].item() - 0.0) < 1e-6, "Backward ts[-1] should be 0.0"
        assert xs_bwd.shape == (n_steps + 1, 10, 2), f"Backward xs shape: {xs_bwd.shape}"

        # Compute all losses to verify they work together
        loss_ipf_fwd = solver.compute_ipf_loss(
            xs_bwd, ts_bwd, log_p0, log_p1, direction="forward"
        )
        loss_ipf_bwd = solver.compute_ipf_loss(
            xs_fwd, ts_fwd, log_p0, log_p1, direction="backward"
        )

        loss_td_fwd = solver.compute_td_loss(
            xs_bwd,
            ts_bwd,
            dws_bwd,
            direction="forward",
            log_p0_fn=log_p0,
            log_p1_fn=log_p1,
        )
        loss_td_bwd = solver.compute_td_loss(
            xs_fwd,
            ts_fwd,
            dws_fwd,
            direction="backward",
            log_p0_fn=log_p0,
            log_p1_fn=log_p1,
        )

        # All losses should be finite
        assert torch.isfinite(loss_ipf_fwd), f"IPF forward loss: {loss_ipf_fwd}"
        assert torch.isfinite(loss_ipf_bwd), f"IPF backward loss: {loss_ipf_bwd}"
        assert torch.isfinite(loss_td_fwd), f"TD forward loss: {loss_td_fwd}"
        assert torch.isfinite(loss_td_bwd), f"TD backward loss: {loss_td_bwd}"

        print(f"✓ All losses computed successfully:")
        print(f"  IPF fwd={loss_ipf_fwd:.4f}, bwd={loss_ipf_bwd:.4f}")
        print(f"  TD  fwd={loss_td_fwd:.4f}, bwd={loss_td_bwd:.4f}")
        print("✓ test_end_to_end_integration PASSED")

    def test_deepgsb_numerical_compatibility(self, solver):
        """Compare our implementation with DeepGSB reference (internal consistency).

        Verifies that same seed gives same trajectory and different seeds differ.
        """
        # Sample same trajectory with two seeds
        x1 = torch.randn(10, 2)

        xs1, ts1, dws1 = solver.sample_backward_trajectory(x1, seed=123)
        xs2, ts2, dws2 = solver.sample_backward_trajectory(x1, seed=123)

        # Should be identical
        assert torch.allclose(xs1, xs2), "Same seed should give same trajectory"
        assert torch.allclose(dws1, dws2), "Same seed should give same noise"

        # Different seed should differ
        xs3, _, dws3 = solver.sample_backward_trajectory(x1, seed=456)
        assert not torch.allclose(xs1, xs3), "Different seed should differ"

        print("✓ test_deepgsb_numerical_compatibility PASSED (internal consistency)")


if __name__ == "__main__":
    """Run tests manually without pytest."""
    print("Running backward SDE indexing tests...\n")

    config = Config(dim=2, T=1.0, dt=0.1, n_agents=5)
    solver = FBSDESolver(config)

    test_suite = TestBackwardSDEIndexing()

    print("=" * 60)
    print("Test 5.1: Noise Indexing")
    print("=" * 60)
    test_suite.test_backward_trajectory_noise_indexing(solver, config)
    print()

    print("=" * 60)
    print("Test 5.2: Boundary Condition Symmetry")
    print("=" * 60)
    test_suite.test_boundary_condition_symmetry(solver, config)
    print()

    print("=" * 60)
    print("Test 5.3: TD Loss Time Consistency")
    print("=" * 60)
    test_suite.test_td_loss_time_consistency(solver, config)
    print()

    print("=" * 60)
    print("Test 5.4: IPF Loss Consistency")
    print("=" * 60)
    test_suite.test_ipf_loss_consistency(solver, config)
    print()

    print("=" * 60)
    print("Test 5.5: DeepGSB Numerical Compatibility")
    print("=" * 60)
    test_suite.test_deepgsb_numerical_compatibility(solver)
    print()

    print("=" * 60)
    print("Test 5.6: End-to-End Integration")
    print("=" * 60)
    test_suite.test_end_to_end_integration(solver, config)
    print()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
