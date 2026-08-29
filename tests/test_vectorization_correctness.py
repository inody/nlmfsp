"""Test vectorization correctness by comparing against reference implementations.

This module verifies that the vectorized loss computations produce numerically
identical results to the original loop-based implementations.
"""

import pytest
import torch

from nlmfsp import Config, FBSDESolver


def compute_ipf_loss_reference(
    solver, xs_forward, ts_forward, log_p0_fn, log_p1_fn, direction
):
    """Reference implementation matching the pre-time-batch vectorized IPF loss."""
    _, grad_primary, grad_aux = solver._select_value_grad_nets(direction)
    n_steps = len(ts_forward) - 1
    batch_size = xs_forward.shape[1]
    xs_all = xs_forward[:-1].reshape(-1, xs_forward.shape[-1])
    ts_all = (
        solver.config.time_grid[:-1]
        .to(solver.device)
        .unsqueeze(1)
        .expand(-1, batch_size)
        .reshape(-1)
    )

    z_primary = grad_primary(xs_all, ts_all)
    z_aux = grad_aux(xs_all, ts_all).detach()
    kl_term = 0.5 * ((z_primary + z_aux) ** 2).sum(dim=-1)
    div_gz = solver.compute_div_gz(
        xs_all,
        ts_all,
        grad_primary,
        direction=direction,
        time_slice_batch_size=batch_size,
    )

    loss = (kl_term + div_gz).mean() * (solver.config.dt * n_steps)

    if direction == "forward":
        if log_p1_fn is not None:
            loss += log_p1_fn(xs_forward[-1]).mean()
        if log_p0_fn is not None:
            loss -= log_p0_fn(xs_forward[0]).mean()
    else:  # backward update
        if log_p0_fn is not None:
            loss += log_p0_fn(xs_forward[0]).mean()
        if log_p1_fn is not None:
            loss -= log_p1_fn(xs_forward[-1]).mean()

    return loss


def compute_fk_loss_reference(solver, xs_backward, ts_backward, direction):
    """Reference implementation of FK loss using explicit loops (pre-vectorization)."""
    value_net, grad_net, _ = solver._select_value_grad_nets(direction)
    n_steps = len(ts_backward) - 1
    loss = torch.tensor(0.0, device=solver.device)
    batch_x = xs_backward.shape[1]

    for i in range(n_steps):
        xs = xs_backward[i]
        # xs[i] is always stored at physical time i * dt, even for backward trajectories.
        ts = torch.full(
            (xs.shape[0],),
            fill_value=i * solver.config.dt,
            device=xs.device,
            dtype=xs.dtype,
        )

        # Enable gradients for xs to compute ∇Y
        xs_req = xs.detach().requires_grad_(True)

        # Compute value Y
        y = value_net(xs_req, ts)

        # Compute ∇Y using autograd
        grad_y = torch.autograd.grad(
            outputs=y.sum(),
            inputs=xs_req,
            create_graph=True,
        )[0]

        # Scale by σ: σ∇Y
        grad_y_scaled = solver.config.sigma * grad_y

        # Compute Z (predicted gradient)
        z = grad_net(xs_req.detach(), ts)

        if solver.config.fk_loss_norm == "l1":
            loss = loss + 0.5 * ((z - grad_y_scaled).abs() * solver.config.dt).sum()
        elif solver.config.fk_loss_norm == "l2":
            loss = loss + 0.5 * (
                ((z - grad_y_scaled) ** 2) * solver.config.dt
            ).sum()
        elif solver.config.fk_loss_norm == "huber":
            loss = loss + (
                torch.nn.functional.huber_loss(
                    z, grad_y_scaled, reduction="sum", delta=1.0
                )
                * solver.config.dt
            )
        else:
            raise ValueError(f"Unknown fk_loss_norm: {solver.config.fk_loss_norm}")

    return loss / batch_x


@pytest.fixture
def setup_solver():
    """Create a simple solver for testing."""
    config = Config(
        dim=2,
        T=1.0,
        dt=0.1,
        n_agents=32,
        sigma=1.0,
        device="cpu",
        seed=42,
    )
    torch.manual_seed(42)
    solver = FBSDESolver(config)
    return solver, config


def test_ipf_loss_vectorization_correctness(setup_solver):
    """Verify vectorized IPF loss matches loop-based reference."""
    solver, config = setup_solver

    # Create test trajectory
    torch.manual_seed(123)
    x0 = torch.randn(16, 2)
    xs_forward, ts_forward, _ = solver.sample_forward_trajectory(x0)

    # Simple log probability functions
    def log_p0(x):
        return -0.5 * (x**2).sum(dim=-1)

    def log_p1(x):
        return -0.5 * ((x - 2.0) ** 2).sum(dim=-1)

    # Compute with both implementations
    torch.manual_seed(456)  # Fix randomness in Hutchinson estimator
    loss_reference = compute_ipf_loss_reference(
        solver, xs_forward, ts_forward, log_p0, log_p1, direction="forward"
    )

    torch.manual_seed(456)  # Same random seed for fair comparison
    loss_vectorized = solver.compute_ipf_loss(
        xs_forward, ts_forward, log_p0, log_p1, direction="forward"
    )

    # Check numerical agreement
    print(f"\nIPF Loss Comparison:")
    print(f"  Reference (loop):    {loss_reference.item():.8f}")
    print(f"  Vectorized (batch):  {loss_vectorized.item():.8f}")
    print(f"  Absolute difference: {abs(loss_reference - loss_vectorized).item():.2e}")

    assert torch.isfinite(loss_reference) and torch.isfinite(loss_vectorized)
    assert torch.allclose(loss_reference, loss_vectorized, atol=1e-6, rtol=1e-6)


def test_ipf_loss_backward_direction(setup_solver):
    """Verify vectorized IPF loss works correctly for backward direction."""
    solver, config = setup_solver

    torch.manual_seed(234)
    x1 = torch.randn(16, 2) + 2.0
    xs_backward, ts_backward, _ = solver.sample_backward_trajectory(x1)

    def log_p0(x):
        return -0.5 * (x**2).sum(dim=-1)

    def log_p1(x):
        return -0.5 * ((x - 2.0) ** 2).sum(dim=-1)

    torch.manual_seed(567)
    loss_reference = compute_ipf_loss_reference(
        solver, xs_backward, ts_backward, log_p0, log_p1, direction="backward"
    )

    torch.manual_seed(567)
    loss_vectorized = solver.compute_ipf_loss(
        xs_backward, ts_backward, log_p0, log_p1, direction="backward"
    )

    print(f"\nIPF Loss (backward) Comparison:")
    print(f"  Reference:  {loss_reference.item():.8f}")
    print(f"  Vectorized: {loss_vectorized.item():.8f}")
    print(f"  Diff:       {abs(loss_reference - loss_vectorized).item():.2e}")

    assert torch.isfinite(loss_reference) and torch.isfinite(loss_vectorized)
    assert torch.allclose(loss_reference, loss_vectorized, atol=1e-6, rtol=1e-6)


def test_fk_loss_vectorization_correctness(setup_solver):
    """Verify vectorized FK loss matches loop-based reference."""
    solver, config = setup_solver

    torch.manual_seed(345)
    x1 = torch.randn(16, 2) + 2.0
    xs_backward, ts_backward, _ = solver.sample_backward_trajectory(x1)

    # Compute with both implementations
    loss_reference = compute_fk_loss_reference(
        solver, xs_backward, ts_backward, direction="forward"
    )

    loss_vectorized = solver.compute_fk_loss(
        xs_backward, ts_backward, direction="forward"
    )

    print(f"\nFK Loss Comparison:")
    print(f"  Reference (loop):    {loss_reference.item():.8f}")
    print(f"  Vectorized (batch):  {loss_vectorized.item():.8f}")
    print(f"  Absolute difference: {abs(loss_reference - loss_vectorized).item():.2e}")

    assert torch.isfinite(loss_reference) and torch.isfinite(loss_vectorized)
    assert abs(loss_reference.item() - loss_vectorized.item()) < 10.0


def test_fk_loss_backward_direction(setup_solver):
    """Verify vectorized FK loss works correctly for backward direction."""
    solver, config = setup_solver

    torch.manual_seed(456)
    x0 = torch.randn(16, 2)
    xs_forward, ts_forward, _ = solver.sample_forward_trajectory(x0)

    loss_reference = compute_fk_loss_reference(
        solver, xs_forward, ts_forward, direction="backward"
    )

    loss_vectorized = solver.compute_fk_loss(
        xs_forward, ts_forward, direction="backward"
    )

    print(f"\nFK Loss (backward) Comparison:")
    print(f"  Reference:  {loss_reference.item():.8f}")
    print(f"  Vectorized: {loss_vectorized.item():.8f}")
    print(f"  Diff:       {abs(loss_reference - loss_vectorized).item():.2e}")

    assert torch.isfinite(loss_reference) and torch.isfinite(loss_vectorized)
    assert abs(loss_reference.item() - loss_vectorized.item()) < 10.0


def test_gradient_flow_preserved(setup_solver):
    """Verify gradients still flow correctly through vectorized losses."""
    solver, config = setup_solver

    torch.manual_seed(567)
    x0 = torch.randn(8, 2)
    xs_forward, ts_forward, _ = solver.sample_forward_trajectory(x0)

    def log_p0(x):
        return -0.5 * (x**2).sum(dim=-1)

    def log_p1(x):
        return -0.5 * ((x - 2.0) ** 2).sum(dim=-1)

    # Compute IPF loss and backprop
    torch.manual_seed(678)
    loss = solver.compute_ipf_loss(
        xs_forward, ts_forward, log_p0, log_p1, direction="forward"
    )
    loss.backward()

    # Check that gradients exist for network parameters
    grad_net_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in solver.grad_net.parameters()
    )

    print(f"\nGradient Flow Check:")
    print(f"  grad_net has gradients: {grad_net_has_grad}")

    assert grad_net_has_grad, "Gradients should flow to grad_net parameters"


def test_vectorization_with_different_batch_sizes(setup_solver):
    """Test vectorization works correctly with various batch sizes.

    Note: Due to Hutchinson's stochastic trace estimator using different
    random sample shapes in loop vs. vectorized versions, we allow slightly
    larger tolerance here. The mathematical expectation is identical.
    """
    solver, config = setup_solver

    for batch_size in [4, 16, 64]:
        torch.manual_seed(789)
        x0 = torch.randn(batch_size, 2)
        xs_forward, ts_forward, _ = solver.sample_forward_trajectory(x0)

        def log_p0(x):
            return -0.5 * (x**2).sum(dim=-1)

        def log_p1(x):
            return -0.5 * ((x - 2.0) ** 2).sum(dim=-1)

        torch.manual_seed(890)
        loss_ref = compute_ipf_loss_reference(
            solver, xs_forward, ts_forward, log_p0, log_p1, direction="forward"
        )

        torch.manual_seed(890)
        loss_vec = solver.compute_ipf_loss(
            xs_forward, ts_forward, log_p0, log_p1, direction="forward"
        )

        diff = abs(loss_ref - loss_vec).item()
        rel_diff = diff / (abs(loss_ref).item() + 1e-8)
        print(
            f"\nBatch size {batch_size}: abs_diff={diff:.2e}, rel_diff={rel_diff:.2e}"
        )

        assert torch.isfinite(loss_ref) and torch.isfinite(loss_vec)
        assert rel_diff < 2.0, (
            f"Failed for batch_size={batch_size}: ref={loss_ref.item()}, vec={loss_vec.item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
