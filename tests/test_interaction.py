"""Tests for interaction learning with mathematical validation."""

import pytest
import torch

from nlmfsp.config import Config
from nlmfsp.interaction import InteractionLearner


def test_linear_interaction():
    """Test learning a linear mean-field interaction: f(x,ρ) = E[X] (mean of distribution).

    This tests whether the network can learn:
        f(x,t,ρ) = ∫ y ρ(y,t) dy ≈ (1/N)Σ_j x_j

    Mathematically: The network should approximate the empirical mean.
    We use a trajectory with controlled mean to make the test more stable.
    """
    torch.manual_seed(42)
    config = Config(
        dim=2,
        T=1.0,
        dt=0.1,
        n_agents=100,
        n_interaction_steps=1000,
        hidden_dims=(64, 64),
        learning_rate_interaction=1e-3,
    )

    # Define interaction: f(x, ρ) = mean(ρ) (independent of x)
    def f_interaction(x, empirical_samples, t=None):
        """Compute mean of empirical distribution."""
        return empirical_samples.mean(dim=0, keepdim=True).expand_as(x)

    # Define cost interaction: F(x, ρ) = 0
    def F_interaction(x, empirical_samples):
        return torch.zeros(x.shape[0], 1, device=x.device)

    # Create learner
    learner = InteractionLearner(config, f_interaction, F_interaction)

    # Generate trajectory with slowly varying mean (more realistic)
    n_steps = int(config.T / config.dt)
    xs = torch.zeros(n_steps + 1, config.n_agents or 256, config.dim, device=config.device)
    time_grid = torch.linspace(0, config.T, n_steps + 1, device=config.device)

    # Mean drifts from [0,0] to [1,1] over time
    for i in range(n_steps + 1):
        mean_at_t = time_grid[i] * torch.ones(config.dim, device=config.device)
        xs[i] = mean_at_t + 0.5 * torch.randn(config.n_agents or 256, config.dim, device=config.device)

    # Train
    print("\n  Training linear interaction...")
    history = learner.train(xs, time_grid, n_steps=1000)

    final_f_loss = history["f_loss"][-1]
    print(f"  Final f_loss: {final_f_loss:.6f}")

    # Mathematical validation: Test at multiple time points
    errors = []
    for t_idx in [0, n_steps//2, n_steps]:
        t_val = time_grid[t_idx]
        empirical_samples_train = xs[t_idx]
        true_mean = empirical_samples_train.mean(dim=0)

        x_test = torch.randn(50, config.dim, device=config.device)
        t_test = torch.full((50,), t_val.item(), device=config.device)

        f_pred, _ = learner.evaluate(x_test, t_test)
        pred_mean = f_pred.mean(dim=0)

        rel_error = (pred_mean - true_mean).norm() / (true_mean.norm() + 1e-8)
        errors.append(rel_error.item())

    avg_error = sum(errors) / len(errors)
    print(f"  Average relative error across times: {avg_error:.6f}")

    # Check convergence
    assert final_f_loss < 0.01, f"Training did not converge: loss = {final_f_loss}"
    assert avg_error < 0.15, f"Approximation error too large: {avg_error}"

    print("✓ Linear interaction learning test passed")


def test_quadratic_cost_interaction():
    """Test learning quadratic cost: F(x,ρ) = ‖x - E[X]‖².

    Mathematically: The network should learn the variance from the mean.
    """
    torch.manual_seed(42)
    config = Config(
        dim=2,
        T=1.0,
        dt=0.1,
        n_agents=100,
        n_interaction_steps=1000,
        hidden_dims=(64, 64),
        learning_rate_interaction=1e-3,
    )

    # Define drift interaction: f = 0
    def f_interaction(x, empirical_samples, t=None):
        return torch.zeros_like(x)

    # Define cost interaction: F(x, ρ) = ‖x - mean(ρ)‖²
    def F_interaction(x, empirical_samples):
        mean = empirical_samples.mean(dim=0, keepdim=True)
        cost = ((x - mean) ** 2).sum(dim=1, keepdim=True)
        return cost

    # Create learner
    learner = InteractionLearner(config, f_interaction, F_interaction)

    # Generate random trajectory
    n_steps = int(config.T / config.dt)
    xs = torch.randn(n_steps + 1, config.n_agents or 256, config.dim, device=config.device)
    time_grid = torch.linspace(0, config.T, n_steps + 1, device=config.device)

    # Train
    print("\n  Training quadratic cost interaction...")
    history = learner.train(xs, time_grid, n_steps=1000)

    final_F_loss = history["F_loss"][-1]
    print(f"  Final F_loss: {final_F_loss:.6f}")

    # Mathematical validation
    # Test on training data at a specific time
    t_idx = 5
    t_val = time_grid[t_idx]
    empirical_samples_train = xs[t_idx]

    # Test points
    x_test = torch.randn(50, config.dim, device=config.device)
    t_test = torch.full((50,), t_val.item(), device=config.device)

    # True interaction for test points
    F_true = F_interaction(x_test, empirical_samples_train)

    # Learned interaction
    _, F_pred = learner.evaluate(x_test, t_test)

    # Compute relative error
    rel_error = (F_pred - F_true).norm() / F_true.norm()
    print(f"  Relative error on test set: {rel_error.item():.6f}")

    # Check convergence
    assert final_F_loss < 0.5, f"Training did not converge: loss = {final_F_loss}"
    assert rel_error < 0.25, f"Approximation error too large: {rel_error.item()}"

    print("✓ Quadratic cost interaction learning test passed")


def test_gradient_interaction():
    """Test learning gradient interaction: f(x,ρ) = -∇W * ρ for W(x) = ‖x‖²/2.

    This is a typical McKean-Vlasov interaction:
        f(x,ρ) = -∫ ∇W(x-y) ρ(y) dy = -∫ (x-y) ρ(y) dy = -x + E[Y]

    Mathematically: Should learn f(x,ρ) = E[X] - x
    """
    torch.manual_seed(42)
    config = Config(
        dim=2,
        T=1.0,
        dt=0.1,
        n_agents=100,
        n_interaction_steps=1000,
        hidden_dims=(64, 64),
        learning_rate_interaction=1e-3,
    )

    # Define gradient interaction: f(x,ρ) = mean(ρ) - x
    def f_interaction(x, empirical_samples, t=None):
        mean = empirical_samples.mean(dim=0, keepdim=True)
        return mean.expand_as(x) - x

    def F_interaction(x, empirical_samples):
        return torch.zeros(x.shape[0], 1, device=x.device)

    # Create learner
    learner = InteractionLearner(config, f_interaction, F_interaction)

    # Generate random trajectory
    n_steps = int(config.T / config.dt)
    xs = torch.randn(n_steps + 1, config.n_agents or 256, config.dim, device=config.device)
    time_grid = torch.linspace(0, config.T, n_steps + 1, device=config.device)

    # Train
    print("\n  Training gradient interaction...")
    history = learner.train(xs, time_grid, n_steps=1000)

    final_f_loss = history["f_loss"][-1]
    print(f"  Final f_loss: {final_f_loss:.6f}")

    # Mathematical validation: Test linearity
    # For this interaction, f(x,ρ) = μ - x where μ = mean(ρ)
    t_idx = 5
    t_val = time_grid[t_idx]
    empirical_samples_train = xs[t_idx]

    x_test = torch.randn(50, config.dim, device=config.device)
    t_test = torch.full((50,), t_val.item(), device=config.device)

    # True interaction using training distribution
    f_true = f_interaction(x_test, empirical_samples_train)

    # Learned interaction
    f_pred, _ = learner.evaluate(x_test, t_test)

    # Compute relative error
    rel_error = (f_pred - f_true).norm() / f_true.norm()
    print(f"  Relative error on test set: {rel_error.item():.6f}")

    # Additional test: Check if it satisfies f(x,ρ) ≈ const - x
    # This means ∂f/∂x ≈ -I
    x_test.requires_grad_(True)
    f_pred_grad, _ = learner.f_net(x_test, t_test), None

    # For each component, compute gradient
    gradients = []
    for i in range(config.dim):
        grad = torch.autograd.grad(
            f_pred_grad[:, i].sum(),
            x_test,
            retain_graph=True,
            create_graph=False,
        )[0]
        gradients.append(grad)

    # Stack gradients: (batch_size, dim, dim)
    jacobian = torch.stack([g for g in gradients], dim=1)

    # Average Jacobian over batch
    avg_jacobian = jacobian.mean(dim=0)
    expected_jacobian = -torch.eye(config.dim, device=config.device)

    jacobian_error = (avg_jacobian - expected_jacobian).norm() / expected_jacobian.norm()
    print(f"  Jacobian error (should be ≈ -I): {jacobian_error.item():.6f}")

    # Check convergence
    assert final_f_loss < 0.01, f"Training did not converge: loss = {final_f_loss}"
    assert rel_error < 0.15, f"Approximation error too large: {rel_error.item()}"
    assert jacobian_error < 0.3, f"Jacobian structure not learned: {jacobian_error.item()}"

    print("✓ Gradient interaction learning test passed")


def test_time_dependent_interaction():
    """Test learning time-dependent interaction: f(x,t,ρ) = t * mean(ρ).

    Mathematically: Should learn that interaction scales linearly with time.
    """
    torch.manual_seed(42)
    config = Config(
        dim=2,
        T=1.0,
        dt=0.1,
        n_agents=100,
        n_interaction_steps=1000,
        hidden_dims=(64, 64),
        learning_rate_interaction=1e-3,
    )

    # Define time-dependent interaction
    def f_interaction_factory(t_val):
        def f_interaction(x, empirical_samples, t=None):
            mean = empirical_samples.mean(dim=0, keepdim=True)
            return t_val * mean.expand_as(x)
        return f_interaction

    def F_interaction(x, empirical_samples):
        return torch.zeros(x.shape[0], 1, device=x.device)

    # We need to modify compute_true_interactions to handle time-dependence
    # For simplicity, we'll create trajectory and compute interactions manually
    n_steps = int(config.T / config.dt)
    xs = torch.randn(n_steps + 1, config.n_agents or 256, config.dim, device=config.device)
    time_grid = torch.linspace(0, config.T, n_steps + 1, device=config.device)

    # Compute time-dependent targets
    f_targets = torch.zeros_like(xs)
    for i in range(n_steps + 1):
        t_val = time_grid[i].item()
        empirical_samples = xs[i]
        f_interaction = f_interaction_factory(t_val)
        f_targets[i] = f_interaction(xs[i], empirical_samples)

    F_targets = torch.zeros(n_steps + 1, config.n_agents or 256, 1, device=config.device)

    # Create learner with dummy functions (we'll override targets)
    learner = InteractionLearner(config, lambda x, s, t: x * 0, F_interaction)

    # Train with precomputed targets
    print("\n  Training time-dependent interaction...")
    history = {"f_loss": [], "F_loss": []}
    for step in range(1000):
        f_loss, F_loss = learner.train_step(xs, time_grid, f_targets, F_targets)
        history["f_loss"].append(f_loss)
        history["F_loss"].append(F_loss)

    final_f_loss = history["f_loss"][-1]
    print(f"  Final f_loss: {final_f_loss:.6f}")

    # Mathematical validation: Check time-dependence
    x_test = torch.randn(50, config.dim, device=config.device)
    # Test at different times
    times = [0.0, 0.5, 1.0]
    errors = []
    for t_val in times:
        t_test = torch.tensor([t_val] * 50, device=config.device)
        t_idx = int(round(t_val / config.dt))
        mean_empirical = xs[t_idx].mean(dim=0)
        f_true = t_val * mean_empirical.unsqueeze(0).expand(50, -1)
        f_pred, _ = learner.evaluate(x_test, t_test)
        error = (f_pred - f_true).norm() / (f_true.norm() + 1e-8)
        errors.append(error.item())
        print(f"  Time t={t_val:.1f}: relative error = {error.item():.6f}")

    avg_error = sum(errors[1:]) / len(errors[1:])

    # Check convergence
    assert final_f_loss < 0.02, f"Training did not converge: loss = {final_f_loss}"
    assert avg_error < 0.5, f"Time-dependent approximation error too large: {avg_error}"

    print("✓ Time-dependent interaction learning test passed")


def test_evaluate_requires_both_interaction_networks():
    """InteractionLearner.evaluate should fail fast on partial initialization."""
    config = Config(
        dim=2,
        T=1.0,
        dt=0.1,
        n_agents=32,
        learning_rate_interaction=1e-3,
    )

    def f_interaction(x, empirical_samples, t=None):
        return empirical_samples.mean(dim=0, keepdim=True).expand_as(x)

    def F_interaction(x, empirical_samples):
        mean = empirical_samples.mean(dim=0, keepdim=True)
        return ((x - mean) ** 2).sum(dim=1, keepdim=True)

    learner_drift_only = InteractionLearner(config, f_interaction, None)
    learner_cost_only = InteractionLearner(config, None, F_interaction)
    x_test = torch.randn(4, config.dim, device=config.device)
    t_test = torch.zeros(4, device=config.device)

    with pytest.raises(ValueError, match="F_net"):
        learner_drift_only.evaluate(x_test, t_test)
    with pytest.raises(ValueError, match="f_net"):
        learner_cost_only.evaluate(x_test, t_test)


if __name__ == "__main__":
    print("\n=== Testing Interaction Learning (with mathematical validation) ===")
    test_linear_interaction()
    test_quadratic_cost_interaction()
    test_gradient_interaction()
    test_time_dependent_interaction()
    print("\n=== All interaction learning tests passed! ===\n")
