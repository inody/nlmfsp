"""Tests for analytical interaction functions."""

import torch
from nlmfsp.analytical_interactions import (
    AnalyticalGaussianConvolutionMeanCost,
    AnalyticalPedestrianVisualCost,
    AnalyticalMeanReversionDrift,
    AnalyticalGaussianDrift,
    AnalyticalSmoothBoundedConfidenceDrift,
    AnalyticalZeroDrift,
    AnalyticalQuadraticDeviationCost,
    AnalyticalGaussianConvolutionCost,
    AnalyticalZeroCost,
)


def test_mean_reversion_drift():
    """Test AnalyticalMeanReversionDrift with empirical mean."""
    drift = AnalyticalMeanReversionDrift(weight=0.5)

    x = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    t = torch.tensor([0.5, 0.5])

    f = drift(x, t)

    mean = x.mean(dim=0, keepdim=True)
    expected = 0.5 * (mean - x)

    assert f.shape == x.shape
    assert torch.allclose(f, expected, atol=1e-6)

    print("✓ Mean reversion drift test passed")


def test_zero_drift():
    """Test AnalyticalZeroDrift."""
    drift = AnalyticalZeroDrift()

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    t = torch.tensor([0.5, 0.5])

    f = drift(x, t)

    expected = torch.zeros_like(x)

    assert f.shape == x.shape
    assert torch.allclose(f, expected, atol=1e-6)

    print("✓ Zero drift test passed")


def test_gaussian_drift():
    """Test AnalyticalGaussianDrift."""
    drift = AnalyticalGaussianDrift(sigma=1.0)

    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    t = torch.tensor([0.5, 0.5, 0.5])

    f = drift(x, t)

    assert f.shape == x.shape

    print("✓ Gaussian drift test passed")


def test_smooth_bounded_confidence_drift_matches_weighted_average():
    """Test smooth bounded-confidence drift against a hand-computable case."""
    drift = AnalyticalSmoothBoundedConfidenceDrift(sigma=1.0, weight=2.0)

    x = torch.tensor([[0.0], [1.0]])
    t = torch.tensor([0.0, 0.0])

    f = drift(x, t)

    k = torch.exp(torch.tensor(-0.5))
    expected = torch.tensor([[2.0 * k / (1.0 + k)], [-2.0 * k / (1.0 + k)]])

    assert f.shape == x.shape
    assert torch.allclose(f, expected, atol=1e-6)

    print("✓ Smooth bounded-confidence drift test passed")


def test_quadratic_deviation_cost():
    """Test AnalyticalQuadraticDeviationCost."""
    cost = AnalyticalQuadraticDeviationCost(weight=1.0)

    x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    t = torch.tensor([0.5, 0.5, 0.5])

    F = cost(x, t)

    mean = x.mean(dim=0, keepdim=True)
    expected = ((x - mean) ** 2).sum(dim=1, keepdim=True)

    assert F.shape == (3, 1)
    assert torch.allclose(F, expected, atol=1e-6)

    print("✓ Quadratic deviation cost test passed")


def test_zero_cost():
    """Test AnalyticalZeroCost."""
    cost = AnalyticalZeroCost()

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    t = torch.tensor([0.5, 0.5])

    F = cost(x, t)

    expected = torch.zeros(2, 1)

    assert F.shape == (2, 1)
    assert torch.allclose(F, expected, atol=1e-6)

    print("✓ Zero cost test passed")


def test_gaussian_convolution_cost():
    """Test AnalyticalGaussianConvolutionCost."""
    cost = AnalyticalGaussianConvolutionCost(sigma=1.0)

    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    t = torch.tensor([0.5, 0.5, 0.5])

    F = cost(x, t)

    assert F.shape == (3, 1)
    assert torch.all(F > 0)

    print("✓ Gaussian convolution cost test passed")


def test_gaussian_convolution_mean_cost_matches_empirical_average():
    """Test AnalyticalGaussianConvolutionMeanCost with a hand-computable example."""
    cost = AnalyticalGaussianConvolutionMeanCost(sigma=1.0, weight=2.0)

    x = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    t = torch.tensor([0.5, 0.5])

    F = cost(x, t)

    kernel_same = 1.0
    kernel_offset = torch.exp(torch.tensor(-0.5)).item()
    expected_value = 2.0 * (kernel_same + kernel_offset) / 2.0
    expected = torch.full((2, 1), expected_value)

    assert F.shape == (2, 1)
    assert torch.allclose(F, expected, atol=1e-6)

    print("✓ Gaussian convolution mean cost test passed")


def test_pedestrian_visual_cost_is_directional_and_excludes_self():
    cost = AnalyticalPedestrianVisualCost(range_scale=1.0, weight=2.0)
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    t = torch.zeros(2)

    values = cost(x, t)

    expected = torch.tensor([[2.0 * torch.exp(torch.tensor(-1.0))], [0.0]])
    assert torch.allclose(values, expected, atol=1e-6)


def test_scalar_time_input():
    """Test that all functions work with scalar time input."""
    drift = AnalyticalMeanReversionDrift(weight=0.5)
    cost = AnalyticalQuadraticDeviationCost(weight=1.0)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    t_scalar = torch.tensor(0.5)

    f = drift(x, t_scalar)
    F = cost(x, t_scalar)

    assert f.shape == x.shape
    assert F.shape == (2, 1)

    print("✓ Scalar time input test passed")


def test_interface_consistency():
    """Test that all classes follow the same interface."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    t = torch.tensor([0.5, 0.5])

    drifts = [
        AnalyticalMeanReversionDrift(),
        AnalyticalZeroDrift(),
        AnalyticalGaussianDrift(),
        AnalyticalSmoothBoundedConfidenceDrift(),
    ]

    for drift in drifts:
        f = drift(x, t)
        assert f.shape == x.shape, f"Drift {drift.__class__.__name__} shape mismatch"

    costs = [
        AnalyticalZeroCost(),
        AnalyticalQuadraticDeviationCost(),
        AnalyticalGaussianConvolutionCost(),
    ]

    for cost in costs:
        F = cost(x, t)
        assert F.shape == (2, 1), f"Cost {cost.__class__.__name__} shape mismatch"

    print("✓ Interface consistency test passed")


def test_mean_reversion_interaction_dependency():
    """Test that MeanReversionDrift output changes with empirical distribution.
    
    This verifies it's a true interaction function that depends on all particles,
    not just a single-particle function.
    """
    drift = AnalyticalMeanReversionDrift(weight=1.0)
    
    x = torch.tensor([[0.0]])
    t = torch.tensor([0.0])
    
    # Distribution 1: mean = 1.0
    dist1 = torch.tensor([[1.0]])
    x_combined1 = torch.cat([x, dist1], dim=0)
    f1 = drift(x_combined1, t.expand(2))
    
    # Distribution 2: mean = 2.0
    dist2 = torch.tensor([[3.0]])
    x_combined2 = torch.cat([x, dist2], dim=0)
    f2 = drift(x_combined2, t.expand(2))
    
    # The drift at x=0 should be different for different empirical distributions
    # because it depends on the empirical mean
    assert not torch.allclose(f1[0], f2[0], atol=1e-6), \
        "MeanReversionDrift should depend on empirical distribution"
    
    # Verify the expected values
    # dist1: mean = 0.5, f(0) = 1.0 * (0.5 - 0) = 0.5
    # dist2: mean = 1.5, f(0) = 1.0 * (1.5 - 0) = 1.5
    assert torch.allclose(f1[0], torch.tensor([0.5]), atol=1e-6)
    assert torch.allclose(f2[0], torch.tensor([1.5]), atol=1e-6)
    
    print("✓ Mean reversion interaction dependency test passed")


def test_quadratic_deviation_interaction_dependency():
    """Test that QuadraticDeviationCost output changes with empirical distribution.
    
    This verifies it's a true interaction function that depends on all particles,
    not just a single-particle function.
    """
    cost = AnalyticalQuadraticDeviationCost(weight=1.0)
    
    x = torch.tensor([[0.0]])
    t = torch.tensor([0.0])
    
    # Distribution 1: mean = 1.0
    dist1 = torch.tensor([[1.0]])
    x_combined1 = torch.cat([x, dist1], dim=0)
    F1 = cost(x_combined1, t.expand(2))
    
    # Distribution 2: mean = 2.0
    dist2 = torch.tensor([[3.0]])
    x_combined2 = torch.cat([x, dist2], dim=0)
    F2 = cost(x_combined2, t.expand(2))
    
    # The cost at x=0 should be different for different empirical distributions
    assert not torch.allclose(F1[0], F2[0], atol=1e-6), \
        "QuadraticDeviationCost should depend on empirical distribution"
    
    # Verify the expected values
    # dist1: mean = 0.5, F(0) = (0 - 0.5)^2 = 0.25
    # dist2: mean = 1.5, F(0) = (0 - 1.5)^2 = 2.25
    assert torch.allclose(F1[0], torch.tensor([0.25]), atol=1e-6)
    assert torch.allclose(F2[0], torch.tensor([2.25]), atol=1e-6)
    
    print("✓ Quadratic deviation interaction dependency test passed")


def test_gaussian_drift_interaction_dependency():
    """Test that GaussianDrift output changes with empirical distribution."""
    drift = AnalyticalGaussianDrift(sigma=1.0)
    
    x = torch.tensor([[0.0, 0.0]])
    t = torch.tensor([0.0])
    
    # Distribution 1: particle at (1, 0)
    dist1 = torch.tensor([[1.0, 0.0]])
    x_combined1 = torch.cat([x, dist1], dim=0)
    f1 = drift(x_combined1, t.expand(2))
    
    # Distribution 2: particle at (0, 1)
    dist2 = torch.tensor([[0.0, 1.0]])
    x_combined2 = torch.cat([x, dist2], dim=0)
    f2 = drift(x_combined2, t.expand(2))
    
    # The drift at x=(0,0) should be different for different empirical distributions
    assert not torch.allclose(f1[0], f2[0], atol=1e-6), \
        "GaussianDrift should depend on empirical distribution"
    
    # Direction should be roughly towards the empirical mean
    # dist1: should point in +x direction
    # dist2: should point in +y direction
    assert f1[0, 0] > f1[0, 1]  # More in x direction
    assert f2[0, 1] > f2[0, 0]  # More in y direction
    
    print("✓ Gaussian drift interaction dependency test passed")


def test_gaussian_convolution_interaction_dependency():
    """Test that GaussianConvolutionCost output changes with empirical distribution."""
    cost = AnalyticalGaussianConvolutionCost(sigma=1.0)
    
    x = torch.tensor([[0.0, 0.0]])
    t = torch.tensor([0.0])
    
    # Distribution 1: particle nearby at (0.1, 0)
    dist1 = torch.tensor([[0.1, 0.0]])
    x_combined1 = torch.cat([x, dist1], dim=0)
    F1 = cost(x_combined1, t.expand(2))
    
    # Distribution 2: particle far away at (5, 0)
    dist2 = torch.tensor([[5.0, 0.0]])
    x_combined2 = torch.cat([x, dist2], dim=0)
    F2 = cost(x_combined2, t.expand(2))
    
    # The cost at x=(0,0) should be higher when particles are nearby
    assert F1[0] > F2[0], \
        "GaussianConvolutionCost should be higher when particles are nearby"
    
    print("✓ Gaussian convolution interaction dependency test passed")


def test_zero_functions_no_interaction_dependency():
    """Test that Zero functions don't change with empirical distribution.
    
    These are trivial cases but verify they work correctly.
    """
    drift = AnalyticalZeroDrift()
    cost = AnalyticalZeroCost()
    
    x = torch.tensor([[0.0]])
    t = torch.tensor([0.0])
    
    # Two different distributions
    dist1 = torch.tensor([[1.0]])
    dist2 = torch.tensor([[5.0]])
    
    x_combined1 = torch.cat([x, dist1], dim=0)
    x_combined2 = torch.cat([x, dist2], dim=0)
    
    f1 = drift(x_combined1, t.expand(2))
    f2 = drift(x_combined2, t.expand(2))
    
    F1 = cost(x_combined1, t.expand(2))
    F2 = cost(x_combined2, t.expand(2))
    
    # Zero functions should always return zero regardless of distribution
    assert torch.allclose(f1[0], f2[0], atol=1e-6)
    assert torch.allclose(F1[0], F2[0], atol=1e-6)
    assert torch.allclose(f1[0], torch.zeros(1), atol=1e-6)
    assert torch.allclose(F1[0], torch.zeros(1, 1), atol=1e-6)
    
    print("✓ Zero functions no interaction dependency test passed")


if __name__ == "__main__":
    print("\n=== Testing Analytical Interactions ===\n")

    test_mean_reversion_drift()
    test_zero_drift()
    test_gaussian_drift()
    test_quadratic_deviation_cost()
    test_zero_cost()
    test_gaussian_convolution_cost()
    test_gaussian_convolution_mean_cost_matches_empirical_average()
    test_scalar_time_input()
    test_interface_consistency()
    
    print("\n=== Testing Interaction Dependencies ===\n")
    
    test_mean_reversion_interaction_dependency()
    test_quadratic_deviation_interaction_dependency()
    test_gaussian_drift_interaction_dependency()
    test_gaussian_convolution_interaction_dependency()
    test_zero_functions_no_interaction_dependency()

    print("\n=== All tests passed! ===\n")
