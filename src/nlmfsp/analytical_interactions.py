"""Analytical interaction functions for testing and benchmarking.

These functions compute mean-field interactions analytically without
requiring learned neural networks. They are useful for:

1. Validating the FBSDE solver (Option A)
2. Benchmarking against learned interactions (Option B)
3. Testing the training loop without interaction learning overhead

All functions follow the same interface as InteractionNet:
    forward(x: Tensor, t: Tensor) -> Tensor

This ensures compatibility with FBSDESolver and enables fair comparison
between analytical (Option A) and learned (Option B) approaches.

Design: These classes wrap the base interaction classes from costs.py/drifts.py
using composition, adding only a forward() method for compatibility with
the neural network interface.
"""

from __future__ import annotations

import torch
from torch import nn

from .costs import (
    CongestionCost,
    CoulombCost,
    EntropyCost,
    FlockingAlignmentCost,
    GaussianConvolutionCost,
    GaussianConvolutionMeanCost,
    LocalExclusionCost,
    MeanDeviationCost,
    PedestrianVisualCost,
    QuadraticDeviationCost,
    QuadraticPotentialCost,
    ZeroCost,
)
from .drifts import (
    AttractionRepulsionDrift,
    CoulombDrift,
    GaussianDrift,
    LaplacianDiffusionDrift,
    MeanReversionDrift,
    NonstationaryAnisotropicDrift,
    SmoothBoundedConfidenceDrift,
    ZeroDrift,
)

Tensor = torch.Tensor


class AnalyticalMeanReversionDrift(nn.Module):
    """Mean reversion drift to empirical mean: f(x,t) = w (E[X] - x).

    Args:
        weight: Scalar weight w
    """

    drift: MeanReversionDrift

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.drift = MeanReversionDrift(weight=weight)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.drift.evaluate(x, x)


class AnalyticalZeroDrift(nn.Module):
    """Zero drift: f(x) = 0"""

    drift: ZeroDrift

    def __init__(self):
        super().__init__()
        self.drift = ZeroDrift()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.drift.evaluate(x, x)


class AnalyticalGaussianDrift(nn.Module):
    """Non-local Gaussian convolution drift."""

    drift: GaussianDrift

    def __init__(self, sigma: float = 1.0, weight: float = 1.0):
        super().__init__()
        self.drift = GaussianDrift(sigma=sigma, weight=weight)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.drift.evaluate(x, x)


class AnalyticalNonstationaryAnisotropicDrift(nn.Module):
    """Non-stationary anisotropic pairwise drift."""

    drift: NonstationaryAnisotropicDrift

    def __init__(
        self, sigma: float = 1.0, weight: float = 1.0, alpha: float = 1.0
    ):
        super().__init__()
        self.drift = NonstationaryAnisotropicDrift(
            sigma=sigma, weight=weight, alpha=alpha
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.drift.evaluate(x, x)


class AnalyticalAttractionRepulsionDrift(nn.Module):
    """Attraction-repulsion drift."""

    drift: AttractionRepulsionDrift

    def __init__(
        self,
        attract: float = 1.0,
        sigma_attract: float = 1.0,
        repel: float = 0.5,
        sigma_repel: float = 0.3,
        weight: float = 1.0,
    ):
        super().__init__()
        self.drift = AttractionRepulsionDrift(
            attract=attract,
            sigma_attract=sigma_attract,
            repel=repel,
            sigma_repel=sigma_repel,
            weight=weight,
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.drift.evaluate(x, x)


class AnalyticalCoulombDrift(nn.Module):
    """Coulomb drift."""

    drift: CoulombDrift

    def __init__(self, alpha: float = 1.0, eps: float = 0.1, weight: float = 1.0):
        super().__init__()
        self.drift = CoulombDrift(alpha=alpha, eps=eps, weight=weight)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.drift.evaluate(x, x)


class AnalyticalLaplacianDiffusionDrift(nn.Module):
    """Laplacian diffusion drift."""

    drift: LaplacianDiffusionDrift

    def __init__(
        self, sigma: float = 1.0, kernel_type: str = "gaussian", weight: float = 1.0
    ):
        super().__init__()
        self.drift = LaplacianDiffusionDrift(
            sigma=sigma, kernel_type=kernel_type, weight=weight
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.drift.evaluate(x, x)


class AnalyticalSmoothBoundedConfidenceDrift(nn.Module):
    """Smooth bounded-confidence opinion drift."""

    drift: SmoothBoundedConfidenceDrift

    def __init__(self, sigma: float = 1.0, weight: float = 1.0):
        super().__init__()
        self.drift = SmoothBoundedConfidenceDrift(sigma=sigma, weight=weight)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.drift.evaluate(x, x)


class AnalyticalZeroCost(nn.Module):
    """Zero cost: F(x) = 0"""

    cost: ZeroCost

    def __init__(self):
        super().__init__()
        self.cost = ZeroCost()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.cost.evaluate(x, x)


class AnalyticalQuadraticDeviationCost(nn.Module):
    """Quadratic deviation from empirical mean."""

    cost: QuadraticDeviationCost

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.cost = QuadraticDeviationCost(weight=weight)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.cost.evaluate(x, x)


class AnalyticalGaussianConvolutionCost(nn.Module):
    """Smoothed crowd-density cost via Gaussian convolution."""

    cost: GaussianConvolutionCost

    def __init__(self, sigma: float = 1.0, weight: float = 1.0):
        super().__init__()
        self.cost = GaussianConvolutionCost(sigma=sigma, weight=weight)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.cost.evaluate(x, x)


class AnalyticalGaussianConvolutionMeanCost(nn.Module):
    """Smoothed crowd-density cost via Gaussian convolution with mean aggregation."""

    cost: GaussianConvolutionMeanCost

    def __init__(self, sigma: float = 1.0, weight: float = 1.0):
        super().__init__()
        self.cost = GaussianConvolutionMeanCost(sigma=sigma, weight=weight)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.cost.evaluate(x, x)


class AnalyticalPedestrianVisualCost(nn.Module):
    """Pedestrian-inspired, goal-oriented visual congestion cost."""

    cost: PedestrianVisualCost

    def __init__(
        self,
        range_scale: float = 1.0,
        weight: float = 1.0,
        goal: tuple[float, float] = (7.0, 0.0),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.cost = PedestrianVisualCost(
            range_scale=range_scale,
            weight=weight,
            goal=goal,
            eps=eps,
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.cost.evaluate(x, x)


class AnalyticalMeanDeviationCost(nn.Module):
    """Mean deviation cost."""

    cost: MeanDeviationCost

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.cost = MeanDeviationCost(weight=weight)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.cost.evaluate(x, x)


class AnalyticalQuadraticPotentialCost(nn.Module):
    """Quadratic potential interaction: F(x) = E[‖x-X‖²]."""

    cost: QuadraticPotentialCost

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.cost = QuadraticPotentialCost(weight=weight)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.cost.evaluate(x, x)


class AnalyticalCoulombCost(nn.Module):
    """Coulomb cost."""

    cost: CoulombCost

    def __init__(self, alpha: float = 1.0, eps: float = 0.01):
        super().__init__()
        self.cost = CoulombCost(alpha=alpha, eps=eps)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.cost.evaluate(x, x)


class AnalyticalLocalExclusionCost(nn.Module):
    """Local exclusion cost."""

    cost: LocalExclusionCost

    def __init__(self, epsilon: float = 0.5, strength: float = 1.0):
        super().__init__()
        self.cost = LocalExclusionCost(epsilon=epsilon, strength=strength)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.cost.evaluate(x, x)


class AnalyticalEntropyCost(nn.Module):
    """Entropy minimization cost (Liu et al. 2022)."""

    cost: EntropyCost

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.cost = EntropyCost(weight=weight)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.cost.evaluate(x, x)


class AnalyticalCongestionCost(nn.Module):
    """Congestion cost (Liu et al. 2022)."""

    cost: CongestionCost

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.cost = CongestionCost(weight=weight)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.cost.evaluate(x, x)


class AnalyticalFlockingAlignmentCost(nn.Module):
    """Cauchy-kernel flocking alignment cost."""

    cost: FlockingAlignmentCost

    def __init__(
        self,
        sigma: float = 1.0,
        weight: float = 1.0,
        position_dim: int | None = None,
    ):
        super().__init__()
        self.cost = FlockingAlignmentCost(
            sigma=sigma, weight=weight, position_dim=position_dim
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.cost.evaluate(x, x)


__all__ = [
    "AnalyticalMeanReversionDrift",
    "AnalyticalZeroDrift",
    "AnalyticalGaussianDrift",
    "AnalyticalAttractionRepulsionDrift",
    "AnalyticalCoulombDrift",
    "AnalyticalLaplacianDiffusionDrift",
    "AnalyticalSmoothBoundedConfidenceDrift",
    "AnalyticalZeroCost",
    "AnalyticalQuadraticDeviationCost",
    "AnalyticalGaussianConvolutionCost",
    "AnalyticalGaussianConvolutionMeanCost",
    "AnalyticalPedestrianVisualCost",
    "AnalyticalMeanDeviationCost",
    "AnalyticalQuadraticPotentialCost",
    "AnalyticalCoulombCost",
    "AnalyticalLocalExclusionCost",
    "AnalyticalEntropyCost",
    "AnalyticalCongestionCost",
    "AnalyticalFlockingAlignmentCost",
]
