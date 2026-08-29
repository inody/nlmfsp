"""
Base cost functions (no interaction)
"""

import numpy as np
import torch


class QuadraticCost:
    """
    F(x) = (x_target - x)^T Q (x_target - x)
    """

    def __init__(self, Q, x_target):
        self.Q = Q  # np.ndarray or torch.Tensor
        self.x_target = x_target

    def __call__(self, x):
        diff = x - self.x_target
        if isinstance(self.Q, np.ndarray):
            return diff.T @ self.Q @ diff
        else:
            # Use @ operator which handles 1D tensors correctly
            return diff @ self.Q @ diff


class ObstacleCost:
    """
    Penalty if x is within radius of center
    """

    def __init__(self, center, radius, penalty):
        self.center = center
        self.radius = radius
        self.penalty = penalty

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            dist = np.linalg.norm(x - self.center)
        else:
            dist = torch.norm(x - self.center)
        return self.penalty if dist < self.radius else 0.0


class SmoothObstacleCost:
    """
    Differentiable barrier around circular obstacles.

    Uses a smooth approximation so gradients remain informative near the boundary.
    By default we apply a logistic transition:
        penalty * sigmoid(beta * (radius - ||x - center||))
    where beta controls how sharp the wall is (larger beta -> closer to indicator).
    """

    def __init__(
        self, center: torch.Tensor, radius: float, penalty: float, *, beta: float = 5.0
    ):
        if beta <= 0:
            raise ValueError("beta must be > 0 for SmoothObstacleCost.")
        self.center = center
        self.radius = radius
        self.penalty = penalty
        self.beta = beta

    def __call__(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xs: (..., dim) tensor

        Returns:
            (...,) tensor of smooth penalties.
        """
        if xs.ndim == 1:
            # Promote to batch for unified math, then squeeze afterwards.
            xs = xs.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Ensure operand on same device as xs to avoid device mismatches.
        center = self.center.to(xs.device, dtype=xs.dtype)
        dist = torch.norm(xs - center, dim=-1)
        scaled = self.beta * (self.radius - dist)
        cost = self.penalty * torch.sigmoid(scaled)
        return cost.squeeze(0) if squeeze else cost


class VNeckObstacleCost:
    """
    V-neck parabolic obstacle (Liu et al. 2022)

    Penalizes points inside V-shaped region: coef*x² - y² < -c_sq

    Args:
        c_sq: Threshold parameter (default: 0.36)
        coef: Coefficient for x² term (default: 5.0)
        penalty: Cost value inside obstacle (default: 1500.0)
    """

    def __init__(self, c_sq: float = 0.36, coef: float = 5.0, penalty: float = 1500.0):
        self.c_sq = c_sq
        self.coef = coef
        self.penalty = penalty

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, dim) or (dim,) tensor

        Returns:
            (batch,) or scalar tensor of costs
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        x_sq = x[..., 0] ** 2
        y_sq = x[..., 1] ** 2
        d = self.coef * x_sq - y_sq

        cost = self.penalty * (d < -self.c_sq).float()

        return cost.squeeze() if squeeze else cost


class STunnelObstacleCost:
    """
    S-tunnel elliptic obstacles (Liu et al. 2022)

    Two elliptic obstacles forming S-shaped tunnel.
    Penalizes points inside ellipses: a*(x-cx)² + b*(y-cy)² < c

    Args:
        ellipses: List of dicts with keys: center, a, b, c, penalty

    Example:
        >>> ellipses = [
        ...     {"center": [5, 6], "a": 20, "b": 1, "c": 90, "penalty": 1500},
        ...     {"center": [-5, -6], "a": 20, "b": 1, "c": 90, "penalty": 1500}
        ... ]
        >>> cost = STunnelObstacleCost(ellipses)
    """

    def __init__(self, ellipses: list):
        self.ellipses = ellipses

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, dim) or (dim,) tensor

        Returns:
            (batch,) or scalar tensor of costs
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        total_cost = torch.zeros(x.shape[0], device=x.device)

        for ellipse in self.ellipses:
            center = torch.tensor(
                ellipse["center"], device=x.device, dtype=torch.float32
            )
            a = ellipse["a"]
            b = ellipse["b"]
            c = ellipse["c"]
            penalty = ellipse["penalty"]

            dx = x[..., 0] - center[0]
            dy = x[..., 1] - center[1]
            d = a * dx**2 + b * dy**2

            cost = penalty * (d < c).float()
            total_cost += cost

        return total_cost.squeeze() if squeeze else total_cost


class ZeroBaseCost:
    """
    F(x) = 0 (zero cost)
    """

    def __call__(self, x):
        return 0.0
