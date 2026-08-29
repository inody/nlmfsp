"""Reusable cost potentials for mean-field games."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import torch

Tensor = torch.Tensor


class CostPotential(ABC):
    """Base class for scalar interaction costs."""

    name: str = "cost_potential"
    formula: str = ""
    random_batch_aggregation: str = "mean"
    query_chunk_size: Optional[int] = None
    ref_chunk_size: Optional[int] = None

    def evaluate(self, x: Tensor, empirical: Optional[Tensor] = None) -> Tensor:
        if empirical is None:
            empirical = x
        return self._evaluate(x, empirical)

    def evaluate_batched(
        self,
        x: Tensor,
        empirical: Optional[Tensor] = None,
        *,
        time_block_size: Optional[int] = None,
    ) -> Tensor:
        """Evaluate exact time-local interactions for `(time, batch, dim)` inputs."""
        if x.dim() != 3:
            raise ValueError(
                f"evaluate_batched expects x with shape (time, batch, dim), got {tuple(x.shape)}"
            )
        if empirical is None:
            empirical = x
        if empirical.dim() != 3:
            raise ValueError(
                "evaluate_batched expects empirical with shape (time, batch, dim)."
            )
        n_steps = x.shape[0]
        block = n_steps if time_block_size is None else min(time_block_size, n_steps)
        out = x.new_empty((n_steps, x.shape[1], 1))
        for start in range(0, n_steps, block):
            end = min(start + block, n_steps)
            out[start:end] = self._evaluate_batched_block(x[start:end], empirical[start:end])
        return out

    def set_chunk_sizes(
        self,
        *,
        query_chunk_size: Optional[int] = None,
        ref_chunk_size: Optional[int] = None,
    ) -> None:
        """Set optional chunk sizes for pairwise interaction evaluation."""
        self.query_chunk_size = query_chunk_size
        self.ref_chunk_size = ref_chunk_size

    @abstractmethod
    def _evaluate(self, x: Tensor, empirical: Tensor) -> Tensor:
        raise NotImplementedError

    def _evaluate_batched_block(self, x: Tensor, empirical: Tensor) -> Tensor:
        return torch.stack(
            [self._evaluate(x_t, empirical_t) for x_t, empirical_t in zip(x, empirical)],
            dim=0,
        )


class ZeroCost(CostPotential):
    """Cost identically zero."""

    random_batch_aggregation = "mean"

    def __init__(self):
        self.name = "zero"
        self.formula = "F(x) = 0"

    def _evaluate(self, x: Tensor, empirical: Tensor) -> Tensor:
        return torch.zeros(x.shape[0], 1, device=x.device)

    def _evaluate_batched_block(self, x: Tensor, empirical: Tensor) -> Tensor:
        return torch.zeros(x.shape[0], x.shape[1], 1, device=x.device)


class QuadraticDeviationCost(CostPotential):
    """Quadratic deviation from empirical mean."""

    random_batch_aggregation = "mean"

    def __init__(self, weight: float = 1.0):
        self.weight = weight
        self.name = "quadratic_mean_gap"
        self.formula = f"F(x) = {self.weight}‖x - 𝔼[X]‖²"

    def _evaluate(self, x: Tensor, empirical: Tensor) -> Tensor:
        mean = empirical.mean(dim=0, keepdim=True)
        return self.weight * ((x - mean) ** 2).sum(dim=1, keepdim=True)

    def _evaluate_batched_block(self, x: Tensor, empirical: Tensor) -> Tensor:
        mean = empirical.mean(dim=1, keepdim=True)
        return self.weight * ((x - mean) ** 2).sum(dim=2, keepdim=True)


class GaussianConvolutionCost(CostPotential):
    """Smoothed crowd-density cost via Gaussian convolution."""

    random_batch_aggregation = "sum"

    def __init__(self, sigma: float = 1.0, weight: float = 1.0):
        self.sigma = sigma
        self.weight = weight
        self.name = "gaussian_conv"
        self.formula = f"F(x) = {weight:.2f}∑ exp(-‖x-X‖²/2σ²), σ={sigma:.2f}"

    def _evaluate(self, x: Tensor, empirical: Tensor) -> Tensor:
        bsz = x.shape[0]
        n_ref = empirical.shape[0]
        q_chunk = (
            bsz
            if self.query_chunk_size is None or self.query_chunk_size <= 0
            else min(self.query_chunk_size, bsz)
        )
        r_chunk = (
            n_ref
            if self.ref_chunk_size is None or self.ref_chunk_size <= 0
            else min(self.ref_chunk_size, n_ref)
        )
        out = x.new_zeros(bsz, 1)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[q_start:q_end]
            acc = x.new_zeros(q_end - q_start, 1)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[r_start:r_end]
                diff = emp.unsqueeze(0) - xq.unsqueeze(1)
                dist_sq = (diff**2).sum(dim=2)
                weights = torch.exp(-dist_sq / (2 * self.sigma**2))
                acc = acc + weights.sum(dim=1, keepdim=True)
            out[q_start:q_end] = acc
        return self.weight * out

    def _evaluate_batched_block(self, x: Tensor, empirical: Tensor) -> Tensor:
        n_steps, bsz, _ = x.shape
        n_ref = empirical.shape[1]
        q_chunk = (
            bsz
            if self.query_chunk_size is None or self.query_chunk_size <= 0
            else min(self.query_chunk_size, bsz)
        )
        r_chunk = (
            n_ref
            if self.ref_chunk_size is None or self.ref_chunk_size <= 0
            else min(self.ref_chunk_size, n_ref)
        )
        out = x.new_zeros(n_steps, bsz, 1)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[:, q_start:q_end]
            acc = x.new_zeros(n_steps, q_end - q_start, 1)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[:, r_start:r_end]
                diff = emp.unsqueeze(1) - xq.unsqueeze(2)
                dist_sq = (diff**2).sum(dim=3)
                weights = torch.exp(-dist_sq / (2 * self.sigma**2))
                acc = acc + weights.sum(dim=2, keepdim=True)
            out[:, q_start:q_end] = acc
        return self.weight * out


class GaussianConvolutionMeanCost(GaussianConvolutionCost):
    """Smoothed crowd-density cost via Gaussian convolution with mean aggregation."""

    random_batch_aggregation = "mean"

    def __init__(self, sigma: float = 1.0, weight: float = 1.0):
        super().__init__(sigma=sigma, weight=weight)
        self.name = "gaussian_conv_mean"
        self.formula = f"F(x) = {weight:.2f}·𝔼[exp(-‖x-X‖²/2σ²)], σ={sigma:.2f}"

    def _evaluate(self, x: Tensor, empirical: Tensor) -> Tensor:
        out = super()._evaluate(x, empirical)
        n_ref = empirical.shape[0]
        if n_ref <= 0:
            return out
        return out / float(n_ref)

    def _evaluate_batched_block(self, x: Tensor, empirical: Tensor) -> Tensor:
        out = super()._evaluate_batched_block(x, empirical)
        n_ref = empirical.shape[1]
        if n_ref <= 0:
            return out
        return out / float(n_ref)


class PedestrianVisualCost(CostPotential):
    """Pedestrian-inspired visual congestion cost.

    The nonsymmetric pairwise kernel gives larger weight to nearby particles
    located in front of the query particle:

        K(x, y) = 0.5 * (1 + <e(x), (y-x) / (||y-x|| + eps)>)
                  * exp(-||y-x|| / B),

    where e(x) points from x toward a fixed goal. When queries and references
    are the same tensor, self-interactions are removed and the empirical
    average uses 1 / (N - 1). For distinct query/reference sets, the average
    uses all reference particles.
    """

    random_batch_aggregation = "mean"

    def __init__(
        self,
        range_scale: float = 1.0,
        weight: float = 1.0,
        goal: tuple[float, float] = (7.0, 0.0),
        eps: float = 1e-6,
    ):
        if range_scale <= 0:
            raise ValueError("PedestrianVisualCost range_scale must be positive")
        if eps <= 0:
            raise ValueError("PedestrianVisualCost eps must be positive")
        self.range_scale = float(range_scale)
        self.weight = float(weight)
        self.goal = tuple(float(value) for value in goal)
        self.eps = float(eps)
        self.name = "pedestrian_visual"
        self.formula = (
            f"F(x) = {weight:.2f} E[p(x,X) exp(-||x-X||/{range_scale:.2f})]"
        )

    @staticmethod
    def _same_particle_set(x: Tensor, empirical: Tensor) -> bool:
        return x.shape == empirical.shape and x.data_ptr() == empirical.data_ptr()

    def _goal_tensor(self, x: Tensor) -> Tensor:
        if x.shape[-1] != len(self.goal):
            raise ValueError(
                "PedestrianVisualCost goal dimension does not match state dimension: "
                f"goal={len(self.goal)}, state={x.shape[-1]}"
            )
        return x.new_tensor(self.goal)

    def _evaluate(self, x: Tensor, empirical: Tensor) -> Tensor:
        bsz = x.shape[0]
        n_ref = empirical.shape[0]
        same_particles = self._same_particle_set(x, empirical)
        denominator = n_ref - 1 if same_particles else n_ref
        if denominator <= 0:
            return x.new_zeros((bsz, 1))

        q_chunk = (
            bsz
            if self.query_chunk_size is None or self.query_chunk_size <= 0
            else min(self.query_chunk_size, bsz)
        )
        r_chunk = (
            n_ref
            if self.ref_chunk_size is None or self.ref_chunk_size <= 0
            else min(self.ref_chunk_size, n_ref)
        )
        goal = self._goal_tensor(x)
        out = x.new_zeros(bsz, 1)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[q_start:q_end]
            goal_diff = goal.unsqueeze(0) - xq
            desired = goal_diff / (
                torch.linalg.vector_norm(goal_diff, dim=1, keepdim=True) + self.eps
            )
            acc = x.new_zeros(q_end - q_start, 1)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                diff = empirical[r_start:r_end].unsqueeze(0) - xq.unsqueeze(1)
                dist = torch.linalg.vector_norm(diff, dim=2)
                direction = diff / (dist.unsqueeze(-1) + self.eps)
                visibility = 0.5 * (
                    1.0 + (desired.unsqueeze(1) * direction).sum(dim=2)
                )
                kernel = visibility * torch.exp(-dist / self.range_scale)
                if same_particles:
                    q_indices = torch.arange(q_start, q_end, device=x.device)
                    mask = (q_indices >= r_start) & (q_indices < r_end)
                    if mask.any():
                        rows = torch.nonzero(mask, as_tuple=False).squeeze(1)
                        cols = q_indices[mask] - r_start
                        kernel[rows, cols] = 0.0
                acc = acc + kernel.sum(dim=1, keepdim=True)
            out[q_start:q_end] = acc / float(denominator)
        return self.weight * out

    def _evaluate_batched_block(self, x: Tensor, empirical: Tensor) -> Tensor:
        n_steps, bsz, _ = x.shape
        n_ref = empirical.shape[1]
        same_particles = self._same_particle_set(x, empirical)
        denominator = n_ref - 1 if same_particles else n_ref
        if denominator <= 0:
            return x.new_zeros((n_steps, bsz, 1))

        q_chunk = (
            bsz
            if self.query_chunk_size is None or self.query_chunk_size <= 0
            else min(self.query_chunk_size, bsz)
        )
        r_chunk = (
            n_ref
            if self.ref_chunk_size is None or self.ref_chunk_size <= 0
            else min(self.ref_chunk_size, n_ref)
        )
        goal = self._goal_tensor(x)
        out = x.new_zeros(n_steps, bsz, 1)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[:, q_start:q_end]
            goal_diff = goal.view(1, 1, -1) - xq
            desired = goal_diff / (
                torch.linalg.vector_norm(goal_diff, dim=2, keepdim=True) + self.eps
            )
            acc = x.new_zeros(n_steps, q_end - q_start, 1)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                diff = empirical[:, r_start:r_end].unsqueeze(1) - xq.unsqueeze(2)
                dist = torch.linalg.vector_norm(diff, dim=3)
                direction = diff / (dist.unsqueeze(-1) + self.eps)
                visibility = 0.5 * (
                    1.0 + (desired.unsqueeze(2) * direction).sum(dim=3)
                )
                kernel = visibility * torch.exp(-dist / self.range_scale)
                if same_particles:
                    q_indices = torch.arange(q_start, q_end, device=x.device)
                    mask = (q_indices >= r_start) & (q_indices < r_end)
                    if mask.any():
                        rows = torch.nonzero(mask, as_tuple=False).squeeze(1)
                        cols = q_indices[mask] - r_start
                        kernel[:, rows, cols] = 0.0
                acc = acc + kernel.sum(dim=2, keepdim=True)
            out[:, q_start:q_end] = acc / float(denominator)
        return self.weight * out


class MeanDeviationCost(CostPotential):
    """Absolute deviation from empirical mean (L1-type)."""

    random_batch_aggregation = "mean"

    def __init__(self, weight: float = 1.0):
        self.weight = weight
        self.name = "mean_deviation"
        self.formula = f"F(x) = {self.weight}‖x - 𝔼[X]‖₁"

    def _evaluate(self, x: Tensor, empirical: Tensor) -> Tensor:
        mean = empirical.mean(dim=0, keepdim=True)
        return self.weight * torch.norm(x - mean, p=1, dim=1, keepdim=True)

    def _evaluate_batched_block(self, x: Tensor, empirical: Tensor) -> Tensor:
        mean = empirical.mean(dim=1, keepdim=True)
        return self.weight * torch.norm(x - mean, p=1, dim=2, keepdim=True)


class QuadraticPotentialCost(CostPotential):
    """Quadratic potential interaction: F(x) = 𝔼[‖x-X‖²].

    Pairwise distance cost that penalizes agents based on their
    average squared distance to all other agents in the population.
    Unlike QuadraticDeviationCost which measures distance from the mean,
    this cost captures the full pairwise interaction structure.

    Args:
        weight: Scaling factor for the cost (default: 1.0)

    Applications:
        - Social distance minimization
        - Consensus problems with pairwise penalties
        - Variance-like cost functionals in mean field games
    """

    random_batch_aggregation = "mean"

    def __init__(self, weight: float = 1.0):
        self.weight = weight
        self.name = "quadratic_potential"
        self.formula = f"F(x) = {self.weight}𝔼[‖x - X‖²]"

    def _evaluate(self, x: Tensor, empirical: Tensor) -> Tensor:
        bsz = x.shape[0]
        n_ref = empirical.shape[0]
        q_chunk = (
            bsz
            if self.query_chunk_size is None or self.query_chunk_size <= 0
            else min(self.query_chunk_size, bsz)
        )
        r_chunk = (
            n_ref
            if self.ref_chunk_size is None or self.ref_chunk_size <= 0
            else min(self.ref_chunk_size, n_ref)
        )
        out = x.new_zeros(bsz, 1)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[q_start:q_end]
            acc = x.new_zeros(q_end - q_start, 1)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[r_start:r_end]
                diff = emp.unsqueeze(0) - xq.unsqueeze(1)
                dist_sq = (diff**2).sum(dim=2)
                acc = acc + dist_sq.sum(dim=1, keepdim=True)
            out[q_start:q_end] = acc / float(n_ref)
        return self.weight * out

    def _evaluate_batched_block(self, x: Tensor, empirical: Tensor) -> Tensor:
        n_steps, bsz, _ = x.shape
        n_ref = empirical.shape[1]
        q_chunk = (
            bsz
            if self.query_chunk_size is None or self.query_chunk_size <= 0
            else min(self.query_chunk_size, bsz)
        )
        r_chunk = (
            n_ref
            if self.ref_chunk_size is None or self.ref_chunk_size <= 0
            else min(self.ref_chunk_size, n_ref)
        )
        out = x.new_zeros(n_steps, bsz, 1)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[:, q_start:q_end]
            acc = x.new_zeros(n_steps, q_end - q_start, 1)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[:, r_start:r_end]
                diff = emp.unsqueeze(1) - xq.unsqueeze(2)
                dist_sq = (diff**2).sum(dim=3)
                acc = acc + dist_sq.sum(dim=2, keepdim=True)
            out[:, q_start:q_end] = acc / float(n_ref)
        return self.weight * out


class CoulombCost(CostPotential):
    """Coulomb-type long-range cost: F(x) = 𝔼[1/‖x-X‖^α].

    Power-law interaction cost with singularity regularization.
    Models repulsive (or attractive if negated) long-range forces
    similar to electrostatic or gravitational potentials.

    Args:
        alpha: Power exponent (default: 1.0)
            - α = 1: Coulomb/logarithmic potential
            - α < 1: Super-Coulombic, stronger long-range interaction
            - α > 1: Sub-Coulombic, weaker long-range interaction
        eps: Regularization parameter to avoid singularity (default: 0.01)

    Applications:
        - Electrostatic and gravitational systems
        - Optimal transport with power-law costs
        - Long-range repulsion in crowd dynamics
    """

    random_batch_aggregation = "mean"

    def __init__(self, alpha: float = 1.0, eps: float = 0.01):
        self.alpha = alpha
        self.eps = eps
        self.name = "coulomb_cost"
        self.formula = f"F(x) = 𝔼[1/‖x-X‖^{alpha:.1f}], ε={eps:.2f}"

    def _evaluate(self, x: Tensor, empirical: Tensor) -> Tensor:
        bsz = x.shape[0]
        n_ref = empirical.shape[0]
        q_chunk = (
            bsz
            if self.query_chunk_size is None or self.query_chunk_size <= 0
            else min(self.query_chunk_size, bsz)
        )
        r_chunk = (
            n_ref
            if self.ref_chunk_size is None or self.ref_chunk_size <= 0
            else min(self.ref_chunk_size, n_ref)
        )
        out = x.new_zeros(bsz, 1)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[q_start:q_end]
            acc = x.new_zeros(q_end - q_start, 1)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[r_start:r_end]
                diff = emp.unsqueeze(0) - xq.unsqueeze(1)
                dist = torch.sqrt((diff**2).sum(dim=2) + self.eps**2)
                cost = 1.0 / (dist**self.alpha)
                acc = acc + cost.sum(dim=1, keepdim=True)
            out[q_start:q_end] = acc / float(n_ref)
        return out

    def _evaluate_batched_block(self, x: Tensor, empirical: Tensor) -> Tensor:
        n_steps, bsz, _ = x.shape
        n_ref = empirical.shape[1]
        q_chunk = (
            bsz
            if self.query_chunk_size is None or self.query_chunk_size <= 0
            else min(self.query_chunk_size, bsz)
        )
        r_chunk = (
            n_ref
            if self.ref_chunk_size is None or self.ref_chunk_size <= 0
            else min(self.ref_chunk_size, n_ref)
        )
        out = x.new_zeros(n_steps, bsz, 1)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[:, q_start:q_end]
            acc = x.new_zeros(n_steps, q_end - q_start, 1)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[:, r_start:r_end]
                diff = emp.unsqueeze(1) - xq.unsqueeze(2)
                dist = torch.sqrt((diff**2).sum(dim=3) + self.eps**2)
                cost = 1.0 / (dist**self.alpha)
                acc = acc + cost.sum(dim=2, keepdim=True)
            out[:, q_start:q_end] = acc / float(n_ref)
        return out


class LocalExclusionCost(CostPotential):
    """Local exclusion cost: F(x) = (strength/ε^d) 𝔼[φ((x-X)/ε)].

    Short-range repulsion with compact support kernel φ(r) = max(0, 1-r).
    Only particles within distance ε contribute to the cost, making this
    suitable for modeling hard-core or soft-exclusion constraints.

    Args:
        epsilon: Exclusion radius (default: 0.5)
            Particles outside this radius have no interaction
        strength: Cost intensity (default: 1.0)
            Controls the penalty magnitude

    Applications:
        - Pedestrian and crowd dynamics with personal space
        - Collision avoidance in multi-agent systems
        - Hard-sphere approximations in particle systems
        - Traffic flow with minimum safe distances
    """

    random_batch_aggregation = "mean"

    def __init__(self, epsilon: float = 0.5, strength: float = 1.0):
        self.epsilon = epsilon
        self.strength = strength
        self.name = "local_exclusion"
        self.formula = f"F(x) = {strength:.2f}/ε^d·𝔼[φ((x-X)/ε)], ε={epsilon:.2f}"

    def _evaluate(self, x: Tensor, empirical: Tensor) -> Tensor:
        bsz = x.shape[0]
        n_ref = empirical.shape[0]
        q_chunk = (
            bsz
            if self.query_chunk_size is None or self.query_chunk_size <= 0
            else min(self.query_chunk_size, bsz)
        )
        r_chunk = (
            n_ref
            if self.ref_chunk_size is None or self.ref_chunk_size <= 0
            else min(self.ref_chunk_size, n_ref)
        )
        out = x.new_zeros(bsz, 1)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[q_start:q_end]
            acc = x.new_zeros(q_end - q_start, 1)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[r_start:r_end]
                diff = emp.unsqueeze(0) - xq.unsqueeze(1)
                scaled_dist = torch.sqrt((diff**2).sum(dim=2)) / self.epsilon
                phi = torch.relu(1.0 - scaled_dist)
                acc = acc + phi.sum(dim=1, keepdim=True)
            out[q_start:q_end] = acc / float(n_ref)

        d = x.shape[1]
        normalization = self.strength / (self.epsilon**d)
        return normalization * out

    def _evaluate_batched_block(self, x: Tensor, empirical: Tensor) -> Tensor:
        n_steps, bsz, dim = x.shape
        n_ref = empirical.shape[1]
        q_chunk = (
            bsz
            if self.query_chunk_size is None or self.query_chunk_size <= 0
            else min(self.query_chunk_size, bsz)
        )
        r_chunk = (
            n_ref
            if self.ref_chunk_size is None or self.ref_chunk_size <= 0
            else min(self.ref_chunk_size, n_ref)
        )
        out = x.new_zeros(n_steps, bsz, 1)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[:, q_start:q_end]
            acc = x.new_zeros(n_steps, q_end - q_start, 1)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[:, r_start:r_end]
                diff = emp.unsqueeze(1) - xq.unsqueeze(2)
                scaled_dist = torch.sqrt((diff**2).sum(dim=3)) / self.epsilon
                phi = torch.relu(1.0 - scaled_dist)
                acc = acc + phi.sum(dim=2, keepdim=True)
            out[:, q_start:q_end] = acc / float(n_ref)

        normalization = self.strength / (self.epsilon**dim)
        return normalization * out


class EntropyCost(CostPotential):
    """Entropy minimization cost (Liu et al. 2022).

    Penalizes high probability density: F(x) = w·(log ρ(x) + 1)

    This encourages the distribution to spread out and avoid concentration.
    Used in V-neck navigation problem.

    Note: Requires log-density to be available, typically computed from
    empirical distribution or via kernel density estimation.

    Args:
        weight: Multiplicative weight for the entropy penalty (default: 1.0)

    Applications:
        - Crowd navigation with density-dependent costs
        - Bottleneck navigation (V-neck)
        - Entropy regularization in mean-field games
    """

    random_batch_aggregation = "mean"

    def __init__(self, weight: float = 1.0, bandwidth: float = 1.0):
        if bandwidth <= 0:
            raise ValueError("EntropyCost bandwidth must be positive")
        self.weight = weight
        self.bandwidth = bandwidth
        self.name = "entropy"
        self.formula = f"F(x) = {weight:.3f}·(log ρ_h(x) + 1), h={bandwidth:.3f}"

    def _evaluate(self, x: Tensor, empirical: Tensor) -> Tensor:
        """
        For entropy cost, we need the log-density which should be passed
        as the empirical distribution's log_prob.

        This is a placeholder - proper implementation requires log_prob
        to be available in the solver.
        """
        bsz = x.shape[0]
        n_ref = empirical.shape[0]
        q_chunk = (
            bsz
            if self.query_chunk_size is None or self.query_chunk_size <= 0
            else min(self.query_chunk_size, bsz)
        )
        r_chunk = (
            n_ref
            if self.ref_chunk_size is None or self.ref_chunk_size <= 0
            else min(self.ref_chunk_size, n_ref)
        )
        log_density = x.new_full((bsz, 1), -float("inf"))

        h = float(self.bandwidth)
        log_norm = -x.shape[-1] * math.log(h) - 0.5 * x.shape[-1] * math.log(
            2.0 * math.pi
        )
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[q_start:q_end]
            log_acc = x.new_full((q_end - q_start, 1), -float("inf"))
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[r_start:r_end]
                diff = emp.unsqueeze(0) - xq.unsqueeze(1)
                dist_sq = (diff**2).sum(dim=2)
                log_kernel = -dist_sq / (2 * h**2)
                log_chunk = torch.logsumexp(log_kernel, dim=1, keepdim=True)
                log_acc = torch.logaddexp(log_acc, log_chunk)
            log_density[q_start:q_end] = log_acc - math.log(float(n_ref)) + log_norm

        # log ρ + 1
        return self.weight * (log_density + 1.0)

    def _evaluate_batched_block(self, x: Tensor, empirical: Tensor) -> Tensor:
        n_steps, bsz, _ = x.shape
        n_ref = empirical.shape[1]
        q_chunk = (
            bsz
            if self.query_chunk_size is None or self.query_chunk_size <= 0
            else min(self.query_chunk_size, bsz)
        )
        r_chunk = (
            n_ref
            if self.ref_chunk_size is None or self.ref_chunk_size <= 0
            else min(self.ref_chunk_size, n_ref)
        )
        log_density = x.new_full((n_steps, bsz, 1), -float("inf"))
        h = float(self.bandwidth)
        log_norm = -x.shape[-1] * math.log(h) - 0.5 * x.shape[-1] * math.log(
            2.0 * math.pi
        )
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[:, q_start:q_end]
            log_acc = x.new_full((n_steps, q_end - q_start, 1), -float("inf"))
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[:, r_start:r_end]
                diff = emp.unsqueeze(1) - xq.unsqueeze(2)
                dist_sq = (diff**2).sum(dim=3)
                log_kernel = -dist_sq / (2 * h**2)
                log_chunk = torch.logsumexp(log_kernel, dim=2, keepdim=True)
                log_acc = torch.logaddexp(log_acc, log_chunk)
            log_density[:, q_start:q_end] = (
                log_acc - math.log(float(n_ref)) + log_norm
            )

        return self.weight * (log_density + 1.0)


class CongestionCost(CostPotential):
    """Congestion cost (Liu et al. 2022).

    Pairwise repulsion cost: F(x) = w·𝔼[2 / (‖x-X‖² + 1)]

    Penalizes proximity to other particles, encouraging spread.
    Used in S-tunnel navigation problem.

    Args:
        weight: Multiplier for cost intensity (default: 1.0)

    Applications:
        - S-tunnel navigation with congestion avoidance
        - Traffic flow with congestion penalty
        - Crowd dynamics with personal space
    """

    random_batch_aggregation = "mean"

    def __init__(self, weight: float = 1.0):
        self.weight = weight
        self.name = "congestion"
        self.formula = f"F(x) = {weight:.1f}·𝔼[2/(‖x-X‖²+1)]"

    def _evaluate(self, x: Tensor, empirical: Tensor) -> Tensor:
        bsz = x.shape[0]
        n_ref = empirical.shape[0]
        q_chunk = (
            bsz
            if self.query_chunk_size is None or self.query_chunk_size <= 0
            else min(self.query_chunk_size, bsz)
        )
        r_chunk = (
            n_ref
            if self.ref_chunk_size is None or self.ref_chunk_size <= 0
            else min(self.ref_chunk_size, n_ref)
        )
        out = x.new_zeros(bsz, 1)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[q_start:q_end]
            acc = x.new_zeros(q_end - q_start, 1)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[r_start:r_end]
                diff = emp.unsqueeze(0) - xq.unsqueeze(1)
                dist_sq = (diff**2).sum(dim=2)
                cost_per_particle = 2.0 / (dist_sq + 1.0)
                acc = acc + cost_per_particle.sum(dim=1, keepdim=True)
            out[q_start:q_end] = acc / float(n_ref)

        return self.weight * out


class FlockingAlignmentCost(CostPotential):
    """Cauchy-kernel flocking alignment cost inspired by Cao et al. (2026).

    For a state x=(p, v), where p is position and v is velocity, the cost is

        A(p, v) = v C0(p) - C1(p),
        C0(p)   = E[K(p-p')],
        C1(p)   = E[K(p-p') v'],
        F(x)    = weight * ||A(p, v)||^2,

    with the Cauchy kernel

        K(z) = 1 / (1 + ||z||^2 / sigma^2).

    By default the state is split evenly into position and velocity blocks.
    """

    random_batch_aggregation = "mean"

    def __init__(
        self,
        sigma: float = 1.0,
        weight: float = 1.0,
        position_dim: Optional[int] = None,
    ):
        self.sigma = sigma
        self.weight = weight
        self.position_dim = position_dim
        self.name = "flocking_alignment"
        self.formula = (
            f"F(x)= {weight:.2f}||v E[K] - E[K v']||^2, "
            f"K(z)=1/(1+||z||^2/{sigma:.2f}^2)"
        )

    def _split_state(self, x: Tensor) -> tuple[Tensor, Tensor]:
        pos_dim = self.position_dim or (x.shape[-1] // 2)
        if pos_dim <= 0 or pos_dim >= x.shape[-1]:
            raise ValueError(
                "FlockingAlignmentCost expects state dim >= 2 and a valid position_dim."
            )
        return x[..., :pos_dim], x[..., pos_dim:]

    def _evaluate(self, x: Tensor, empirical: Tensor) -> Tensor:
        p, v = self._split_state(x)
        p_emp, v_emp = self._split_state(empirical)
        bsz = x.shape[0]
        n_ref = empirical.shape[0]
        q_chunk = (
            bsz
            if self.query_chunk_size is None or self.query_chunk_size <= 0
            else min(self.query_chunk_size, bsz)
        )
        r_chunk = (
            n_ref
            if self.ref_chunk_size is None or self.ref_chunk_size <= 0
            else min(self.ref_chunk_size, n_ref)
        )
        out = x.new_zeros(bsz, 1)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            pq = p[q_start:q_end]
            vq = v[q_start:q_end]
            c0_acc = x.new_zeros(q_end - q_start, 1)
            c1_acc = x.new_zeros(q_end - q_start, v.shape[1])
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                pe = p_emp[r_start:r_end]
                ve = v_emp[r_start:r_end]
                diff = pe.unsqueeze(0) - pq.unsqueeze(1)
                dist_sq = (diff**2).sum(dim=2)
                kernel = 1.0 / (1.0 + dist_sq / (self.sigma**2))
                c0_acc = c0_acc + kernel.sum(dim=1, keepdim=True)
                c1_acc = c1_acc + kernel @ ve
            c0 = c0_acc / float(n_ref)
            c1 = c1_acc / float(n_ref)
            align = vq * c0 - c1
            out[q_start:q_end] = (align**2).sum(dim=1, keepdim=True)
        return self.weight * out

    def _evaluate_batched_block(self, x: Tensor, empirical: Tensor) -> Tensor:
        p, v = self._split_state(x)
        p_emp, v_emp = self._split_state(empirical)
        n_steps, bsz, _ = x.shape
        vel_dim = v.shape[-1]
        n_ref = empirical.shape[1]
        q_chunk = (
            bsz
            if self.query_chunk_size is None or self.query_chunk_size <= 0
            else min(self.query_chunk_size, bsz)
        )
        r_chunk = (
            n_ref
            if self.ref_chunk_size is None or self.ref_chunk_size <= 0
            else min(self.ref_chunk_size, n_ref)
        )
        out = x.new_zeros(n_steps, bsz, 1)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            pq = p[:, q_start:q_end]
            vq = v[:, q_start:q_end]
            c0_acc = x.new_zeros(n_steps, q_end - q_start, 1)
            c1_acc = x.new_zeros(n_steps, q_end - q_start, vel_dim)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                pe = p_emp[:, r_start:r_end]
                ve = v_emp[:, r_start:r_end]
                diff = pe.unsqueeze(1) - pq.unsqueeze(2)
                dist_sq = (diff**2).sum(dim=3)
                kernel = 1.0 / (1.0 + dist_sq / (self.sigma**2))
                c0_acc = c0_acc + kernel.sum(dim=2, keepdim=True)
                c1_acc = c1_acc + torch.einsum("tqr,trd->tqd", kernel, ve)
            c0 = c0_acc / float(n_ref)
            c1 = c1_acc / float(n_ref)
            align = vq * c0 - c1
            out[:, q_start:q_end] = (align**2).sum(dim=2, keepdim=True)
        return self.weight * out


__all__ = [
    "CongestionCost",
    "CostPotential",
    "CoulombCost",
    "EntropyCost",
    "FlockingAlignmentCost",
    "GaussianConvolutionCost",
    "GaussianConvolutionMeanCost",
    "LocalExclusionCost",
    "MeanDeviationCost",
    "QuadraticDeviationCost",
    "QuadraticPotentialCost",
    "ZeroCost",
]
