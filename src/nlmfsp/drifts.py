"""Reusable drift functions for mean-field dynamics."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from .rng_utils import torch_seed_context

Tensor = torch.Tensor
EPS = 1e-8


class Drift(ABC):
    """Base class for local/non-local drift functions."""

    name: str = "drift"
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
        out = x.new_empty(x.shape)
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


class MeanReversionDrift(Drift):
    """Local drift pulling states towards the empirical mean."""

    random_batch_aggregation = "mean"

    def __init__(self, weight: float = 1.0):
        self.weight = weight
        self.name = "mean_reversion"
        self.formula = f"f(x) = {self.weight}(𝔼[X] - x)"

    def _evaluate(self, x: Tensor, empirical: Tensor) -> Tensor:
        mean = empirical.mean(dim=0, keepdim=True)
        return self.weight * (mean.expand_as(x) - x)

    def _evaluate_batched_block(self, x: Tensor, empirical: Tensor) -> Tensor:
        mean = empirical.mean(dim=1, keepdim=True)
        return self.weight * (mean.expand_as(x) - x)


class ZeroDrift(Drift):
    """Zero drift."""

    random_batch_aggregation = "mean"

    def __init__(self):
        self.name = "zero"
        self.formula = "f(x) = 0"

    def _evaluate(self, x: Tensor, empirical: Tensor) -> Tensor:
        return torch.zeros_like(x)

    def _evaluate_batched_block(self, x: Tensor, empirical: Tensor) -> Tensor:
        return torch.zeros_like(x)


class GaussianDrift(Drift):
    """Non-local Gaussian convolution drift.

    Special case of LaplacianDiffusionDrift with Gaussian kernel.
    """

    random_batch_aggregation = "mean"

    def __init__(self, sigma: float = 1.0, weight: float = 1.0):
        self.sigma = sigma
        self.weight = weight
        self.name = "gaussian"
        self.formula = f"f(x) = {weight:.2f}·𝔼[exp(-‖x-X‖²/2σ²)(X-x)], σ={sigma:.2f}"

    def _evaluate(self, x: Tensor, empirical: Tensor) -> Tensor:
        bsz, dim = x.shape
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

        out = x.new_zeros(bsz, dim)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[q_start:q_end]
            acc = x.new_zeros(q_end - q_start, dim)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[r_start:r_end]
                diff = emp.unsqueeze(0) - xq.unsqueeze(1)
                dist_sq = (diff**2).sum(dim=2)
                weights = torch.exp(-dist_sq / (2 * self.sigma**2))
                acc = acc + (weights.unsqueeze(2) * diff).sum(dim=1)
            out[q_start:q_end] = acc / float(n_ref)

        return self.weight * out

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

        out = x.new_zeros(n_steps, bsz, dim)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[:, q_start:q_end]
            acc = x.new_zeros(n_steps, q_end - q_start, dim)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[:, r_start:r_end]
                diff = emp.unsqueeze(1) - xq.unsqueeze(2)
                dist_sq = (diff**2).sum(dim=3)
                weights = torch.exp(-dist_sq / (2 * self.sigma**2))
                acc = acc + (weights.unsqueeze(3) * diff).sum(dim=2)
            out[:, q_start:q_end] = acc / float(n_ref)

        return self.weight * out


class NonstationaryAnisotropicDrift(Drift):
    """Pairwise drift with a spatially varying anisotropic Gaussian kernel.

    The interaction has the form

        f(x) = weight * E[K(x, X) (X - x)],

    where K is a Gaussian kernel whose local metric depends on the midpoint
    m=(x+X)/2. In the first two coordinates,

        A(m) = R(theta(m)) diag(exp(alpha), exp(-alpha)) R(theta(m))^T,
        theta(m) = m_1 / sigma.

    Remaining coordinates, if present, use the isotropic Gaussian metric. The
    special case alpha=0 exactly recovers GaussianDrift.
    """

    random_batch_aggregation = "mean"

    def __init__(self, sigma: float = 1.0, weight: float = 1.0, alpha: float = 1.0):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.sigma = sigma
        self.weight = weight
        self.alpha = alpha
        self.name = "nonstationary_anisotropic"
        self.formula = (
            f"f(x) = {weight:.2f}*E[K_A(x,X)(X-x)], "
            f"sigma={sigma:.2f}, alpha={alpha:.2f}"
        )

    def _quadratic_form(self, diff: Tensor, midpoint: Tensor) -> Tensor:
        if diff.shape[-1] == 1:
            return math.exp(self.alpha) * diff[..., 0] ** 2

        theta = midpoint[..., 0] / self.sigma
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # u = R(theta)^T diff in the first two coordinates.
        u0 = cos_theta * diff[..., 0] + sin_theta * diff[..., 1]
        u1 = -sin_theta * diff[..., 0] + cos_theta * diff[..., 1]
        quad = math.exp(self.alpha) * u0**2 + math.exp(-self.alpha) * u1**2
        if diff.shape[-1] > 2:
            quad = quad + (diff[..., 2:] ** 2).sum(dim=-1)
        return quad

    def _evaluate(self, x: Tensor, empirical: Tensor) -> Tensor:
        bsz, dim = x.shape
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

        out = x.new_zeros(bsz, dim)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[q_start:q_end]
            acc = x.new_zeros(q_end - q_start, dim)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[r_start:r_end]
                diff = emp.unsqueeze(0) - xq.unsqueeze(1)
                midpoint = 0.5 * (emp.unsqueeze(0) + xq.unsqueeze(1))
                quad = self._quadratic_form(diff, midpoint)
                weights = torch.exp(-quad / (2 * self.sigma**2))
                acc = acc + (weights.unsqueeze(2) * diff).sum(dim=1)
            out[q_start:q_end] = acc / float(n_ref)

        return self.weight * out

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

        out = x.new_zeros(n_steps, bsz, dim)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[:, q_start:q_end]
            acc = x.new_zeros(n_steps, q_end - q_start, dim)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[:, r_start:r_end]
                diff = emp.unsqueeze(1) - xq.unsqueeze(2)
                midpoint = 0.5 * (emp.unsqueeze(1) + xq.unsqueeze(2))
                quad = self._quadratic_form(diff, midpoint)
                weights = torch.exp(-quad / (2 * self.sigma**2))
                acc = acc + (weights.unsqueeze(3) * diff).sum(dim=2)
            out[:, q_start:q_end] = acc / float(n_ref)

        return self.weight * out


class AttractionRepulsionDrift(Drift):
    """Balanced attraction/repulsion drift via Gaussian components."""

    random_batch_aggregation = "mean"

    def __init__(
        self,
        attract: float = 1.0,
        sigma_attract: float = 1.0,
        repel: float = 0.5,
        sigma_repel: float = 0.3,
        weight: float = 1.0,
    ):
        self.attract = attract
        self.sigma_attract = sigma_attract
        self.repel = repel
        self.sigma_repel = sigma_repel
        self.weight = weight
        self.name = "attraction_repulsion"
        self.formula = (
            f"f(x) = {weight:.2f}[{attract:.2f}⋅𝔼[K_{sigma_attract:.2f}(X-x)]"
            f" - {repel:.2f}⋅𝔼[K_{sigma_repel:.2f}(X-x)]]"
        )

    def _gaussian(self, x: Tensor, empirical: Tensor, sigma: float) -> Tensor:
        bsz, dim = x.shape
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
        out = x.new_zeros(bsz, dim)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[q_start:q_end]
            acc = x.new_zeros(q_end - q_start, dim)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[r_start:r_end]
                diff = emp.unsqueeze(0) - xq.unsqueeze(1)
                dist_sq = (diff**2).sum(dim=2)
                weights = torch.exp(-dist_sq / (2 * sigma**2))
                acc = acc + (weights.unsqueeze(2) * diff).sum(dim=1)
            out[q_start:q_end] = acc / float(n_ref)
        return out

    def _gaussian_batched_block(
        self, x: Tensor, empirical: Tensor, sigma: float
    ) -> Tensor:
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
        out = x.new_zeros(n_steps, bsz, dim)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[:, q_start:q_end]
            acc = x.new_zeros(n_steps, q_end - q_start, dim)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[:, r_start:r_end]
                diff = emp.unsqueeze(1) - xq.unsqueeze(2)
                dist_sq = (diff**2).sum(dim=3)
                weights = torch.exp(-dist_sq / (2 * sigma**2))
                acc = acc + (weights.unsqueeze(3) * diff).sum(dim=2)
            out[:, q_start:q_end] = acc / float(n_ref)
        return out

    def _evaluate(self, x: Tensor, empirical: Tensor) -> Tensor:
        attraction = self._gaussian(x, empirical, self.sigma_attract)
        repulsion = self._gaussian(x, empirical, self.sigma_repel)
        return self.weight * (self.attract * attraction - self.repel * repulsion)

    def _evaluate_batched_block(self, x: Tensor, empirical: Tensor) -> Tensor:
        attraction = self._gaussian_batched_block(x, empirical, self.sigma_attract)
        repulsion = self._gaussian_batched_block(x, empirical, self.sigma_repel)
        return self.weight * (self.attract * attraction - self.repel * repulsion)


class CoulombDrift(Drift):
    """Coulomb-type drift with softening epsilon."""

    random_batch_aggregation = "mean"

    def __init__(self, alpha: float = 1.0, eps: float = 0.1, weight: float = 1.0):
        self.alpha = alpha
        self.eps = eps
        self.weight = weight
        self.name = "coulomb"
        self.formula = (
            f"f(x) = {weight:.2f}·𝔼[(x-X)/|x-X|^({alpha:.1f}+1)], ε={eps:.2f}"
        )

    def _evaluate(self, x: Tensor, empirical: Tensor) -> Tensor:
        bsz, dim = x.shape
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

        out = x.new_zeros(bsz, dim)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[q_start:q_end]
            acc = x.new_zeros(q_end - q_start, dim)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[r_start:r_end]
                diff = xq.unsqueeze(1) - emp.unsqueeze(0)
                dist = torch.sqrt((diff**2).sum(dim=2, keepdim=True) + self.eps**2)
                kernel = diff / (dist ** (self.alpha + 1))
                acc = acc + kernel.sum(dim=1)
            out[q_start:q_end] = acc / float(n_ref)

        return self.weight * out

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

        out = x.new_zeros(n_steps, bsz, dim)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[:, q_start:q_end]
            acc = x.new_zeros(n_steps, q_end - q_start, dim)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[:, r_start:r_end]
                diff = xq.unsqueeze(2) - emp.unsqueeze(1)
                dist = torch.sqrt((diff**2).sum(dim=3, keepdim=True) + self.eps**2)
                kernel = diff / (dist ** (self.alpha + 1))
                acc = acc + kernel.sum(dim=2)
            out[:, q_start:q_end] = acc / float(n_ref)

        return self.weight * out


class LaplacianDiffusionDrift(Drift):
    """Laplacian-based diffusion coupling: f(x) = 𝔼[ψ(‖x-X‖)(X-x)].

    Generalizes GaussianDrift to support multiple kernel types.
    The drift couples particles through a weighted average of relative
    positions, where weights are determined by a distance-based kernel.

    Args:
        sigma: Kernel bandwidth parameter (default: 1.0)
        kernel_type: Type of distance kernel (default: "gaussian")
            - "gaussian": exp(-‖x-X‖²/2σ²) - smooth, localized influence
            - "exponential": exp(-‖x-X‖/σ) - medium-range coupling
            - "inverse": 1/(‖x-X‖/σ + 1) - long-range, heavy-tailed

    Applications:
        - Consensus dynamics and opinion formation
        - Swarm behavior and flocking models
        - Diffusion-based particle systems
    """

    random_batch_aggregation = "mean"

    def __init__(
        self, sigma: float = 1.0, kernel_type: str = "gaussian", weight: float = 1.0
    ):
        self.sigma = sigma
        self.kernel_type = kernel_type
        self.weight = weight
        self.name = "laplacian_diffusion"
        self.formula = (
            f"f(x) = {weight:.2f}·𝔼[ψ(‖x-X‖)(X-x)], ψ={kernel_type}, σ={sigma:.2f}"
        )

    def _evaluate(self, x: Tensor, empirical: Tensor) -> Tensor:
        bsz, dim = x.shape
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
        out = x.new_zeros(bsz, dim)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[q_start:q_end]
            acc = x.new_zeros(q_end - q_start, dim)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[r_start:r_end]
                diff = emp.unsqueeze(0) - xq.unsqueeze(1)
                dist = torch.sqrt((diff**2).sum(dim=2, keepdim=True) + 1e-8)

                if self.kernel_type == "gaussian":
                    weights = torch.exp(-(dist**2) / (2 * self.sigma**2))
                elif self.kernel_type == "exponential":
                    weights = torch.exp(-dist / self.sigma)
                elif self.kernel_type == "inverse":
                    weights = 1.0 / (dist / self.sigma + 1.0)
                else:
                    raise ValueError(f"Unknown kernel type: {self.kernel_type}")

                acc = acc + (weights * diff).sum(dim=1)
            out[q_start:q_end] = acc / float(n_ref)
        return self.weight * out

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
        out = x.new_zeros(n_steps, bsz, dim)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[:, q_start:q_end]
            acc = x.new_zeros(n_steps, q_end - q_start, dim)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[:, r_start:r_end]
                diff = emp.unsqueeze(1) - xq.unsqueeze(2)
                dist = torch.sqrt((diff**2).sum(dim=3, keepdim=True) + 1e-8)

                if self.kernel_type == "gaussian":
                    weights = torch.exp(-(dist**2) / (2 * self.sigma**2))
                elif self.kernel_type == "exponential":
                    weights = torch.exp(-dist / self.sigma)
                elif self.kernel_type == "inverse":
                    weights = 1.0 / (dist / self.sigma + 1.0)
                else:
                    raise ValueError(f"Unknown kernel type: {self.kernel_type}")

                acc = acc + (weights * diff).sum(dim=2)
            out[:, q_start:q_end] = acc / float(n_ref)
        return self.weight * out


class SmoothBoundedConfidenceDrift(Drift):
    """Smooth bounded-confidence opinion drift.

    This is a differentiable Hegselmann-Krause-style interaction:

        f(x) = weight * sum_j K(||X_j - x||) (X_j - x) / sum_j K(||X_j - x||).

    The Gaussian kernel gives a smooth confidence radius and keeps the exact
    evaluation genuinely pairwise O(N^2 d), unlike the DeepGSB party-model
    opinion drift whose population term can be aggregated once per time slice.
    """

    random_batch_aggregation = "mean"

    def __init__(
        self,
        sigma: float = 1.0,
        weight: float = 1.0,
        eps: float = 1e-8,
    ):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.sigma = sigma
        self.weight = weight
        self.eps = eps
        self.name = "bounded_confidence"
        self.formula = (
            f"f(x) = {weight:.2f}·Σ Kσ(X-x)(X-x) / Σ Kσ(X-x), σ={sigma:.2f}"
        )

    def _evaluate(self, x: Tensor, empirical: Tensor) -> Tensor:
        bsz, dim = x.shape
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

        out = x.new_zeros(bsz, dim)
        denom = x.new_zeros(bsz, 1)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[q_start:q_end]
            acc = x.new_zeros(q_end - q_start, dim)
            den = x.new_zeros(q_end - q_start, 1)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[r_start:r_end]
                diff = emp.unsqueeze(0) - xq.unsqueeze(1)
                dist_sq = (diff**2).sum(dim=2, keepdim=True)
                weights = torch.exp(-dist_sq / (2 * self.sigma**2))
                acc = acc + (weights * diff).sum(dim=1)
                den = den + weights.sum(dim=1)
            out[q_start:q_end] = acc
            denom[q_start:q_end] = den

        return self.weight * out / denom.clamp_min(self.eps)

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

        out = x.new_zeros(n_steps, bsz, dim)
        denom = x.new_zeros(n_steps, bsz, 1)
        for q_start in range(0, bsz, q_chunk):
            q_end = min(q_start + q_chunk, bsz)
            xq = x[:, q_start:q_end]
            acc = x.new_zeros(n_steps, q_end - q_start, dim)
            den = x.new_zeros(n_steps, q_end - q_start, 1)
            for r_start in range(0, n_ref, r_chunk):
                r_end = min(r_start + r_chunk, n_ref)
                emp = empirical[:, r_start:r_end]
                diff = emp.unsqueeze(1) - xq.unsqueeze(2)
                dist_sq = (diff**2).sum(dim=3, keepdim=True)
                weights = torch.exp(-dist_sq / (2 * self.sigma**2))
                acc = acc + (weights * diff).sum(dim=2)
                den = den + weights.sum(dim=2)
            out[:, q_start:q_end] = acc
            denom[:, q_start:q_end] = den

        return self.weight * out / denom.clamp_min(self.eps)


class OpinionMeanFieldDrift(nn.Module):
    """Opinion dynamics mean-field drift (DeepGSB-inspired).

    Uses a time-varying direction schedule xi[t] and computes the mean-field
    polarization term from the current particle cloud at that time step.

    Args:
        xi_schedule: (n_steps, dim) unit vectors (Brownian-smoothed)
        f_mul: (n_steps,) decay schedule to avoid blow-up
        dt: Time step size
        T: Horizon (for index clamping)
        polarize_strength: Multiplier (default 1.0 for low-dim, 6.0 for 1kD)
    """

    random_batch_aggregation = "mean"
    requires_time = True

    def __init__(
        self,
        xi_schedule: Tensor,
        f_mul: Tensor,
        dt: float,
        T: float,
        polarize_strength: float = 1.0,
    ):
        super().__init__()
        # Register as buffers to keep on device and out of optimizer
        self.register_buffer("xi_schedule", xi_schedule)
        self.register_buffer("f_mul", f_mul)
        self.dt = dt
        self.T = T
        self.n_steps = xi_schedule.shape[0]
        self.polarize_strength = polarize_strength

    def _time_to_index(self, t: Tensor) -> Tensor:
        idx = torch.round(t / self.dt).long()
        idx = torch.clamp(idx, 0, self.n_steps - 1)
        return idx

    @staticmethod
    def _signed_projection(x: Tensor, xi: Tensor) -> Tensor:
        sign = torch.sign(torch.sum(x * xi, dim=-1, keepdim=True))
        return torch.where(sign == 0, torch.ones_like(sign), sign)

    def _evaluate_index(self, x: Tensor, empirical: Tensor, idx: Tensor) -> Tensor:
        idx_int = int(idx.item())
        xi = self.xi_schedule[idx_int].view(1, -1)
        fmul = self.f_mul[idx_int]

        y_norm = torch.linalg.norm(empirical, dim=-1, keepdim=True).clamp_min(EPS)
        y_bar = empirical / y_norm
        agree_y = self._signed_projection(empirical, xi)
        mf_term = (agree_y * y_bar).mean(dim=0, keepdim=True)
        mf_norm = torch.linalg.norm(mf_term, dim=-1, keepdim=True).clamp_min(EPS)
        mf_drift = mf_term / mf_norm

        agree_x = self._signed_projection(x, xi)
        return self.polarize_strength * fmul * agree_x * mf_drift

    def evaluate(
        self,
        x: Tensor,
        empirical: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        if empirical is None:
            empirical = x
        if t is None:
            idx = x.new_tensor(0, dtype=torch.long)
            return self._evaluate_index(x, empirical, idx)
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        idx = self._time_to_index(t.to(x.device))
        unique_idx = torch.unique(idx)
        out = torch.empty_like(x)
        for idx_val in unique_idx:
            mask = idx == idx_val
            empirical_t = (
                empirical[mask] if empirical.shape[0] == x.shape[0] else empirical
            )
            out[mask] = self._evaluate_index(x[mask], empirical_t, idx_val)
        return out

    def evaluate_batched(
        self,
        x: Tensor,
        empirical: Optional[Tensor] = None,
        *,
        time_grid: Optional[Tensor] = None,
        time_block_size: Optional[int] = None,
    ) -> Tensor:
        """Evaluate time-local opinion drift for `(time, batch, dim)` inputs."""
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
        if time_grid is None:
            idx_grid = torch.arange(n_steps, device=x.device).clamp_max(self.n_steps - 1)
        else:
            if time_grid.dim() == 2:
                time_grid = time_grid[:, 0]
            idx_grid = self._time_to_index(time_grid.to(x.device))

        block = n_steps if time_block_size is None else min(time_block_size, n_steps)
        out = torch.empty_like(x)
        for start in range(0, n_steps, block):
            end = min(start + block, n_steps)
            for local_i, idx_val in enumerate(idx_grid[start:end]):
                i = start + local_i
                out[i] = self._evaluate_index(x[i], empirical[i], idx_val)
        return out

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # x: (B, dim), t: (B,). Particles sharing a time value form the
        # empirical cloud for that time slice.
        return self.evaluate(x, x, t)


def build_opinion_xi_schedule(
    n_steps: int,
    dim: int,
    seed: Optional[int] = None,
    bm_std: float = 0.4,
    dt: float = 0.01,
) -> Tensor:
    """Generate Brownian-smoothed unit vectors for xi schedule."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    with torch_seed_context(seed):
        rng = torch.randn(n_steps, dim)
    xi = rng[0]
    xs = [xi]
    for t in range(1, n_steps):
        xi = xi - (2.0 * xi) * dt + bm_std * math.sqrt(dt) * rng[t]
        xs.append(xi)
    xis = torch.stack(xs, dim=0)
    xis = xis / torch.linalg.norm(xis, dim=-1, keepdim=True).clamp_min(EPS)
    return xis


def build_opinion_f_mul(
    n_steps: int, T: float, coeff: float = 8.0, power: float = 5.0
) -> Tensor:
    """Build decay schedule f_mul to prevent drift blow-up."""
    ts = torch.linspace(0.0, T, n_steps)
    f_mul = torch.clip(1.0 - torch.exp(coeff * (ts - T)) + 1e-5, min=1e-4, max=1.0)
    f_mul = f_mul**power
    return f_mul


__all__ = [
    "AttractionRepulsionDrift",
    "CoulombDrift",
    "Drift",
    "GaussianDrift",
    "LaplacianDiffusionDrift",
    "MeanReversionDrift",
    "OpinionMeanFieldDrift",
    "SmoothBoundedConfidenceDrift",
    "build_opinion_xi_schedule",
    "build_opinion_f_mul",
    "ZeroDrift",
]
