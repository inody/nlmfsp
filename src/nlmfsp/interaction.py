"""Interaction learning for mean-field terms.

This module implements supervised learning of interaction terms following Han et al.:
    f_θ(x,t) ≈ f(x,t,ρ(·,t))  (mean-field drift)
    F_φ(x,t) ≈ F(x,t,ρ(·,t))  (mean-field running cost)

The key idea is to:
1. Compute true interaction from empirical distribution (O(N²))
2. Train neural networks to approximate these interactions (supervised learning)
3. Use learned networks for O(N) evaluation in FBSDE solving
"""

import time
import copy
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from nlmfsp.drifts import (
    OpinionMeanFieldDrift,
    build_opinion_f_mul,
    build_opinion_xi_schedule,
)

from .analytical_interactions import (
    AnalyticalAttractionRepulsionDrift,
    AnalyticalCongestionCost,
    AnalyticalCoulombCost,
    AnalyticalCoulombDrift,
    AnalyticalEntropyCost,
    AnalyticalFlockingAlignmentCost,
    AnalyticalGaussianConvolutionCost,
    AnalyticalGaussianConvolutionMeanCost,
    AnalyticalGaussianDrift,
    AnalyticalLaplacianDiffusionDrift,
    AnalyticalLocalExclusionCost,
    AnalyticalMeanDeviationCost,
    AnalyticalMeanReversionDrift,
    AnalyticalNonstationaryAnisotropicDrift,
    AnalyticalPedestrianVisualCost,
    AnalyticalQuadraticDeviationCost,
    AnalyticalQuadraticPotentialCost,
    AnalyticalSmoothBoundedConfidenceDrift,
    AnalyticalZeroCost,
    AnalyticalZeroDrift,
)
from .config import Config
from .networks import InteractionNet


class TimedInteractionModule(nn.Module):
    """Wrap an interaction network and accumulate inference timing stats."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self._timing_enabled = True
        self._eval_time_total = 0.0
        self._call_count = 0.0

    def set_timing_enabled(self, enabled: bool) -> None:
        self._timing_enabled = enabled

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not self._timing_enabled:
            return self.net(x, t)
        eval_start = time.perf_counter()
        out = self.net(x, t)
        self._eval_time_total += time.perf_counter() - eval_start
        self._call_count += 1.0
        return out

    def timing_stats(self) -> Dict[str, float]:
        return {
            "kind": "nn",
            "sample_time_total": 0.0,
            "eval_time_total": self._eval_time_total,
            "call_count": float(self._call_count),
        }


class TransformedInteractionModule(nn.Module):
    """Apply interaction input scaling and optional target standardization."""

    def __init__(
        self,
        net: nn.Module,
        output_dim: int,
        *,
        input_scale: float,
        standardize_targets: bool,
    ):
        super().__init__()
        self.net = net
        self.input_scale = float(input_scale)
        self.standardize_targets = bool(standardize_targets)
        self.register_buffer("target_mean", torch.zeros(output_dim))
        self.register_buffer("target_std", torch.ones(output_dim))

    def set_target_stats(self, targets: torch.Tensor) -> None:
        if not self.standardize_targets:
            self.target_mean.zero_()
            self.target_std.fill_(1.0)
            return
        flat = targets.detach().reshape(-1, targets.shape[-1])
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, correction=0).clamp_min(1e-8)
        self.target_mean.copy_(mean)
        self.target_std.copy_(std)

    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets - self.target_mean) / self.target_std

    def forward_normalized(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(x / self.input_scale, t)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.forward_normalized(x, t) * self.target_std + self.target_mean

    def set_timing_enabled(self, enabled: bool) -> None:
        if hasattr(self.net, "set_timing_enabled"):
            self.net.set_timing_enabled(enabled)

    def timing_stats(self) -> Dict[str, float]:
        if hasattr(self.net, "timing_stats"):
            return self.net.timing_stats()
        return {
            "kind": "nn",
            "sample_time_total": 0.0,
            "eval_time_total": 0.0,
            "call_count": 0.0,
        }


def _make_learned_interaction_module(
    config: Config, *, output_dim: int
) -> nn.Module:
    module: nn.Module = TimedInteractionModule(InteractionNet(config, output_dim=output_dim))
    if config.standardize_interaction_targets or config.interaction_input_scale != 1.0:
        module = TransformedInteractionModule(
            module,
            output_dim=output_dim,
            input_scale=config.interaction_input_scale,
            standardize_targets=config.standardize_interaction_targets,
        )
    return module.to(config.device)


class RandomBatchDrift(nn.Module):
    """Approximate analytical drift using a random empirical subset of size M."""

    def __init__(self, drift: nn.Module, sample_size: int):
        super().__init__()
        self._drift_module = drift
        self.sample_size = sample_size
        base_drift = getattr(drift, "drift", drift)
        self.requires_time = bool(getattr(base_drift, "requires_time", False))
        self._sample_time_total = 0.0
        self._eval_time_total = 0.0
        self._call_count = 0

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x.new_zeros(x.shape)
        base_drift = getattr(self._drift_module, "drift", self._drift_module)
        take = min(self.sample_size, x.shape[0])
        empirical_subset = self._sample_empirical_subset(x, take)
        eval_start = time.perf_counter()
        if hasattr(base_drift, "evaluate"):
            if getattr(base_drift, "requires_time", False):
                out = base_drift.evaluate(x, empirical_subset, t)
            else:
                out = base_drift.evaluate(x, empirical_subset)
        else:
            out = base_drift(x, t)
        out = self._apply_aggregation_scaling(
            base_drift, out, n_ref=x.shape[0], take=take
        )
        self._eval_time_total += time.perf_counter() - eval_start
        self._call_count += 1
        return out

    def evaluate_batched(
        self,
        x: torch.Tensor,
        empirical: torch.Tensor,
        *,
        time_grid: Optional[torch.Tensor] = None,
        time_block_size: Optional[int] = None,
    ) -> torch.Tensor:
        if x.dim() != 3 or empirical.dim() != 3:
            raise ValueError(
                "RandomBatchDrift.evaluate_batched expects (time, batch, dim) tensors."
            )
        base_drift = getattr(self._drift_module, "drift", self._drift_module)
        if not hasattr(base_drift, "evaluate_batched"):
            return torch.stack(
                [
                    self.forward(x_t, x_t.new_zeros(x_t.shape[0]))
                    for x_t in x
                ],
                dim=0,
            )
        take = min(self.sample_size, empirical.shape[1])
        empirical_subset = self._sample_empirical_subset_batched(empirical, take)
        eval_start = time.perf_counter()
        kwargs = {"time_block_size": time_block_size}
        if getattr(base_drift, "requires_time", False):
            kwargs["time_grid"] = time_grid
        out = base_drift.evaluate_batched(x, empirical_subset, **kwargs)
        out = self._apply_aggregation_scaling(
            base_drift,
            out,
            n_ref=empirical.shape[1],
            take=take,
        )
        self._eval_time_total += time.perf_counter() - eval_start
        self._call_count += float(x.shape[0])
        return out

    def _apply_aggregation_scaling(
        self,
        base_drift: nn.Module,
        out: torch.Tensor,
        *,
        n_ref: int,
        take: int,
    ) -> torch.Tensor:
        aggregation = getattr(base_drift, "random_batch_aggregation", "mean")
        if aggregation == "mean":
            return out
        if aggregation != "sum":
            raise ValueError(
                "random_batch_aggregation must be 'mean' or 'sum', "
                f"got {aggregation!r} for {base_drift.__class__.__name__}."
            )
        if take <= 0 or take >= n_ref:
            return out
        return out * (float(n_ref) / float(take))

    def _sample_empirical_subset(self, empirical: torch.Tensor, take: int) -> torch.Tensor:
        sample_start = time.perf_counter()
        if take >= empirical.shape[0]:
            subset = empirical
        else:
            indices = torch.randperm(empirical.shape[0], device=empirical.device)[:take]
            subset = empirical.index_select(0, indices)
        self._sample_time_total += time.perf_counter() - sample_start
        return subset

    def _sample_empirical_subset_batched(
        self, empirical: torch.Tensor, take: int
    ) -> torch.Tensor:
        sample_start = time.perf_counter()
        if take >= empirical.shape[1]:
            subset = empirical
        else:
            subsets = []
            for empirical_t in empirical:
                indices = torch.randperm(empirical_t.shape[0], device=empirical_t.device)[
                    :take
                ]
                subsets.append(empirical_t.index_select(0, indices))
            subset = torch.stack(subsets, dim=0)
        self._sample_time_total += time.perf_counter() - sample_start
        return subset

    def timing_stats(self) -> Dict[str, float]:
        return {
            "kind": "random_batch",
            "sample_time_total": self._sample_time_total,
            "eval_time_total": self._eval_time_total,
            "call_count": float(self._call_count),
        }


class RandomBatchCost(nn.Module):
    """Approximate analytical running cost using a random empirical subset of size M."""

    def __init__(self, cost: nn.Module, sample_size: int):
        super().__init__()
        self._cost_module = cost
        self.sample_size = sample_size
        self._sample_time_total = 0.0
        self._eval_time_total = 0.0
        self._call_count = 0

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x.new_zeros((0, 1))
        base_cost = getattr(self._cost_module, "cost", self._cost_module)
        take = min(self.sample_size, x.shape[0])
        empirical_subset = self._sample_empirical_subset(x, take)
        eval_start = time.perf_counter()
        if hasattr(base_cost, "evaluate"):
            out = base_cost.evaluate(x, empirical_subset)
        else:
            out = base_cost(x, t)
        out = self._apply_aggregation_scaling(
            base_cost, out, n_ref=x.shape[0], take=take
        )
        self._eval_time_total += time.perf_counter() - eval_start
        self._call_count += 1
        return out

    def evaluate_batched(
        self,
        x: torch.Tensor,
        empirical: torch.Tensor,
        *,
        time_block_size: Optional[int] = None,
    ) -> torch.Tensor:
        if x.dim() != 3 or empirical.dim() != 3:
            raise ValueError(
                "RandomBatchCost.evaluate_batched expects (time, batch, dim) tensors."
            )
        base_cost = getattr(self._cost_module, "cost", self._cost_module)
        if not hasattr(base_cost, "evaluate_batched"):
            return torch.stack(
                [
                    self.forward(x_t, x_t.new_zeros(x_t.shape[0])).squeeze(-1)
                    for x_t in x
                ],
                dim=0,
            ).unsqueeze(-1)
        take = min(self.sample_size, empirical.shape[1])
        empirical_subset = self._sample_empirical_subset_batched(empirical, take)
        eval_start = time.perf_counter()
        out = base_cost.evaluate_batched(
            x,
            empirical_subset,
            time_block_size=time_block_size,
        )
        out = self._apply_aggregation_scaling(
            base_cost,
            out,
            n_ref=empirical.shape[1],
            take=take,
        )
        self._eval_time_total += time.perf_counter() - eval_start
        self._call_count += float(x.shape[0])
        return out

    def _apply_aggregation_scaling(
        self,
        base_cost: nn.Module,
        out: torch.Tensor,
        *,
        n_ref: int,
        take: int,
    ) -> torch.Tensor:
        aggregation = getattr(base_cost, "random_batch_aggregation", "mean")
        if aggregation == "mean":
            return out
        if aggregation != "sum":
            raise ValueError(
                "random_batch_aggregation must be 'mean' or 'sum', "
                f"got {aggregation!r} for {base_cost.__class__.__name__}."
            )
        if take <= 0 or take >= n_ref:
            return out
        return out * (float(n_ref) / float(take))

    def _sample_empirical_subset(self, empirical: torch.Tensor, take: int) -> torch.Tensor:
        sample_start = time.perf_counter()
        if take >= empirical.shape[0]:
            subset = empirical
        else:
            indices = torch.randperm(empirical.shape[0], device=empirical.device)[:take]
            subset = empirical.index_select(0, indices)
        self._sample_time_total += time.perf_counter() - sample_start
        return subset

    def _sample_empirical_subset_batched(
        self, empirical: torch.Tensor, take: int
    ) -> torch.Tensor:
        sample_start = time.perf_counter()
        if take >= empirical.shape[1]:
            subset = empirical
        else:
            subsets = []
            for empirical_t in empirical:
                indices = torch.randperm(empirical_t.shape[0], device=empirical_t.device)[
                    :take
                ]
                subsets.append(empirical_t.index_select(0, indices))
            subset = torch.stack(subsets, dim=0)
        self._sample_time_total += time.perf_counter() - sample_start
        return subset

    def timing_stats(self) -> Dict[str, float]:
        return {
            "kind": "random_batch",
            "sample_time_total": self._sample_time_total,
            "eval_time_total": self._eval_time_total,
            "call_count": float(self._call_count),
        }


def create_interactions(
    config: Config,
) -> Tuple[
    torch.nn.Module,
    torch.nn.Module,
    Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]],
    Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    torch.nn.Module,
    torch.nn.Module,
]:
    """Create analytical or learned interaction modules from Config."""
    interaction_backend = config.interaction_backend
    learn_interaction_drift = config.learn_interaction_drift
    learn_interaction_cost = config.learn_interaction_cost
    learn_interaction_enabled = bool(learn_interaction_drift or learn_interaction_cost)

    int_drift_weight: float = float(config.int_drift_weight or 0.0)
    int_drift_sigma: float = float(config.int_drift_sigma or 1.0)
    int_cost_weight: float = float(config.int_cost_weight or 0.0)
    int_cost_sigma: float = float(
        config.int_cost_sigma if config.int_cost_sigma is not None else int_drift_sigma
    )

    drift_type = config.drift_type
    if drift_type == "mean_reversion":
        drift = AnalyticalMeanReversionDrift(weight=int_drift_weight)
    elif drift_type == "zero":
        drift = AnalyticalZeroDrift()
    elif drift_type == "gaussian":
        drift = AnalyticalGaussianDrift(sigma=int_drift_sigma, weight=int_drift_weight)
    elif drift_type == "nonstationary_anisotropic":
        alpha = (
            1.0
            if config.int_drift_strength is None
            else float(config.int_drift_strength)
        )
        drift = AnalyticalNonstationaryAnisotropicDrift(
            sigma=int_drift_sigma,
            weight=int_drift_weight,
            alpha=alpha,
        )
    elif drift_type == "attraction_repulsion":
        drift = AnalyticalAttractionRepulsionDrift(weight=int_drift_weight)
    elif drift_type == "coulomb":
        drift = AnalyticalCoulombDrift(weight=int_drift_weight)
    elif drift_type == "laplacian":
        drift = AnalyticalLaplacianDiffusionDrift(
            sigma=int_drift_sigma, weight=int_drift_weight
        )
    elif drift_type == "bounded_confidence":
        drift = AnalyticalSmoothBoundedConfidenceDrift(
            sigma=int_drift_sigma, weight=int_drift_weight
        )
    elif drift_type == "opinion":
        schedule_steps = config.n_time_steps + 1
        xi_schedule = build_opinion_xi_schedule(
            n_steps=schedule_steps,
            dim=config.dim,
            seed=getattr(config, "seed", None),
            dt=config.dt,
        )
        f_mul = build_opinion_f_mul(schedule_steps, config.T)
        polarize_strength = 1.0 if config.dim <= 2 else 6.0
        drift = OpinionMeanFieldDrift(
            xi_schedule=xi_schedule.to(config.device),
            f_mul=f_mul.to(config.device),
            dt=config.dt,
            T=config.T,
            polarize_strength=polarize_strength,
        )
    else:
        raise ValueError(f"Unknown drift type: {drift_type}")
    drift = drift.to(config.device)

    cost_type = config.cost_type
    if cost_type == "zero":
        cost = AnalyticalZeroCost()
    elif cost_type == "quadratic_deviation":
        cost = AnalyticalQuadraticDeviationCost(weight=int_cost_weight)
    elif cost_type == "gaussian_conv":
        cost = AnalyticalGaussianConvolutionCost(
            sigma=int_cost_sigma, weight=int_cost_weight
        )
    elif cost_type == "gaussian_conv_mean":
        cost = AnalyticalGaussianConvolutionMeanCost(
            sigma=int_cost_sigma, weight=int_cost_weight
        )
    elif cost_type == "pedestrian_visual":
        cost = AnalyticalPedestrianVisualCost(
            range_scale=int_cost_sigma,
            weight=int_cost_weight,
        )
    elif cost_type == "mean_deviation":
        cost = AnalyticalMeanDeviationCost(weight=int_cost_weight)
    elif cost_type == "quadratic_potential":
        cost = AnalyticalQuadraticPotentialCost(weight=int_cost_weight)
    elif cost_type == "coulomb":
        cost = AnalyticalCoulombCost()
    elif cost_type == "local_exclusion":
        cost = AnalyticalLocalExclusionCost()
    elif cost_type == "entropy":
        cost = AnalyticalEntropyCost(weight=int_cost_weight)
    elif cost_type == "congestion":
        cost = AnalyticalCongestionCost(weight=int_cost_weight)
    elif cost_type == "flocking_alignment":
        cost = AnalyticalFlockingAlignmentCost(
            sigma=int_cost_sigma, weight=int_cost_weight
        )
    else:
        raise ValueError(f"Unknown cost type: {cost_type}")
    cost = cost.to(config.device)

    analytical_drift_module = drift
    analytical_cost_module = cost

    base_drift = getattr(drift, "drift", drift)
    if hasattr(base_drift, "set_chunk_sizes"):
        base_drift.set_chunk_sizes(
            query_chunk_size=config.interaction_query_chunk_size,
            ref_chunk_size=config.interaction_ref_chunk_size,
        )

    base_cost = getattr(cost, "cost", cost)
    if hasattr(base_cost, "set_chunk_sizes"):
        base_cost.set_chunk_sizes(
            query_chunk_size=config.interaction_query_chunk_size,
            ref_chunk_size=config.interaction_ref_chunk_size,
        )

    drift_module: Optional[torch.nn.Module]
    cost_module: Optional[torch.nn.Module]
    f_interaction_fn: Optional[
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None
    F_interaction_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = (
        None
    )

    if interaction_backend == "random_batch":
        sample_size = int(config.random_batch_size or 0)
        drift_module = RandomBatchDrift(drift, sample_size).to(config.device)
        cost_module = RandomBatchCost(cost, sample_size).to(config.device)
    elif learn_interaction_enabled:
        if learn_interaction_drift:
            drift_net = _make_learned_interaction_module(
                config, output_dim=config.dim
            )

            def f_interaction_fn(
                x: torch.Tensor, empirical: torch.Tensor, t: torch.Tensor
            ) -> torch.Tensor:
                base_drift = getattr(drift, "drift", drift)
                if hasattr(base_drift, "evaluate"):
                    if getattr(base_drift, "requires_time", False):
                        return base_drift.evaluate(x, empirical, t)
                    return base_drift.evaluate(x, empirical)
                return base_drift(x, t)

            setattr(f_interaction_fn, "base_drift", base_drift)

            drift_module = drift_net
        else:
            drift_module = drift

        if learn_interaction_cost:
            cost_net = _make_learned_interaction_module(config, output_dim=1)

            def F_interaction_fn(
                x: torch.Tensor, empirical: torch.Tensor
            ) -> torch.Tensor:
                return cost.cost.evaluate(x, empirical)

            setattr(F_interaction_fn, "base_cost", base_cost)

            cost_module = cost_net
        else:
            cost_module = cost
    else:
        drift_module = drift
        cost_module = cost

    return (
        drift_module,
        cost_module,
        f_interaction_fn,
        F_interaction_fn,
        analytical_drift_module,
        analytical_cost_module,
    )


class InteractionLearner:
    """Learner for mean-field interaction terms.

    This class handles:
    1. Computing true interactions f̃(x,t) and F̃(x,t) from particle samples
    2. Training neural networks f_θ and F_φ to approximate these interactions
    """

    def __init__(
        self,
        config: Config,
        f_interaction_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        F_interaction_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ):
        """Initialize interaction learner.

        Args:
            config: Configuration object
            f_interaction_fn: Optional function to compute f(x,t,ρ) from empirical distribution
                Signature: (x, empirical_samples, t) -> f̃(x,t)
                where x: (batch_size, dim), empirical_samples: (N, dim)
                If None, f is not learned (drift interaction disabled)
            F_interaction_fn: Optional function to compute F(x,t,ρ) from empirical distribution
                Signature: (x, empirical_samples) -> F̃(x,t)
                where x: (batch_size, dim), empirical_samples: (N, dim)
                If None, F is not learned (cost interaction disabled)

        Note:
            At least one of f_interaction_fn or F_interaction_fn must be provided.
            This allows flexible combinations:
            - f only: Learn drift interaction, use analytical cost
            - F only: Use analytical drift, learn cost interaction
            - Both: Learn both interactions (Han et al. full method)
        """
        if f_interaction_fn is None and F_interaction_fn is None:
            raise ValueError(
                "At least one of f_interaction_fn or F_interaction_fn must be provided"
            )

        self.config = config

        # True interaction functions (O(N²) computation)
        self.f_interaction_fn = f_interaction_fn
        self.F_interaction_fn = F_interaction_fn

        # Neural network approximators (conditional creation)
        self.f_net = None
        self.F_net = None
        self.f_optimizer = None
        self.F_optimizer = None
        self.optimizer = None

        if f_interaction_fn is not None:
            self.f_net = _make_learned_interaction_module(
                config, output_dim=config.dim
            )
            self.f_optimizer = torch.optim.Adam(
                self.f_net.parameters(),
                lr=config.learning_rate_interaction,
            )

        if F_interaction_fn is not None:
            self.F_net = _make_learned_interaction_module(config, output_dim=1)
            self.F_optimizer = torch.optim.Adam(
                self.F_net.parameters(),
                lr=config.learning_rate_interaction,
            )

    def compute_true_interactions(
        self,
        xs: torch.Tensor,
        time_grid: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute true interactions from trajectory samples.

        Given trajectory samples {X_t^i}_{i=1}^N, compute empirical interactions:
            f̃_t^i = f(X_t^i, m^(k-1)(·,t))
            F̃_t^i = F(X_t^i, m^(k-1)(·,t))
        where m^(k-1)(·,t) = (1/N)Σ_j δ(· - X_t^j)

        Args:
            xs: Trajectory samples (n_steps+1, n_agents, dim)
            time_grid: Time points (n_steps+1,)

        Returns:
            f_targets: Target drift interactions (n_steps+1, n_agents, dim) or None
            F_targets: Target cost interactions (n_steps+1, n_agents, 1) or None
        """
        n_steps = xs.shape[0] - 1
        n_agents = xs.shape[1]  # Number of agents (N)

        f_targets = None
        F_targets = None

        with torch.no_grad():
            if self.f_interaction_fn is not None:
                base_drift = getattr(self.f_interaction_fn, "base_drift", None)
                time_block_size = getattr(
                    self.config, "interaction_time_block_size", None
                )
                if base_drift is not None and hasattr(base_drift, "evaluate_batched"):
                    kwargs = {"time_block_size": time_block_size}
                    if getattr(base_drift, "requires_time", False):
                        kwargs["time_grid"] = time_grid
                    f_targets = base_drift.evaluate_batched(xs, xs, **kwargs)
                else:
                    f_targets = torch.zeros_like(xs)
                    for i in range(n_steps + 1):
                        empirical_samples = xs[i]
                        t_val = time_grid[i]
                        f_targets[i] = self.f_interaction_fn(
                            xs[i], empirical_samples, t_val
                        )

            if self.F_interaction_fn is not None:
                base_cost = getattr(self.F_interaction_fn, "base_cost", None)
                time_block_size = getattr(
                    self.config, "interaction_time_block_size", None
                )
                if base_cost is not None and hasattr(base_cost, "evaluate_batched"):
                    F_targets = base_cost.evaluate_batched(
                        xs,
                        xs,
                        time_block_size=time_block_size,
                    )
                else:
                    F_targets = torch.zeros(n_steps + 1, n_agents, 1, device=xs.device)
                    for i in range(n_steps + 1):
                        empirical_samples = xs[i]
                        F_targets[i] = self.F_interaction_fn(xs[i], empirical_samples)

        return f_targets, F_targets

    def _prepare_training_batch(
        self,
        xs: torch.Tensor,
        time_grid: torch.Tensor,
        f_targets: Optional[torch.Tensor],
        F_targets: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Flatten trajectory inputs and labels once for repeated optimizer steps."""
        batch_size = xs.shape[1]
        xs_flat = xs.reshape(-1, self.config.dim)
        time_flat = time_grid.repeat_interleave(batch_size)
        f_targets_flat = (
            None if f_targets is None else f_targets.reshape(-1, self.config.dim)
        )
        F_targets_flat = None if F_targets is None else F_targets.reshape(-1, 1)
        return xs_flat, time_flat, f_targets_flat, F_targets_flat

    def _train_step_precomputed(
        self,
        xs_flat: torch.Tensor,
        time_flat: torch.Tensor,
        f_targets_flat: Optional[torch.Tensor],
        F_targets_flat: Optional[torch.Tensor],
    ) -> Tuple[float, float]:
        """Perform one optimizer step from pre-flattened inputs and labels."""
        f_loss_val = 0.0
        F_loss_val = 0.0

        if self.optimizer is not None:
            total_loss = 0.0
            if self.f_net is not None and f_targets_flat is not None:
                f_pred = self._forward_for_training(self.f_net, xs_flat, time_flat)
                f_loss = nn.functional.mse_loss(f_pred, f_targets_flat)
                f_loss_val = f_loss.item()
                total_loss = total_loss + f_loss

            if self.F_net is not None and F_targets_flat is not None:
                F_pred = self._forward_for_training(self.F_net, xs_flat, time_flat)
                F_loss = nn.functional.mse_loss(F_pred, F_targets_flat)
                F_loss_val = F_loss.item()
                total_loss = total_loss + F_loss

            if total_loss != 0.0:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            return f_loss_val, F_loss_val

        if self.f_net is not None and f_targets_flat is not None:
            self.f_optimizer.zero_grad()
            f_pred = self._forward_for_training(self.f_net, xs_flat, time_flat)
            f_loss = nn.functional.mse_loss(f_pred, f_targets_flat)
            f_loss.backward()
            self.f_optimizer.step()
            f_loss_val = f_loss.item()

        if self.F_net is not None and F_targets_flat is not None:
            self.F_optimizer.zero_grad()
            F_pred = self._forward_for_training(self.F_net, xs_flat, time_flat)
            F_loss = nn.functional.mse_loss(F_pred, F_targets_flat)
            F_loss.backward()
            self.F_optimizer.step()
            F_loss_val = F_loss.item()

        return f_loss_val, F_loss_val

    @staticmethod
    def _forward_for_training(
        module: nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if hasattr(module, "forward_normalized"):
            return module.forward_normalized(x, t)
        return module(x, t)

    @staticmethod
    def _prepare_targets_for_training(
        module: Optional[nn.Module], targets: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if module is None or targets is None:
            return targets
        if hasattr(module, "set_target_stats") and hasattr(module, "normalize_targets"):
            module.set_target_stats(targets)
            return module.normalize_targets(targets)
        return targets

    def train_step(
        self,
        xs: torch.Tensor,
        time_grid: torch.Tensor,
        f_targets: Optional[torch.Tensor],
        F_targets: Optional[torch.Tensor],
    ) -> Tuple[float, float]:
        """Perform one training step.

        Args:
            xs: State samples (n_steps+1, batch_size, dim)
            time_grid: Time points (n_steps+1,)
            f_targets: Target drift interactions (n_steps+1, batch_size, dim) or None
            F_targets: Target cost interactions (n_steps+1, batch_size, 1) or None

        Returns:
            f_loss: Loss for f_θ network (0.0 if not learning f)
            F_loss: Loss for F_φ network (0.0 if not learning F)
        """
        xs_flat, time_flat, f_targets_flat, F_targets_flat = self._prepare_training_batch(
            xs, time_grid, f_targets, F_targets
        )
        return self._train_step_precomputed(
            xs_flat,
            time_flat,
            f_targets_flat,
            F_targets_flat,
        )

    def train(
        self,
        xs: torch.Tensor,
        time_grid: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> dict:
        """Train interaction networks on trajectory samples.

        Args:
            xs: Trajectory samples (n_steps+1, batch_size, dim)
            time_grid: Time points (n_steps+1,)
            n_steps: Number of training steps (default: config.n_interaction_steps)

        Returns:
            Training history dict with losses
        """
        if n_steps is None:
            n_steps = self.config.n_interaction_steps

        timed_modules = [module for module in (self.f_net, self.F_net) if module is not None]
        for module in timed_modules:
            if hasattr(module, "set_timing_enabled"):
                module.set_timing_enabled(False)

        try:
            # Compute true interactions (O(N²) computation)
            f_targets, F_targets = self.compute_true_interactions(xs, time_grid)
            f_targets = self._prepare_targets_for_training(self.f_net, f_targets)
            F_targets = self._prepare_targets_for_training(self.F_net, F_targets)
            xs_flat, time_flat, f_targets_flat, F_targets_flat = self._prepare_training_batch(
                xs,
                time_grid,
                f_targets,
                F_targets,
            )
            # Train networks
            history = {"f_loss": [], "F_loss": []}
            best_total_loss = float("inf")
            best_f_state = None
            best_F_state = None

            for step in range(n_steps):
                f_loss, F_loss = self._train_step_precomputed(
                    xs_flat,
                    time_flat,
                    f_targets_flat,
                    F_targets_flat,
                )
                history["f_loss"].append(f_loss)
                history["F_loss"].append(F_loss)
                total_loss = 0.0
                if self.f_net is not None:
                    total_loss += f_loss
                if self.F_net is not None:
                    total_loss += F_loss
                if total_loss < best_total_loss:
                    best_total_loss = total_loss
                    if self.f_net is not None:
                        best_f_state = copy.deepcopy(self.f_net.state_dict())
                    if self.F_net is not None:
                        best_F_state = copy.deepcopy(self.F_net.state_dict())

            if self.f_net is not None and best_f_state is not None:
                self.f_net.load_state_dict(best_f_state)
            if self.F_net is not None and best_F_state is not None:
                self.F_net.load_state_dict(best_F_state)

            return history
        finally:
            for module in timed_modules:
                if hasattr(module, "set_timing_enabled"):
                    module.set_timing_enabled(True)

    def evaluate(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate learned interactions (O(N) complexity).

        Args:
            x: State (batch_size, dim)
            t: Time (batch_size,) or scalar

        Returns:
            f_pred: Predicted drift (batch_size, dim)
            F_pred: Predicted cost (batch_size, 1)
        """
        if self.f_net is None:
            raise ValueError("InteractionLearner.evaluate requires f_net to be initialized.")
        if self.F_net is None:
            raise ValueError("InteractionLearner.evaluate requires F_net to be initialized.")

        with torch.no_grad():
            f_pred = self.f_net(x, t)
            F_pred = self.F_net(x, t)

        return f_pred, F_pred

    def benchmark_analytical_vs_learned(
        self,
        xs: torch.Tensor,
        time_grid: torch.Tensor,
        n_warmup: int = 5,
        n_trials: int = 20,
    ) -> Dict[str, any]:
        """Benchmark analytical vs learned interaction computation.

        This method measures the wall-clock time for:
        1. Analytical computation: O(N²) for N particles per timestep
        2. Learned network inference: O(N) for N particles (O(1) per particle)

        Only benchmarks the interactions that are being learned (f and/or F).

        Note on complexity:
        - Analytical: Each particle requires O(N) comparisons → O(N²) total
        - Learned: Each particle requires O(1) NN evaluation → O(N) total
        - For trajectory of T timesteps: O(TN²) vs O(TN)

        Args:
            xs: Trajectory samples (n_steps+1, batch_size, dim)
            time_grid: Time points (n_steps+1,)
            n_warmup: Number of warmup runs (default: 5)
            n_trials: Number of benchmark trials (default: 20)

        Returns:
            results: Dictionary containing timing statistics. Fields depend on which
                    interactions are being learned:
                - If learning f: 'analytical_f', 'learned_f', 'speedup_f'
                - If learning F: 'analytical_F', 'learned_F', 'speedup_F'
                - Always: 'batch_size', 'n_steps', 'dim', 'device'
        """
        from .benchmark_utils import compare_functions

        n_steps = xs.shape[0] - 1
        batch_size = xs.shape[1]
        device = xs.device

        # Prepare flattened inputs for learned networks
        xs_flat = xs.reshape(-1, self.config.dim)
        time_flat = time_grid.repeat_interleave(batch_size)

        results = {
            "batch_size": batch_size,
            "n_steps": n_steps + 1,
            "dim": self.config.dim,
            "device": str(device),
        }

        # === Benchmark f (drift) if learning ===
        if self.f_interaction_fn is not None and self.f_net is not None:

            def analytical_f():
                for i in range(n_steps + 1):
                    t = time_grid[i].expand(xs[i].shape[0])
                    _ = self.f_interaction_fn(xs[i], xs[i], t)

            def learned_f():
                with torch.no_grad():
                    _ = self.f_net(xs_flat, time_flat)

            comparison = compare_functions(
                analytical_f, learned_f, n_trials, n_warmup, device
            )

            results["analytical_f"] = comparison["analytical"]
            results["learned_f"] = comparison["learned"]
            results["speedup_f"] = comparison["speedup"]

        # === Benchmark F (cost) if learning ===
        if self.F_interaction_fn is not None and self.F_net is not None:

            def analytical_F():
                for i in range(n_steps + 1):
                    _ = self.F_interaction_fn(xs[i], xs[i])

            def learned_F():
                with torch.no_grad():
                    _ = self.F_net(xs_flat, time_flat)

            comparison = compare_functions(
                analytical_F, learned_F, n_trials, n_warmup, device
            )

            results["analytical_F"] = comparison["analytical"]
            results["learned_F"] = comparison["learned"]
            results["speedup_F"] = comparison["speedup"]

        return results
