"""Forward-Backward SDE solver for Mean-Field Schrödinger Bridge.

This module implements the FBSDE-based solver following Liu et al. (2022)
"Deep Generalized Schrödinger Bridge" with extensions for learned interaction terms.

References:
    Liu, G.-H., et al. (2022). Deep Generalized Schrödinger Bridge. NeurIPS.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import time
from typing import Callable, Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn

from . import tensor_utils
from .config import Config
from .networks import ToyGradientNet, ToyValueNet
from .rng_utils import seed_offset, torch_seed_context

Tensor = torch.Tensor
logger = logging.getLogger(__name__)


class FBSDESolver:
    """FBSDE solver with learned mean-field interaction terms.

    This solver implements Liu (2022)'s FBSDE losses (L_IPF, L_TD, L_FK) with
    extensions to handle learned interaction terms f_α(x,t) and F_β(x,t) that
    approximate f(x,t,ρ) and F(x,t,ρ) without requiring O(N²) computations.

    Attributes:
        config: Configuration object
        value_net: Y(x,t) network (log Ψ)
        value_hat_net: Ŷ(x,t) network (log Ψ̂)
        grad_net: Z(x,t) = σ∇Y network
        grad_hat_net: Ẑ(x,t) = σ∇Ŷ network
        drift_net: Learned f_α(x,t) network (optional)
        cost_net: Learned F_β(x,t) network (optional)
        base_drift_fn: Optional base drift function (e.g., λx)

    Example:
        >>> config = Config(dim=2, T=1.0, dt=0.01, sigma=1.0)
        >>> solver = FBSDESolver(config)
        >>> # Sample forward trajectory
        >>> x0 = torch.randn(256, 2)
        >>> xs, ts, dws = solver.sample_forward_trajectory(x0)
    """

    def __init__(
        self,
        config: Config,
        value_net: Optional[nn.Module] = None,
        value_hat_net: Optional[nn.Module] = None,
        grad_net: Optional[nn.Module] = None,
        grad_hat_net: Optional[nn.Module] = None,
        drift_net: Optional[nn.Module] = None,
        cost_net: Optional[nn.Module] = None,
        base_drift_fn: Optional[Callable[[Tensor], Tensor]] = None,
        base_cost_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ):
        """Initialize FBSDE solver.

        Args:
            config: Configuration object
            value_net: Y(x,t) network (if None, created automatically)
            value_hat_net: Ŷ(x,t) network (if None, created automatically)
            grad_net: Z(x,t) network (if None, created automatically)
            grad_hat_net: Ẑ(x,t) network (if None, created automatically)
            drift_net: Learned f_α(x,t) network (optional)
            cost_net: Learned F_β(x,t) network (optional)
            base_drift_fn: Optional base drift (e.g., lambda x: -0.1*x)
            base_cost_fn: Optional base cost added to nonlocal cost (e.g., obstacle)
        """
        self.config = config
        self.device = config.device
        self._time_grid_cache: dict[tuple[str, int], tuple[Tensor, Tensor]] = {}

        # Create DeepGSB ToyY/ToyZ dual-stream networks if not provided.
        self.value_net = value_net or ToyValueNet(config).to(self.device)
        self.value_hat_net = value_hat_net or ToyValueNet(config).to(self.device)
        self.grad_net = grad_net or ToyGradientNet(config).to(self.device)
        self.grad_hat_net = grad_hat_net or ToyGradientNet(config).to(self.device)

        # Interaction terms (optional - can be None if not using MF interaction)
        self.drift_net = drift_net
        self.cost_net = cost_net
        self.base_drift_fn = base_drift_fn
        self.base_cost_fn = base_cost_fn
        self._exact_interaction_drift_eval_time_total = 0.0
        self._exact_interaction_cost_eval_time_total = 0.0
        self._exact_interaction_drift_call_count = 0.0
        self._exact_interaction_cost_call_count = 0.0
        self._exact_interaction_timing_enabled = True
        self._stochastic_seed_base: Optional[int] = None
        self._stochastic_seed_counters: dict[str, int] = {}

    # ========== Helper Functions ==========

    @staticmethod
    def _module_has_trainable_params(module: Optional[nn.Module]) -> bool:
        """Return whether the currently attached module has trainable parameters."""
        return module is not None and any(p.requires_grad for p in module.parameters())

    # ========== Core Methods ==========

    @staticmethod
    def _trajectory_step_seed(
        seed: Optional[int],
        *,
        phase_offset: int,
        step_idx: int,
        kind_offset: int,
    ) -> Optional[int]:
        """Derive a deterministic per-step seed for trajectory sampling."""
        return seed_offset(seed, phase_offset + step_idx * 100 + kind_offset)

    @contextmanager
    def stochastic_seed_scope(self, seed: Optional[int]) -> Iterator[None]:
        """Scope solver stochastic substreams under a stable base seed."""
        prev_base = self._stochastic_seed_base
        prev_counters = self._stochastic_seed_counters
        self._stochastic_seed_base = seed
        self._stochastic_seed_counters = {}
        try:
            yield
        finally:
            self._stochastic_seed_base = prev_base
            self._stochastic_seed_counters = prev_counters

    @contextmanager
    def suspend_exact_interaction_timing(self) -> Iterator[None]:
        """Temporarily disable exact interaction timing accumulation."""
        prev = self._exact_interaction_timing_enabled
        self._exact_interaction_timing_enabled = False
        try:
            yield
        finally:
            self._exact_interaction_timing_enabled = prev

    def _next_stochastic_seed(self, kind: str) -> Optional[int]:
        """Return a deterministic sub-seed for a stochastic component."""
        if self._stochastic_seed_base is None:
            return None
        kind_offsets = {
            "hutchinson_flat": 100_000,
            "hutchinson_time_batch": 200_000,
            "analytical_drift_eval": 300_000,
            "analytical_cost_eval": 400_000,
        }
        counter = self._stochastic_seed_counters.get(kind, 0)
        self._stochastic_seed_counters[kind] = counter + 1
        return seed_offset(self._stochastic_seed_base, kind_offsets[kind] + counter)

    def _evaluate_direct_exact_drift(
        self,
        x: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Evaluate a non-trainable drift module and accumulate exact timing."""
        if self.drift_net is None:
            return torch.zeros_like(x)
        if self._module_has_trainable_params(self.drift_net):
            return self.drift_net(x, t)
        eval_start = time.perf_counter()
        out = self.drift_net(x, t)
        if self._exact_interaction_timing_enabled:
            self._exact_interaction_drift_eval_time_total += (
                time.perf_counter() - eval_start
            )
            self._exact_interaction_drift_call_count += 1.0
        return out

    def compute_div_gz(
        self,
        xs: Tensor,
        ts: Tensor,
        z_net: nn.Module,
        direction: str = "forward",
        time_slice_batch_size: Optional[int] = None,
    ) -> Tensor:
        """Compute divergence ∇·(σZ - f) on a flat batch using Hutchinson's estimator.

        This method uses Hutchinson's stochastic trace estimator to efficiently
        compute the divergence term without explicit summation over dimensions:
            Tr(∇·drift) = E[ε^T ∇(drift) ε], where ε ~ N(0, I)

        Args:
            xs: States with shape ``(batch_size, dim)``. For time-batch callers,
                use ``_compute_divergence_hutchinson_time_batch`` instead.
            ts: Times with shape ``(batch_size,)``
            z_net: Gradient network (Z or Ẑ)
            direction: 'forward' or 'backward'

        Returns:
            div_gz: Divergence (batch_size,)

        References:
            Hutchinson, M.F. (1990). A stochastic estimator of the trace of
            the influence matrix for Laplacian smoothing splines.

        Optimization notes:
            - Optimized for n_hutchinson_samples=1 (no loop, no retain_graph)
            - Uses detach() instead of clone() for efficiency
            - Avoids unnecessary graph retention
        """
        # Ensure xs requires gradient
        # (use detach() instead of clone() for efficiency)
        if not xs.requires_grad:
            xs = xs.detach().requires_grad_(True)

        # batch_size = xs.shape[0]

        # Compute σZ
        z = z_net(xs, ts)  # (batch_size, dim)
        gz = self.config.sigma * z

        # Compute drift term: f = f_interaction + f_base
        # IMPORTANT: The sign of f depends on the direction parameter, which indicates
        # which policy is being updated (not which trajectory is being used).
        # Following DeepGSB's implementation:
        # - direction="forward": policy Y,Z being updated → use +f
        # - direction="backward": policy Ŷ,Ẑ being updated → use -f
        # This matches the SDE definitions:
        # - Forward SDE: dX = (f + σZ)dt + σdW
        # - Backward SDE: dX̄ = (-f + σẐ)ds + σdW
        f = torch.zeros_like(xs)

        # Add learned interaction term if available
        if self.drift_net is not None:
            if (
                self._module_has_trainable_params(self.drift_net)
                or time_slice_batch_size is None
            ):
                f = f + self._evaluate_direct_exact_drift(xs, ts)
            else:
                if (
                    time_slice_batch_size <= 0
                    or xs.shape[0] % time_slice_batch_size != 0
                ):
                    raise ValueError(
                        "time_slice_batch_size must be a positive divisor of "
                        f"batch size {xs.shape[0]}, got {time_slice_batch_size}"
                    )
                xs_steps = xs.reshape(-1, time_slice_batch_size, xs.shape[-1])
                ts_steps = ts.reshape(-1, time_slice_batch_size)
                f_steps = self._evaluate_analytical_drift_time_local(xs_steps, ts_steps)
                f = f + f_steps.reshape_as(xs)

        # Add base drift if provided
        if self.base_drift_fn is not None:
            f = f + self.base_drift_fn(xs)

        # Apply direction-dependent sign to base drift
        sign = 1.0 if direction == "forward" else -1.0
        f = sign * f

        # Compute drift = σZ - f
        drift = gz - f

        # Hutchinson's estimator with optimized paths for common cases
        n_samples = self.config.n_hutchinson_samples

        if n_samples == 1:
            # Optimized path for single sample (most common case)
            # No loop, no retain_graph needed
            with torch_seed_context(self._next_stochastic_seed("hutchinson_flat")):
                epsilon = torch.randn_like(xs)
            grad_drift = torch.autograd.grad(
                outputs=drift,
                inputs=xs,
                grad_outputs=epsilon,
                create_graph=True,
                # No retain_graph needed for single sample
            )[0]
            div_gz = (grad_drift * epsilon).sum(dim=-1)
        else:
            # Multiple samples path (variance reduction)
            # E[ε^T ∇(drift) ε] where ε ~ N(0, I)
            div_estimates = []
            for i in range(n_samples):
                with torch_seed_context(self._next_stochastic_seed("hutchinson_flat")):
                    epsilon = torch.randn_like(xs)
                grad_drift = torch.autograd.grad(
                    outputs=drift,
                    inputs=xs,
                    grad_outputs=epsilon,
                    create_graph=True,
                    # Only retain for non-final samples
                    retain_graph=(i < n_samples - 1),
                )[0]
                div_estimates.append((grad_drift * epsilon).sum(dim=-1))

            # Average over multiple samples
            div_gz = torch.stack(div_estimates).mean(dim=0)

        return div_gz

    def sample_forward_trajectory(
        self,
        x0: Tensor,
        seed: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Sample forward SDE trajectory: dX = (f + σZ) dt + σ dW.

        Args:
            x0: Initial states (batch_size, dim)
            seed: Random seed for reproducibility

        Returns:
            xs: Trajectory (n_steps+1, batch_size, dim)
            ts: Time grid (n_steps+1,)
            dws: Noise increments (n_steps, batch_size, dim)
        """
        batch_size = x0.shape[0]
        n_steps = self.config.n_time_steps
        dim = self.config.dim

        # Initialize trajectory
        xs = torch.zeros(n_steps + 1, batch_size, dim, device=self.device)
        xs[0] = x0.to(self.device)

        # Time grid
        ts = self.config.time_grid

        # Noise increments
        sqrt_dt = torch.sqrt(torch.tensor(self.config.dt, device=self.device))
        dws = []

        # Euler-Maruyama integration
        with torch.no_grad():
            for i in range(n_steps):
                t = ts[i].expand(batch_size)
                x_current = xs[i]

                # Compute drift: f + σZ
                z = self.grad_net(x_current, t)
                drift = self.config.sigma * z

                # Add interaction term
                if self.drift_net is not None:
                    interaction_seed = self._trajectory_step_seed(
                        seed,
                        phase_offset=10_000,
                        step_idx=i,
                        kind_offset=1,
                    )
                    with torch_seed_context(interaction_seed):
                        drift = drift + self._evaluate_direct_exact_drift(x_current, t)

                # Add base drift
                if self.base_drift_fn is not None:
                    drift = drift + self.base_drift_fn(x_current)

                # Generate noise
                noise_seed = self._trajectory_step_seed(
                    seed,
                    phase_offset=10_000,
                    step_idx=i,
                    kind_offset=2,
                )
                with torch_seed_context(noise_seed):
                    dw = torch.randn_like(x_current) * sqrt_dt
                dws.append(dw)

                # Euler step
                xs[i + 1] = x_current + drift * self.config.dt + self.config.sigma * dw

        dws_tensor = torch.stack(dws, dim=0)  # (n_steps, batch_size, dim)

        return xs, ts, dws_tensor

    def sample_backward_trajectory(
        self,
        x1: Tensor,
        seed: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Sample backward SDE trajectory: dX̄ = (-f + σẐ) ds + σ dW.

        The backward SDE evolves in time-reversed coordinate s = T - t,
        but we store the trajectory in PHYSICAL TIME order for consistency.

        Args:
            x1: Terminal states at t=T (batch_size, dim)
            seed: Random seed for reproducibility

        Returns:
            xs: Trajectory (n_steps+1, batch_size, dim) in PHYSICAL TIME order
                xs[0] at t=0, xs[1] at t=dt, ..., xs[n_steps] at t=T
            ts: Time grid (n_steps+1,) FLIPPED: [T, T-dt, ..., 0]
                For iteration convenience (matches DeepGSB convention)
            dws: Noise increments (n_steps, batch_size, dim) in PHYSICAL TIME order
                dws[i] used for transition from xs[i+1] to xs[i]

        Example:
            With n_steps=3, dt=0.1, T=0.3:
            - ts = [0.3, 0.2, 0.1, 0.0] (flipped for iteration)
            - xs[3] at t=0.3 (starting point, = x1)
            - xs[2] at t=0.2
            - xs[1] at t=0.1
            - xs[0] at t=0.0 (ending point)
            - dws[2] used for xs[3]→xs[2] transition
            - dws[1] used for xs[2]→xs[1] transition
            - dws[0] used for xs[1]→xs[0] transition
        """
        batch_size = x1.shape[0]
        n_steps = self.config.n_time_steps
        dim = self.config.dim

        # Initialize arrays in PHYSICAL TIME order
        xs = torch.zeros(n_steps + 1, batch_size, dim, device=self.device)
        dws = torch.zeros(n_steps, batch_size, dim, device=self.device)

        # Flip time grid for iteration (DeepGSB convention)
        ts_backward = torch.flip(self.config.time_grid, dims=[0])
        # Result: ts_backward = [T, T-dt, ..., 0]

        # Initialize at terminal condition
        x_current = x1.to(self.device)

        sqrt_dt = torch.sqrt(torch.tensor(self.config.dt, device=self.device))

        # Euler-Maruyama integration with PHYSICAL TIME indexing
        # We iterate forward in s-coordinate (backward in physical time t)
        with torch.no_grad():
            for iter_idx in range(n_steps + 1):
                # Map iteration index to PHYSICAL TIME INDEX
                # iter_idx=0 → phys_idx=n_steps (t=T, end of array)
                # iter_idx=1 → phys_idx=n_steps-1 (t=T-dt)
                # ...
                # iter_idx=n_steps → phys_idx=0 (t=0, start of array)
                phys_idx = n_steps - iter_idx

                # Store current state at PHYSICAL TIME INDEX
                xs[phys_idx] = x_current

                # If not at final iteration, compute next state
                if iter_idx < n_steps:
                    t_current = ts_backward[iter_idx]
                    t_current_expanded = t_current.expand(batch_size)

                    # Compute drift: -f + σẐ
                    z_hat = self.grad_hat_net(x_current, t_current_expanded)
                    drift = self.config.sigma * z_hat

                    # Subtract interaction term (backward SDE has -f)
                    if self.drift_net is not None:
                        interaction_seed = self._trajectory_step_seed(
                            seed,
                            phase_offset=20_000,
                            step_idx=iter_idx,
                            kind_offset=1,
                        )
                        with torch_seed_context(interaction_seed):
                            drift = drift - self._evaluate_direct_exact_drift(
                                x_current, t_current_expanded
                            )

                    # Subtract base drift
                    if self.base_drift_fn is not None:
                        drift = drift - self.base_drift_fn(x_current)

                    # Generate noise
                    noise_seed = self._trajectory_step_seed(
                        seed,
                        phase_offset=20_000,
                        step_idx=iter_idx,
                        kind_offset=2,
                    )
                    with torch_seed_context(noise_seed):
                        dw = torch.randn_like(x_current) * sqrt_dt

                    # CRITICAL: Store noise at PHYSICAL TIME index
                    # Transition FROM phys_idx TO phys_idx-1 (time decreasing)
                    # dws array has n_steps elements (indices 0 to n_steps-1)
                    # phys_idx ranges from n_steps down to 1 during this loop
                    # So dws index is phys_idx - 1
                    dws[phys_idx - 1] = dw

                    # Euler step: update to next state (earlier in physical time)
                    x_current = (
                        x_current + drift * self.config.dt + self.config.sigma * dw
                    )

        # Return with physical time ordering
        # xs[i] corresponds to physical time t = i * dt
        # ts_backward for iteration convenience (flipped)
        # dws[i] used for transition xs[i+1] → xs[i]
        return xs, ts_backward, dws

    def compute_ipf_loss(
        self,
        xs_traj: Tensor,
        ts_traj: Tensor,
        log_p0_fn: Optional[Callable[[Tensor], Tensor]] = None,
        log_p1_fn: Optional[Callable[[Tensor], Tensor]] = None,
        direction: str = "forward",
    ) -> Tensor:
        """Compute L_IPF (Iterative Proportional Fitting) loss.

        L_IPF = ∫₀ᵀ E[Z·(0.5Z + Ẑ) + ∇·(σZ - f)] dt + boundary terms

        Args:
            xs_traj: Trajectory from the complementary policy
                - If direction="forward": backward trajectory (for updating Y,Z)
                  Physical indexing: xs[0] at t=0, xs[n_steps] at t=T
                - If direction="backward": forward trajectory (for updating Ŷ,Ẑ)
                  Standard indexing: xs[0] at t=0, xs[-1] at t=T
            ts_traj: Time grid corresponding to xs_traj
            direction: Which networks to update ("forward" or "backward")

        Boundary terms (CORRECTED for physical time indexing):
            - direction="forward" (using backward traj):
                + log_p0(xs_traj[0]) where xs_traj[0] is at t=0
                - log_p1(xs_traj[-1]) where xs_traj[-1]=xs_traj[n_steps] is at t=T
            - direction="backward" (using forward traj):
                + log_p0(xs_traj[0]) where xs_traj[0] is at t=0
                - log_p1(xs_traj[-1]) where xs_traj[-1] is at t=T

        Uses the time-batch representation directly.
        """
        _, grad_primary, grad_aux = self._select_value_grad_nets(direction)
        n_steps = len(ts_traj) - 1
        batch_size = xs_traj.shape[1]

        # Always reason in physical time order: xs_traj[i] is the state at i * dt.
        xs = xs_traj[:-1].detach().requires_grad_(True)  # (n_steps, batch, dim)
        ts_steps = self._get_time_grid_steps(batch_size)

        # Compute primary and auxiliary gradients for all time steps
        z_primary = self._eval_net_time_batch(grad_primary, xs, ts_steps)
        z_aux = self._eval_net_time_batch(grad_aux, xs, ts_steps).detach()

        # KL term: 0.5 * ‖Z_primary + Z_aux‖²
        # This matches Liu et al. (2022) Proposition 9: 0.5‖Ẑ + Z‖²
        kl_term = 0.5 * ((z_primary + z_aux) ** 2).sum(dim=-1)

        # Divergence term ∇·(σZ - f), keeping the current direction-dependent
        # sign convention but evaluating it on the time-batch representation.
        drift = self.config.sigma * z_primary
        f = torch.zeros_like(z_primary)
        if self.drift_net is not None:
            if self._module_has_trainable_params(self.drift_net):
                f_learned = self._eval_net_time_batch(self.drift_net, xs, ts_steps)
                if self.config.detach_interaction_drift_divergence:
                    f_learned = f_learned.detach()
                f = f + f_learned
            else:
                f = f + self._evaluate_analytical_drift_time_local(xs, ts_steps)
        if self.base_drift_fn is not None:
            f = f + self._eval_base_drift_time_batch(xs)

        sign = 1.0 if direction == "forward" else -1.0
        drift = drift - sign * f
        div_gz = self._compute_divergence_hutchinson_time_batch(drift, xs).sum(dim=-1)

        # Average over all (time, batch) points, then scale by total time
        integrand_mean = (kl_term + div_gz).mean()
        loss = integrand_mean * (self.config.dt * n_steps)

        # Boundary terms follow DeepGSB's get_init_term_bound pattern:
        # - init boundary: starting point of the TRAJECTORY being sampled
        # - term boundary: ending point of the TRAJECTORY being sampled
        # Formula: +log p_init - log p_term (where p_init/p_term match trajectory endpoints)

        if direction == "forward":
            # Forward networks (Y, Z) update using backward trajectory
            # Backward trajectory: starts at t=T (init), ends at t=0 (term)
            # init = xs_traj[-1] at t=T → p_init = ρ_1
            # term = xs_traj[0] at t=0 → p_term = ρ_0
            if log_p1_fn is not None:
                loss += log_p1_fn(xs_traj[-1]).mean()  # + log ρ_1(X̄ at t=T) [init]
            if log_p0_fn is not None:
                loss -= log_p0_fn(xs_traj[0]).mean()  # - log ρ_0(X̄ at t=0) [term]
        else:  # backward
            # Backward networks (Ŷ, Ẑ) update using forward trajectory
            # Forward trajectory: starts at t=0 (init), ends at t=T (term)
            # init = xs_traj[0] at t=0 → p_init = ρ_0
            # term = xs_traj[-1] at t=T → p_term = ρ_1
            if log_p0_fn is not None:
                loss += log_p0_fn(xs_traj[0]).mean()  # + log ρ_0(X at t=0) [init]
            if log_p1_fn is not None:
                loss -= log_p1_fn(xs_traj[-1]).mean()  # - log ρ_1(X at t=T) [term]

        return loss

    def _select_value_grad_nets(
        self, direction: str
    ) -> tuple[nn.Module, nn.Module, nn.Module]:
        """Return (value_net, grad_net, aux_grad_net) for a given direction.

        Args:
            direction: 'forward' or 'backward'

        Returns:
            Tuple of (primary value net, primary grad net, auxiliary grad net)

        Raises:
            ValueError: If direction is not recognised.
        """
        if direction == "forward":
            return self.value_net, self.grad_net, self.grad_hat_net
        if direction == "backward":
            return self.value_hat_net, self.grad_hat_net, self.grad_net
        raise ValueError(
            f"Unknown direction '{direction}'. Expected 'forward' or 'backward'."
        )

    def compute_fk_loss(
        self,
        xs_backward: Tensor,
        ts_backward: Tensor,
        direction: str = "forward",
    ) -> Tensor:
        """Compute L_FK (Fokker-Planck / Gradient Matching) loss.

        L_FK = E[‖σ∇Y(X̄_s, s) - Z(X̄_s, s)‖²]

        Uses the time-batch representation directly.
        """
        value_net, grad_net, _ = self._select_value_grad_nets(direction)
        xs = xs_backward[:-1].detach().requires_grad_(True)  # (n_steps, batch, dim)
        ts_steps = self._get_time_grid_steps(xs_backward.shape[1])

        # Compute value and its spatial gradient on the time-batch representation.
        y = self._eval_net_time_batch(value_net, xs, ts_steps)
        grad_y = torch.autograd.grad(outputs=y.sum(), inputs=xs, create_graph=True)[0]

        # Label: σ∇Y (detached)
        label = self.config.sigma * grad_y

        # Predicted gradient from Z network
        predict = self._eval_net_time_batch(grad_net, xs.detach(), ts_steps)

        # DeepGSB-style normalization: 0.5 * (error^p * dt).sum() / batch_x
        batch_x = xs_backward.shape[1]  # batch size
        if self.config.fk_loss_norm == "l1":
            loss = 0.5 * ((predict - label).abs() * self.config.dt).sum() / batch_x
        elif self.config.fk_loss_norm == "l2":
            loss = 0.5 * ((predict - label) ** 2 * self.config.dt).sum() / batch_x
        elif self.config.fk_loss_norm == "huber":
            loss = (
                torch.nn.functional.huber_loss(
                    predict, label, reduction="sum", delta=1.0
                )
                * self.config.dt
                / batch_x
            )
        else:
            raise ValueError(f"Unknown fk_loss_norm: {self.config.fk_loss_norm}")

        return loss

    def compute_td_loss(
        self,
        xs_traj: Tensor,
        ts_traj: Tensor,
        dws_traj: Tensor,
        direction: str = "forward",
        target_value_net: Optional[nn.Module] = None,
        target_aux_grad_net: Optional[nn.Module] = None,
        target_value_hat_net: Optional[nn.Module] = None,
        target_grad_hat_net: Optional[nn.Module] = None,
        log_p0_fn: Optional[Callable[[Tensor], Tensor]] = None,
        log_p1_fn: Optional[Callable[[Tensor], Tensor]] = None,
        zs_aux_override: Optional[Tensor] = None,
        return_aux: bool = False,
        trajectory_direction: Optional[str] = None,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """Compute TD loss (single or multi-step) following DeepGSB implementation.

        When ``return_aux`` is True, the function also returns the auxiliary gradient
        (Z from the opposite policy). When ``zs_aux_override`` is provided, it is
        used instead of evaluating the auxiliary gradient network.

        Algorithm:
            1. Work directly on trajectories shaped ``(n_steps, batch, dim)``
            2. Compute KL term: ``kl = z * (0.5*z + z_aux) + div_gz`` [per dimension]
            3. Compute values using current and EMA networks
            4. Compute running cost ``F(x, t)``
            5. Compute 1-step increment: ``δY = (kl.sum() - F) * dt + Z^T dW``
            6. Transpose to ``(batch, n_steps)`` for DeepGSB-style targets
            7. Compute TD loss:
               - Forward: Y_{t+1} vs Y_t^{target} + δY_t
               - Backward: Ŷ_{t-1} vs Ŷ_t^{target} + δŶ_t

        Key features:
            - Vectorized: processes all time steps at once
            - Uses EMA for 1-step target for stability
            - No explicit boundary bootstrapping (handled by slicing)
            - DeepGSB compatible implementation

        Args:
            xs_traj: Trajectory states (n_steps+1, batch_size, dim)
                     ALWAYS in physical time order: xs[0] at t=0, xs[-1] at t=T
            ts_traj: Time grid (n_steps+1,)
                     For backward: flipped [T, T-dt, ..., 0] (iteration convenience)
                     For forward: normal [0, dt, ..., T]
            dws_traj: Noise increments (n_steps, batch_size, dim)
            direction: "forward" or "backward" - which policy to train
            target_value_net: EMA value network for current direction
            target_aux_grad_net: EMA grad network for opposite direction
            target_value_hat_net: EMA value_hat network (opposite of value_net)
            target_grad_hat_net: EMA grad_hat network (for opposite direction)
            log_p0_fn, log_p1_fn: Boundary densities (required for multi-step TD)
            zs_aux_override: Optional precomputed auxiliary gradients with shape
                ``(n_steps, batch_size, dim)``
            return_aux: Whether to return auxiliary gradients

        Returns:
            TD loss scalar, and optionally auxiliary gradients with shape
            ``(n_steps, batch_size, dim)`` if ``return_aux`` is True.

        References:
            DeepGSB loss_lib.py compute_bsde_td_loss_{singlestep,multistep}
        """
        if trajectory_direction is None:
            trajectory_direction = "backward" if direction == "forward" else "forward"

        # Select networks based on direction
        value_net, grad_net, aux_grad_net = self._select_value_grad_nets(direction)

        # Default to current networks if no targets provided
        if target_value_net is None:
            target_value_net = value_net
        if target_aux_grad_net is None:
            target_aux_grad_net = aux_grad_net

        # Get opposite direction networks for value computation
        if direction == "forward":
            target_value_opposite_net = target_value_hat_net or self.value_hat_net
        else:
            target_value_opposite_net = target_value_hat_net or self.value_net

        # Prepare data: remove last state, keep only transitions
        # xs_traj: (n_steps+1, batch, dim) → (n_steps, batch, dim)
        n_steps = len(ts_traj) - 1
        batch_size = xs_traj.shape[1]
        dim = xs_traj.shape[2]
        use_multistep = self.config.use_multistep_td_loss

        xs = xs_traj[:-1]  # (n_steps, batch, dim) - states at t=0, dt, ..., T-dt

        # Time grid: use PHYSICAL time, not ts_traj (which may be flipped)
        # We need times corresponding to xs[i] = state at physical time i*dt
        ts_steps = self._get_time_grid_steps(batch_size)

        # === Compute KL term ===
        # We need to compute: kl = z * (0.5*z + z_aux) + div_gz
        # where div_gz is per-dimension divergence

        # Enable gradients for divergence computation
        xs_grad = xs.detach().requires_grad_(True)

        # Compute Z (primary gradient)
        z_primary = self._eval_net_time_batch(grad_net, xs_grad, ts_steps)

        # Compute Ẑ (auxiliary gradient) - detached
        if zs_aux_override is not None:
            if zs_aux_override.shape != z_primary.shape:
                raise ValueError(
                    f"zs_aux_override shape {zs_aux_override.shape} does not match "
                    f"expected time-batch shape {tuple(z_primary.shape)}"
                )
            z_aux = zs_aux_override.detach()
        else:
            z_aux = self._eval_net_time_batch(aux_grad_net, xs_grad, ts_steps).detach()

        # Compute drift = σZ - f
        drift = self.config.sigma * z_primary

        # Subtract drift term f with direction-dependent sign
        f = torch.zeros_like(z_primary)
        if self.drift_net is not None:
            if self._module_has_trainable_params(self.drift_net):
                f_learned = self._eval_net_time_batch(self.drift_net, xs_grad, ts_steps)
                if self.config.detach_interaction_drift_divergence:
                    f_learned = f_learned.detach()
                f = f + f_learned
            else:
                f_steps = self._evaluate_analytical_drift_time_local(xs, ts_steps)
                f = f + f_steps
        if self.base_drift_fn is not None:
            f = f + self._eval_base_drift_time_batch(xs_grad)

        sign = 1.0 if direction == "forward" else -1.0
        drift = drift - sign * f

        # Compute divergence per dimension using Hutchinson estimator
        div_gz = self._compute_divergence_hutchinson_time_batch(drift, xs_grad)

        # KL term: kl = z * (0.5*z + z_aux) + div_gz  [per dimension]
        # This matches DeepGSB line 84
        kl = z_primary * (0.5 * z_primary + z_aux) + div_gz  # (n_steps, batch, dim)

        # === Compute Values ===
        # Current value (for prediction)
        value = self._eval_net_time_batch(value_net, xs, ts_steps).squeeze(-1)

        # Target value (EMA, for 1-step target)
        with torch.no_grad():
            value_target = self._eval_net_time_batch(
                target_value_net, xs, ts_steps
            ).squeeze(-1)
            value_opposite_target = self._eval_net_time_batch(
                target_value_opposite_net, xs, ts_steps
            ).squeeze(-1)

        # === Compute Running Cost ===
        # log_p = Y^{target} + Ŷ^{target} (estimate of log density)
        # Running cost F(x, t)
        # IMPORTANT:
        # - Trainable learned cost net: evaluate on flattened batch (O(T*B))
        # - Analytical non-trainable cost: evaluate per-time-slice to avoid
        #   cross-time interactions and accidental O((T*B)^2) complexity.
        F = xs.new_zeros((n_steps, batch_size))
        if self.cost_net is not None:
            if self._module_has_trainable_params(self.cost_net):
                F = F + self._eval_net_time_batch(self.cost_net, xs, ts_steps).squeeze(-1)
            else:
                use_cpu = bool(self.config.evaluate_interaction_cpu)
                F_steps = self._evaluate_analytical_cost_time_local(
                    xs,
                    ts_steps,
                    use_cpu=use_cpu,
                )
                F = F + F_steps
        if self.base_cost_fn is not None:
            F = F + self._eval_base_cost_time_batch(xs, ts_steps)

        # === Compute 1-Step Increment ===
        # δY = (kl.sum(dim=-1) - F) * dt + Z^T dW
        # This matches DeepGSB line 237
        bsde_1step = (kl.sum(dim=-1) - F) * self.config.dt + (z_primary * dws_traj).sum(
            dim=-1
        )
        # Shape: (n_steps, batch)

        # Transpose to (batch, n_steps) to match DeepGSB convention
        value_predict = value.transpose(0, 1)  # (batch, n_steps)
        bsde_1step = bsde_1step.transpose(0, 1)  # (batch, n_steps)
        value_target = value_target.transpose(0, 1)  # (batch, n_steps)
        value_opposite_target = value_opposite_target.transpose(
            0, 1
        )  # (batch, n_steps)

        # === Compute TD Target and Loss ===
        if use_multistep:
            if trajectory_direction == "forward":
                if log_p0_fn is None:
                    raise ValueError(
                        "log_p0_fn is required for multi-step TD (forward)."
                    )
                boundary_x = xs_traj[0]  # t = 0
                log_p_boundary = log_p0_fn(boundary_x).detach()
                target_bnd = (log_p_boundary - value_opposite_target[:, 0]).detach()
                bsde_cum = torch.cumsum(bsde_1step, dim=1)
                label = (target_bnd[:, None] + bsde_cum)[:, :-1].detach()
                predict = value_predict[:, 1:]
            else:
                if log_p1_fn is None:
                    raise ValueError(
                        "log_p1_fn is required for multi-step TD (backward)."
                    )
                boundary_x = xs_traj[-1]  # t = T
                log_p_boundary = log_p1_fn(boundary_x).detach()
                target_bnd = (log_p_boundary - value_opposite_target[:, -1]).detach()
                bsde_flip = torch.flip(bsde_1step, dims=[1])
                bsde_cum = torch.cumsum(bsde_flip, dim=1)
                bsde_cum = torch.flip(bsde_cum, dims=[1])
                label = (target_bnd[:, None] + bsde_cum)[:, 1:].detach()
                predict = value_predict[:, :-1]
        else:
            if trajectory_direction == "forward":
                label = (value_target + bsde_1step)[:, :-1].detach()
                predict = value_predict[:, 1:]
            else:
                label = (value_target + bsde_1step)[:, 1:].detach()
                predict = value_predict[:, :-1]

        # L2 loss: 0.5 * Σ (predict - label)² * dt / batch_size
        loss = 0.5 * ((predict - label) ** 2 * self.config.dt).sum() / batch_size

        if return_aux:
            return loss, z_aux.detach()
        return loss

    def zero_grad(self) -> None:
        """Zero gradients of all learnable networks."""
        self.value_net.zero_grad()
        self.value_hat_net.zero_grad()
        self.grad_net.zero_grad()
        self.grad_hat_net.zero_grad()
        if self.drift_net is not None:
            self.drift_net.zero_grad()
        if self.cost_net is not None:
            self.cost_net.zero_grad()

    def _evaluate_analytical_drift_time_local(
        self,
        xs_steps: Tensor,
        ts_steps: Tensor,
    ) -> Tensor:
        """Evaluate an analytical drift exactly without cross-time mixing."""
        eval_start = time.perf_counter()
        with torch_seed_context(self._next_stochastic_seed("analytical_drift_eval")):
            if hasattr(self.drift_net, "evaluate_batched"):
                kwargs = {
                    "time_block_size": getattr(
                        self.config, "interaction_time_block_size", None
                    )
                }
                if getattr(self.drift_net, "requires_time", False):
                    kwargs["time_grid"] = ts_steps
                result = self.drift_net.evaluate_batched(xs_steps, xs_steps, **kwargs)
            else:
                base_drift = getattr(self.drift_net, "drift", None)
                time_block_size = getattr(
                    self.config, "interaction_time_block_size", None
                )
                if base_drift is not None and hasattr(base_drift, "evaluate_batched"):
                    kwargs = {"time_block_size": time_block_size}
                    if getattr(base_drift, "requires_time", False):
                        kwargs["time_grid"] = ts_steps
                    result = base_drift.evaluate_batched(xs_steps, xs_steps, **kwargs)
                else:
                    n_steps, batch_size, dim = xs_steps.shape
                    f_steps = torch.zeros(n_steps, batch_size, dim, device=self.device)
                    for i in range(n_steps):
                        f_steps[i] = self.drift_net(xs_steps[i], ts_steps[i])
                    result = f_steps
        if self._exact_interaction_timing_enabled:
            self._exact_interaction_drift_eval_time_total += (
                time.perf_counter() - eval_start
            )
            self._exact_interaction_drift_call_count += float(xs_steps.shape[0])
        return result

    def _get_time_grid_steps(self, batch_size: int) -> Tensor:
        """Return cached physical time grid with shape (time, batch)."""
        device_key = str(self.device)
        cache_key = (device_key, batch_size)
        cached = self._time_grid_cache.get(cache_key)
        if cached is not None:
            return cached
        ts_steps = (
            self.config.time_grid[:-1]
            .to(self.device)
            .unsqueeze(1)
            .expand(-1, batch_size)
        )
        self._time_grid_cache[cache_key] = ts_steps
        return ts_steps

    @staticmethod
    def _flatten_time_batch(x: Tensor) -> Tensor:
        """Flatten leading `(time, batch)` dimensions into a single batch."""
        return tensor_utils.flatten_dim01(x)

    @staticmethod
    def _unflatten_time_batch(x: Tensor, n_steps: int, batch_size: int) -> Tensor:
        """Restore flattened `(time * batch, ...)` tensors to `(time, batch, ...)`."""
        return tensor_utils.unflatten_dim01(x, [n_steps, batch_size])

    def _eval_net_time_batch(
        self,
        net: nn.Module,
        xs_steps: Tensor,
        ts_steps: Tensor,
    ) -> Tensor:
        """Evaluate a network on `(time, batch, ...)` inputs and preserve layout."""
        n_steps, batch_size = xs_steps.shape[:2]
        out = net(
            self._flatten_time_batch(xs_steps),
            self._flatten_time_batch(ts_steps),
        )
        return self._unflatten_time_batch(out, n_steps, batch_size)

    def _eval_base_drift_time_batch(
        self,
        xs_steps: Tensor,
    ) -> Tensor:
        """Evaluate optional base drift and preserve `(time, batch, dim)` layout."""
        if self.base_drift_fn is None:
            return torch.zeros_like(xs_steps)
        n_steps, batch_size, _ = xs_steps.shape
        out = self.base_drift_fn(self._flatten_time_batch(xs_steps))
        return self._unflatten_time_batch(out, n_steps, batch_size)

    def _eval_base_cost_time_batch(
        self,
        xs_steps: Tensor,
        ts_steps: Tensor,
    ) -> Tensor:
        """Evaluate optional base cost and preserve `(time, batch)` layout."""
        if self.base_cost_fn is None:
            return xs_steps.new_zeros(xs_steps.shape[:2])
        n_steps, batch_size, _ = xs_steps.shape
        out = self.base_cost_fn(
            self._flatten_time_batch(xs_steps),
            self._flatten_time_batch(ts_steps),
        )
        if out.dim() == 2:
            out = out.squeeze(-1)
        return self._unflatten_time_batch(out, n_steps, batch_size)

    def _compute_divergence_hutchinson_time_batch(
        self,
        drift_steps: Tensor,
        xs_steps_grad: Tensor,
    ) -> Tensor:
        """Compute per-dimension divergence for `(time, batch, dim)` drift values."""
        with torch_seed_context(self._next_stochastic_seed("hutchinson_time_batch")):
            epsilon = torch.randn_like(xs_steps_grad)
        grad_drift = torch.autograd.grad(
            outputs=drift_steps,
            inputs=xs_steps_grad,
            grad_outputs=epsilon,
            create_graph=True,
        )[0]
        return grad_drift * epsilon

    def _evaluate_analytical_cost_time_local(
        self,
        xs_steps: Tensor,
        ts_steps: Tensor,
        *,
        use_cpu: bool,
    ) -> Tensor:
        """Evaluate an analytical cost exactly without cross-time mixing."""
        eval_start = time.perf_counter()
        with torch_seed_context(self._next_stochastic_seed("analytical_cost_eval")):
            if hasattr(self.cost_net, "evaluate_batched") and not use_cpu:
                result = self.cost_net.evaluate_batched(
                    xs_steps,
                    xs_steps,
                    time_block_size=getattr(
                        self.config, "interaction_time_block_size", None
                    ),
                ).squeeze(-1)
            else:
                base_cost = getattr(self.cost_net, "cost", None)
                time_block_size = getattr(
                    self.config, "interaction_time_block_size", None
                )
                if base_cost is not None and hasattr(base_cost, "evaluate_batched"):
                    xs_eval = xs_steps.detach().to("cpu") if use_cpu else xs_steps
                    result = base_cost.evaluate_batched(
                        xs_eval,
                        xs_eval,
                        time_block_size=time_block_size,
                    ).squeeze(-1)
                    result = result.to(self.device) if use_cpu else result
                else:
                    n_steps, batch_size, _ = xs_steps.shape
                    f_steps = torch.zeros(n_steps, batch_size, device=self.device)
                    cost_module = self.cost_net
                    if use_cpu:
                        cost_module = cost_module.to("cpu")

                    with torch.no_grad():
                        for i in range(n_steps):
                            x_step = xs_steps[i]
                            t_step = ts_steps[i]
                            if use_cpu:
                                f_steps[i] = cost_module(
                                    x_step.detach().to("cpu"),
                                    t_step.detach().to("cpu"),
                                ).squeeze(-1).to(self.device)
                            else:
                                f_steps[i] = cost_module(x_step, t_step).squeeze(-1)

                    if use_cpu:
                        cost_module.to(self.device)
                    result = f_steps
        if self._exact_interaction_timing_enabled:
            self._exact_interaction_cost_eval_time_total += (
                time.perf_counter() - eval_start
            )
            self._exact_interaction_cost_call_count += float(xs_steps.shape[0])
        return result

    def timing_stats(self) -> dict[str, float]:
        """Return solver-side timing stats for exact analytical interaction evaluation."""
        return {
            "exact_drift_eval_time_total": self._exact_interaction_drift_eval_time_total,
            "exact_cost_eval_time_total": self._exact_interaction_cost_eval_time_total,
            "exact_drift_call_count": self._exact_interaction_drift_call_count,
            "exact_cost_call_count": self._exact_interaction_cost_call_count,
        }


__all__ = ["FBSDESolver"]
