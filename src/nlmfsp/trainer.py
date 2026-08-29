"""Trainer for Mean-Field Schrödinger Bridge with 4-step alternating optimization.

This module implements the training loop that coordinates FBSDE learning and
interaction learning in an alternating fashion.

References:
    Liu, G.-H., et al. (2022). Deep Generalized Schrödinger Bridge. NeurIPS.
"""

from __future__ import annotations

import copy
import logging
import time
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch_ema import ExponentialMovingAverage
except Exception:  # pragma: no cover - optional dependency
    ExponentialMovingAverage = None  # type: ignore

from .config import Config
from .fbsde import FBSDESolver
from .file_utils import (create_trainer_logger, load_trainer_checkpoint,
                         save_trainer_checkpoint)
from .interaction import InteractionLearner
from .rng_utils import seed_offset, torch_seed_context
from .validation import evaluate_losses_with_analytical

Tensor = torch.Tensor


def _build_param_groups(
    value_params: list[nn.Parameter],
    grad_params: list[nn.Parameter],
    *,
    lr_value: float,
    lr_grad: float,
) -> list[dict[str, Any]]:
    """Create optimizer param groups with separate learning rates."""
    groups: list[dict[str, Any]] = []
    if value_params:
        groups.append({"params": value_params, "lr": lr_value})
    if grad_params:
        groups.append({"params": grad_params, "lr": lr_grad})
    if not groups:
        raise ValueError("No trainable parameters found for optimizer setup.")
    return groups


def _create_optimizer_from_config(
    config: Config,
    param_groups: list[dict[str, Any]],
) -> optim.Optimizer:
    """Instantiate optimizer based on Config fields."""
    opt_type = config.optimizer_type.lower()
    base_lr = param_groups[0].get("lr")
    if base_lr is None:
        raise ValueError("Optimizer param group missing learning rate.")
    weight_decay = config.optimizer_weight_decay
    momentum = config.optimizer_momentum

    if opt_type == "adam":
        return optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)
    if opt_type == "adamw":
        return optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)
    if opt_type == "sgd":
        mom = momentum if momentum is not None else 0.0
        return optim.SGD(param_groups, lr=base_lr, momentum=mom, weight_decay=weight_decay)
    if opt_type == "rmsprop":
        mom = momentum if momentum is not None else 0.0
        return optim.RMSprop(
            param_groups, lr=base_lr, momentum=mom, weight_decay=weight_decay
        )
    if opt_type == "adagrad":
        return optim.Adagrad(param_groups, lr=base_lr, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer_type: {config.optimizer_type}")


def _create_scheduler_from_config(
    config: Config,
    optimizer: optim.Optimizer,
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Create LR scheduler matching Config."""
    sched_type = config.lr_scheduler_type.lower()
    if sched_type == "none":
        return None
    if sched_type == "exponential":
        return optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config.lr_decay_gamma, last_epoch=-1
        )
    if sched_type == "step":
        step_size = config.lr_decay_step or 1
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=config.lr_decay_gamma
        )
    raise ValueError(f"Unsupported lr_scheduler_type: {config.lr_scheduler_type}")


def setup_optimizers_and_schedulers(
    *,
    config: Config,
    solver: Any,
    learn_interaction_enabled: bool,
) -> tuple[
    optim.Optimizer,
    optim.Optimizer,
    Optional[optim.Optimizer],
    Optional[optim.lr_scheduler._LRScheduler],
    Optional[optim.lr_scheduler._LRScheduler],
    Optional[optim.lr_scheduler._LRScheduler],
]:
    """Build forward/backward/interaction optimizers and schedulers from Config."""
    if config.learning_rate_value is None or config.learning_rate_grad is None:
        raise ValueError(
            "learning_rate_value and learning_rate_grad must be specified in Config"
        )
    if learn_interaction_enabled and config.learning_rate_interaction is None:
        raise ValueError(
            "learning_rate_interaction must be specified when interaction learning is enabled"
        )

    lr_value = config.learning_rate_value
    lr_grad = config.learning_rate_grad

    forward_value_params = [p for p in solver.value_net.parameters() if p.requires_grad]
    forward_grad_params = [p for p in solver.grad_net.parameters() if p.requires_grad]
    forward_groups = _build_param_groups(
        forward_value_params, forward_grad_params, lr_value=lr_value, lr_grad=lr_grad
    )
    opt_forward = _create_optimizer_from_config(config, forward_groups)

    backward_value_params = [
        p for p in solver.value_hat_net.parameters() if p.requires_grad
    ]
    backward_grad_params = [p for p in solver.grad_hat_net.parameters() if p.requires_grad]
    backward_groups = _build_param_groups(
        backward_value_params, backward_grad_params, lr_value=lr_value, lr_grad=lr_grad
    )
    opt_backward = _create_optimizer_from_config(config, backward_groups)

    params_interaction: list[nn.Parameter] = []
    if solver.drift_net is not None:
        params_interaction.extend(
            p for p in solver.drift_net.parameters() if p.requires_grad
        )
    if solver.cost_net is not None:
        params_interaction.extend(
            p for p in solver.cost_net.parameters() if p.requires_grad
        )

    if params_interaction:
        opt_interaction: Optional[optim.Optimizer] = optim.Adam(
            params_interaction, lr=config.learning_rate_interaction
        )
    else:
        opt_interaction = None

    scheduler_forward = _create_scheduler_from_config(config, opt_forward)
    scheduler_backward = _create_scheduler_from_config(config, opt_backward)
    scheduler_interaction = (
        _create_scheduler_from_config(config, opt_interaction)
        if opt_interaction is not None
        else None
    )

    return (
        opt_forward,
        opt_backward,
        opt_interaction,
        scheduler_forward,
        scheduler_backward,
        scheduler_interaction,
    )


class Trainer:
    """End-to-end trainer for Mean-Field Schrödinger Bridge.

    This trainer implements the 4-step cycle:
        1. Backward drift learning: Update (Ŷ, Ẑ) from forward trajectories
        2. Interaction learning: Update (f_α, F_β) from forward policy
        3. Forward drift learning: Update (Y, Z) from backward trajectories
        4. Interaction learning: Update (f_α, F_β) from backward policy

    Attributes:
        config: Configuration object
        solver: FBSDESolver instance
        opt_forward: Optimizer for (Y, Z)
        opt_backward: Optimizer for (Ŷ, Ẑ)
        opt_interaction: Optimizer for (f_α, F_β)
        scheduler_forward: LR scheduler for forward networks
        scheduler_backward: LR scheduler for backward networks
        scheduler_interaction: LR scheduler for interaction networks
        metrics: Dictionary storing training metrics
        logger: Logger instance

    Example:
        >>> config = Config(dim=2, T=1.0, n_outer_iterations=10)
        >>> solver = FBSDESolver(config)
        >>> trainer = Trainer(config, solver)
        >>> trainer.train(rho_0, rho_1)
    """

    def __init__(
        self,
        config: Config,
        solver: FBSDESolver,
        log_p0_fn: Optional[Callable[[Tensor], Tensor]] = None,
        log_p1_fn: Optional[Callable[[Tensor], Tensor]] = None,
        f_interaction_fn: Optional[Callable[[Tensor, Tensor, Tensor], Tensor]] = None,
        F_interaction_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        analytical_drift_module: Optional[nn.Module] = None,
        analytical_cost_module: Optional[nn.Module] = None,
        checkpoint_callbacks: Optional[List] = None,
        *,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize Trainer.

        Args:
            config: Configuration object
            solver: FBSDESolver instance with all networks
            log_p0_fn: Log-density function for initial distribution ρ₀
            log_p1_fn: Log-density function for target distribution ρ₁
            f_interaction_fn: True interaction drift function f(x, empirical, t) for training labels
                Required if solver.drift_net is trainable (learned interactions)
            F_interaction_fn: True interaction cost function F(x, empirical) for training labels
                Required if solver.cost_net is trainable (learned interactions)
            checkpoint_callbacks: Optional list of callback functions to execute at checkpoints.
                Each callback should accept (iteration, metrics, trainer) as arguments.
            logger: Optional preconfigured logger to reuse for run/training output.
        """
        self.config = config
        self.checkpoint_callbacks = checkpoint_callbacks or []
        self.solver = solver
        self.logger = logger
        self.log_p0_fn = log_p0_fn
        self.log_p1_fn = log_p1_fn
        self.learn_interaction_drift = config.learn_interaction_drift
        self.learn_interaction_cost = config.learn_interaction_cost
        self.learn_interaction_enabled = bool(
            self.learn_interaction_drift or self.learn_interaction_cost
        )
        self.resample_interaction_sample = config.resample_interaction_sample
        self.analytical_drift_module = analytical_drift_module
        if self.analytical_drift_module is not None:
            self.analytical_drift_module = self.analytical_drift_module.to(
                self.config.device
            )
        self.analytical_cost_module = analytical_cost_module
        if self.analytical_cost_module is not None:
            self.analytical_cost_module = self.analytical_cost_module.to(
                self.config.device
            )
        self._sample_rho0_fn: Optional[Callable[[int], Tensor]] = None
        self._sample_rho1_fn: Optional[Callable[[int], Tensor]] = None

        (
            self.opt_forward,
            self.opt_backward,
            self.opt_interaction,
            self.scheduler_forward,
            self.scheduler_backward,
            self.scheduler_interaction,
        ) = setup_optimizers_and_schedulers(
            config=self.config,
            solver=self.solver,
            learn_interaction_enabled=self.learn_interaction_enabled,
        )

        # Setup logging
        self._setup_logging()

        if self.learn_interaction_enabled and self.opt_interaction is None:
            raise ValueError(
                "Interaction learning is enabled but no interaction optimizer was created."
            )

        # Initialize metrics storage
        self.metrics: Dict[str, Any] = {
            "loss_ipf_forward": [],
            "loss_ipf_backward": [],
            "loss_td_forward": [],
            "loss_td_backward": [],
            "loss_fk_forward": [],
            "loss_fk_backward": [],
            "loss_boundary_forward": [],
            "loss_boundary_backward": [],
            "loss_interaction_drift": [],
            "loss_interaction_cost": [],
            "loss_ipf_forward_analytical": [],
            "loss_ipf_backward_analytical": [],
            "loss_td_forward_analytical": [],
            "loss_td_backward_analytical": [],
            "loss_fk_forward_analytical": [],
            "loss_fk_backward_analytical": [],
            "loss_boundary_forward_analytical": [],
            "loss_boundary_backward_analytical": [],
            "time_per_iteration": [],
            "time_drift_per_iteration": [],
            "time_interaction_per_iteration": [],
            "time_total": 0.0,
            "time_drift_total": 0.0,
            "time_interaction_total": 0.0,
            "time_checkpoint_eval_total": 0.0,
            "time_random_batch_drift_sampling_total": 0.0,
            "time_random_batch_drift_eval_total": 0.0,
            "random_batch_drift_call_count": 0.0,
            "time_random_batch_cost_sampling_total": 0.0,
            "time_random_batch_cost_eval_total": 0.0,
            "random_batch_cost_call_count": 0.0,
            "time_nn_drift_inference_total": 0.0,
            "nn_drift_call_count": 0.0,
            "time_nn_cost_inference_total": 0.0,
            "nn_cost_call_count": 0.0,
            "time_exact_drift_eval_total": 0.0,
            "exact_drift_call_count": 0.0,
            "time_exact_cost_eval_total": 0.0,
            "exact_cost_call_count": 0.0,
            "lr_forward": [],
            "lr_backward": [],
            "lr_interaction": [],
            "lr_forward_grad": [],
            "lr_backward_grad": [],
        }

        # Setup output directories
        self._setup_directories()

        self.global_step = 0

        # Exponential moving average (EMA) targets
        self.use_ema = config.ema_decay > 0.0
        self.ema_active = False
        if self.use_ema and ExponentialMovingAverage is None:
            raise RuntimeError(
                "EMA requested (ema_decay>0) but torch_ema is not installed."
            )
        if self.use_ema:
            self._init_ema_targets()

        # Initialize InteractionLearner if we have trainable interaction networks
        self.interaction_learner = None
        if self.learn_interaction_enabled:
            if self.learn_interaction_drift:
                if solver.drift_net is None:
                    raise ValueError(
                        "learn_interaction_drift=True requires solver.drift_net."
                    )
                if f_interaction_fn is None:
                    raise ValueError(
                        "learn_interaction_drift=True requires f_interaction_fn."
                    )
            if self.learn_interaction_cost:
                if solver.cost_net is None:
                    raise ValueError(
                        "learn_interaction_cost=True requires solver.cost_net."
                    )
                if F_interaction_fn is None:
                    raise ValueError(
                        "learn_interaction_cost=True requires F_interaction_fn."
                    )

            # Create InteractionLearner
            # It will use solver.drift_net and solver.cost_net internally
            self.interaction_learner = InteractionLearner(
                config=config,
                f_interaction_fn=f_interaction_fn,
                F_interaction_fn=F_interaction_fn,
            )

            # Replace InteractionLearner's networks with solver's networks
            # and optimizer with the one we created
            if self.learn_interaction_drift:
                self.interaction_learner.f_net = solver.drift_net
            if self.learn_interaction_cost:
                self.interaction_learner.F_net = solver.cost_net

            # Replace optimizer
            self.interaction_learner.optimizer = self.opt_interaction

        if self.interaction_learner is not None:
            self.logger.info("InteractionLearner initialized for learned interactions")

    @contextmanager
    def _override_interactions(
        self,
        drift_module: Optional[nn.Module],
        cost_module: Optional[nn.Module],
    ):
        original_interaction = self.solver.drift_net
        original_cost = self.solver.cost_net
        if drift_module is not None:
            self.solver.drift_net = drift_module
        else:
            self.solver.drift_net = None
        if cost_module is not None:
            self.solver.cost_net = cost_module
        else:
            self.solver.cost_net = None
        try:
            yield
        finally:
            self.solver.drift_net = original_interaction
            self.solver.cost_net = original_cost

    def _compute_boundary_loss(
        self,
        xs: Tensor,
        target_fn: Optional[Callable[[Tensor], Tensor]],
        value_net: nn.Module,
        value_other_net: nn.Module,
        t_value: float,
    ) -> Tensor:
        """Compute boundary loss with correct indexing for both trajectory types.

        CRITICAL: After backward SDE indexing fix, both forward and backward
        trajectories use physical time indexing:
            - Forward traj: xs[0] at t=0, xs[-1] at t=T
            - Backward traj: xs[0] at t=0, xs[-1] at t=T (after fix)

        Therefore, we select boundary based on t_value:
            - t_value=0.0 → use xs[0] (both trajectory types)
            - t_value=T → use xs[-1] (both trajectory types)

        This enforces self-consistency between Y and Ŷ networks at boundaries
        using the formula: Y(x,t) = log_p(x) - Ŷ(x,t)

        Args:
            xs: Trajectory samples (n_steps+1, batch, dim)
            target_fn: Log-density function for boundary distribution
            value_net: Value network to train (Y or Ŷ)
            value_other_net: The other value network (Ŷ or Y)
            t_value: Time value at boundary (0.0 or T)

        Returns:
            Mean squared error loss between network output and target.
        """
        if target_fn is None:
            return torch.tensor(0.0, device=self.config.device)

        # CRITICAL FIX: Select boundary state based on t_value, not trajectory type
        if abs(t_value - 0.0) < 1e-6:
            # Boundary at t=0: use xs[0] for BOTH forward and backward trajectories
            x_boundary = xs[0]
        elif abs(t_value - self.config.T) < 1e-6:
            # Boundary at t=T: use xs[-1] for BOTH forward and backward trajectories
            x_boundary = xs[-1]
        else:
            raise ValueError(
                f"Invalid t_value {t_value}, expected 0.0 or {self.config.T}"
            )

        batch = x_boundary.shape[0]
        t_tensor = torch.full((batch,), t_value, device=self.config.device)

        with torch.no_grad():
            log_p = target_fn(x_boundary)
            value_other = value_other_net(x_boundary, t_tensor).squeeze(-1)
            # DeepGSB formula: Y = log_p - Ŷ (or Ŷ = log_p - Y)
            target = (log_p - value_other).detach()

        prediction = value_net(x_boundary, t_tensor).squeeze(-1)
        boundary_loss = torch.nn.functional.mse_loss(prediction, target)
        return boundary_loss

    def _compute_td_loss(
        self,
        xs: Tensor,
        ts: Tensor,
        dws: Tensor,
        direction: str,
        target_value_net: Optional[nn.Module] = None,
        target_aux_grad_net: Optional[nn.Module] = None,
        target_value_hat_net: Optional[nn.Module] = None,
        target_grad_hat_net: Optional[nn.Module] = None,
    ) -> Tensor:
        """Compute on-policy TD loss for the current trajectory."""
        loss_td_on = self.solver.compute_td_loss(
            xs,
            ts,
            dws,
            direction=direction,
            target_value_net=target_value_net,
            target_aux_grad_net=target_aux_grad_net,
            target_value_hat_net=target_value_hat_net,
            target_grad_hat_net=target_grad_hat_net,
            log_p0_fn=self.log_p0_fn,
            log_p1_fn=self.log_p1_fn,
            trajectory_direction="backward" if direction == "forward" else "forward",
        )
        if torch.isnan(loss_td_on):
            # Gather diagnostic information
            diagnostics = {
                "direction": direction,
                "loss_value": float(loss_td_on),
                "xs_has_nan": torch.isnan(xs).any().item(),
                "xs_has_inf": torch.isinf(xs).any().item(),
                "ts_has_nan": torch.isnan(ts).any().item(),
                "dws_has_nan": torch.isnan(dws).any().item(),
                "xs_min": float(xs.min()),
                "xs_max": float(xs.max()),
                "xs_mean": float(xs.mean()),
                "xs_std": float(xs.std()),
            }

            # Check network parameters for NaN/Inf
            if direction == "backward":
                value_net = self.solver.value_hat_net
                grad_net = self.solver.grad_hat_net
            else:
                value_net = self.solver.value_net
                grad_net = self.solver.grad_net

            value_params_nan = any(torch.isnan(p).any() for p in value_net.parameters())
            grad_params_nan = any(torch.isnan(p).any() for p in grad_net.parameters())

            diagnostics["value_net_has_nan"] = value_params_nan
            diagnostics["grad_net_has_nan"] = grad_params_nan
            diagnostics["global_step"] = self.global_step

            error_msg = (
                f"NaN in on-policy TD loss (direction={direction}).\nDiagnostics:\n"
            )
            for key, val in diagnostics.items():
                error_msg += f"  {key}: {val}\n"
            error_msg += (
                "\nPossible causes:\n"
                "  1. Gradient explosion - try reducing learning rate\n"
                "  2. Numerical instability on MPS - try CPU device\n"
                "  3. Invalid input distributions\n"
                "  4. Divergence term computation issue\n"
                "  5. Large td_horizon causing accumulation errors\n"
            )

            raise RuntimeError(error_msg)

        return loss_td_on

    def _init_ema_targets(self) -> None:
        """Initialise EMA trackers and shadow networks."""
        self._forward_params = list(
            chain(
                self.solver.value_net.parameters(),
                self.solver.grad_net.parameters(),
            )
        )
        self._backward_params = list(
            chain(
                self.solver.value_hat_net.parameters(),
                self.solver.grad_hat_net.parameters(),
            )
        )
        self.ema_forward = ExponentialMovingAverage(
            self._forward_params,
            decay=self.config.ema_decay,
        )
        self.ema_backward = ExponentialMovingAverage(
            self._backward_params,
            decay=self.config.ema_decay,
        )

        self.target_value_net_forward = copy.deepcopy(self.solver.value_net).to(
            self.config.device
        )
        self.target_grad_net_forward = copy.deepcopy(self.solver.grad_net).to(
            self.config.device
        )
        self.target_value_net_backward = copy.deepcopy(self.solver.value_hat_net).to(
            self.config.device
        )
        self.target_grad_net_backward = copy.deepcopy(self.solver.grad_hat_net).to(
            self.config.device
        )

        for net in (
            self.target_value_net_forward,
            self.target_grad_net_forward,
            self.target_value_net_backward,
            self.target_grad_net_backward,
        ):
            net.eval()
            for param in net.parameters():
                param.requires_grad_(False)

        self._forward_target_params = list(
            chain(
                self.target_value_net_forward.parameters(),
                self.target_grad_net_forward.parameters(),
            )
        )
        self._backward_target_params = list(
            chain(
                self.target_value_net_backward.parameters(),
                self.target_grad_net_backward.parameters(),
            )
        )

    def _apply_ema_to_targets(self, direction: str) -> None:
        """Copy EMA parameters to shadow networks."""
        if direction == "forward":
            self.ema_forward.copy_to(self._forward_target_params)
            self.target_value_net_forward.eval()
            self.target_grad_net_forward.eval()
        elif direction == "backward":
            self.ema_backward.copy_to(self._backward_target_params)
            self.target_value_net_backward.eval()
            self.target_grad_net_backward.eval()
        self.ema_active = True

    def _maybe_update_ema(self, direction: str) -> None:
        """Update EMA trackers if scheduling permits."""
        if not getattr(self, "use_ema", False):
            return
        if self.global_step < self.config.ema_start_step:
            return
        interval = max(1, self.config.ema_update_interval)
        if (self.global_step - self.config.ema_start_step) % interval != 0:
            return

        if direction == "forward":
            self.ema_forward.update(self._forward_params)
            self._apply_ema_to_targets(direction)
        elif direction == "backward":
            self.ema_backward.update(self._backward_params)
            self._apply_ema_to_targets(direction)

    def _get_target_networks(
        self, direction: str
    ) -> tuple[
        Optional[nn.Module],  # target_value_net
        Optional[nn.Module],  # target_aux_grad_net
        Optional[nn.Module],  # target_value_hat_net
        Optional[nn.Module],  # target_grad_hat_net
    ]:
        """Return all 4 EMA shadow networks for TD loss.

        Args:
            direction: "forward" or "backward" - which policy is being trained

        Returns:
            Tuple of (target_value_net, target_aux_grad_net,
                      target_value_hat_net, target_grad_hat_net)

            For forward: (Y_target, Ẑ_target, Ŷ_target, Z_target)
            For backward: (Ŷ_target, Z_target, Y_target, Ẑ_target)
        """
        if not (getattr(self, "use_ema", False) and getattr(self, "ema_active", False)):
            return None, None, None, None

        if direction == "forward":
            return (
                self.target_value_net_forward,  # Y
                self.target_grad_net_backward,  # Ẑ
                self.target_value_net_backward,  # Ŷ
                self.target_grad_net_forward,  # Z
            )
        if direction == "backward":
            return (
                self.target_value_net_backward,  # Ŷ
                self.target_grad_net_forward,  # Z
                self.target_value_net_forward,  # Y
                self.target_grad_net_backward,  # Ẑ
            )
        raise ValueError(f"Unknown direction '{direction}'")

    def _setup_logging(self) -> None:
        """Setup logging to file and console."""
        if self.logger is not None:
            return
        self.logger = create_trainer_logger(self.config, name="nlmfsp.trainer")

    def _setup_directories(self) -> None:
        """Create output directories if they don't exist."""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _log_training_header(self, start_iteration: int = 0) -> None:
        """Log training start/resume message."""
        if start_iteration > 0:
            self.logger.info(f"Resuming Training from Iteration {start_iteration}")
        else:
            self.logger.info("Starting Training")
        if start_iteration > 0:
            self.logger.info(f"  Resuming from: iteration {start_iteration}")
        self.logger.info("")

    def _update_schedulers_and_record_lr(
        self,
        *,
        step_forward: bool = True,
        step_backward: bool = True,
        step_interaction: bool = True,
    ) -> None:
        """Update learning rate schedulers and record learning rates."""
        if self.scheduler_forward is not None and step_forward:
            self.scheduler_forward.step()
        if self.scheduler_backward is not None and step_backward:
            self.scheduler_backward.step()

        lr_interaction = 0.0
        if self.scheduler_interaction is not None and step_interaction:
            self.scheduler_interaction.step()
        if self.opt_interaction is not None:
            lr_interaction = self.opt_interaction.param_groups[0]["lr"]
        elif self.scheduler_interaction is not None:
            # Scheduler exists only when opt_interaction exists, but guard for safety
            lr_interaction = 0.0

        self.metrics["lr_interaction"].append(lr_interaction)
        self.metrics["lr_forward"].append(self.opt_forward.param_groups[0]["lr"])
        self.metrics["lr_backward"].append(self.opt_backward.param_groups[0]["lr"])
        forward_grad_lr = (
            self.opt_forward.param_groups[1]["lr"]
            if len(self.opt_forward.param_groups) > 1
            else self.opt_forward.param_groups[0]["lr"]
        )
        backward_grad_lr = (
            self.opt_backward.param_groups[1]["lr"]
            if len(self.opt_backward.param_groups) > 1
            else self.opt_backward.param_groups[0]["lr"]
        )
        self.metrics["lr_forward_grad"].append(forward_grad_lr)
        self.metrics["lr_backward_grad"].append(backward_grad_lr)

    def _log_training_footer(self) -> None:
        """Log training completion message.
        """
        self.logger.info("")
        self.logger.info("Training Complete")

        total_time = float(self.metrics.get("time_total", 0.0))
        if total_time == 0.0 and self.metrics.get("time_per_iteration"):
            total_time = float(sum(self.metrics["time_per_iteration"]))

        if total_time > 0.0:
            self.logger.info(f"Total runtime: {total_time:.2f}s")

    def _evaluate_losses_with_analytical(
        self, batch_size: Optional[int] = None
    ) -> Optional[Dict[str, float]]:
        """Evaluate losses using analytical interaction functions on fresh trajectories."""
        return evaluate_losses_with_analytical(
            config=self.config,
            solver=self.solver,
            logger=self.logger,
            metrics=self.metrics,
            sample_rho0_fn=self._sample_rho0_fn,
            sample_rho1_fn=self._sample_rho1_fn,
            log_p0_fn=self.log_p0_fn,
            log_p1_fn=self.log_p1_fn,
            analytical_drift_module=self.analytical_drift_module,
            analytical_cost_module=self.analytical_cost_module,
            get_target_networks=self._get_target_networks,
            override_interactions=self._override_interactions,
            compute_boundary_loss=self._compute_boundary_loss,
            batch_size=batch_size,
            seed=self.config.validation_seed,
        )

    def train(
        self,
        sample_rho0_fn: Callable[[int], Tensor],
        sample_rho1_fn: Callable[[int], Tensor],
        start_iteration: int = 0,
    ) -> Dict[str, List[float]]:
        """Main training loop with support for resuming from checkpoints.

        Args:
            sample_rho0_fn: Function to sample from initial distribution ρ₀
                           Takes batch_size and returns (batch_size, dim) tensor
            sample_rho1_fn: Function to sample from target distribution ρ₁
            start_iteration: Starting iteration number (default: 0).
                           Use this when resuming from a checkpoint.

        Returns:
            metrics: Dictionary of training metrics

        Example:
            >>> rho_0 = torch.distributions.Normal(0, 1)
            >>> rho_1 = torch.distributions.Normal(2, 1)
            >>> # Normal training
            >>> trainer.train(
            ...     lambda n: rho_0.sample((n, config.dim)),
            ...     lambda n: rho_1.sample((n, config.dim))
            ... )
            >>> # Resume from iteration 10
            >>> trainer.load_checkpoint("checkpoint_0010.pt")
            >>> trainer.train(
            ...     lambda n: rho_0.sample((n, config.dim)),
            ...     lambda n: rho_1.sample((n, config.dim)),
            ...     start_iteration=10
            ... )
        """
        self._sample_rho0_fn = sample_rho0_fn
        self._sample_rho1_fn = sample_rho1_fn
        rb_stats_start = self._collect_random_batch_timing_stats()
        self._log_training_header(start_iteration)

        for iteration in range(start_iteration, self.config.n_outer_iterations):
            iter_start_time = time.time()
            drift_time = 0.0
            interaction_time = 0.0

            self.logger.info(
                f"Iteration {iteration + 1}/{self.config.n_outer_iterations}"
            )

            # Step 1: Backward drift learning
            drift_start = time.time()
            (
                avg_ipf_bwd,
                avg_td_bwd,
                avg_fk_bwd,
                avg_boundary_bwd,
            ) = self.backward_drift_step(sample_rho0_fn, iteration_idx=iteration)
            drift_time += time.time() - drift_start
            self.metrics["loss_ipf_backward"].append(avg_ipf_bwd)
            self.metrics["loss_td_backward"].append(avg_td_bwd)
            self.metrics["loss_fk_backward"].append(avg_fk_bwd)
            self.metrics["loss_boundary_backward"].append(avg_boundary_bwd)

            # Step 2: Interaction learning (forward policy, starting from ρ₀)
            if self.learn_interaction_enabled and self.opt_interaction is not None:
                interaction_start = time.time()
                loss_drift, loss_cost = self.interaction_learning_step(
                    sample_rho0_fn, mode="forward", iteration_idx=iteration
                )
                interaction_time += time.time() - interaction_start
                self.metrics["loss_interaction_drift"].append(loss_drift)
                self.metrics["loss_interaction_cost"].append(loss_cost)

            # Step 3: Forward drift learning
            drift_start = time.time()
            (
                avg_ipf_fwd,
                avg_td_fwd,
                avg_fk_fwd,
                avg_boundary_fwd,
            ) = self.forward_drift_step(sample_rho1_fn, iteration_idx=iteration)
            drift_time += time.time() - drift_start
            self.metrics["loss_ipf_forward"].append(avg_ipf_fwd)
            self.metrics["loss_td_forward"].append(avg_td_fwd)
            self.metrics["loss_fk_forward"].append(avg_fk_fwd)
            self.metrics["loss_boundary_forward"].append(avg_boundary_fwd)

            # Step 4: Interaction learning (backward policy, starting from ρ₁)
            if self.learn_interaction_enabled and self.opt_interaction is not None:
                interaction_start = time.time()
                loss_drift, loss_cost = self.interaction_learning_step(
                    sample_rho1_fn, mode="backward", iteration_idx=iteration
                )
                interaction_time += time.time() - interaction_start
                # Average with backward interaction losses
                self.metrics["loss_interaction_drift"][-1] = (
                    self.metrics["loss_interaction_drift"][-1] + loss_drift
                ) / 2
                self.metrics["loss_interaction_cost"][-1] = (
                    self.metrics["loss_interaction_cost"][-1] + loss_cost
                ) / 2
            else:
                if len(self.metrics["loss_interaction_drift"]) < iteration + 1:
                    self.metrics["loss_interaction_drift"].append(0.0)
                    self.metrics["loss_interaction_cost"].append(0.0)

            # Update learning rates
            self._update_schedulers_and_record_lr()

            # Record time
            iter_time = time.time() - iter_start_time
            self.metrics["time_per_iteration"].append(iter_time)
            self.metrics["time_drift_per_iteration"].append(drift_time)
            self.metrics["time_interaction_per_iteration"].append(interaction_time)
            self.metrics["time_total"] += iter_time
            self.metrics["time_drift_total"] += drift_time
            self.metrics["time_interaction_total"] += interaction_time

            # Logging
            if (iteration + 1) % self.config.log_interval == 0:
                self._log_metrics(iteration + 1)

            # Checkpointing
            if (iteration + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(iteration + 1)
                # Execute checkpoint callbacks
                for callback in self.checkpoint_callbacks:
                    callback(iteration + 1, self.metrics, self)

        self._log_training_footer()
        rb_stats_end = self._collect_random_batch_timing_stats()
        for key, end_value in rb_stats_end.items():
            self.metrics[key] = end_value - rb_stats_start.get(key, 0.0)

        # Save final checkpoint
        self.save_checkpoint("final")

        return self.metrics

    def _collect_random_batch_timing_stats(self) -> Dict[str, float]:
        stats: Dict[str, float] = {
            "time_random_batch_drift_sampling_total": 0.0,
            "time_random_batch_drift_eval_total": 0.0,
            "random_batch_drift_call_count": 0.0,
            "time_random_batch_cost_sampling_total": 0.0,
            "time_random_batch_cost_eval_total": 0.0,
            "random_batch_cost_call_count": 0.0,
            "time_nn_drift_inference_total": 0.0,
            "nn_drift_call_count": 0.0,
            "time_nn_cost_inference_total": 0.0,
            "nn_cost_call_count": 0.0,
            "time_exact_drift_eval_total": 0.0,
            "exact_drift_call_count": 0.0,
            "time_exact_cost_eval_total": 0.0,
            "exact_cost_call_count": 0.0,
        }
        module_specs = (
            (
                self.solver.drift_net,
                "time_random_batch_drift_sampling_total",
                "time_random_batch_drift_eval_total",
                "random_batch_drift_call_count",
                "time_nn_drift_inference_total",
                "nn_drift_call_count",
            ),
            (
                self.solver.cost_net,
                "time_random_batch_cost_sampling_total",
                "time_random_batch_cost_eval_total",
                "random_batch_cost_call_count",
                "time_nn_cost_inference_total",
                "nn_cost_call_count",
            ),
        )
        for module, sample_key, eval_key, count_key, nn_eval_key, nn_count_key in module_specs:
            if module is None or not hasattr(module, "timing_stats"):
                continue
            module_stats = module.timing_stats()
            kind = module_stats.get("kind")
            if kind == "random_batch":
                stats[sample_key] = float(module_stats.get("sample_time_total", 0.0))
                stats[eval_key] = float(module_stats.get("eval_time_total", 0.0))
                stats[count_key] = float(module_stats.get("call_count", 0.0))
            elif kind == "nn":
                stats[nn_eval_key] = float(module_stats.get("eval_time_total", 0.0))
                stats[nn_count_key] = float(module_stats.get("call_count", 0.0))
        if hasattr(self.solver, "timing_stats"):
            solver_stats = self.solver.timing_stats()
            stats["time_exact_drift_eval_total"] = float(
                solver_stats.get("exact_drift_eval_time_total", 0.0)
            )
            stats["exact_drift_call_count"] = float(
                solver_stats.get("exact_drift_call_count", 0.0)
            )
            stats["time_exact_cost_eval_total"] = float(
                solver_stats.get("exact_cost_eval_time_total", 0.0)
            )
            stats["exact_cost_call_count"] = float(
                solver_stats.get("exact_cost_call_count", 0.0)
            )
        return stats

    def _make_training_seed(self, *parts: int) -> Optional[int]:
        """Derive a stable step-local seed from the configured training seed."""
        base = self.config.training_seed
        if base is None:
            return None
        seed = int(base)
        modulus = 2_147_483_647
        for part in parts:
            seed = (seed * 1_000_003 + int(part) + 0x9E3779B9) % modulus
        if seed == 0:
            seed = 1
        return seed

    def _sample_drift_trajectory(
        self,
        *,
        direction: str,
        sample_fn: Callable[[int], Tensor],
        seed: Optional[int] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if direction == "backward":
            with torch_seed_context(seed_offset(seed, 0)):
                x0 = sample_fn(self.config.n_agents)
            return self.solver.sample_forward_trajectory(
                x0, seed=seed_offset(seed, 1)
            )
        if direction == "forward":
            with torch_seed_context(seed_offset(seed, 0)):
                x1 = sample_fn(self.config.n_agents)
            return self.solver.sample_backward_trajectory(
                x1, seed=seed_offset(seed, 1)
            )
        raise ValueError(f"Invalid direction '{direction}'")

    def _run_drift_step_single(
        self,
        *,
        direction: str,
        xs: Tensor,
        ts: Tensor,
        dws: Tensor,
        seed: Optional[int] = None,
    ) -> Tensor:

        if direction == "backward":
            optimizer = self.opt_backward
            clip_modules = (
                self.solver.value_hat_net.parameters(),
                self.solver.grad_hat_net.parameters(),
            )
            boundary_log_fn = self.log_p1_fn
            boundary_value_net = self.solver.value_hat_net
            boundary_reference_net = self.solver.value_net
            boundary_time = self.config.T
        elif direction == "forward":
            optimizer = self.opt_forward
            clip_modules = (
                self.solver.value_net.parameters(),
                self.solver.grad_net.parameters(),
            )
            boundary_log_fn = self.log_p0_fn
            boundary_value_net = self.solver.value_net
            boundary_reference_net = self.solver.value_hat_net
            boundary_time = 0.0
        else:
            raise ValueError(f"Invalid direction '{direction}'")

        with torch_seed_context(seed):
            with self.solver.stochastic_seed_scope(seed):
                loss_ipf = self.solver.compute_ipf_loss(
                    xs, ts, self.log_p0_fn, self.log_p1_fn, direction=direction
                )

                (
                    target_value_net,
                    target_aux_grad_net,
                    target_value_hat_net,
                    target_grad_hat_net,
                ) = self._get_target_networks(direction)
                loss_td = self._compute_td_loss(
                    xs,
                    ts,
                    dws,
                    direction=direction,
                    target_value_net=target_value_net,
                    target_aux_grad_net=target_aux_grad_net,
                    target_value_hat_net=target_value_hat_net,
                    target_grad_hat_net=target_grad_hat_net,
                )

                loss_fk = self.solver.compute_fk_loss(xs, ts, direction=direction)

        if self.config.lambda_boundary > 0.0:
            loss_boundary = self._compute_boundary_loss(
                xs,
                boundary_log_fn,
                boundary_value_net,
                boundary_reference_net,
                t_value=boundary_time,
            )
        else:
            loss_boundary = xs.new_tensor(0.0)

        if torch.isnan(loss_ipf) or torch.isnan(loss_fk) or torch.isnan(loss_boundary):
            raise RuntimeError(f"NaN encountered in {direction} drift step")

        loss_total = (
            self.config.lambda_ipf * loss_ipf
            + self.config.lambda_td * loss_td
            + self.config.lambda_fk * loss_fk
        )
        if self.config.lambda_boundary > 0.0:
            loss_total = loss_total + self.config.lambda_boundary * loss_boundary

        optimizer.zero_grad()
        loss_total.backward()

        if self.config.grad_clip is not None:
            for params in clip_modules:
                nn.utils.clip_grad_norm_(params, self.config.grad_clip)

        optimizer.step()

        self.global_step += 1
        self._maybe_update_ema(direction)

        losses_tensor = torch.stack(
            [
                loss_ipf.detach(),
                loss_td.detach(),
                loss_fk.detach(),
                loss_boundary.detach(),
            ]
        )

        return losses_tensor

    def backward_drift_step(
        self, sample_rho0_fn: Callable[[int], Tensor], *, iteration_idx: int
    ) -> tuple[float, float, float, float]:
        """Step 1: Update (Ŷ, Ẑ) using forward trajectories.

        Args:
            sample_rho0_fn: Function to sample from ρ₀

        Returns:
            (loss_ipf, loss_td, loss_fk, loss_boundary): Average losses
        """
        sum_losses = torch.zeros(4, device=self.config.device)

        for inner_idx in range(self.config.n_inner_steps):
            step_seed = self._make_training_seed(0, iteration_idx, inner_idx)
            xs, ts, dws = self._sample_drift_trajectory(
                direction="backward", sample_fn=sample_rho0_fn, seed=step_seed
            )
            losses = self._run_drift_step_single(
                direction="backward",
                xs=xs,
                ts=ts,
                dws=dws,
                seed=seed_offset(step_seed, 1),
            )
            sum_losses = sum_losses + losses

        avg_losses = sum_losses / float(self.config.n_inner_steps)
        return tuple(float(x) for x in avg_losses.cpu().tolist())

    def forward_drift_step(
        self, sample_rho1_fn: Callable[[int], Tensor], *, iteration_idx: int
    ) -> tuple[float, float, float, float]:
        """Step 3: Update (Y, Z) using backward trajectories.

        Args:
            sample_rho1_fn: Function to sample from ρ₁

        Returns:
            (loss_ipf, loss_td, loss_fk, loss_boundary): Average losses
        """
        sum_losses = torch.zeros(4, device=self.config.device)

        for inner_idx in range(self.config.n_inner_steps):
            step_seed = self._make_training_seed(1, iteration_idx, inner_idx)
            xs, ts, dws = self._sample_drift_trajectory(
                direction="forward", sample_fn=sample_rho1_fn, seed=step_seed
            )
            losses = self._run_drift_step_single(
                direction="forward",
                xs=xs,
                ts=ts,
                dws=dws,
                seed=seed_offset(step_seed, 1),
            )
            sum_losses = sum_losses + losses

        avg_losses = sum_losses / float(self.config.n_inner_steps)
        return tuple(float(x) for x in avg_losses.cpu().tolist())

    def interaction_learning_step(
        self,
        sample_fn: Callable[[int], Tensor],
        mode: str = "forward",
        *,
        iteration_idx: int = 0,
    ) -> tuple[float, float]:
        """Step 2 or 4: Update (f_α, F_β) using current FBSDE policy.

        This method implements the interaction learning steps from the 4-step cycle:
        - Step 2: Train interaction networks using forward policy samples
        - Step 4: Train interaction networks using backward policy samples

        The method:
        1. Uses cached trajectory if provided, otherwise samples from current FBSDE policy
        2. Computes true interactions O(N²) to create training labels
        3. Trains interaction networks (f_α, F_β) via supervised learning

        Args:
            sample_fn: Function to sample from the boundary distribution matching the
                requested policy:
                - mode='forward': source boundary ρ₀
                - mode='backward': terminal boundary ρ₁
            mode: 'forward' or 'backward' - which policy to use for sampling

        Returns:
            (loss_drift, loss_cost): Average drift and cost losses over training steps
        """
        # Skip if no interaction networks to train
        if self.opt_interaction is None:
            raise RuntimeError(
                "interaction_learning_step called without an interaction optimizer."
            )

        # Check if InteractionLearner is available
        if self.interaction_learner is None:
            raise RuntimeError(
                "interaction_learning_step called without an initialized InteractionLearner."
            )

        # Helper to sample one trajectory.
        def _sample_traj(sample_seed: Optional[int]) -> Tuple[Tensor, Tensor]:
            with torch_seed_context(seed_offset(sample_seed, 0)):
                x_boundary = sample_fn(self.config.n_agents)
                if mode == "forward":
                    xs_t, ts_t, _ = self.solver.sample_forward_trajectory(
                        x_boundary, seed=seed_offset(sample_seed, 1)
                    )
                elif mode == "backward":
                    xs_t, ts_t, _ = self.solver.sample_backward_trajectory(
                        x_boundary, seed=seed_offset(sample_seed, 1)
                    )
                    # sample_backward_trajectory returns xs in physical-time order but a
                    # flipped time grid for DeepGSB-style iteration. Interaction learning
                    # trains on xs[i] as the state at physical time t_i, so pass the
                    # matching physical grid here as well.
                    ts_t = self.config.time_grid[: xs_t.shape[0]]
                else:
                    raise ValueError(
                        f"Invalid mode: {mode}. Must be 'forward' or 'backward'."
                    )
            return xs_t, ts_t

        # Resample mode: recompute labels each step
        if self.resample_interaction_sample:
            loss_drift_list = []
            loss_cost_list = []
            mode_tag = 0 if mode == "backward" else 1
            for step_idx in range(self.config.n_interaction_steps):
                step_seed = self._make_training_seed(
                    2, mode_tag, iteration_idx, step_idx
                )
                xs, ts = _sample_traj(step_seed)
                with torch_seed_context(seed_offset(step_seed, 1)):
                    history = self.interaction_learner.train(xs, ts, n_steps=1)
                if self.learn_interaction_drift and history["f_loss"]:
                    loss_drift_list.extend(history["f_loss"])
                if self.learn_interaction_cost and history["F_loss"]:
                    loss_cost_list.extend(history["F_loss"])
            avg_loss_drift = (
                sum(loss_drift_list) / len(loss_drift_list) if loss_drift_list else 0.0
            )
            avg_loss_cost = (
                sum(loss_cost_list) / len(loss_cost_list) if loss_cost_list else 0.0
            )
            return avg_loss_drift, avg_loss_cost

        # Default mode: one trajectory, labels once, multiple optimizer steps
        mode_tag = 0 if mode == "backward" else 1
        traj_seed = self._make_training_seed(3, mode_tag, iteration_idx)
        xs, ts = _sample_traj(traj_seed)
        with torch_seed_context(seed_offset(traj_seed, 1)):
            history = self.interaction_learner.train(
                xs, ts, n_steps=self.config.n_interaction_steps
            )
        avg_loss_drift = (
            sum(history["f_loss"]) / len(history["f_loss"])
            if self.learn_interaction_drift and history["f_loss"]
            else 0.0
        )
        avg_loss_cost = (
            sum(history["F_loss"]) / len(history["F_loss"])
            if self.learn_interaction_cost and history["F_loss"]
            else 0.0
        )
        return avg_loss_drift, avg_loss_cost

    def _log_metrics(self, iteration: int) -> None:
        """Log current training metrics.

        Args:
            iteration: Current iteration number
        """
        self.logger.info(f"  L_IPF (fwd): {self.metrics['loss_ipf_forward'][-1]:.6f}")
        self.logger.info(f"  L_IPF (bwd): {self.metrics['loss_ipf_backward'][-1]:.6f}")
        self.logger.info(f"  L_TD  (bwd): {self.metrics['loss_td_backward'][-1]:.6f}")
        if self.metrics["loss_td_forward"]:
            self.logger.info(
                f"  L_TD  (fwd): {self.metrics['loss_td_forward'][-1]:.6f}"
            )
        self.logger.info(f"  L_FK (fwd): {self.metrics['loss_fk_forward'][-1]:.6f}")
        self.logger.info(f"  L_FK (bwd): {self.metrics['loss_fk_backward'][-1]:.6f}")
        if self.config.lambda_boundary > 0.0:
            self.logger.info(
                f"  L_Boundary (fwd): {self.metrics['loss_boundary_forward'][-1]:.6f}"
            )
            self.logger.info(
                f"  L_Boundary (bwd): {self.metrics['loss_boundary_backward'][-1]:.6f}"
            )

        if self.learn_interaction_enabled and self.opt_interaction is not None:
            if self.learn_interaction_drift and self.metrics["loss_interaction_drift"]:
                self.logger.info(
                    f"  L_drift: {self.metrics['loss_interaction_drift'][-1]:.6f}"
                )
            if self.learn_interaction_cost and self.metrics["loss_interaction_cost"]:
                self.logger.info(
                    f"  L_cost: {self.metrics['loss_interaction_cost'][-1]:.6f}"
                )
        else:
            self.logger.info("  Interaction learning: skipped")

        if self.metrics["time_interaction_per_iteration"]:
            self.logger.info(
                f"  Interaction time: {self.metrics['time_interaction_per_iteration'][-1]:.2f}s"
            )
        self.logger.info(f"  Time: {self.metrics['time_per_iteration'][-1]:.2f}s")
        self.logger.info("")

    def save_checkpoint(self, iteration: int | str) -> None:
        """Save model checkpoint.

        Args:
            iteration: Current iteration number or checkpoint name (e.g., "final")
        """
        if self.analytical_drift_module is not None or self.analytical_cost_module is not None:
            eval_start = time.time()
            with self.solver.suspend_exact_interaction_timing():
                analytical_metrics = self._evaluate_losses_with_analytical()
            self.metrics["time_checkpoint_eval_total"] += time.time() - eval_start
        else:
            analytical_metrics = None
        save_trainer_checkpoint(
            self,
            iteration=iteration,
            analytical_metrics=analytical_metrics,
        )

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            iteration: Iteration number from checkpoint
        """
        return load_trainer_checkpoint(self, checkpoint_path)


__all__ = ["Trainer"]
