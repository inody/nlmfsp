"""Validation and evaluation utilities for training results.

This module provides functions for:
- Evaluating transport quality (mean/std error)
- Summarizing training metrics
- Generating validation reports
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .rng_utils import seed_offset, torch_seed_context

if TYPE_CHECKING:
    from .config import Config
    from .fbsde import FBSDESolver


def _emit_info(logger: Optional[logging.Logger], message: str = "") -> None:
    """Emit an informational message to logger or stdout."""
    if logger is None:
        print(message)
        return
    logger.info(message)


def log_configuration(
    config: "Config", logger: Optional[logging.Logger] = None
) -> None:
    """Log a compact run summary."""
    _emit_info(logger, "")
    _emit_info(logger, "Configuration:")
    _emit_info(logger, f"  Device: {config.device}")
    _emit_info(
        logger,
        f"  State space: dim={config.dim}, T={config.T}, steps={config.n_time_steps}",
    )
    _emit_info(
        logger,
        f"  Training: outer={config.n_outer_iterations}, inner={config.n_inner_steps}, agents={config.n_agents}",
    )
    _emit_info(
        logger,
        f"  Learning rates: value={config.learning_rate_value}, grad={config.learning_rate_grad}",
    )
    if config.learning_rate_interaction is not None:
        _emit_info(logger, f"  Interaction LR: {config.learning_rate_interaction}")
    optimizer_line = f"  Optimizer:    {config.optimizer_type} (weight_decay={config.optimizer_weight_decay}"
    if config.optimizer_momentum is not None:
        optimizer_line += f", momentum={config.optimizer_momentum}"
    optimizer_line += ")"
    _emit_info(logger, optimizer_line)
    scheduler_desc = config.lr_scheduler_type
    if config.lr_scheduler_type.lower() == "step":
        scheduler_desc += (
            f" (step={config.lr_decay_step}, gamma={config.lr_decay_gamma})"
        )
    elif config.lr_scheduler_type.lower() == "exponential":
        scheduler_desc += f" (gamma={config.lr_decay_gamma})"
    else:
        scheduler_desc += " (disabled)"
    _emit_info(logger, f"  LR scheduler: {scheduler_desc}")
    _emit_info(
        logger,
        f"  Loss weights: λ_IPF={config.lambda_ipf}, λ_TD={config.lambda_td}, λ_FK={config.lambda_fk}"
    )
    _emit_info(
        logger,
        f"  TD horizon:   {config.td_horizon} (bootstrap={'stop' if config.td_bootstrap_from_target else 'grad'})"
    )
    _emit_info(
        logger,
        f"  TD multistep: {config.use_multistep_td_loss}, FK norm: {config.fk_loss_norm}",
    )
    _emit_info(
        logger,
        "  Learned drift divergence: {}".format(
            "detached" if config.detach_interaction_drift_divergence else "active"
        ),
    )
    _emit_info(logger, f"  Network widths: x={config.hidden_dims}, t={config.t_hidden_dims}")
    _emit_info(
        logger,
        "  Time input: mode={}, embedding={} (legacy index norm={}, interval={})".format(
            config.time_input_mode,
            "enabled" if config.use_time_embedding else "disabled",
            "on" if config.use_time_index_normalization else "off",
            config.time_index_interval,
        )
    )
    _emit_info(
        logger,
        f"  Interaction batching: query={config.interaction_query_chunk_size}, ref={config.interaction_ref_chunk_size}, time={config.interaction_time_block_size}"
    )
    _emit_info(
        logger,
        "  Interaction normalization: input_scale={}, target_standardization={}".format(
            config.interaction_input_scale,
            "on" if config.standardize_interaction_targets else "off",
        ),
    )
    if config.ema_decay > 0.0:
        _emit_info(
            logger,
            f"  EMA:          decay={config.ema_decay}, start_step={config.ema_start_step}, interval={config.ema_update_interval}"
        )
    else:
        _emit_info(logger, "  EMA:          disabled")
    _emit_info(logger, f"  Diffusion coefficient: σ={config.sigma}")
    _emit_info(logger, f"  Gradient clip: {config.grad_clip}")


def summarize_metrics(
    metrics: Dict[str, list], logger: Optional[logging.Logger] = None
) -> None:
    """Log a summary of final training metrics.

    Args:
        metrics: Dictionary containing training metrics with keys like:
            - loss_ipf_forward, loss_ipf_backward
            - loss_td_forward, loss_td_backward
            - loss_fk_forward, loss_fk_backward
            - loss_boundary_forward, loss_boundary_backward
            Each value is a list of metric values across iterations.

    Prints:
        Final and initial values for each tracked loss component.
    """
    tracked = [
        ("loss_ipf_forward", "L_IPF (fwd)"),
        ("loss_ipf_backward", "L_IPF (bwd)"),
        ("loss_td_forward", "L_TD (fwd)"),
        ("loss_td_backward", "L_TD (bwd)"),
        ("loss_fk_forward", "L_FK (fwd)"),
        ("loss_fk_backward", "L_FK (bwd)"),
        ("loss_boundary_forward", "L_B (fwd)"),
        ("loss_boundary_backward", "L_B (bwd)"),
    ]
    _emit_info(logger, "")
    _emit_info(logger, "Final Metrics:")
    for key, label in tracked:
        if metrics.get(key):
            initial = metrics[key][0]
            final = metrics[key][-1]
            _emit_info(
                logger,
                f"  {label:>16}: {final:.6f} (initial: {initial:.6f})",
            )


def log_memory_profile(
    config: "Config", logger: Optional[logging.Logger] = None
) -> None:
    """Log a rough memory footprint estimate for trajectories."""
    n_steps = config.n_time_steps + 1
    n_agents = config.n_agents
    dim = config.dim
    bytes_fp32 = n_steps * n_agents * dim * 4  # one trajectory
    bytes_two_traj = bytes_fp32 * 2  # forward + backward
    _emit_info(
        logger,
        f"  Estimated trajectory buffer (fwd+bwd, float32): "
        f"{bytes_two_traj / 1e6:.1f} MB ({bytes_two_traj / 1e9:.2f} GB)"
    )
    if torch.cuda.is_available() and "cuda" in str(config.device):
        free, total = torch.cuda.mem_get_info()
        try:
            dev_name = torch.cuda.get_device_name()
        except Exception:
            dev_name = "cuda"
        _emit_info(
            logger,
            f"  CUDA memory: free={free / 1e9:.2f} GB / total={total / 1e9:.2f} GB "
            f"on {dev_name}"
        )


def evaluate_transport_quality(
    solver: FBSDESolver,
    sample_rho0: Callable[[int], torch.Tensor],
    sample_rho1: Callable[[int], torch.Tensor],
    config: Config,
    n_test: int = 1000,
    logger: Optional[logging.Logger] = None,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate the quality of learned transport between distributions.

    Samples test particles from the source distribution, transports them using
    the learned forward solver, and compares the transported distribution with
    the target distribution.

    Args:
        solver: Trained FBSDESolver instance.
        sample_rho0: Function to sample from source distribution ρ₀.
        sample_rho1: Function to sample from target distribution ρ₁.
        config: Configuration object (not currently used but kept for consistency).
        n_test: Number of test samples to use for evaluation.

    Returns:
        Dictionary containing:
            - mean_true: True target mean (list)
            - mean_pred: Predicted mean after transport (list)
            - mean_error: L2 norm of mean difference (float)
            - std_true: True target standard deviation (list)
            - std_pred: Predicted standard deviation after transport (list)
            - std_error: L2 norm of std difference (float)

    Note:
        Prints evaluation results to console including mean/std errors.
    """
    _emit_info(logger, "")
    _emit_info(logger, "Evaluating Transport Quality...")
    with torch_seed_context(seed_offset(seed, 0)):
        with torch.no_grad():
            x0_test = sample_rho0(n_test)
            xs_forward, _, _ = solver.sample_forward_trajectory(
                x0_test, seed=seed_offset(seed, 1)
            )
            x1_pred = xs_forward[-1]
    with torch_seed_context(seed_offset(seed, 2)):
        with torch.no_grad():
            x1_true = sample_rho1(n_test)

    mean_pred = x1_pred.mean(dim=0)
    mean_true = x1_true.mean(dim=0)
    std_pred = x1_pred.std(dim=0)
    std_true = x1_true.std(dim=0)

    mean_error = torch.norm(mean_pred - mean_true).item()
    std_error = torch.norm(std_pred - std_true).item()

    _emit_info(logger, f"  Target mean:     {mean_true.cpu().numpy()}")
    _emit_info(logger, f"  Predicted mean:  {mean_pred.cpu().numpy()}")
    _emit_info(logger, f"  Mean error:      {mean_error:.6f}")
    _emit_info(logger, f"  Target std:      {std_true.cpu().numpy()}")
    _emit_info(logger, f"  Predicted std:   {std_pred.cpu().numpy()}")
    _emit_info(logger, f"  Std error:       {std_error:.6f}")

    return {
        "mean_true": mean_true.cpu().tolist(),
        "mean_pred": mean_pred.cpu().tolist(),
        "mean_error": mean_error,
        "std_true": std_true.cpu().tolist(),
        "std_pred": std_pred.cpu().tolist(),
        "std_error": std_error,
    }


def _time_grid_steps_for(xs: torch.Tensor, config: Config) -> torch.Tensor:
    """Expand the physical time grid to match a trajectory tensor."""
    n_steps = xs.shape[0]
    batch = xs.shape[1]
    return config.time_grid[:n_steps].to(xs.device).unsqueeze(1).expand(n_steps, batch)


def _relative_l2_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    denom = torch.norm(target)
    if float(denom) == 0.0:
        return float(torch.norm(pred - target).item())
    return float((torch.norm(pred - target) / denom).item())


def _evaluate_module_time_batch(
    module: nn.Module,
    xs: torch.Tensor,
    ts_steps: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    """Evaluate a drift/cost module on (time, batch, dim) tensors."""
    if hasattr(module, "evaluate_batched"):
        kwargs = {
            "time_block_size": getattr(config, "interaction_time_block_size", None)
        }
        if getattr(module, "requires_time", False):
            kwargs["time_grid"] = ts_steps
        return module.evaluate_batched(xs, xs, **kwargs)
    for attr_name in ("drift", "cost"):
        base_module = getattr(module, attr_name, None)
        if base_module is not None and hasattr(base_module, "evaluate_batched"):
            kwargs = {
                "time_block_size": getattr(config, "interaction_time_block_size", None)
            }
            if getattr(base_module, "requires_time", False):
                kwargs["time_grid"] = ts_steps
            return base_module.evaluate_batched(xs, xs, **kwargs)
    xs_flat = xs.reshape(-1, xs.shape[-1])
    ts_flat = ts_steps.reshape(-1)
    values = module(xs_flat, ts_flat)
    if values.ndim == 1:
        values = values.unsqueeze(-1)
    return values.reshape(xs.shape[0], xs.shape[1], -1)


def _evaluate_analytical_reference(
    module: nn.Module,
    xs: torch.Tensor,
    ts_steps: torch.Tensor,
    config: Config,
    *,
    attr_name: str,
) -> torch.Tensor:
    """Evaluate the exact analytical interaction on the same trajectories."""
    base_module = getattr(module, attr_name, module)
    time_block_size = getattr(config, "interaction_time_block_size", None)
    if hasattr(base_module, "evaluate_batched"):
        kwargs = {"time_block_size": time_block_size}
        if getattr(base_module, "requires_time", False):
            kwargs["time_grid"] = ts_steps
        return base_module.evaluate_batched(xs, xs, **kwargs)

    outputs = []
    for x_t, t_t in zip(xs, ts_steps):
        if hasattr(base_module, "evaluate"):
            if getattr(base_module, "requires_time", False):
                out_t = base_module.evaluate(x_t, x_t, t_t)
            else:
                out_t = base_module.evaluate(x_t, x_t)
        else:
            out_t = base_module(x_t, t_t)
        if out_t.ndim == 1:
            out_t = out_t.unsqueeze(-1)
        outputs.append(out_t)
    return torch.stack(outputs, dim=0)


def _compute_interaction_error_stats(
    approx: torch.Tensor, exact: torch.Tensor
) -> Dict[str, float]:
    diff = approx - exact
    return {
        "relative_l2": _relative_l2_error(approx, exact),
        "mse": float(torch.mean(diff.pow(2)).item()),
    }


def evaluate_interaction_approximation(
    *,
    solver: "FBSDESolver",
    config: Config,
    analytical_drift_module: Optional[nn.Module],
    analytical_cost_module: Optional[nn.Module],
    xs_forward: torch.Tensor,
    xs_backward: torch.Tensor,
    logger: Optional[logging.Logger] = None,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate final interaction approximation errors on rollout trajectories."""
    stats: Dict[str, Dict[str, float]] = {}
    trajectory_specs = (
        ("forward", xs_forward),
        ("backward", xs_backward),
    )

    with torch_seed_context(seed):
        with torch.no_grad():
            for direction, xs in trajectory_specs:
                ts_steps = _time_grid_steps_for(xs, config)

                if solver.drift_net is not None and analytical_drift_module is not None:
                    approx_drift = _evaluate_module_time_batch(
                        solver.drift_net, xs, ts_steps, config
                    )
                    exact_drift = _evaluate_analytical_reference(
                        analytical_drift_module,
                        xs,
                        ts_steps,
                        config,
                        attr_name="drift",
                    )
                    drift_stats = _compute_interaction_error_stats(
                        approx_drift, exact_drift
                    )
                    stats[f"{direction}_drift"] = drift_stats
                    _emit_info(
                        logger,
                        f"  Interaction {direction} drift error: relL2={drift_stats['relative_l2']:.6f}, mse={drift_stats['mse']:.6f}",
                    )

                if solver.cost_net is not None and analytical_cost_module is not None:
                    approx_cost = _evaluate_module_time_batch(
                        solver.cost_net, xs, ts_steps, config
                    )
                    exact_cost = _evaluate_analytical_reference(
                        analytical_cost_module,
                        xs,
                        ts_steps,
                        config,
                        attr_name="cost",
                    )
                    cost_stats = _compute_interaction_error_stats(approx_cost, exact_cost)
                    stats[f"{direction}_cost"] = cost_stats
                    _emit_info(
                        logger,
                        f"  Interaction {direction} cost error: relL2={cost_stats['relative_l2']:.6f}, mse={cost_stats['mse']:.6f}",
                    )

    return stats


def evaluate_losses_with_analytical(
    *,
    config: "Config",
    solver: "FBSDESolver",
    logger,
    metrics: Dict[str, list],
    sample_rho0_fn: Optional[Callable[[int], torch.Tensor]],
    sample_rho1_fn: Optional[Callable[[int], torch.Tensor]],
    log_p0_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    log_p1_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    analytical_drift_module: Optional[nn.Module],
    analytical_cost_module: Optional[nn.Module],
    get_target_networks: Callable[[str], tuple[Optional[nn.Module], Optional[nn.Module], Optional[nn.Module], Optional[nn.Module]]],
    override_interactions: Callable[[Optional[nn.Module], Optional[nn.Module]], object],
    compute_boundary_loss: Callable[[torch.Tensor, Optional[Callable[[torch.Tensor], torch.Tensor]], nn.Module, nn.Module, float], torch.Tensor],
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> Optional[Dict[str, float]]:
    """Evaluate training losses with analytical interactions on fresh trajectories."""
    eval_on_cpu = bool(getattr(config, "evaluate_interaction_cpu", False))

    if analytical_drift_module is None and analytical_cost_module is None:
        raise ValueError(
            "Analytical loss evaluation requires analytical_drift_module or analytical_cost_module."
        )
    if sample_rho0_fn is None or sample_rho1_fn is None:
        raise ValueError(
            "Analytical loss evaluation requires sample_rho0_fn and sample_rho1_fn."
        )
    if log_p0_fn is None or log_p1_fn is None:
        raise ValueError(
            "Analytical loss evaluation requires log_p0_fn and log_p1_fn."
        )

    eval_batch = int(batch_size or config.n_agents_test or config.n_agents)
    eval_device = torch.device("cpu") if eval_on_cpu else torch.device(config.device)
    original_device = torch.device(config.device)
    target_networks = {
        direction: get_target_networks(direction)
        for direction in ("forward", "backward")
    }

    modules_moved: List[nn.Module] = []
    if eval_on_cpu:
        candidates: List[Optional[nn.Module]] = [
            getattr(solver, "value_net", None),
            getattr(solver, "value_hat_net", None),
            getattr(solver, "grad_net", None),
            getattr(solver, "grad_hat_net", None),
            getattr(solver, "drift_net", None),
            getattr(solver, "cost_net", None),
            analytical_drift_module,
            analytical_cost_module,
            *(mod for nets in target_networks.values() for mod in nets),
        ]
        seen_ids = set()
        for module in candidates:
            if module is None:
                continue
            module_id = id(module)
            if module_id in seen_ids:
                continue
            seen_ids.add(module_id)
            module.to(eval_device)
            modules_moved.append(module)
        solver.device = eval_device
        config.device = eval_device
    else:
        config.device = original_device

    def _restore_modules() -> None:
        if not eval_on_cpu:
            return
        config.device = original_device
        solver.device = original_device
        for module in modules_moved:
            module.to(original_device)

    try:
        with torch_seed_context(seed_offset(seed, 0)):
            x0_eval = sample_rho0_fn(eval_batch)
        with torch_seed_context(seed_offset(seed, 1)):
            x1_eval = sample_rho1_fn(eval_batch)
        if x0_eval.device != eval_device:
            x0_eval = x0_eval.to(eval_device)
        if x1_eval.device != eval_device:
            x1_eval = x1_eval.to(eval_device)

        if eval_on_cpu:

            def log_p0_eval(x: torch.Tensor) -> torch.Tensor:
                return log_p0_fn(x.to(original_device)).to(eval_device)

            def log_p1_eval(x: torch.Tensor) -> torch.Tensor:
                return log_p1_fn(x.to(original_device)).to(eval_device)

        else:
            log_p0_eval = log_p0_fn
            log_p1_eval = log_p1_fn

        with override_interactions(analytical_drift_module, analytical_cost_module):
            xs_fwd, ts_fwd, dws_fwd = solver.sample_forward_trajectory(
                x0_eval, seed=seed_offset(seed, 2)
            )
            xs_bwd, ts_bwd, dws_bwd = solver.sample_backward_trajectory(
                x1_eval, seed=seed_offset(seed, 3)
            )

            loss_ipf_bwd = solver.compute_ipf_loss(
                xs_fwd, ts_fwd, log_p0_eval, log_p1_eval, direction="backward"
            )
            (
                target_value_net_bwd,
                target_aux_grad_net_bwd,
                target_value_hat_net_bwd,
                target_grad_hat_net_bwd,
            ) = target_networks["backward"]
            loss_td_bwd = solver.compute_td_loss(
                xs_fwd,
                ts_fwd,
                dws_fwd,
                direction="backward",
                target_value_net=target_value_net_bwd,
                target_aux_grad_net=target_aux_grad_net_bwd,
                target_value_hat_net=target_value_hat_net_bwd,
                target_grad_hat_net=target_grad_hat_net_bwd,
                log_p0_fn=log_p0_eval,
                log_p1_fn=log_p1_eval,
                trajectory_direction="forward",
            )
            loss_fk_bwd = solver.compute_fk_loss(xs_fwd, ts_fwd, direction="backward")
            loss_boundary_bwd = compute_boundary_loss(
                xs_fwd,
                log_p1_eval,
                solver.value_hat_net,
                solver.value_net,
                config.T,
            )

            loss_ipf_fwd = solver.compute_ipf_loss(
                xs_bwd, ts_bwd, log_p0_eval, log_p1_eval, direction="forward"
            )
            (
                target_value_net_fwd,
                target_aux_grad_net_fwd,
                target_value_hat_net_fwd,
                target_grad_hat_net_fwd,
            ) = target_networks["forward"]
            loss_td_fwd = solver.compute_td_loss(
                xs_bwd,
                ts_bwd,
                dws_bwd,
                direction="forward",
                target_value_net=target_value_net_fwd,
                target_aux_grad_net=target_aux_grad_net_fwd,
                target_value_hat_net=target_value_hat_net_fwd,
                target_grad_hat_net=target_grad_hat_net_fwd,
                log_p0_fn=log_p0_eval,
                log_p1_fn=log_p1_eval,
                trajectory_direction="backward",
            )
            loss_fk_fwd = solver.compute_fk_loss(xs_bwd, ts_bwd, direction="forward")
            loss_boundary_fwd = compute_boundary_loss(
                xs_bwd,
                log_p0_eval,
                solver.value_net,
                solver.value_hat_net,
                0.0,
            )
    finally:
        _restore_modules()

    pairs = {
        "loss_ipf_backward_analytical": loss_ipf_bwd,
        "loss_td_backward_analytical": loss_td_bwd,
        "loss_fk_backward_analytical": loss_fk_bwd,
        "loss_boundary_backward_analytical": loss_boundary_bwd,
        "loss_ipf_forward_analytical": loss_ipf_fwd,
        "loss_td_forward_analytical": loss_td_fwd,
        "loss_fk_forward_analytical": loss_fk_fwd,
        "loss_boundary_forward_analytical": loss_boundary_fwd,
    }
    result: Dict[str, float] = {}
    for key, tensor_value in pairs.items():
        value = float(tensor_value.detach())
        result[key] = value
        metrics.setdefault(key, []).append(value)
    return result
