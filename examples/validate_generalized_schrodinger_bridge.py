"""Training engine for the nonlocal mean-field Schrödinger bridge.

**This is the internal engine, not the entry point.** To reproduce a figure or
table from the paper, use::

    uv run python scripts/reproduce.py --list
    uv run python scripts/reproduce.py --config configs/paper/vneck/main_nn.json

Each config under ``configs/paper/`` carries every setting its artifact needs,
so nothing has to be passed on the command line. This script additionally
accepts overrides for most hyperparameters; those exist for development and are
not part of the published interface. Using them means the run is no longer
described by any config file, which is precisely the situation the paper
configs were introduced to avoid.

The interaction terms can be evaluated in three ways, selected by
``interactions.interaction_backend`` in the config:

    exact         pairwise sums over all particles, O(N^2) per step
    nn            learned surrogates for the nonlocal drift and cost, O(N)
    random_batch  the same sums over a random subset of size M_RB
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from nlmfsp import (
    Config,
    ExperimentManager,
    FBSDESolver,
    Trainer,
    build_config,
    create_interactions,
    create_realtime_visualization_callback,
    evaluate_transport_quality,
    log_configuration,
    plot_metrics,
    plot_trajectories,
    plot_opinion_directional_similarity,
    plot_opinion_snapshots,
    sample_visualization_trajectories,
    summarize_metrics,
)
from nlmfsp.cli_helpers import (
    apply_interaction_config_overrides,
)
from nlmfsp.config import ensure_config_time_input_fields
from nlmfsp.file_utils import create_trainer_logger
from nlmfsp.problem_setup import load_problem_config, setup_problem_artifacts
from nlmfsp.validation import (
    evaluate_interaction_approximation,
    log_memory_profile,
)


def _emit_info(logger: Optional[logging.Logger], message: str = "") -> None:
    """Emit informational output to logger or stdout."""
    if logger is None:
        print(message)
        return
    logger.info(message)


def print_problem_overview(
    problem_cfg: dict, logger: Optional[logging.Logger] = None
) -> None:
    """Log a compact description of the loaded problem."""
    _emit_info(logger, "")
    _emit_info(logger, f"Problem: {problem_cfg['name']} ({problem_cfg['problem_type']})")
    if "description" in problem_cfg:
        _emit_info(logger, f"  {problem_cfg['description']}")


def _format_distribution(cfg: dict) -> str:
    dist_type = cfg["type"]
    if dist_type == "gaussian":
        if "std" in cfg:
            return f"N({cfg['mean']}, {cfg['std']}^2 I)"
        if "covar_diag" in cfg:
            return f"N({cfg['mean']}, diag[{len(cfg['covar_diag'])}])"
        if "covar_override" in cfg:
            return f"N({cfg['mean']}, custom covariance)"
        return f"N({cfg['mean']}, covariance)"
    if dist_type == "gaussian_mixture":
        return f"GMM[{len(cfg['components'])}]"
    return dist_type


def print_distribution_summary(
    problem_cfg: dict, logger: Optional[logging.Logger] = None
) -> None:
    """Log source/target distribution summary once."""
    rho_0_cfg = problem_cfg["distributions"]["rho_0"]
    rho_1_cfg = problem_cfg["distributions"]["rho_1"]
    _emit_info(logger, "")
    _emit_info(logger, "Distributions:")
    _emit_info(logger, f"  rho_0: {_format_distribution(rho_0_cfg)}")
    _emit_info(logger, f"  rho_1: {_format_distribution(rho_1_cfg)}")


def print_obstacle_summary(
    obstacle_config, logger: Optional[logging.Logger] = None
) -> None:
    """Log obstacle summary if present."""
    if not obstacle_config or obstacle_config.get("type") != "multi_obstacle":
        return
    obstacles = obstacle_config["obstacles"]
    _emit_info(logger, "")
    _emit_info(logger, f"Obstacles: {len(obstacles)}")
    for i, obs in enumerate(obstacles):
        desc = obs.get("description", f"Obstacle {i + 1}")
        mode = obs.get("mode", "hard")
        extra = ""
        if mode == "soft" and obs.get("beta") is not None:
            extra = f", beta={obs['beta']}"
        _emit_info(
            logger,
            f"  {desc}: center={obs['center']}, radius={obs['radius']}, penalty={obs['penalty']}, mode={mode}{extra}"
        )


def print_runtime_setup(
    config: Config,
    problem_cfg: Optional[dict],
    exp_manager,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log compact runtime summary once before training."""
    _emit_info(logger, "")
    _emit_info(logger, "Run Setup:")
    _emit_info(
        logger,
        f"  Solver steps: inner={config.n_inner_steps}, interaction={config.n_interaction_steps if config.interaction_backend == 'nn' else 0}"
    )
    _emit_info(
        logger,
        f"  Interaction backend: {config.interaction_backend} "
        f"(drift={config.drift_type}, cost={config.cost_type})"
    )
    if config.interaction_backend == "random_batch":
        _emit_info(logger, f"  Random-batch size: M={config.random_batch_size}")
    base_functions_cfg = (problem_cfg or {}).get("base_functions", {})
    base_drift_type = base_functions_cfg.get("drift", {}).get("type", "zero")
    base_cost_type = base_functions_cfg.get("cost", {}).get("type", "zero")
    _emit_info(
        logger,
        f"  Base functions: drift={base_drift_type}, cost={base_cost_type}"
    )
    if exp_manager is not None:
        _emit_info(logger, f"  Output dir: {exp_manager.exp_dir}")


def _make_realtime_callback(
    *,
    exp_manager,
    config,
    sample_rho0,
    sample_rho1,
    mu_0,
    mu_1,
    mu_0_components=None,
    mu_1_components=None,
):
    """Create the realtime visualization callback."""
    return create_realtime_visualization_callback(
        exp_manager=exp_manager,
        config=config,
        sample_rho0=sample_rho0,
        sample_rho1=sample_rho1,
        mu_0=mu_0,
        mu_1=mu_1,
        mu_0_components=mu_0_components,
        mu_1_components=mu_1_components,
    )


def _build_summary(
    config: Config,
    args: argparse.Namespace,
    metrics: Mapping[str, Any],
    transport_stats: Mapping[str, float],
    interaction_stats: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, object]:
    """Assemble summary dictionary for metrics export."""
    if config.interaction_backend == "nn":
        hidden_dims_source = config.interaction_hidden_dims or config.hidden_dims
        interaction_hidden_dims = list(hidden_dims_source) if hidden_dims_source else None
        method_param = (
            f"H={hidden_dims_source[0]}"
            if hidden_dims_source
            else "H=?"
        )
    elif config.interaction_backend == "random_batch":
        interaction_hidden_dims = None
        method_param = f"M={config.random_batch_size}"
    else:
        interaction_hidden_dims = None
        method_param = None

    summary = {
        "method": {
            "interaction_backend": config.interaction_backend,
            "method_param": method_param,
            "interaction_hidden_dims": interaction_hidden_dims,
            "learn_interaction_drift": config.learn_interaction_drift,
            "learn_interaction_cost": config.learn_interaction_cost,
        },
        "config": {
            "n_agents": config.n_agents,
            "interaction_backend": config.interaction_backend,
            "random_batch_size": config.random_batch_size,
            "interaction_hidden_dims": interaction_hidden_dims,
            "learn_interaction_drift": config.learn_interaction_drift,
            "learn_interaction_cost": config.learn_interaction_cost,
            "n_outer_iterations": config.n_outer_iterations,
            "n_inner_steps": config.n_inner_steps,
            "n_interaction_steps": config.n_interaction_steps,
            "standardize_interaction_targets": config.standardize_interaction_targets,
            "interaction_input_scale": config.interaction_input_scale,
            "dt": config.dt,
            "learning_rate_value": config.learning_rate_value,
            "learning_rate_grad": config.learning_rate_grad,
            "learning_rate_interaction": config.learning_rate_interaction,
            "int_drift_weight": config.int_drift_weight,
            "int_cost_weight": config.int_cost_weight,
            "sigma": config.sigma,
            "training_seed": config.training_seed,
            "validation_seed": config.validation_seed,
            "visualization_seed": config.visualization_seed,
        },
        "timing": {
            "total_wall_time_sec": metrics.get("time_total", 0.0),
            "drift_step_time_sec_mean": (
                float(sum(metrics["time_drift_per_iteration"]))
                / len(metrics["time_drift_per_iteration"])
                if metrics.get("time_drift_per_iteration")
                else 0.0
            ),
            "interaction_training_time_sec": metrics.get("time_interaction_total", 0.0),
            "checkpoint_eval_time_sec_total": metrics.get(
                "time_checkpoint_eval_total", 0.0
            ),
            "interaction_step_time_sec_mean": (
                float(sum(metrics["time_interaction_per_iteration"]))
                / len(metrics["time_interaction_per_iteration"])
                if metrics.get("time_interaction_per_iteration")
                else 0.0
            ),
            "random_batch_sampling_time_sec_total": (
                metrics.get("time_random_batch_drift_sampling_total", 0.0)
                + metrics.get("time_random_batch_cost_sampling_total", 0.0)
            ),
            "random_batch_eval_time_sec_total": (
                metrics.get("time_random_batch_drift_eval_total", 0.0)
                + metrics.get("time_random_batch_cost_eval_total", 0.0)
            ),
            "random_batch_drift_sampling_time_sec_total": metrics.get(
                "time_random_batch_drift_sampling_total", 0.0
            ),
            "random_batch_drift_eval_time_sec_total": metrics.get(
                "time_random_batch_drift_eval_total", 0.0
            ),
            "random_batch_drift_call_count": metrics.get(
                "random_batch_drift_call_count", 0.0
            ),
            "random_batch_cost_sampling_time_sec_total": metrics.get(
                "time_random_batch_cost_sampling_total", 0.0
            ),
            "random_batch_cost_eval_time_sec_total": metrics.get(
                "time_random_batch_cost_eval_total", 0.0
            ),
            "random_batch_cost_call_count": metrics.get(
                "random_batch_cost_call_count", 0.0
            ),
            "nn_drift_inference_time_sec_total": metrics.get(
                "time_nn_drift_inference_total", 0.0
            ),
            "nn_drift_call_count": metrics.get("nn_drift_call_count", 0.0),
            "nn_cost_inference_time_sec_total": metrics.get(
                "time_nn_cost_inference_total", 0.0
            ),
            "nn_cost_call_count": metrics.get("nn_cost_call_count", 0.0),
            "nn_inference_time_sec_total": (
                metrics.get("time_nn_drift_inference_total", 0.0)
                + metrics.get("time_nn_cost_inference_total", 0.0)
            ),
            "exact_drift_eval_time_sec_total": metrics.get(
                "time_exact_drift_eval_total", 0.0
            ),
            "exact_drift_call_count": metrics.get("exact_drift_call_count", 0.0),
            "exact_cost_eval_time_sec_total": metrics.get(
                "time_exact_cost_eval_total", 0.0
            ),
            "exact_cost_call_count": metrics.get("exact_cost_call_count", 0.0),
            "exact_interaction_eval_time_sec_total": (
                metrics.get("time_exact_drift_eval_total", 0.0)
                + metrics.get("time_exact_cost_eval_total", 0.0)
            ),
        },
        "final_losses": {
            "loss_ipf_forward": metrics["loss_ipf_forward"][-1],
            "loss_ipf_backward": metrics["loss_ipf_backward"][-1],
            "loss_td_forward": metrics["loss_td_forward"][-1],
            "loss_td_backward": metrics["loss_td_backward"][-1],
            "loss_fk_forward": metrics["loss_fk_forward"][-1],
            "loss_fk_backward": metrics["loss_fk_backward"][-1],
            "loss_boundary_forward": metrics["loss_boundary_forward"][-1],
            "loss_boundary_backward": metrics["loss_boundary_backward"][-1],
        },
        "initial_losses": {
            "loss_ipf_forward": metrics["loss_ipf_forward"][0],
            "loss_ipf_backward": metrics["loss_ipf_backward"][0],
            "loss_td_forward": metrics["loss_td_forward"][0],
            "loss_td_backward": metrics["loss_td_backward"][0],
            "loss_fk_forward": metrics["loss_fk_forward"][0],
            "loss_fk_backward": metrics["loss_fk_backward"][0],
            "loss_boundary_forward": metrics["loss_boundary_forward"][0],
            "loss_boundary_backward": metrics["loss_boundary_backward"][0],
        },
    }
    analytical_keys = [
        "loss_ipf_forward_analytical",
        "loss_ipf_backward_analytical",
        "loss_td_forward_analytical",
        "loss_td_backward_analytical",
        "loss_fk_forward_analytical",
        "loss_fk_backward_analytical",
        "loss_boundary_forward_analytical",
        "loss_boundary_backward_analytical",
    ]
    analytical_summary = {
        key: metrics[key][-1] for key in analytical_keys if metrics.get(key)
    }
    if analytical_summary:
        summary["analytical_losses"] = analytical_summary
    summary["final_total_losses"] = {
        "forward_ipf_td_fk": (
            summary["final_losses"]["loss_ipf_forward"]
            + summary["final_losses"]["loss_td_forward"]
            + summary["final_losses"]["loss_fk_forward"]
        ),
        "backward_ipf_td_fk": (
            summary["final_losses"]["loss_ipf_backward"]
            + summary["final_losses"]["loss_td_backward"]
            + summary["final_losses"]["loss_fk_backward"]
        ),
        "bidirectional_ipf_td_fk": (
            summary["final_losses"]["loss_ipf_forward"]
            + summary["final_losses"]["loss_td_forward"]
            + summary["final_losses"]["loss_fk_forward"]
            + summary["final_losses"]["loss_ipf_backward"]
            + summary["final_losses"]["loss_td_backward"]
            + summary["final_losses"]["loss_fk_backward"]
        ),
        "forward_with_boundary": (
            summary["final_losses"]["loss_ipf_forward"]
            + summary["final_losses"]["loss_td_forward"]
            + summary["final_losses"]["loss_fk_forward"]
            + summary["final_losses"]["loss_boundary_forward"]
        ),
        "backward_with_boundary": (
            summary["final_losses"]["loss_ipf_backward"]
            + summary["final_losses"]["loss_td_backward"]
            + summary["final_losses"]["loss_fk_backward"]
            + summary["final_losses"]["loss_boundary_backward"]
        ),
        "bidirectional_with_boundary": (
            summary["final_losses"]["loss_ipf_forward"]
            + summary["final_losses"]["loss_td_forward"]
            + summary["final_losses"]["loss_fk_forward"]
            + summary["final_losses"]["loss_boundary_forward"]
            + summary["final_losses"]["loss_ipf_backward"]
            + summary["final_losses"]["loss_td_backward"]
            + summary["final_losses"]["loss_fk_backward"]
            + summary["final_losses"]["loss_boundary_backward"]
        ),
    }
    if interaction_stats:
        summary["interaction_errors"] = interaction_stats
    summary.update(transport_stats)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Option B: Learned interactions
    parser.add_argument(
        "--learn-interaction-drift",
        dest="learn_interaction_drift",
        action="store_true",
        help="Enable learning of the drift interaction network.",
    )
    parser.add_argument(
        "--learn-interaction-cost",
        dest="learn_interaction_cost",
        action="store_true",
        help="Enable learning of the cost interaction network.",
    )
    parser.set_defaults(
        interaction_backend=None,
        learn_interaction_drift=None,
        learn_interaction_cost=None,
        random_batch_size=None,
        evaluate_interaction_cpu=None,
        resample_interaction_sample=None,
        interaction_query_chunk_size=None,
        interaction_ref_chunk_size=None,
        interaction_time_block_size=None,
        use_multistep_td=None,
        td_bootstrap_from_target=None,
        use_time_embedding=None,
        time_input_mode=None,
        use_time_index_normalization=None,
    )
    parser.add_argument(
        "--interaction-backend",
        type=str,
        default=None,
        choices=["exact", "nn", "random_batch"],
        help="Interaction approximation backend. If omitted, inferred from learn-interaction flags.",
    )
    parser.add_argument(
        "--evaluate-interaction-cpu",
        action="store_true",
        default=None,
        help="Run analytical interaction evaluation on CPU to reduce GPU memory usage.",
    )
    parser.add_argument(
        "--no-evaluate-interaction-cpu",
        dest="evaluate_interaction_cpu",
        action="store_false",
        default=None,
        help="Run analytical interaction evaluation on the configured device.",
    )
    parser.add_argument(
        "--random-batch-size",
        type=int,
        default=None,
        help="Reference subset size M for random-batch interaction evaluation.",
    )
    parser.add_argument(
        "--interaction-steps",
        type=int,
        default=None,
        help="Interaction learning steps per cycle (M_int). If omitted, uses Config default.",
    )
    parser.add_argument(
        "--standardize-interaction-targets",
        dest="standardize_interaction_targets",
        action="store_true",
        default=None,
        help=(
            "Standardize learned interaction targets per output component and "
            "unstandardize network outputs during simulation."
        ),
    )
    parser.add_argument(
        "--no-standardize-interaction-targets",
        dest="standardize_interaction_targets",
        action="store_false",
        default=None,
        help="Disable learned interaction target standardization.",
    )
    parser.add_argument(
        "--interaction-input-scale",
        type=float,
        default=None,
        help="Divide learned interaction-network spatial inputs by this positive scale.",
    )
    parser.add_argument(
        "--interaction-query-chunk-size",
        type=int,
        default=None,
        help="Chunk size over query particles when computing analytical interaction labels.",
    )
    parser.add_argument(
        "--interaction-ref-chunk-size",
        type=int,
        default=None,
        help="Chunk size over reference particles when computing analytical interaction labels.",
    )
    parser.add_argument(
        "--interaction-time-block-size",
        type=int,
        default=None,
        help="Number of time slices to batch together for exact time-local analytical interactions.",
    )
    parser.add_argument(
        "--resample-interaction-sample",
        action="store_true",
        help="Resample a fresh trajectory and recompute interaction labels at each interaction step (Algorithm {alg:fourstep_alt_resample}).",
    )
    parser.add_argument(
        "--interaction-hidden-dims",
        type=int,
        nargs="+",
        default=None,
        help="Hidden layer sizes for the learned interaction network space branch.",
    )
    parser.add_argument(
        "--interaction-t-hidden-dims",
        type=int,
        nargs="+",
        default=None,
        help="Hidden layer sizes for the learned interaction network time branch (default: match space branch).",
    )
    parser.add_argument(
        "--interaction-activation",
        type=str,
        default=None,
        help="Activation override for interaction networks (defaults to main activation).",
    )
    parser.add_argument(
        "--lr-interaction",
        type=float,
        default=None,
        help="Learning rate for interaction networks. Required when interaction learning is enabled.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device for computation. If omitted, auto-detection is used.",
    )
    parser.add_argument(
        "--drift-type",
        type=str,
        default=None,
        choices=[
            "mean_reversion",
            "zero",
            "gaussian",
            "nonstationary_anisotropic",
            "attraction_repulsion",
            "coulomb",
            "laplacian",
            "bounded_confidence",
            "opinion",
        ],
        help="Type of drift interaction (analytical). If omitted, uses Config default.",
    )
    parser.add_argument(
        "--cost-type",
        type=str,
        default=None,
        choices=[
            "zero",
            "quadratic_deviation",
            "gaussian_conv",
            "gaussian_conv_mean",
            "pedestrian_visual",
            "mean_deviation",
            "quadratic_potential",
            "coulomb",
            "local_exclusion",
            "entropy",
            "congestion",
            "flocking_alignment",
        ],
        help="Type of cost interaction (analytical). If omitted, uses Config default.",
    )
    # New unified parameter naming: --{scope}-{component}-{param}
    # Interaction drift parameters
    parser.add_argument(
        "--int-drift-weight",
        type=float,
        default=None,
        help="Weight for interaction drift (mean_reversion, etc.).",
    )
    parser.add_argument(
        "--int-drift-sigma",
        type=float,
        default=None,
        help="Sigma for interaction drift (gaussian, nonstationary_anisotropic, laplacian).",
    )
    parser.add_argument(
        "--int-drift-strength",
        type=float,
        default=None,
        help="Strength for interaction drift (nonstationary_anisotropic alpha, coulomb, attraction_repulsion).",
    )
    # Interaction cost parameters
    parser.add_argument(
        "--int-cost-weight",
        type=float,
        default=None,
        help="Weight for interaction cost (quadratic_deviation, etc.).",
    )
    parser.add_argument(
        "--int-cost-sigma",
        type=float,
        default=None,
        help=(
            "Range scale for interaction cost "
            "(gaussian_conv, gaussian_conv_mean, pedestrian_visual)."
        ),
    )
    parser.add_argument(
        "--int-cost-strength",
        type=float,
        default=None,
        help="Strength for interaction cost (coulomb, local_exclusion).",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=None,
        help="Number of agents (particles) for Mean Field approximation.",
    )
    parser.add_argument(
        "--n-agents-test",
        type=int,
        default=None,
        help="Number of particles for evaluation/plots (defaults to n_agents).",
    )
    # Training parameters
    parser.add_argument(
        "--n-outer-iterations",
        type=int,
        default=None,
        help="Number of outer IPF iterations. If omitted, uses Config default.",
    )
    parser.add_argument(
        "--n-inner-steps",
        type=int,
        default=None,
        help="Number of gradient steps per iteration/stage. If omitted, uses Config default.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Time step. If omitted, uses Config default.",
    )
    parser.add_argument(
        "--lr-value",
        type=float,
        default=None,
        help="Learning rate for value networks (Y/Ŷ). If omitted, uses Config default.",
    )
    parser.add_argument(
        "--lr-grad",
        type=float,
        default=None,
        help="Learning rate for gradient networks (Z/Ẑ). If omitted, uses Config default.",
    )
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=None,
        help="Exponential LR decay per outer iteration. If omitted, uses Config default.",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default=None,
        choices=["exponential", "step", "none"],
        help="Scheduler type for FBSDE optimizers. If omitted, uses Config default.",
    )
    parser.add_argument(
        "--lr-step-size",
        type=int,
        default=None,
        help="Step size for StepLR scheduler (used when --lr-scheduler-type=step).",
    )
    parser.add_argument(
        "--optimizer-type",
        type=str,
        default=None,
        choices=["adam", "adamw", "sgd", "rmsprop", "adagrad"],
        help="Optimizer for FBSDE networks. If omitted, uses Config default.",
    )
    parser.add_argument(
        "--optimizer-momentum",
        type=float,
        default=None,
        help="Momentum for optimizers that support it (e.g., SGD/RMSprop).",
    )
    parser.add_argument(
        "--optimizer-weight-decay",
        type=float,
        default=None,
        help="Weight decay applied to FBSDE optimizers. If omitted, uses Config default.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Gradient clipping threshold. If omitted, uses Config default.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Diffusion coefficient σ. If omitted, uses Config default.",
    )

    parser.add_argument(
        "--lambda-ipf", type=float, default=None, help="Weight for L_IPF. If omitted, uses Config default."
    )
    parser.add_argument(
        "--lambda-td", type=float, default=None, help="Weight for L_TD. If omitted, uses Config default."
    )
    parser.add_argument(
        "--lambda-fk",
        type=float,
        default=None,
        help="Weight for L_FK. If omitted, uses Config default.",
    )
    parser.add_argument(
        "--lambda-boundary",
        type=float,
        default=None,
        help="Weight for boundary loss. If omitted, uses Config default.",
    )
    parser.add_argument(
        "--td-horizon",
        type=int,
        default=None,
        help="Multi-step TD horizon. If omitted, uses Config default.",
    )
    parser.add_argument(
        "--use-multistep-td",
        action="store_true",
        help="Enable DeepGSB-style multi-step TD loss.",
    )
    parser.add_argument(
        "--td-bootstrap-from-target",
        action="store_true",
        help="Enable TD bootstrapping from target (default stops gradients).",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=None,
        help="Hidden layer sizes for Value/Gradient networks (space-separated ints). "
        "If omitted, uses Config defaults or values from --problem-config.",
    )
    parser.add_argument(
        "--t-hidden-dims",
        type=int,
        nargs="*",
        default=None,
        help="Hidden dimensions for toy network time branch (space-separated ints).",
    )
    parser.add_argument(
        "--fk-loss-norm",
        type=str,
        default=None,
        choices=["l1", "l2", "huber"],
        help="Norm used in FK loss / grad matching. If omitted, uses Config default.",
    )
    parser.add_argument(
        "--detach-interaction-drift-divergence",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Detach learned interaction drift in divergence terms. "
            "This keeps f_hat values in the drift but removes derivatives of f_hat "
            "from IPF/TD divergence calculations. If omitted, uses Config default."
        ),
    )
    parser.add_argument(
        "--use-time-embedding",
        action="store_true",
        help="Enable sinusoidal time embedding input.",
    )
    parser.add_argument(
        "--time-input-mode",
        type=str,
        default=None,
        choices=["raw", "normalized"],
        help="Time preprocessing mode for neural networks. If omitted, uses Config default or problem config.",
    )
    parser.add_argument(
        "--time-embed-dim",
        type=int,
        default=None,
        help="Time embedding dimension. If omitted, uses Config default.",
    )
    parser.add_argument(
        "--use-time-index-normalization",
        action="store_true",
        help="Normalize time indices before feeding networks.",
    )
    parser.add_argument(
        "--time-index-interval",
        type=float,
        default=None,
        help="Interval applied during time index normalization. If omitted, uses Config default.",
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "tanh", "elu", "silu"],
        default=None,
        help="Activation function for neural networks. If omitted, uses Config default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. If omitted, uses Config default.",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=None,
        help="Training-phase seed. If omitted, falls back to --seed.",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=None,
        help="Validation/checkpoint-eval seed. If omitted, falls back to --seed.",
    )
    parser.add_argument(
        "--visualization-seed",
        type=int,
        default=None,
        help="Visualization/plot seed. If omitted, falls back to --seed.",
    )
    parser.add_argument(
        "--T",
        type=float,
        default=None,
        help="Time horizon for the FBSDE. If omitted, uses Config default.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="Iterations between metric logs. If omitted, uses Config default.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Iterations between checkpoints. If omitted, uses Config default.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to experiment directory to resume from. "
        "Automatically loads the latest checkpoint and continues training.",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default=None,
        help="Optional path to save summary metrics (JSON).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Disable metric/trajectory plots (useful for sweeps).",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=None,
        help="EMA decay (0 disables EMA targets). If omitted, uses Config default.",
    )
    parser.add_argument(
        "--ema-start-step",
        type=int,
        default=None,
        help="Global optimizer step to start EMA updates. If omitted, uses Config default.",
    )
    parser.add_argument(
        "--ema-update-interval",
        type=int,
        default=None,
        help="Number of optimizer steps between EMA updates. If omitted, uses Config default.",
    )
    parser.add_argument(
        "--no-experiment-dir",
        action="store_true",
        help="Disable experiment directory management (use default outputs/).",
    )
    parser.add_argument(
        "--problem-config",
        type=str,
        required=True,
        help="Path to problem configuration JSON (required). "
        "CLI arguments can still override config values.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional scenario name override for experiment directory naming.",
    )
    args = parser.parse_args()
    # Attach parser to args for later access to defaults (for problem config override logic)
    args._parser = parser

    return args


def run_training(
    config: Config,
    args: argparse.Namespace,
    mu_0: torch.Tensor,
    mu_1: torch.Tensor,
    logger: Optional[logging.Logger] = None,
    problem_cfg: Optional[dict] = None,
    resume_checkpoint: Optional[Path] = None,
    start_iteration: int = 0,
    exp_manager: Optional[ExperimentManager] = None,
    sample_rho0=None,
    sample_rho1=None,
    log_p0=None,
    log_p1=None,
    mu_0_components=None,
    mu_1_components=None,
    base_drift_fn=None,
    base_cost_fn=None,
    obstacle_config=None,
    xlim=None,
    ylim=None,
) -> Dict[str, float]:
    """Run training with flexible distribution and base function setup.

    Args:
        config: Training configuration
        args: CLI arguments
        mu_0, mu_1: Distribution means
        resume_checkpoint: Optional checkpoint to resume from
        start_iteration: Starting iteration (for resume)
        exp_manager: Optional experiment manager
        sample_rho0, sample_rho1: Optional sampling functions
            (if None, created from Gaussian)
        log_p0, log_p1: Optional log probability functions
            (if None, created from Gaussian)
        mu_0_components, mu_1_components: Optional GMM components
            for visualization
        base_drift_fn, base_cost_fn: Optional base functions
            (None means no explicit base function)
        obstacle_config: Optional obstacle configuration for visualization
        xlim, ylim: Optional axis limits for trajectory plots
    """
    interaction_enabled = config.learn_interaction

    # Setup experiment management
    if exp_manager is None and not args.no_experiment_dir:
        scenario_name = (
            args.name
            if args.name
            else (problem_cfg["name"] if problem_cfg is not None else "problem_config")
        )
        exp_manager = ExperimentManager(
            config=config,
            scenario_name=scenario_name,
            method=config.interaction_backend,
            auto_timestamp=True,
        )
    # Save config at the very beginning (before training starts)
    if exp_manager:
        exp_manager.save_config(config, cli_args=args)

    # Create interactions (always needed for drift_net and cost_net)
    (
        drift_net,
        cost_net,
        f_interaction_fn,
        F_interaction_fn,
        analytical_drift_module,
        analytical_cost_module,
    ) = create_interactions(config)

    solver = FBSDESolver(
        config=config,
        drift_net=drift_net,
        cost_net=cost_net,
        base_drift_fn=base_drift_fn,
        base_cost_fn=base_cost_fn,
    )

    trainable_params = sum(
        p.numel()
        for net in [
            solver.value_net,
            solver.value_hat_net,
            solver.grad_net,
            solver.grad_hat_net,
        ]
        for p in net.parameters()
        if p.requires_grad
    )
    if sample_rho0 is None or sample_rho1 is None:
        raise ValueError(
            "run_training requires sample_rho0 and sample_rho1. "
            "Callers must provide explicit sampling functions."
        )
    if log_p0 is None or log_p1 is None:
        raise ValueError(
            "run_training requires log_p0 and log_p1. "
            "Callers must provide explicit log-density functions."
        )

    # Setup callbacks for real-time visualization
    callbacks = []
    if not args.skip_plots and exp_manager:
        callback = _make_realtime_callback(
            exp_manager=exp_manager,
            config=config,
            sample_rho0=sample_rho0,
            sample_rho1=sample_rho1,
            mu_0=mu_0,
            mu_1=mu_1,
            mu_0_components=mu_0_components,
            mu_1_components=mu_1_components,
        )
        callbacks.append(callback)

    trainer = Trainer(
        config=config,
        solver=solver,
        logger=logger,
        log_p0_fn=log_p0,
        log_p1_fn=log_p1,
        f_interaction_fn=f_interaction_fn,
        F_interaction_fn=F_interaction_fn,
        analytical_drift_module=analytical_drift_module,
        analytical_cost_module=analytical_cost_module,
        checkpoint_callbacks=callbacks,
    )

    # Load checkpoint if resuming
    if resume_checkpoint is not None:
        logger.info("")
        logger.info("Loading Checkpoint")
        loaded_iteration = trainer.load_checkpoint(str(resume_checkpoint))
        logger.info(
            f"Successfully loaded checkpoint from iteration {loaded_iteration}"
        )
    metrics = trainer.train(
        sample_rho0_fn=sample_rho0,
        sample_rho1_fn=sample_rho1,
        start_iteration=start_iteration,
    )

    summarize_metrics(metrics, logger=logger)
    if not args.skip_plots:
        plot_metrics(metrics, exp_manager, config)

    n_agents_vis = config.n_agents_test or config.n_agents or 256
    xs_forward, xs_backward = sample_visualization_trajectories(
        trainer,
        sample_rho0,
        sample_rho1,
        n_agents_vis,
        seed=config.visualization_seed,
    )
    if not args.skip_plots:
        plot_trajectories(
            xs_forward,
            xs_backward,
            mu_0,
            mu_1,
            exp_manager,
            obstacle_config=obstacle_config,
            xlim=xlim,
            ylim=ylim,
            mu_0_components=mu_0_components,
            mu_1_components=mu_1_components,
        )
        # Opinion-specific projections/histograms for dim > 2
        if problem_cfg and "opinion" in problem_cfg.get("problem_type", ""):
            plot_opinion_snapshots(
                xs_forward,
                xs_backward,
                solver.config.time_grid,
                exp_manager,
            )
            plot_opinion_directional_similarity(
                xs_forward,
                xs_backward,
                exp_manager,
            )

    transport_stats = evaluate_transport_quality(
        solver,
        sample_rho0,
        sample_rho1,
        config,
        n_test=config.n_agents_test or 1000,
        logger=logger,
        seed=config.validation_seed,
    )
    logger.info("")
    logger.info("Evaluating Interaction Approximation...")
    interaction_stats = evaluate_interaction_approximation(
        solver=solver,
        config=config,
        analytical_drift_module=analytical_drift_module,
        analytical_cost_module=analytical_cost_module,
        xs_forward=xs_forward,
        xs_backward=xs_backward,
        logger=logger,
        seed=config.validation_seed,
    )

    trainer.save_checkpoint(len(metrics["loss_ipf_forward"]))

    # Save experiment metadata if using experiment management
    # (Already saved at the beginning)

    logger.info("")
    logger.info("Validation Complete")
    if exp_manager:
        logger.info(f"All results saved to: {exp_manager.exp_dir}")
    else:
        logger.info("Results saved under outputs/figures and outputs/checkpoints.")

    summary = _build_summary(
        config,
        args,
        metrics,
        transport_stats,
        interaction_stats,
    )

    if exp_manager:
        exp_manager.save_metrics(summary)

    if args.metrics_output:
        with open(args.metrics_output, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary metrics saved to {args.metrics_output}")

    return summary


def main() -> None:
    args = parse_args()
    logger: Optional[logging.Logger] = None

    # Resolve device once at the start
    # (handles auto-detection and prints CUDA mapping)
    from nlmfsp import resolve_device

    args.device = resolve_device(args.device or "auto", print_info=True)

    # Load problem configuration (required)
    problem_cfg = load_problem_config(Path(args.problem_config))

    parser = args._parser  # Access parser attached in parse_args()

    # Handle resume mode
    if args.resume:
        # Load existing experiment
        exp_manager = ExperimentManager.from_existing(Path(args.resume))

        # Find latest checkpoint
        checkpoint_path = exp_manager.find_latest_checkpoint()
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No checkpoints found in {exp_manager.exp_dir / 'checkpoints'}"
            )

        # Load checkpoint to get config
        checkpoint = torch.load(
            checkpoint_path, map_location=args.device, weights_only=False
        )
        config = checkpoint["config"]
        ensure_config_time_input_fields(config)
        start_iteration = checkpoint["iteration"]
        if "dim" in problem_cfg:
            config.dim = int(problem_cfg["dim"])
        config.problem_config_path = str(Path(args.problem_config).resolve())

        logger = create_trainer_logger(config, name="nlmfsp.run")
        logger.info("")
        logger.info("Resume Mode")
        logger.info(f"Found checkpoint: {checkpoint_path}")
        logger.info(f"  Original iterations: {config.n_outer_iterations}")
        logger.info(f"  Completed: {start_iteration}")

        # If CLI explicitly specifies n_outer_iterations, use it as the new total.
        if args.n_outer_iterations is not None:
            if args.n_outer_iterations < start_iteration:
                raise ValueError(
                    f"--n-outer-iterations ({args.n_outer_iterations}) must be >= completed iterations ({start_iteration}) in resume mode."
                )
            config.n_outer_iterations = args.n_outer_iterations
            logger.info(
                f"  Overridden total (--n-outer-iterations): {config.n_outer_iterations}"
            )
        remaining = config.n_outer_iterations - start_iteration
        logger.info(f"  Remaining: {remaining}")

        (
            mu_0,
            mu_1,
            sample_rho0,
            sample_rho1,
            mu_0_components,
            mu_1_components,
            obstacle_config,
            xlim_tuple,
            ylim_tuple,
            base_drift_fn,
            base_cost_fn,
            log_p0,
            log_p1,
        ) = setup_problem_artifacts(config, problem_cfg)
        print_problem_overview(problem_cfg, logger)
        print_distribution_summary(problem_cfg, logger)
        print_obstacle_summary(obstacle_config, logger)
        log_configuration(config, logger)
        log_memory_profile(config, logger)
        print_runtime_setup(config, problem_cfg, exp_manager, logger)

        try:
            summary = run_training(
                config,
                args,
                mu_0,
                mu_1,
                logger=logger,
                problem_cfg=problem_cfg,
                resume_checkpoint=checkpoint_path,
                start_iteration=start_iteration,
                exp_manager=exp_manager,
                sample_rho0=sample_rho0,
                sample_rho1=sample_rho1,
                log_p0=log_p0,
                log_p1=log_p1,
                mu_0_components=mu_0_components,
                mu_1_components=mu_1_components,
                base_drift_fn=base_drift_fn,
                base_cost_fn=base_cost_fn,
                obstacle_config=obstacle_config,
                xlim=xlim_tuple,
                ylim=ylim_tuple,
            )
        except Exception:
            logger.exception("Training failed")
            raise

    else:
        from nlmfsp import apply_problem_config_overrides

        # Priority: CLI explicit > problem JSON > CLI default(None)
        args = apply_problem_config_overrides(args, problem_cfg, parser)

        apply_interaction_config_overrides(args, problem_cfg, parser)

        config = build_config(args)
        if "dim" in problem_cfg:
            config.dim = int(problem_cfg["dim"])
        config.problem_config_path = str(Path(args.problem_config).resolve())

        if not args.no_experiment_dir:
            scenario_name = args.name if args.name else problem_cfg["name"]
            exp_manager = ExperimentManager(
                config=config,
                scenario_name=scenario_name,
                method=config.interaction_backend,
                auto_timestamp=True,
            )
        else:
            exp_manager = None

        logger = create_trainer_logger(config, name="nlmfsp.run")

        (
            mu_0,
            mu_1,
            sample_rho0,
            sample_rho1,
            mu_0_components,
            mu_1_components,
            obstacle_config,
            xlim_tuple,
            ylim_tuple,
            base_drift_fn,
            base_cost_fn,
            log_p0,
            log_p1,
        ) = setup_problem_artifacts(config, problem_cfg)
        print_problem_overview(problem_cfg, logger)
        log_configuration(config, logger)
        print_distribution_summary(problem_cfg, logger)
        print_obstacle_summary(obstacle_config, logger)

        base_functions_cfg = problem_cfg.get("base_functions", {})
        if base_functions_cfg.get("drift", {}).get("type") == "constant":
            drift_value = base_functions_cfg["drift"]["value"]
            logger.info("")
            logger.info(f"Base drift override: constant {drift_value}")

        log_memory_profile(config, logger)
        print_runtime_setup(config, problem_cfg, exp_manager, logger)

        if "visualization" in problem_cfg and not xlim_tuple and not ylim_tuple:
            vis_cfg = problem_cfg["visualization"]
            xlim_tuple = tuple(vis_cfg["xlim"]) if "xlim" in vis_cfg else xlim_tuple
            ylim_tuple = tuple(vis_cfg["ylim"]) if "ylim" in vis_cfg else ylim_tuple

        try:
            summary = run_training(
                config=config,
                args=args,
                mu_0=mu_0,
                mu_1=mu_1,
                logger=logger,
                problem_cfg=problem_cfg,
                exp_manager=exp_manager,
                sample_rho0=sample_rho0,
                sample_rho1=sample_rho1,
                log_p0=log_p0,
                log_p1=log_p1,
                mu_0_components=mu_0_components,
                mu_1_components=mu_1_components,
                base_drift_fn=base_drift_fn,
                base_cost_fn=base_cost_fn,
                obstacle_config=obstacle_config,
                xlim=xlim_tuple,
                ylim=ylim_tuple,
            )
        except Exception:
            logger.exception("Training failed")
            raise

    summary["problem_type"] = problem_cfg["problem_type"]
    summary["problem_name"] = problem_cfg["name"]

    if args.metrics_output is None:
        if logger is not None:
            logger.info("Summary (JSON):")
            logger.info(json.dumps(summary, indent=2))
        else:
            print("Summary (JSON):")
            print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
