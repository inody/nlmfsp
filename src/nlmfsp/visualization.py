"""Visualization utilities for training metrics and trajectories.

This module provides plotting functions for:
- Training metrics (losses, learning rates)
- Particle trajectories (forward/backward)
- Real-time visualization during training
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.patches import Circle, Ellipse

from .config import Config
from .plot_style import apply_paper_style
from .rng_utils import seed_offset, torch_seed_context

if TYPE_CHECKING:
    from .experiment_manager import ExperimentManager


apply_paper_style()


def _emit_info(message: str) -> None:
    """Log visualization info through the standard run/trainer loggers."""
    logger = logging.getLogger("nlmfsp.run")
    if not logger.handlers:
        logger = logging.getLogger("nlmfsp.trainer")
    if logger.handlers:
        logger.info(message)
    else:
        print(message)


def _emit_warning(message: str) -> None:
    """Log visualization warnings through the standard run/trainer loggers."""
    logger = logging.getLogger("nlmfsp.run")
    if not logger.handlers:
        logger = logging.getLogger("nlmfsp.trainer")
    if logger.handlers:
        logger.warning(message)
    else:
        print(message)


def _get_figure_dir(exp_manager: Optional["ExperimentManager"]) -> Path:
    """Resolve the figure output directory."""
    if exp_manager is not None:
        figures_dir = exp_manager.exp_dir / "figures"
    else:
        figures_dir = Path("outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _save_figure_outputs(
    fig: plt.Figure,
    *,
    exp_manager: Optional["ExperimentManager"],
    filenames: Sequence[str],
    dpi: int = 150,
) -> list[Path]:
    """Save one figure to one or more filenames and log the result."""
    figures_dir = _get_figure_dir(exp_manager)
    saved_paths: list[Path] = []
    for filename in filenames:
        save_path = figures_dir / filename
        save_kwargs = {"bbox_inches": "tight"}
        if save_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            save_kwargs["dpi"] = dpi
        fig.savefig(save_path, **save_kwargs)
        saved_paths.append(save_path)
    if not saved_paths:
        return saved_paths
    if len(saved_paths) == 1:
        _emit_info(f"  Saved: {saved_paths[0]}")
    else:
        joined = " and ".join(str(path) for path in saved_paths)
        _emit_info(f"  Saved: {joined}")
    return saved_paths


def sample_visualization_trajectories(
    trainer,
    sample_rho0: Callable[[int], torch.Tensor],
    sample_rho1: Callable[[int], torch.Tensor],
    num_vis: int,
    *,
    seed_forward: Optional[int] = None,
    seed_backward: Optional[int] = None,
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample forward/backward trajectories for visualization."""
    if seed is not None:
        with torch_seed_context(seed_offset(seed, 0)):
            x0_vis = sample_rho0(num_vis)
        with torch_seed_context(seed_offset(seed, 1)):
            x1_vis = sample_rho1(num_vis)
        seed_forward = seed_offset(seed, 2) or seed_forward
        seed_backward = seed_offset(seed, 3) or seed_backward
    else:
        x0_vis = sample_rho0(num_vis)
        x1_vis = sample_rho1(num_vis)
    with torch.no_grad():
        xs_forward, _, _ = trainer.solver.sample_forward_trajectory(
            x0_vis, seed=seed_forward
        )
        xs_backward, _, _ = trainer.solver.sample_backward_trajectory(
            x1_vis, seed=seed_backward
        )
    return xs_forward, xs_backward


def _save_metrics_figure(
    metrics: Dict[str, list],
    config: Config,
    exp_manager: "ExperimentManager",
    filenames: Sequence[str],
) -> None:
    """Create and save metrics plots."""
    fig = _create_metrics_plot(metrics, config)
    _save_figure_outputs(
        fig,
        exp_manager=exp_manager,
        filenames=filenames,
    )
    plt.close(fig)


def _save_trajectory_figure(
    *,
    exp_manager: Optional["ExperimentManager"],
    xs_forward: torch.Tensor,
    xs_backward: torch.Tensor,
    mu_0: torch.Tensor,
    mu_1: torch.Tensor,
    filenames: Sequence[str],
    obstacle_config: Optional[Dict] = None,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    mu_0_components: Optional[list] = None,
    mu_1_components: Optional[list] = None,
) -> None:
    """Create and save 2D trajectory plots."""
    fig = _create_trajectories_figure(
        xs_forward,
        xs_backward,
        mu_0,
        mu_1,
        obstacle_config=obstacle_config,
        xlim=xlim,
        ylim=ylim,
        mu_0_components=mu_0_components,
        mu_1_components=mu_1_components,
    )
    if fig is None:
        return
    _save_figure_outputs(
        fig,
        exp_manager=exp_manager,
        filenames=filenames,
    )
    plt.close(fig)


def _save_opinion_snapshots_figure(
    *,
    exp_manager: Optional["ExperimentManager"],
    xs_forward: torch.Tensor,
    xs_backward: torch.Tensor,
    time_grid: torch.Tensor,
    filenames: Sequence[str],
    seed: Optional[int] = None,
) -> None:
    """Create and save opinion PCA snapshot figure."""
    snap_fig = _create_opinion_snapshots_figure(
        xs_forward,
        xs_backward,
        time_grid,
        seed=seed,
    )
    if snap_fig is not None:
        _save_figure_outputs(
            snap_fig,
            exp_manager=exp_manager,
            filenames=filenames,
        )
        plt.close(snap_fig)


def _save_opinion_directional_similarity_figure(
    *,
    exp_manager: Optional["ExperimentManager"],
    xs_forward: torch.Tensor,
    xs_backward: torch.Tensor,
    filenames: Sequence[str],
    n_est: int = 5000,
    seed: Optional[int] = None,
) -> None:
    """Create and save opinion directional-similarity figure."""
    fig = _create_opinion_directional_similarity_figure(
        xs_forward,
        xs_backward,
        n_est=n_est,
        seed=seed,
    )
    if fig is None:
        return
    _save_figure_outputs(
        fig,
        exp_manager=exp_manager,
        filenames=filenames,
    )
    plt.close(fig)


def setup_problem_visualization(
    problem_cfg: Dict[str, object],
) -> Dict[str, Optional[tuple[float, float]]]:
    """Extract visualization metadata from problem config."""
    xlim: Optional[tuple[float, float]] = None
    ylim: Optional[tuple[float, float]] = None
    visualization_cfg = problem_cfg.get("visualization", {})
    if isinstance(visualization_cfg, dict) and visualization_cfg:
        xlim_vals = visualization_cfg.get("xlim")
        ylim_vals = visualization_cfg.get("ylim")
        if xlim_vals:
            xlim = (float(xlim_vals[0]), float(xlim_vals[1]))
        if ylim_vals:
            ylim = (float(ylim_vals[0]), float(ylim_vals[1]))
    return {"xlim": xlim, "ylim": ylim}


def _plot_obstacles(
    axes,
    obstacle_config: Dict,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
) -> None:
    """Plot obstacles on given axes based on configuration.

    Args:
        axes: List of matplotlib axes to plot on
        obstacle_config: Dict with 'type' and obstacle parameters
        xlim: Optional x-axis limits (used for V-neck plotting range)
        ylim: Optional y-axis limits (used for V-neck obstacle extent)
    """
    obstacle_type = obstacle_config.get("type")

    if obstacle_type == "multi_obstacle":
        # Circular obstacles
        shown_label = False
        for obs in obstacle_config["obstacles"]:
            center = obs["center"]
            radius = obs["radius"]
            mode = obs.get("mode", "hard")
            is_soft = mode == "soft"
            # Use same color scheme across scenarios (gray fill), keep dashed/dotted edge for soft
            fill_color = "gray"
            edge_style = ":" if is_soft else "--"
            label = "Obstacle" if not shown_label else ""
            shown_label = True
            # if is_soft:
            # label = f"{desc} (soft)"

            for ax in axes:
                circle = Circle(
                    center,
                    radius,
                    color=fill_color,
                    alpha=0.3,
                    zorder=1,
                    label=label if ax == axes[0] else "",
                )
                ax.add_patch(circle)
                circle_edge = Circle(
                    center,
                    radius,
                    color="black",
                    fill=False,
                    linestyle=edge_style,
                    linewidth=2,
                    zorder=2,
                )
                ax.add_patch(circle_edge)

    elif obstacle_type == "vneck_parabolic":
        # V-neck parabolic obstacle: coef*x^2 - y^2 < -c_sq
        # => y^2 > coef*x^2 + c_sq
        # => |y| > sqrt(coef*x^2 + c_sq)
        # This is the region OUTSIDE the V-shape (beyond the two parabolas)
        c_sq = obstacle_config.get("c_sq", 0.36)
        coef = obstacle_config.get("coef", 5.0)

        # Determine plotting range
        if xlim:
            x_min, x_max = xlim
        else:
            x_min, x_max = -8, 8

        if ylim:
            y_min, y_max = ylim
        else:
            y_min, y_max = -4, 4

        # Boundary curves: y = ± sqrt(coef*x^2 + c_sq)
        x_range = np.linspace(x_min, x_max, 400)
        y_boundary = np.sqrt(coef * x_range**2 + c_sq)

        for ax in axes:
            # Fill UPPER obstacle region (y > y_boundary)
            ax.fill_between(
                x_range, y_boundary, y_max, color="gray", alpha=0.3, zorder=1
            )
            # Fill LOWER obstacle region (y < -y_boundary)
            ax.fill_between(
                x_range,
                y_min,
                -y_boundary,
                color="gray",
                alpha=0.3,
                zorder=1,
                label="Obstacle" if ax == axes[0] else "",
            )
            # Draw boundary curves
            ax.plot(x_range, y_boundary, "k--", linewidth=2, zorder=2)
            ax.plot(x_range, -y_boundary, "k--", linewidth=2, zorder=2)

    elif obstacle_type == "stunnel_elliptic":
        # S-tunnel elliptic obstacles
        ellipses_config = obstacle_config.get("ellipses", [])

        shown_label = False
        for i, ell in enumerate(ellipses_config):
            center = ell["center"]
            a = ell["a"]
            b = ell["b"]
            c = ell["c"]

            # Ellipse: a*(x-cx)^2 + b*(y-cy)^2 = c
            # => (x-cx)^2/r_x^2 + (y-cy)^2/r_y^2 = 1
            # where r_x = sqrt(c/a), r_y = sqrt(c/b)
            width = 2 * np.sqrt(c / a)
            height = 2 * np.sqrt(c / b)

            for ax in axes:
                ellipse = Ellipse(
                    center,
                    width,
                    height,
                    color="gray",
                    alpha=0.3,
                    zorder=1,
                    label="Obstacle" if (ax == axes[0] and not shown_label) else "",
                )
                ax.add_patch(ellipse)
                shown_label = True
                ellipse_edge = Ellipse(
                    center,
                    width,
                    height,
                    color="black",
                    fill=False,
                    linestyle="--",
                    linewidth=2,
                    zorder=2,
                )
                ax.add_patch(ellipse_edge)


def _create_metrics_plot(
    metrics: Dict[str, list],
    config: Config,
) -> plt.Figure:
    """Create a metrics plot figure.

    Args:
        metrics: Dictionary containing training metrics.
        config: Config object for applying loss weights.

    Returns:
        Matplotlib figure object.
    """
    iterations_forward = list(range(1, len(metrics["loss_ipf_forward"]) + 1))
    iterations_backward = list(range(1, len(metrics["loss_ipf_backward"]) + 1))
    learn_interaction_drift = getattr(config, "learn_interaction_drift", False)
    learn_interaction_cost = getattr(config, "learn_interaction_cost", False)
    drift_losses = metrics.get("loss_interaction_drift", [])
    cost_losses = metrics.get("loss_interaction_cost", [])
    has_drift_metrics = any(abs(v) > 1e-9 for v in drift_losses)
    has_cost_metrics = any(abs(v) > 1e-9 for v in cost_losses)
    has_interaction_metrics = has_drift_metrics or has_cost_metrics

    # Check if interaction losses are present and non-zero
    has_interaction = (
        learn_interaction_drift or learn_interaction_cost or has_interaction_metrics
    )

    # Use 3x2 layout if interaction losses present, otherwise 2x2
    if has_interaction:
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Backward losses
    axes[0, 0].plot(
        iterations_backward, metrics["loss_ipf_backward"], "g-", label="L_IPF"
    )
    axes[0, 0].plot(
        iterations_backward, metrics["loss_td_backward"], "b-", label="L_TD"
    )
    axes[0, 0].plot(
        iterations_backward, metrics["loss_fk_backward"], "r-", label="L_FK"
    )
    axes[0, 0].plot(
        iterations_backward, metrics["loss_boundary_backward"], "m-", label="L_B"
    )
    axes[0, 0].set_title("Backward Step Losses")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Forward losses
    axes[0, 1].plot(
        iterations_forward, metrics["loss_ipf_forward"], "g-", label="L_IPF"
    )
    axes[0, 1].plot(iterations_forward, metrics["loss_td_forward"], "b-", label="L_TD")
    axes[0, 1].plot(iterations_forward, metrics["loss_fk_forward"], "r-", label="L_FK")
    axes[0, 1].plot(
        iterations_forward, metrics["loss_boundary_forward"], "m-", label="L_B"
    )
    axes[0, 1].set_title("Forward Step Losses")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Combined loss with Forward/Backward/Total breakdown
    forward_len = min(
        len(metrics["loss_ipf_forward"]),
        len(metrics["loss_td_forward"]),
        len(metrics["loss_fk_forward"]),
        len(metrics["loss_boundary_forward"]),
    )
    backward_len = min(
        len(metrics["loss_ipf_backward"]),
        len(metrics["loss_td_backward"]),
        len(metrics["loss_fk_backward"]),
        len(metrics["loss_boundary_backward"]),
    )
    combined_len = min(forward_len, backward_len)

    forward_loss = [
        config.lambda_ipf * metrics["loss_ipf_forward"][i]
        + config.lambda_td * metrics["loss_td_forward"][i]
        + config.lambda_fk * metrics["loss_fk_forward"][i]
        + config.lambda_boundary * metrics["loss_boundary_forward"][i]
        for i in range(forward_len)
    ][:combined_len]
    backward_loss = [
        config.lambda_ipf * metrics["loss_ipf_backward"][i]
        + config.lambda_td * metrics["loss_td_backward"][i]
        + config.lambda_fk * metrics["loss_fk_backward"][i]
        + config.lambda_boundary * metrics["loss_boundary_backward"][i]
        for i in range(backward_len)
    ][:combined_len]
    total_loss = [forward_loss[i] + backward_loss[i] for i in range(combined_len)]
    iterations_combined = list(range(1, combined_len + 1))

    axes[1, 0].plot(
        iterations_combined,
        forward_loss,
        "b-",
        label="Forward",
        linewidth=1.5,
        alpha=0.7,
    )
    axes[1, 0].plot(
        iterations_combined,
        backward_loss,
        "r-",
        label="Backward",
        linewidth=1.5,
        alpha=0.7,
    )
    axes[1, 0].plot(iterations_combined, total_loss, "k-", label="Total", linewidth=2)
    axes[1, 0].set_title("Combined Loss (Weighted)")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rates
    lr_ax = axes[1, 1]
    if metrics.get("lr_forward"):
        lr_forward_x = list(range(1, len(metrics["lr_forward"]) + 1))
        lr_ax.plot(lr_forward_x, metrics["lr_forward"], "b-", label="Forward Y")
    if metrics.get("lr_forward_grad"):
        lr_forward_grad_x = list(range(1, len(metrics["lr_forward_grad"]) + 1))
        lr_ax.plot(
            lr_forward_grad_x, metrics["lr_forward_grad"], "b--", label="Forward Z"
        )
    if metrics.get("lr_backward"):
        lr_backward_x = list(range(1, len(metrics["lr_backward"]) + 1))
        lr_ax.plot(lr_backward_x, metrics["lr_backward"], "r-", label=r"Backward $\hat{Y}$")
    if metrics.get("lr_backward_grad"):
        lr_backward_grad_x = list(range(1, len(metrics["lr_backward_grad"]) + 1))
        lr_ax.plot(
            lr_backward_grad_x,
            metrics["lr_backward_grad"],
            "r--",
            label=r"Backward $\hat{Z}$",
        )
    if metrics.get("lr_interaction") and any(v > 0 for v in metrics["lr_interaction"]):
        lr_interaction_x = list(range(1, len(metrics["lr_interaction"]) + 1))
        lr_ax.plot(
            lr_interaction_x, metrics["lr_interaction"], "g-", label="Interaction"
        )
    lr_ax.set_title("Learning Rates")
    lr_ax.set_xlabel("Iteration")
    lr_ax.set_ylabel("LR")
    lr_ax.legend()
    lr_ax.grid(True, alpha=0.3)

    # Interaction losses (Option B only)
    if has_interaction:
        interaction_ax = axes[2, 0]
        if learn_interaction_drift or has_drift_metrics:
            drift_x = list(range(1, len(drift_losses) + 1))
            interaction_ax.plot(
                drift_x,
                drift_losses,
                "c-",
                label="L_drift",
                linewidth=2,
            )
        if learn_interaction_cost or has_cost_metrics:
            cost_x = list(range(1, len(cost_losses) + 1))
            interaction_ax.plot(
                cost_x,
                cost_losses,
                "m-",
                label="L_cost",
                linewidth=2,
            )
        if interaction_ax.lines:
            interaction_ax.set_title("Interaction Learning Losses")
            interaction_ax.set_xlabel("Iteration")
            interaction_ax.set_ylabel("Loss")
            interaction_ax.legend()
            interaction_ax.grid(True, alpha=0.3)
        else:
            interaction_ax.axis("off")

        # Hide the extra subplot
        axes[2, 1].axis("off")

    plt.tight_layout()
    return fig


def plot_metrics(
    metrics: Dict[str, list],
    exp_manager: Optional[ExperimentManager] = None,
    config: Optional[Config] = None,
) -> None:
    """Plot training metrics including losses and learning rates.

    Args:
        metrics: Dictionary containing training metrics with keys:
            - loss_ipf_forward, loss_ipf_backward
            - loss_td_forward, loss_td_backward
            - loss_fk_forward, loss_fk_backward
            - loss_boundary_forward, loss_boundary_backward
            - loss_interaction_drift, loss_interaction_cost (optional)
            - lr_forward, lr_backward, lr_interaction
        exp_manager: Optional ExperimentManager for saving to
            experiment directory.
            If None, saves to outputs/figures/.
        config: Optional Config object for applying loss weights.
            If None, uses default Config values.
    """
    # Use provided config or default
    if config is None:
        config = SimpleNamespace(
            lambda_ipf=1.0,
            lambda_td=1.0,
            lambda_fk=0.1,
            lambda_boundary=0.0,
            learn_interaction_drift=False,
            learn_interaction_cost=False,
        )

    _save_metrics_figure(
        metrics,
        config,
        exp_manager=exp_manager,
        filenames=["validate_generalized_schrodinger_bridge_metrics.png"],
    )


def _deepgsb_style_pca_project(
    xs_forward: torch.Tensor,
    xs_backward: torch.Tensor,
    n_components: int = 2,
    max_pca_samples: int = 200,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project trajectories with DeepGSB-style PCA basis selection.

    DeepGSB computes PCA basis from the terminal forward states, then projects
    both forward/backward trajectories using the same basis and centering.
    """
    if n_components != 2:
        raise ValueError("Only n_components=2 is supported for opinion snapshots.")

    # Inputs are (T+1, B, D). Convert to (B, T+1, D) to mirror DeepGSB layout.
    xs_f = xs_forward.transpose(0, 1)
    xs_b = xs_backward.transpose(0, 1)
    batch, _, dim = xs_f.shape

    final_xs_f = xs_f[:, -1, :]
    mean = final_xs_f.mean(dim=0, keepdim=True)
    centered_final = final_xs_f - mean

    if batch > max_pca_samples:
        with torch_seed_context(seed_offset(seed, 0)):
            rand_idxs = torch.randperm(batch, device=centered_final.device)[
                :max_pca_samples
            ]
        centered_final = centered_final[rand_idxs]

    # MPS can be unstable/slow for SVD in this shape; perform SVD on CPU.
    centered_for_svd = (
        centered_final.cpu() if centered_final.device.type == "mps" else centered_final
    )
    _, _, vt = torch.linalg.svd(centered_for_svd, full_matrices=False)
    v = vt[:n_components, :].T.to(xs_f.device)  # (D, 2)

    flat_xsf = xs_f.reshape(-1, dim) - mean
    flat_xsb = xs_b.reshape(-1, dim) - mean

    proj_f = (flat_xsf @ v).reshape(batch, xs_f.shape[1], n_components)
    proj_b = (flat_xsb @ v).reshape(batch, xs_b.shape[1], n_components)

    # Return as (T+1, B, 2), which matches caller expectations.
    return proj_f.transpose(0, 1).cpu().numpy(), proj_b.transpose(0, 1).cpu().numpy()


def _select_steps(n_steps: int, n_snapshot: int = 5) -> np.ndarray:
    """Select snapshot steps including both endpoints."""
    return np.linspace(0, n_steps, n_snapshot).astype(int)


def _create_opinion_snapshots_figure(
    xs_forward: torch.Tensor,
    xs_backward: torch.Tensor,
    time_grid: torch.Tensor,
    n_snapshot: int = 5,
    seed: Optional[int] = None,
) -> Optional[plt.Figure]:
    """Create opinion PCA snapshot figure (no save side-effects)."""
    dim = xs_forward.shape[-1]
    if dim <= 2:
        return None

    steps = _select_steps(xs_forward.shape[0] - 1, n_snapshot=n_snapshot)
    xs_f_proj, xs_b_proj = _deepgsb_style_pca_project(
        xs_forward, xs_backward, n_components=2, seed=seed
    )

    # Keep a fixed color across time for consistency with other scenarios.
    point_color = "#1f77b4"
    titles = [r"$t$ = 0", r"$t$ = 0.25$T$", r"$t$ = 0.50$T$", r"$t$ = 0.75$T$", r"$t = T$"]
    if len(steps) != 5:
        titles = [f"t={float(time_grid[s].item()):.2f}" for s in steps]

    fig, axes = plt.subplots(
        2, len(steps), figsize=(4.8 * len(steps), 9.5), constrained_layout=True
    )

    for col, step in enumerate(steps):
        axes[0, col].scatter(
            xs_f_proj[step, :, 0],
            xs_f_proj[step, :, 1],
            s=14.0,
            alpha=0.5,
            color=point_color,
            linewidths=0,
            rasterized=True,
        )
        axes[0, col].set_title(f"Forward {titles[col]}", fontsize=12, fontweight="normal")
        axes[1, col].scatter(
            xs_b_proj[step, :, 0],
            xs_b_proj[step, :, 1],
            s=14.0,
            alpha=0.5,
            color=point_color,
            linewidths=0,
            rasterized=True,
        )
        axes[1, col].set_title(
            f"Backward {titles[col]}", fontsize=12, fontweight="normal"
        )
        for ax in (axes[0, col], axes[1, col]):
            ax.set_aspect("equal")
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            ax.tick_params(axis="both", which="major", labelsize=10)
            ax.grid(True, alpha=0.18, linewidth=0.6)

    return fig


def plot_opinion_snapshots(
    xs_forward: torch.Tensor,
    xs_backward: torch.Tensor,
    time_grid: torch.Tensor,
    exp_manager: Optional["ExperimentManager"] = None,
    n_snapshot: int = 5,
    seed: Optional[int] = None,
) -> None:
    """Plot opinion trajectories with PCA projection (for dim>2).

    Produces two rows (forward/backward) with snapshots across time, and saves to figures/.
    """
    _save_opinion_snapshots_figure(
        exp_manager=exp_manager,
        xs_forward=xs_forward,
        xs_backward=xs_backward,
        time_grid=time_grid,
        filenames=["opinion_pca_snapshots.png", "opinion_pca_snapshots.pdf"],
        seed=seed,
    )


def _create_opinion_directional_similarity_figure(
    xs_forward: torch.Tensor,
    xs_backward: torch.Tensor,
    n_est: int = 5000,
    seed: Optional[int] = None,
) -> Optional[plt.Figure]:
    """Create directional similarity histogram figure (no save side-effects)."""
    dim = xs_forward.shape[-1]
    if dim < 2:
        return None

    def _est_directional_similarity(
        xs_term: torch.Tensor,
        *,
        local_seed: Optional[int],
    ) -> np.ndarray:
        # xs_term: (B, dim)
        xs_centered = xs_term - xs_term.mean(dim=0, keepdim=True)
        batch = xs_centered.shape[0]
        with torch_seed_context(local_seed):
            rand1 = torch.randint(batch, (n_est,), device=xs_term.device)
            rand2 = torch.randint(batch, (n_est,), device=xs_term.device)
        v1 = xs_centered[rand1]
        v2 = xs_centered[rand2]
        v1 = v1 / torch.linalg.norm(v1, dim=-1, keepdim=True).clamp_min(1e-8)
        v2 = v2 / torch.linalg.norm(v2, dim=-1, keepdim=True).clamp_min(1e-8)
        cos_angle = (v1 * v2).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.acos(cos_angle)
        return (1.0 - angle / torch.pi).cpu().numpy()

    sim_f = _est_directional_similarity(xs_forward[-1], local_seed=seed_offset(seed, 0))
    sim_b = _est_directional_similarity(xs_backward[0], local_seed=seed_offset(seed, 1))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].hist(sim_f, bins=100, color="blue", alpha=0.7)
    axes[0].set_title("Directional Similarity (Forward end)")
    axes[0].set_xlim(0, 1)
    axes[1].hist(sim_b, bins=100, color="orange", alpha=0.7)
    axes[1].set_title("Directional Similarity (Backward start)")
    axes[1].set_xlim(0, 1)
    for ax in axes:
        ax.set_xlabel(r"$1 - \mathrm{angle}/\pi$")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.2)
    return fig


def plot_opinion_directional_similarity(
    xs_forward: torch.Tensor,
    xs_backward: torch.Tensor,
    exp_manager: Optional["ExperimentManager"] = None,
    n_est: int = 5000,
    seed: Optional[int] = None,
) -> None:
    """Plot directional similarity histograms for opinion models (dim>=2)."""
    _save_opinion_directional_similarity_figure(
        exp_manager=exp_manager,
        xs_forward=xs_forward,
        xs_backward=xs_backward,
        filenames=["opinion_directional_similarity.png"],
        n_est=n_est,
        seed=seed,
    )


def _create_trajectories_figure(
    xs_forward: torch.Tensor,
    xs_backward: torch.Tensor,
    mu_0: torch.Tensor,
    mu_1: torch.Tensor,
    obstacle_config: Optional[Dict] = None,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    mu_0_components: Optional[list] = None,
    mu_1_components: Optional[list] = None,
) -> Optional[plt.Figure]:
    """Create 2D forward/backward trajectory figure (no save side-effects)."""
    if xs_forward.shape[-1] != 2:
        return None

    def _legend_without_obstacle(ax: plt.Axes) -> None:
        handles, labels = ax.get_legend_handles_labels()
        filtered = [(h, l) for h, l in zip(handles, labels) if l and l != "Obstacle"]
        if filtered:
            ax.legend(*zip(*filtered), loc="upper right")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    num_show = xs_forward.shape[1]
    traj_alpha = 0.16 if mu_1_components else 0.08

    if obstacle_config:
        _plot_obstacles(axes, obstacle_config, xlim=xlim, ylim=ylim)

    for i in range(num_show):
        traj = xs_forward[:, i, :].cpu().numpy()
        axes[0].plot(traj[:, 0], traj[:, 1], "b-", alpha=traj_alpha, linewidth=0.4)
        axes[0].plot(
            traj[0, 0],
            traj[0, 1],
            "o",
            color="green",
            alpha=0.5,
            markersize=2.5,
            label="Initial particles" if i == 0 else None,
        )
        axes[0].plot(
            traj[-1, 0],
            traj[-1, 1],
            "o",
            color="red",
            alpha=0.5,
            markersize=2.5,
            label="Terminal particles" if i == 0 else None,
        )
    axes[0].set_title("Forward trajectories")
    axes[0].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    _legend_without_obstacle(axes[0])
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect("equal")
    if xlim:
        axes[0].set_xlim(xlim)
    if ylim:
        axes[0].set_ylim(ylim)

    for i in range(num_show):
        traj = xs_backward[:, i, :].cpu().numpy()
        axes[1].plot(traj[:, 0], traj[:, 1], "b-", alpha=traj_alpha, linewidth=0.4)
        axes[1].plot(
            traj[-1, 0],
            traj[-1, 1],
            "o",
            color="red",
            alpha=0.5,
            markersize=2.5,
            label="Terminal particles" if i == 0 else None,
        )
        axes[1].plot(
            traj[0, 0],
            traj[0, 1],
            "o",
            color="green",
            alpha=0.5,
            markersize=2.5,
            label="Initial particles" if i == 0 else None,
        )
    axes[1].set_title("Backward trajectories")
    axes[1].set_xlabel(r"$x_1$")
    axes[1].set_ylabel(r"$x_2$")
    _legend_without_obstacle(axes[1])
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect("equal")
    if xlim:
        axes[1].set_xlim(xlim)
    if ylim:
        axes[1].set_ylim(ylim)

    plt.tight_layout()
    return fig


def plot_trajectories(
    xs_forward: torch.Tensor,
    xs_backward: torch.Tensor,
    mu_0: torch.Tensor,
    mu_1: torch.Tensor,
    exp_manager: Optional[ExperimentManager] = None,
    obstacle_config: Optional[Dict] = None,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    mu_0_components: Optional[list] = None,
    mu_1_components: Optional[list] = None,
) -> None:
    """Plot forward and backward particle trajectories.

    Args:
        xs_forward: Forward trajectories tensor of shape
            (T+1, batch_size, dim).
        xs_backward: Backward trajectories tensor of shape
            (T+1, batch_size, dim).
        mu_0: Initial distribution mean (tuple or tensor).
        mu_1: Target distribution mean (tuple or tensor).
        exp_manager: Optional ExperimentManager for saving to
            experiment directory.
            If None, saves to outputs/figures/.
        obstacle_config: Obstacle configuration dict with 'type' and parameters
        xlim: Optional x-axis limits (min, max)
        ylim: Optional y-axis limits (min, max)
        mu_0_components: Optional list of GMM component centers for μ₀
        mu_1_components: Optional list of GMM component centers for μ₁

    Note:
        Only works for 2D problems (dim=2). Returns silently if dim != 2.
    """
    _save_trajectory_figure(
        exp_manager=exp_manager,
        xs_forward=xs_forward,
        xs_backward=xs_backward,
        mu_0=mu_0,
        mu_1=mu_1,
        filenames=[
            "validate_generalized_schrodinger_bridge_trajectories.png",
            "validate_generalized_schrodinger_bridge_trajectories.pdf",
        ],
        obstacle_config=obstacle_config,
        xlim=xlim,
        ylim=ylim,
        mu_0_components=mu_0_components,
        mu_1_components=mu_1_components,
    )


def save_trajectory_animation(
    xs: torch.Tensor,
    exp_manager: Optional["ExperimentManager"],
    obstacle_config: Optional[Dict] = None,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    start: Optional[torch.Tensor] = None,
    target: Optional[torch.Tensor] = None,
    start_components: Optional[Sequence] = None,
    target_components: Optional[Sequence] = None,
    direction: str = "forward",
    fps: int = 10,
    reverse: bool = False,
) -> Optional[Path]:
    """
    Save an animated GIF of trajectories (2D only).

    Args:
        xs: Trajectories tensor of shape (T+1, N, 2).
        exp_manager: ExperimentManager for save path.
        obstacle_config: Obstacle config dict.
        xlim, ylim: Axis limits.
        start, target: Optional start/target means for markers.
        start_components, target_components: Optional lists of component centers to plot.
        direction: "forward" or "backward" (used in filename and color).
        fps: Frames per second for the GIF.
        reverse: If True, play frames in reverse order (useful for backward runs).
    """
    if xs.shape[-1] != 2:
        return None

    xs_np = xs.cpu().numpy()
    if reverse:
        xs_np = xs_np[::-1]
    T_plus_1, N, _ = xs_np.shape
    # Use the same color for forward/backward for consistency in GIFs
    color = "#1f77b4"

    fig, ax = plt.subplots(figsize=(6, 6))
    if obstacle_config:
        _plot_obstacles([ax], obstacle_config, xlim=xlim, ylim=ylim)
    if start_components:
        for i, comp in enumerate(start_components):
            label = "start" if i == 0 else None
            ax.scatter(
                comp[0], comp[1], c="green", s=150, marker="*", label=label, zorder=10
            )
    elif start is not None and start.numel() >= 2:
        ax.scatter(
            start[0].cpu(),
            start[1].cpu(),
            c="green",
            s=150,
            marker="*",
            label="start",
            zorder=10,
        )

    if target_components:
        for i, comp in enumerate(target_components):
            label = "target" if i == 0 else None
            ax.scatter(
                comp[0], comp[1], c="red", s=150, marker="*", label=label, zorder=10
            )
    elif target is not None and target.numel() >= 2:
        ax.scatter(
            target[0].cpu(),
            target[1].cpu(),
            c="red",
            s=150,
            marker="*",
            label="target",
            zorder=10,
        )
    scatter = ax.scatter(xs_np[0, :, 0], xs_np[0, :, 1], s=10, c=color, alpha=0.7)
    ax.set_title(f"{direction.capitalize()} trajectories")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if (
        (start is not None or target is not None)
        or start_components
        or target_components
    ):
        ax.legend(loc="upper right")

    def init():
        scatter.set_offsets(xs_np[0, :, :2])
        return (scatter,)

    def update(frame):
        scatter.set_offsets(xs_np[frame, :, :2])
        return (scatter,)

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=T_plus_1,
        interval=1000.0 / fps,
        blit=True,
    )

    figures_dir = _get_figure_dir(exp_manager)
    save_path = figures_dir / f"trajectory_{direction}.gif"

    try:
        anim.save(save_path, writer="pillow", dpi=120)
        _emit_info(f"  Saved animation: {save_path}")
    except Exception as e:
        _emit_warning(f"  Warning: failed to save animation ({e})")
        save_path = None
    plt.close(fig)
    return save_path


def save_opinion_pca_animation(
    xs_forward: torch.Tensor,
    xs_backward: torch.Tensor,
    exp_manager: Optional["ExperimentManager"],
    direction: str = "forward",
    fps: int = 10,
    reverse: bool = False,
    seed: Optional[int] = None,
) -> Optional[Path]:
    """Save an animated GIF for high-dimensional opinion trajectories via PCA."""
    if xs_forward.shape[-1] <= 2:
        return None

    proj_f, proj_b = _deepgsb_style_pca_project(
        xs_forward, xs_backward, n_components=2, seed=seed
    )
    xs_np = proj_f if direction == "forward" else proj_b
    if reverse:
        xs_np = xs_np[::-1]

    t_plus_1 = xs_np.shape[0]
    color = "#1f77b4"

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(xs_np[0, :, 0], xs_np[0, :, 1], s=10, c=color, alpha=0.7)
    ax.set_title(f"{direction.capitalize()} trajectories (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    def init():
        scatter.set_offsets(xs_np[0, :, :2])
        return (scatter,)

    def update(frame):
        scatter.set_offsets(xs_np[frame, :, :2])
        return (scatter,)

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=t_plus_1,
        interval=1000.0 / fps,
        blit=True,
    )

    figures_dir = _get_figure_dir(exp_manager)
    save_path = figures_dir / f"trajectory_{direction}.gif"

    try:
        anim.save(save_path, writer="pillow", dpi=120)
        _emit_info(f"  Saved animation: {save_path}")
    except Exception as e:
        _emit_warning(f"  Warning: failed to save opinion PCA animation ({e})")
        save_path = None
    plt.close(fig)
    return save_path


def create_realtime_visualization_callback(
    exp_manager: ExperimentManager,
    config: Config,
    sample_rho0: Callable[[int], torch.Tensor],
    sample_rho1: Callable[[int], torch.Tensor],
    mu_0: torch.Tensor,
    mu_1: torch.Tensor,
    mu_0_components: Optional[list] = None,
    mu_1_components: Optional[list] = None,
):
    """Create a callback function for real-time visualization during training.

    Args:
        exp_manager: ExperimentManager instance for determining save paths.
        config: Configuration object (for dim check).
        sample_rho0: Function to sample from source distribution.
        sample_rho1: Function to sample from target distribution.
        mu_0: Initial distribution mean.
        mu_1: Target distribution mean.
        mu_0_components: Optional list of GMM component centers for μ₀
        mu_1_components: Optional list of GMM component centers for μ₁

    Returns:
        Callback function with signature (iteration, metrics, trainer) -> None.
        The callback generates and saves real-time plots of:
        - Training metrics (metrics_realtime_iterXXXX.png +
          metrics_realtime.png)
        - Particle trajectories (trajectories_realtime_iterXXXX.png +
          trajectories_realtime.png, 2D only)

    Note:
        The callback saves plots to {exp_dir}/figures/ at each checkpoint.
        Both numbered versions (e.g., _iter0001.png) and a latest version
        (e.g., metrics_realtime.png) are saved to preserve intermediate
        results and provide stable filenames for quick inspection.
    """
    def callback(iteration: int, metrics: Dict[str, list], trainer):
        _emit_info("")
        _emit_info(f"[Checkpoint {iteration}] Updating real-time plots...")

        _save_metrics_figure(
            metrics=metrics,
            config=config,
            exp_manager=exp_manager,
            filenames=[
                f"metrics_realtime_iter{iteration:04d}.png",
                "metrics_realtime.png",
            ],
        )

        num_vis = config.n_agents_test or 64
        if config.dim == 2 and mu_0.numel() >= 2 and mu_1.numel() >= 2:
            with torch_seed_context(config.visualization_seed):
                xs_forward, xs_backward = sample_visualization_trajectories(
                    trainer,
                    sample_rho0,
                    sample_rho1,
                    num_vis,
                    seed=config.visualization_seed,
                )
                _save_trajectory_figure(
                    exp_manager=exp_manager,
                    xs_forward=xs_forward,
                    xs_backward=xs_backward,
                    mu_0=mu_0,
                    mu_1=mu_1,
                    filenames=[
                        f"trajectories_realtime_iter{iteration:04d}.png",
                        "trajectories_realtime.png",
                    ],
                    mu_0_components=mu_0_components,
                    mu_1_components=mu_1_components,
                )
        elif "opinion" in getattr(config, "problem_type", "") and config.dim > 2:
            with torch_seed_context(config.visualization_seed):
                xs_forward, xs_backward = sample_visualization_trajectories(
                    trainer,
                    sample_rho0,
                    sample_rho1,
                    num_vis,
                    seed=config.visualization_seed,
                )
            _save_opinion_snapshots_figure(
                exp_manager=exp_manager,
                xs_forward=xs_forward,
                xs_backward=xs_backward,
                time_grid=trainer.solver.config.time_grid,
                filenames=[
                    f"opinion_pca_snapshots_iter{iteration:04d}.png",
                    "opinion_pca_snapshots_realtime.png",
                ],
                seed=config.visualization_seed,
            )
            _save_opinion_directional_similarity_figure(
                exp_manager=exp_manager,
                xs_forward=xs_forward,
                xs_backward=xs_backward,
                filenames=[
                    f"opinion_directional_similarity_iter{iteration:04d}.png",
                    "opinion_directional_similarity_realtime.png",
                ],
                seed=config.visualization_seed,
            )

        _emit_info(f"  Updated plots saved to {_get_figure_dir(exp_manager)}")

    return callback
