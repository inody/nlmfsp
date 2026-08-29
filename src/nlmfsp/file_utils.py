"""Utility functions for NLMFSP."""

import logging
import time
from pathlib import Path
from typing import Any

import torch


def get_project_root() -> Path:
    """Get project root directory."""
    # This file is in src/nlmfsp/file_utils.py, so go up 3 levels
    return Path(__file__).parent.parent.parent


def get_output_dir(subdir: str = "") -> Path:
    """Get output directory, creating it if necessary.

    Args:
        subdir: Subdirectory within outputs (e.g., 'figures', 'logs', 'models')

    Returns:
        Path to the output directory
    """
    root = get_project_root()
    output_dir = root / "outputs"

    if subdir:
        output_dir = output_dir / subdir

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_figure(fig, filename: str, subdir: str = "figures", **kwargs):
    """Save matplotlib figure to outputs directory.

    Args:
        fig: Matplotlib figure object
        filename: Filename (without path)
        subdir: Subdirectory within outputs
        **kwargs: Additional arguments passed to fig.savefig()
    """
    output_path = get_output_dir(subdir) / filename

    # Default save arguments
    save_kwargs = {
        "dpi": 150,
        "bbox_inches": "tight",
    }
    save_kwargs.update(kwargs)

    fig.savefig(output_path, **save_kwargs)
    relative_path = output_path.relative_to(get_project_root())
    logger = logging.getLogger("nlmfsp.run")
    if not logger.handlers:
        logger = logging.getLogger("nlmfsp.trainer")
    if logger.handlers:
        logger.info(f"  Saved: {relative_path}")
    else:
        print(f"  Saved: {relative_path}")
    return output_path


def save_trainer_checkpoint(
    trainer: Any,
    *,
    iteration: int | str,
    analytical_metrics: dict[str, float] | None = None,
) -> None:
    """Serialize and save a trainer checkpoint."""
    actual_iteration = (
        len(trainer.metrics["loss_ipf_forward"])
        if isinstance(iteration, str)
        else iteration
    )

    checkpoint = {
        "iteration": actual_iteration,
        "config": trainer.config,
        "solver_state": {
            "value_net": trainer.solver.value_net.state_dict(),
            "value_hat_net": trainer.solver.value_hat_net.state_dict(),
            "grad_net": trainer.solver.grad_net.state_dict(),
            "grad_hat_net": trainer.solver.grad_hat_net.state_dict(),
        },
        "optimizer_state": {
            "opt_forward": trainer.opt_forward.state_dict(),
            "opt_backward": trainer.opt_backward.state_dict(),
        },
        "scheduler_state": {
            "scheduler_forward": trainer.scheduler_forward.state_dict()
            if trainer.scheduler_forward is not None
            else None,
            "scheduler_backward": trainer.scheduler_backward.state_dict()
            if trainer.scheduler_backward is not None
            else None,
        },
        "metrics": trainer.metrics,
    }
    if analytical_metrics is not None:
        checkpoint["metrics_analytical"] = analytical_metrics

    if trainer.solver.drift_net is not None:
        checkpoint["solver_state"]["drift_net"] = trainer.solver.drift_net.state_dict()
    if trainer.solver.cost_net is not None:
        checkpoint["solver_state"]["cost_net"] = trainer.solver.cost_net.state_dict()

    if trainer.opt_interaction is not None:
        checkpoint["optimizer_state"]["opt_interaction"] = (
            trainer.opt_interaction.state_dict()
        )
        checkpoint["scheduler_state"]["scheduler_interaction"] = (
            trainer.scheduler_interaction.state_dict()
            if trainer.scheduler_interaction is not None
            else None
        )

    if isinstance(iteration, str):
        checkpoint_path = Path(trainer.config.checkpoint_dir) / f"checkpoint_{iteration}.pt"
    else:
        checkpoint_path = Path(trainer.config.checkpoint_dir) / f"checkpoint_{iteration:04d}.pt"
    torch.save(checkpoint, checkpoint_path)
    trainer.logger.info(f"Checkpoint saved: {checkpoint_path}")


def load_trainer_checkpoint(trainer: Any, checkpoint_path: str) -> int:
    """Load trainer state from checkpoint and return saved iteration."""
    checkpoint = torch.load(
        checkpoint_path, map_location=trainer.config.device, weights_only=False
    )

    trainer.solver.value_net.load_state_dict(checkpoint["solver_state"]["value_net"])
    trainer.solver.value_hat_net.load_state_dict(
        checkpoint["solver_state"]["value_hat_net"]
    )
    trainer.solver.grad_net.load_state_dict(checkpoint["solver_state"]["grad_net"])
    trainer.solver.grad_hat_net.load_state_dict(checkpoint["solver_state"]["grad_hat_net"])

    solver_state = checkpoint["solver_state"]
    if "interaction_net" in solver_state:
        raise ValueError(
            "Legacy checkpoint format is not supported: solver_state['interaction_net'] detected. "
            "Use a checkpoint that stores 'drift_net' and 'cost_net'."
        )
    if "drift_net" in solver_state:
        trainer.solver.drift_net.load_state_dict(solver_state["drift_net"])
    if "cost_net" in checkpoint["solver_state"]:
        trainer.solver.cost_net.load_state_dict(checkpoint["solver_state"]["cost_net"])

    trainer.opt_forward.load_state_dict(checkpoint["optimizer_state"]["opt_forward"])
    trainer.opt_backward.load_state_dict(checkpoint["optimizer_state"]["opt_backward"])
    if (
        "opt_interaction" in checkpoint["optimizer_state"]
        and trainer.opt_interaction is not None
    ):
        trainer.opt_interaction.load_state_dict(
            checkpoint["optimizer_state"]["opt_interaction"]
        )

    if (
        trainer.scheduler_forward is not None
        and checkpoint["scheduler_state"]["scheduler_forward"] is not None
    ):
        trainer.scheduler_forward.load_state_dict(
            checkpoint["scheduler_state"]["scheduler_forward"]
        )
    if (
        trainer.scheduler_backward is not None
        and checkpoint["scheduler_state"]["scheduler_backward"] is not None
    ):
        trainer.scheduler_backward.load_state_dict(
            checkpoint["scheduler_state"]["scheduler_backward"]
        )
    if "scheduler_interaction" in checkpoint["scheduler_state"]:
        sched_state = checkpoint["scheduler_state"]["scheduler_interaction"]
        if trainer.scheduler_interaction is not None and sched_state is not None:
            trainer.scheduler_interaction.load_state_dict(sched_state)

    trainer.metrics = checkpoint["metrics"]
    time_total = trainer.metrics.get("time_total")
    if isinstance(time_total, list):
        raise ValueError(
            "Invalid checkpoint metrics: time_total must be a scalar, not a list."
        )
    if time_total is None or not isinstance(time_total, (int, float)):
        raise ValueError(
            "Invalid checkpoint metrics: time_total must be present as a numeric scalar."
        )
    trainer.metrics["time_total"] = float(time_total)

    iteration = checkpoint["iteration"]
    trainer.logger.info(f"Checkpoint loaded from iteration {iteration}")
    return iteration


def create_trainer_logger(config: Any, *, name: str = "nlmfsp.trainer") -> logging.Logger:
    """Create and configure trainer logger with file + optional console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    if config.verbose:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(console_formatter)
        logger.addHandler(ch)

    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger
