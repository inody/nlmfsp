#!/usr/bin/env python3
"""
Plot forward/backward trajectories from an experiment checkpoint.

Given an experiment directory (outputs/experiments/.../exp_YYYYMMDD_HHMMSS),
this script:
  1. Loads the latest (or specified) checkpoint and serialized Config.
  2. Reconstructs the problem setup from the stored CLI arguments or
     problem configuration JSON.
  3. Restores the FBSDESolver weights.
  4. Samples trajectories and saves a plot via nlmfsp.visualization.plot_trajectories.
"""

from __future__ import annotations

import argparse
import json
import shutil

# Ensure the local package is importable when running as a script
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlmfsp.base_costs import STunnelObstacleCost, VNeckObstacleCost
from nlmfsp.base_drifts import ConstantDrift
from nlmfsp.drifts import OpinionMeanFieldDrift
from nlmfsp.config import Config, migrate_legacy_config_dict
from nlmfsp.fbsde import FBSDESolver
from nlmfsp.interaction import create_interactions
from nlmfsp.problem_setup import load_problem_config, setup_problem_artifacts
from nlmfsp.cli_helpers import resolve_device
from nlmfsp.rng_utils import seed_offset, torch_seed_context
from nlmfsp.visualization import (
    plot_trajectories,
    plot_opinion_snapshots,
    plot_opinion_directional_similarity,
    save_trajectory_animation,
    save_opinion_pca_animation,
)

DEFAULT_FIGURE_NAME = "validate_generalized_schrodinger_bridge_trajectories.png"
DEFAULT_FIGURE_NAME_PDF = "validate_generalized_schrodinger_bridge_trajectories.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot trajectories from a checkpoint in an experiment directory."
    )
    parser.add_argument(
        "experiment_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Path to experiment directory containing config.json and checkpoints/. "
        "You can also pass the same path via --experiment-dir.",
    )
    parser.add_argument(
        "--experiment-dir",
        dest="experiment_dir_flag_dash",
        type=Path,
        default=None,
        help="Optional flag form of the experiment directory path.",
    )
    parser.add_argument(
        "--experiment_dir",
        dest="experiment_dir_flag_underscore",
        type=Path,
        default=None,
        help="Optional underscore variant of --experiment-dir.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint identifier. Provide an iteration number (e.g., 42), "
            "a filename (checkpoint_0042.pt), or an explicit path. "
            "Default selects the highest-iteration checkpoint."
        ),
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=256,
        help="Number of trajectories to sample for plotting (default: 256).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device for loading/sampling (cpu, cuda, mps, or auto).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for trajectory sampling. Defaults to the run's "
            "stored visualization_seed, or 123 if unavailable."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to copy the generated figure to after plotting.",
    )
    parser.add_argument(
        "--save-gif",
        action="store_true",
        help="Also save animated GIFs of forward/backward trajectories (2D only).",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=10,
        help="Frames per second for GIF animation (default: 10).",
    )
    args = parser.parse_args()
    exp_dir_value = (
        args.experiment_dir_flag_underscore
        or args.experiment_dir_flag_dash
        or args.experiment_dir
    )
    if exp_dir_value is None:
        parser.error(
            "An experiment directory must be provided as a positional argument or via --experiment-dir."
        )

    args.experiment_dir = exp_dir_value

    # Normalize auxiliary attributes to avoid confusion downstream
    args.experiment_dir_flag_dash = None
    args.experiment_dir_flag_underscore = None
    return args


def _checkpoint_sort_key(path: Path) -> Tuple[int, float]:
    """Sort checkpoints by numeric suffix (fall back to mtime)."""
    stem = path.stem  # e.g. checkpoint_0042
    suffix = stem.split("_", 1)[-1] if "_" in stem else stem
    score: int
    lowered = suffix.lower()
    if lowered == "latest":
        score = 10**9
    elif lowered == "final":
        score = 10**9 + 1
    else:
        try:
            score = int(suffix)
        except ValueError:
            score = -1
    return score, path.stat().st_mtime


def resolve_checkpoint(exp_dir: Path, checkpoint_arg: Optional[str]) -> Path:
    """Resolve checkpoint path from user input or pick the latest one."""
    ckpt_dir = exp_dir / "checkpoints"
    if checkpoint_arg:
        candidate = Path(checkpoint_arg)
        if candidate.is_file():
            return candidate

        if candidate.suffix == ".pt" and (ckpt_dir / candidate.name).is_file():
            return ckpt_dir / candidate.name

        if candidate.name.isdigit():
            padded = ckpt_dir / f"checkpoint_{int(candidate.name):04d}.pt"
            if padded.is_file():
                return padded

        numbered = ckpt_dir / f"checkpoint_{checkpoint_arg}.pt"
        if numbered.is_file():
            return numbered

        raise FileNotFoundError(f"Could not resolve checkpoint '{checkpoint_arg}'.")

    checkpoints = sorted(ckpt_dir.glob("checkpoint_*.pt"), key=_checkpoint_sort_key)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return checkpoints[-1]


def load_experiment_config(
    exp_dir: Path,
) -> Tuple[Config, argparse.Namespace, Dict[str, Any]]:
    """Load stored config and CLI args from experiment directory."""
    config_path = exp_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found in {exp_dir}")

    with config_path.open() as f:
        payload = json.load(f)

    if "config" not in payload:
        raise ValueError(f"'config' section missing in {config_path}")

    config_data: Dict[str, Any] = payload["config"]
    cli_args_dict: Dict[str, Any] = payload.get("cli_args", {})
    merged_args: Dict[str, Any] = config_data.copy()
    for key, value in cli_args_dict.items():
        if value is not None:
            merged_args[key] = value

    problem_config_path = merged_args.get("problem_config_path")
    problem_config = cli_args_dict.get("problem_config")
    if problem_config_path:
        candidate_path = Path(problem_config_path)
        if not candidate_path.exists() and problem_config:
            merged_args["problem_config_path"] = str(Path(problem_config).resolve())
    elif problem_config:
        merged_args["problem_config_path"] = str(Path(problem_config).resolve())

    valid_fields = {f.name for f in fields(Config)}
    sanitized_config = {k: v for k, v in merged_args.items() if k in valid_fields}
    sanitized_config = migrate_legacy_config_dict(sanitized_config)
    backend = payload.get("interaction_backend") or sanitized_config.get(
        "interaction_backend"
    )
    if backend not in {"exact", "nn", "random_batch"}:
        if sanitized_config.get("learn_interaction_drift") or sanitized_config.get(
            "learn_interaction_cost"
        ) or payload.get("method") == "learned":
            backend = "nn"
        elif sanitized_config.get("random_batch_size") is not None:
            backend = "random_batch"
        else:
            backend = "exact"
    sanitized_config["interaction_backend"] = backend
    dropped_keys = sorted(set(config_data) - valid_fields)
    if dropped_keys:
        print(
            "warning: ignoring legacy config keys not supported by Config: "
            + ", ".join(dropped_keys)
        )

    config_obj = Config(**sanitized_config)
    config_obj.output_dir = str(exp_dir)
    config_obj.checkpoint_dir = str(exp_dir / "checkpoints")
    config_obj.log_dir = str(exp_dir / "logs")
    args_namespace = argparse.Namespace(**merged_args)

    return config_obj, args_namespace, payload


def _load_checkpoint_interaction_modules(
    solver: FBSDESolver, solver_state: Dict[str, Any], config: Config
) -> None:
    """Load optional drift/cost modules from checkpoint with shape-aware recovery."""
    def _load_with_legacy_prefix(module: torch.nn.Module, state: Dict[str, Any]) -> None:
        try:
            module.load_state_dict(state)
            return
        except RuntimeError as exc:
            if any(key.startswith("impl.") for key in state):
                remapped = {
                    (f"net.{key}" if key.startswith("impl.") else key): value
                    for key, value in state.items()
                }
                module.load_state_dict(remapped)
                return
            raise exc

    drift_state = solver_state.get("drift_net")
    if drift_state is not None:
        try:
            if solver.drift_net is not None:
                _load_with_legacy_prefix(solver.drift_net, drift_state)
            else:
                raise RuntimeError("solver.drift_net is None")
        except RuntimeError as exc:
            # Legacy checkpoints may store opinion analytical drift buffers while
            # current config resolves to zero drift. Recover from checkpoint state.
            if set(drift_state.keys()) == {"xi_schedule", "f_mul"}:
                polarize_strength = 1.0 if config.dim <= 2 else 6.0
                recovered_drift = OpinionMeanFieldDrift(
                    xi_schedule=drift_state["xi_schedule"].to(config.device),
                    f_mul=drift_state["f_mul"].to(config.device),
                    dt=config.dt,
                    T=config.T,
                    polarize_strength=polarize_strength,
                ).to(config.device)
                recovered_drift.load_state_dict(drift_state)
                solver.drift_net = recovered_drift
                print(
                    "Recovered opinion drift module from checkpoint state "
                    "(xi_schedule/f_mul)."
                )
            else:
                raise RuntimeError(
                    f"Failed to load drift_net from checkpoint: {exc}"
                ) from exc

    cost_state = solver_state.get("cost_net")
    if cost_state is not None and solver.cost_net is not None:
        _load_with_legacy_prefix(solver.cost_net, cost_state)


def main() -> None:
    args = parse_args()
    exp_dir = args.experiment_dir.expanduser().resolve()

    config, cli_args, _ = load_experiment_config(exp_dir)
    checkpoint_path = resolve_checkpoint(exp_dir, args.checkpoint)

    # Device resolution priority: CLI override > stored CLI > config default
    default_device = getattr(cli_args, "device", None) or str(config.device) or "auto"
    device_choice = args.device or default_device
    device_str = resolve_device(device_choice, print_info=False)

    if device_str == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA device but none is available; falling back to CPU.")
        device_str = "cpu"
    elif device_str == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not has_mps:
            print("Requested MPS device but it is unavailable; falling back to CPU.")
            device_str = "cpu"

    device = torch.device(device_str)

    config.device = device
    cli_args.device = device_str

    if getattr(cli_args, "learn_interaction_drift", None) is not None:
        config.learn_interaction_drift = bool(cli_args.learn_interaction_drift)
    if getattr(cli_args, "learn_interaction_cost", None) is not None:
        config.learn_interaction_cost = bool(cli_args.learn_interaction_cost)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    sampling_seed = (
        args.seed
        if args.seed is not None
        else getattr(config, "visualization_seed", None)
    )
    if sampling_seed is None:
        sampling_seed = 123

    (
        mu_0,
        mu_1,
        sample_rho0,
        sample_rho1,
        mu_0_components,
        mu_1_components,
        obstacle_config,
        xlim,
        ylim,
        base_drift_fn,
        base_cost_fn,
        _log_p0,
        _log_p1,
    ) = setup_problem_artifacts(config)

    (
        drift_net,
        cost_net,
        _,
        _,
        *_,
    ) = create_interactions(config)

    solver = FBSDESolver(
        config=config,
        drift_net=drift_net,
        cost_net=cost_net,
        base_drift_fn=base_drift_fn,
        base_cost_fn=base_cost_fn,
    )

    solver_state: Dict[str, Any] = checkpoint["solver_state"]
    solver.value_net.load_state_dict(solver_state["value_net"])
    solver.value_hat_net.load_state_dict(solver_state["value_hat_net"])
    solver.grad_net.load_state_dict(solver_state["grad_net"])
    solver.grad_hat_net.load_state_dict(solver_state["grad_hat_net"])

    _load_checkpoint_interaction_modules(solver, solver_state, config)

    n_samples = max(1, int(args.num_trajectories))
    with torch_seed_context(seed_offset(sampling_seed, 0)):
        x0 = sample_rho0(n_samples)
    with torch_seed_context(seed_offset(sampling_seed, 1)):
        x1 = sample_rho1(n_samples)

    fwd_gif_path: Optional[Path] = None
    bwd_gif_path: Optional[Path] = None

    with torch.no_grad():
        xs_forward, _, _ = solver.sample_forward_trajectory(
            x0, seed=seed_offset(sampling_seed, 2)
        )
        xs_backward, _, _ = solver.sample_backward_trajectory(
            x1, seed=seed_offset(sampling_seed, 3)
        )

    exp_stub = SimpleNamespace(exp_dir=exp_dir)
    problem_type = getattr(config, "problem_type", None)
    if problem_type is None and getattr(config, "problem_config_path", None):
        problem_type = load_problem_config(Path(config.problem_config_path)).get(
            "problem_type"
        )

    if config.dim == 2:
        plot_trajectories(
            xs_forward,
            xs_backward,
            mu_0,
            mu_1,
            exp_manager=exp_stub,
            obstacle_config=obstacle_config,
            xlim=xlim,
            ylim=ylim,
            mu_0_components=mu_0_components,
            mu_1_components=mu_1_components,
        )
        if args.save_gif:
            fwd_gif_path = save_trajectory_animation(
                xs_forward,
                exp_stub,
                obstacle_config=obstacle_config,
                xlim=xlim,
                ylim=ylim,
                start=mu_0,
                target=mu_1,
                start_components=mu_0_components,
                target_components=mu_1_components,
                direction="forward",
                fps=args.gif_fps,
            )
            bwd_gif_path = save_trajectory_animation(
                xs_backward,
                exp_stub,
                obstacle_config=obstacle_config,
                xlim=xlim,
                ylim=ylim,
                start=mu_1,
                target=mu_0,
                start_components=mu_1_components,
                target_components=mu_0_components,
                direction="backward",
                fps=args.gif_fps,
                reverse=True,
            )
    elif problem_type and "opinion" in problem_type and config.dim > 2:
        # High-dimensional opinion: PCA projection + directional similarity
        snap_fig = plot_opinion_snapshots(
            xs_forward,
            xs_backward,
            config.time_grid,
            exp_manager=exp_stub,
            seed=sampling_seed,
        )
        dirsim_fig = plot_opinion_directional_similarity(
            xs_forward,
            xs_backward,
            exp_manager=exp_stub,
            seed=sampling_seed,
        )
        if args.save_gif:
            fwd_gif_path = save_opinion_pca_animation(
                xs_forward,
                xs_backward,
                exp_stub,
                direction="forward",
                fps=args.gif_fps,
                reverse=False,
                seed=sampling_seed,
            )
            bwd_gif_path = save_opinion_pca_animation(
                xs_forward,
                xs_backward,
                exp_stub,
                direction="backward",
                fps=args.gif_fps,
                reverse=True,
                seed=sampling_seed,
            )

    # Decide which default figure path to report/copy
    if config.dim == 2:
        default_figure_path = exp_dir / "figures" / DEFAULT_FIGURE_NAME
        default_figure_pdf = exp_dir / "figures" / DEFAULT_FIGURE_NAME_PDF
    elif problem_type and "opinion" in problem_type and config.dim > 2:
        default_figure_path = exp_dir / "figures" / "opinion_pca_snapshots.png"
        default_figure_pdf = exp_dir / "figures" / "opinion_pca_snapshots.pdf"
    else:
        default_figure_path = exp_dir / "figures" / DEFAULT_FIGURE_NAME
        default_figure_pdf = exp_dir / "figures" / DEFAULT_FIGURE_NAME_PDF
    if args.output and default_figure_path.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(default_figure_path, args.output)
        print(f"Copied figure to {args.output}")
        if default_figure_pdf and default_figure_pdf.exists():
            target_pdf = args.output.with_suffix(".pdf")
            shutil.copy2(default_figure_pdf, target_pdf)
            print(f"Copied PDF figure to {target_pdf}")
    elif args.output:
        print(f"Warning: default trajectory figure not found at {default_figure_path}, skipping copy.")

    # Copy GIFs next to the output (if requested and available)
    if args.output and args.save_gif:
        out_dir = args.output.parent
        base = args.output.stem
        if fwd_gif_path and Path(fwd_gif_path).exists():
            target = out_dir / f"{base}_forward.gif"
            shutil.copy2(fwd_gif_path, target)
            print(f"Copied forward GIF to {target}")
        if bwd_gif_path and Path(bwd_gif_path).exists():
            target = out_dir / f"{base}_backward.gif"
            shutil.copy2(bwd_gif_path, target)
            print(f"Copied backward GIF to {target}")

    # For opinion (dim>2), also copy directional similarity histogram if available
    if (
        args.output
        and problem_type
        and "opinion" in problem_type
        and config.dim > 2
    ):
        dirsim_path = exp_dir / "figures" / "opinion_directional_similarity.png"
        if dirsim_path.exists():
            tag = exp_dir.name
            target = args.output.parent / f"{tag}_opinion_directional_similarity.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dirsim_path, target)
            print(f"Copied directional similarity figure to {target}")
        else:
            print(f"Warning: directional similarity figure not found at {dirsim_path}, skipping copy.")

    iteration = checkpoint.get("iteration")
    print(
        f"Plotted trajectories for checkpoint {checkpoint_path.name} (iteration {iteration})."
    )
    print(f"Saved figure to {default_figure_path}")


if __name__ == "__main__":
    main()
