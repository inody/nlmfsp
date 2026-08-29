"""CLI helper utilities for training scripts.

This module provides functions for:
- Building Config from command-line arguments
- Creating analytical interaction functions (drift/cost)
- Logging configuration details
"""

from __future__ import annotations

import argparse
from dataclasses import MISSING
from typing import Any, Dict, Optional

import torch
from .config import Config


def _sequence_to_tuple(value: Any) -> Any:
    """Convert list-based dimension specs to tuple while leaving scalars untouched."""
    if isinstance(value, list):
        return tuple(value)
    return value


def get_default_device() -> str:
    """Get default device based on available hardware.

    Returns:
        Device string: "cuda" if NVIDIA GPU available,
                      "mps" if Apple Silicon GPU available,
                      "cpu" otherwise.

    Example:
        >>> device = get_default_device()
        >>> print(f"Using device: {device}")
        Using device: cuda
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def resolve_device(device: str, print_info: bool = True) -> str:
    """Resolve device string and optionally print CUDA device mapping.

    This function:
    1. Resolves "auto" to the best available device (cuda > mps > cpu)
    2. Validates explicit device choices (cuda/mps/cpu) without fallback
    3. Optionally prints CUDA[0] physical device name for verification

    Args:
        device: Device string ("auto", "cuda", "mps", or "cpu")
        print_info: If True, prints auto-detection result and CUDA[0] mapping

    Returns:
        Resolved device string ("cuda", "mps", or "cpu")

    Example:
        >>> device = resolve_device("auto")
        Auto-detected device: cuda
        CUDA[0]: NVIDIA GeForce RTX 3090
        >>> device
        'cuda'

    Note:
        When CUDA_VISIBLE_DEVICES is set, CUDA[0] maps to the selected
        physical GPU, helping verify GPU selection in shared environments.
    """
    import warnings

    # Handle auto-detection
    if device == "auto":
        resolved = get_default_device()
        if print_info:
            print(f"Auto-detected device: {resolved}")
    else:
        resolved = device

    # Print CUDA device mapping if available and requested
    if print_info and resolved == "cuda":
        try:
            if torch.cuda.is_available():
                print("CUDA[0]:", torch.cuda.get_device_name(0))
        except Exception as e:
            # Non-fatal: just warn and continue
            warnings.warn(f"Could not query CUDA device name: {e}")

    return resolved


def apply_problem_config_overrides(
    args: argparse.Namespace,
    problem_cfg: Dict[str, Any],
    parser: argparse.ArgumentParser,
) -> argparse.Namespace:
    """Apply problem configuration overrides to command-line arguments.

    This function processes the 'training' section of a problem configuration
    JSON file and applies its values to the argparse Namespace, respecting
    the priority: CLI explicit > problem JSON > CLI default.

    Args:
        args: Parsed command-line arguments (will be modified in-place)
        problem_cfg: Problem configuration dictionary loaded from JSON
        parser: ArgumentParser instance used to check default values

    Returns:
        Modified argparse namespace.

    Priority Logic:
        1. If user explicitly set a CLI arg (value != default), keep CLI value
        2. Otherwise, use problem config value if available
        3. Otherwise, fall back to CLI default

    Example:
        >>> args = apply_problem_config_overrides(args, problem_cfg, parser)

    Raises:
        ValueError: If a training key cannot be mapped to a known CLI argument.
    """
    training_cfg = problem_cfg.get("training", {})

    for key, value in training_cfg.items():
        attr_name = key

        # Convert list to tuple for hidden dims variants
        if key in (
            "hidden_dims",
            "t_hidden_dims",
            "interaction_hidden_dims",
            "interaction_t_hidden_dims",
        ):
            value = _sequence_to_tuple(value)

        # Only override if user didn't explicitly set via CLI
        if hasattr(args, attr_name):
            # Check if value differs from parser default (i.e., user set it)
            default_value = parser.get_default(attr_name)
            current_value = getattr(args, attr_name)

            # If current value == default, user didn't set it -> use config
            if current_value == default_value:
                setattr(args, attr_name, value)
                print(f"  Using config value for {key}: {value}")
            else:
                print(f"  CLI override for {key}: {current_value} (config has {value})")
        else:
            raise ValueError(
                f"Training config key '{key}' is not recognized by CLI parser "
                f"(mapped attribute: '{attr_name}')."
            )

    return args


def apply_interaction_config_overrides(
    args: argparse.Namespace,
    problem_cfg: Dict[str, Any],
    parser: argparse.ArgumentParser,
) -> None:
    """Merge interaction settings from JSON into CLI namespace.

    Priority: CLI explicit > interactions section.
    """

    interactions_cfg = problem_cfg.get("interactions", {}) or {}

    def _set_attr(attr: str, value: Any) -> None:
        if value is None:
            return
        value = _sequence_to_tuple(value)
        if not hasattr(args, attr):
            setattr(args, attr, value)
            return
        try:
            default = parser.get_default(attr)
        except (AttributeError, ValueError):
            default = getattr(args, attr)
        current = getattr(args, attr)
        if current == default or current is None:
            setattr(args, attr, value)

    # Keep learn flags synced when provided in interactions section.
    _set_attr(
        "learn_interaction_drift", interactions_cfg.get("learn_interaction_drift")
    )
    _set_attr("learn_interaction_cost", interactions_cfg.get("learn_interaction_cost"))
    _set_attr("interaction_backend", interactions_cfg.get("interaction_backend"))
    _set_attr("random_batch_size", interactions_cfg.get("random_batch_size"))

    # Interaction drift/cost types from interactions section
    _set_attr("drift_type", interactions_cfg.get("drift_type"))
    _set_attr("cost_type", interactions_cfg.get("cost_type"))

    for key, attr in (
        ("weight", "int_drift_weight"),
        ("sigma", "int_drift_sigma"),
        ("strength", "int_drift_strength"),
    ):
        _set_attr(attr, interactions_cfg.get("drift_params", {}).get(key))

    for key, attr in (
        ("weight", "int_cost_weight"),
        ("sigma", "int_cost_sigma"),
        ("strength", "int_cost_strength"),
    ):
        _set_attr(attr, interactions_cfg.get("cost_params", {}).get(key))

    return None


def build_config(args: argparse.Namespace) -> Config:
    """Build Config object from command-line arguments.

    Args:
        args: Parsed command-line arguments from argparse.

    Returns:
        Config object with parameters from args.

    Note:
        Some parameters are hardcoded for the current validation setup:
        - dim: 2
        - T: 1.0
        - n_interaction_steps: 0 unless interaction learning is enabled

        - device: Uses args.device if available, otherwise defaults to "cpu"
    """

    def _config_default(field_name: str) -> Any:
        field = Config.__dataclass_fields__[field_name]  # type: ignore[index]
        if field.default is not MISSING:
            return field.default
        if field.default_factory is not MISSING:
            return field.default_factory()  # type: ignore[misc]
        raise ValueError(f"Config field '{field_name}' has no default")

    def _arg_or_default(
        arg_name: str,
        *,
        config_field: Optional[str] = None,
        none_uses_default: bool = False,
    ) -> Any:
        default_field = config_field or arg_name
        if hasattr(args, arg_name):
            value = getattr(args, arg_name)
            if value is None and none_uses_default:
                return _config_default(default_field)
            return value
        return _config_default(default_field)

    device = _arg_or_default("device", none_uses_default=True)
    inner_steps = _arg_or_default("n_inner_steps", none_uses_default=True)
    outer_iterations = _arg_or_default("n_outer_iterations", none_uses_default=True)
    time_horizon = _arg_or_default("T", none_uses_default=True)
    dt = _arg_or_default("dt", none_uses_default=True)

    learn_interaction_drift = _arg_or_default(
        "learn_interaction_drift", none_uses_default=True
    )
    learn_interaction_cost = _arg_or_default(
        "learn_interaction_cost", none_uses_default=True
    )
    learn_interaction_enabled = bool(learn_interaction_drift or learn_interaction_cost)
    interaction_backend = _arg_or_default("interaction_backend")
    if interaction_backend is None:
        interaction_backend = "nn" if learn_interaction_enabled else _config_default(
            "interaction_backend"
        )
    interaction_steps = (
        _arg_or_default(
            "interaction_steps",
            config_field="n_interaction_steps",
            none_uses_default=True,
        )
        if learn_interaction_enabled
        else 0
    )
    resample_interaction_sample = _arg_or_default(
        "resample_interaction_sample", none_uses_default=True
    )
    lr_value = _arg_or_default(
        "lr_value", config_field="learning_rate_value", none_uses_default=True
    )
    lr_grad = _arg_or_default(
        "lr_grad", config_field="learning_rate_grad", none_uses_default=True
    )
    lr_interaction = _arg_or_default(
        "lr_interaction", config_field="learning_rate_interaction"
    )
    time_input_mode = _arg_or_default("time_input_mode", none_uses_default=True)
    use_time_index_normalization = _arg_or_default(
        "use_time_index_normalization", none_uses_default=True
    )
    if getattr(args, "time_input_mode", None) is None and use_time_index_normalization:
        time_input_mode = "normalized"

    if learn_interaction_enabled and lr_interaction is None:
        raise ValueError(
            "learning_rate_interaction must be provided when interaction learning is enabled."
        )
    interaction_hidden_dims = _sequence_to_tuple(
        _arg_or_default("interaction_hidden_dims", none_uses_default=True)
    )
    interaction_t_hidden_dims = _sequence_to_tuple(
        _arg_or_default("interaction_t_hidden_dims", none_uses_default=True)
    )

    # Use canonical parameter names directly.
    n_agents = _arg_or_default("n_agents")
    n_agents_test = _arg_or_default("n_agents_test")

    hidden_dims = _sequence_to_tuple(
        _arg_or_default("hidden_dims", none_uses_default=True)
    )

    return Config(
        dim=2,
        T=time_horizon,
        dt=dt,
        n_agents=n_agents,
        n_agents_test=n_agents_test,
        interaction_backend=interaction_backend,
        learn_interaction_drift=learn_interaction_drift,
        learn_interaction_cost=learn_interaction_cost,
        random_batch_size=_arg_or_default(
            "random_batch_size", none_uses_default=True
        ),
        n_outer_iterations=outer_iterations,
        n_inner_steps=inner_steps,
        n_interaction_steps=interaction_steps,
        standardize_interaction_targets=_arg_or_default(
            "standardize_interaction_targets", none_uses_default=True
        ),
        interaction_input_scale=_arg_or_default(
            "interaction_input_scale", none_uses_default=True
        ),
        resample_interaction_sample=resample_interaction_sample,
        learning_rate_interaction=lr_interaction,
        interaction_hidden_dims=interaction_hidden_dims,
        interaction_t_hidden_dims=interaction_t_hidden_dims,
        interaction_activation=_arg_or_default(
            "interaction_activation", none_uses_default=True
        ),
        drift_type=_arg_or_default("drift_type", none_uses_default=True),
        cost_type=_arg_or_default("cost_type", none_uses_default=True),
        int_drift_weight=_arg_or_default("int_drift_weight", none_uses_default=True),
        int_drift_sigma=_arg_or_default("int_drift_sigma", none_uses_default=True),
        int_drift_strength=_arg_or_default(
            "int_drift_strength", none_uses_default=True
        ),
        int_cost_weight=_arg_or_default("int_cost_weight", none_uses_default=True),
        int_cost_sigma=_arg_or_default("int_cost_sigma", none_uses_default=True),
        int_cost_strength=_arg_or_default("int_cost_strength", none_uses_default=True),
        learning_rate_value=lr_value,
        learning_rate_grad=lr_grad,
        lambda_ipf=_arg_or_default("lambda_ipf", none_uses_default=True),
        lambda_td=_arg_or_default("lambda_td", none_uses_default=True),
        lambda_fk=_arg_or_default("lambda_fk", none_uses_default=True),
        lambda_boundary=_arg_or_default("lambda_boundary", none_uses_default=True),
        td_horizon=_arg_or_default("td_horizon", none_uses_default=True),
        td_bootstrap_from_target=_arg_or_default(
            "td_bootstrap_from_target", none_uses_default=True
        ),
        use_multistep_td_loss=_arg_or_default(
            "use_multistep_td",
            config_field="use_multistep_td_loss",
            none_uses_default=True,
        ),
        ema_decay=_arg_or_default("ema_decay", none_uses_default=True),
        ema_start_step=_arg_or_default("ema_start_step", none_uses_default=True),
        ema_update_interval=_arg_or_default(
            "ema_update_interval", none_uses_default=True
        ),
        sigma=_arg_or_default("sigma", none_uses_default=True),
        hidden_dims=hidden_dims,
        t_hidden_dims=_sequence_to_tuple(
            _arg_or_default("t_hidden_dims", none_uses_default=True)
        ),
        fk_loss_norm=_arg_or_default("fk_loss_norm", none_uses_default=True),
        detach_interaction_drift_divergence=_arg_or_default(
            "detach_interaction_drift_divergence", none_uses_default=True
        ),
        activation=_arg_or_default("activation", none_uses_default=True),
        use_time_embedding=_arg_or_default(
            "use_time_embedding", none_uses_default=True
        ),
        time_embed_dim=_arg_or_default("time_embed_dim", none_uses_default=True),
        time_input_mode=time_input_mode,
        use_time_index_normalization=use_time_index_normalization,
        time_index_interval=_arg_or_default(
            "time_index_interval", none_uses_default=True
        ),
        optimizer_type=_arg_or_default("optimizer_type", none_uses_default=True),
        optimizer_momentum=_arg_or_default(
            "optimizer_momentum", none_uses_default=True
        ),
        optimizer_weight_decay=_arg_or_default(
            "optimizer_weight_decay", none_uses_default=True
        ),
        lr_scheduler_type=_arg_or_default("lr_scheduler_type", none_uses_default=True),
        lr_decay_step=_arg_or_default(
            "lr_step_size", config_field="lr_decay_step", none_uses_default=True
        ),
        lr_decay_gamma=_arg_or_default(
            "lr_decay", config_field="lr_decay_gamma", none_uses_default=True
        ),
        grad_clip=_arg_or_default("grad_clip", none_uses_default=True),
        log_interval=_arg_or_default("log_interval", none_uses_default=True),
        checkpoint_interval=_arg_or_default(
            "checkpoint_interval", none_uses_default=True
        ),
        verbose=True,
        device=device,
        seed=_arg_or_default("seed", none_uses_default=True),
        training_seed=_arg_or_default("training_seed", none_uses_default=True),
        validation_seed=_arg_or_default("validation_seed", none_uses_default=True),
        visualization_seed=_arg_or_default(
            "visualization_seed", none_uses_default=True
        ),
        evaluate_interaction_cpu=_arg_or_default(
            "evaluate_interaction_cpu", none_uses_default=True
        ),
        interaction_query_chunk_size=_arg_or_default(
            "interaction_query_chunk_size", none_uses_default=True
        ),
        interaction_ref_chunk_size=_arg_or_default(
            "interaction_ref_chunk_size", none_uses_default=True
        ),
        interaction_time_block_size=_arg_or_default(
            "interaction_time_block_size", none_uses_default=True
        ),
    )
