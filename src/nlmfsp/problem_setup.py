"""Problem configuration and setup utilities.

This module provides:
- Strict loading and schema validation for problem JSON configs
- Distribution builders (Gaussian / Gaussian-mixture / opinion variants)
- Base drift/cost function builders
- Visualization metadata extraction
- End-to-end problem artifact reconstruction for training/plotting
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .base_costs import (
    ObstacleCost,
    SmoothObstacleCost,
    STunnelObstacleCost,
    VNeckObstacleCost,
)
from .base_drifts import ConstantDrift
from .visualization import setup_problem_visualization

VALID_PROBLEM_TYPES = {
    "gaussian_transport",
    "gmm_navigation",
    "vneck_navigation",
    "stunnel_navigation",
    "opinion_2d",
    "opinion_1k",
}

ALLOWED_TOP_LEVEL_KEYS = {
    "problem_type",
    "name",
    "description",
    "dim",
    "distributions",
    "interactions",
    "base_functions",
    "training",
    "visualization",
}

ALLOWED_DISTRIBUTION_KEYS = {
    "type",
    "mean",
    "std",
    "covar_diag",
    "components",
    "covar_override",
    "dim",
}
ALLOWED_GMM_COMPONENT_KEYS = {
    "mean",
    "std",
    "covar_diag",
    "covar_override",
    "dim",
    "weight",
}

ALLOWED_INTERACTIONS_KEYS = {
    "drift_type",
    "cost_type",
    "drift_params",
    "cost_params",
    "interaction_backend",
    "random_batch_size",
    "learn_interaction_drift",
    "learn_interaction_cost",
}
ALLOWED_INTERACTION_PARAM_KEYS = {"weight", "sigma", "strength"}

ALLOWED_BASE_FUNCTIONS_KEYS = {"drift", "cost"}
ALLOWED_BASE_DRIFT_KEYS = {"type", "weight", "target", "value", "description"}
ALLOWED_BASE_COST_KEYS = {
    "type",
    "description",
    "c_sq",
    "coef",
    "penalty",
    "obstacles",
    "ellipses",
}
ALLOWED_OBSTACLE_KEYS = {"center", "radius", "penalty", "description", "mode", "beta"}
ALLOWED_ELLIPSE_KEYS = {"center", "a", "b", "c", "penalty", "description"}
ALLOWED_BASE_DRIFT_TYPES = {"zero", "constant"}
ALLOWED_BASE_COST_TYPES = {
    "zero",
    "multi_obstacle",
    "vneck_parabolic",
    "stunnel_elliptic",
}

ALLOWED_VISUALIZATION_KEYS = {"xlim", "ylim"}

ALLOWED_TRAINING_KEYS = {
    "n_outer_iterations",
    "n_inner_steps",
    "n_agents",
    "n_agents_test",
    "T",
    "dt",
    "lr_value",
    "lr_grad",
    "lr_interaction",
    "activation",
    "hidden_dims",
    "t_hidden_dims",
    "interaction_hidden_dims",
    "interaction_t_hidden_dims",
    "interaction_activation",
    "learn_interaction_drift",
    "learn_interaction_cost",
    "interaction_steps",
    "standardize_interaction_targets",
    "interaction_input_scale",
    "interaction_query_chunk_size",
    "interaction_ref_chunk_size",
    "interaction_time_block_size",
    "resample_interaction_sample",
    "evaluate_interaction_cpu",
    "sigma",
    "lambda_ipf",
    "lambda_td",
    "lambda_fk",
    "lambda_boundary",
    "td_horizon",
    "td_bootstrap_from_target",
    "use_multistep_td",
    "fk_loss_norm",
    "detach_interaction_drift_divergence",
    "optimizer_type",
    "optimizer_momentum",
    "optimizer_weight_decay",
    "lr_scheduler_type",
    "lr_step_size",
    "lr_decay",
    "grad_clip",
    "log_interval",
    "checkpoint_interval",
    "seed",
    "device",
    "ema_decay",
    "ema_start_step",
    "ema_update_interval",
    "use_time_embedding",
    "time_input_mode",
    "time_embed_dim",
    "use_time_index_normalization",
    "time_index_interval",
}


def _unknown_keys(mapping: Dict[str, Any], allowed: set[str]) -> list[str]:
    return sorted(set(mapping.keys()) - allowed)


def _ensure_no_unknown_keys(
    mapping: Dict[str, Any], allowed: set[str], *, path: str
) -> None:
    unknown = _unknown_keys(mapping, allowed)
    if unknown:
        unknown_str = ", ".join(unknown)
        raise ValueError(f"Unknown key(s) at {path}: {unknown_str}")


def _validate_problem_config_schema(
    problem_cfg: Dict[str, Any], config_path: Path
) -> None:
    """Validate problem config strictly and reject unknown keys everywhere."""
    _ensure_no_unknown_keys(problem_cfg, ALLOWED_TOP_LEVEL_KEYS, path="root")

    distributions = problem_cfg.get("distributions")
    if not isinstance(distributions, dict):
        raise ValueError(f"Field 'distributions' must be an object in {config_path}")
    _ensure_no_unknown_keys(distributions, {"rho_0", "rho_1"}, path="distributions")

    for rho_key in ("rho_0", "rho_1"):
        rho_cfg = distributions.get(rho_key)
        if not isinstance(rho_cfg, dict):
            raise ValueError(
                f"Field 'distributions.{rho_key}' must be an object in {config_path}"
            )
        _ensure_no_unknown_keys(
            rho_cfg,
            ALLOWED_DISTRIBUTION_KEYS,
            path=f"distributions.{rho_key}",
        )
        rho_type = rho_cfg.get("type")
        if rho_type not in {"gaussian", "gaussian_mixture"}:
            raise ValueError(
                f"Invalid distributions.{rho_key}.type='{rho_type}'. "
                "Expected 'gaussian' or 'gaussian_mixture'."
            )
        if rho_type == "gaussian_mixture":
            components = rho_cfg.get("components")
            if not isinstance(components, list) or len(components) == 0:
                raise ValueError(
                    f"distributions.{rho_key}.components must be a non-empty list for gaussian_mixture."
                )
            for i, comp in enumerate(components):
                if not isinstance(comp, dict):
                    raise ValueError(
                        f"distributions.{rho_key}.components[{i}] must be an object."
                    )
                _ensure_no_unknown_keys(
                    comp,
                    ALLOWED_GMM_COMPONENT_KEYS,
                    path=f"distributions.{rho_key}.components[{i}]",
                )

    interactions = problem_cfg.get("interactions")
    if interactions is not None:
        if not isinstance(interactions, dict):
            raise ValueError(f"Field 'interactions' must be an object in {config_path}")
        _ensure_no_unknown_keys(
            interactions, ALLOWED_INTERACTIONS_KEYS, path="interactions"
        )
        for pkey in ("drift_params", "cost_params"):
            params = interactions.get(pkey)
            if params is not None:
                if not isinstance(params, dict):
                    raise ValueError(f"Field 'interactions.{pkey}' must be an object.")
                _ensure_no_unknown_keys(
                    params,
                    ALLOWED_INTERACTION_PARAM_KEYS,
                    path=f"interactions.{pkey}",
                )

    base_functions = problem_cfg.get("base_functions")
    if base_functions is not None:
        if not isinstance(base_functions, dict):
            raise ValueError(
                f"Field 'base_functions' must be an object in {config_path}"
            )
        _ensure_no_unknown_keys(
            base_functions, ALLOWED_BASE_FUNCTIONS_KEYS, path="base_functions"
        )

        base_drift = base_functions.get("drift")
        if base_drift is not None:
            if not isinstance(base_drift, dict):
                raise ValueError("Field 'base_functions.drift' must be an object.")
            _ensure_no_unknown_keys(
                base_drift, ALLOWED_BASE_DRIFT_KEYS, path="base_functions.drift"
            )
            drift_type = base_drift.get("type")
            if drift_type not in ALLOWED_BASE_DRIFT_TYPES:
                raise ValueError(
                    f"Invalid base_functions.drift.type='{drift_type}'. "
                    f"Expected one of {sorted(ALLOWED_BASE_DRIFT_TYPES)}."
                )
            if drift_type == "constant" and "value" not in base_drift:
                raise ValueError(
                    "base_functions.drift.value is required when type='constant'."
                )
        base_cost = base_functions.get("cost")
        if base_cost is not None:
            if not isinstance(base_cost, dict):
                raise ValueError("Field 'base_functions.cost' must be an object.")
            _ensure_no_unknown_keys(
                base_cost, ALLOWED_BASE_COST_KEYS, path="base_functions.cost"
            )
            cost_type = base_cost.get("type")
            if cost_type not in ALLOWED_BASE_COST_TYPES:
                raise ValueError(
                    f"Invalid base_functions.cost.type='{cost_type}'. "
                    f"Expected one of {sorted(ALLOWED_BASE_COST_TYPES)}."
                )
            if cost_type == "multi_obstacle" and "obstacles" not in base_cost:
                raise ValueError(
                    "base_functions.cost.obstacles is required when type='multi_obstacle'."
                )
            if cost_type == "vneck_parabolic":
                for key in ("c_sq", "coef", "penalty"):
                    if key not in base_cost:
                        raise ValueError(
                            f"base_functions.cost.{key} is required when type='vneck_parabolic'."
                        )
            if cost_type == "stunnel_elliptic" and "ellipses" not in base_cost:
                raise ValueError(
                    "base_functions.cost.ellipses is required when type='stunnel_elliptic'."
                )
            if "obstacles" in base_cost:
                obstacles = base_cost["obstacles"]
                if not isinstance(obstacles, list):
                    raise ValueError(
                        "Field 'base_functions.cost.obstacles' must be a list."
                    )
                for i, obs in enumerate(obstacles):
                    if not isinstance(obs, dict):
                        raise ValueError(
                            f"base_functions.cost.obstacles[{i}] must be an object."
                        )
                    _ensure_no_unknown_keys(
                        obs,
                        ALLOWED_OBSTACLE_KEYS,
                        path=f"base_functions.cost.obstacles[{i}]",
                    )
                    mode = obs.get("mode")
                    if mode not in {"hard", "soft"}:
                        raise ValueError(
                            f"base_functions.cost.obstacles[{i}].mode must be 'hard' or 'soft'."
                        )
                    if mode == "soft" and "beta" not in obs:
                        raise ValueError(
                            f"base_functions.cost.obstacles[{i}].beta is required when mode='soft'."
                        )
            if "ellipses" in base_cost:
                ellipses = base_cost["ellipses"]
                if not isinstance(ellipses, list):
                    raise ValueError(
                        "Field 'base_functions.cost.ellipses' must be a list."
                    )
                for i, ellipse in enumerate(ellipses):
                    if not isinstance(ellipse, dict):
                        raise ValueError(
                            f"base_functions.cost.ellipses[{i}] must be an object."
                        )
                    _ensure_no_unknown_keys(
                        ellipse,
                        ALLOWED_ELLIPSE_KEYS,
                        path=f"base_functions.cost.ellipses[{i}]",
                    )

    visualization = problem_cfg.get("visualization")
    if visualization is not None:
        if not isinstance(visualization, dict):
            raise ValueError(
                f"Field 'visualization' must be an object in {config_path}"
            )
        _ensure_no_unknown_keys(
            visualization, ALLOWED_VISUALIZATION_KEYS, path="visualization"
        )

    training = problem_cfg.get("training")
    if not isinstance(training, dict):
        raise ValueError(f"Field 'training' must be an object in {config_path}")
    _ensure_no_unknown_keys(training, ALLOWED_TRAINING_KEYS, path="training")


def load_problem_config(config_path: Path) -> Dict[str, Any]:
    """Load problem configuration from JSON file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Problem config not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    required_fields = ["problem_type", "name", "distributions", "training"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in {config_path}")

    if config["problem_type"] not in VALID_PROBLEM_TYPES:
        raise ValueError(
            f"Invalid problem_type: {config['problem_type']}. "
            f"Must be one of {sorted(VALID_PROBLEM_TYPES)}"
        )

    _validate_problem_config_schema(config, config_path)
    return config


def make_sampling_functions(
    mu_start: torch.Tensor,
    mu_target: torch.Tensor,
    std: float,
    device: torch.device,
) -> Tuple[Callable[[int], torch.Tensor], Callable[[int], torch.Tensor]]:
    """Create sampling functions for source and target distributions.

    Args:
        mu_start: Mean of source distribution ρ₀.
        mu_target: Mean of target distribution ρ₁.
        std: Standard deviation (isotropic Gaussian).
        device: torch device for tensor creation.

    Returns:
        Tuple of (sample_rho0, sample_rho1):
        - sample_rho0: Function that takes batch size and returns samples from ρ₀
        - sample_rho1: Function that takes batch size and returns samples from ρ₁

    Note:
        Both distributions are isotropic Gaussians: N(μ, σ²I).
    """

    def sample_rho0(n: int) -> torch.Tensor:
        return torch.randn(n, mu_start.shape[0], device=device) * std + mu_start

    def sample_rho1(n: int) -> torch.Tensor:
        return torch.randn(n, mu_target.shape[0], device=device) * std + mu_target

    return sample_rho0, sample_rho1


def make_log_prob_functions(
    mu_start: torch.Tensor,
    mu_target: torch.Tensor,
    std: float,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Create log-probability functions for source and target distributions.

    Args:
        mu_start: Mean of source distribution ρ₀.
        mu_target: Mean of target distribution ρ₁.
        std: Standard deviation (isotropic Gaussian).

    Returns:
        Tuple of (log_p0, log_p1):
        - log_p0: Function that computes log ρ₀(x)
        - log_p1: Function that computes log ρ₁(x)

    Note:
        Both distributions are isotropic Gaussians: N(μ, σ²I).
        Log-probability is computed as: -d/2 log(2πσ²) - ‖x-μ‖²/(2σ²)
    """
    d = mu_start.shape[0]
    norm_const = -0.5 * d * np.log(2 * np.pi * std**2)

    def log_p0(x: torch.Tensor) -> torch.Tensor:
        diff = x - mu_start
        return norm_const - 0.5 * torch.sum(diff**2, dim=-1) / (std**2)

    def log_p1(x: torch.Tensor) -> torch.Tensor:
        diff = x - mu_target
        return norm_const - 0.5 * torch.sum(diff**2, dim=-1) / (std**2)

    return log_p0, log_p1


def make_gmm_log_prob_functions(
    rho_0_config: Dict[str, Any],
    rho_1_config: Dict[str, Any],
    device: torch.device,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]
]:
    """Create log-probability functions for Gaussian or Gaussian Mixture distributions.

    Args:
        rho_0_config: Initial distribution config with fields:
            - type: "gaussian" or "gaussian_mixture"
            - For "gaussian": mean, std
            - For "gaussian_mixture": components (list of {mean, std, weight})
        rho_1_config: Target distribution config (same structure as rho_0_config)
        device: PyTorch device for tensors

    Returns:
        Tuple of (log_p0, log_p1):
        - log_p0: Function that computes log ρ₀(x)
        - log_p1: Function that computes log ρ₁(x)

    Note:
        For GMM: log p(x) = log(Σᵢ wᵢ N(x; μᵢ, σᵢ²I))
        Uses log-sum-exp trick for numerical stability.
    """

    def _make_log_prob(
        config: Dict[str, Any],
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        if config["type"] == "gaussian":
            # Single Gaussian
            mean = torch.tensor(config["mean"], device=device, dtype=torch.float32)
            std = config["std"]
            d = len(config["mean"])
            norm_const = -0.5 * d * np.log(2 * np.pi * std**2)

            def log_prob(x: torch.Tensor) -> torch.Tensor:
                diff = x - mean
                return norm_const - 0.5 * torch.sum(diff**2, dim=-1) / (std**2)

            return log_prob

        elif config["type"] == "gaussian_mixture":
            # Gaussian Mixture Model
            components = config["components"]
            n_components = len(components)

            # Prepare component parameters
            means = torch.tensor(
                [comp["mean"] for comp in components],
                device=device,
                dtype=torch.float32,
            )  # (K, d)
            stds = torch.tensor(
                [comp["std"] for comp in components], device=device, dtype=torch.float32
            )  # (K,)
            weights = torch.tensor(
                [comp["weight"] for comp in components],
                device=device,
                dtype=torch.float32,
            )  # (K,)
            log_weights = torch.log(weights)  # (K,)

            d = len(components[0]["mean"])

            def log_prob(x: torch.Tensor) -> torch.Tensor:
                """Compute log p(x) = log(Σᵢ wᵢ N(x; μᵢ, σᵢ²I))"""
                # x: (batch, d)
                batch_size = x.shape[0]

                # Compute log prob for each component
                # diff: (batch, K, d)
                diff = x.unsqueeze(1) - means.unsqueeze(0)  # (batch, 1, d) - (1, K, d)

                # Compute -‖x - μᵢ‖²/(2σᵢ²) for each component
                # (batch, K)
                sq_dist = torch.sum(diff**2, dim=-1) / (2 * stds.unsqueeze(0) ** 2)

                # Normalization constant for each component
                # (K,)
                norm_consts = -0.5 * d * torch.log(2 * np.pi * stds**2)

                # Log probability for each component: log wᵢ + log N(x; μᵢ, σᵢ²I)
                # (batch, K)
                log_probs_per_component = (
                    log_weights.unsqueeze(0) + norm_consts.unsqueeze(0) - sq_dist
                )

                # Log-sum-exp for numerical stability
                # log(Σᵢ wᵢ N(x; μᵢ, σᵢ²I)) = log-sum-exp(log wᵢ + log N(...))
                log_prob_total = torch.logsumexp(log_probs_per_component, dim=1)

                return log_prob_total

            return log_prob

        else:
            raise ValueError(f"Unknown distribution type: {config['type']}")

    log_p0 = _make_log_prob(rho_0_config)
    log_p1 = _make_log_prob(rho_1_config)

    return log_p0, log_p1


def make_gmm_sampling_functions(
    rho_0_config: Dict[str, Any],
    rho_1_config: Dict[str, Any],
    device: torch.device,
) -> Tuple[Callable[[int], torch.Tensor], Callable[[int], torch.Tensor]]:
    """Create sampling functions from Gaussian or Gaussian Mixture config.

    Args:
        rho_0_config: Initial distribution config with fields:
            - type: "gaussian" or "gaussian_mixture"
            - For "gaussian": mean, std
            - For "gaussian_mixture": components (list of {mean, std, weight})
        rho_1_config: Target distribution config (same structure)
        device: torch device for tensor creation

    Returns:
        Tuple of (sample_rho0, sample_rho1):
        - sample_rho0: Function that takes batch size and returns samples
          from ρ₀
        - sample_rho1: Function that takes batch size and returns samples
          from ρ₁

    Example:
        >>> rho0_cfg = {
        ...     "type": "gaussian_mixture",
        ...     "components": [
        ...         {"mean": [-3, -3], "std": 0.5, "weight": 0.5},
        ...         {"mean": [3, 3], "std": 0.5, "weight": 0.5}
        ...     ]
        ... }
        >>> sample_rho0, _ = make_gmm_sampling_functions(
        ...     rho0_cfg, rho0_cfg, torch.device("cpu")
        ... )
        >>> samples = sample_rho0(100)  # (100, 2)
    """

    def sample_gaussian(
        n: int,
        mean: List[float],
        std: float,
    ) -> torch.Tensor:
        """Sample n points from single Gaussian."""
        dim = len(mean)
        mean_t = torch.tensor(mean, device=device, dtype=torch.float32)
        samples = torch.randn(n, dim, device=device) * std + mean_t
        return samples

    def sample_gmm(
        n: int,
        components: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """Sample n points from Gaussian mixture."""
        # Extract component parameters
        means = [comp["mean"] for comp in components]
        stds = [comp["std"] for comp in components]
        weights = torch.tensor([comp["weight"] for comp in components])

        # Sample component indices
        component_indices = torch.multinomial(weights, n, replacement=True)

        # Sample from each component
        dim = len(means[0])
        samples = torch.zeros(n, dim, device=device)

        for i in range(n):
            k = int(component_indices[i].item())
            mean = torch.tensor(means[k], device=device, dtype=torch.float32)
            std = float(stds[k])
            sample = torch.randn(dim, device=device) * std + mean
            samples[i] = sample

        return samples

    # Create sampling function for rho_0
    if rho_0_config["type"] == "gaussian":
        mean_0 = rho_0_config["mean"]
        std_0 = rho_0_config["std"]
        sample_rho0 = lambda n: sample_gaussian(n, mean_0, std_0)
    elif rho_0_config["type"] == "gaussian_mixture":
        components_0 = rho_0_config["components"]
        sample_rho0 = lambda n: sample_gmm(n, components_0)
    else:
        raise ValueError(f"Unsupported rho_0 type: {rho_0_config['type']}")

    # Create sampling function for rho_1
    if rho_1_config["type"] == "gaussian":
        mean_1 = rho_1_config["mean"]
        std_1 = rho_1_config["std"]
        sample_rho1 = lambda n: sample_gaussian(n, mean_1, std_1)
    elif rho_1_config["type"] == "gaussian_mixture":
        components_1 = rho_1_config["components"]
        sample_rho1 = lambda n: sample_gmm(n, components_1)
    else:
        raise ValueError(f"Unsupported rho_1 type: {rho_1_config['type']}")

    return sample_rho0, sample_rho1


def setup_problem_distributions(
    problem_cfg: Dict[str, Any], config: Any
) -> Dict[str, Any]:
    """Setup distributions for a problem configuration.

    This function handles all supported problem types (gaussian_transport,
    gmm_navigation, vneck_navigation, stunnel_navigation) and creates
    the appropriate sampling functions, log-probability functions, and
    extracts necessary metadata like GMM components.

    Args:
        problem_cfg: Problem configuration dictionary with 'problem_type' and
                    'distributions' keys.
        config: Training configuration object with 'device' attribute.

    Returns:
        Dictionary containing:
            - mu_0: torch.Tensor - Mean/center of source distribution
            - mu_1: torch.Tensor - Mean/center of target distribution
            - sample_rho0: Callable - Source distribution sampling function
            - sample_rho1: Callable - Target distribution sampling function
            - log_p0: Callable - Source log-probability function
            - log_p1: Callable - Target log-probability function
            - mu_0_components: Optional[List] - Source GMM component means (or None)
            - mu_1_components: Optional[List] - Target GMM component means (or None)

    Raises:
        ValueError: If problem_type is not supported.

    Example:
        >>> dist_setup = setup_problem_distributions(problem_cfg, config)
        >>> samples = dist_setup['sample_rho0'](100)
        >>> log_prob = dist_setup['log_p0'](samples)
    """
    problem_type = problem_cfg["problem_type"]
    rho_0_cfg = problem_cfg["distributions"]["rho_0"]
    rho_1_cfg = problem_cfg["distributions"]["rho_1"]

    # Initialize optional components
    mu_0_components = None
    mu_1_components = None

    if problem_type in ("opinion_2d", "opinion_1k"):
        # Opinion models may use anisotropic Gaussian endpoints or mixtures.
        # Build torch distributions on CPU to avoid MPS Cholesky limitations;
        # samples/log-probabilities are moved to the configured training device.
        def _mean_from_cfg(cfg: Dict[str, Any], default_dim: int) -> torch.Tensor:
            base_mean = torch.tensor(cfg["mean"], device="cpu", dtype=torch.float32)
            dim_target = int(cfg.get("dim", default_dim))
            if base_mean.numel() == 1 and dim_target > 1:
                return torch.full((dim_target,), float(base_mean.item()), device="cpu")
            mean_cpu = base_mean.reshape(-1)
            if mean_cpu.numel() != dim_target:
                raise ValueError(
                    f"Opinion distribution mean has dimension {mean_cpu.numel()}, "
                    f"expected {dim_target}."
                )
            return mean_cpu

        def _covariance_from_cfg(
            cfg: Dict[str, Any], dim_target: int
        ) -> torch.Tensor:
            if "covar_override" in cfg:
                covar_cpu = torch.tensor(
                    cfg["covar_override"], device="cpu", dtype=torch.float32
                )
                if covar_cpu.numel() == 1:
                    return covar_cpu.item() * torch.eye(dim_target, device="cpu")
                if covar_cpu.ndim == 1:
                    if covar_cpu.numel() != dim_target:
                        raise ValueError(
                            "covar_override vector length does not match opinion dimension."
                        )
                    return torch.diag(covar_cpu)
                if covar_cpu.shape != (dim_target, dim_target):
                    raise ValueError(
                        f"covar_override has shape {tuple(covar_cpu.shape)}, "
                        f"expected {(dim_target, dim_target)}."
                    )
                return covar_cpu
            if "covar_diag" in cfg:
                covar_diag = torch.tensor(
                    cfg["covar_diag"], device="cpu", dtype=torch.float32
                ).reshape(-1)
                if covar_diag.numel() == 1:
                    covar_diag = covar_diag.repeat(dim_target)
                if covar_diag.numel() != dim_target:
                    raise ValueError(
                        "covar_diag length does not match opinion dimension."
                    )
                return torch.diag(covar_diag)
            std = float(cfg["std"])
            return (std**2) * torch.eye(dim_target, device="cpu")

        def _make_mvn_from_cfg(
            cfg: Dict[str, Any], default_dim: int
        ) -> tuple[torch.Tensor, torch.distributions.MultivariateNormal]:
            mean_cpu = _mean_from_cfg(cfg, default_dim)
            covar_cpu = _covariance_from_cfg(cfg, mean_cpu.numel())
            return mean_cpu, torch.distributions.MultivariateNormal(
                mean_cpu, covariance_matrix=covar_cpu
            )

        def setup_opinion_distribution(
            cfg: Dict[str, Any],
        ) -> tuple[
            Callable[[int], Tensor],
            Callable[[Tensor], Tensor],
            torch.Tensor,
            Optional[list],
        ]:
            default_dim = int(getattr(config, "dim", cfg.get("dim", 1)))
            if cfg["type"] == "gaussian":
                mean_cpu, mvn_cpu = _make_mvn_from_cfg(cfg, default_dim)

                def sampler(n: int) -> Tensor:
                    return mvn_cpu.sample((n,)).to(config.device)

                def log_prob(x: Tensor) -> Tensor:
                    return mvn_cpu.log_prob(x.to("cpu")).to(config.device)

                return sampler, log_prob, mean_cpu.to(config.device), None

            if cfg["type"] == "gaussian_mixture":
                components = cfg["components"]
                weights_cpu = torch.tensor(
                    [comp["weight"] for comp in components],
                    device="cpu",
                    dtype=torch.float32,
                )
                weights_cpu = weights_cpu / weights_cpu.sum()
                mvns_cpu = []
                means_cpu = []
                for comp in components:
                    mean_cpu, mvn_cpu = _make_mvn_from_cfg(comp, default_dim)
                    means_cpu.append(mean_cpu)
                    mvns_cpu.append(mvn_cpu)
                means_stack = torch.stack(means_cpu, dim=0)

                def sampler(n: int) -> Tensor:
                    component_indices = torch.multinomial(
                        weights_cpu, n, replacement=True
                    )
                    samples_cpu = torch.empty(
                        n, means_stack.shape[1], device="cpu", dtype=torch.float32
                    )
                    for k, mvn_cpu in enumerate(mvns_cpu):
                        mask = component_indices == k
                        count = int(mask.sum().item())
                        if count > 0:
                            samples_cpu[mask] = mvn_cpu.sample((count,))
                    return samples_cpu.to(config.device)

                def log_prob(x: Tensor) -> Tensor:
                    x_cpu = x.to("cpu")
                    log_weights = torch.log(weights_cpu)
                    per_comp = [
                        log_weights[k] + mvn_cpu.log_prob(x_cpu)
                        for k, mvn_cpu in enumerate(mvns_cpu)
                    ]
                    return torch.logsumexp(torch.stack(per_comp, dim=0), dim=0).to(
                        config.device
                    )

                mixture_mean = (weights_cpu[:, None] * means_stack).sum(dim=0)
                components_out = [mean.cpu().tolist() for mean in means_cpu]
                return (
                    sampler,
                    log_prob,
                    mixture_mean.to(config.device),
                    components_out,
                )

            raise ValueError(f"Unsupported opinion distribution type: {cfg['type']}")

        sample_rho0, log_p0, mu_0, mu_0_components = setup_opinion_distribution(
            rho_0_cfg
        )
        sample_rho1, log_p1, mu_1, mu_1_components = setup_opinion_distribution(
            rho_1_cfg
        )

    elif problem_type == "gaussian_transport":
        # Standard Gaussian transport
        mu_0 = torch.tensor(
            rho_0_cfg["mean"], device=config.device, dtype=torch.float32
        )
        mu_1 = torch.tensor(
            rho_1_cfg["mean"], device=config.device, dtype=torch.float32
        )
        sigma_dist = rho_0_cfg["std"]

        sample_rho0, sample_rho1 = make_sampling_functions(
            mu_0, mu_1, sigma_dist, config.device
        )
        log_p0, log_p1 = make_log_prob_functions(mu_0, mu_1, sigma_dist)

    elif problem_type == "gmm_navigation":
        # Gaussian Mixture Model navigation
        sample_rho0, sample_rho1 = make_gmm_sampling_functions(
            rho_0_cfg, rho_1_cfg, config.device
        )

        # Create proper GMM log-probability functions
        log_p0, log_p1 = make_gmm_log_prob_functions(
            rho_0_cfg, rho_1_cfg, config.device
        )

        # Extract mu_0 and mu_0_components
        if rho_0_cfg["type"] == "gaussian":
            mu_0 = torch.tensor(
                rho_0_cfg["mean"], device=config.device, dtype=torch.float32
            )
        else:
            means_0 = [comp["mean"] for comp in rho_0_cfg["components"]]
            mu_0 = torch.tensor(means_0[0], device=config.device, dtype=torch.float32)
            mu_0_components = means_0

        # Extract mu_1 and mu_1_components
        if rho_1_cfg["type"] == "gaussian":
            mu_1 = torch.tensor(
                rho_1_cfg["mean"], device=config.device, dtype=torch.float32
            )
        else:
            means_1 = [comp["mean"] for comp in rho_1_cfg["components"]]
            mu_1 = torch.tensor(means_1[0], device=config.device, dtype=torch.float32)
            mu_1_components = means_1

    elif problem_type in ("vneck_navigation", "stunnel_navigation"):
        # V-neck or S-tunnel navigation (both use simple Gaussian)
        mu_0 = torch.tensor(
            rho_0_cfg["mean"], device=config.device, dtype=torch.float32
        )
        mu_1 = torch.tensor(
            rho_1_cfg["mean"], device=config.device, dtype=torch.float32
        )
        sigma_dist = rho_0_cfg["std"]

        sample_rho0, sample_rho1 = make_sampling_functions(
            mu_0, mu_1, sigma_dist, config.device
        )
        log_p0, log_p1 = make_log_prob_functions(mu_0, mu_1, sigma_dist)

    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    return {
        "mu_0": mu_0,
        "mu_1": mu_1,
        "sample_rho0": sample_rho0,
        "sample_rho1": sample_rho1,
        "log_p0": log_p0,
        "log_p1": log_p1,
        "mu_0_components": mu_0_components,
        "mu_1_components": mu_1_components,
    }


def create_multi_obstacle_cost(
    obstacles: List[Dict[str, Any]],
    device: torch.device,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Create combined obstacle cost from list of obstacles."""
    hard_costs = []
    soft_costs = []
    for obs_dict in obstacles:
        center = torch.tensor(obs_dict["center"], device=device, dtype=torch.float32)
        radius = obs_dict["radius"]
        penalty = obs_dict["penalty"]
        mode = obs_dict.get("mode")
        if mode is None:
            raise ValueError("Obstacle 'mode' is required ('hard' or 'soft').")
        if mode not in {"hard", "soft"}:
            raise ValueError(
                f"Unknown obstacle mode '{mode}'. Expected 'hard' or 'soft'."
            )
        if mode == "hard":
            obstacle = ObstacleCost(center, radius, penalty)
            hard_costs.append(obstacle)
        else:
            if "beta" not in obs_dict:
                raise ValueError("Obstacle 'beta' is required when mode='soft'.")
            beta = float(obs_dict["beta"])
            obstacle = SmoothObstacleCost(center, radius, penalty, beta=beta)
            soft_costs.append(obstacle)

    def multi_obstacle_cost_fn(xs: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """Evaluate combined obstacle cost for batch."""
        batch_size = xs.shape[0]
        total_cost = torch.zeros(batch_size, device=xs.device)

        if hard_costs:
            for i in range(batch_size):
                for obs in hard_costs:
                    total_cost[i] += obs(xs[i])

        for obs in soft_costs:
            total_cost = total_cost + obs(xs)

        return total_cost

    return multi_obstacle_cost_fn


def setup_problem_base_functions(
    problem_cfg: Dict[str, Any],
    config: Any,
    mu_0: torch.Tensor,
    mu_1: torch.Tensor,
) -> Dict[str, Any]:
    """Build base drift/cost callables from problem config."""
    base_functions_cfg = problem_cfg.get("base_functions", {})
    obstacle_config = None
    base_drift_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    base_cost_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None

    if "cost" in base_functions_cfg:
        base_cost_cfg = base_functions_cfg["cost"]
        obstacle_config = base_cost_cfg
        cost_type = base_cost_cfg.get("type")
        if cost_type == "zero":
            base_cost_fn = None
        elif cost_type == "multi_obstacle":
            base_cost_fn = create_multi_obstacle_cost(
                base_cost_cfg["obstacles"], config.device
            )
        elif cost_type == "vneck_parabolic":
            c_sq = base_cost_cfg["c_sq"]
            coef = base_cost_cfg["coef"]
            penalty = base_cost_cfg["penalty"]
            vneck_cost = VNeckObstacleCost(c_sq=c_sq, coef=coef, penalty=penalty)

            def base_cost_fn(xs: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
                return vneck_cost(xs)

        elif cost_type == "stunnel_elliptic":
            stunnel_cost = STunnelObstacleCost(base_cost_cfg["ellipses"])

            def base_cost_fn(xs: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
                return stunnel_cost(xs)
        else:
            raise ValueError(
                f"Unknown base_functions.cost.type in problem config: '{cost_type}'"
            )

    if "drift" in base_functions_cfg:
        base_drift_cfg = base_functions_cfg["drift"]
        drift_type = base_drift_cfg["type"]
        if drift_type == "zero":
            base_drift_fn = None
        elif drift_type == "constant":
            drift_value = base_drift_cfg["value"]
            base_drift_fn = ConstantDrift(drift_value)
        else:
            raise ValueError(
                f"Unknown base_functions.drift.type in problem config: '{drift_type}'"
            )

    return {
        "obstacle_config": obstacle_config,
        "base_drift_fn": base_drift_fn,
        "base_cost_fn": base_cost_fn,
    }


def setup_problem_artifacts(
    config: Any,
    problem_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Callable[[int], torch.Tensor],
    Callable[[int], torch.Tensor],
    Optional[list],
    Optional[list],
    Optional[Dict[str, Any]],
    Optional[Tuple[float, float]],
    Optional[Tuple[float, float]],
    Optional[Callable[[torch.Tensor], torch.Tensor]],
    Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Recreate problem distributions, base functions, and visualization metadata."""
    if problem_cfg is None:
        problem_config_path = getattr(config, "problem_config_path", None)
        if problem_config_path is None:
            raise ValueError(
                "Problem configuration path is required on config. "
                "Callers must set config.problem_config_path before setup."
            )
        problem_cfg = load_problem_config(Path(problem_config_path))

    dist_setup = setup_problem_distributions(problem_cfg, config)
    mu_0 = dist_setup["mu_0"]
    mu_1 = dist_setup["mu_1"]
    sample_rho0 = dist_setup["sample_rho0"]
    sample_rho1 = dist_setup["sample_rho1"]
    mu_0_components = dist_setup["mu_0_components"]
    mu_1_components = dist_setup["mu_1_components"]
    log_p0 = dist_setup["log_p0"]
    log_p1 = dist_setup["log_p1"]

    base_setup = setup_problem_base_functions(problem_cfg, config, mu_0, mu_1)
    vis_setup = setup_problem_visualization(problem_cfg)
    obstacle_config = base_setup["obstacle_config"]
    base_drift_fn = base_setup["base_drift_fn"]
    base_cost_fn = base_setup["base_cost_fn"]
    xlim = vis_setup["xlim"]
    ylim = vis_setup["ylim"]
    return (
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
        log_p0,
        log_p1,
    )
