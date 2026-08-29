"""
Nonlocal Mean Field Schrödinger Bridge (NLMFSP)

A deep learning framework for solving McKean-Vlasov FBSDEs with general
mean-field interactions using Schrödinger Bridge formulation.

References:
- Liu et al. (2022): Deep Generalized Schrödinger Bridge
- Han et al. (2023): Learning High-Dimensional McKean-Vlasov FBSDEs
"""

__version__ = "0.1.0"

# Keep top-level imports reasonably light but expose the symbols
# that examples and users expect to import from `nlmfsp`.

# CLI helpers commonly used by scripts/examples
from .cli_helpers import (apply_interaction_config_overrides,
                          apply_problem_config_overrides, build_config,
                          get_default_device, resolve_device)
# Core configuration and components
from .config import Config
from .experiment_manager import ExperimentManager
from .fbsde import FBSDESolver
from .interaction import create_interactions
# Problem setup utilities (distribution sampling)
from .problem_setup import (make_gmm_log_prob_functions,
                            make_gmm_sampling_functions,
                            make_log_prob_functions, make_sampling_functions,
                            setup_problem_distributions)
from .trainer import Trainer
from .validation import (evaluate_transport_quality, log_configuration,
                         log_memory_profile, summarize_metrics)
# Visualization and validation utilities
from .visualization import (create_realtime_visualization_callback,
                            plot_metrics, plot_opinion_directional_similarity,
                            plot_opinion_snapshots, plot_trajectories,
                            sample_visualization_trajectories)

__all__ = [
    # Core
    "Config",
    "FBSDESolver",
    "Trainer",
    "ExperimentManager",
    # CLI helpers
    "apply_problem_config_overrides",
    "apply_interaction_config_overrides",
    "build_config",
    "create_interactions",
    "log_configuration",
    "get_default_device",
    "resolve_device",
    # Problem setup (distributions)
    "make_sampling_functions",
    "make_log_prob_functions",
    "make_gmm_sampling_functions",
    "make_gmm_log_prob_functions",
    "setup_problem_distributions",
    # Visualization
    "plot_metrics",
    "plot_trajectories",
    "plot_opinion_snapshots",
    "plot_opinion_directional_similarity",
    "create_realtime_visualization_callback",
    "sample_visualization_trajectories",
    # Validation
    "summarize_metrics",
    "evaluate_transport_quality",
    "log_memory_profile",
]
