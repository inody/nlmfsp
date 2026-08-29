"""Configuration management for NLMFSP."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch

TIME_INPUT_MODES = {"raw", "normalized"}


def migrate_legacy_config_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return a constructor-ready config dict with legacy time semantics.

    Historical experiment configs did not contain ``time_input_mode``. Even when
    they serialized ``use_time_index_normalization=true``, the active Toy-network
    path still used raw physical time. Missing ``time_input_mode`` therefore
    means a legacy raw-time run.
    """
    migrated = dict(config_dict)
    migrated.pop("n_time_steps", None)
    if migrated.get("time_input_mode") is None:
        migrated["time_input_mode"] = "raw"
    return migrated


def ensure_config_time_input_fields(config: Any) -> Any:
    """Patch Config-like objects loaded from old checkpoints in-place."""
    if not hasattr(config, "time_input_mode") or config.time_input_mode is None:
        config.time_input_mode = "raw"
    if config.time_input_mode not in TIME_INPUT_MODES:
        raise ValueError(
            "time_input_mode must be one of: " + ", ".join(sorted(TIME_INPUT_MODES))
        )
    config.use_time_index_normalization = config.time_input_mode == "normalized"
    if not hasattr(config, "detach_interaction_drift_divergence"):
        config.detach_interaction_drift_divergence = False
    return config


@dataclass
class Config:
    """Configuration for Nonlocal Mean Field Schrödinger Bridge solver.

    Attributes:
        # Problem parameters
        dim: State space dimension
        T: Terminal time
        dt: Time step size
        n_time_steps: Number of time steps (computed from T/dt)

        # SDE parameters
        sigma: Diffusion coefficient (assumed constant)

        # Problem definition
        drift_type: Type of interaction drift (mean_reversion, zero, gaussian, etc.)
        cost_type: Type of interaction cost (zero, quadratic_deviation, gaussian_conv,
            gaussian_conv_mean, pedestrian_visual, etc.)
        int_drift_weight: Weight for interaction drift
        int_drift_sigma: Sigma for interaction drift (gaussian, nonstationary_anisotropic, laplacian)
        int_drift_strength: Strength for interaction drift (coulomb, attraction_repulsion)
        int_cost_weight: Weight for interaction cost
        int_cost_sigma: Range scale for interaction cost (gaussian_conv,
            gaussian_conv_mean, pedestrian_visual)
        int_cost_strength: Strength for interaction cost (coulomb, local_exclusion)
        base_drift_type: Legacy serialized base-drift field retained for compatibility.
            Active runs take base drift from problem JSON `base_functions`.
        base_drift_value: Legacy serialized constant base-drift value.
        base_cost_type: Legacy serialized base-cost field retained for compatibility.
            Active runs take base cost from problem JSON `base_functions`.
        base_cost_weight: Legacy serialized base-cost weight.

        # Training parameters
        n_agents: Number of agents in the mean field (N)
        n_outer_iterations: Number of fictitious play iterations (K) - for iteration-based training
        n_inner_steps: Number of gradient steps per iteration/stage
        n_interaction_steps: Number of gradient steps for interaction learning
        standardize_interaction_targets: Train learned interactions on standardized
            targets and unstandardize their outputs during simulation
        interaction_input_scale: Divide interaction-network spatial inputs by this
            positive scalar during both training and simulation
        learning_rate_interaction: Learning rate for interaction networks

        # Network architecture
        hidden_dims: Hidden layer dimensions for networks
        activation: Activation function name ('relu', 'tanh', 'elu')

        # Loss weights (Liu et al.)
        lambda_ipf: Weight for LIPF (Iterative Proportional Fitting) loss
        lambda_td: Weight for LTD (Temporal Difference) loss
        lambda_fk: Weight for LFK (Fokker-Planck) loss
        lambda_boundary: Weight for boundary loss

        # Optimization (Phase 5)
        lr_decay_gamma: Learning rate decay factor for ExponentialLR
        grad_clip: Gradient clipping threshold (None = no clipping)

        # Logging (Phase 5)
        log_interval: Log metrics every N outer iterations
        checkpoint_interval: Save checkpoint every N outer iterations
        verbose: If True, print progress to console

        # Paths (Phase 5)
        output_dir: Root directory for outputs
        log_dir: Directory for log files
        checkpoint_dir: Directory for model checkpoints

        # Device
        device: torch device ('cpu' or 'cuda')
    """

    # Problem parameters
    dim: int = 2
    T: float = 1.0
    dt: float = 0.01

    # Training parameters - New unified parameters (v2.0)
    n_agents: Optional[int] = None  # Mean Field agent count (N) - both modes
    n_agents_test: Optional[int] = None  # Particle count for evaluation/plots

    # Interaction learning mode
    interaction_backend: str = "exact"  # "exact", "nn", or "random_batch"
    learn_interaction_drift: bool = False  # Learn only the drift interaction network
    learn_interaction_cost: bool = False  # Learn only the cost interaction network
    random_batch_size: Optional[int] = None  # Reference subset size M for random-batch interactions
    evaluate_interaction_cpu: bool = (
        False  # Run analytical evaluation on CPU to reduce GPU memory use
    )
    resample_interaction_sample: bool = False  # Resample trajectories at each interaction step (Alg. {alg:fourstep_alt_resample})
    interaction_query_chunk_size: Optional[int] = (
        256  # Chunk size over query particles for analytical O(N^2) interactions
    )
    interaction_ref_chunk_size: Optional[int] = (
        1024  # Chunk size over reference particles for analytical O(N^2) interactions
    )
    interaction_time_block_size: Optional[int] = (
        None  # Optional number of time slices to batch together for exact time-local analytical interactions
    )
    problem_config_path: Optional[str] = (
        None  # Path to the source problem JSON used to reconstruct artifacts
    )

    n_outer_iterations: int = 10  # Number of outer IPF iterations
    n_inner_steps: int = 1000  # Number of gradient steps per iteration/stage

    # Interaction learning
    n_interaction_steps: int = 500
    standardize_interaction_targets: bool = False
    interaction_input_scale: float = 1.0
    learning_rate_interaction: Optional[float] = None
    learning_rate_value: Optional[float] = (
        1e-4  # Learning rate for value networks (Y/Ŷ)
    )
    learning_rate_grad: Optional[float] = (
        1e-4  # Learning rate for gradient networks (Z/Ẑ)
    )

    # Network architecture
    hidden_dims: Tuple[int, ...] = (
        64,
        64,
        64,
    )  # Space branch (x) for toy, all layers for mlp
    t_hidden_dims: Optional[Tuple[int, ...]] = (
        None  # Time branch (t) for toy; None = use hidden_dims
    )
    activation: str = "relu"
    interaction_hidden_dims: Optional[Tuple[int, ...]] = (
        None  # Space branch sizes for toy interaction
    )
    interaction_t_hidden_dims: Optional[Tuple[int, ...]] = (
        None  # Time branch sizes for toy interaction
    )
    interaction_activation: Optional[str] = (
        None  # Optional override for interaction nets
    )

    # Time embedding configuration (DeepGSB compatibility)
    use_time_embedding: bool = False  # Enable sinusoidal timestep embedding
    time_embed_dim: int = 128  # Embedding dimension (when enabled)
    time_input_mode: str = "raw"  # "raw" or "normalized" physical-time input
    use_time_index_normalization: bool = False  # Normalize time: t → t/T * interval
    time_index_interval: float = 100.0  # Interval for time normalization

    # Loss weights
    lambda_ipf: float = 1.0
    lambda_td: float = 1.0
    lambda_fk: float = 0.1

    # Boundary loss weight
    lambda_boundary: float = 0.0

    # FK loss (grad loss) options
    fk_loss_norm: str = "l1"  # "l1", "l2", or "huber" (DeepGSB uses "l1" for GMM)

    # Temporal-difference options
    td_horizon: int = 1  # Number of rollout steps for multi-step TD (1 = single-step)
    td_bootstrap_from_target: bool = (
        True  # If True, stop gradients through bootstrap target
    )
    use_multistep_td_loss: bool = False  # Enable DeepGSB-style multi-step TD loss

    # Optimizer options for FBSDE networks
    optimizer_type: str = (
        "adam"  # Optimizer for FBSDE nets (adam, adamw, sgd, rmsprop, adagrad)
    )
    optimizer_momentum: Optional[float] = (
        None  # Momentum (used for SGD/RMSprop when specified)
    )
    optimizer_weight_decay: float = (
        0.0  # Weight decay applied to optimizer param groups
    )

    # Learning rate scheduler options
    lr_scheduler_type: str = "exponential"  # "exponential", "step", or "none"
    lr_decay_step: Optional[int] = None  # Step size for StepLR (ignored otherwise)

    # EMA options
    ema_decay: float = 0.0  # 0 disables EMA; otherwise 0 < decay < 1
    ema_start_step: int = 0  # Global optimization step to start EMA updates
    ema_update_interval: int = 1  # Update EMA every N optimizer steps

    # SDE parameters
    sigma: float = 1.0

    # Problem definition parameters (drift/cost types and their parameters)
    drift_type: str = "zero"  # Type of interaction drift
    cost_type: str = "zero"  # Type of interaction cost

    # Interaction drift parameters
    int_drift_weight: Optional[float] = 1.0  # Weight for mean_reversion, etc.
    int_drift_sigma: Optional[float] = 1.0  # Sigma for gaussian, nonstationary_anisotropic, laplacian
    int_drift_strength: Optional[float] = (
        None  # Strength for nonstationary_anisotropic, coulomb, attraction_repulsion
    )

    # Interaction cost parameters
    int_cost_weight: Optional[float] = 1.0  # Weight for quadratic_deviation, etc.
    # Range for gaussian_conv / gaussian_conv_mean / pedestrian_visual
    int_cost_sigma: Optional[float] = 1.0
    int_cost_strength: Optional[float] = None  # Strength for coulomb, local_exclusion

    # Legacy serialized base-function fields. Active runtime behavior comes from
    # problem JSON `base_functions`, not these config attributes.
    base_drift_type: str = "zero"
    base_drift_value: Optional[list] = None

    base_cost_type: str = "zero"
    base_cost_weight: Optional[float] = None

    # Numerical stability
    n_hutchinson_samples: int = 1  # Number of samples for Hutchinson's trace estimator (1=original, 3=variance reduction)
    detach_interaction_drift_divergence: bool = (
        False  # If True, exclude learned interaction drift derivatives from divergence terms
    )

    # Optimization (Phase 5)
    lr_decay_gamma: float = 0.999  # Learning rate decay factor
    grad_clip: Optional[float] = None  # Gradient clipping value (None = no clipping)

    # Logging (Phase 5)
    log_interval: int = 1  # Log metrics every N iterations
    checkpoint_interval: int = 10  # Save checkpoint every N iterations
    verbose: bool = True  # Print progress to console

    # Paths (Phase 5)
    output_dir: str = "outputs"
    log_dir: str = "outputs/logs"
    checkpoint_dir: str = "outputs/checkpoints"

    # Experiment management
    experiment_root: str = "outputs/experiments"
    experiment_id: Optional[str] = None  # Auto-generated experiment ID

    # Device
    device: Union[str, torch.device] = "cpu"

    # Random seed
    seed: Optional[int] = 42
    training_seed: Optional[int] = None
    validation_seed: Optional[int] = None
    visualization_seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Compute derived quantities and handle parameter setup."""
        # Set default parameters if not provided
        self._set_default_parameters()

        # Validate and auto-adjust parameters
        self._validate_and_adjust_parameters()

        # Determine number of FBSDE time steps from dt and T
        # Round to nearest integer to avoid truncation errors
        # Then adjust dt to ensure exact arrival at T
        self.n_time_steps = int(round(self.T / self.dt))
        self.dt = self.T / self.n_time_steps

        # Convert device string to torch.device
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        if self.time_input_mode not in TIME_INPUT_MODES:
            raise ValueError(
                "time_input_mode must be one of: "
                + ", ".join(sorted(TIME_INPUT_MODES))
            )
        # Keep the legacy flag as a readable alias for serialization/logging.
        self.use_time_index_normalization = self.time_input_mode == "normalized"
        if self.interaction_input_scale <= 0:
            raise ValueError("interaction_input_scale must be positive")

        if self.td_horizon < 1:
            raise ValueError("td_horizon must be >= 1")
        if self.ema_decay < 0.0 or self.ema_decay >= 1.0:
            raise ValueError("ema_decay must satisfy 0 <= ema_decay < 1")
        if self.ema_decay > 0.0:
            if self.ema_start_step < 0:
                raise ValueError("ema_start_step must be >= 0")
            if self.ema_update_interval < 1:
                raise ValueError("ema_update_interval must be >= 1")
        if self.interaction_query_chunk_size is not None:
            if self.interaction_query_chunk_size <= 0:
                raise ValueError("interaction_query_chunk_size must be positive")
        if self.interaction_ref_chunk_size is not None:
            if self.interaction_ref_chunk_size <= 0:
                raise ValueError("interaction_ref_chunk_size must be positive")
        if self.interaction_time_block_size is not None:
            if self.interaction_time_block_size <= 0:
                raise ValueError("interaction_time_block_size must be positive")
        if self.random_batch_size is not None and self.random_batch_size <= 0:
            raise ValueError("random_batch_size must be positive")
        if self.interaction_backend not in {"exact", "nn", "random_batch"}:
            raise ValueError(
                "interaction_backend must be one of: exact, nn, random_batch"
            )
        if self.interaction_backend == "exact":
            if self.learn_interaction:
                raise ValueError(
                    "interaction_backend='exact' does not allow learn_interaction_* flags."
                )
        elif self.interaction_backend == "nn":
            if not self.learn_interaction:
                raise ValueError(
                    "interaction_backend='nn' requires learn_interaction_drift or learn_interaction_cost."
                )
        elif self.interaction_backend == "random_batch":
            if self.learn_interaction:
                raise ValueError(
                    "interaction_backend='random_batch' does not allow learn_interaction_* flags."
                )
            if self.random_batch_size is None:
                raise ValueError(
                    "interaction_backend='random_batch' requires random_batch_size."
                )

        if self.training_seed is None:
            self.training_seed = self.seed
        if self.validation_seed is None:
            self.validation_seed = self.seed
        if self.visualization_seed is None:
            self.visualization_seed = self.seed

        if self.training_seed is not None:
            torch.manual_seed(self.training_seed)

    def _set_default_parameters(self):
        """Validate required particle counts and derive dependent values."""
        if self.n_agents is None:
            raise ValueError("n_agents must be specified explicitly.")
        if self.n_agents_test is None:
            self.n_agents_test = self.n_agents

    def _validate_and_adjust_parameters(self):
        """Validate parameter consistency and auto-adjust."""
        return None

    @property
    def learn_interaction(self) -> bool:
        """Derived convenience flag indicating whether any interaction learning is enabled."""
        return self.learn_interaction_drift or self.learn_interaction_cost

    @property
    def time_grid(self) -> torch.Tensor:
        """Return time grid from 0 to T."""
        return torch.linspace(0, self.T, self.n_time_steps + 1, device=self.device)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization.

        Uses dataclasses.asdict() for automatic field conversion and handles
        torch.device serialization. Excludes computed fields like n_time_steps.

        Returns:
            Dictionary representation of config parameters suitable for JSON
        """
        # dataclasses.asdict()で自動変換
        config_dict = asdict(self)

        # 計算済みフィールドと非シリアライズ可能なフィールドを除外
        excluded_fields = {
            "n_time_steps",  # Tとdtから計算
            "device",  # torch.deviceはJSONシリアライズ不可
        }

        # deviceフィールドを文字列に変換
        result = {}
        for k, v in config_dict.items():
            if k in excluded_fields:
                if k == "device" and v is not None:
                    result[k] = str(v)  # deviceを文字列に変換
                # n_time_stepsは除外
            else:
                result[k] = v

        return result
