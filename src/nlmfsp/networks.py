"""Neural network architectures for NLMFSP.

Parts of this module are adapted from DeepGSB (Liu et al., NeurIPS 2022),
https://github.com/ghliu/DeepGSB, which is distributed under the Apache License
2.0. Specifically:

- ``timestep_embedding`` follows ``models/util.py`` closely, with shape
  validation added and the odd-dimension branch removed;
- ``ToyValueNet`` and ``ToyGradientNet`` generalize the ``ToyY`` and ``ToyZ``
  architectures of ``models/toy_net.py``, replacing fixed layer widths with
  configuration-driven ones and adding optional time embedding and time
  normalization. The layer topology is otherwise the same, which
  ``tests/test_toy_networks.py`` checks by comparing parameter counts against
  the upstream reference.

See NOTICE for the full attribution.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .config import Config


def _process_time_input(
    t: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    """Process physical-time input with optional normalization and embedding.

    Args:
        t: Physical time tensor (batch_size,), (batch_size, 1), or scalar
        config: Configuration object

    Returns:
        Processed time tensor:
            - (batch_size, 1) if no embedding
            - (batch_size, time_embed_dim) if using embedding
    """
    if t.dim() == 2 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    elif t.dim() not in (0, 1):
        raise ValueError(f"t must be scalar, 1D, or shape (batch, 1); got {t.shape}")

    if config.time_input_mode == "normalized":
        t_processed = t / config.T * config.time_index_interval
    elif config.time_input_mode == "raw":
        t_processed = t
    else:
        raise ValueError(f"Unknown time_input_mode: {config.time_input_mode}")

    # Process time
    if config.use_time_embedding:
        return timestep_embedding(t_processed, config.time_embed_dim)
    else:
        return t_processed.unsqueeze(-1)


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.

    Adapted from DeepGSB ``models/util.py`` (Apache License 2.0). The frequency
    bands and the ``[cos, sin]`` concatenation are unchanged; this version
    validates the input shape and requires an even dimension.

    Based on Transformer positional encoding and used in Diffusion Models.
    This encoding helps the network distinguish between different time steps
    by mapping a scalar time value to a high-dimensional vector.

    Reference:
        - DeepGSB (Liu et al., NeurIPS 2022)
        - Denoising Diffusion Probabilistic Models (Ho et al., NeurIPS 2020)
        - Attention Is All You Need (Vaswani et al., NeurIPS 2017)

    Args:
        timesteps: (batch_size,) or (batch_size, 1) tensor of timestep values
        dim: Embedding dimension (must be even)
        max_period: Maximum period for sinusoidal functions (default: 10000)

    Returns:
        Embedding tensor of shape (batch_size, dim)

    Example:
        >>> t = torch.tensor([0.0, 0.5, 1.0])
        >>> emb = timestep_embedding(t, dim=128)
        >>> emb.shape
        torch.Size([3, 128])

    Note:
        The embedding uses the formula:
            PE(t, 2i)   = sin(t / max_period^(2i/dim))
            PE(t, 2i+1) = cos(t / max_period^(2i/dim))
        This creates a smooth, unique representation for each timestep.
    """
    if dim % 2 != 0:
        raise ValueError(f"Embedding dimension must be even, got {dim}")

    # Ensure timesteps is 1D
    if timesteps.dim() == 2:
        timesteps = timesteps.squeeze(-1)
    elif timesteps.dim() != 1:
        raise ValueError(f"timesteps must be 1D or 2D, got shape {timesteps.shape}")

    # Compute frequency bands:
    # [max_period^(-0/half), ..., max_period^(-(half-1)/half)]
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)

    # Compute arguments for sin and cos: (batch_size, half)
    args = timesteps[:, None].float() * freqs[None]

    # Concatenate [cos, sin] to get (batch_size, dim)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    return embedding


def _get_activation(activation: str) -> nn.Module:
    """Get activation module from string name.

    Args:
        activation: Activation function name ('relu', 'tanh', 'elu', 'silu')

    Returns:
        Activation module instance
    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown activation: {activation}")


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...],
        activation: str = "relu",
    ):
        """Initialize MLP.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: Tuple of hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'elu')
        """
        super().__init__()

        # Build layers
        layers = []
        dims = [input_dim] + list(hidden_dims)

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(_get_activation(activation))

        # Output layer (no activation)
        layers.append(nn.Linear(dims[-1], output_dim))

        self.net = nn.Sequential(*layers)

        # Apply Xavier/Glorot initialization for better stability with tanh
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization for stability."""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (..., input_dim)

        Returns:
            Output tensor (..., output_dim)
        """
        return self.net(x)


class InteractionNet(nn.Module):
    """Neural network for learning interaction terms f_θ(x,t) or F_φ(x,t).

    This network approximates mean-field interactions:
        f_θ(x,t) ≈ f(x,t,ρ(·,t))  (mean-field drift)
        F_φ(x,t) ≈ F(x,t,ρ(·,t))  (mean-field running cost)

    Following Han et al., we use (x,t) as input to achieve O(N) complexity.
    """

    def __init__(
        self,
        config: Config,
        output_dim: Optional[int] = None,
    ):
        """Initialize interaction network.

        Args:
            config: Configuration object
            output_dim: Output dimension
                (default: config.dim for f_θ, 1 for F_φ)
        """
        super().__init__()
        self.dim = config.dim
        self.output_dim = output_dim if output_dim is not None else config.dim
        activation = config.interaction_activation or config.activation
        self.impl = ToyInteractionNet(config, self.output_dim, activation)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: State (batch_size, dim)
            t: Time (batch_size,) or scalar

        Returns:
            Output (batch_size, output_dim)
        """
        return self.impl(x, t)


def _resolve_interaction_hidden_dims(config: Config) -> Tuple[int, ...]:
    if config.interaction_hidden_dims is not None:
        return tuple(config.interaction_hidden_dims)
    if isinstance(config.hidden_dims, tuple):
        return config.hidden_dims
    return tuple(config.hidden_dims)


def _resolve_interaction_time_dims(
    config: Config, x_dims: Tuple[int, ...]
) -> Tuple[int, ...]:
    if config.interaction_t_hidden_dims is not None:
        return tuple(config.interaction_t_hidden_dims)
    return x_dims


class ToyInteractionNet(nn.Module):
    """Dual-branch interaction network inspired by DeepGSB ToyY/ToyZ."""

    def __init__(self, config: Config, output_dim: int, activation: str):
        super().__init__()
        self.config = config
        self.output_dim = output_dim
        self.use_time_embedding = config.use_time_embedding
        self.activation = activation

        x_hidden_dims = _resolve_interaction_hidden_dims(config)
        t_hidden_dims = _resolve_interaction_time_dims(config, x_hidden_dims)

        time_input_dim = config.time_embed_dim if self.use_time_embedding else 1

        self.x_branch, x_out_dim = self._build_branch(config.dim, x_hidden_dims)
        self.t_branch, t_out_dim = self._build_branch(time_input_dim, t_hidden_dims)

        concat_dim = x_out_dim + t_out_dim
        head_hidden = (concat_dim,)
        self.head = MLP(
            input_dim=concat_dim,
            output_dim=output_dim,
            hidden_dims=head_hidden,
            activation=activation,
        )

    def _build_branch(
        self, input_dim: int, hidden_dims: Tuple[int, ...]
    ) -> Tuple[nn.Module, int]:
        if hidden_dims is None or len(hidden_dims) == 0:
            return nn.Identity(), input_dim

        layers: List[nn.Module] = []
        in_dim = input_dim
        for idx, hid in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hid))
            if idx != len(hidden_dims) - 1:
                layers.append(_get_activation(self.activation))
            in_dim = hid
        branch = nn.Sequential(*layers) if layers else nn.Identity()
        out_dim = hidden_dims[-1] if len(hidden_dims) > 0 else input_dim
        return branch, out_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        elif t.shape[0] != x.shape[0]:
            raise ValueError(f"Shape mismatch: x {x.shape}, t {t.shape}")

        t_input = _process_time_input(t, self.config)

        x_feat = self.x_branch(x)
        t_feat = self.t_branch(t_input)
        xt = torch.cat([x_feat, t_feat], dim=-1)
        return self.head(xt)


class ValueNet(nn.Module):
    """Neural network for value functions Y(x,t) or Ŷ(x,t).

    In Liu's formulation:
        Y_t represents the forward value function
        Ŷ_t represents the backward value function

    Supports both simple MLP and DeepGSB-style architectures:
    - use_time_embedding=False: Direct concatenation [x, t]
    - use_time_embedding=True: Sinusoidal embedding [x, emb(t)]
    """

    def __init__(self, config: Config):
        """Initialize value network.

        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.use_time_embedding = config.use_time_embedding

        # Determine input dimensions
        if self.use_time_embedding:
            time_input_dim = config.time_embed_dim
        else:
            time_input_dim = 1  # Raw time scalar

        # Input: (x, t_processed) where t_processed is either R or R^128
        input_dim = config.dim + time_input_dim

        self.net = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
        )

        # Use PyTorch default initialization (works better for this task)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: State (batch_size, dim)
            t: Time (batch_size,) or scalar

        Returns:
            Value (batch_size, 1)
        """
        # Ensure t has the right shape
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        elif t.shape[0] != x.shape[0]:
            raise ValueError(f"Shape mismatch: x {x.shape}, t {t.shape}")

        # Process time with optional normalization and embedding
        t_processed = _process_time_input(t, self.config)

        # Concatenate (x, t_processed)
        xt = torch.cat([x, t_processed], dim=-1)
        return self.net(xt)


class GradientNet(nn.Module):
    """Neural network for gradient/Z functions Z(x,t) or Ẑ(x,t).

    In Liu's formulation:
        Z_t = ∇Y_t represents the gradient of forward value
        Ẑ_t = ∇Ŷ_t represents the gradient of backward value

    Supports both simple MLP and DeepGSB-style architectures:
    - use_time_embedding=False: Direct concatenation [x, t]
    - use_time_embedding=True: Sinusoidal embedding [x, emb(t)]
    """

    def __init__(self, config: Config):
        """Initialize gradient network.

        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.use_time_embedding = config.use_time_embedding

        # Determine input dimensions
        if self.use_time_embedding:
            time_input_dim = config.time_embed_dim
        else:
            time_input_dim = 1  # Raw time scalar

        # Input: (x, t_processed)
        input_dim = config.dim + time_input_dim

        self.net = MLP(
            input_dim=input_dim,
            output_dim=config.dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
        )

        # Use PyTorch default initialization (works better for this task)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: State (batch_size, dim)
            t: Time (batch_size,) or scalar

        Returns:
            Gradient (batch_size, dim)
        """
        # Ensure t has the right shape
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        elif t.shape[0] != x.shape[0]:
            raise ValueError(f"Shape mismatch: x {x.shape}, t {t.shape}")

        # Process time with optional normalization and embedding
        t_processed = _process_time_input(t, self.config)

        # Concatenate (x, t_processed)
        xt = torch.cat([x, t_processed], dim=-1)
        return self.net(xt)


# ============================================================================
# DeepGSB Toy Networks (Dual-Stream Architecture)
# ============================================================================


class ToyValueNet(nn.Module):
    """Value network generalizing the DeepGSB ``ToyY`` architecture.

    Adapted from DeepGSB ``models/toy_net.py`` (Apache License 2.0): the
    time/space branches, their concatenation and the three-layer head follow
    ``ToyY``, with fixed widths replaced by configuration-driven ones.

    Processes time and space in separate branches before combining via concatenation.

    Architecture:
        - Time branch: time_embed_dim → t_hidden_dims
        - Space branch: dim → hidden_dims
        - Combination: concatenation [x_out, t_out]
        - Output module: 3 layers (hardcoded per DeepGSB)

    Reference: Liu et al. (2022) Deep Generalized Schrödinger Bridge
    """

    def __init__(self, config: Config):
        """Initialize ToyY-style value network.

        Args:
            config: Configuration object with:
                - hidden_dims: Space branch layer sizes
                - t_hidden_dims: Time branch layer sizes (None = use hidden_dims)
                - time_embed_dim: Sinusoidal embedding dimension (if use_time_embedding=True)
                - use_time_embedding: Whether to use sinusoidal time embedding
                - activation: Activation function name
                - dim: State space dimension
        """
        super().__init__()
        self.config = config
        self.use_time_embedding = config.use_time_embedding

        # Determine time input dimension
        if self.use_time_embedding:
            time_input_dim = config.time_embed_dim
        else:
            time_input_dim = 1  # Raw time scalar

        # Determine branch configurations
        x_hidden_dims = list(config.hidden_dims)
        t_hidden_dims = (
            list(config.t_hidden_dims)
            if config.t_hidden_dims is not None
            else list(config.hidden_dims)
        )

        # Build time branch
        t_layers = []
        t_in = time_input_dim
        for t_hid in t_hidden_dims:
            t_layers.append(nn.Linear(t_in, t_hid))
            t_layers.append(_get_activation(config.activation))
            t_in = t_hid
        # Remove last activation
        self.t_module = nn.Sequential(*t_layers[:-1])

        # Build space branch
        x_layers = []
        x_in = config.dim
        for x_hid in x_hidden_dims:
            x_layers.append(nn.Linear(x_in, x_hid))
            x_layers.append(_get_activation(config.activation))
            x_in = x_hid
        # Remove last activation
        self.x_module = nn.Sequential(*x_layers[:-1])

        # Output module (hardcoded: 3 layers for ToyY)
        t_out_dim = t_hidden_dims[-1]
        x_out_dim = x_hidden_dims[-1]
        concat_dim = t_out_dim + x_out_dim

        self.out_module = nn.Sequential(
            nn.Linear(concat_dim, x_out_dim),
            _get_activation(config.activation),
            nn.Linear(x_out_dim, x_out_dim),
            _get_activation(config.activation),
            nn.Linear(x_out_dim, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: State (batch_size, dim)
            t: Time (batch_size,) or scalar

        Returns:
            Value (batch_size, 1)
        """
        # Handle scalar time input
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        elif t.shape[0] != x.shape[0]:
            raise ValueError(f"Shape mismatch: x {x.shape}, t {t.shape}")

        t_input = _process_time_input(t, self.config)

        # Process time and space separately
        t_out = self.t_module(t_input)
        x_out = self.x_module(x)

        # Concatenate and pass through output module
        xt_out = torch.cat([x_out, t_out], dim=1)
        return self.out_module(xt_out)


class ToyGradientNet(nn.Module):
    """Gradient network generalizing the DeepGSB ``ToyZ`` architecture.

    Adapted from DeepGSB ``models/toy_net.py`` (Apache License 2.0): the
    time/space branches, their summation and the two-layer head follow
    ``ToyZ``, with fixed widths replaced by configuration-driven ones.

    Processes time and space in separate branches before combining via addition.

    Architecture:
        - Time branch: time_embed_dim → t_hidden_dims
        - Space branch: dim → hidden_dims
        - Combination: addition x_out + t_out (residual-like)
        - Output module: 2 layers (hardcoded per DeepGSB)

    Note: Requires t_hidden_dims[-1] == hidden_dims[-1] for addition.

    Reference: Liu et al. (2022) Deep Generalized Schrödinger Bridge
    """

    def __init__(self, config: Config):
        """Initialize ToyZ-style gradient network.

        Args:
            config: Configuration object with:
                - hidden_dims: Space branch layer sizes
                - t_hidden_dims: Time branch layer sizes (None = use hidden_dims)
                - time_embed_dim: Sinusoidal embedding dimension (if use_time_embedding=True)
                - use_time_embedding: Whether to use sinusoidal time embedding
                - activation: Activation function name
                - dim: State space dimension

        Raises:
            ValueError: If output dimensions don't match for addition
        """
        super().__init__()
        self.config = config
        self.use_time_embedding = config.use_time_embedding

        # Determine time input dimension
        if self.use_time_embedding:
            time_input_dim = config.time_embed_dim
        else:
            time_input_dim = 1  # Raw time scalar

        # Determine branch configurations
        x_hidden_dims = list(config.hidden_dims)
        t_hidden_dims = (
            list(config.t_hidden_dims)
            if config.t_hidden_dims is not None
            else list(config.hidden_dims)
        )

        # Build time branch (same structure as ToyValueNet)
        t_layers = []
        t_in = time_input_dim
        for t_hid in t_hidden_dims:
            t_layers.append(nn.Linear(t_in, t_hid))
            t_layers.append(_get_activation(config.activation))
            t_in = t_hid
        self.t_module = nn.Sequential(*t_layers[:-1])

        # Build space branch (same structure as ToyValueNet)
        x_layers = []
        x_in = config.dim
        for x_hid in x_hidden_dims:
            x_layers.append(nn.Linear(x_in, x_hid))
            x_layers.append(_get_activation(config.activation))
            x_in = x_hid
        self.x_module = nn.Sequential(*x_layers[:-1])

        # Check dimensions match for addition
        t_out_dim = t_hidden_dims[-1]
        x_out_dim = x_hidden_dims[-1]
        if t_out_dim != x_out_dim:
            raise ValueError(
                f"ToyGradientNet requires matching output dims for addition: "
                f"t_hidden_dims[-1]={t_out_dim} != hidden_dims[-1]={x_out_dim}. "
                f"Please adjust hidden_dims or t_hidden_dims."
            )

        # Output module (hardcoded: 2 layers for ToyZ)
        combined_dim = x_out_dim
        self.out_module = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            _get_activation(config.activation),
            nn.Linear(combined_dim, config.dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: State (batch_size, dim)
            t: Time (batch_size,) or scalar

        Returns:
            Gradient (batch_size, dim)
        """
        # Handle scalar time input
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        elif t.shape[0] != x.shape[0]:
            raise ValueError(f"Shape mismatch: x {x.shape}, t {t.shape}")

        t_input = _process_time_input(t, self.config)

        # Process time and space separately
        t_out = self.t_module(t_input)
        x_out = self.x_module(x)

        # Addition (residual-like) and pass through output module
        return self.out_module(x_out + t_out)
