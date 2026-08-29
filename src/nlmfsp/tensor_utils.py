"""Utility functions for tensor manipulation and batch processing."""

from typing import List

import torch


def flatten_dim01(x: torch.Tensor) -> torch.Tensor:
    """Flatten first two dimensions: (d0, d1, ...) -> (d0*d1, ...).

    Useful for converting trajectory data from (batch, time, ...) to
    a flat batch dimension (batch*time, ...) for vectorized processing.

    Args:
        x: Tensor with shape (d0, d1, d2, ...)

    Returns:
        Tensor with shape (d0*d1, d2, ...)

    Example:
        >>> x = torch.randn(4, 5, 2)  # (batch, time, dim)
        >>> x_flat = flatten_dim01(x)
        >>> x_flat.shape
        torch.Size([20, 2])  # (batch*time, dim)
    """
    return x.reshape(-1, *x.shape[2:])


def unflatten_dim01(x: torch.Tensor, shape: List[int]) -> torch.Tensor:
    """Unflatten first dimension: (d0*d1, ...) -> (d0, d1, ...).

    Inverse operation of flatten_dim01. Useful for converting flat batch
    back to (batch, time, ...) format.

    Args:
        x: Tensor with shape (d0*d1, d2, ...)
        shape: Target shape [d0, d1] for first two dimensions

    Returns:
        Tensor with shape (d0, d1, d2, ...)

    Example:
        >>> x_flat = torch.randn(20, 2)  # (batch*time, dim)
        >>> x = unflatten_dim01(x_flat, [4, 5])
        >>> x.shape
        torch.Size([4, 5, 2])  # (batch, time, dim)
    """
    return x.reshape(*shape, *x.shape[1:])


def reverse_cumsum(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Compute cumulative sum in reverse order along specified dimension.

    This is useful for backward TD learning where we accumulate rewards
    from the end of trajectory to the beginning.

    Args:
        x: Input tensor
        dim: Dimension along which to compute reverse cumsum

    Returns:
        Tensor with same shape as input, containing reverse cumulative sums

    Example:
        >>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> reverse_cumsum(x, dim=1)
        tensor([[ 6,  5,  3],
                [15, 11,  6]])
    """
    return torch.flip(torch.cumsum(torch.flip(x, dims=[dim]), dim=dim), dims=[dim])


def batch_trajectory_forward(
    xs: torch.Tensor, ts: torch.Tensor, include_boundary: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare trajectory data for vectorized forward processing.

    Converts trajectory from (T+1, B, nx) format to flat (T*B, nx) for
    efficient batch processing through networks.

    Args:
        xs: Trajectory states (T+1, batch, dim) or (T, batch, dim)
        ts: Time grid (T+1,) or (T,)
        include_boundary: If True, include final state; if False, exclude it

    Returns:
        xs_flat: Flattened states (T*B, dim)
        ts_flat: Repeated times (T*B,)

    Example:
        >>> xs = torch.randn(51, 128, 2)  # 50 steps + 1 boundary
        >>> ts = torch.linspace(0, 1, 51)
        >>> xs_flat, ts_flat = batch_trajectory_forward(xs, ts)
        >>> xs_flat.shape
        torch.Size([6400, 2])  # 50 * 128
    """
    batch_size = xs.shape[1]

    if not include_boundary:
        # Exclude final state (typical for TD loss)
        xs = xs[:-1]  # (T, B, nx)
        ts = ts[:-1]  # (T,)

    # Transpose to (B, T, nx) then flatten
    xs_transposed = xs.transpose(0, 1)  # (B, T, nx)
    xs_flat = flatten_dim01(xs_transposed)  # (B*T, nx)

    # Repeat times for each batch element
    ts_flat = ts.repeat_interleave(batch_size)  # (B*T,)

    return xs_flat, ts_flat


def unflatten_trajectory(
    x_flat: torch.Tensor, batch_size: int, n_steps: int
) -> torch.Tensor:
    """Convert flat trajectory back to (batch, time, ...) format.

    Args:
        x_flat: Flat tensor (B*T, ...)
        batch_size: Number of trajectories
        n_steps: Number of time steps

    Returns:
        Unflattened tensor (B, T, ...)
    """
    return unflatten_dim01(x_flat, [batch_size, n_steps])


__all__ = [
    "flatten_dim01",
    "unflatten_dim01",
    "reverse_cumsum",
    "batch_trajectory_forward",
    "unflatten_trajectory",
]
