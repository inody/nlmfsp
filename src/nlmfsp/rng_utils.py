"""Utilities for phase-local RNG control."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import torch


def seed_offset(seed: Optional[int], offset: int) -> Optional[int]:
    """Return a deterministic offset from a base seed while preserving None."""
    if seed is None:
        return None
    return int(seed) + int(offset)


@contextmanager
def torch_seed_context(seed: Optional[int]) -> Iterator[None]:
    """Temporarily set and later restore global torch RNG state."""
    if seed is None:
        yield
        return

    cpu_state = torch.random.get_rng_state()
    cuda_available = torch.cuda.is_available()
    cuda_states = torch.cuda.get_rng_state_all() if cuda_available else None

    torch.manual_seed(seed)
    if cuda_available:
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_available and cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
