"""
Base drift functions (no interaction)
"""

from typing import Union

import numpy as np
import torch


class ConstantDrift:
    """
    f(x) = b (constant drift vector)

    Always returns torch.Tensor for compatibility with FBSDESolver.
    """

    def __init__(self, b):
        if isinstance(b, (list, tuple)):
            self.b = torch.tensor(b, dtype=torch.float32)
        elif isinstance(b, np.ndarray):
            self.b = torch.from_numpy(b).float()
        else:
            self.b = b

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        b = self.b.to(x.device)
        return b.expand_as(x)


class ZeroBaseDrift:
    """
    Zero drift: f(x) = 0

    Args:
        None (no parameters required)
    """

    def __call__(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            x: Current position (np.ndarray or torch.Tensor)

        Returns:
            Zero drift with same type and shape as input
        """
        if torch.is_tensor(x):
            return torch.zeros_like(x)
        else:
            return np.zeros_like(x)
