"""Reproducibility utilities for research experiments.

Provides functions to seed all random number generators across Python,
NumPy, and PyTorch (CPU and CUDA) and toggle deterministic backend behaviors.
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seed all pseudo-random number generators for reproducible research results.

    Args:
        seed (int): The integer seed value to use across all libraries. Defaults to 42.
        deterministic (bool): If True, configures cuDNN and PyTorch to enforce
            deterministic algorithm execution. Note that this may incur a minor
            performance trade-off. Defaults to True.

    Returns:
        int: The seed that was set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Enforce deterministic convolution algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            # Enforce deterministic algorithms across PyTorch operations (supported in PyTorch >= 1.8)
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass
    else:
        torch.backends.cudnn.benchmark = True

    return seed
