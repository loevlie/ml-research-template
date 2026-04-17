"""Reproducibility utilities.

Rule 19: Every run is reproducible. This module seeds all RNGs and
configures PyTorch for deterministic execution.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, *, deterministic: bool = True) -> None:
    """Seed all random number generators for reproducibility.

    Sets seeds for: Python stdlib, NumPy, PyTorch CPU, PyTorch CUDA (all GPUs),
    and PYTHONHASHSEED. Optionally enables fully deterministic execution.

    Args:
        seed: Integer seed value.
        deterministic: If True, enables cuDNN deterministic mode and
            disables cuDNN benchmark autotuning. Slower but reproducible.
            Use True during development; consider False for final large runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn for reproducible multi-worker loading.

    Usage:
        DataLoader(..., worker_init_fn=seed_worker, generator=g)
        where g = torch.Generator().manual_seed(seed)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
