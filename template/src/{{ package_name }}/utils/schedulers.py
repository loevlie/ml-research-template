"""Learning-rate schedulers.

One factory, stepped once per optimizer step (not per epoch). Configured via
`trainer.scheduler` in configs/trainer/default.yaml; `name: none` keeps the
constant-LR behavior.
"""

import math

import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str = "none",
    *,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    """Build a per-step LR scheduler.

    Args:
        optimizer: The optimizer to schedule.
        name: "none" (constant LR, returns None), "cosine" (linear warmup then
            cosine decay to min_lr_ratio * base_lr), or "constant" (linear
            warmup then constant).
        total_steps: Total optimizer steps over the run (epochs * steps/epoch).
        warmup_steps: Steps to linearly ramp LR from 0 to base.
        min_lr_ratio: Floor for cosine decay, as a fraction of the base LR.

    Returns:
        A LambdaLR scheduler, or None for "none".
    """
    if name == "none":
        return None

    def warmup(step: int) -> float:
        return (step + 1) / max(1, warmup_steps)

    if name == "constant":

        def lr_lambda(step: int) -> float:
            return warmup(step) if step < warmup_steps else 1.0

    elif name == "cosine":

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return warmup(step)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    else:
        raise ValueError(f"Unknown scheduler: {name!r}. Use 'none', 'cosine', or 'constant'.")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
