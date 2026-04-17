"""Explicit training loop primitives.

train.py orchestrates setup and invokes these. Keep them free of Hydra config
parsing — they accept plain args so they're testable without Hydra in the loop.

One objective, one loop — swap objectives (supervised / contrastive / masked /
next-state) via Hydra without touching this file.
"""

from typing import Any

import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_epoch(
    fabric: L.Fabric,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    objective: Any,
    train_loader: DataLoader,
    epoch: int,
) -> float:
    """Run one training epoch.

    Args:
        fabric: Lightning Fabric (for logging + backward).
        model: The model to train.
        optimizer: The optimizer.
        objective: Any callable with signature (model, batch) -> dict containing "loss".
        train_loader: Training dataloader (already wrapped by Fabric).
        epoch: Current epoch number.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        out = objective(model, batch)
        loss = out["loss"]
        fabric.backward(loss)
        fabric.clip_gradients(model, optimizer, max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 10 == 0:
            fabric.log("train/loss_step", loss.item(), step=epoch * len(train_loader) + batch_idx)

    avg_loss = total_loss / n_batches
    fabric.log("train/loss_epoch", avg_loss, step=epoch)
    return avg_loss


@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: nn.Module,
    objective: Any,
    val_loader: DataLoader,
    epoch: int,
) -> tuple[float, float]:
    """Run validation.

    Computes loss always. Computes argmax-based accuracy automatically iff the
    objective emits both `logits` and `targets` in its output dict. For other
    metrics (retrieval@k, regression MSE, per-dataset rank), extend this
    function or compute them externally from the saved checkpoint.

    Args:
        fabric: Lightning Fabric (for logging).
        model: The model to evaluate.
        objective: Same callable shape as in train_epoch.
        val_loader: Validation dataloader (already wrapped by Fabric).
        epoch: Current epoch number.

    Returns:
        Tuple of (average val loss, accuracy). Accuracy is 0.0 when the
        objective doesn't emit logits/targets.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in val_loader:
        out = objective(model, batch)
        total_loss += out["loss"].item()
        if "logits" in out and "targets" in out:
            logits, targets = out["logits"], out["targets"]
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total if total > 0 else 0.0

    fabric.log("val/loss", avg_loss, step=epoch)
    if total > 0:
        fabric.log("val/acc", accuracy, step=epoch)
    return avg_loss, accuracy
