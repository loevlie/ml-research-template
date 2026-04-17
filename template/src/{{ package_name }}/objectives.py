"""Training objectives.

An Objective bundles the forward pass + loss computation in one callable so the
training loop stays a tiny orchestrator. Each Objective owns:
  - how to call the model on a batch (batch structure is objective-specific)
  - what loss to return
  - (optionally) what intermediates to surface for metrics/logging

Swap objectives via Hydra:  `python train.py loss=contrastive`
Configs live in `configs/loss/`.

Contract
--------
An Objective is any callable with the signature:

    def __call__(self, model: nn.Module, batch: Any) -> dict[str, Any]

The returned dict MUST contain a scalar tensor under key `"loss"`. Anything
else is optional. Two conventional keys that enable automatic metrics:

  - `"logits"`  — raw classifier outputs
  - `"targets"` — integer class labels

When both are present, the validation loop computes argmax-based accuracy
automatically. Objectives for regression, contrastive, or masked prediction
just omit these keys and the accuracy branch stays inactive.

Adding a new objective
----------------------
1. Define a class with `__call__(model, batch) -> dict`.
2. Create `configs/loss/<name>.yaml` with `_target_: <pkg>.objectives.YourObj`.
3. If your batch structure differs from `(x, y)`, pair with a matching
   DataLoader (contrastive → pairs; masked-prediction → (x, mask, y); etc.).
"""

from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


class Objective(Protocol):
    """Minimal interface every objective satisfies.

    Duck-typed — no need to inherit. Any class with a matching `__call__`
    works as an Objective.
    """

    def __call__(self, model: nn.Module, batch: Any) -> dict[str, Any]:
        ...


class SupervisedObjective:
    """Cross-entropy supervised classification.

    Args:
        label_smoothing: Passed to F.cross_entropy (0.0 = off).
    """

    def __init__(self, label_smoothing: float = 0.0) -> None:
        self.label_smoothing = label_smoothing

    def __call__(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> dict[str, Any]:
        x, y = batch
        logits = model(x)
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        return {"loss": loss, "logits": logits, "targets": y}
