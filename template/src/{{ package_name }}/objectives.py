"""Training objectives.

An Objective bundles the forward pass + loss computation in one callable so the
training loop stays a tiny orchestrator. Each Objective owns:
  - how to call the model on a batch (batch structure is objective-specific)
  - what loss to return
  - (optionally) what intermediates to surface for metrics/logging

Swap objectives from the CLI:  `python train.py loss=contrastive`
Config classes live in `configs.py` (GROUPS registers the variants).

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
2. Add a config class with a `build()` in `configs.py` and register it in
   `GROUPS["loss"]` — `loss=<name>` then works on every entry point.
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

    def __call__(self, model: nn.Module, batch: Any) -> dict[str, Any]: ...


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


class ContrastiveObjective:
    """Symmetric InfoNCE (CLIP-style) over paired views.

    Works for two modalities (image/text) or two augmented views of one
    modality (SimCLR-style tabular/image SSL). Expects:

      - batch: a pair ``(x_a, x_b)`` where row i of x_a matches row i of x_b
      - model: callable as ``model(x_a, x_b) -> (z_a, z_b)``, two [B, D]
        embedding tensors (e.g. a two-tower module)

    Pair with a DataLoader that yields view pairs (see data/augmentations.py).

    Args:
        temperature: Softmax temperature for the similarity logits.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        self.temperature = temperature

    def __call__(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> dict[str, Any]:
        x_a, x_b = batch
        z_a, z_b = model(x_a, x_b)
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)

        logits = (z_a @ z_b.T) / self.temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
        # No "logits"/"targets" keys: classification accuracy doesn't apply.
        return {"loss": loss}
