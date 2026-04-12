"""Example model.

Replace this with your actual architecture. Demonstrates shape-annotated
signatures with jaxtyping + beartype.
"""

import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, Int
from torch import Tensor


class ExampleModel(nn.Module):
    """Simple MLP classifier -- replace with your architecture.

    Args:
        n_features: Input feature dimensionality.
        hidden_dim: Hidden layer size.
        n_classes: Number of output classes.
    """

    def __init__(
        self,
        n_features: int = 32,
        hidden_dim: int = 128,
        n_classes: int = 10,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    @beartype
    def forward(
        self, x: Float[Tensor, "batch features"]
    ) -> Float[Tensor, "batch classes"]:
        return self.net(x)
