"""MLP extractor — a 2-layer MLP from hidden state to env state."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .base import StateDefinition
from .matching import identity_mse
from .training import train_extractor


class MLPExtractor(nn.Module):
    """Maps hidden state → env state via a 2-layer MLP.

    More expressive than LinearExtractor but not compatible with
    pseudoinverse-based editing (use LinearExtractor for probe_steering).

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the input hidden state.
    state_def : StateDefinition
        Defines the output shape and target quantity.
    mlp_hidden : int
        Width of each hidden layer.
    n_hidden_layers : int
        Number of hidden layers. **Default 1 reproduces the original architecture
        exactly** (same `state_dict` keys), so existing frozen probes — notably the
        one the MLP Grad Steering editor writes through — are unaffected.
        For *reporting* readability R² use `pim.extractors.fit_readability_probes`,
        which standardises this at 2 and scores on held-out sequences.
    n_epochs, lr : gradient-descent training hyperparameters.
    """

    def __init__(
        self,
        hidden_size: int,
        state_def: StateDefinition,
        *,
        mlp_hidden: int = 128,
        n_hidden_layers: int = 1,
        n_epochs: int = 30,
        lr: float = 5e-3,
    ) -> None:
        super().__init__()
        self.state_def = state_def
        if n_hidden_layers < 1:
            raise ValueError(f"n_hidden_layers must be >= 1, got {n_hidden_layers}")
        layers: list[nn.Module] = [nn.Linear(hidden_size, mlp_hidden), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU()]
        layers.append(nn.Linear(mlp_hidden, state_def.output_dim))
        self.net = nn.Sequential(*layers)
        self.n_epochs = n_epochs
        self.lr = lr

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        flat = self.net(h)
        return flat.reshape(*h.shape[:-1], *self.state_def.state_shape)

    def fit(
        self,
        internal_states: np.ndarray,
        env_states_gt: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        loss_fn: Callable = identity_mse,
        device: str = "cpu",
    ) -> float:
        """Train this probe in place via gradient descent. Returns final train loss."""
        losses = train_extractor(
            self,
            internal_states,
            env_states_gt,
            n_epochs=self.n_epochs,
            lr=self.lr,
            loss_fn=loss_fn,
            mask=mask,
            device=device,
        )
        return losses[-1]
