"""Linear extractor — a single linear layer from hidden state to env state."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .base import StateDefinition
from .matching import identity_mse
from .training import fit_lstsq, train_extractor


class LinearExtractor(nn.Module):
    """Maps hidden state → env state via a single linear layer.

    Compatible with exact least-squares fitting (use_lstsq=True) and with
    pseudoinverse-based hidden-state editing (probe_steering).

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the input hidden state.
    state_def : StateDefinition
        Defines the output shape and target quantity.
    use_lstsq : bool
        If True, fit() solves the closed-form least-squares problem.
        If False, fit() does gradient descent.
    n_epochs, lr : training hyperparameters when use_lstsq=False.
    """

    def __init__(
        self,
        hidden_size: int,
        state_def: StateDefinition,
        *,
        use_lstsq: bool = True,
        n_epochs: int = 30,
        lr: float = 5e-3,
    ) -> None:
        super().__init__()
        self.state_def = state_def
        self.linear = nn.Linear(hidden_size, state_def.output_dim)
        self.use_lstsq = use_lstsq
        self.n_epochs = n_epochs
        self.lr = lr

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        flat = self.linear(h)
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
        """Train this probe in place. Returns final train MSE."""
        if self.use_lstsq:
            return fit_lstsq(self, internal_states, env_states_gt, mask=mask)
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
