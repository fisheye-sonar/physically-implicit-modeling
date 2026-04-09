"""MLP extractor — a 2-layer MLP from hidden state to env state."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import StateDefinition


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
        Width of the hidden layer.
    """

    def __init__(
        self,
        hidden_size: int,
        state_def: StateDefinition,
        mlp_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.state_def = state_def
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, state_def.output_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Map hidden states to decoded env state.

        Parameters
        ----------
        h : (..., hidden_size)

        Returns
        -------
        decoded : (..., *state_shape)
        """
        flat = self.net(h)  # (..., output_dim)
        return flat.reshape(*h.shape[:-1], *self.state_def.state_shape)
