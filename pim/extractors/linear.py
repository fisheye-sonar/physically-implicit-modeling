"""Linear extractor — a single linear layer from hidden state to env state."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import StateDefinition


class LinearExtractor(nn.Module):
    """Maps hidden state → env state via a single linear layer.

    Compatible with exact least-squares fitting (fit_lstsq) and with
    pseudoinverse-based hidden-state editing (probe_steering).

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the input hidden state.
    state_def : StateDefinition
        Defines the output shape and target quantity.
    """

    def __init__(self, hidden_size: int, state_def: StateDefinition) -> None:
        super().__init__()
        self.state_def = state_def
        self.linear = nn.Linear(hidden_size, state_def.output_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Map hidden states to decoded env state.

        Parameters
        ----------
        h : (..., hidden_size)

        Returns
        -------
        decoded : (..., *state_shape)
        """
        flat = self.linear(h)  # (..., output_dim)
        return flat.reshape(*h.shape[:-1], *self.state_def.state_shape)
