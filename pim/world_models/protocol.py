"""Model interface protocols.

These Protocols define the expected interface for world models without
imposing any inheritance. Models only need to implement the methods;
type checkers will validate conformance structurally.

WorldModel       — minimal contract: forward + step (any predictive model)
HiddenStateModel — extended contract: also exposes hidden_size + get_hidden_states
                   (models whose internal state can be probed / steered)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class WorldModel(Protocol):
    """Minimal contract — any model that predicts next observations."""

    def forward(
        self,
        obs: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forcing forward pass.

        Parameters
        ----------
        obs : (B, T, R) observation sequence
        h0  : optional initial state

        Returns
        -------
        pred     : (B, T-1, R) predicted next observations
        state_out: final model state (shape model-dependent)
        """
        ...

    def step(
        self,
        obs_t: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step autoregressive forward.

        Parameters
        ----------
        obs_t : (B, R) current observation
        state : current model state, or None for initial state

        Returns
        -------
        pred_t    : (B, R) predicted next observation
        state_next: updated model state (shape model-dependent)
        """
        ...


@runtime_checkable
class HiddenStateModel(WorldModel, Protocol):
    """Extended contract — models that expose a probe-compatible hidden state.

    Only models that implement this interface are compatible with extractors
    and editors. Models that don't expose internal states can still be
    evaluated on predictive quality and rollout consistency.
    """

    @property
    def hidden_size(self) -> int:
        """Dimensionality of the hidden state vector."""
        ...

    def get_hidden_states(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract per-timestep hidden states via teacher-forcing.

        Parameters
        ----------
        obs : (B, T, R) observation sequence

        Returns
        -------
        h : (B, T-1, hidden_size)
            h[:, t, :] is the hidden state produced after seeing obs[:, t, :].
            Aligns with positions[:, t, :] and is_visible[:, t, :].
        """
        ...
