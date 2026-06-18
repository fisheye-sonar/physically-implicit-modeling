"""Model interface protocols.

WorldModel       — minimal contract: forward + step (any predictive model)
HiddenStateModel — extended contract: also exposes a probe-compatible flat hidden
                   state plus five SSM operations that allow model-agnostic
                   inference across GRU, RSSM, and future architectures.

The five SSM operations on HiddenStateModel:

  flat_state(state)       : model-native state → (B, H) flat tensor for probes/editors
  state_from_flat(flat)   : (B, H) flat tensor → model-native state
  decode(state)           : model-native state → (B, R) observation (no state advance)
  observe_sequence(obs)   : (B, T, R) → ((B, T-1, R) preds, (B, T-1, H) hidden states)
                            teacher-forcing pass, combines forward() + get_hidden_states()
  predict_step(state)     : model-native state → (pred (B, R), next_state)
                            free-running step with no real observation
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class WorldModel(Protocol):
    """Minimal contract — any model that predicts next observations."""

    def forward(
        self,
        obs: torch.Tensor,
        h0: Any = None,
    ) -> tuple[torch.Tensor, Any]:
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
        state: Any = None,
    ) -> tuple[torch.Tensor, Any]:
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

    In addition to WorldModel, these models support a general SSM interface
    (flat_state / state_from_flat / observe_sequence / predict_step) that
    lets eval, probe, and editor code remain model-agnostic.
    """

    @property
    def hidden_size(self) -> int:
        """Dimensionality of the flat hidden state vector."""
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

    def flat_state(self, state: Any) -> torch.Tensor:
        """Convert model-native state to a flat (B, hidden_size) tensor.

        Used by probes and editors that operate on flat hidden vectors.

        Parameters
        ----------
        state : model-native state (e.g. GRU: (num_layers, B, H); RSSM: RSSMState)

        Returns
        -------
        flat : (B, hidden_size)
        """
        ...

    def state_from_flat(self, flat: torch.Tensor) -> Any:
        """Reconstruct model-native state from a flat (B, hidden_size) tensor.

        Inverse of flat_state — used after probe injection.

        Parameters
        ----------
        flat : (B, hidden_size)

        Returns
        -------
        state : model-native state
        """
        ...

    def decode(self, state: Any) -> torch.Tensor:
        """Decode the current state to an observation without advancing it.

        The inverse of observe_step in the SSM sense: given a state, what
        observation does the model think you'd see right now?

        Parameters
        ----------
        state : model-native state

        Returns
        -------
        obs : (B, R) decoded observation
        """
        ...

    def observe_sequence(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forcing pass returning predictions AND flat hidden states.

        Single-pass equivalent of forward() + get_hidden_states().  The
        hidden states may use stochastic sampling during training (RSSM)
        but are deterministic for GRU.

        Parameters
        ----------
        obs : (B, T, R) observation sequence

        Returns
        -------
        pred   : (B, T-1, R)  next-step predictions
        h_flat : (B, T-1, hidden_size)  flat hidden states aligned to obs[:, :-1]
        """
        ...

    def predict_step(self, state: Any) -> tuple[torch.Tensor, Any]:
        """Free-running step with no real observation.

        Advances the model one step without conditioning on a real observation.
        Semantics are model-specific:
          GRU  — decodes current h to get a synthetic obs, feeds it back through step
          RSSM — pure prior imagination step (no obs used)

        Parameters
        ----------
        state : current model-native state

        Returns
        -------
        pred_next  : (B, R) predicted observation for the next frame
        state_next : model-native state after the step
        """
        ...
