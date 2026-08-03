"""Continuous-action-conditioned GRU implicit world model.

NEW module — does not modify the existing GRU or the discrete ``action_gru``.
A GRU world model whose encoder ingests a **continuous** per-object action vector
(``[active, a1, a2]`` per object, projected by a small MLP) concatenated to the
1D observation.

Crucially, it conforms to the ``HiddenStateModel`` protocol with **actions
defaulting to no-op (zeros)**.  Every protocol method (``forward``, ``step``,
``get_hidden_states``, ``observe_sequence``, ``predict_step``, ``decode``,
``flat_state``, ``state_from_flat``, ``hidden_size``) has the same signature as
``GRUModel`` — the optional ``actions``/``action`` argument is appended, never
required.  So the entire model-agnostic eval / extractor / editor suite runs
UNCHANGED on this model in passive (no-op) mode; no ``isinstance`` branch anywhere.

Alignment
---------
Training forward (teacher forcing), with actions:
    pred, h_n = model(obs, actions=actions)     # obs:(B,T,R), actions:(B,T,n_obj,3)
    loss = MSE(pred, obs[:, 1:, :])
At input step t (encoding obs[:, t]) the model is told a_t = actions[:, t], the
action driving the transition into t+1, and predicts obs[:, t+1].

Passive inference (eval): pass no ``actions`` -> all zeros (no-op), so the
recurrent state unfolds passively exactly as a plain GRU would.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ActionContinuousModelConfig:
    input_dim: int = 128            # obs_res
    hidden_size: int = 256
    num_layers: int = 1
    dropout: float = 0.0            # inter-layer dropout; ignored when num_layers == 1
    n_obj: int = 2                  # objects (action vector = n_obj * action_feat_dim)
    action_feat_dim: int = 3        # [active, a1, a2] per object
    action_proj_dim: int = 16       # width of the projected action embedding


class ActionGRUContinuousModel(nn.Module):
    """GRU world model conditioned on a continuous per-object action vector."""

    def __init__(self, cfg: ActionContinuousModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._act_dim = cfg.n_obj * cfg.action_feat_dim
        self.action_proj = nn.Linear(self._act_dim, cfg.action_proj_dim)
        self.encoder = nn.Linear(cfg.input_dim + cfg.action_proj_dim, cfg.hidden_size)
        self.gru = nn.GRU(
            input_size=cfg.hidden_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.decoder = nn.Linear(cfg.hidden_size, cfg.input_dim)

    @property
    def hidden_size(self) -> int:
        return self.cfg.hidden_size

    # ── internal: encode (obs, action) -> gru input ───────────────────────────
    def _encode(self, obs: torch.Tensor, actions: torch.Tensor | None) -> torch.Tensor:
        """obs:(..., R) -> encoded (..., H).

        actions:(..., n_obj, action_feat_dim) or (..., n_obj*action_feat_dim) or
        None (no-op = zeros).  A zeros vector still projects through
        ``action_proj`` (+bias, ReLU) -> a fixed learned no-op embedding, exactly
        analogous to the discrete no-op token's embedding.
        """
        lead = obs.shape[:-1]
        if actions is None:
            flat = torch.zeros(*lead, self._act_dim, dtype=obs.dtype, device=obs.device)
        else:
            flat = actions.reshape(*lead, self._act_dim).to(obs.dtype)
        a = F.relu(self.action_proj(flat))                  # (..., action_proj_dim)
        return F.relu(self.encoder(torch.cat([obs, a], dim=-1)))

    def forward(
        self,
        obs: torch.Tensor,
        h0: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forcing pass.  obs:(B,T,R); actions:(B,T,n_obj,3) or None.

        Returns pred:(B,T-1,R), h_n:(num_layers,B,H).
        """
        a = None if actions is None else actions[:, :-1]
        x = self._encode(obs[:, :-1, :], a)                 # (B, T-1, H)
        h, h_n = self.gru(x, h0)
        pred = self.decoder(h)
        return pred, h_n

    def step(
        self,
        obs_t: torch.Tensor,
        state: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single autoregressive step.  obs_t:(B,R); action:(B,n_obj,3) or None."""
        x = self._encode(obs_t, action).unsqueeze(1)        # (B, 1, H)
        h_out, h_next = self.gru(x, state)
        pred_t = self.decoder(h_out.squeeze(1))
        return pred_t, h_next

    @torch.no_grad()
    def get_hidden_states(
        self, obs: torch.Tensor, actions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Per-timestep hidden states via teacher forcing.  Returns (B,T-1,H)."""
        a = None if actions is None else actions[:, :-1]
        x = self._encode(obs[:, :-1, :], a)
        h, _ = self.gru(x)
        return h

    # ── SSM protocol methods (identical semantics to GRUModel) ────────────────

    def flat_state(self, state: torch.Tensor) -> torch.Tensor:
        return state[-1]

    def state_from_flat(self, flat: torch.Tensor) -> torch.Tensor:
        return flat.unsqueeze(0)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return self.decoder(state[-1])

    @torch.no_grad()
    def observe_sequence(
        self, obs: torch.Tensor, actions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forcing pass: (pred (B,T-1,R), h_flat (B,T-1,H))."""
        a = None if actions is None else actions[:, :-1]
        x = self._encode(obs[:, :-1, :], a)
        h_seq, _ = self.gru(x)
        pred = self.decoder(h_seq)
        return pred, h_seq

    def predict_step(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Free-running step (no-op action): decode current h, feed back through step."""
        obs_hat = self.decoder(state[-1])
        pred_next, state_next = self.step(obs_hat, state, action=None)
        return pred_next, state_next
