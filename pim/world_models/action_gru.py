"""Action-conditioned GRU implicit world model.

NEW module — does not modify the existing GRU.  A GRU world model whose encoder
is widened to ingest a discrete action token (embedded and concatenated to the
observation) in addition to the 1D observation.

Crucially, it conforms to the ``HiddenStateModel`` protocol with **actions
defaulting to no-op (token 0)**.  Every protocol method (``forward``, ``step``,
``get_hidden_states``, ``observe_sequence``, ``predict_step``, ``decode``,
``flat_state``, ``state_from_flat``, ``hidden_size``) has the *same* signature
as ``GRUModel`` — the optional ``actions`` argument is appended, never required.
So the entire model-agnostic eval / extractor / editor suite runs UNCHANGED on
this model in passive (no-op) mode; no ``isinstance`` branch anywhere.

Alignment
---------
Training forward (teacher forcing), with actions:
    pred, h_n = model(obs, actions=actions)   # obs:(B,T,R), actions:(B,T) long
    loss = MSE(pred, obs[:, 1:, :])
At input step t (encoding obs[:, t]) the model is told a_t = actions[:, t], the
token that drives the transition into t+1, and predicts obs[:, t+1].

Passive inference (eval): pass no ``actions`` -> all no-op (token 0), so the
recurrent state unfolds passively exactly as a plain GRU would.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ActionModelConfig:
    input_dim: int = 128        # obs_res
    hidden_size: int = 256
    num_layers: int = 1
    dropout: float = 0.0        # inter-layer dropout; ignored when num_layers == 1
    n_actions: int = 9          # 1 no-op + 4 per obj * 2 objs
    action_embed_dim: int = 16


class ActionGRUModel(nn.Module):
    """GRU world model conditioned on a discrete action token via the encoder."""

    def __init__(self, cfg: ActionModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_embed = nn.Embedding(cfg.n_actions, cfg.action_embed_dim)
        self.encoder = nn.Linear(cfg.input_dim + cfg.action_embed_dim, cfg.hidden_size)
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
        """obs:(..., R) -> encoded (..., H).  actions:(...,) long or None (no-op)."""
        if actions is None:
            actions = torch.zeros(obs.shape[:-1], dtype=torch.long, device=obs.device)
        emb = self.action_embed(actions)                    # (..., E)
        return F.relu(self.encoder(torch.cat([obs, emb], dim=-1)))

    def forward(
        self,
        obs: torch.Tensor,
        h0: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forcing pass.  obs:(B,T,R); actions:(B,T) long or None -> no-op.

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
        """Single autoregressive step.  obs_t:(B,R); action:(B,) long or None."""
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
