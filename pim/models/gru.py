"""GRU implicit world model.

Trained with teacher forcing to predict the next 1D observation given the
current one.  The GRU hidden state serves as the implicit world state — it
is never supervised directly, only shaped by the predictive loss.

Architecture
------------
    obs[t]  →  encoder (Linear + ReLU)  →  GRU  →  decoder (Linear)  →  pred[t+1]

Training forward (teacher forcing):
    pred, h_n = model(obs)      # obs: (B, T, R)
    loss = MSE(pred, obs[:, 1:, :])

Autoregressive rollout (evaluation):
    h = None
    for t in range(T):
        pred_t, h = model.step(obs_t, h)   # obs_t: (B, R)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    input_dim: int = 128    # obs_res — overridden from dataset at train time
    hidden_size: int = 256
    num_layers: int = 1
    dropout: float = 0.0    # inter-layer dropout; ignored when num_layers == 1


class GRUModel(nn.Module):
    """GRU-based implicit world model.

    Parameters
    ----------
    cfg:
        Model configuration.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = nn.Linear(cfg.input_dim, cfg.hidden_size)
        self.gru = nn.GRU(
            input_size=cfg.hidden_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.decoder = nn.Linear(cfg.hidden_size, cfg.input_dim)

    def forward(
        self,
        obs: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forcing forward pass over a full sequence.

        Parameters
        ----------
        obs:
            Observation sequence, shape ``(B, T, R)``.
        h0:
            Optional initial hidden state, shape ``(num_layers, B, hidden_size)``.
            Defaults to zeros.

        Returns
        -------
        pred:
            Predicted next observations, shape ``(B, T-1, R)``.
        h_n:
            Final hidden state, shape ``(num_layers, B, hidden_size)``.
        """
        # Encode obs[0..T-2]; the GRU at step t predicts obs[t+1]
        x = F.relu(self.encoder(obs[:, :-1, :]))   # (B, T-1, H)
        h, h_n = self.gru(x, h0)                   # (B, T-1, H)
        pred = self.decoder(h)                      # (B, T-1, R)
        return pred, h_n

    def step(
        self,
        obs_t: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step autoregressive forward (for rollout / evaluation).

        Parameters
        ----------
        obs_t:
            Current observation, shape ``(B, R)``.
        h:
            Current hidden state, shape ``(num_layers, B, hidden_size)``.
            Defaults to zeros on first call.

        Returns
        -------
        pred_t:
            Predicted next observation, shape ``(B, R)``.
        h_next:
            Updated hidden state, shape ``(num_layers, B, hidden_size)``.
        """
        x = F.relu(self.encoder(obs_t)).unsqueeze(1)    # (B, 1, H)
        h_out, h_next = self.gru(x, h)                  # (B, 1, H)
        pred_t = self.decoder(h_out.squeeze(1))         # (B, R)
        return pred_t, h_next
