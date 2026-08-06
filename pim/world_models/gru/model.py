"""GRU implicit world model.

Trained with teacher forcing to predict the next 1D observation given the
current one.  The GRU hidden state serves as the implicit world state — it
is never supervised directly, only shaped by the predictive loss.

Architecture
------------
    obs[t]  →  encoder (Linear + ReLU)  →  GRU  →  decoder (Linear)  →  pred[t+1]

``enc_hidden_layers`` / ``dec_hidden_layers`` add extra ``Linear + activation``
blocks on either side of the recurrence, both defaulting to 0 (the architecture
above, unchanged).  With ``dec_hidden_layers = 0`` the decoder is a single
``nn.Linear``, i.e. **affine** — so ``decode(h0 + d1 + d2)`` is identically
``decode(h0+d1) + decode(h0+d2) − decode(h0)`` for any ``d1, d2``, and any
"edits superpose" result read off the *decoded observation* is forced by algebra
rather than by structure in the latent.  Set ``dec_hidden_layers >= 1`` to break
that identity.  The extra blocks live in separate ``enc_trunk`` / ``dec_trunk``
submodules that are absent at depth 0, so pre-existing checkpoints load
unchanged and produce bit-identical outputs.

Training forward (teacher forcing):
    pred, h_n = model(obs)      # obs: (B, T, R)
    loss = MSE(pred, obs[:, 1:, :])

Autoregressive rollout (evaluation):
    h = None
    for t in range(T):
        pred_t, h = model.step(obs_t, h)   # obs_t: (B, R)

Hidden state extraction (for probes):
    h = model.get_hidden_states(obs)   # (B, T-1, hidden_size)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


@dataclass
class ModelConfig:
    input_dim: int = 128  # obs_res — overridden from dataset at train time
    hidden_size: int = 256
    num_layers: int = 1
    dropout: float = 0.0  # inter-layer dropout; ignored when num_layers == 1
    # Extra Linear+activation blocks around the recurrence.  0 = the original
    # single-Linear encoder / single-Linear (affine) decoder.
    enc_hidden_layers: int = 0
    dec_hidden_layers: int = 0
    mlp_activation: str = "relu"


def _trunk(hidden_size: int, n_blocks: int, act: str) -> nn.Sequential | None:
    """``n_blocks`` × (Linear(H, H) + activation), or None when ``n_blocks == 0``.

    Returning None rather than an empty Sequential keeps the module absent from
    ``state_dict`` at depth 0, so old checkpoints load without ``strict=False``.
    """
    if n_blocks <= 0:
        return None
    if act not in _ACTIVATIONS:
        raise ValueError(
            f"unknown mlp_activation {act!r}; choose from {sorted(_ACTIVATIONS)}"
        )
    layers: list[nn.Module] = []
    for _ in range(n_blocks):
        layers += [nn.Linear(hidden_size, hidden_size), _ACTIVATIONS[act]()]
    return nn.Sequential(*layers)


class GRUModel(nn.Module):
    """GRU-based implicit world model.

    Implements both WorldModel and HiddenStateModel protocols.

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

        # Optional depth.  enc_trunk runs after the encoder Linear+ReLU;
        # dec_trunk runs before the decoder Linear.  Both None by default.
        act = getattr(cfg, "mlp_activation", "relu")
        self.enc_trunk = _trunk(
            cfg.hidden_size, getattr(cfg, "enc_hidden_layers", 0), act
        )
        self.dec_trunk = _trunk(
            cfg.hidden_size, getattr(cfg, "dec_hidden_layers", 0), act
        )

    @property
    def hidden_size(self) -> int:
        return self.cfg.hidden_size

    @property
    def has_affine_decoder(self) -> bool:
        """True when ``decode`` is a single Linear, hence exactly affine in ``h``."""
        return self.dec_trunk is None

    # ── Single choke-points: every encode / decode in this class goes through
    #    these two, so depth can never be applied on some code paths and not others.

    def _enc(self, obs: torch.Tensor) -> torch.Tensor:
        """obs (..., R) → embedding (..., H).  Broadcasts over any leading dims."""
        x = F.relu(self.encoder(obs))
        return x if self.enc_trunk is None else self.enc_trunk(x)

    def _dec(self, h: torch.Tensor) -> torch.Tensor:
        """hidden (..., H) → observation (..., R).  Broadcasts over any leading dims."""
        return self.decoder(h if self.dec_trunk is None else self.dec_trunk(h))

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
        x = self._enc(obs[:, :-1, :])  # (B, T-1, H)
        h, h_n = self.gru(x, h0)  # (B, T-1, H)
        pred = self._dec(h)  # (B, T-1, R)
        return pred, h_n

    def step(
        self,
        obs_t: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step autoregressive forward (for rollout / evaluation).

        Parameters
        ----------
        obs_t:
            Current observation, shape ``(B, R)``.
        state:
            Current hidden state, shape ``(num_layers, B, hidden_size)``.
            Defaults to zeros on first call.

        Returns
        -------
        pred_t:
            Predicted next observation, shape ``(B, R)``.
        h_next:
            Updated hidden state, shape ``(num_layers, B, hidden_size)``.
        """
        x = self._enc(obs_t).unsqueeze(1)  # (B, 1, H)
        h_out, h_next = self.gru(x, state)  # (B, 1, H)
        pred_t = self._dec(h_out.squeeze(1))  # (B, R)
        return pred_t, h_next

    @torch.no_grad()
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
        x = self._enc(obs[:, :-1, :])  # (B, T-1, H)
        h, _ = self.gru(x)  # (B, T-1, H)
        return h

    # ── SSM protocol methods ──────────────────────────────────────────────────

    def flat_state(self, state: torch.Tensor) -> torch.Tensor:
        """Last recurrent layer: (num_layers, B, H) → (B, H)."""
        return state[-1]

    def state_from_flat(self, flat: torch.Tensor) -> torch.Tensor:
        """(B, H) → (1, B, H) GRU hidden state (num_layers=1)."""
        return flat.unsqueeze(0)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        """Decode the last-layer hidden state to an observation (no advance).

        Parameters
        ----------
        state : (num_layers, B, H)

        Returns
        -------
        obs : (B, R)
        """
        return self._dec(state[-1])

    @torch.no_grad()
    def observe_sequence(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-pass teacher-forcing: predictions + flat hidden states.

        Parameters
        ----------
        obs : (B, T, R)

        Returns
        -------
        pred   : (B, T-1, R)
        h_flat : (B, T-1, hidden_size)
        """
        x = self._enc(obs[:, :-1, :])  # (B, T-1, H)
        h_seq, _ = self.gru(x)  # (B, T-1, H)
        pred = self._dec(h_seq)  # (B, T-1, R)
        return pred, h_seq

    def predict_step(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Free-running step: decode current h, feed decoded obs back through step.

        Returns the prediction for the NEXT frame and the updated state.

        Parameters
        ----------
        state : (num_layers, B, H)

        Returns
        -------
        pred_next  : (B, R) — prediction for frame t+1
        state_next : (num_layers, B, H)
        """
        obs_hat = self._dec(state[-1])  # (B, R) — decode current h
        pred_next, state_next = self.step(obs_hat, state)
        return pred_next, state_next
