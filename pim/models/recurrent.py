"""Recurrent-L — a stacked GRU world model at Transformer-L's parameter count.

Added 2026-09-02 to test ONE candidate gate for the discworld editability negative:
**recomputation**. A transformer's readable state at the last position is rebuilt every
step from the observation window by attention, so a write there can be overwritten by
later layers re-deriving the same quantity from earlier positions. A recurrent model has
no such route: its hidden state is the only summary of the past it carries, and a write to
it has nowhere to be overwritten *from*. If Recurrent-L edits where Transformer-L does not,
recomputation is the gate; if it stays flat, that candidate drops.

Architecture — each component mirrors Transformer-S (which was itself built to mirror the
project's original GRU), so the only change is the block stack:

    obs (B,T,R)  →  Linear(R → d) + ReLU            ← the encoder port, residual point 0
                 →  n_layers × GRU(d → d)             ← the ONLY change (dropout between)
                 →  LayerNorm → Linear(d → R)         ← same decoder shape
    loss = MSE(pred[:, t], obs[:, t+1])               ← identical objective

Sized to the canonical Transformer-L: d=1024, 4 layers ≈ 25.4M parameters (Sevan's call,
2026-09-02: parameter-matched rather than width-matched). The carried state is
4 × 1024 = 4,096 numbers, close to the transformer's 39-frame × 128-ray window.

Residual points
---------------
``n_points = n_layers + 1``: point 0 is the embedding, point ℓ the output (= hidden) of
GRU layer ℓ at that step. The layers are held as separate one-layer ``nn.GRU`` modules and
run one after another over the whole sequence — which is exactly how a stacked GRU computes
anyway — so the ``_run`` edit socket has a natural home BETWEEN layers: a write at point ℓ
is applied to layer ℓ's output before layer ℓ+1 consumes it. Both edit forms of the
protocol are honoured unchanged (a tuple at the last position; a callable at every point),
so Li's sequential gradient-steering schedule runs as-is — no "pretend it is one layer"
trick is needed.

The state object, and what is — and is not — the hidden state
---------------------------------------------------------------
``RecurrentState(h_prev, obs_t, length)``. The **hidden state is ``h_prev`` alone**: the
(n_layers, B, d) hiddens after the previous frame. ``obs_t`` is the newest frame, *pending
and not yet consumed* — the same role the newest frame plays inside Transformer-S's window.
It is there so that ``decode`` computes the current step in full, which is what lets an
edit at ANY residual point ℓ shape the *immediate* prediction (layers above ℓ recompute
at this step) exactly as the transformer's ``(layer, vector)`` edit does at the last
position. Probes, ``flat_state`` and edits all refer to the post-step hiddens ``h_t``,
computed on demand from ``(h_prev, obs_t)``; ``tests/test_recurrent.py`` pins that this
per-step path and the full-sequence ``_run`` agree exactly.

Edit semantics — CARRIED (the architecture-forced difference)
--------------------------------------------------------------
``rollout_with_edit`` writes point ℓ at the current step, decodes, and then carries the
**edited** hiddens forward (at layer ℓ and, through recomputation, above it; layers below
ℓ carry their own unedited hiddens). That is what "editing an RNN's world state" means and
it is the hypothesis under test. ``carry_edits = False`` reproduces the transformer's
carry-nothing semantics — the write shapes one prediction and the unedited hiddens are
carried — for a matched comparison. Default ``True`` (Sevan, 2026-09-02).

``state_span`` is unbounded: an RNN's state is a function of its entire history. It is
reported as a large sentinel so callers that truncate to the span keep every frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentState(NamedTuple):
    """h_prev : (n_layers, B, d) hiddens after the PREVIOUS frame — THE hidden state.
    obs_t  : (B, R) the newest frame, pending (not yet consumed).
    length : (B,) frames seen so far, including obs_t."""

    h_prev: torch.Tensor
    obs_t: torch.Tensor
    length: torch.Tensor


@dataclass
class RecurrentConfig:
    input_dim: int = 128     # obs_res — overridden from the dataset at train time
    d_model: int = 1024      # hidden width; 1024 × 4 layers ≈ Transformer-L's 25.4M params
    n_layers: int = 4
    dropout: float = 0.1     # between GRU layers (never after the last), as nn.GRU does


class RecurrentL(nn.Module):
    """Recurrent-L: stacked GRU world model (regression head)."""

    STATE_VIEWS = ("activations",)
    UNBOUNDED_SPAN = 10_000

    def __init__(self, cfg: RecurrentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Linear(cfg.input_dim, cfg.d_model)
        self.layers = nn.ModuleList(
            [nn.GRU(cfg.d_model, cfg.d_model, batch_first=True) for _ in range(cfg.n_layers)]
        )
        self.norm_out = nn.LayerNorm(cfg.d_model)
        self.decoder = nn.Linear(cfg.d_model, cfg.input_dim)
        self.state_view: str = "activations"
        self.probe_layer: int = cfg.n_layers
        self.carry_edits: bool = True

    # ── protocol scalars ──────────────────────────────────────────────────────

    @property
    def n_layers(self) -> int:
        return self.cfg.n_layers

    @property
    def state_span(self) -> int:
        """Unbounded — see the module docstring. Any episode is shorter than this."""
        return self.UNBOUNDED_SPAN

    @property
    def hidden_size(self) -> int:
        return self.cfg.d_model

    # ── core ──────────────────────────────────────────────────────────────────

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        """The encoder port — residual point 0, same form as Transformer-S."""
        return F.relu(self.encoder(obs))

    def _seq_mask(self, T: int, device):
        """No attention, no mask — kept so callers written for the transformers work."""
        return None

    def _run(self, tokens, attn_mask=None, edit=None, want_resid=False, h0=None):
        """Run the layer stack over pre-embedded tokens ``(B, T, d)``.

        ``edit`` follows the protocol exactly: ``(layer, vector)`` forces residual point
        ``layer`` at the last position; a callable fires at every point with the whole
        stream. ``None`` takes neither branch (bit-identical, gated by tests).
        ``h0`` (n_layers, B, d) seeds the recurrence — the per-step path; ``None`` = zeros.
        Returns ``(h, resids)`` with ``h`` the top layer's stream and ``resids`` the list of
        every residual point (edits included) when ``want_resid``.
        """
        hook = edit if callable(edit) else None
        x = tokens
        resids = [x] if want_resid else None
        for i, gru in enumerate(self.layers):
            if hook is not None:
                x = hook(i, x)
                if want_resid:
                    resids[i] = x
            elif edit is not None and edit[0] == i:
                x = x.clone()
                x[:, -1] = edit[1]
                if want_resid:
                    resids[i] = x
            if i > 0 and self.cfg.dropout > 0:
                x = F.dropout(x, self.cfg.dropout, self.training)
            h_i = None if h0 is None else h0[i][None].contiguous()
            x, _ = gru(x, h_i)
            if want_resid:
                resids.append(x)
        if hook is not None:
            x = hook(len(self.layers), x)
            if want_resid:
                resids[-1] = x
        elif edit is not None and edit[0] == len(self.layers):
            x = x.clone()
            x[:, -1] = edit[1]
            if want_resid:
                resids[-1] = x
        return x, resids

    # ── training / teacher forcing ────────────────────────────────────────────

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict the next frame at EVERY given position — Transformer-L's convention, so
        the trainer's ``pred = model(x[:, :-1])`` alignment applies unchanged."""
        h, _ = self._run(self.embed(obs))
        return self.decoder(self.norm_out(h))

    # ── the per-step path ─────────────────────────────────────────────────────

    def _step(self, state: RecurrentState, edit=None):
        """Consume ``obs_t`` from ``h_prev``: (pred, h_t, resids).

        ``h_t`` (n_layers, B, d) are the hiddens to carry — taken from ``resids`` so that
        an edit at point ℓ is carried at layer ℓ, and the layers above it carry what they
        recomputed from the edited value."""
        tokens = self.embed(state.obs_t)[:, None]                       # (B, 1, d)
        h, resids = self._run(tokens, edit=edit, want_resid=True, h0=state.h_prev)
        pred = self.decoder(self.norm_out(h[:, -1]))
        h_t = torch.stack([r[:, -1] for r in resids[1:]], 0)
        return pred, h_t, resids

    def _zeros(self, B: int, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_zeros(self.cfg.n_layers, B, self.cfg.d_model)

    def state_from_obs(self, frames: torch.Tensor) -> RecurrentState:
        """Consume all but the newest frame; the newest becomes the pending input."""
        B, n, _ = frames.shape
        if n > 1:
            _, resids = self._run(self.embed(frames[:, :-1]), want_resid=True)
            h_prev = torch.stack([r[:, -1] for r in resids[1:]], 0)
        else:
            h_prev = self._zeros(B, frames)
        length = torch.full((B,), n, dtype=torch.long, device=frames.device)
        return RecurrentState(h_prev, frames[:, -1], length)

    def flat_state(self, state: RecurrentState) -> torch.Tensor:
        """Residual point ``probe_layer`` at the current step, (B, d)."""
        _, _, resids = self._step(state)
        return resids[self.probe_layer][:, -1]

    def state_from_flat(self, flat: torch.Tensor):
        raise ValueError("the activation view is read-only; edit through rollout_with_edit")

    @torch.no_grad()
    def residual_stack(self, x, edit=None) -> torch.Tensor:
        """(n_layers+1, B, T, d) over an observation sequence, or (n_layers+1, B, 1, d)
        for a state (the current step)."""
        if isinstance(x, RecurrentState):
            _, _, resids = self._step(x, edit=edit)
        else:
            _, resids = self._run(self.embed(x), edit=edit, want_resid=True)
        return torch.stack(resids, 0)

    # ── prediction ────────────────────────────────────────────────────────────

    def decode(self, state: RecurrentState, edit=None) -> torch.Tensor:
        return self._step(state, edit=edit)[0]

    def decode_with_edit(self, state, layer: int, resid: torch.Tensor):
        return self.decode(state, edit=(layer, resid))

    def advance(self, state: RecurrentState, obs_next: torch.Tensor) -> RecurrentState:
        _, h_t, _ = self._step(state)
        return RecurrentState(h_t, obs_next, state.length + 1)

    def predict_step(self, state: RecurrentState):
        pred, h_t, _ = self._step(state)
        return pred, RecurrentState(h_t, pred, state.length + 1)

    def step(self, obs_t: torch.Tensor, state: RecurrentState | None = None):
        if state is None:
            B = obs_t.shape[0]
            state = RecurrentState(self._zeros(B, obs_t), obs_t,
                                   torch.ones(B, dtype=torch.long, device=obs_t.device))
        else:
            state = self.advance(state, obs_t)
        return self.decode(state), state

    def _rollout(self, state: RecurrentState, edit, steps: int) -> torch.Tensor:
        pred, h_t, _ = self._step(state, edit=edit)
        if not self.carry_edits:                    # transformer-style: carry nothing
            _, h_t, _ = self._step(state)
        out, s = [pred], RecurrentState(h_t, pred, state.length + 1)
        for _ in range(steps - 1):
            p, s = self.predict_step(s)
            out.append(p)
        return torch.stack(out, 1)

    def rollout_with_edit(self, state, layer: int, resid: torch.Tensor, steps: int):
        """Free-run whose first step is produced under a single-site edit, which is then
        CARRIED in the hidden state (``carry_edits``) — the recurrent measurement."""
        return self._rollout(state, (layer, resid), steps)

    def rollout_with_hook(self, state, hook, steps: int):
        """The same for a callable (multi-layer) edit — gradient steering's schedule."""
        return self._rollout(state, hook, steps)
