"""Causal transformer world model — the attention counterpart to the GRU/RSSM.

Trained on exactly the same task and with exactly the same objective as the GRU:
predict `obs[t+1]` from `obs[<=t]` under MSE.  Deliberately *not* a diffusion
model (see `pim/world_models/dit/`) — this is the minimal architecture change
from the GRU so that "recurrence vs attention" is the only variable.

Architecture (each component chosen to mirror `GRUModel`)
--------------------------------------------------------
    obs (B,T,R)  →  Linear(R → d_model) + ReLU        ← same encoder as the GRU
                 →  N × pre-norm transformer blocks    ← the ONLY change
                 →  LayerNorm → Linear(d_model → R)    ← same shape as the GRU decoder
    loss = MSE(pred[:, t], obs[:, t+1])                ← identical objective

`d_model` defaults to **256 = the GRU's hidden size**, not to a parameter-matched
width.  The whole analysis is about the *geometry* of the state (row-space
fractions have chance level `sqrt(d/H)`), so the state width is the quantity that
must match for numbers to be comparable across architectures; parameter count is
the lesser confound and is handled by a capacity control instead.

What is the "state"?
--------------------
This is the substantive question the architecture forces, and the answer is that
a transformer has **two** objects where the GRU has one, and they come apart:

  * the **carried** state — the buffer of recent observations the model must be
    given to reproduce its own prediction.  It is genuinely what persists between
    steps, it is exactly invertible, and each of its elements is a
    *history-independent* function of one frame.

    Its size is **not** the attention window.  Stacking layers widens the
    receptive field: at layer L, position t depends on positions
    `t - L*(W-1) .. t`, because each layer's keys were themselves computed from
    a window.  So the carried state spans

        state_span = n_layers * (window - 1) + 1

    frames, and `window` is only the *per-layer attention span*.  This is
    verified numerically (`tests/test_transformer.py`): a one-pass banded-mask
    forward and a step-by-step buffer rollout agree to float tolerance only when
    the buffer is `state_span` frames, and diverge from exactly `t = window`
    onward if the buffer is `window` frames.  Getting this wrong would badly
    mis-state "how much history must be overwritten for an edit to stick".
  * the **readable** state — the residual stream at (layer ℓ, current position),
    which *is* history-dependent because attention has mixed the window in.  It
    is the closest analogue to the GRU's `h` — but it is **recomputed** every
    step, never carried, so a write to it does not survive to the next step
    unless the window itself is changed.

In a GRU these coincide.  Here they do not, and that is the point.  Three views
are exposed via `model.state_view` (a runtime toggle, like the RSSM's `sample`):

  * `"obs_window"` (default) — the raw W-frame buffer, flattened (W·R dims).
    Invertible, so `state_from_flat` works and every editor runs unchanged.
  * `"activations"` — residual stream at `model.probe_layer`, current position
    (d_model dims).  The GRU-`h` analogue.  Read-only for `state_from_flat`;
    edited through `decode_with_edit` / `rollout_with_edit`, which is where the
    *transient vs persistent* distinction is measured.
  * `"kv_cache"` — post-RoPE K/V for every layer over the window.  Large
    (`n_layers·2·W·d_model`); for targeted probing, not the full suite.

`probe_layer` indexes **residual points**, of which there are `n_layers + 1`:
0 = the encoder output (i.e. the *encoder port*, matching the GRU's `x`), and
`n_layers` = the final pre-LayerNorm stream that the decoder reads.

Layer choice has a structural consequence worth stating: an edit at residual
point ℓ changes the block inputs for layers > ℓ only.  Editing the *final* point
therefore alters this position's own prediction and propagates to nothing, while
editing point 0 propagates furthest.  "Which layer is the world state" is thus
both a readability and a persistence question, and they need not agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pim.world_models.dit.blocks import (
    CausalSelfAttention,
    RotaryEmbedding,
    band_causal_mask,
)

# ── State ─────────────────────────────────────────────────────────────────────


class TransformerState(NamedTuple):
    """Sliding-window state: everything the model carries between observations.

    obs_buffer : (B, S, R) last S = `state_span` observations, right-aligned
                 (newest last), zero-padded while fewer than S have been seen.
    length     : (B,) int64 number of valid frames in the buffer (<= S).
    """

    obs_buffer: torch.Tensor
    length: torch.Tensor


@dataclass
class ModelConfig:
    input_dim: int = 128  # obs_res — overridden from the dataset at train time
    d_model: int = 256  # token width; matches the GRU's hidden size
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: float = 4.0
    window: int = 16  # sliding attention window = frames of state carried


# ── Blocks ────────────────────────────────────────────────────────────────────


class Block(nn.Module):
    """Standard pre-norm transformer block (no conditioning — this is not a DiT)."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )

    def forward(self, x, attn_mask, rope, kv_sink=None):
        x = x + self.attn(self.norm1(x), attn_mask, rope, kv_sink)
        x = x + self.mlp(self.norm2(x))
        return x


# ── Model ─────────────────────────────────────────────────────────────────────


class TransformerModel(nn.Module):
    """Causal transformer world model. Implements WorldModel + HiddenStateModel."""

    STATE_VIEWS = ("obs_window", "activations", "kv_cache")

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Linear(cfg.input_dim, cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                Block(cfg.d_model, cfg.n_heads, cfg.mlp_ratio)
                for _ in range(cfg.n_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(cfg.d_model)
        self.decoder = nn.Linear(cfg.d_model, cfg.input_dim)
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_heads)
        self.state_view: str = "obs_window"
        self.probe_layer: int = cfg.n_layers  # residual point read by "activations"

    @property
    def state_span(self) -> int:
        """Frames the model must be given to reproduce its own prediction.

        `n_layers*(window-1)+1`, NOT `window` — see the module docstring.  This is
        the true size of the carried state and the quantity an edit has to
        overwrite.
        """
        return self.cfg.n_layers * (self.cfg.window - 1) + 1

    # ── core ──────────────────────────────────────────────────────────────────

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        """The encoder port — identical in form to the GRU's `relu(Linear(obs))`."""
        return F.relu(self.encoder(obs))

    def _run(self, tokens, attn_mask, edit=None, want_resid=False, kv_sink=None):
        """Run the block stack over pre-embedded tokens.

        edit : optional (layer, vector) forcing the residual stream at residual
               point `layer` to `vector` at the LAST position only.
        """
        B, T, _ = tokens.shape
        cos, sin = self.rope(T, tokens.device)
        x = tokens
        resids = [x] if want_resid else None
        for i, blk in enumerate(self.blocks):
            if edit is not None and edit[0] == i:
                x = x.clone()
                x[:, -1] = edit[1]
                if want_resid:
                    resids[i] = x  # keep the recorded stream consistent with the edit
            x = blk(x, attn_mask, (cos, sin), kv_sink)
            if want_resid:
                resids.append(x)
        if edit is not None and edit[0] == len(self.blocks):
            x = x.clone()
            x[:, -1] = edit[1]
            if want_resid:
                resids[-1] = x
        return x, resids

    def _seq_mask(self, T: int, device) -> torch.Tensor:
        return band_causal_mask(T, self.cfg.window, device)[None, None]

    def _win_mask(self, lengths: torch.Tensor, device) -> torch.Tensor:
        """(N,1,S,S) banded-causal mask restricted to the valid right-aligned frames.

        Uses the SAME band width as the training-time mask, so a buffer rollout
        reproduces the one-pass forward exactly."""
        S = self.state_span
        causal = band_causal_mask(S, self.cfg.window, device)
        valid = torch.arange(S, device=device)[None, :] >= (S - lengths[:, None])
        return causal[None, None] & valid[:, None, None, :]

    # ── training / teacher forcing ────────────────────────────────────────────

    def forward(self, obs: torch.Tensor, h0=None):
        """Teacher forcing over a whole sequence in ONE pass (banded causal mask).

        Equivalent to running the sliding window at every position — the band mask
        *is* the sliding window — but O(T) cheaper than unfolding, which matters
        because it is the training path.  Returns (pred (B,T-1,R), final state).
        """
        x = self.embed(obs[:, :-1, :])
        h, _ = self._run(x, self._seq_mask(x.shape[1], obs.device))
        pred = self.decoder(self.norm_out(h))
        return pred, self.state_from_obs(obs[:, :-1, :])

    @torch.no_grad()
    def get_hidden_states(self, obs: torch.Tensor) -> torch.Tensor:
        """Per-timestep flat state, aligned so index t follows obs[:, t]."""
        return self.observe_sequence(obs)[1]

    @torch.no_grad()
    def observe_sequence(self, obs: torch.Tensor):
        x = self.embed(obs[:, :-1, :])
        h, resids = self._run(
            x, self._seq_mask(x.shape[1], obs.device), want_resid=True
        )
        pred = self.decoder(self.norm_out(h))
        if self.state_view == "activations":
            flat = resids[self.probe_layer]
        elif self.state_view == "obs_window":
            B, T, R = obs[:, :-1, :].shape
            W = self.state_span
            pad = obs.new_zeros(B, W - 1, R)
            padded = torch.cat([pad, obs[:, :-1, :]], dim=1)
            flat = padded.unfold(1, W, 1).permute(0, 1, 3, 2).reshape(B, T, W * R)
        else:
            raise ValueError(f"observe_sequence unsupported for {self.state_view!r}")
        return pred, flat

    # ── state plumbing ────────────────────────────────────────────────────────

    @property
    def hidden_size(self) -> int:
        if self.state_view == "obs_window":
            return self.state_span * self.cfg.input_dim
        if self.state_view == "activations":
            return self.cfg.d_model
        if self.state_view == "kv_cache":
            return self.cfg.n_layers * 2 * self.state_span * self.cfg.d_model
        raise ValueError(f"unknown state_view: {self.state_view!r}")

    def _pad_window(self, frames: torch.Tensor) -> torch.Tensor:
        B, n, R = frames.shape
        W = self.state_span
        if n >= W:
            return frames[:, -W:].contiguous()
        return torch.cat([frames.new_zeros(B, W - n, R), frames], dim=1)

    def state_from_obs(self, frames: torch.Tensor) -> TransformerState:
        """Build the carried state directly from a run of observations.

        The transformer's state is a function of the last `state_span` frames, so
        this is exact and O(1) — no need to replay `step` frame by frame. It is
        also how a *history overwrite* edit is applied: hand it a buffer whose
        last n frames come from a counterfactual world.
        """
        B, n, _ = frames.shape
        length = torch.full(
            (B,), min(n, self.state_span), dtype=torch.long, device=frames.device
        )
        return TransformerState(self._pad_window(frames), length)

    def flat_state(self, state: TransformerState) -> torch.Tensor:
        if self.state_view == "obs_window":
            return state.obs_buffer.reshape(state.obs_buffer.shape[0], -1)
        if self.state_view == "activations":
            return self._activations(state)
        if self.state_view == "kv_cache":
            return self._kv_cache(state)
        raise ValueError(f"unknown state_view: {self.state_view!r}")

    def state_from_flat(self, flat: torch.Tensor) -> TransformerState:
        """Only the (invertible) obs_window view supports this.

        The activation view is deliberately read-only here: a d_model vector
        cannot reconstruct the window, and pretending otherwise would silently
        fabricate context.  Edit activations via `decode_with_edit` /
        `rollout_with_edit`, which make the transience explicit.
        """
        if self.state_view != "obs_window":
            raise ValueError(
                "state_from_flat requires state_view='obs_window'; the "
                f"{self.state_view!r} view is read-only (use rollout_with_edit)"
            )
        B = flat.shape[0]
        buf = flat.reshape(B, self.state_span, self.cfg.input_dim)
        length = torch.full((B,), self.state_span, dtype=torch.long, device=flat.device)
        return TransformerState(buf, length)

    def _activations(self, state, edit=None) -> torch.Tensor:
        tokens = self.embed(state.obs_buffer)
        _, resids = self._run(
            tokens,
            self._win_mask(state.length, tokens.device),
            edit=edit,
            want_resid=True,
        )
        return resids[self.probe_layer][:, -1]

    def _kv_cache(self, state) -> torch.Tensor:
        tokens = self.embed(state.obs_buffer)
        sink: list = []
        self._run(tokens, self._win_mask(state.length, tokens.device), kv_sink=sink)
        return torch.cat([torch.cat([k, v], -1).flatten(1) for k, v in sink], dim=-1)

    # ── prediction ────────────────────────────────────────────────────────────

    def decode(self, state: TransformerState, edit=None) -> torch.Tensor:
        """Predict the next observation from the current window (GRU convention)."""
        tokens = self.embed(state.obs_buffer)
        h, _ = self._run(tokens, self._win_mask(state.length, tokens.device), edit=edit)
        return self.decoder(self.norm_out(h[:, -1]))

    def decode_with_edit(self, state, layer: int, resid: torch.Tensor):
        """Decode after forcing the residual stream at `layer`, current position."""
        return self.decode(state, edit=(layer, resid))

    def advance(self, state: TransformerState, obs_t: torch.Tensor):
        """Append an observation to the window."""
        buf = torch.cat([state.obs_buffer[:, 1:], obs_t[:, None, :]], dim=1)
        length = torch.clamp(state.length + 1, max=self.state_span)
        return TransformerState(buf, length)

    def step(self, obs_t: torch.Tensor, state: TransformerState | None = None):
        if state is None:
            B, R = obs_t.shape
            state = TransformerState(
                obs_t.new_zeros(B, self.state_span, R),
                torch.zeros(B, dtype=torch.long, device=obs_t.device),
            )
        state = self.advance(state, obs_t)
        return self.decode(state), state

    def predict_step(self, state: TransformerState):
        """Free-run: decode the next frame, then feed it back into the window."""
        pred = self.decode(state)
        return pred, self.advance(state, pred)

    def rollout_with_edit(self, state, layer: int, resid: torch.Tensor, steps: int):
        """Free-run whose FIRST step is produced under an activation edit.

        This is the honest measurement of a transformer activation write: the
        edit shapes the immediate prediction, that prediction enters the window,
        and every later step is recomputed from the window with no edit applied.
        Any persistence therefore has to travel through the observations — which
        is exactly the property under test.
        """
        pred = self.decode_with_edit(state, layer, resid)
        out = [pred]
        s = self.advance(state, pred)
        for _ in range(steps - 1):
            p, s = self.predict_step(s)
            out.append(p)
        return torch.stack(out, 1)
