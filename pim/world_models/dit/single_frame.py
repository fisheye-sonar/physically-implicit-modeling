"""Single-frame-token DiT world model — vanilla diffusion forcing.

The (b) variant of the DiT comparison: identical trunk to the concat DiT
(``pim/world_models/dit/model.py``) but with the *standard* diffusion-forcing
tokenization (Chen et al. 2024) instead of the paired-frame concat.

Formulation — how it differs from the concat DiT
------------------------------------------------
One token per frame, and the token IS the (noised) frame:

    token_t = x_τ(obs[t]) = (1-τ_t)·obs[t] + τ_t·ε_t,   per-token τ_t.

The network predicts the flow velocity v = ε − obs[t] at every position from
the token itself plus the *preceding* tokens (causal banded attention).  A
frame therefore plays two roles with ONE tensor: denoising target at its own
position, and history for later positions — so when a frame draws a high τ
during training, later positions see a noised version of it as context.
That is the package deal of vanilla diffusion forcing, and exactly what this
variant exists to measure against the concat design (whose clean channel
guarantees clean history at every noise pattern).

Prediction appends a pure-noise token after the observed frames and denoises
only that slot, context tokens held clean (τ=0).

Window semantics — off by one vs the concat DiT
-----------------------------------------------
``window`` is the attention band width in TOKENS.  The prediction slot is one
of them, so the model conditions on ``window − 1`` past frames — whereas the
concat DiT at the same ``window`` conditions on ``window`` past frames (each
of its tokens carries a frame pair).  Frame-span matching therefore pairs
single-frame W=3 with concat W=2, and W=5 with concat W=4.  Registry:
``notebooks/experiments/editability/transformers/DIT_RUNS.md``.

Everything else — rectified-flow interpolant, τ-mixture training
(``p_clean``/``p_one``), deterministic "mean" / "sample" prediction modes with
the fixed noise bank, the three state views, and the HiddenStateModel
protocol — mirrors the concat DiT; see its module docstring for the shared
rationale.  The carried state is the last ``window − 1`` observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn

from pim.world_models.dit.blocks import (
    DiTBlock,
    FinalLayer,
    RotaryEmbedding,
    TimestepEmbedder,
    band_causal_mask,
)

# ── State type ────────────────────────────────────────────────────────────────


class SingleFrameDiTState(NamedTuple):
    """Sliding-window state: the last ``window − 1`` observations.

    obs_buffer : (B, window-1, R) — right-aligned (newest at index -1),
                 zero-padded at the front while length < window-1.
    length     : (B,) int64 — number of valid frames in the buffer.
    """

    obs_buffer: torch.Tensor
    length: torch.Tensor


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class SingleFrameConfig:
    input_dim: int = 128  # obs_res — overridden from dataset at train time
    d_model: int = 128  # token width
    n_layers: int = 4  # transformer depth
    n_heads: int = 4  # attention heads
    mlp_ratio: float = 4.0  # MLP hidden width / d_model
    window: int = 16  # attention band width in tokens; context = window - 1
    n_sample_steps: int = 8  # Euler steps for the deterministic ODE sampler
    n_mean_eps: int = 8  # noise-bank size for the mean-mode readout
    noise_seed: int = 0  # seed for the fixed noise bank (deterministic modes)
    # "unit_interval" (observations in [0,1]) or "identity" (pre-normalised
    # latents); see pim/world_models/dit/model.py.
    data_transform: str = "unit_interval"
    variant: str = "single_frame"  # loader dispatch marker — do not change


# ── Model ─────────────────────────────────────────────────────────────────────


class SingleFrameDiTModel(nn.Module):
    """Vanilla diffusion-forcing DiT world model (single-frame tokens).

    Implements both WorldModel and HiddenStateModel protocols.

    Parameters
    ----------
    cfg:
        Model configuration.  ``window`` must be ≥ 2 (one context frame plus
        the prediction slot).
    """

    STATE_VIEWS = ("obs_window", "activations", "kv_cache")

    def __init__(self, cfg: SingleFrameConfig) -> None:
        super().__init__()
        if cfg.window < 2:
            raise ValueError("window must be >= 2 (context frame + prediction slot)")
        self.cfg = cfg

        # Runtime toggles, mirroring DiTModel (see its module docstring):
        # predict_mode ∈ {"mean", "sample", "sample_fresh"}.
        self.state_view: str = "obs_window"
        self.predict_mode: str = "mean"
        self.noise_gen: torch.Generator | None = None

        # Token input: one (possibly noised) frame → d_model
        self.token_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.t_embed = TimestepEmbedder(cfg.d_model)
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_heads)
        self.blocks = nn.ModuleList(
            DiTBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio)
            for _ in range(cfg.n_layers)
        )
        self.final_layer = FinalLayer(cfg.d_model, cfg.input_dim)

        # Fixed noise bank: mean mode averages its τ=1 readout over all rows;
        # sample mode starts its ODE from row 0.
        gen = torch.Generator().manual_seed(cfg.noise_seed)
        self.register_buffer(
            "_eps_bank",
            torch.randn(cfg.n_mean_eps, cfg.input_dim, generator=gen),
            persistent=False,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def _ctx(self) -> int:
        """Context frames carried between observations (= window − 1)."""
        return self.cfg.window - 1

    @property
    def hidden_size(self) -> int:
        """Flat state dimensionality under the active state view."""
        cfg = self.cfg
        if self.state_view == "obs_window":
            return self._ctx * cfg.input_dim
        if self.state_view == "activations":
            return cfg.d_model
        if self.state_view == "kv_cache":
            return cfg.n_layers * 2 * self._ctx * cfg.d_model
        raise ValueError(f"unknown state_view: {self.state_view!r}")

    # ── Diffusion space ───────────────────────────────────────────────────────

    def _to_diff(self, obs: torch.Tensor) -> torch.Tensor:
        if self.cfg.data_transform == "identity":
            return obs
        return 2.0 * obs - 1.0

    def _from_diff(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.data_transform == "identity":
            return x
        return (x + 1.0) / 2.0

    # ── Core network ──────────────────────────────────────────────────────────

    def _trunk(
        self,
        tokens_diff: torch.Tensor,
        tau: torch.Tensor,
        attn_mask: torch.Tensor,
        kv_sink: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the transformer over single-frame tokens.

        Parameters
        ----------
        tokens_diff : (B, T, R) frames (clean or noised), diff space
        tau         : (B, T) per-token diffusion times
        attn_mask   : bool, broadcastable to (B, n_heads, T, T)

        Returns
        -------
        feats : (B, T, d_model) final-block token features
        c     : (B, T, d_model) conditioning vectors (needed by final layer)
        """
        x = self.token_proj(tokens_diff)
        c = self.t_embed(tau)
        rope = self.rope(x.shape[1], x.device)
        for blk in self.blocks:
            x = blk(x, c, attn_mask, rope, kv_sink)
        return x, c

    def _denoise(
        self,
        tokens_diff: torch.Tensor,
        tau: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the flow velocity v = dx/dτ at every position. (B, T, R)."""
        feats, c = self._trunk(tokens_diff, tau, attn_mask)
        return self.final_layer(feats, c)

    # ── Training objective ────────────────────────────────────────────────────

    def diffusion_loss(
        self, obs: torch.Tensor, p_clean: float = 0.3, p_one: float = 0.1
    ) -> torch.Tensor:
        """Flow-matching loss over every position of a sequence batch.

        Same τ-mixture as the concat DiT (see its docstring), but each token
        is the noised frame itself, so a clean (τ=0) position is BOTH
        loss-excluded and the clean-history pattern later positions consume.

        Parameters
        ----------
        obs : (B, T, R) raw observation sequences

        Returns
        -------
        loss : scalar
        """
        x0 = self._to_diff(obs)  # (B, T, R)
        B, T, _ = x0.shape
        device = obs.device

        tau = torch.rand(B, T, device=device)
        r = torch.rand(B, T, device=device)
        clean = r < p_clean
        one = r > 1.0 - p_one  # disjoint from `clean` for p_clean + p_one < 1
        tau = tau.masked_fill(clean, 0.0).masked_fill(one, 1.0)

        eps = torch.randn_like(x0)
        x = (1.0 - tau[..., None]) * x0 + tau[..., None] * eps
        v_target = eps - x0

        mask = band_causal_mask(T, self.cfg.window, device)
        v = self._denoise(x, tau, mask)
        return ((v - v_target) ** 2)[~clean].mean()

    # ── Deterministic sampler ─────────────────────────────────────────────────

    def _pred_attn_mask(
        self, lengths: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """(N, 1, W, W) bool mask over [ctx tokens..., prediction slot].

        Causal + only the valid (right-aligned) context keys; the prediction
        slot is always a valid key for itself.
        """
        W = self.cfg.window
        causal = band_causal_mask(W, W, device)
        valid = torch.ones(lengths.shape[0], W, dtype=torch.bool, device=device)
        idx = torch.arange(self._ctx, device=device)
        valid[:, : self._ctx] = idx[None, :] >= (self._ctx - lengths[:, None])
        return causal[None, None] & valid[:, None, None, :]

    def _sample_next(
        self, windows: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Predict the next observation from a context window.

        Parameters
        ----------
        windows : (N, ctx, R) raw observations, right-aligned, zero-padded front
        lengths : (N,) valid frame counts

        Returns
        -------
        pred : (N, R) predicted next observation, raw space
        """
        if self.predict_mode == "mean":
            return self._predict_mean(windows, lengths)
        if self.predict_mode in ("sample", "sample_fresh"):
            return self._predict_sample(windows, lengths)
        raise ValueError(f"unknown predict_mode: {self.predict_mode!r}")

    def _start_noise(self, n: int, device: torch.device) -> torch.Tensor:
        """(n, R) ODE start noise; fresh per-row for "sample_fresh" (see DiTModel)."""
        R = self.cfg.input_dim
        if self.predict_mode == "sample_fresh":
            return torch.randn(n, R, generator=self.noise_gen).to(device)
        return self._eps_bank[0].expand(n, R)

    def _predict_mean(
        self, windows: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Conditional-mean readout: x̂₀ = mean_i[ε_i − v̂(ε_i, τ=1)]."""
        N, C, R = windows.shape
        E = self._eps_bank.shape[0]
        device = windows.device
        ctx = self._to_diff(windows)

        eps = self._eps_bank.repeat_interleave(N, dim=0)  # (E·N, R)
        tokens = torch.cat([ctx.repeat(E, 1, 1), eps.unsqueeze(1)], dim=1)
        tau = torch.zeros(E * N, C + 1, device=device)
        tau[:, -1] = 1.0
        attn_mask = self._pred_attn_mask(lengths, device).repeat(E, 1, 1, 1)

        v = self._denoise(tokens, tau, attn_mask)[:, -1]  # (E·N, R)
        x0 = (eps - v).view(E, N, R).mean(dim=0)  # x − τ·v at τ=1
        return self._from_diff(x0)

    def _predict_sample(
        self, windows: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Distribution-typical prediction: Euler ODE from the fixed noise."""
        N, C, R = windows.shape
        device = windows.device
        ctx = self._to_diff(windows)
        attn_mask = self._pred_attn_mask(lengths, device)

        tau = torch.zeros(N, C + 1, device=device)
        x = self._start_noise(N, device)
        taus = torch.linspace(1.0, 0.0, self.cfg.n_sample_steps + 1, device=device)
        for k in range(self.cfg.n_sample_steps):
            tokens = torch.cat([ctx, x.unsqueeze(1)], dim=1)
            tau_k = tau.clone()
            tau_k[:, -1] = taus[k]
            v = self._denoise(tokens, tau_k, attn_mask)[:, -1]
            x = x + (taus[k + 1] - taus[k]) * v
        return self._from_diff(x)

    # ── Window bookkeeping ────────────────────────────────────────────────────

    def _initial_state(
        self, batch_size: int, device: torch.device
    ) -> SingleFrameDiTState:
        C, R = self._ctx, self.cfg.input_dim
        return SingleFrameDiTState(
            obs_buffer=torch.zeros(batch_size, C, R, device=device),
            length=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    def _append(
        self, state: SingleFrameDiTState, obs_t: torch.Tensor
    ) -> SingleFrameDiTState:
        """Push one observation into the sliding window."""
        buffer = torch.cat([state.obs_buffer[:, 1:], obs_t.unsqueeze(1)], dim=1)
        length = torch.clamp(state.length + 1, max=self._ctx)
        return SingleFrameDiTState(buffer, length)

    def _unfold_windows(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice a sequence into per-timestep context windows (teacher forcing).

        Parameters
        ----------
        obs : (B, T, R)

        Returns
        -------
        windows : (B, T-1, ctx, R) — windows[:, t] holds frames t-ctx+1..t,
                  right-aligned, zero-padded at the front for t < ctx-1
        lengths : (T-1,) valid frame counts per timestep
        """
        B, T, R = obs.shape
        C = self._ctx
        pad = obs.new_zeros(B, C - 1, R)
        padded = torch.cat([pad, obs[:, :-1]], dim=1)  # (B, C-1 + T-1, R)
        windows = padded.unfold(1, C, 1).permute(0, 1, 3, 2).contiguous()
        lengths = torch.clamp(
            torch.arange(1, T, device=obs.device, dtype=torch.long), max=C
        )
        return windows, lengths

    _OBSERVE_CHUNK = 8192  # max forward rows per sampler call (memory bound)

    @property
    def _pred_chunk(self) -> int:
        """Windows per sampler call; mean mode folds the ε-bank into the batch."""
        if self.predict_mode == "mean":
            return max(1, self._OBSERVE_CHUNK // self._eps_bank.shape[0])
        return self._OBSERVE_CHUNK

    def _observe_core(self, obs: torch.Tensor) -> torch.Tensor:
        """Teacher-forced next-step predictions for every position. (B, T-1, R)."""
        B, T, R = obs.shape
        windows, lengths = self._unfold_windows(obs)
        flat_win = windows.reshape(B * (T - 1), self._ctx, R)
        flat_len = lengths.unsqueeze(0).expand(B, -1).reshape(-1)
        chunk = self._pred_chunk
        preds = torch.cat(
            [
                self._sample_next(flat_win[i : i + chunk], flat_len[i : i + chunk])
                for i in range(0, flat_win.shape[0], chunk)
            ]
        )
        return preds.reshape(B, T - 1, R)

    # ── State views ───────────────────────────────────────────────────────────

    def _probe_pass(
        self, windows: torch.Tensor, lengths: torch.Tensor, collect_kv: bool
    ) -> tuple[torch.Tensor, list]:
        """One deterministic forward for representation probing.

        Context tokens are clean (τ=0); the prediction slot carries the first
        bank noise vector at τ=1 — the configuration both prediction modes
        start from.
        """
        N, C, R = windows.shape
        device = windows.device
        ctx = self._to_diff(windows)
        tokens = torch.cat([ctx, self._eps_bank[0].expand(N, 1, R)], dim=1)
        tau = torch.zeros(N, C + 1, device=device)
        tau[:, -1] = 1.0
        attn_mask = self._pred_attn_mask(lengths, device)
        kv_sink: list = [] if collect_kv else None
        feats, _ = self._trunk(tokens, tau, attn_mask, kv_sink)
        return feats, kv_sink

    def _flat_view(self, windows: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Flatten a batch of windows under the active state view."""
        N, C, R = windows.shape
        if self.state_view == "obs_window":
            return windows.reshape(N, C * R)
        if self.state_view == "activations":
            feats, _ = self._probe_pass(windows, lengths, collect_kv=False)
            return feats[:, -1]
        if self.state_view == "kv_cache":
            _, kv = self._probe_pass(windows, lengths, collect_kv=True)
            # Per layer: keep the C context tokens, zero the padded ones.
            valid = torch.arange(C, device=windows.device)[None, :] >= (
                C - lengths[:, None]
            )  # (N, C)
            parts = []
            for k, v in kv:
                for x in (k, v):
                    x = x[:, :, :C]  # (N, heads, C, head_dim)
                    x = x * valid[:, None, :, None]
                    parts.append(x.transpose(1, 2).reshape(N, C, -1))
            return torch.cat(parts, dim=-1).reshape(N, -1)
        raise ValueError(f"unknown state_view: {self.state_view!r}")

    # ── SSM protocol methods ──────────────────────────────────────────────────

    def flat_state(self, state: SingleFrameDiTState) -> torch.Tensor:
        """Model-native state → (B, hidden_size) under the active view."""
        return self._flat_view(state.obs_buffer, state.length)

    def state_from_flat(self, flat: torch.Tensor) -> SingleFrameDiTState:
        """(B, ctx·R) flat obs-window → SingleFrameDiTState.

        Only supported for the "obs_window" view.  Injected states are
        assumed fully warmed (length = ctx).
        """
        if self.state_view != "obs_window":
            raise ValueError(
                "state_from_flat requires state_view='obs_window' "
                f"(active view {self.state_view!r} is read-only)"
            )
        C, R = self._ctx, self.cfg.input_dim
        buffer = flat.reshape(-1, C, R)
        length = torch.full((buffer.shape[0],), C, dtype=torch.long, device=flat.device)
        return SingleFrameDiTState(buffer, length)

    def decode(self, state: SingleFrameDiTState) -> torch.Tensor:
        """Predict the next observation from the current window (no advance).

        GRU convention: decode(state after obs_t) ≈ obs_{t+1}.  Differentiable
        for gradient-based editors.
        """
        return self._sample_next(state.obs_buffer, state.length)

    @torch.no_grad()
    def observe_sequence(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forcing pass: predictions + flat states per view.

        Returns
        -------
        pred   : (B, T-1, R)  deterministic next-step predictions
        h_flat : (B, T-1, hidden_size)  flat states aligned to obs[:, :-1]
        """
        B, T, R = obs.shape
        pred = self._observe_core(obs)
        windows, lengths = self._unfold_windows(obs)
        flat_win = windows.reshape(B * (T - 1), self._ctx, R)
        flat_len = lengths.unsqueeze(0).expand(B, -1).reshape(-1)
        h_flat = torch.cat(
            [
                self._flat_view(
                    flat_win[i : i + self._OBSERVE_CHUNK],
                    flat_len[i : i + self._OBSERVE_CHUNK],
                )
                for i in range(0, flat_win.shape[0], self._OBSERVE_CHUNK)
            ]
        )
        return pred, h_flat.reshape(B, T - 1, -1)

    @torch.no_grad()
    def predict_step(
        self, state: SingleFrameDiTState
    ) -> tuple[torch.Tensor, SingleFrameDiTState]:
        """Free-running step: decode the current window, feed the prediction back."""
        obs_hat = self.decode(state)
        return self.step(obs_hat, state)

    # ── Protocol interface ────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(
        self,
        obs: torch.Tensor,
        h0: SingleFrameDiTState | None = None,
    ) -> tuple[torch.Tensor, SingleFrameDiTState]:
        """Teacher-forcing pass over a full sequence (protocol; eval-only).

        Training does NOT go through forward — use ``diffusion_loss``.
        """
        if h0 is not None:
            raise NotImplementedError(
                "SingleFrameDiTModel.forward does not support h0"
            )
        pred = self._observe_core(obs)
        B, T, R = obs.shape
        C = self._ctx
        state = SingleFrameDiTState(
            obs_buffer=self._pad_window(obs[:, -C:]),
            length=torch.full((B,), min(T, C), dtype=torch.long, device=obs.device),
        )
        return pred, state

    def _pad_window(self, frames: torch.Tensor) -> torch.Tensor:
        """Right-align (B, ≤ctx, R) frames into a (B, ctx, R) buffer."""
        B, n, R = frames.shape
        C = self._ctx
        if n == C:
            return frames
        return torch.cat([frames.new_zeros(B, C - n, R), frames], dim=1)

    def step(
        self,
        obs_t: torch.Tensor,
        state: SingleFrameDiTState | None = None,
    ) -> tuple[torch.Tensor, SingleFrameDiTState]:
        """Single-step autoregressive forward (for rollout / evaluation)."""
        if state is None:
            state = self._initial_state(obs_t.shape[0], obs_t.device)
        state = self._append(state, obs_t)
        pred = self._sample_next(state.obs_buffer, state.length)
        return pred, state

    @torch.no_grad()
    def get_hidden_states(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract per-timestep flat states (active view) via teacher-forcing.

        Returns
        -------
        h : (B, T-1, hidden_size)
            h[:, t, :] is the state after seeing obs[:, t, :].
        """
        B, T, R = obs.shape
        windows, lengths = self._unfold_windows(obs)
        flat_win = windows.reshape(B * (T - 1), self._ctx, R)
        flat_len = lengths.unsqueeze(0).expand(B, -1).reshape(-1)
        h = torch.cat(
            [
                self._flat_view(
                    flat_win[i : i + self._OBSERVE_CHUNK],
                    flat_len[i : i + self._OBSERVE_CHUNK],
                )
                for i in range(0, flat_win.shape[0], self._OBSERVE_CHUNK)
            ]
        )
        return h.reshape(B, T - 1, -1)
