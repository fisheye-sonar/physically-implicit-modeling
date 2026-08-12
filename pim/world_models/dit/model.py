"""Diffusion transformer (DiT) world model — causal, observation-space.

Trained on the same task as the GRU/RSSM: predict obs[t+1] from obs[<=t],
receiving one observation at a time.  No latent space, no action/text
conditioning — the only conditioning is the diffusion time τ, as is standard.

Formulation (diffusion forcing over a causal token stream)
----------------------------------------------------------
One token per frame.  The token at position t packs
    concat( obs[t],  x_τ(obs[t+1]) )  +  AdaLN conditioning on τ_t,
where x_τ = (1-τ)·obs[t+1] + τ·ε is the rectified-flow interpolant of the
*next* frame.  Attention is causal with a banded window of W tokens, so
position t sees clean obs[t-W+1..t] plus the noised next frame — exactly the
information available at deployment.  The network predicts the flow velocity
    v = dx/dτ = ε − obs[t+1]
at every position in a single forward pass (per-position independent τ during
training; past positions are fed τ=0, i.e. their next frame is the *observed*
clean frame, which is what inference provides).

Prediction modes (``model.predict_mode``)
-----------------------------------------
The first two are exact functions of the input (fixed seeded noise buffers, no
sampling), so the model can be evaluated deterministically despite the diffusion
parameterisation; the third is the honest stochastic generator.

  * "mean" (default) — one forward at τ=1 per noise vector in a fixed bank,
    averaged:  x̂₀ = mean_i[ ε_i − v̂(ε_i, τ=1, ctx) ].  At τ=1 the optimal
    velocity field is v*(x, 1) = x − E[x₀ | ctx] for every x, so this reads
    out the model's CONDITIONAL MEAN — the deterministic predictor the
    GRU approximates with MSE training, and the analogue of evaluating the
    RSSM in prior-mean mode.  Averaging over a small bank washes out the
    ε-specific transport structure a single vector inherits (empirically
    1→4 vectors is a large gain, 4→16 marginal).
  * "sample" — integrate the ODE dx/dτ = v from τ=1 → 0 with
    ``n_sample_steps`` Euler steps starting from the FIXED first bank
    vector: a distribution-typical prediction.  NB this reproduces a
    realisation of the observation noise (it is a generative sample), so
    its MSE vs clean/noisy targets is intrinsically worse than mean mode —
    that is faithfulness, not error.  Use it for generative-quality
    questions, not prediction metrics.
  * "sample_fresh" — the same ODE from INDEPENDENT per-row noise, redrawn on
    every call (seed ``model.noise_gen`` for reproducibility).  The only
    non-deterministic mode, and **the one to use for autoregressive
    rollouts**: reusing the fixed vector at every step makes the model
    re-generate one ε-specific texture and the rollout degenerates into
    stripes (diagnosed 2026-08-11).

What is the "state"?
--------------------
The transformer recomputes attention every step, so the only thing genuinely
carried forward from one observation to the next is the sliding window of the
last W observations.  The KV cache is a deterministic *function* of that
window (a cache, not extra state).  Three views of the state are exposed via
``model.state_view`` (runtime toggle, like RSSM's ``sample``):

  * "obs_window"  (default) — the raw W-frame buffer, flattened (W·R dims).
    Minimal, exact, and invertible: ``state_from_flat`` works, so editors and
    the controllability eval operate on this view.
  * "activations" — final-block token features at the current position,
    computed at τ=1 from the fixed noise (the deterministic "pre-denoising"
    representation the model builds from context alone; d_model dims).  The
    closest analogue to a GRU's h.  Read-only (not invertible).
  * "kv_cache"    — post-RoPE K/V of every layer for the W-1 completed
    tokens in the window: what an incremental implementation would carry
    instead of recomputing.  Large (n_layers·2·(W-1)·d_model dims); intended
    for targeted probing on subsets, not the full eval suite.  Read-only.

``hidden_size``, ``get_hidden_states``, ``observe_sequence`` and
``flat_state`` all follow the active view, so probes fit on whichever state
notion is selected.

Protocol
--------
Implements WorldModel + HiddenStateModel unchanged.  ``decode(state)``
predicts the next observation from the current window (the GRU convention:
decoder(h_t) ≈ obs[t+1]); ``predict_step`` = decode + step, mirroring the GRU.
Training uses ``diffusion_loss(obs)`` (flow-matching MSE), not ``forward`` —
sampled predictions are eval-only.
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


class DiTState(NamedTuple):
    """Sliding-window state: everything the DiT carries between observations.

    obs_buffer : (B, W, R) — last W observations, right-aligned (newest at
                 index -1), zero-padded at the front while length < W.
    length     : (B,) int64 — number of valid frames in the buffer (≤ W).
    """

    obs_buffer: torch.Tensor
    length: torch.Tensor


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    input_dim: int = 128  # obs_res — overridden from dataset at train time
    d_model: int = 128  # token width
    n_layers: int = 4  # transformer depth
    n_heads: int = 4  # attention heads
    mlp_ratio: float = 4.0  # MLP hidden width / d_model
    window: int = 16  # sliding attention window = frames of state carried
    n_sample_steps: int = 8  # Euler steps for the deterministic ODE sampler
    n_mean_eps: int = 8  # noise-bank size for the mean-mode readout
    noise_seed: int = 0  # seed for the fixed noise bank (deterministic modes)
    # How raw data maps into the space the flow runs in:
    #   "unit_interval" — observations in [0,1] → [-1,1]  (pixel-space DiT)
    #   "identity"      — already ≈zero-mean unit-scale   (latent DiT; the VAE's
    #                     `latent_scale` has done the normalisation)
    data_transform: str = "unit_interval"


# ── Model ─────────────────────────────────────────────────────────────────────


class DiTModel(nn.Module):
    """Causal diffusion transformer world model.

    Implements both WorldModel and HiddenStateModel protocols.

    Parameters
    ----------
    cfg:
        Model configuration.
    """

    STATE_VIEWS = ("obs_window", "activations", "kv_cache")

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Which notion of state flat_state/get_hidden_states expose.
        # Runtime toggle (not part of the checkpointed config), see module
        # docstring.  Only "obs_window" supports state_from_flat.
        self.state_view: str = "obs_window"

        # Prediction mode: "mean" (conditional-mean readout, default),
        # "sample" (K-step ODE from the FIXED bank vector — deterministic), or
        # "sample_fresh" (K-step ODE from PER-SAMPLE fresh noise — the honest
        # generative mode; the only non-deterministic one).  Runtime toggle,
        # see module docstring.
        self.predict_mode: str = "mean"
        # Optional torch.Generator for "sample_fresh"; None → global RNG.
        # Seed it for reproducible stochastic rollouts.
        self.noise_gen: torch.Generator | None = None

        # Token input: concat(current obs, noised next obs) → d_model
        self.token_proj = nn.Linear(2 * cfg.input_dim, cfg.d_model)
        self.t_embed = TimestepEmbedder(cfg.d_model)
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_heads)
        self.blocks = nn.ModuleList(
            DiTBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio)
            for _ in range(cfg.n_layers)
        )
        self.final_layer = FinalLayer(cfg.d_model, cfg.input_dim)

        # Fixed noise bank: mean mode averages its τ=1 readout over all rows;
        # sample mode starts its ODE from row 0.  Regenerated from the seed
        # (non-persistent), which is what makes both modes deterministic.
        gen = torch.Generator().manual_seed(cfg.noise_seed)
        self.register_buffer(
            "_eps_bank",
            torch.randn(cfg.n_mean_eps, cfg.input_dim, generator=gen),
            persistent=False,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def hidden_size(self) -> int:
        """Flat state dimensionality under the active state view."""
        cfg = self.cfg
        if self.state_view == "obs_window":
            return cfg.window * cfg.input_dim
        if self.state_view == "activations":
            return cfg.d_model
        if self.state_view == "kv_cache":
            return cfg.n_layers * 2 * (cfg.window - 1) * cfg.d_model
        raise ValueError(f"unknown state_view: {self.state_view!r}")

    # ── Diffusion space ───────────────────────────────────────────────────────
    # The flow runs between N(0,1) noise and data at comparable scale.  For
    # observations in [0, 1] that means rescaling to [-1, 1]; for VAE latents
    # already normalised by `latent_scale` it is the identity.  See
    # `cfg.data_transform`.

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
        cur_diff: torch.Tensor,
        nxt_diff: torch.Tensor,
        tau: torch.Tensor,
        attn_mask: torch.Tensor,
        kv_sink: list | None = None,
        resid_sink: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the transformer over paired-frame tokens.

        Parameters
        ----------
        cur_diff : (B, T, R) current frames, diff space
        nxt_diff : (B, T, R) next-frame channel (clean or noised), diff space
        tau      : (B, T) per-token diffusion times
        attn_mask: bool, broadcastable to (B, n_heads, T, T)
        resid_sink : if given, the residual stream is appended at every residual
            point — n_layers+1 entries: the token embedding (input to block 1),
            the input to each later block, and the final-block output (which
            equals the returned ``feats``).  For layer-resolved probing.

        Returns
        -------
        feats : (B, T, d_model) final-block token features
        c     : (B, T, d_model) conditioning vectors (needed by final layer)
        """
        x = self.token_proj(torch.cat([cur_diff, nxt_diff], dim=-1))
        c = self.t_embed(tau)
        rope = self.rope(x.shape[1], x.device)
        for blk in self.blocks:
            if resid_sink is not None:
                resid_sink.append(x)
            x = blk(x, c, attn_mask, rope, kv_sink)
        if resid_sink is not None:
            resid_sink.append(x)
        return x, c

    def _denoise(
        self,
        cur_diff: torch.Tensor,
        nxt_diff: torch.Tensor,
        tau: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the flow velocity v = dx/dτ at every position. (B, T, R)."""
        feats, c = self._trunk(cur_diff, nxt_diff, tau, attn_mask)
        return self.final_layer(feats, c)

    # ── Training objective ────────────────────────────────────────────────────

    def diffusion_loss(
        self, obs: torch.Tensor, p_clean: float = 0.3, p_one: float = 0.1
    ) -> torch.Tensor:
        """Flow-matching loss over every position of a sequence batch.

        Each position draws its own τ, from a mixture targeting the patterns
        inference actually uses:
          * with prob ``p_clean`` — τ=0 exactly (clean next frame): the past-
            context pattern.  Excluded from the loss (the velocity target is
            unpredictable noise there); these positions exist to train *later*
            positions against clean context.
          * with prob ``p_one`` — τ=1 exactly (pure noise input): the mean-
            readout point.  At τ=1 the loss regresses
            v̂(ε, 1, ctx) → ε − E[obs_{t+1} | ctx], i.e. it directly trains
            the conditional-mean readout that predict_mode="mean" evaluates.
          * otherwise — τ ~ U(0, 1): the ODE-path interior.

        Parameters
        ----------
        obs : (B, T, R) raw observation sequences

        Returns
        -------
        loss : scalar
        """
        cur = self._to_diff(obs[:, :-1])  # (B, T-1, R)
        target = self._to_diff(obs[:, 1:])  # (B, T-1, R)
        B, Tm1, _ = cur.shape
        device = obs.device

        tau = torch.rand(B, Tm1, device=device)
        r = torch.rand(B, Tm1, device=device)
        clean = r < p_clean
        one = r > 1.0 - p_one  # disjoint from `clean` for p_clean + p_one < 1
        tau = tau.masked_fill(clean, 0.0).masked_fill(one, 1.0)

        eps = torch.randn_like(target)
        x = (1.0 - tau[..., None]) * target + tau[..., None] * eps
        v_target = eps - target

        mask = band_causal_mask(Tm1, self.cfg.window, device)
        v = self._denoise(cur, x, tau, mask)
        return ((v - v_target) ** 2)[~clean].mean()

    # ── Deterministic sampler ─────────────────────────────────────────────────

    def _window_attn_mask(
        self, lengths: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """(N, 1, W, W) bool mask: causal + only the valid (right-aligned) keys."""
        W = self.cfg.window
        causal = band_causal_mask(W, W, device)
        valid = torch.arange(W, device=device)[None, :] >= (W - lengths[:, None])
        return causal[None, None] & valid[:, None, None, :]

    def _window_tokens(
        self, windows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split a window into token channels: (cur, nxt) in diff space.

        Token j pairs frame j with frame j+1; the last token (the current
        frame) has no observed next frame — its nxt channel is filled by the
        caller (noise/iterate for sampling, _eps0 for probing).
        """
        cur = self._to_diff(windows)  # (N, W, R)
        nxt = torch.zeros_like(cur)
        nxt[:, :-1] = cur[:, 1:]  # completed pairs are clean (τ=0)
        return cur, nxt

    def _sample_next(
        self, windows: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Predict the next observation from an observation window.

        Dispatches on ``predict_mode`` (see module docstring).  Both modes
        are fully deterministic.

        Parameters
        ----------
        windows : (N, W, R) raw observations, right-aligned, zero-padded front
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
        """(n, R) ODE start noise for the sampling modes.

        "sample"       — the fixed bank vector, shared by every row (deterministic).
        "sample_fresh" — independent noise per row, redrawn on every call.  This is
        what an autoregressive rollout needs: reusing one fixed vector at every
        rollout step makes the model re-generate the same ε-specific texture and
        the rollout degenerates (diagnosed 2026-08-11).
        """
        R = self.cfg.input_dim
        if self.predict_mode == "sample_fresh":
            return torch.randn(n, R, generator=self.noise_gen).to(device)
        return self._eps_bank[0].expand(n, R)

    def _predict_mean(
        self, windows: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Conditional-mean readout: x̂₀ = mean_i[ε_i − v̂(ε_i, τ=1)].

        The bank dimension is folded into the batch so all noise vectors run
        in one forward.
        """
        N, W, R = windows.shape
        E = self._eps_bank.shape[0]
        device = windows.device
        cur, nxt_base = self._window_tokens(windows)

        cur_t = cur.repeat(E, 1, 1)  # (E·N, W, R), block e = bank row e
        eps = self._eps_bank.repeat_interleave(N, dim=0)  # (E·N, R)
        nxt = torch.cat([nxt_base[:, :-1].repeat(E, 1, 1), eps.unsqueeze(1)], dim=1)
        tau = torch.zeros(E * N, W, device=device)
        tau[:, -1] = 1.0
        attn_mask = self._window_attn_mask(lengths, device).repeat(E, 1, 1, 1)

        v = self._denoise(cur_t, nxt, tau, attn_mask)[:, -1]  # (E·N, R)
        x0 = (eps - v).view(E, N, R).mean(dim=0)  # x − τ·v at τ=1
        return self._from_diff(x0)

    def _predict_sample(
        self, windows: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Distribution-typical prediction: Euler ODE from the start noise.

        Start noise per ``predict_mode`` — fixed bank vector ("sample") or fresh
        per-sample noise ("sample_fresh"); see ``_start_noise``.
        """
        N, W, R = windows.shape
        device = windows.device
        cur, nxt = self._window_tokens(windows)
        attn_mask = self._window_attn_mask(lengths, device)

        tau = torch.zeros(N, W, device=device)
        x = self._start_noise(N, device)
        taus = torch.linspace(1.0, 0.0, self.cfg.n_sample_steps + 1, device=device)
        for k in range(self.cfg.n_sample_steps):
            nxt_k = torch.cat([nxt[:, :-1], x.unsqueeze(1)], dim=1)
            tau_k = tau.clone()
            tau_k[:, -1] = taus[k]
            v = self._denoise(cur, nxt_k, tau_k, attn_mask)[:, -1]
            x = x + (taus[k + 1] - taus[k]) * v
        return self._from_diff(x)

    # ── Window bookkeeping ────────────────────────────────────────────────────

    def _initial_state(self, batch_size: int, device: torch.device) -> DiTState:
        W, R = self.cfg.window, self.cfg.input_dim
        return DiTState(
            obs_buffer=torch.zeros(batch_size, W, R, device=device),
            length=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    def _append(self, state: DiTState, obs_t: torch.Tensor) -> DiTState:
        """Push one observation into the sliding window."""
        buffer = torch.cat([state.obs_buffer[:, 1:], obs_t.unsqueeze(1)], dim=1)
        length = torch.clamp(state.length + 1, max=self.cfg.window)
        return DiTState(buffer, length)

    def _unfold_windows(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice a sequence into per-timestep windows (teacher forcing).

        Parameters
        ----------
        obs : (B, T, R)

        Returns
        -------
        windows : (B, T-1, W, R) — windows[:, t] holds frames t-W+1..t,
                  right-aligned, zero-padded at the front for t < W-1
        lengths : (T-1,) valid frame counts per timestep
        """
        B, T, R = obs.shape
        W = self.cfg.window
        pad = obs.new_zeros(B, W - 1, R)
        padded = torch.cat([pad, obs[:, :-1]], dim=1)  # (B, W-1 + T-1, R)
        windows = padded.unfold(1, W, 1).permute(0, 1, 3, 2).contiguous()
        lengths = torch.clamp(
            torch.arange(1, T, device=obs.device, dtype=torch.long), max=W
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
        """Teacher-forced next-step predictions for every position.

        Each position is denoised against its own window with *clean* past
        pairs — exactly matching what step() computes sequentially.

        Parameters
        ----------
        obs : (B, T, R)

        Returns
        -------
        pred : (B, T-1, R)
        """
        B, T, R = obs.shape
        windows, lengths = self._unfold_windows(obs)
        flat_win = windows.reshape(B * (T - 1), self.cfg.window, R)
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

        Completed pairs are clean (τ=0); the current token carries the first
        bank noise vector at τ=1 — the configuration both prediction modes
        start from, i.e. the representation the model computes from context
        alone.
        """
        N, W, R = windows.shape
        device = windows.device
        cur, nxt = self._window_tokens(windows)
        nxt = torch.cat([nxt[:, :-1], self._eps_bank[0].expand(N, 1, R)], dim=1)
        tau = torch.zeros(N, W, device=device)
        tau[:, -1] = 1.0
        attn_mask = self._window_attn_mask(lengths, device)
        kv_sink: list = [] if collect_kv else None
        feats, _ = self._trunk(cur, nxt, tau, attn_mask, kv_sink)
        return feats, kv_sink

    def _flat_view(self, windows: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Flatten a batch of windows under the active state view. (N, hidden_size)."""
        N, W, R = windows.shape
        if self.state_view == "obs_window":
            return windows.reshape(N, W * R)
        if self.state_view == "activations":
            feats, _ = self._probe_pass(windows, lengths, collect_kv=False)
            return feats[:, -1]
        if self.state_view == "kv_cache":
            _, kv = self._probe_pass(windows, lengths, collect_kv=True)
            # Per layer: (N, heads, W, head_dim) → keep the W-1 completed
            # tokens, zero the padded ones, flatten everything.
            valid = torch.arange(W - 1, device=windows.device)[None, :] >= (
                W - lengths[:, None]
            )  # (N, W-1)
            parts = []
            for k, v in kv:
                for x in (k, v):
                    x = x[:, :, : W - 1]  # (N, heads, W-1, head_dim)
                    x = x * valid[:, None, :, None]
                    parts.append(x.transpose(1, 2).reshape(N, W - 1, -1))
            return torch.cat(parts, dim=-1).reshape(N, -1)
        raise ValueError(f"unknown state_view: {self.state_view!r}")

    # ── SSM protocol methods ──────────────────────────────────────────────────

    def flat_state(self, state: DiTState) -> torch.Tensor:
        """Model-native state → (B, hidden_size) under the active view."""
        return self._flat_view(state.obs_buffer, state.length)

    def state_from_flat(self, flat: torch.Tensor) -> DiTState:
        """(B, W·R) flat obs-window → DiTState.

        Only supported for the "obs_window" view (the other views are not
        invertible).  Injected states are assumed fully warmed (length = W);
        eval warm-ups run ≥ W frames before editing, so this holds in
        practice.
        """
        if self.state_view != "obs_window":
            raise ValueError(
                "state_from_flat requires state_view='obs_window' "
                f"(active view {self.state_view!r} is read-only)"
            )
        W, R = self.cfg.window, self.cfg.input_dim
        buffer = flat.reshape(-1, W, R)
        length = torch.full((buffer.shape[0],), W, dtype=torch.long, device=flat.device)
        return DiTState(buffer, length)

    def decode(self, state: DiTState) -> torch.Tensor:
        """Predict the next observation from the current window (no advance).

        GRU convention: decode(state after obs_t) ≈ obs_{t+1}.  Differentiable
        (backprop through the sampler) for gradient-based editors.

        Parameters
        ----------
        state : DiTState

        Returns
        -------
        obs : (B, R)
        """
        return self._sample_next(state.obs_buffer, state.length)

    @torch.no_grad()
    def observe_sequence(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forcing pass: sampled predictions + flat states per view.

        Parameters
        ----------
        obs : (B, T, R)

        Returns
        -------
        pred   : (B, T-1, R)  deterministic sampled next-step predictions
        h_flat : (B, T-1, hidden_size)  flat states aligned to obs[:, :-1]
        """
        B, T, R = obs.shape
        pred = self._observe_core(obs)
        windows, lengths = self._unfold_windows(obs)
        flat_win = windows.reshape(B * (T - 1), self.cfg.window, R)
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
    def predict_step(self, state: DiTState) -> tuple[torch.Tensor, DiTState]:
        """Free-running step: decode the current window, feed the prediction back.

        Mirrors the GRU: obs_hat = decode(state) ≈ obs_{t+1} enters the window
        as if observed, and the returned prediction is for the frame after.

        Parameters
        ----------
        state : DiTState

        Returns
        -------
        pred_next  : (B, R) — prediction for frame t+2
        state_next : DiTState advanced by the imagined frame
        """
        obs_hat = self.decode(state)
        return self.step(obs_hat, state)

    # ── Protocol interface ────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(
        self,
        obs: torch.Tensor,
        h0: DiTState | None = None,
    ) -> tuple[torch.Tensor, DiTState]:
        """Teacher-forcing pass over a full sequence (protocol; eval-only).

        Training does NOT go through forward — use ``diffusion_loss``.

        Parameters
        ----------
        obs : (B, T, R)
        h0  : must be None (windowed attention starts from an empty buffer)

        Returns
        -------
        pred      : (B, T-1, R) sampled next-step predictions
        state_out : DiTState after the full sequence
        """
        if h0 is not None:
            raise NotImplementedError("DiTModel.forward does not support h0")
        pred = self._observe_core(obs)
        W = self.cfg.window
        B, T, R = obs.shape
        state = DiTState(
            obs_buffer=self._pad_window(obs[:, -W:]),
            length=torch.full((B,), min(T, W), dtype=torch.long, device=obs.device),
        )
        return pred, state

    def _pad_window(self, frames: torch.Tensor) -> torch.Tensor:
        """Right-align (B, ≤W, R) frames into a (B, W, R) buffer."""
        B, n, R = frames.shape
        W = self.cfg.window
        if n == W:
            return frames
        return torch.cat([frames.new_zeros(B, W - n, R), frames], dim=1)

    def step(
        self,
        obs_t: torch.Tensor,
        state: DiTState | None = None,
    ) -> tuple[torch.Tensor, DiTState]:
        """Single-step autoregressive forward (for rollout / evaluation).

        Parameters
        ----------
        obs_t : (B, R) current observation
        state : DiTState, or None for an empty window

        Returns
        -------
        pred_t     : (B, R) predicted next observation
        state_next : DiTState including obs_t
        """
        if state is None:
            state = self._initial_state(obs_t.shape[0], obs_t.device)
        state = self._append(state, obs_t)
        pred = self._sample_next(state.obs_buffer, state.length)
        return pred, state

    @torch.no_grad()
    def get_hidden_states(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract per-timestep flat states (active view) via teacher-forcing.

        Parameters
        ----------
        obs : (B, T, R) observation sequence

        Returns
        -------
        h : (B, T-1, hidden_size)
            h[:, t, :] is the state after seeing obs[:, t, :].
            Aligns with positions[:, t, :] and is_visible[:, t, :].
        """
        B, T, R = obs.shape
        windows, lengths = self._unfold_windows(obs)
        flat_win = windows.reshape(B * (T - 1), self.cfg.window, R)
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
