"""Latent diffusion transformer world model — a frozen per-frame VAE + a DiT in latent space.

The modern video-model recipe (VAE + latent DiT) applied to this repo's 1D world,
and treated as a **wholly separate architecture** from the pixel-space DiT: the
world model never sees a 128-ray observation, only the VAE's z-dim code.

    obs (B, T, R) --VAE encode--> z (B, T, Z) --DiT--> ẑ_{t+1} --VAE decode--> ôbs_{t+1}

Why this architecture is worth having
-------------------------------------
Every probe-gradient editor in this repo has failed the same way in observation
space (`notebooks/experiments/editability/input_grad_steering/`): the gradient
flips the readout without moving content, because a 128-d surface has a large
subspace of readout-changing directions orthogonal to anything semantic.  A z≈16
code has almost no room for that, and the frozen decoder is a second,
unconditional manifold projector — anything done in z decodes to a valid-looking
observation.  Whether that fixes controllability is the question this
architecture exists to answer; see `research/directions/latent-dit-vae.md`.

Normalisation
-------------
The VAE's `latent_scale` (measured post-training) puts codes at ≈unit scale, and
the DiT core is built with `data_transform="identity"` so its rectified-flow
interpolant runs directly on them.  All state, prediction, and editing inside the
core happen in **normalised latent space**; the wrapper converts at the boundary.

State
-----
Same three views as the pixel DiT, one rename: the carried state is a window of
**latents**, not observations.

  * "latent_window" (default) — the W·Z normalised-latent buffer.  Invertible, so
    `state_from_flat` works and window-write editors operate here.
  * "activations" — final-block token features at the current position (d_model).
  * "kv_cache" — read-only, as in the pixel DiT.

Protocol
--------
Implements WorldModel + HiddenStateModel in **observation space** — `step()` takes
an observation and encodes it, `decode()` returns a decoded observation — so the
whole eval suite, §4 metrics and waterfalls run unchanged.  Latent-space handles
(`encode`, `decode_latent`, `core`) are exposed for latent-native editors.
Training uses `diffusion_loss(obs)`, which encodes and delegates to the core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import torch
import torch.nn as nn

from pim.world_models.dit.model import DiTModel, DiTState
from pim.world_models.dit.model import ModelConfig as DiTCoreConfig
from pim.world_models.vae import ObsVAE, VAEConfig


class LatentDiTState(NamedTuple):
    """Sliding window of normalised latents (the carried state).

    latent_buffer : (B, W, Z) — newest at index -1, zero-padded at the front.
    length        : (B,) int64 — valid frames in the buffer (≤ W).
    """

    latent_buffer: torch.Tensor
    length: torch.Tensor


@dataclass
class LatentDiTConfig:
    """Config for the composite model.  `vae` and `core` are the sub-configs."""

    vae: dict = field(default_factory=dict)  # VAEConfig as a dict
    core: dict = field(default_factory=dict)  # DiT ModelConfig as a dict
    vae_checkpoint: str = ""  # provenance only — weights live in this checkpoint
    variant: str = "latent_dit"  # loader dispatch marker — do not change


class LatentDiTModel(nn.Module):
    """Frozen per-frame VAE + causal DiT over its latents.

    Parameters
    ----------
    cfg : LatentDiTConfig

    Notes
    -----
    The VAE is frozen (`requires_grad_(False)`, kept in eval mode) — it is a
    fixed representation, not a jointly-trained component.  Its weights are
    saved inside this model's `state_dict` so a checkpoint is self-contained.
    """

    STATE_VIEWS = ("latent_window", "activations", "kv_cache")

    def __init__(self, cfg: LatentDiTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vae = ObsVAE(VAEConfig(**cfg.vae))
        self.vae.requires_grad_(False)
        self.vae.eval()
        core_cfg = dict(cfg.core)
        core_cfg["data_transform"] = "identity"  # latents are pre-normalised
        core_cfg["input_dim"] = self.vae.cfg.latent_dim
        self.core = DiTModel(DiTCoreConfig(**core_cfg))
        self.state_view: str = "latent_window"

    def train(self, mode: bool = True):  # noqa: D102 — keep the VAE frozen/eval
        super().train(mode)
        self.vae.eval()
        return self

    # ── passthrough toggles (the core owns prediction behaviour) ──────────────

    @property
    def predict_mode(self) -> str:
        return self.core.predict_mode

    @predict_mode.setter
    def predict_mode(self, value: str) -> None:
        self.core.predict_mode = value

    @property
    def noise_gen(self) -> torch.Generator | None:
        return self.core.noise_gen

    @noise_gen.setter
    def noise_gen(self, value: torch.Generator | None) -> None:
        self.core.noise_gen = value

    @property
    def latent_dim(self) -> int:
        return self.vae.cfg.latent_dim

    @property
    def window(self) -> int:
        return self.core.cfg.window

    @property
    def hidden_size(self) -> int:
        """Flat state dimensionality under the active view."""
        if self.state_view == "latent_window":
            return self.core.cfg.window * self.latent_dim
        self.core.state_view = self.state_view
        return self.core.hidden_size

    # ── VAE boundary ─────────────────────────────────────────────────────────

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """(..., R) observation → (..., Z) normalised latent (differentiable)."""
        return self.vae.encode_normalized(obs)

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """(..., Z) normalised latent → (..., R) observation (differentiable)."""
        return self.vae.decode_normalized(z)

    # ── state plumbing ───────────────────────────────────────────────────────

    def _to_core(self, state: LatentDiTState) -> DiTState:
        return DiTState(state.latent_buffer, state.length)

    def state_from_latents(self, latents: torch.Tensor) -> LatentDiTState:
        """Build the carried state from a run of latents (newest last)."""
        B, n, Z = latents.shape
        W = self.core.cfg.window
        buf = (
            latents[:, -W:].contiguous()
            if n >= W
            else torch.cat([latents.new_zeros(B, W - n, Z), latents], dim=1)
        )
        length = torch.full((B,), min(n, W), dtype=torch.long, device=latents.device)
        return LatentDiTState(buf, length)

    def state_from_obs(self, frames: torch.Tensor) -> LatentDiTState:
        """Build the carried state directly from a run of observations."""
        return self.state_from_latents(self.encode(frames))

    def flat_state(self, state: LatentDiTState) -> torch.Tensor:
        if self.state_view == "latent_window":
            return state.latent_buffer.reshape(state.latent_buffer.shape[0], -1)
        self.core.state_view = self.state_view
        return self.core.flat_state(self._to_core(state))

    def state_from_flat(self, flat: torch.Tensor) -> LatentDiTState:
        """(B, W·Z) → state.  Only the invertible latent-window view supports this."""
        if self.state_view != "latent_window":
            raise ValueError(
                "state_from_flat requires state_view='latent_window' "
                f"(active view {self.state_view!r} is read-only)"
            )
        W, Z = self.core.cfg.window, self.latent_dim
        buf = flat.reshape(-1, W, Z)
        length = torch.full((buf.shape[0],), W, dtype=torch.long, device=flat.device)
        return LatentDiTState(buf, length)

    # ── prediction ───────────────────────────────────────────────────────────

    def decode_next_latent(self, state: LatentDiTState) -> torch.Tensor:
        """Predict the next NORMALISED LATENT from the current window."""
        return self.core.decode(self._to_core(state))

    def decode(self, state: LatentDiTState) -> torch.Tensor:
        """Predict the next OBSERVATION (GRU convention: decode(state after obs_t) ≈ obs_{t+1})."""
        return self.decode_latent(self.decode_next_latent(state))

    def advance(self, state: LatentDiTState, z_t: torch.Tensor) -> LatentDiTState:
        """Append one latent to the window."""
        buf = torch.cat([state.latent_buffer[:, 1:], z_t.unsqueeze(1)], dim=1)
        length = torch.clamp(state.length + 1, max=self.core.cfg.window)
        return LatentDiTState(buf, length)

    def step(
        self, obs_t: torch.Tensor, state: LatentDiTState | None = None
    ) -> tuple[torch.Tensor, LatentDiTState]:
        """Observe one frame, predict the next (observation space)."""
        if state is None:
            B = obs_t.shape[0]
            state = LatentDiTState(
                obs_t.new_zeros(B, self.core.cfg.window, self.latent_dim),
                torch.zeros(B, dtype=torch.long, device=obs_t.device),
            )
        state = self.advance(state, self.encode(obs_t))
        return self.decode(state), state

    def step_latent(
        self, z_t: torch.Tensor, state: LatentDiTState
    ) -> tuple[torch.Tensor, LatentDiTState]:
        """Latent-native step: append a latent, return the next predicted latent."""
        state = self.advance(state, z_t)
        return self.decode_next_latent(state), state

    @torch.no_grad()
    def predict_step(
        self, state: LatentDiTState
    ) -> tuple[torch.Tensor, LatentDiTState]:
        """Free-running step in LATENT space (no VAE round-trip between steps).

        Returns the decoded observation prediction for the frame after next, and
        the advanced state.  Keeping the feedback in latent space is the honest
        rollout for this architecture — a decode/encode round-trip each step
        would inject VAE reconstruction error that the model never sees.
        """
        z_hat = self.decode_next_latent(state)
        z_next, state = self.step_latent(z_hat, state)
        return self.decode_latent(z_next), state

    # ── sequence interfaces ──────────────────────────────────────────────────

    @torch.no_grad()
    def observe_sequence(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher forcing: (pred obs (B,T-1,R), flat states (B,T-1,hidden_size))."""
        z = self.encode(obs)
        pred_z, core_flat = self._core_observe(z)
        return self.decode_latent(pred_z), core_flat

    def _core_observe(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.state_view == "latent_window":
            self.core.state_view = "obs_window"  # = the latent window here
        else:
            self.core.state_view = self.state_view
        return self.core.observe_sequence(z)

    @torch.no_grad()
    def get_hidden_states(self, obs: torch.Tensor) -> torch.Tensor:
        """Per-timestep flat states under the active view; h[:, t] follows obs[:, t]."""
        return self.observe_sequence(obs)[1]

    @torch.no_grad()
    def forward(
        self, obs: torch.Tensor, h0: LatentDiTState | None = None
    ) -> tuple[torch.Tensor, LatentDiTState]:
        """Teacher-forcing pass (protocol; eval-only — training uses diffusion_loss)."""
        if h0 is not None:
            raise NotImplementedError("LatentDiTModel.forward does not support h0")
        z = self.encode(obs)
        pred_z, _ = self._core_observe(z)
        return self.decode_latent(pred_z), self.state_from_latents(z[:, :-1])

    # ── training ─────────────────────────────────────────────────────────────

    def diffusion_loss(
        self, obs: torch.Tensor, p_clean: float = 0.3, p_one: float = 0.1
    ) -> torch.Tensor:
        """Flow-matching loss in latent space (the VAE stays frozen)."""
        with torch.no_grad():
            z = self.encode(obs)
        return self.core.diffusion_loss(z, p_clean=p_clean, p_one=p_one)
