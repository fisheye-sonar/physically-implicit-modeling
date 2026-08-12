"""Per-frame VAE over 1D observation scans — the latent space the latent DiT runs in.

Scope and design
----------------
This is a **per-frame** autoencoder: one 128-ray observation in, one continuous
vector latent out.  It carries no temporal information whatsoever — all dynamics
live in the world model that consumes the latents (`pim.world_models.latent_dit`).
That separation is deliberate: it keeps "what a frame looks like" (VAE) apart from
"what happens next" (world model), so an editability result in latent space is a
statement about the world model, not about a sequence autoencoder.

Following the latent-diffusion recipe, the KL term is weighted very low
(`kl_weight` ~1e-6): the encoder is effectively deterministic and the latent is a
compressed *code*, not a strongly regularised probabilistic latent.  There is no
perceptual loss and no adversarial term — at 128 rays plain MSE is the right
objective, and the extra machinery of image LDM autoencoders would buy nothing.

Reconstruction target
---------------------
Trained on the **noisy** observations the world models consume (`obs_intensity`),
because the latent space must represent what the model actually sees.  A tight
latent nevertheless *partially denoises* — that is expected and is why
`scripts/train_vae.py` reports reconstruction RMSE against BOTH the noisy input
(what it was trained on) and the clean render (what the simulator drew).  Quote
both; the clean-target number is the one that says whether the code kept the
world state.

Normalisation for the diffusion model
-------------------------------------
Latent scale is a free parameter of the autoencoder, and a rectified-flow model
needs its data at roughly unit scale to balance the two ODE endpoints.  The VAE
therefore carries `latent_scale` (the LDM "scale factor"): `encode_normalized`
divides by it and `decode_normalized` multiplies back.  It is *measured* from the
training set after fitting (`fit_latent_scale`) and stored in the checkpoint, so
the downstream model never has to re-derive it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn


class VAEOutput(NamedTuple):
    """Forward pass results (training-time)."""

    recon: torch.Tensor  # (..., R)
    mu: torch.Tensor  # (..., z)
    logvar: torch.Tensor  # (..., z)
    z: torch.Tensor  # (..., z) reparameterised sample


@dataclass
class VAEConfig:
    input_dim: int = 128  # obs_res
    latent_dim: int = 16  # z
    hidden: int = 256  # width of each MLP hidden layer
    n_layers: int = 2  # hidden layers per side (encoder / decoder)
    kl_weight: float = 1e-6  # LDM-style: near-deterministic encoder
    latent_scale: float = 1.0  # measured post-training by fit_latent_scale


def _mlp(in_dim: int, hidden: int, n_layers: int, out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.SiLU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class ObsVAE(nn.Module):
    """Per-frame VAE: (..., R) observation ↔ (..., z) continuous latent.

    All methods are shape-transparent over leading batch dimensions, so a
    ``(B, T, R)`` sequence encodes to ``(B, T, z)`` without reshaping.

    Parameters
    ----------
    cfg : VAEConfig
    """

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.enc = _mlp(cfg.input_dim, cfg.hidden, cfg.n_layers, 2 * cfg.latent_dim)
        self.dec = _mlp(cfg.latent_dim, cfg.hidden, cfg.n_layers, cfg.input_dim)

    # ── core ──────────────────────────────────────────────────────────────────

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(..., R) → (mu, logvar), each (..., z)."""
        mu, logvar = self.enc(obs).chunk(2, dim=-1)
        return mu, logvar.clamp(-30.0, 20.0)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(..., z) → (..., R) reconstruction, clamped to the observation range."""
        return self.dec(z).clamp(0.0, 1.0)

    def forward(self, obs: torch.Tensor, sample: bool = True) -> VAEOutput:
        mu, logvar = self.encode(obs)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
        return VAEOutput(self.decode(z), mu, logvar, z)

    def loss(self, obs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Reconstruction MSE + `kl_weight` · KL.  Returns (loss, parts)."""
        out = self.forward(obs, sample=True)
        recon = ((out.recon - obs) ** 2).mean()
        kl = -0.5 * (1 + out.logvar - out.mu.pow(2) - out.logvar.exp()).sum(-1).mean()
        return recon + self.cfg.kl_weight * kl, {
            "recon_mse": float(recon.detach()),
            "kl": float(kl.detach()),
        }

    # ── deterministic interface used by the world model ───────────────────────

    @torch.no_grad()
    def encode_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Posterior mean — the deterministic code the world model consumes."""
        return self.encode(obs)[0]

    def encode_normalized(self, obs: torch.Tensor) -> torch.Tensor:
        """Posterior mean divided by `latent_scale` (≈ unit-scale for the flow).

        Differentiable (no `no_grad`): gradient-based editors backprop through it.
        """
        return self.encode(obs)[0] / self.cfg.latent_scale

    def decode_normalized(self, z_norm: torch.Tensor) -> torch.Tensor:
        """Inverse of `encode_normalized` on the decode side."""
        return self.decode(z_norm * self.cfg.latent_scale)


@torch.no_grad()
def fit_latent_scale(model: ObsVAE, obs: torch.Tensor, chunk: int = 65536) -> float:
    """Measure the std of the posterior mean over a data sample.

    Stored as `cfg.latent_scale` so `encode_normalized` yields ≈ unit-scale
    latents — the rectified-flow interpolant needs both endpoints at comparable
    scale, and this is the LDM "scale factor" by another name.

    Parameters
    ----------
    obs : (N, R) flattened observation frames
    """
    flat = obs.reshape(-1, obs.shape[-1])
    parts = [
        model.encode(flat[i : i + chunk])[0] for i in range(0, flat.shape[0], chunk)
    ]
    return float(torch.cat(parts).std())
