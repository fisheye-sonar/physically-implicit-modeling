"""Transformer building blocks for the DiT world model.

Standard DiT components (Peebles & Xie, 2023) adapted to *causal sequence*
modelling with *per-token* diffusion times:

  * timestep_embedding / TimestepEmbedder — sinusoidal features of the
    diffusion time τ ∈ [0, 1], mapped to a conditioning vector.
  * RotaryEmbedding — relative positions via RoPE on q/k.  Relative (not
    absolute) positions matter here: at inference the model sees a sliding
    window of the last W frames, so token identities shift every step and
    only query–key *offsets* are stable between training and deployment.
  * DiTBlock — pre-norm transformer block with AdaLN-Zero conditioning.
    Conditioning is per-token (B, T, d), not per-sample: diffusion forcing
    gives every sequence position its own τ (past = clean τ=0, current =
    noised), so each token needs its own modulation.
  * FinalLayer — AdaLN-modulated linear projection back to observation space.

All blocks are shape-transparent: (B, T, d) in, (B, T, d) out.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Timestep conditioning ─────────────────────────────────────────────────────


def timestep_embedding(
    tau: torch.Tensor, dim: int, max_period: float = 10_000.0
) -> torch.Tensor:
    """Sinusoidal features of the diffusion time τ ∈ [0, 1].

    τ is scaled by 1000 so the sinusoid frequencies cover the same range as
    the integer-timestep embeddings used by standard diffusion models.

    Parameters
    ----------
    tau : (...,) diffusion times in [0, 1]
    dim : embedding dimension (must be even)

    Returns
    -------
    emb : (..., dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=tau.device)
        / half
    )
    args = tau.float().unsqueeze(-1) * 1000.0 * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal τ features → MLP → conditioning vector (per token)."""

    def __init__(self, d_model: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """(..., ) τ values → (..., d_model) conditioning vectors."""
        return self.mlp(timestep_embedding(tau, self.freq_dim))


# ── Rotary position embedding ─────────────────────────────────────────────────


class RotaryEmbedding(nn.Module):
    """RoPE cos/sin tables for a given head dimension."""

    def __init__(self, head_dim: int, theta: float = 10_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, n_pos: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) tables of shape (n_pos, head_dim // 2)."""
        pos = torch.arange(n_pos, dtype=torch.float32, device=device)
        angles = pos[:, None] * self.inv_freq[None, :].to(device)
        return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate query/key features by position-dependent angles.

    Parameters
    ----------
    x        : (B, n_heads, T, head_dim)
    cos, sin : (T, head_dim // 2)
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)


# ── Attention masks ───────────────────────────────────────────────────────────


def band_causal_mask(n_pos: int, window: int, device: torch.device) -> torch.Tensor:
    """Boolean (n_pos, n_pos) mask: attend to self and the window-1 tokens before.

    True = may attend.  Row i attends columns j with  i-window < j <= i.
    This is the training-time equivalent of the inference-time sliding
    window: a fixed-size state carrying only the last `window` frames.
    """
    i = torch.arange(n_pos, device=device)[:, None]
    j = torch.arange(n_pos, device=device)[None, :]
    return (j <= i) & (j > i - window)


# ── Transformer blocks ────────────────────────────────────────────────────────


def _modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE and an arbitrary boolean mask."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        kv_sink: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x         : (B, T, d_model)
        attn_mask : bool, broadcastable to (B, n_heads, T, T); True = attend
        rope      : (cos, sin) tables of shape (T, head_dim // 2)
        kv_sink   : if given, the post-RoPE (k, v) tensors are appended —
                    this is what a real KV cache would store, exposed for
                    the "kv_cache" state view.
        """
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (B, T, d) → (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = rope
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if kv_sink is not None:
            kv_sink.append((k, v))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.proj(out)


class DiTBlock(nn.Module):
    """Pre-norm transformer block with per-token AdaLN-Zero conditioning.

    The AdaLN projection is zero-initialised so every block starts as the
    identity function — the standard DiT trick for stable early training.
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        kv_sink: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, d_model) token features
        c : (B, T, d_model) per-token conditioning (from TimestepEmbedder)
        """
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=-1)
        x = x + gate1 * self.attn(
            _modulate(self.norm1(x), shift1, scale1), attn_mask, rope, kv_sink
        )
        x = x + gate2 * self.mlp(_modulate(self.norm2(x), shift2, scale2))
        return x


class FinalLayer(nn.Module):
    """AdaLN-modulated projection back to observation space (zero-initialised)."""

    def __init__(self, d_model: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.proj = nn.Linear(d_model, out_dim)
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return self.proj(_modulate(self.norm(x), shift, scale))
