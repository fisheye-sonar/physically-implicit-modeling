"""Attention building blocks shared by the canonical architectures.

Extracted verbatim 2026-08-31 from ``pim/world_models/dit/blocks.py`` (dropping the
DiT-only conditioning pieces), because Transformer-S depends on these and the DiT tree
is retired. Nothing here was changed — checkpoints trained through the old module load
through this one unchanged.

  * RotaryEmbedding / apply_rope — relative positions via RoPE on q/k. Relative (not
    absolute) positions matter for Transformer-S: at inference it sees a sliding window
    of the last W frames, so token identities shift every step and only query–key
    *offsets* are stable between training and deployment.
  * band_causal_mask — the training-time equivalent of the inference-time sliding
    window: a fixed-size state carrying only the last ``window`` frames.
  * CausalSelfAttention — multi-head attention taking an arbitrary boolean mask, with
    an optional ``kv_sink`` exposing the post-RoPE (k, v) a real KV cache would store.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def band_causal_mask(n_pos: int, window: int, device: torch.device) -> torch.Tensor:
    """Boolean (n_pos, n_pos) mask: attend to self and the window-1 tokens before.

    True = may attend. Row i attends columns j with  i-window < j <= i.
    """
    i = torch.arange(n_pos, device=device)[:, None]
    j = torch.arange(n_pos, device=device)[None, :]
    return (j <= i) & (j > i - window)


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
        kv_sink   : if given, the post-RoPE (k, v) tensors are appended.
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
