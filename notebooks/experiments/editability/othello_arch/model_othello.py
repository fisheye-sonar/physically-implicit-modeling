"""Li et al.'s minGPT on Othello — **unmodified**, at OUR pilot's training conditions.

The control Sevan asked for on 2026-08-22: *"we don't yet have any example where the same setup
and conditions, the only thing changed being the environment, has editability on Othello but not
discworld."*

What the 2x2 looked like before this run
----------------------------------------
| model | env | data | epochs | editable |
|---|---|---|---|---|
| `A_pilot` — their arch | discworld | 900k eps | 4 | **no** (best EI −0.496) |
| Li et al.'s published checkpoint | Othello | 20M games | many | **yes** (EI −0.829 → +0.697) |

Those two differ in **environment AND data volume AND training length**, so "the environment is
what matters" was not licensed. This run holds architecture, data volume (900k), epochs (4),
batch, optimiser and schedule fixed at `A_pilot`'s values and changes **only the environment**.

- **Editable** → the environment is the cause, and the discworld negative is about discworld.
- **Not editable** → 900k games / 4 epochs is simply too little for editability to appear at all,
  and the `A_pilot` negative says nothing about environment. In that case the honest reading is
  that we still lack the clean comparison.

There are **no substitutions here.** On Othello their architecture is used exactly as published:
`nn.Embedding(61, 512)` in, `Linear(512, 61)` out, cross-entropy. The only wrapper is `logits`,
so the trainer in `ours_on_othello/train.py` can drive it through the same code path as ours.
Evaluation then goes through `othello_transfer/othello_shim.OthelloGPTShim`, which already exists
precisely to put a minGPT behind our probe and editor names.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

OTHELLO_ROOT = Path("/home/sevan/research/PIM/othello_world")
if str(OTHELLO_ROOT) not in sys.path:
    sys.path.insert(0, str(OTHELLO_ROOT))

from mingpt.model import GPT, GPTConfig  # noqa: E402

VOCAB, BLOCK = 61, 59


class OthelloGPTNative(nn.Module):
    """Their `GPT`, untouched, with the one accessor our trainer expects."""

    def __init__(self, vocab: int = VOCAB, block_size: int = BLOCK, n_layer: int = 8,
                 n_head: int = 8, n_embd: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        cfg = GPTConfig(vocab_size=vocab, block_size=block_size, n_layer=n_layer,
                        n_head=n_head, n_embd=n_embd,
                        embd_pdrop=dropout, resid_pdrop=dropout, attn_pdrop=dropout)
        self.gpt = GPT(cfg)
        self.cfg = cfg
        self.vocab = vocab
        self.n_layers = n_layer

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        """(B, T) move tokens -> (B, T, vocab). `GPT.forward` returns (logits, loss)."""
        out = self.gpt(idx)
        return out[0] if isinstance(out, tuple) else out

    @property
    def state_span(self) -> int:
        return self.cfg.block_size


def build(vocab: int = VOCAB, block_size: int = BLOCK, n_layer: int = 8, n_head: int = 8,
          n_embd: int = 512, dropout: float = 0.1, **_) -> OthelloGPTNative:
    """Signature-compatible with `ours_on_othello.model.build` so the trainer can swap them."""
    return OthelloGPTNative(vocab, block_size, n_layer, n_head, n_embd, dropout)
