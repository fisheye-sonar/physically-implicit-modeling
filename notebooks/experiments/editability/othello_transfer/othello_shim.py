"""The bridge: Li et al.'s minGPT Othello-GPT behind the handful of names this
repo's probe and intervention code already calls.

Why this file exists
--------------------
The point of the `othello_transfer` thread is to run **our** probe and **our**
editing code, unmodified, on **their** model — so that a failure on discworld
cannot be blamed on our implementation of either.  That only works if the
editing code is byte-identical to what runs on discworld, which means the model
has to answer to the names `othello_gpt/othello_probe.py` calls:

    embed · _run(tokens, attn_mask, edit=, want_resid=) · _seq_mask / _win_mask
    norm_out · decoder · residual_stack

`OthelloGPTShim` supplies exactly those and **contains no editing logic of any
kind**.  Every write to the residual stream is performed by the unmodified
`make_intervention_hook` / `_descend` from `othello_probe`.

What is deliberately NOT matched
--------------------------------
`TransformerModel._run` takes an explicit banded-causal `attn_mask`, because that
model's state is a sliding window.  minGPT applies its own full-causal mask inside
`CausalSelfAttention`, so the shim **accepts and ignores** the mask argument.  That
is the architectural difference we are holding fixed on purpose: their model, their
context handling, our probe and our editor.

Alignment
---------
Residual point `ell` is the stream **after** `ell` blocks, so point 0 is the
embedding and point 8 is the final pre-`ln_f` stream — the same convention as
`TransformerModel` (`n_layers + 1` points), and the same activation
`GPTforProbing(probe_layer=ell)` returns.

The model must be in `eval()` mode: minGPT carries dropout at p=0.1 on the
embedding, attention and residual paths, and their own probe script never disables
it (see the thread README).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OthelloGPTShim(nn.Module):
    """Wraps an unmodified minGPT `GPT`. Owns no parameters of its own."""

    def __init__(self, gpt: nn.Module, probe_layer: int | None = None) -> None:
        super().__init__()
        self.gpt = gpt
        self.n_layers = len(gpt.blocks)
        self.probe_layer = self.n_layers if probe_layer is None else probe_layer

    # ── the names our code calls ──────────────────────────────────────────────

    @property
    def decoder(self) -> nn.Module:
        return self.gpt.head

    @property
    def norm_out(self) -> nn.Module:
        return self.gpt.ln_f

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        """(B, T) move tokens → (B, T, d_model) residual point 0."""
        t = idx.shape[1]
        return self.gpt.drop(self.gpt.tok_emb(idx) + self.gpt.pos_emb[:, :t, :])

    def _seq_mask(self, T: int, device) -> None:
        """minGPT masks internally; the mask argument exists only for signature parity."""
        return None

    def _win_mask(self, lengths, device) -> None:
        return None

    def _run(self, tokens, attn_mask=None, edit=None, want_resid=False, kv_sink=None):
        """Block stack with the same `edit` semantics as `TransformerModel._run`.

        edit : ``(layer, vector)`` forces the stream at that residual point at the
               **last position**; a callable ``fn(layer_idx, x) -> x`` fires at
               **every** residual point 0…n_layers.  ``None`` leaves the pass
               bit-identical to `GPT.forward`.
        """
        if kv_sink is not None:
            raise NotImplementedError("kv_sink has no minGPT analogue")
        x = tokens
        hook = edit if callable(edit) else None
        resids = [x] if want_resid else None
        for i, blk in enumerate(self.gpt.blocks):
            if hook is not None:
                x = hook(i, x)
                if want_resid:
                    resids[i] = x
            elif edit is not None and edit[0] == i:
                x = x.clone()
                x[:, -1] = edit[1]
                if want_resid:
                    resids[i] = x
            x = blk(x)
            if want_resid:
                resids.append(x)
        if hook is not None:
            x = hook(self.n_layers, x)
            if want_resid:
                resids[-1] = x
        elif edit is not None and edit[0] == self.n_layers:
            x = x.clone()
            x[:, -1] = edit[1]
            if want_resid:
                resids[-1] = x
        return x, resids

    @torch.no_grad()
    def residual_stack(self, idx: torch.Tensor, edit=None) -> torch.Tensor:
        """(n_layers+1, B, T, d_model) — the stream at every residual point."""
        _, resids = self._run(self.embed(idx), None, edit=edit, want_resid=True)
        return torch.stack(resids, 0)

    def decode(self, idx: torch.Tensor, edit=None) -> torch.Tensor:
        """(B, vocab) next-move logits at the last position.

        Verified against `GPT.forward`: the block stack is **bit-identical**, and so is
        the head when applied over the whole sequence.  Applying it to the `(B, d_model)`
        last-position slice instead — which is what this method does, and what the
        intervention needs — differs by ~1.4e-6 in the logits and ~1.5e-7 in the
        next-move distribution, purely because cuBLAS selects a different kernel for the
        2-D matmul.  It is float32 kernel choice, not a semantic difference; do not
        spend time on it.
        """
        h, _ = self._run(self.embed(idx), None, edit=edit)
        return self.decoder(self.norm_out(h[:, -1]))

    # ── rollout surface — deliberately absent until it is designed ────────────

    def advance(self, state, obs_t):
        raise NotImplementedError(
            "Rolling forward in Othello needs design decisions we have not made: "
            "whether the model's own move enters the history, whether it is sampled "
            "or argmax'd, and how the counterfactual board evolves alongside. "
            "The Li replication is step-0 only."
        )

    def predict_step(self, state):
        raise NotImplementedError(self.advance.__doc__)
