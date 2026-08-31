"""Transformer-L — the large (~25M) canonical architecture: Li et al.'s minGPT.

ONE bridge (2026-08-31) replacing the four that used to exist —
``othello_arch/model.py`` (regression), ``othello_arch/model_othello.py`` (tokens),
``othello_transfer/othello_shim.py`` (tokens, wrapping their published checkpoint), and
the ``_run`` copy each carried. The GPT itself comes from the vendored byte-identical
copy in ``pim.environments.othello.vendor`` (that is its provenance, not a claim that
the architecture is Othello-specific).

Two task heads share one body:

  * ``TransformerL`` — regression: the brief's substitutions, exactly —

        tok_emb  nn.Embedding(61, 512)      ->  nn.Linear(obs_res, 512)
        head     nn.Linear(512, 61, b=F)    ->  nn.Linear(512, obs_res)
        loss     cross-entropy              ->  MSE on the next observation

    **No ReLU** after the input projection (pinned by
    ``directions/othello-architecture-on-discworld.md``), so residual point 0 is a bare
    affine map of the observation. ``block_size`` 39 for dataset-4's 40-frame episodes.

  * ``TransformerLTokens`` — their GPT untouched: ``Embedding(61, 512)`` in,
    ``Linear(512, 61)`` out, cross-entropy. Loads their published checkpoint as well as
    our retrained ones.

Everything else is **their** ``GPT``, used unmodified: 8 blocks, 8 heads, ``n_embd``
512, full causal attention, **learned absolute** position embeddings, dropout 0.1 on
the embedding / attention / residual paths, post-block LayerNorm, their weight init.
Models must be in ``eval()`` for analysis: minGPT carries live dropout otherwise.

Residual-point convention (same as Transformer-S and everywhere else): point ``ell`` is
the stream **after** ``ell`` blocks — point 0 the embedding, point ``n_layers`` the
final pre-``ln_f`` stream — ``n_layers + 1`` points in all.

State alignment note: ``TransformerL``'s carried state is the observation prefix,
**left-aligned**. Transformer-S needs a right-aligned sliding buffer because its
attention is banded and its positions are RoPE (relative); minGPT is full-causal with
learned **absolute** positions, so a frame's index is semantically load-bearing and
right-aligning a partly-filled buffer would silently read the wrong position embedding.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from pim.environments.othello.vendor.mingpt_model import GPT, GPTConfig


class ArchState(NamedTuple):
    """Everything TransformerL carries between observations: the left-aligned history."""

    obs: torch.Tensor  # (B, T, obs_res), T <= block_size


class _MinGPTCore(nn.Module):
    """The shared body: their block stack behind this repo's probe/editor names.

    Subclasses supply ``embed`` (residual point 0) and ``decoder``; everything an editor
    or probe calls — ``_run``'s edit hook, ``residual_stack``, ``norm_out`` — lives here
    exactly once.
    """

    def __init__(self, vocab_size: int, block_size: int, n_layer: int = 8,
                 n_head: int = 8, n_embd: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer,
                        n_head=n_head, n_embd=n_embd,
                        embd_pdrop=dropout, resid_pdrop=dropout, attn_pdrop=dropout)
        self.gpt = GPT(cfg)
        self.cfg = cfg
        self.n_layers = n_layer
        self.probe_layer = n_layer

    # ── the names the probe/editor suite calls ────────────────────────────────

    @property
    def norm_out(self) -> nn.Module:
        return self.gpt.ln_f

    def _seq_mask(self, T: int, device) -> None:
        """minGPT masks internally; the argument exists only for signature parity."""
        return None

    def _win_mask(self, lengths, device) -> None:
        return None

    def _run(self, tokens, attn_mask=None, edit=None, want_resid=False, kv_sink=None):
        """Block stack with the same ``edit`` semantics as ``TransformerS._run``.

        edit : ``(layer, vector)`` forces the stream at that residual point at the
               **last position**; a callable ``fn(layer_idx, x) -> x`` fires at
               **every** residual point 0…n_layers. ``None`` leaves the pass
               bit-identical to ``GPT.forward``.
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
    def residual_stack(self, inp: torch.Tensor, edit=None) -> torch.Tensor:
        """(n_layers+1, B, T, n_embd) — the stream at every residual point."""
        _, resids = self._run(self.embed(inp), edit=edit, want_resid=True)
        return torch.stack(resids, 0)

    @property
    def state_span(self) -> int:
        """Frames carried. Full causal attention ⇒ the whole history, capped at block_size."""
        return self.cfg.block_size


class TransformerL(_MinGPTCore):
    """Their minGPT with a continuous encoder in and a continuous decoder out (discworld)."""

    def __init__(self, obs_res: int = 128, block_size: int = 39, n_layer: int = 8,
                 n_head: int = 8, n_embd: int = 512, dropout: float = 0.1) -> None:
        super().__init__(vocab_size=1, block_size=block_size, n_layer=n_layer,
                         n_head=n_head, n_embd=n_embd, dropout=dropout)
        self.obs_res = obs_res
        # the two substitutions; `tok_emb` and `head` are replaced, never wrapped
        self.gpt.tok_emb = nn.Identity()          # unused — `embed` bypasses it
        self.encoder = nn.Linear(obs_res, n_embd)
        self.gpt.head = nn.Identity()             # unused — `decoder` replaces it
        self.decoder = nn.Linear(n_embd, obs_res)

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, T, obs_res) -> (B, T, n_embd). Residual point 0.

        Their learned absolute position embedding is added and their embedding dropout
        applied, exactly as ``GPT.forward`` does — the substitution is the projection,
        not the surrounding machinery. No ReLU (see module docstring).
        """
        t = obs.shape[1]
        return self.gpt.drop(self.encoder(obs) + self.gpt.pos_emb[:, :t, :])

    def forward(self, obs: torch.Tensor, edit=None) -> torch.Tensor:
        """(B, T, obs_res) -> (B, T, obs_res): the predicted NEXT observation everywhere."""
        h, _ = self._run(self.embed(obs), edit=edit)
        return self.decoder(self.norm_out(h))

    # ── the carried-state surface (what editors and rollouts drive) ───────────

    def state_from_obs(self, frames: torch.Tensor) -> ArchState:
        """(B, T, obs_res) observed so far → the carried state."""
        return ArchState(frames[:, -self.state_span :].contiguous())

    def advance(self, state: ArchState, obs_t: torch.Tensor) -> ArchState:
        buf = torch.cat([state.obs, obs_t[:, None, :]], dim=1)
        return ArchState(buf[:, -self.state_span :])

    def flat_state(self, state: ArchState) -> torch.Tensor:
        """(B, n_embd) residual stream at ``probe_layer``, current position."""
        _, resids = self._run(self.embed(state.obs), want_resid=True)
        return resids[self.probe_layer][:, -1]

    def decode(self, state_or_obs, edit=None) -> torch.Tensor:
        """(B, obs_res) prediction at the last position. Accepts a state or raw (B,T,R)."""
        obs = state_or_obs.obs if isinstance(state_or_obs, ArchState) else state_or_obs
        h, _ = self._run(self.embed(obs), edit=edit)
        return self.decoder(self.norm_out(h[:, -1]))

    def decode_with_edit(self, state, layer: int, resid: torch.Tensor) -> torch.Tensor:
        return self.decode(state, edit=(layer, resid))

    def predict_step(self, state: ArchState):
        pred = self.decode(state)
        return pred, self.advance(state, pred)

    @torch.no_grad()
    def rollout_with_edit(self, state: ArchState, layer: int, resid: torch.Tensor, steps: int):
        """Free-run whose FIRST step is produced under an activation edit.

        Identical contract to ``TransformerS.rollout_with_edit``: the edit shapes the
        immediate prediction, that prediction enters the history, and every later step
        is recomputed with no edit applied — so any persistence has to travel through
        the observations.
        """
        pred = self.decode_with_edit(state, layer, resid)
        out = [pred]
        s = self.advance(state, pred)
        for _ in range(steps - 1):
            p, s = self.predict_step(s)
            out.append(p)
        return torch.stack(out, 1)


class TransformerLTokens(_MinGPTCore):
    """Their ``GPT``, untouched, behind the shared surface (Othello)."""

    def __init__(self, vocab: int = 61, block_size: int = 59, n_layer: int = 8,
                 n_head: int = 8, n_embd: int = 512, dropout: float = 0.1) -> None:
        super().__init__(vocab_size=vocab, block_size=block_size, n_layer=n_layer,
                         n_head=n_head, n_embd=n_embd, dropout=dropout)
        self.vocab = vocab

    @property
    def decoder(self) -> nn.Module:
        return self.gpt.head

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        """(B, T) move tokens → (B, T, n_embd) residual point 0."""
        t = idx.shape[1]
        return self.gpt.drop(self.gpt.tok_emb(idx) + self.gpt.pos_emb[:, :t, :])

    def logits(self, idx: torch.Tensor, edit=None) -> torch.Tensor:
        """(B, T, vocab) next-move logits at every position — the training view.

        With ``edit=None`` this is bit-identical to ``GPT.forward`` (gated in the old
        shim's notebook at all nine residual points).
        """
        h, _ = self._run(self.embed(idx), edit=edit)
        return self.decoder(self.norm_out(h))

    def decode(self, idx: torch.Tensor, edit=None) -> torch.Tensor:
        """(B, vocab) next-move logits at the last position — the intervention view.

        Applying the head to the last-position slice differs from the full-sequence
        forward by ~1.4e-6 in the logits purely from cuBLAS kernel choice for the 2-D
        matmul; it is not a semantic difference.
        """
        h, _ = self._run(self.embed(idx), edit=edit)
        return self.decoder(self.norm_out(h[:, -1]))

    # ── rollout surface — deliberately absent until it is designed ────────────

    def advance(self, state, obs_t):
        raise NotImplementedError(
            "Rolling forward in Othello needs design decisions not yet made (does the "
            "model's own move enter the history? sampled or argmax? how does the "
            "counterfactual board evolve alongside?). Every Othello measurement is step-0."
        )

    def predict_step(self, state):
        raise NotImplementedError(self.advance.__doc__)
