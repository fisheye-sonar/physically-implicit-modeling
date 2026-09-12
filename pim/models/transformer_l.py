"""Transformer-L — the large (~25M) canonical architecture: Li et al.'s minGPT.

ONE bridge (2026-08-31) replacing the four that used to exist —
``othello_arch/model.py`` (regression), ``othello_arch/model_othello.py`` (tokens),
``othello_transfer/othello_shim.py`` (tokens, wrapping their published checkpoint), and
the ``_run`` copy each carried. The GPT itself comes from the vendored byte-identical
copy in ``pim.environments.othello.vendor`` (that is its provenance, not a claim that
the architecture is Othello-specific).

ONE class, two parameters (2026-09-09; before that, two classes whose input layer and
head were welded together). The body is always **their** ``GPT``, used unmodified: 8
blocks, 8 heads, ``n_embd`` 512, full causal attention, **learned absolute** position
embeddings, dropout 0.1 on the embedding / attention / residual paths, post-block
LayerNorm, their weight init. What a run chooses is the interface at each end:

    input="linear"      float observations (B, T, obs_res) through ``Linear(obs_res, d)``
                        — NO ReLU (pinned by ``directions/othello-architecture-on-discworld.md``),
                        so residual point 0 is a bare affine map of the observation
    input="embedding"   integer ids (B, T) through their ``Embedding(vocab, d)``

    head="regression"   ``Linear(d, obs_res)`` — the next observation, in its own units;
                        the model EMITS A FRAME and carries the rollout surface below
    head="categorical"  their ``Linear(d, vocab, bias=False)`` — a distribution over the
                        vocabulary; read per ``output_kind``: "logits" (trained with
                        cross-entropy; softmax to read) or "raw" (trained with MSE against
                        the one-hot next token, 2026-09-04: the outputs ARE the estimates)

The two canonical combinations keep their names as presets and their registry entries:
``TransformerL`` (linear + regression — discworld frames) and ``TransformerLTokens``
(embedding + categorical — Othello moves, and discworld frames-as-tokens). Loads their
published checkpoint as well as every retrained one.

⛔ Parameter names and registration order are load-bearing: the probe cache fingerprints
a model by its state dict, so ``encoder.*`` / ``decoder.*`` (regression) and
``gpt.tok_emb.*`` / ``gpt.head.*`` (categorical) must stay exactly where they are. The
2026-09-09 unification was gated on every canonical checkpoint: identical fingerprints,
identical state-dict keys, bit-identical residual streams and head outputs.

Models must be in ``eval()`` for analysis: minGPT carries live dropout otherwise.

Residual-point convention (same as Transformer-S and everywhere else): point ``ell`` is
the stream **after** ``ell`` blocks — point 0 the embedding, point ``n_layers`` the
final pre-``ln_f`` stream — ``n_layers + 1`` points in all.

State alignment note: the regression head's carried state is the observation prefix,
**left-aligned**. Transformer-S needs a right-aligned sliding buffer because its
attention is banded and its positions are RoPE (relative); minGPT is full-causal with
learned **absolute** positions, so a frame's index is semantically load-bearing and
right-aligning a partly-filled buffer would silently read the wrong position embedding.
The categorical head has no rollout: rolling forward in a vocabulary world needs design
decisions not yet made (does the model's own move enter the history? sampled or argmax?
how does the counterfactual board evolve alongside?) — every such measurement is step-0.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from pim.environments.othello.vendor.mingpt_model import GPT, GPTConfig
from pim.models.protocol import free_run

INPUTS = ("linear", "embedding")
HEADS = ("regression", "categorical")
OUTPUT_KINDS = ("logits", "raw")          # categorical head only; see data.move_probs

_NO_ROLLOUT = ("Rolling forward with a categorical head needs design decisions not yet made "
               "(does the model's own move enter the history? sampled or argmax? how does the "
               "counterfactual board evolve alongside?). Every such measurement is step-0.")


class ArchState(NamedTuple):
    """Everything a frame-emitting TransformerL carries between observations: the
    left-aligned history."""

    obs: torch.Tensor  # (B, T, obs_res), T <= block_size


class TransformerL(nn.Module):
    """Their minGPT behind this repo's probe/editor names, with a chosen input layer and head.

    ``TransformerL(obs_res=…, block_size=…)`` is the regression preset (linear in,
    regression out); ``TransformerL(vocab=…, block_size=…)`` the categorical one (embedding
    in, categorical out) — ``input`` / ``head`` default from which size was given, and can
    be set explicitly for any other combination. (Transformer-S and Recurrent-L carry their
    own ``_run`` with the same edit contract; the three are deliberately separate block
    stacks that share a protocol, not code — see ``pim/models/protocol.py``.)
    """

    def __init__(self, obs_res: int | None = None, block_size: int = 39, n_layer: int = 8,
                 n_head: int = 8, n_embd: int = 512, dropout: float = 0.1, *,
                 vocab: int | None = None, input: str | None = None, head: str | None = None,
                 output_kind: str | None = None) -> None:
        super().__init__()
        if input is None:
            input = "embedding" if (vocab is not None and obs_res is None) else "linear"
        if head is None:
            head = "categorical" if (vocab is not None and obs_res is None) else "regression"
        if input not in INPUTS or head not in HEADS:
            raise ValueError(f"input must be one of {INPUTS} and head one of {HEADS}, "
                             f"got {input!r} / {head!r}")
        if (input == "linear" or head == "regression") and obs_res is None:
            raise ValueError("obs_res is required for a linear input or a regression head")
        if (input == "embedding" or head == "categorical") and vocab is None:
            raise ValueError("vocab is required for an embedding input or a categorical head")
        if head == "categorical":
            output_kind = output_kind or "logits"
            if output_kind not in OUTPUT_KINDS:
                raise ValueError(f"output_kind must be one of {OUTPUT_KINDS}, got {output_kind!r}")
        elif output_kind is not None:
            raise ValueError("output_kind applies to the categorical head only")
        self.input, self.head, self.output_kind = input, head, output_kind
        self.obs_res, self.vocab = obs_res, vocab

        # Their GPT, sized for the vocabulary it carries (1 when neither end uses it);
        # registered FIRST so gpt.* keys precede encoder/decoder as they always have.
        cfg = GPTConfig(vocab_size=vocab if vocab is not None else 1, block_size=block_size,
                        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                        embd_pdrop=dropout, resid_pdrop=dropout, attn_pdrop=dropout)
        self.gpt = GPT(cfg)
        self.cfg = cfg
        self.n_layers = n_layer
        self.probe_layer = n_layer
        # The interfaces: their layer where it is used, replaced (never wrapped) where not,
        # so a state dict names exactly the parameters that exist.
        if input == "linear":
            self.gpt.tok_emb = nn.Identity()          # unused — `embed` bypasses it
            self.encoder = nn.Linear(obs_res, n_embd)
        if head == "regression":
            self.gpt.head = nn.Identity()             # unused — `decoder` replaces it
            # registered under the name the `decoder` property serves (state-dict key
            # `decoder.*`, as always); `add_module` would refuse because the property exists
            self._modules["decoder"] = nn.Linear(n_embd, obs_res)

    # ── what the model is ─────────────────────────────────────────────────────

    @property
    def emits(self) -> str:
        """``"frame"`` (regression head: the next observation, roll-out-able) or
        ``"distribution"`` (categorical head: a vector over the vocabulary, step-0 only)."""
        return "frame" if self.head == "regression" else "distribution"

    # ── the names the probe/editor suite calls ────────────────────────────────

    @property
    def norm_out(self) -> nn.Module:
        return self.gpt.ln_f

    @property
    def decoder(self) -> nn.Module:
        """The head: our ``Linear(d, obs_res)`` (registered as ``decoder``) or their
        ``gpt.head``. An editor reads ``decoder(norm_out(h))`` itself."""
        return self._modules["decoder"] if self.head == "regression" else self.gpt.head

    def _seq_mask(self, T: int, device) -> None:
        """minGPT masks internally; the argument exists only for signature parity."""
        return None

    def _win_mask(self, lengths, device) -> None:
        return None

    def embed(self, inp: torch.Tensor) -> torch.Tensor:
        """Residual point 0: ``(B, T, obs_res)`` floats through the linear encoder, or
        ``(B, T)`` ids through their embedding — then their learned absolute position
        embedding and embedding dropout, exactly as ``GPT.forward`` does. The
        substitution is the projection, not the surrounding machinery."""
        t = inp.shape[1]
        x = self.encoder(inp) if self.input == "linear" else self.gpt.tok_emb(inp)
        return self.gpt.drop(x + self.gpt.pos_emb[:, :t, :])

    def _run(self, tokens, attn_mask=None, edit=None, want_resid=False):
        """Block stack with the same ``edit`` semantics as ``TransformerS._run``.

        edit : ``(layer, vector)`` forces the stream at that residual point at the
               **last position**; a callable ``fn(layer_idx, x) -> x`` fires at
               **every** residual point 0…n_layers. ``None`` leaves the pass
               bit-identical to ``GPT.forward``.
        """
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
        """Positions carried. Full causal attention ⇒ the whole history, capped at block_size."""
        return self.cfg.block_size

    # ── the two views of the head ─────────────────────────────────────────────

    def forward(self, inp: torch.Tensor, edit=None) -> torch.Tensor:
        """The training view: the head at EVERY position — ``(B, T, obs_res)`` predicted
        next observations, or ``(B, T, vocab)`` next-token logits / raw estimates."""
        h, _ = self._run(self.embed(inp), edit=edit)
        return self.decoder(self.norm_out(h))

    def logits(self, inp: torch.Tensor, edit=None) -> torch.Tensor:
        """The training view under its categorical name (``ce_next_move`` calls this). With
        ``edit=None`` bit-identical to ``GPT.forward`` (gated at all nine residual points)."""
        return self.forward(inp, edit=edit)

    def decode(self, state_or_input, edit=None) -> torch.Tensor:
        """The intervention view: the head at the LAST position — ``(B, obs_res)`` or
        ``(B, vocab)``. Accepts a carried ``ArchState`` or the raw input tensor.

        Applying the head to the last-position slice differs from the full-sequence
        forward by ~1.4e-6 in the logits purely from cuBLAS kernel choice for the 2-D
        matmul; it is not a semantic difference.
        """
        inp = state_or_input.obs if isinstance(state_or_input, ArchState) else state_or_input
        h, _ = self._run(self.embed(inp), edit=edit)
        return self.decoder(self.norm_out(h[:, -1]))

    # ── the carried-state surface (frame-emitting head only) ──────────────────

    def _needs_frames(self) -> None:
        if self.emits != "frame":
            raise NotImplementedError(_NO_ROLLOUT)

    def state_from_obs(self, frames: torch.Tensor) -> ArchState:
        """(B, T, obs_res) observed so far → the carried state."""
        self._needs_frames()
        return ArchState(frames[:, -self.state_span :].contiguous())

    def advance(self, state: ArchState, obs_t: torch.Tensor) -> ArchState:
        self._needs_frames()
        buf = torch.cat([state.obs, obs_t[:, None, :]], dim=1)
        return ArchState(buf[:, -self.state_span :])

    def flat_state(self, state: ArchState) -> torch.Tensor:
        """(B, n_embd) residual stream at ``probe_layer``, current position."""
        self._needs_frames()
        _, resids = self._run(self.embed(state.obs), want_resid=True)
        return resids[self.probe_layer][:, -1]

    def decode_with_edit(self, state, layer: int, resid: torch.Tensor) -> torch.Tensor:
        return self.decode(state, edit=(layer, resid))

    def predict_step(self, state: ArchState):
        self._needs_frames()
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
        self._needs_frames()
        pred = self.decode_with_edit(state, layer, resid)
        return free_run(self, pred, self.advance(state, pred), steps)


class TransformerLTokens(TransformerL):
    """Preset: their ``GPT`` untouched — embedding in, categorical out (Othello moves;
    discworld frames-as-tokens). Kept as a name so checkpoints, tests and call sites read
    as before; everything is ``TransformerL(input="embedding", head="categorical")``."""

    def __init__(self, vocab: int = 61, block_size: int = 59, n_layer: int = 8,
                 n_head: int = 8, n_embd: int = 512, dropout: float = 0.1,
                 output_kind: str = "logits") -> None:
        super().__init__(vocab=vocab, block_size=block_size, n_layer=n_layer, n_head=n_head,
                         n_embd=n_embd, dropout=dropout, input="embedding",
                         head="categorical", output_kind=output_kind)
