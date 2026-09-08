"""A frame-facing view of a frames-as-tokens model, for the canonical discworld bench.

The bench (`pim.environments.discworld.bench`, arms in `.arms`) drives a model through the carried-state
surface of `pim/models/protocol.py`: `state_from_obs`, `decode(edit=)`, `advance`,
`predict_step`, `flat_state`, `rollout_with_edit`. A token model speaks tokens; this wrapper
speaks frames on the outside and tokens on the inside, so every discworld number — the
ray-zone Edit Index, zone RMSEs, fidelity against clean_obs, K-step rollouts, waterfalls —
can be read on the token run with no change to the canonical code.

Two choices (the user's, 2026-09-06):
  render   = "expected"  the reported frame is Σ p_k · frame_k over the real frames (UNK mass
                         renormalised away) — a continuous 8-vector like a regression output;
             "argmax"    the single most probable frame.
  feedback = "argmax"    the free-run feeds the ARGMAX token back (the expected frame is not a
                         token); "sample" draws one.
The edit hook shapes one prediction and later steps are recomputed unedited, exactly as for
the regression transformer (`TransformerL.rollout_with_edit`).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from torch import nn

from pim.environments.discworld.tokens import UNK, FrameVocab, encode


class TokState(NamedTuple):
    tok: torch.Tensor          # (B, t) long — the token window the model will read


class TokenFrameAdapter(nn.Module):
    def __init__(self, model, vocab: FrameVocab, render: str = "expected", feedback: str = "argmax",
                 seed: int = 0):
        super().__init__()
        self.m = model
        self.vocab = vocab
        dev = next(model.parameters()).device
        self.frames = torch.from_numpy(vocab.frames[1:]).float().to(dev)   # (V-1, R)
        self.state_span = int(model.state_span)
        self.n_layers = int(model.n_layers)
        self.probe_layer = self.n_layers
        self.render, self.feedback = render, feedback
        self.gen = torch.Generator(device=dev).manual_seed(seed)
        self.unk_inputs = 0
        self._last_tok = None

    # ── frames ↔ tokens ─────────────────────────────────────────────────────
    def tokenize(self, frames: torch.Tensor) -> torch.Tensor:
        ids = encode(frames.detach().cpu().numpy(), self.vocab)
        self.unk_inputs += int((ids == UNK).sum())
        return torch.from_numpy(ids.astype(np.int64)).to(frames.device)

    def _render(self, logits: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(logits.float(), -1)[:, 1:]
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
        if self.feedback == "sample":
            nxt = torch.multinomial(p, 1, generator=self.gen)[:, 0]
        else:
            nxt = p.argmax(-1)
        self._last_tok = nxt + 1                                   # token id (UNK = 0)
        if self.render == "argmax":
            return self.frames[p.argmax(-1)]
        return p @ self.frames

    # ── the carried-state surface ───────────────────────────────────────────
    def state_from_obs(self, frames: torch.Tensor) -> TokState:
        return TokState(self.tokenize(frames)[:, -self.state_span:].contiguous())

    def _tok(self, state_or_obs) -> torch.Tensor:
        if isinstance(state_or_obs, TokState):
            return state_or_obs.tok
        if state_or_obs.dtype in (torch.int64, torch.int32, torch.int16):
            return state_or_obs.long()
        return self.tokenize(state_or_obs)

    def decode(self, state_or_obs, edit=None) -> torch.Tensor:
        return self._render(self.m.decode(self._tok(state_or_obs), edit=edit))

    def decode_with_edit(self, state, layer: int, resid: torch.Tensor) -> torch.Tensor:
        return self.decode(state, edit=(layer, resid))

    def advance(self, state: TokState, pred: torch.Tensor) -> TokState:
        assert self._last_tok is not None and len(self._last_tok) == len(pred), "advance() follows decode()"
        buf = torch.cat([state.tok, self._last_tok[:, None]], 1)
        return TokState(buf[:, -self.state_span:])

    def predict_step(self, state: TokState):
        pred = self.decode(state)
        return pred, self.advance(state, pred)

    def flat_state(self, state: TokState) -> torch.Tensor:
        rs = self.m.residual_stack(state.tok)
        return rs[self.probe_layer][:, -1]

    def residual_stack(self, x, edit=None):
        return self.m.residual_stack(self._tok(x), edit=edit) if edit is not None else self.m.residual_stack(self._tok(x))

    @torch.no_grad()
    def rollout_with_edit(self, state: TokState, layer: int, resid: torch.Tensor, steps: int):
        pred = self.decode_with_edit(state, layer, resid)
        out, s = [pred], self.advance(state, pred)
        for _ in range(steps - 1):
            p, s = self.predict_step(s)
            out.append(p)
        return torch.stack(out, 1)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.m, name)
