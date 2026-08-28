"""Li et al.'s minGPT, adapted to the discworld task. The port for `run A`.

⚠ **This is the run-A *architecture*, not the full run-A brief.** See
`directions/othello-architecture-on-discworld.md`. That brief also specifies a **25M-episode**
corpus, which needs a streaming dataloader that does not exist (25M episodes is 512 GB of
`obs_intensity`, and `build_inmemory_dataloaders` puts the whole array on the GPU). This module
covers the architecture substitution only, so the architecture variable can be tested at a data
scale the existing machinery handles.

The substitutions, exactly as the brief specifies
-------------------------------------------------
    tok_emb  nn.Embedding(61, 512)        ->  nn.Linear(128, 512)     (continuous observations in)
    head     nn.Linear(512, 61, bias=F)   ->  nn.Linear(512, 128)     (continuous observations out)
    loss     cross-entropy                ->  MSE on the next observation

Everything else is **their** `mingpt.model.GPT`, imported and used unmodified: 8 blocks, 8 heads,
`n_embd` 512, full causal attention, **learned absolute** position embeddings, dropout 0.1 on the
embedding / attention / residual paths, post-block LayerNorm, their weight init.

`block_size` is **39**, not 59: dataset-4 episodes are 40 frames, giving 39 inputs. The brief is
explicit that discworld must *not* be regenerated at 60 frames, because that moves `edit_frame` and
breaks every existing comparison.

Interface
---------
Exposes the same names as `pim.world_models.transformer.TransformerModel` where it matters, so the
probe and editor suite can drive it: `embed`, `_run(tokens, mask, edit=, want_resid=)`,
`_seq_mask`, `norm_out`, `decoder`, `residual_stack`, `decode`. `_run`'s `edit` semantics are
identical to `othello_shim.OthelloGPTShim._run` — `(layer, vector)` writes the last position,
a callable fires at every residual point.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn

OTHELLO_ROOT = Path("/home/sevan/research/PIM/othello_world")
if str(OTHELLO_ROOT) not in sys.path:
    sys.path.insert(0, str(OTHELLO_ROOT))

from mingpt.model import GPT, GPTConfig  # noqa: E402


class ArchState(NamedTuple):
    """Everything the model carries between observations: the left-aligned history."""

    obs: torch.Tensor  # (B, T, obs_res), T <= block_size


class OthelloArchDiscworld(nn.Module):
    """Their minGPT with a continuous encoder in and a continuous decoder out."""

    def __init__(self, obs_res: int = 128, block_size: int = 39, n_layer: int = 8,
                 n_head: int = 8, n_embd: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        cfg = GPTConfig(vocab_size=1, block_size=block_size, n_layer=n_layer,
                        n_head=n_head, n_embd=n_embd,
                        embd_pdrop=dropout, resid_pdrop=dropout, attn_pdrop=dropout)
        self.gpt = GPT(cfg)
        self.cfg = cfg
        self.obs_res = obs_res
        # the two substitutions; `tok_emb` and `head` are replaced, never wrapped
        self.gpt.tok_emb = nn.Identity()          # unused — `embed` bypasses it
        self.encoder = nn.Linear(obs_res, n_embd)
        self.gpt.head = nn.Identity()             # unused — `decoder` replaces it
        self.decoder = nn.Linear(n_embd, obs_res)
        self.n_layers = n_layer
        self.probe_layer = n_layer

    # ── the names the probe/editor suite calls ────────────────────────────────

    @property
    def norm_out(self) -> nn.Module:
        return self.gpt.ln_f

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, T, obs_res) -> (B, T, n_embd). Residual point 0.

        Their learned absolute position embedding is added here, and their embedding dropout is
        applied, exactly as `GPT.forward` does — the substitution is the projection, not the
        surrounding machinery. **No ReLU**: `directions/othello-architecture-on-discworld.md`
        pins "no ReLU after the input projection", so residual point 0 is a bare affine map of the
        observation rather than the GRU's `relu(Linear(obs))` encoder port.
        """
        t = obs.shape[1]
        return self.gpt.drop(self.encoder(obs) + self.gpt.pos_emb[:, :t, :])

    def _seq_mask(self, T: int, device) -> None:
        """minGPT masks internally; the argument exists only for signature parity."""
        return None

    def _run(self, tokens, attn_mask=None, edit=None, want_resid=False, kv_sink=None):
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

    def forward(self, obs: torch.Tensor, edit=None) -> torch.Tensor:
        """(B, T, obs_res) -> (B, T, obs_res): the predicted NEXT observation at every position."""
        h, _ = self._run(self.embed(obs), edit=edit)
        return self.decoder(self.norm_out(h))

    @torch.no_grad()
    def residual_stack(self, obs: torch.Tensor, edit=None) -> torch.Tensor:
        """(n_layers+1, B, T, n_embd) — the stream at every residual point."""
        _, resids = self._run(self.embed(obs), edit=edit, want_resid=True)
        return torch.stack(resids, 0)

    # ── the discworld editor surface ──────────────────────────────────────────
    #
    # `nanda_on_discworld.py` and `pinv_alpha_discworld.py` call exactly four names on a model:
    # `state_from_obs`, `flat_state`, `decode`, `rollout_with_edit`. Supplying them is the whole
    # bridge — the probes (`othello_probe.fit_probe`), the editors
    # (`pim.editors.probe_steering.inject_state`, `othello_probe.make_intervention_hook`) and the
    # metrics (`scripts/editability_metrics`) then run against this model unmodified.
    #
    # The state is simply the observation PREFIX, left-aligned. `TransformerModel` needs a
    # right-aligned sliding buffer because its attention is banded and its positions are RoPE
    # (relative). Their minGPT is full-causal with **learned absolute** position embeddings, so a
    # frame's index in the buffer is semantically load-bearing: right-aligning a partly-filled
    # buffer would silently read the wrong position embedding. Left-aligned needs no masking and
    # no padding, and `block_size` 39 ≥ any prefix a 40-frame episode produces.

    @property
    def state_span(self) -> int:
        """Frames carried. Full causal attention ⇒ the whole history, capped at `block_size`."""
        return self.cfg.block_size

    def state_from_obs(self, frames: torch.Tensor) -> ArchState:
        """(B, T, obs_res) observed so far → the carried state."""
        return ArchState(frames[:, -self.state_span :].contiguous())

    def advance(self, state: ArchState, obs_t: torch.Tensor) -> ArchState:
        buf = torch.cat([state.obs, obs_t[:, None, :]], dim=1)
        return ArchState(buf[:, -self.state_span :])

    def flat_state(self, state: ArchState) -> torch.Tensor:
        """(B, n_embd) residual stream at `probe_layer`, current position — the `h` analogue."""
        _, resids = self._run(self.embed(state.obs), want_resid=True)
        return resids[self.probe_layer][:, -1]

    def decode(self, state_or_obs, edit=None) -> torch.Tensor:
        """(B, obs_res) prediction at the last position.

        Accepts a state or a raw `(B, T, obs_res)` tensor, so the same name serves both the
        editors (which pass a state) and direct calls.
        """
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

        Identical contract to `TransformerModel.rollout_with_edit`: the edit shapes the immediate
        prediction, that prediction enters the history, and every later step is recomputed with no
        edit applied — so any persistence has to travel through the observations.
        """
        pred = self.decode_with_edit(state, layer, resid)
        out = [pred]
        s = self.advance(state, pred)
        for _ in range(steps - 1):
            p, s = self.predict_step(s)
            out.append(p)
        return torch.stack(out, 1)


def build(obs_res: int = 128, block_size: int = 39, **kw) -> OthelloArchDiscworld:
    return OthelloArchDiscworld(obs_res=obs_res, block_size=block_size, **kw)
