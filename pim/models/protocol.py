"""THE model surface — the names probes, editors, and benches call, documented once.

Both canonical architectures (Transformer-S, Transformer-L) implement this surface in
both task forms, which is what lets one editability suite drive every (architecture,
environment) cell without an ``isinstance`` anywhere.

Core surface (every model)
--------------------------
``embed(inp) -> (B, T, d)``
    Input to residual point 0. Regression: observation frames; tokens: move indices.
``_run(tokens, attn_mask=None, edit=None, want_resid=False) -> (h, resids)``
    The block stack. ``edit`` is the intervention socket, with two forms:
      * ``(layer, vector)`` — force the stream at that residual point, LAST position;
      * callable ``fn(layer_idx, x) -> x`` — fires at EVERY residual point 0…n_layers
        with the whole stream, for multi-site / multi-layer writes (gradient steering,
        history edits).
    ``None`` must leave the pass bit-identical — this is load-bearing and gated.
``residual_stack(inp) -> (n_points, B, T, d)``
    The stream at every residual point; ``n_points = n_layers + 1``, where point
    ``ell`` is the stream AFTER ``ell`` blocks (0 = the embedding).
``norm_out`` / ``decoder``
    The output head, split so an editor can read ``decoder(norm_out(h))`` itself.
``decode(state_or_input, edit=None) -> (B, out)``
    Prediction at the last position — the intervention view.

Rollout surface (regression models; token models raise until rollout is designed)
---------------------------------------------------------------------------------
``state_from_obs(frames) -> State`` · ``advance(state, obs) -> State`` ·
``flat_state(state) -> (B, d)`` · ``predict_step(state)`` ·
``rollout_with_edit(state, layer, resid, steps) -> (B, steps, out)``
    The honest measurement of an activation write: the edit shapes the immediate
    prediction, that prediction enters the carried state, and every later step is
    recomputed with NO edit applied — persistence must travel through observations.

The carried state differs by design and that difference is under study:
Transformer-S carries a right-aligned sliding window (banded attention + relative
RoPE positions); Transformer-L carries the left-aligned prefix (full attention +
learned absolute positions). See each module's docstring for why the alignment is
forced by the position encoding.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class WorldModel(Protocol):
    """Static face of the surface above (the docstring is the real contract)."""

    n_layers: int
    probe_layer: int

    def embed(self, inp: torch.Tensor) -> torch.Tensor: ...

    def _run(self, tokens, attn_mask=None, edit=None, want_resid=False): ...

    def residual_stack(self, inp: torch.Tensor, edit=None) -> torch.Tensor: ...

    def decode(self, state_or_input, edit=None) -> torch.Tensor: ...


def n_points(model) -> int:
    """Residual points a model exposes: n_layers + 1."""
    return model.n_layers + 1
