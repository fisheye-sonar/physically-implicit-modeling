"""Editability of a discworld TOKEN model, scored the Othello way (2026-09-05).

A frames-as-tokens model (``TransformerLTokens`` over the instance's frame vocabulary,
``tokens.py``) emits a DISTRIBUTION over next frames. Othello's scoring consumes exactly
that object: the next-move distribution against the legal-move sets of the pre- and
post-edit boards. Here the "legal sets" are the frames the two worlds render at the edit
frame — in a noiseless world each is ONE token — so the machinery transfers verbatim:

* cases, probe targets, change masks and zones are the canonical edit set's, via
  ``bench.bench_arrays`` (the same 192 teleports every discworld number is scored on);
* the editors are the discworld ones — PI (``pim.editors.pinv``, z-space + y-affine) and
  GS (``pim.editors.grad_steer``) on regression probes — writing the residual at the last
  context position through the model's ``decode(idx, edit=hook)``, as Othello's arms do;
* the numbers are ``pim.metrics.set_editability``: ``edit_index_legal`` (reported here as
  ``edit_index`` — the frame-set construction, +1 = the edited world's frame, −1 = the
  unedited one), ``li_error``, the mass on the post-edit frame (``p_post``), and
  ``move_fidelity_ratio`` as the guard.

Also carried, never quoted as the headline: ``zone_edit_index_expected`` — the canonical
ray-zone Edit Index (``pim.metrics.zone_editability.edit_index``) evaluated on the EXPECTED
frame Σ p_k·frame_k, the bridge to the regression model's construction.

Cases whose two worlds render the SAME frame at the edit frame carry no signal and are
dropped (``keep``); so is any case whose frame is outside the vocabulary (never, by
construction — the vocabulary is built over every split).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pim.editors.grad_steer import build_edit_spec, make_intervention_hook
from pim.editors.pinv import pinv_step, readout_error
from pim.environments.discworld import bench as dwb
from pim.environments.discworld.tokens import UNK, FrameVocab, encode
from pim.metrics.zone_editability import edit_index as zone_edit_index
from pim.metrics.set_editability import move_fidelity_ratio, move_scorecard

DEV, EF = dwb.DEV, dwb.EF


@dataclass
class TokenBench:
    tokens: np.ndarray        # (n, EF) int64 — frames 0..EF-1, the model's context
    pre_tok: np.ndarray       # (n,) the frame the UNEDITED world renders at EF
    post_tok: np.ndarray      # (n,) the frame the EDITED world renders at EF
    keep: np.ndarray          # (n,) bool — pre != post and both in the vocabulary
    tgt: torch.Tensor         # (n, d_out) the probe target the edit asks for
    change_mask: torch.Tensor  # (n, d_out) bool — the edited object's dims only
    out_dims: list[int]
    zones: object             # ray zones (for the expected-frame bridge)
    vocab: FrameVocab
    n: int

    @property
    def legal_pre(self) -> list[list[int]]:
        return [[int(t)] if k else [] for t, k in zip(self.pre_tok, self.keep)]

    @property
    def legal_post(self) -> list[list[int]]:
        return [[int(t)] if k else [] for t, k in zip(self.post_tok, self.keep)]


def load_token_bench(vocab: FrameVocab, n: int = 192, target: str = "pos",
                     basis_name: str = "cartesian", data_dir: Path | None = None) -> TokenBench:
    """The canonical edit set as tokens: context, the two worlds' frames at EF, targets."""
    a = dwb.bench_arrays(n, target, basis_name, data_dir)
    tokens = encode(a["obs"][:, :EF], vocab).astype(np.int64)
    post = encode(a["clean"][:, EF], vocab).astype(np.int64)
    pre = encode(a["zones"].gt_unedited, vocab).astype(np.int64)
    keep = (pre != post) & (pre != UNK) & (post != UNK) & (tokens != UNK).all(1)
    return TokenBench(tokens, pre, post, keep, torch.from_numpy(a["y"]).float().to(DEV),
                      torch.from_numpy(a["change_mask"]).to(DEV), a["out_dims"], a["zones"],
                      vocab, a["n"])


def frame_probs(outputs: torch.Tensor, kind: str = "logits") -> torch.Tensor:
    """(B, V) head outputs → (B, V) next-frame estimates, by the model's output kind."""
    if kind == "logits":
        return torch.softmax(outputs.float(), -1)
    if kind == "raw":
        return outputs.float()
    if kind == "clipnorm":
        p = outputs.float().clamp_min(0)
        s = p.sum(-1, keepdim=True)
        return torch.where(s > 0, p / s.clamp_min(1e-12), torch.full_like(p, 1 / p.shape[-1]))
    raise ValueError(f"unknown output kind {kind!r}")


@torch.no_grad()
def probs_at_edit(model, tb: TokenBench, hook=None) -> np.ndarray:
    """(n, V) the model's next-frame distribution after the context, under an edit hook."""
    idx = torch.from_numpy(tb.tokens).to(DEV)
    out = model.decode(idx, edit=hook)
    return frame_probs(out, getattr(model, "output_kind", "logits")).cpu().numpy()


def expected_frame(probs: np.ndarray, vocab: FrameVocab) -> np.ndarray:
    """(n, R) Σ p_k · frame_k over the real frames (UNK mass renormalised away)."""
    p = probs[:, 1:]
    p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
    return p @ vocab.frames[1:]


def scorecard(probs: np.ndarray, tb: TokenBench, uns: np.ndarray | None = None) -> dict:
    """Every number an arm reports — Othello's ``move_scorecard`` on frame sets."""
    c = move_scorecard(probs, tb.legal_pre, tb.legal_post)
    out = {"edit_index": c["edit_index_union"],           # the frame-set construction
           "edit_index_symdiff": c["edit_index_symdiff"],
           "li_error_vs_post": c["li_error_vs_post"], "li_error_vs_pre": c["li_error_vs_pre"],
           "p_post": c["legal_mass"],
           "p_pre": float(np.mean([probs[i, L].sum() for i, L in enumerate(tb.legal_pre) if L])),
           "n_scored": c["n_scored"],
           "edit_index_per_case": c["edit_index_union_per_case"],
           "zone_edit_index_expected": float(zone_edit_index(
               expected_frame(probs, tb.vocab)[tb.keep], _mask_zones(tb.zones, tb.keep)))}
    if uns is not None:
        out["fidelity_ratio"] = move_fidelity_ratio(probs, uns, tb.legal_post)
    return out


def _mask_zones(zones, keep):
    """The zones restricted to the kept cases (``edit_index`` reads the per-case masks)."""
    import dataclasses
    kw = {}
    for f in dataclasses.fields(zones):
        v = getattr(zones, f.name)
        kw[f.name] = v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(keep) else v
    return dataclasses.replace(zones, **kw)


@torch.no_grad()
def unsteered(model, tb: TokenBench) -> tuple[np.ndarray, dict]:
    probs = probs_at_edit(model, tb)
    c = scorecard(probs, tb)
    c["fidelity_ratio"] = 1.0
    return probs, c


@torch.no_grad()
def residuals_last(model, tb: TokenBench) -> dict[int, torch.Tensor]:
    """{point: (n, d)} the residual stream at the LAST context position, every point."""
    rs = model.residual_stack(torch.from_numpy(tb.tokens).to(DEV))
    return {ell: rs[ell][:, -1] for ell in range(len(rs))}


def _write_hook(ell: int, h: torch.Tensor):
    def hook(layer, x):
        if layer != ell:
            return x
        out = x.clone()
        out[:, -1] = h
        return out
    return hook


def pinv_arm(model, tb: TokenBench, probes: dict, alphas, uns: np.ndarray,
             space: str = "zspace", dims: str = "all") -> list[dict]:
    """PI at ONE residual point, every point, α swept — ``arms.pinv_arm`` on tokens."""
    idx = dwb.dim_idx(dims)
    x0 = residuals_last(model, tb)
    recs = []
    for ell, (probe, _) in probes.items():
        h0 = x0[ell]
        step = pinv_step(h0, tb.tgt, probe, space=space, dims=idx)
        err0 = readout_error(h0, tb.tgt, probe, dims=idx)
        for a in alphas:
            h = h0 + a * step
            probs = probs_at_edit(model, tb, hook=_write_hook(ell, h))
            recs.append({"editor": f"PI[{space}]", "point": int(ell), "alpha": float(a),
                         "dims": dims,
                         "write_ratio": float((a * step).norm(dim=1).div(h0.norm(dim=1)).mean()),
                         "readout_err_before": err0,
                         "readout_err_after": readout_error(h, tb.tgt, probe, dims=idx),
                         **scorecard(probs, tb, uns)})
    return recs


def grad_steer_arm(model, tb: TokenBench, probes: dict, start_layers, alphas,
                   uns: np.ndarray, n_steps: int = 100, beta: float = 0.2,
                   dims: str = "all") -> list[dict]:
    """GS from each start layer and every point after it — ``bench.grad_steer_arm`` on tokens."""
    cm = dwb.restrict_mask(tb.change_mask, dims)
    x0 = residuals_last(model, tb)
    recs = []
    for ls in start_layers:
        pts = {e: probes[e][0] for e in probes if e >= ls}
        for a in alphas:
            specs = {e: build_edit_spec(pr, x0[e], cm, tb.tgt, beta=beta) for e, pr in pts.items()}
            rec: dict = {}
            hook = make_intervention_hook(pts, specs, ls, alpha=a, n_steps=n_steps, record=rec)
            probs = probs_at_edit(model, tb, hook=hook)
            recs.append({"editor": f"GS@L{ls}", "point": int(ls), "alpha": float(a), "dims": dims,
                         "write_ratio": float(np.mean(
                             [d["delta_norm"] / d["x_norm"] for d in rec.values()
                              if isinstance(d, dict) and d.get("x_norm", 0) > 0] or [np.nan])),
                         **scorecard(probs, tb, uns)})
    return recs


def token_encoder(vocab: FrameVocab):
    """The ``fit_probes(encoder=...)`` hook for a frames-as-tokens model, plus its cache tag."""
    def enc(obs: np.ndarray) -> np.ndarray:
        return encode(obs, vocab)
    tag = f"tokens:V{vocab.size}:{int(vocab.counts.sum())}"
    return enc, tag
