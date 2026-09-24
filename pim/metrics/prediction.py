"""Predictive quality: a model's held-out loss beside the Bayes floor of its environment.

Arrays in, numbers out (2026-09-19). The floor is a property of the ENVIRONMENT INSTANCE —
the loss of the Bayes-optimal predictor that sees the same observation history the model
sees — and lives in ``runs/_baselines/<instance>/bayes_floor.json``; the loss is a property
of the run and lives in its ``scores.json`` under ``prediction``. Both are averaged exactly
like the training objective (every position, every ray / every move), so ``loss − floor``
is the model's distance from optimal in the units it was trained in.

    next_frame_mse      (N, T, R) prediction vs target → per-sequence MSE     (discworld frames)
    next_token_ce       (N, T, V) logits vs (N, T) ids → per-sequence CE      (token models)
    expected_frame      a distribution over a frame vocabulary → the mean frame, so a token
                        model is scored by ``next_frame_mse`` like a frame model
    mean_se             per-sequence values → (mean, standard error over sequences)
    floor_bracket       a sampled floor is stored as a BRACKET (lo, hi); floor_estimate shows it as value ± pm
    gap_closed          (trivial − loss) / (trivial − floor): 1 at the floor, 0 at the trivial predictor
    excess              loss − floor, against both ends of the bracket

Where the floors come from: Othello's is exact (``pim.environments.othello.bayes``), discworld's
is estimated by posterior sampling over the initial state (``pim.environments.discworld.bayes``).
"""
from __future__ import annotations

import math

import numpy as np


def next_frame_mse(pred, target) -> np.ndarray:
    """(N, T, R) predicted next frames vs (N, T, R) true next frames → (N,) MSE per sequence,
    mean over positions and rays — ``pim.training.train.mse_next_obs`` kept per sequence
    (its mean over sequences IS that objective when every sequence has the same T × R)."""
    p, y = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    if p.shape != y.shape or p.ndim != 3:
        raise ValueError(f"expected matching (N, T, R) arrays, got {p.shape} and {y.shape}")
    return ((p - y) ** 2).mean(axis=(1, 2))


def log_softmax(logits) -> np.ndarray:
    z = np.asarray(logits, np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def next_token_ce(logits, target, ignore_index: int | None = None) -> np.ndarray:
    """(N, T, V) logits vs (N, T) integer targets → (N,) mean cross-entropy per sequence (nats).

    ``pim.training.train.ce_next_move`` kept per sequence: positions whose target equals
    ``ignore_index`` are left out of that sequence's mean. With equal-length sequences the
    mean over sequences equals that objective."""
    lp = log_softmax(logits)
    y = np.asarray(target, np.int64)
    keep = np.ones_like(y, bool) if ignore_index is None else (y != ignore_index)
    nll = -np.take_along_axis(lp, np.where(keep, y, 0)[..., None], axis=-1)[..., 0]
    return (nll * keep).sum(1) / np.maximum(keep.sum(1), 1)


def top1_accuracy(logits, target, ignore_index: int | None = None) -> float:
    y = np.asarray(target, np.int64)
    keep = np.ones_like(y, bool) if ignore_index is None else (y != ignore_index)
    return float(((np.asarray(logits).argmax(-1) == y) & keep).sum() / max(int(keep.sum()), 1))


def expected_frame(probs, frames, drop: tuple[int, ...] = (0,)) -> tuple[np.ndarray, np.ndarray]:
    """A distribution over a frame vocabulary → the MEAN frame.

    ``probs`` (..., V) sums to 1; ``frames`` (V, R) is the vocabulary's frame per id. Ids in
    ``drop`` (the reserved UNK / pad id, whose frame is undefined) are removed and the rest
    renormalised. Returns (mean frame (..., R), dropped mass (...)). If the distribution is
    the true next-frame distribution, the mean frame is the MSE-optimal prediction — so a
    token model read this way is compared with the same MSE floor as a frame model."""
    p = np.array(probs, np.float64, copy=True)
    fr = np.array(frames, np.float64, copy=True)
    dropped = p[..., list(drop)].sum(-1) if drop else np.zeros(p.shape[:-1])
    for d in drop:
        p[..., d] = 0.0
        fr[d] = 0.0                                   # UNK decodes to NaN in the vocabulary
    p /= np.maximum(p.sum(-1, keepdims=True), 1e-300)
    return p @ fr, dropped


def mean_se(per_sequence) -> tuple[float, float]:
    """(mean, standard error of the mean over sequences)."""
    a = np.asarray(per_sequence, np.float64)
    return float(a.mean()), (float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else float("nan"))


def floor_bracket(floor: dict, objective: str) -> tuple[float, float] | None:
    """(lo, hi) of an instance's floor for one objective (``"mse"`` / ``"ce"``), or None.

    An exact floor has lo == hi. A SAMPLED floor is a bracket: ``lo`` is the posterior
    spread of the next observation (biased LOW while the sampler under-explores), ``hi`` is
    the realised loss of the sampler's own predictive (an achievable predictor, so never
    below the true floor in expectation). They meet as the sampler converges."""
    b = (floor or {}).get(objective)
    if not b:
        return None
    if "exact" in b:
        return float(b["exact"]), float(b["exact"])
    return float(b["lo"]), float(b["hi"])


def floor_estimate(floor: dict, objective: str) -> tuple[float, float] | None:
    """(value, ±) — the floor as ONE number with its uncertainty, for display (Sevan, 2026-09-19).

    An exact floor is (value, 0.0): no ±. A sampled floor is the MIDPOINT of its bracket, and the ±
    is half the bracket's width (the sampler's unresolved bias, either way) plus one standard
    error over sequences (the larger of the two ends') — so the ± covers both ends of the
    bracket and the sampling noise on them."""
    b = (floor or {}).get(objective)
    if not b:
        return None
    if "exact" in b:
        return float(b["exact"]), 0.0
    lo, hi = float(b["lo"]), float(b["hi"])
    se = max(float(b.get("lo_se") or 0.0), float(b.get("hi_se") or 0.0))
    return (lo + hi) / 2, abs(hi - lo) / 2 + se


def excess_estimate(loss: float, estimate: tuple[float, float] | None) -> tuple[float, float, float] | None:
    """(loss − floor, ±, relative excess = (loss − floor) / floor). The ± is the floor's: the loss
    is taken on the same sequences as the floor, so their sampling noise is largely shared."""
    if estimate is None or loss is None:
        return None
    v, pm = estimate
    return float(loss - v), float(pm), float((loss - v) / v) if v else float("nan")


def gap_closed(loss: float, trivial: float | None, floor: float | None) -> float:
    """(trivial − loss) / (trivial − floor): the share of the ACHIEVABLE improvement over the
    trivial predictor that the model realises — 1 at the Bayes floor, 0 at the trivial predictor.
    Unit-free, so it is the one predictive number comparable across environments (the analogue of
    Probe Skill for the model's own objective)."""
    if loss is None or trivial is None or floor is None or not trivial > floor:
        return float("nan")
    return float((trivial - loss) / (trivial - floor))


def excess(loss: float, bracket: tuple[float, float] | None) -> tuple[float, float] | None:
    """(loss − hi, loss − lo): the model's distance from optimal against both ends of the
    floor's bracket. The second, larger value is the conservative one to quote."""
    if bracket is None or loss is None:
        return None
    return float(loss - bracket[1]), float(loss - bracket[0])
