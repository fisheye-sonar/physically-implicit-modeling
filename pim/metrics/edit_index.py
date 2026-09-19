"""The Edit Index and the fidelity guard — THE formulas, defined once (2026-09-07).

Two constructions use them and differ only in their INGREDIENTS:

* ``zone_editability`` — a predicted FRAME ``(N, R)`` against the two worlds' clean
  renders, on the rays where the renders differ (discworld regression models);
* ``set_editability`` — a predicted DISTRIBUTION ``(N, V)`` over a vocabulary against
  uniform-over-set references, on the union (or symdiff) of the two sets (Othello legal
  moves; a discworld token model's singleton frame sets).

Each builds ``(pred, ref_edited, ref_unedited, support)`` its own way and calls
``edit_index_per_case``. Nothing below knows which world it is scoring.
"""

from __future__ import annotations

import numpy as np

__all__ = ["masked_rmse_per_case", "edit_index_per_case", "fidelity_ratio_from",
           "case_stats", "ratio_ci95"]


def masked_rmse_per_case(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(N,) RMSE between ``pred`` and ``ref`` over each case's ``mask``; NaN where the
    mask is empty (the case has no support and cannot be scored)."""
    out = np.full(len(pred), np.nan)
    for i in range(len(pred)):
        m = mask[i]
        if m.any():
            out[i] = np.sqrt(((pred[i, m] - ref[i, m]) ** 2).mean())
    return out


def edit_index_per_case(pred: np.ndarray, ref_edited: np.ndarray, ref_unedited: np.ndarray,
                        support: np.ndarray) -> np.ndarray:
    """(N,)  ``(d_uned − d_edit) / (d_uned + d_edit)`` per case, with
    ``d_· = RMSE(pred, ref_·)`` over ``support``.

    +1 = the output IS the edited world, −1 = the unedited one, 0 = equidistant
    (ambiguous, or garbage — an output far from BOTH worlds scores ≈ 0, never a spurious
    success). NaN where the support is empty or both distances vanish; callers average
    with ``nanmean`` so unscoreable cases are dropped, not counted as zero.
    """
    d_e = masked_rmse_per_case(pred, ref_edited, support)
    d_u = masked_rmse_per_case(pred, ref_unedited, support)
    with np.errstate(invalid="ignore", divide="ignore"):
        ei = (d_u - d_e) / (d_u + d_e)
    ei[~(d_u + d_e > 1e-12)] = np.nan
    return ei


def fidelity_ratio_from(rmse_edited: float, rmse_unsteered: float, eps: float = 1e-12) -> float:
    """THE guard: ``RMSE(edited prediction, edited-world GT) / RMSE(unsteered, same GT)``
    at the edit step. **> 1 = the edit left the model further from the truth than doing
    nothing** — degraded, not steered. The absolute counterpart of the (relative) Edit
    Index, which scores a wrecked output mildly positive when it lands marginally nearer
    the edited world."""
    return float(rmse_edited) / max(float(rmse_unsteered), eps)


# ── case-level spread (2026-09-18) ────────────────────────────────────────────
#
# Every arm's Edit Index is a mean over bench cases and its guard a ratio of two
# case-aggregates. The numbers below describe how much those aggregates would move under a
# RESAMPLING OF THE BENCH — the metric's own estimation noise on ONE fixed model. They are
# recorded beside each arm so the option exists later; they are NOT the training-seed spread
# the tables quote (that is the replicate set, ``pim.figures.tables.pool_replicates``), and
# the two are never added. Bootstrap: cases resampled with replacement, fixed seed, so a
# rescoring reproduces the interval exactly.

N_BOOT = 1000


def case_stats(per_case, *, n_boot: int = N_BOOT, seed: int = 0, prefix: str = "") -> dict:
    """Case-level spread of a per-case metric whose scalar is its ``nanmean``: the SD across
    scored cases (ddof 1), the standard error of the mean, the number of scored cases, and a
    percentile-bootstrap 95% interval of the mean. NaN cases (unscoreable) are dropped, as
    the scalar drops them. Keys are prefixed (``<prefix>case_sd`` …) so several metrics can
    live in one record."""
    v = np.asarray(per_case, float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    out = {f"{prefix}case_sd": float("nan"), f"{prefix}case_se": float("nan"), f"{prefix}n_cases": n,
           f"{prefix}ci95_lo": float("nan"), f"{prefix}ci95_hi": float("nan")}
    if n < 2:
        return out
    out[f"{prefix}case_sd"] = float(v.std(ddof=1))
    out[f"{prefix}case_se"] = float(v.std(ddof=1) / np.sqrt(n))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    out[f"{prefix}ci95_lo"], out[f"{prefix}ci95_hi"] = float(lo), float(hi)
    return out


def ratio_ci95(num_per_case, den_per_case, *, root: bool = False, n_boot: int = N_BOOT,
               seed: int = 0, eps: float = 1e-12) -> tuple[float, float]:
    """Percentile-bootstrap 95% interval of a guard-style ratio ``agg(num) / agg(den)`` over the
    SAME resampled cases (paired: numerator and denominator are the edited and unsteered
    errors of one case). ``agg`` is the mean — or the root of the mean when ``root`` is set,
    for a ratio of RMSEs built from per-case mean squared errors (discworld's whole-frame
    guard); Othello's guard is a ratio of means of per-case RMSEs, so ``root`` is off there.
    Cases where either side is NaN are dropped, as the scalars drop them."""
    a = np.asarray(num_per_case, float)
    b = np.asarray(den_per_case, float)
    if a.shape != b.shape:
        raise ValueError(f"per-case arrays differ in shape: {a.shape} vs {b.shape}")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    n = int(a.size)
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    na, nb = a[idx].mean(axis=1), b[idx].mean(axis=1)
    if root:
        na, nb = np.sqrt(na), np.sqrt(nb)
    r = na / np.maximum(nb, eps)
    lo, hi = np.percentile(r, [2.5, 97.5])
    return float(lo), float(hi)
