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

__all__ = ["masked_rmse_per_case", "edit_index_per_case", "fidelity_ratio_from"]


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
