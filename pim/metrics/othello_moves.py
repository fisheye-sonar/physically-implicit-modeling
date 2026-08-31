"""Canonical editability metrics for Othello: Li error, legal mass, the legal-set Edit Index.

The Othello counterpart of ``pim/metrics/editability.py`` (moved here 2026-08-31 from
``othello_transfer/othello_data.py``). Every function takes plain arrays/lists — the
1001-case benchmark object that supplies ``legal_pre``/``legal_post`` lives in
``pim.environments.othello.bench``, and the vocab-specific logits→board mapping in
``pim.environments.othello.data``.

Two deliberate naming choices:

* ``edit_index_legal``, not ``edit_index`` — the discworld Edit Index (ray-zone RMSE) and
  this one (distance to uniform-over-legal distributions) share the formula
  ``(d_uned − d_edit)/(d_uned + d_edit)`` and the axis, but not the ingredients. The old
  tree gave both the same name and left ``sys.path`` order to pick one; never again.
* ``li_error`` keeps Li et al.'s name because it is their §4.2 metric, ported exactly:
  their null-intervention baseline is 2.68 and their best intervention 0.12, and those
  anchors only mean something if the metric is theirs.
"""

from __future__ import annotations

import numpy as np

N_TILES = 64  # 8x8 board; probability vectors over squares are laid out row-major

__all__ = ["N_TILES", "li_error", "uniform_over_legal", "edit_index_legal", "move_scorecard"]


def li_error(probs: np.ndarray, legal: list[list[int]]) -> np.ndarray:
    """Li et al. §4.2: top-N predictions vs the legal-move set, false pos + false neg.

    ``N = len(legal)``, so both sets have the same size and the error is
    ``2 × (N − overlap)``. Lower is better. NaN where the legal set is empty.
    """
    out = np.full(len(probs), np.nan)
    for i, L in enumerate(legal):
        if not L:
            continue
        top = set(np.argsort(-probs[i])[: len(L)].tolist())
        out[i] = 2 * (len(L) - len(top & set(L)))
    return out


def uniform_over_legal(legal: list[int]) -> np.ndarray:
    """(64,) uniform distribution over the legal squares — the Bayes-optimal reference.

    Not an approximation: the synthetic generator draws moves uniformly from the legal
    set, so this IS the true conditional distribution.
    """
    v = np.zeros(N_TILES, np.float32)
    if legal:
        v[list(legal)] = 1.0 / len(legal)
    return v


def edit_index_legal(
    probs: np.ndarray,
    legal_pre: list[list[int]],
    legal_post: list[list[int]],
    support: str = "union",
) -> np.ndarray:
    """The Edit Index translated onto next-move distributions.

    Same formula as the discworld index: ``(d_uned − d_edit)/(d_uned + d_edit)`` with
    ``d_·`` an RMSE against a ground-truth world, scored on the squares where the two
    worlds differ. **+1** = the output is the edited world, **−1** = the unedited one.
    The reference worlds are uniform-over-legal before and after the board flip.

    ``support="union"`` is the faithful translation of "rays where the two worlds
    differ": the two uniform references renormalise (1/|L0| vs 1/|L1|) and so differ on
    *shared* legal squares too (69.9% of cases). ``support="symdiff"`` scores only
    squares whose legality changed — a narrower question, reported alongside but never
    quoted as the same quantity. Floors for the unedited model over all 1001 cases:
    **−0.829 (union)**, −0.943 (symdiff); a perfect predictor of the unedited world
    scores exactly −1 on both. (Measured 2026-08-20: the unedited model sits 0.0016
    RMSE per square from its reference against a 0.0193 separation — a 12× margin, so
    the floors are sharp.)
    """
    out = np.full(len(probs), np.nan)
    for i, (L0, L1) in enumerate(zip(legal_pre, legal_post)):
        s0, s1 = set(L0), set(L1)
        idx = np.array(sorted(s0 | s1 if support == "union" else s0 ^ s1), int)
        if idx.size == 0:
            continue
        g0, g1 = uniform_over_legal(L0), uniform_over_legal(L1)
        d_un = float(np.sqrt(((probs[i, idx] - g0[idx]) ** 2).mean()))
        d_ed = float(np.sqrt(((probs[i, idx] - g1[idx]) ** 2).mean()))
        if d_un + d_ed == 0:
            continue
        out[i] = (d_un - d_ed) / (d_un + d_ed)
    return out


def move_scorecard(
    probs: np.ndarray,
    legal_pre: list[list[int]],
    legal_post: list[list[int]],
) -> dict:
    """Every number an intervention arm reports, in one place.

    ``li_error_vs_post`` is the headline (their metric). ``li_error_vs_pre`` is the
    guard they do not have: a null intervention is low on *pre* and high on *post*, a
    successful one is the reverse, and an arm that **degraded** the model is high on
    both — which the headline alone cannot distinguish.
    """
    e_post = li_error(probs, legal_post)
    e_pre = li_error(probs, legal_pre)
    ei_u = edit_index_legal(probs, legal_pre, legal_post, "union")
    ei_s = edit_index_legal(probs, legal_pre, legal_post, "symdiff")
    return {
        "li_error_vs_post": float(np.nanmean(e_post)),
        "li_error_vs_pre": float(np.nanmean(e_pre)),
        "edit_index_union": float(np.nanmean(ei_u)),
        "edit_index_symdiff": float(np.nanmean(ei_s)),
        "legal_mass": float(
            np.mean([probs[i, L].sum() for i, L in enumerate(legal_post) if L])
        ),
        "n_scored": int(np.isfinite(e_post).sum()),
        "li_error_vs_post_per_case": e_post.tolist(),
        "edit_index_union_per_case": ei_u.tolist(),
    }
