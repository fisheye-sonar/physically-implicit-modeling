"""THE guard: one definition, one polarity, both environments (locked 2026-09-01).

    fidelity = RMSE(edited prediction, edited-world GT)
             / RMSE(unsteered prediction, edited-world GT)     — edit step ONLY

`> 1` = the edit left the model further from the post-edit world than doing nothing.
These tests pin the three properties that make it a guard rather than a restatement of
the Edit Index: it is **absolute** (so it catches a wrecked output the relative index
scores positive), it is **step-0** (so a rollout cannot dilute it), and it is **whole
frame / all 64 squares** (so it sees collateral damage outside the edit's own support).
"""

from __future__ import annotations

import numpy as np
import pytest

from pim.metrics.editability import fidelity_ratio
from pim.metrics.othello_moves import move_fidelity_ratio, move_rmse, uniform_over_legal


# ── discworld ────────────────────────────────────────────────────────────────


def _card(edit_frame_rmse: float, gt_traj_rmse: float = 99.0) -> dict:
    # gt_traj_rmse deliberately absurd: if the guard ever reads it again, these fail.
    return {"edit_frame_rmse": edit_frame_rmse, "gt_traj_rmse": gt_traj_rmse}


def test_uses_the_edit_step_not_the_rollout():
    """Changed 2026-09-01. The rollout dilutes an edit that only touches step 0."""
    assert fidelity_ratio(_card(0.4), _card(0.2)) == pytest.approx(2.0)


def test_polarity_gt_one_is_degraded():
    assert fidelity_ratio(_card(0.30), _card(0.20)) > 1.0   # worse than doing nothing
    assert fidelity_ratio(_card(0.10), _card(0.20)) < 1.0   # a real improvement
    assert fidelity_ratio(_card(0.20), _card(0.20)) == pytest.approx(1.0)


def test_zero_denominator_does_not_explode():
    assert np.isfinite(fidelity_ratio(_card(0.1), _card(0.0)))


# ── othello ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def legal():
    return [[0, 1, 2], [10, 20], [5, 6, 7, 8]]


def test_move_rmse_is_zero_on_the_exact_reference(legal):
    perfect = np.stack([uniform_over_legal(L) for L in legal])
    assert move_rmse(perfect, legal) == pytest.approx(0.0, abs=1e-9)


def test_move_fidelity_matches_the_discworld_formula(legal):
    """Same ratio-of-RMSEs shape, so one number means one thing in both worlds."""
    perfect = np.stack([uniform_over_legal(L) for L in legal])
    rng = np.random.default_rng(0)
    uns = rng.dirichlet(np.ones(64), size=len(legal)).astype(np.float32)
    # a perfect edit sits at 0; the unsteered baseline sits at exactly 1
    assert move_fidelity_ratio(perfect, uns, legal) == pytest.approx(0.0, abs=1e-9)
    assert move_fidelity_ratio(uns, uns, legal) == pytest.approx(1.0)


def test_move_fidelity_flags_a_degraded_prediction(legal):
    """A prediction driven AWAY from the post world must score > 1."""
    ref = np.stack([uniform_over_legal(L) for L in legal])
    uns = 0.5 * ref + 0.5 * np.full((len(legal), 64), 1 / 64, np.float32)
    wrecked = np.zeros((len(legal), 64), np.float32)
    wrecked[:, 63] = 1.0                      # all mass on one square
    assert move_fidelity_ratio(wrecked, uns, legal) > 1.0


def test_move_rmse_sees_damage_outside_the_edit_support(legal):
    """Whole-board, not union-support: the guard must see collateral damage.

    Two predictions identical on the legal squares; one dumps mass elsewhere. A
    support-restricted metric would call them equal.
    """
    clean = np.stack([uniform_over_legal(L) for L in legal])
    dirty = clean.copy()
    off_support = [i for i in range(64) if i not in set().union(*[set(L) for L in legal])]
    dirty[:, off_support[0]] = 0.5
    assert move_rmse(dirty, legal) > move_rmse(clean, legal)
