"""The rules that pick a table's numbers (pim.metrics.selection) and the replicate spread (pim.metrics.replicates)."""
import math

import numpy as np

from pim.metrics.replicates import ci95_halfwidth, pool_replicates, t975
from pim.metrics.selection import best_arm, best_point


def _arm(ed, ei, fid, point=0, alpha=1.0):
    return {"editor": ed, "edit_index": ei, "fidelity_ratio": fid, "point": point, "alpha": alpha}


def test_best_arm_prefers_the_guard_and_falls_back():
    arms = [_arm("PI[zspace]", 0.40, 2.1), _arm("PI[zspace]", 0.35, 0.83, 1), _arm("PI[zspace]", 0.10, 0.5, 2),
            _arm("GS@L0", -0.2, 4.0), _arm("GS@L1", -0.5, 6.0), _arm("PIX", 0.99, 0.1)]
    b = best_arm(arms, "PI", "edit_index")
    assert b["edit_index"] == 0.35 and b["within_guard"] is True          # not the +0.40 at fidelity 2.1
    assert best_arm(arms, "PI", "edit_index", guard=None)["edit_index"] == 0.40
    g = best_arm(arms, "GS", "edit_index")
    assert g["edit_index"] == -0.2 and g["within_guard"] is False         # no arm inside the guard: the unguarded best, flagged
    assert best_arm(arms, "IM", "edit_index") is None
    assert best_arm([_arm("PI", float("nan"), 0.5), _arm("PI", 0.1, 1.0)], "PI", "edit_index")["edit_index"] == 0.1   # 1.0 is inside


def test_best_point():
    assert best_point([0.1, 0.9, 0.3]) == (0.9, 1) and best_point([float("nan"), 0.2]) == (0.2, 1)


def test_replicate_spread():
    rows = [{"parent": "P", "basis": "b", "steps": 512_000, "seed": k, "x": v} for k, v in enumerate((1.0, 2.0, 4.0))]
    rows.append({"parent": "P", "basis": "b", "steps": 780_000, "seed": 9, "x": 100.0})      # another budget: not pooled
    v = pool_replicates(rows, ["x"])[("P", "b")]
    assert v["n"] == 3 and v["dropped_steps"] == [780_000] and math.isclose(v["x"], np.std([1, 2, 4], ddof=1))
    assert math.isclose(v["x_mean"], 7 / 3) and math.isclose(v["x_ci95"], 4.303 * v["x"] / math.sqrt(3)) and v["x_values"] == [1.0, 2.0, 4.0]
    assert t975(2) == 4.303 and t975(22) == t975(20) and t975(40) == 1.960 and math.isnan(ci95_halfwidth([1.0]))


def test_best_arm_by_fidelity():
    from pim.metrics.selection import best_arm_by_fidelity

    arms = [_arm("PI", 0.40, 2.1), _arm("PI", 0.35, 0.83, 1), _arm("PI", 0.10, 0.5, 2), _arm("GS", -0.2, 4.0), _arm("GS", -0.5, 6.0)]
    b = best_arm_by_fidelity(arms, "PI", "edit_index")
    assert (b["fidelity_ratio"], b["edit_index"], b["within_guard"]) == (0.5, 0.10, True)
    g = best_arm_by_fidelity(arms, "GS", "edit_index")
    assert g["fidelity_ratio"] == 4.0 and g["within_guard"] is False and best_arm_by_fidelity(arms, "IM", "edit_index") is None
