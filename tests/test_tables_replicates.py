"""Seed replicates in the tables (2026-09-14): pooled per (parent, basis) at a MATCHED
training budget by default; the override pools every budget."""
import numpy as np

from pim.figures.tables import pool_replicates


def _row(parent, steps, seed, ei, skill=0.9, basis="frustum"):
    return {"parent": parent, "basis": basis, "steps": steps, "seed": seed, "skill_LIN": skill,
            "skill_MLP": skill + 0.05, "unedited": -0.9, "PI EI": ei, "ND EI": np.nan, "GS EI": ei - 0.1,
            "PI fid": 0.9, "ND fid": np.nan, "GS fid": 0.8}


def test_matched_budget_pools_within_tolerance_and_drops_the_rest():
    rows = [_row("P", 390_000, 1, 0.50), _row("P", 390_000, 2, 0.54), _row("P", 421_875, 0, 0.52),
            _row("P", 780_000, 3, 0.70)]                      # a full-budget replicate: not pooled
    out = pool_replicates(rows)
    v = out[("P", "frustum")]
    assert v["n"] == 3 and v["steps"] == [390_000, 421_875] and v["seeds"] == [0, 1, 2]
    assert v["dropped_steps"] == [780_000] and v["pooled_budgets"] is False
    assert abs(v["PI EI_mean"] - 0.52) < 1e-9 and abs(v["PI EI"] - np.std([0.50, 0.54, 0.52], ddof=1)) < 1e-9


def test_override_pools_every_budget():
    rows = [_row("P", 390_000, 1, 0.50), _row("P", 390_000, 2, 0.54), _row("P", 780_000, 3, 0.70)]
    v = pool_replicates(rows, pool_budgets=True)[("P", "frustum")]
    assert v["n"] == 3 and v["steps"] == [390_000, 780_000] and v["dropped_steps"] == [] and v["pooled_budgets"]
    assert abs(v["PI EI_mean"] - 0.58) < 1e-9


def test_largest_budget_group_wins_and_singletons_are_not_a_spread():
    rows = [_row("P", 390_000, 1, 0.5), _row("P", 780_000, 2, 0.7), _row("P", 780_000, 3, 0.72)]
    v = pool_replicates(rows)[("P", "frustum")]
    assert v["steps"] == [780_000] and v["n"] == 2 and v["dropped_steps"] == [390_000]
    assert pool_replicates([_row("P", 390_000, 1, 0.5)]) == {}          # n = 1 is not a spread
    assert pool_replicates([]) == {}


def test_tolerance_is_relative_and_bases_are_separate():
    rows = [_row("P", 390_000, 1, 0.5), _row("P", 430_000, 2, 0.6),          # 10.3% apart: separate at 10%
            _row("P", 390_000, 1, 0.3, basis="appearance-fac"), _row("P", 400_000, 2, 0.4, basis="appearance-fac")]
    out = pool_replicates(rows)
    assert ("P", "frustum") not in out                                   # two singletons after the split
    assert out[("P", "appearance-fac")]["n"] == 2
    assert pool_replicates(rows, budget_tolerance=0.15)[("P", "frustum")]["n"] == 2


# ── the basis switch and the editor list (2026-09-15) ────────────────────────────────────────
from pim.figures import tables as T


def test_set_basis_switches_reg_key_and_stars_fallback():
    try:
        T.set_basis("frustum")
        both = {"frustum": {}, "cartesian": {}}
        assert T.reg_key(both) == "frustum" and T.basis_star(both) == ""
        cart_only = {"cartesian": {}}
        assert T.reg_key(cart_only) == "cartesian" and T.basis_star(cart_only) == "*"
        T.set_basis("cartesian")
        assert T.reg_key(both) == "cartesian" and T.basis_star(both) == ""
        assert T.reg_key({"frustum": {}}) == "frustum" and T.basis_star({"frustum": {}}) == ""   # no star under cartesian
    finally:
        T.set_basis("frustum")


def test_block_row_carries_every_editor_and_shows_pi_gs_im():
    assert T.EDITORS == ("PI", "GS", "IM")
    arms = [{"editor": "PI[zspace]", "point": 2, "alpha": 1.0, "edit_index": 0.2, "fidelity_ratio": 0.9},
            {"editor": "ND", "point": 2, "alpha": 1.0, "edit_index": 0.1, "fidelity_ratio": 1.1},
            {"editor": "GS@L0", "point": 0, "alpha": 0.1, "edit_index": -0.1, "fidelity_ratio": 0.9},
            {"editor": "IM", "point": 4, "alpha": 1.0, "edit_index": 0.7, "fidelity_ratio": 0.3},
            {"editor": "IM-NN", "point": 4, "alpha": 1.0, "edit_index": 0.4, "fidelity_ratio": 0.6}]
    blk = {"probe_skill_linear": [0.5, 0.9], "probe_skill_mlp": [0.6, 0.95], "unedited": {"edit_index": -0.9},
           "best": {}, "arms": arms}
    row = T._block_row({"env": "discworld", "run": "r", "instance": "i", "arch": "transformer_l", "val": 0.0},
                       "frustum", blk, "edit_index", "regression")
    assert row["IM EI"] == 0.7 and row["IM fid"] == 0.3 and row["IM arm"] == "pt4·α1"
    assert row["IM-NN EI"] == 0.4                      # on hand, not shown
    assert np.isnan(row["ND EI"])                       # inapplicable on a regression target
    assert row["PI EI"] == 0.2 and row["GS EI"] == -0.1
