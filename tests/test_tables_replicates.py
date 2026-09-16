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
        # dw-8ray-obs5 has no frustum block BY CONSTRUCTION: it shows in cartesian, starred
        assert T.reg_key(cart_only, "dw-8ray-obs5") == "cartesian" and T.basis_star(cart_only, "dw-8ray-obs5") == "*"
        assert T.reg_key(cart_only) is None            # any other instance: blank
        T.set_basis("cartesian")
        assert T.reg_key(both) == "cartesian" and T.basis_star(both) == ""
        # a run scored in frustum only has NO cartesian result: blank, never the frustum numbers
        assert T.reg_key({"frustum": {}}) is None and T.basis_star({"frustum": {}}) == ""
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


def test_star_label_is_empty_for_rows_without_a_string_mark():
    import pandas as pd
    df = pd.DataFrame([{"env": "othello", "run": "a", "star": np.nan}, {"env": "discworld", "run": "b", "star": "*"},
                       {"env": "discworld", "run": "c", "star": ""}])
    labels = [g[0] for g in T.run_groups(df)]
    assert labels == ["othello · a", "discworld · b*", "discworld · c"]


def test_floor_rows_are_blank_when_the_baselines_lack_the_requested_basis(tmp_path, monkeypatch):
    """dw-smooth / dw-16ray (2026-09-15): the RUN has a cartesian block, its baselines do not — the
    floor cells are BLANK, never the frustum numbers."""
    import pandas as pd
    try:
        T.set_basis("cartesian")
        base = {"archs": {"transformer_l": {"bases": {"frustum": {
            "random_init": {"linear": {"skill": 0.8, "insample_gap": 0.0}, "mlp": {"skill": 0.9, "insample_gap": 0.0}},
            "observation_right_large": {"linear": {"skill": 0.3, "insample_gap": 0.0},
                                        "mlp": {"skill": 0.9, "insample_gap": 0.0, "n_seq": 250000}}}}}}}
        df = pd.DataFrame([{"env": "discworld", "instance": "dw-smooth", "run": "r", "arch": "transformer_l",
                            "canonical": True, "basis": "cartesian", "star": "", "skill_LIN": 0.96,
                            "skill_MLP": 0.99, "gap_LIN": 0.0, "gap_MLP": 0.0}])
        F = T.Frames("t", [], ["r"], df, pd.DataFrame(), {"dw-smooth": base})
        fig = T.table_decodability(F)
        labels = [t.get_text() for ax in fig.axes for t in ax.get_yticklabels()]
        assert any("not fitted in cartesian" in l for l in labels), labels
        assert any(l.startswith("trained") and not l.endswith("*") for l in labels), labels   # the run IS cartesian
        import numpy as _np
        rows = [r for r in T._LAST_DECODABILITY.itertuples()] if hasattr(T, "_LAST_DECODABILITY") else []
        del rows
    finally:
        T.set_basis("frustum")
