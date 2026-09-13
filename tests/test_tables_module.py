"""pim.figures.tables — the master-table renderer behind the two table notebooks (2026-09-12).

Data-dependent (reads runs/**/scores.json); skipped where the canonical runs are absent.
Checks the wiring, not any number: rows collect in list order, Othello before discworld,
Othello's headline is the symmetric-difference index, every table function returns a figure
or None without raising.
"""
from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from pim.figures import tables as T  # noqa: E402

OTH, DW = "L-oth-20m", "L-dw-noiseless-20m"
pytestmark = pytest.mark.skipif(T.find_run(OTH) is None or T.find_run(DW) is None,
                                reason="canonical runs not on disk")


def test_collect_orders_and_symdiff():
    F = T.collect([OTH], [DW], label="test")
    assert list(F.df["env"]).index("discworld") > 0 and F.df["env"].iloc[0] == "othello"
    oth = F.df[(F.df["run"] == OTH) & (F.df["basis"] == "mine/theirs")].iloc[0]
    import json
    s = json.loads(T.find_run(OTH).read_text())
    assert oth["unedited"] == s["unedited"]["edit_index_symdiff"]
    best_pi = max((a for a in s["arms"] if a["editor"] == "PI"), key=lambda a: a["edit_index_symdiff"])
    assert oth["PI EI"] == best_pi["edit_index_symdiff"]
    assert set(F.perdim["probe"]) == {"LIN", "MLP"}


def test_every_table_renders_or_declines():
    F = T.collect([OTH], [DW], label="test")
    for fn in (T.table_decodability, T.table_editability, T.table_arms, T.table_gridified,
               T.table_alignment, T.table_bayes, T.table_seed_variance):
        fig = fn(F)
        assert fig is None or hasattr(fig, "savefig")
    figs = T.tables_components(F) + T.tables_components(F, above_floor=True)
    assert all(hasattr(f, "savefig") for f in figs)
    matplotlib.pyplot.close("all")
