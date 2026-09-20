"""The spread over a SEED-REPLICATE set: the ± and the confidence interval the tables quote.

Moved here 2026-09-19 from ``pim.figures.tables`` (which re-exports these names) so that every
number a table prints is defined under ``pim.metrics``; the arithmetic is unchanged.

    pool_replicates   per (parent run, block): SD (n − 1), mean, t-based 95% half-width and the
                      members' own values, over the replicates pooled at a MATCHED training budget
    ci95_halfwidth    t · SD / √n for a small set
    t975              the Student-t 0.975 quantile table behind it

Rows in, numbers out: a replicate row is a flat dict of the quantities a table shows for one
(replicate run, block) plus ``parent``, ``basis``, ``steps`` and ``seed``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Student-t 0.975 quantiles by degrees of freedom (n − 1), for the small replicate sets the
# tables pool; beyond 30 the normal quantile is used. A 95% interval on the MEAN of n seeds is
# mean ± t · SD / √n — at n = 3 that is 2.48 SD, at n = 5 1.24 SD. A df between two listed
# values takes the LOWER listed df's quantile (the wider interval).
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
         16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def t975(df: int) -> float:
    if df in _T975:
        return _T975[df]
    if df > 30:
        return 1.960
    lower = max(k for k in _T975 if k < df)
    return _T975[lower]


def ci95_halfwidth(values) -> float:
    """t-based 95% half-width of the mean over a replicate set (NaN below n = 2)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n < 2:
        return float("nan")
    return float(t975(n - 1) * v.std(ddof=1) / np.sqrt(n))


def pool_replicates(rep_rows: list[dict], cols: list[str], *, pool_budgets: bool = False,
                    budget_tolerance: float = 0.10) -> dict:
    """The replicate spread per (parent run, basis): ``{"n", "steps", "seeds", "dropped_steps",
    <col>: SD (ddof 1), <col>_mean, <col>_ci95, <col>_values}`` over the pooled replicate set,
    for every quantity in ``cols``.

    GUARD (default): replicates are pooled only at a MATCHED training budget — rows whose
    ``steps`` lie within ``budget_tolerance`` (relative) of each other form one set; when a
    parent has replicates at several budgets, the largest set is used (ties → the larger
    budget) and the others are listed under ``dropped_steps`` so the table can say so. A
    390k re-training and the parent's own 421,875-step checkpoint pool (8% apart); a 390k
    and a 780k replicate do not. ``pool_budgets=True`` overrides the guard and pools every
    replicate of the parent regardless of budget (Sevan, 2026-09-14) — the ± then mixes
    training budgets and Table 5 shows every budget it contains.
    """
    if not rep_rows:
        return {}
    R = pd.DataFrame(rep_rows)
    if "steps" not in R:
        R["steps"] = np.nan
    R["steps"] = R["steps"].fillna(-1).astype(int)
    out = {}
    for (parent, basis), g in R.groupby(["parent", "basis"]):
        dropped: list[int] = []
        if not pool_budgets:
            groups: list[list] = []
            for _, r in g.sort_values("steps").iterrows():
                if groups and r["steps"] <= groups[-1][0]["steps"] * (1 + budget_tolerance):
                    groups[-1].append(r)
                else:
                    groups.append([r])
            chosen = max(groups, key=lambda grp: (len(grp), max(r["steps"] for r in grp)))
            dropped = sorted({int(r["steps"]) for grp in groups if grp is not chosen for r in grp})
            g = pd.DataFrame(chosen)
        if len(g) < 2:
            continue
        g = g.sort_values("seed") if "seed" in g else g
        have = [c for c in cols if c in g and g[c].notna().sum() >= 2]
        out[(parent, basis)] = {
            "n": int(len(g)), "steps": sorted({int(x) for x in g["steps"]}),
            "seeds": sorted(int(x) for x in g["seed"]) if "seed" in g else [],
            "dropped_steps": dropped, "pooled_budgets": bool(pool_budgets),
            **{c: float(g[c].std(ddof=1)) for c in have},
            **{f"{c}_mean": float(g[c].mean()) for c in cols if c in g},
            **{f"{c}_ci95": ci95_halfwidth(g[c].dropna().to_numpy()) for c in have},
            **{f"{c}_values": [float(x) for x in g[c]] for c in cols if c in g}}
    return out
