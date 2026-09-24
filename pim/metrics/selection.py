"""WHICH number a table reports — the selection rules, in one place (2026-09-19).

The scorer records every residual point's probe skill and every (editor, point, step size) arm
in ``scores.json``; a table shows ONE number per cell. These are the rules that pick it. They
read what the scorer wrote, so changing a rule never needs a re-scoring.

    best_point   a probe's reported skill = its maximum over residual points
    best_arm_by_fidelity   the alternative reading (appendix): the arm with the LOWEST fidelity ratio — the write that
                 brings the prediction closest to the edited world in absolute terms — whatever its index
    best_arm     an editor's reported arm = the highest Edit Index AMONG THE ARMS INSIDE THE GUARD
                 (fidelity ratio ≤ 1: the edit did not leave the output further from the edited
                 world than doing nothing). Only if an editor has NO arm inside the guard is its
                 LOWEST-fidelity-ratio arm reported instead — the least-degrading write — and the
                 row says so (``within_guard`` False).

⛔ 2026-09-23 (Sevan): the fallback was the highest-index arm overall (2026-09-19 → 09-23), which on a
row where every editor fails reported the index of a badly degraded output (oth-adjacent GS −0.157 at
fidelity −5.68). It is now the highest-fidelity arm (oth-adjacent GS −0.96 at −0.01), the reading the
paper states. Only editors with no arm inside the guard move; every guarded cell is unchanged.

⛔ Until 2026-09-19 the tables took the highest Edit Index REGARDLESS of the guard (and
``scores.json["best"]`` is the scorer's own argmax, on Othello under the union construction).
The guard moved 16 of the 48 (run, block, editor) cells of the paper's tables, mostly PI — e.g.
oth-adjacent-flip PI: +0.399 at fidelity 2.09 by the old rule, +0.348 at 0.83 by this one;
dw-noiseless PI: +0.197 at 1.71 → −0.096 at 0.96. Pass ``guard=None`` for the old rule.
"""
from __future__ import annotations

import numpy as np

GUARD = 1.0          # a fidelity ratio above this = the edit degraded the prediction


def best_point(values) -> tuple[float, int]:
    """(maximum over residual points, its index); ``(nan, -1)`` when there is no finite value —
    a categorical block's retrieval R² is one NaN per point (no bank), stored since 2026-09-23."""
    v = np.asarray(values, float)
    if v.size == 0 or not np.isfinite(v).any():
        return float("nan"), -1
    return float(np.nanmax(v)), int(np.nanargmax(v))


def arms_of(arms: list[dict], editor: str) -> list[dict]:
    """An editor's arms. Arm labels are "PI[zspace]", "ND", "GS@L0" on discworld and
    "PI" / "ND" / "GS" on Othello."""
    return [a for a in arms if a["editor"] == editor or a["editor"].startswith(editor + "[")
            or a["editor"].startswith(editor + "@")]


def best_arm(arms: list[dict], editor: str, key: str, guard: float | None = GUARD) -> dict | None:
    """The arm an editor is REPORTED at: max ``key`` among arms with ``fidelity_ratio <= guard``;
    if none is inside the guard, the arm with the lowest ``fidelity_ratio`` (max ``key`` overall
    only if no arm carries a ratio). The returned dict is a copy carrying ``within_guard``.
    ``guard=None`` = the unguarded argmax (the pre-2026-09-19 rule)."""
    def ok(v):
        return v is not None and not (isinstance(v, float) and np.isnan(v))

    sub = [a for a in arms_of(arms, editor) if ok(a.get(key))]
    if not sub:
        return None
    rated = [a for a in sub if ok(a.get("fidelity_ratio"))]
    inside = [a for a in rated if guard is not None and a["fidelity_ratio"] <= guard]
    if inside:
        pick = max(inside, key=lambda a: a[key])
    elif guard is not None and rated:
        pick = min(rated, key=lambda a: a["fidelity_ratio"])
    else:
        pick = max(sub, key=lambda a: a[key])
    return {**pick, "within_guard": bool(inside)}


def best_arm_by_fidelity(arms: list[dict], editor: str, key: str) -> dict | None:
    """The arm with the LOWEST fidelity ratio among an editor's arms that carry ``key`` (its Edit
    Index is then read off that same arm). The fidelity ratio is absolute — edited prediction's
    error against the edited world over the unedited prediction's — so this picks the write that
    most improves the output, where ``best_arm`` picks the one that most moves it between the two
    worlds. ``within_guard`` says whether that best ratio is <= ``GUARD`` at all."""
    def ok(v):
        return v is not None and not (isinstance(v, float) and np.isnan(v))

    sub = [a for a in arms_of(arms, editor) if ok(a.get(key)) and ok(a.get("fidelity_ratio"))]
    if not sub:
        return None
    pick = min(sub, key=lambda a: a["fidelity_ratio"])
    return {**pick, "within_guard": bool(pick["fidelity_ratio"] <= GUARD)}
