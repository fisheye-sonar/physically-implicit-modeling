"""Shared plumbing for the main-text qualitative figure (paper/figs/qualitative_main, 2026-09-21).

The two appendix scripts are loaded as modules so their drawing helpers (``_panel``, ``error``,
``_blank``, ``draw_board``, ``mark``) draw every strip and board here too; nothing is re-implemented.
Predictions are read from the appendix caches in ``.scratch/`` (``_catim``: the categorical blocks'
IM through the categorical inverse map, 2026-09-21). Scenarios are chosen by the appendix's filter
(``passing_seeds``: the teleport changes at least 2 rays of the clean 5-ray frame, at least 2 of them by at
least 0.2 in intensity). The only metric touched
is the canonical per-case Othello Edit Index (``pim.metrics.set_editability.edit_index_legal``), used by the
"typical case" rule.

    .pim/bin/python paper/figs/qualitative_main/common.py --build-128ray 2   # the 128-ray cache of a seed (GPU)
"""
from __future__ import annotations

import functools
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent
REPO = FIGS.parents[1]
sys.path.insert(0, str(FIGS))
sys.path.insert(0, str(REPO))
import paper_style as ps  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, FIGS / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rw = _load("rw_appendix", "qualitative_edits/make_figure.py")            # ray-world strips (applies paper_style)
oth = _load("oth_appendix", "qualitative_edits_othello/make_figure.py")  # Othello boards + the marking
ps.apply()
import matplotlib.pyplot as plt  # noqa: E402
from pim.metrics.selection import best_arm  # noqa: E402
from pim.metrics.set_editability import edit_index_legal  # noqa: E402

RW_RUNS = {name: (inst, run) for name, inst, run in rw.VARIANTS}          # display name -> (instance, run)
OTH_RUNS = dict(oth.VARIANTS)                                              # display name -> run
HERO_SEEDS = (0, 1, 2)      # R3: the first three cached seeds whose edited disc is visible before the edit
EDITORS = ("PI", "GS", "IM")
GUTTER_IN = 0.52            # the row-label gutter both panels share (widest label: "Adjacent" / "Unedited" at 8 pt, 0.44 in)
# The 128-ray member of the ray family (radius 1.0): the main text's categorical Standard, the one 128-ray model whose
# categorical block carries every editor's arm, the categorical inverse map included. Not an appendix column.
RAY128 = ("128-ray", "dw-128ray", "ray_ablation/L-dw-128ray-20m")
SOURCES = {"appendix": "qualitative_edits_catim_seed{seed}_ctx{context}.pkl",          # rw.VARIANTS (Standard = dw-noiseless, ..., 5-ray)
           "128ray": "qualitative_edits_catim_128ray_seed{seed}_ctx{context}.pkl"}     # RAY128 only


# ── caches ─────────────────────────────────────────────────────────────────────────────
def rayworld(seed: int, context: int = 8, source: str = "appendix") -> dict:
    """The dict ``build()`` of the appendix script returns for one seed: all five appendix variants
    (``appendix``) or the 128-ray member alone (``128ray``, built by ``build_128ray``)."""
    return pickle.load(open(REPO / ".scratch" / SOURCES[source].format(seed=seed, context=context), "rb"))


def build_128ray(seed: int, context: int = 8) -> dict:
    """The 128-ray cache: the appendix machinery (``rw.build`` with one extra variant) on the same scenario."""
    data = rw.build(seed, context, variants=[RAY128])
    path = REPO / ".scratch" / SOURCES["128ray"].format(seed=seed, context=context)
    pickle.dump(data, open(path, "wb"))
    return data


# ── the scenario filter (the appendix's, ``rw.passing_seeds``) ──────────────────────────
@functools.lru_cache(maxsize=None)
def passing_seeds(n: int) -> tuple[int, ...]:
    """The first ``n`` seeds that pass the 5-ray visibility filter (``rw.passes``: at least ``rw.MIN_RAYS`` changed
    rays of the clean 5-ray frame at the edit frame, at least ``rw.MIN_STRONG`` of them by at least ``rw.MIN_DELTA``
    in intensity). CPU, no model."""
    return tuple(rw.passing_seeds(n))


@functools.lru_cache(maxsize=None)
def changed_rays(seed: int, inst: str) -> list[int]:
    """The rays on which the clean edited and unedited frames at the edit frame differ under ``inst``'s renderer."""
    return rw.visible_change(seed, inst).tolist()


def othello() -> dict:
    """``{variant: compute(run)}`` for all 1000 bench cases of each Othello variant."""
    return pickle.load(open(REPO / ".scratch" / "othello_edits_guarded_cache.pkl", "rb"))


# ── Othello helpers: the squares the Edit Index scores, and the zoom window ────────────
def symdiff(col: dict, i: int) -> list[int]:
    """legal_pre XOR legal_post: the squares whose legality the flip changes (the symdiff support)."""
    return sorted(set(col["legal_pre"][i]) ^ set(col["legal_post"][i]))


def marked(col: dict, i: int) -> list[int]:
    """The squares the figure outlines: the flipped tile plus the changed squares (the appendix's helper)."""
    return oth.marked_squares(col["pos"][i], col["legal_pre"][i], col["legal_post"][i])


def bbox(col: dict, i: int, margin: int = 1) -> tuple[int, int, int, int]:
    """(r0, r1, c0, c1) of {edited tile} plus the symdiff squares, grown by ``margin``, clipped to the board."""
    rc = np.array([divmod(s, 8) for s in marked(col, i)])
    r0, c0 = rc.min(0) - margin
    r1, c1 = rc.max(0) + margin
    return max(0, int(r0)), min(7, int(r1)), max(0, int(c0)), min(7, int(c1))


def side(box) -> int:
    r0, r1, c0, c1 = box
    return max(r1 - r0 + 1, c1 - c0 + 1)


def square(box, S: int) -> tuple[int, int, int, int]:
    """Grow ``box`` to an S x S window, centred where possible and kept inside the board."""
    def grow(lo, hi):
        lo -= (S - (hi - lo + 1)) // 2
        lo = min(max(lo, 0), 8 - S)
        return lo, lo + S - 1
    r0, r1 = grow(box[0], box[1])
    c0, c1 = grow(box[2], box[3])
    return r0, r1, c0, c1


def eligible(col: dict, *, min_symdiff: int = 3, max_side: int = 5) -> list[int]:
    return [i for i in range(len(col["pos"])) if len(symdiff(col, i)) >= min_symdiff and side(bbox(col, i)) <= max_side]


def select(cols: dict, variants, *, rule: str = "random", seed: int = 0, rank: int = 1,
           min_symdiff: int = 3, max_side: int = 5) -> dict:
    """One bench case per variant. Eligible: at least ``min_symdiff`` squares change legality and the
    window (edited tile + changed squares + one-square margin) fits ``max_side`` x ``max_side``.
    ``random``: one eligible case per variant from ``default_rng(seed)``, in ``variants`` order.
    ``typical``: the eligible cases ranked by how close their per-case Edit Indices (symdiff construction,
    the canonical ``edit_index_legal``), summed over PI / GS / IM, sit to the variant's population means
    (the guarded arms' indices); ``rank`` 1 = the closest, 2 = the next, ..."""
    rng = np.random.default_rng(seed)
    picks = {}
    for name, _ in variants:
        col = cols[name]
        elig = eligible(col, min_symdiff=min_symdiff, max_side=max_side)
        if rule == "random":
            picks[name] = int(rng.choice(elig))
        elif rule == "typical":
            ei = {ed: edit_index_legal(col["probs"][ed], col["legal_pre"], col["legal_post"], "symdiff") for ed in EDITORS}
            dist = [sum(abs(ei[ed][i] - col["ei"][ed]) for ed in EDITORS) for i in elig]
            picks[name] = int(elig[int(np.argsort(dist, kind="stable")[rank - 1])])
        else:
            raise ValueError(rule)
    return picks


def case_index(col: dict, i: int) -> dict:
    """Per-case Edit Index (symdiff) of every condition for one case, through the canonical function."""
    return {c: float(edit_index_legal(col["probs"][c][i:i + 1], col["legal_pre"][i:i + 1], col["legal_post"][i:i + 1], "symdiff")[0])
            for c in col["probs"]}


# ── the numbers the panel is read against (Table 2 cells, from scores.json through best_arm) ──
DW_BLOCK = {"cont": "cartesian", "cat": "appearance-fac"}


def _arm_row(a: dict | None, key: str) -> dict | None:
    if a is None:
        return None
    return {"point": int(a["point"]), "alpha": float(a["alpha"]), "edit_index": round(float(a[key]), 3),
            "fidelity_ratio": round(float(a["fidelity_ratio"]), 3), "within_guard": bool(a["within_guard"])}


def table2_rayworld(run: str) -> dict:
    s = json.loads((REPO / "runs" / run / "scores.json").read_text())
    out = {}
    for blk, base in DW_BLOCK.items():
        B = s["bases"][base]
        out[blk] = {"block": base, "unedited": round(float(B["unedited"]["edit_index"]), 3),
                    **{ed: _arm_row(best_arm(B["arms"], ed, "edit_index"), "edit_index") for ed in EDITORS}}
    return out


def table2_othello(run: str) -> dict:
    s = json.loads((REPO / "runs" / run / "scores.json").read_text())
    return {"unedited": round(float(s["unedited"]["edit_index_symdiff"]), 3),
            **{ed: _arm_row(best_arm(s["arms"], ed, "edit_index_symdiff"), "edit_index_symdiff") for ed in EDITORS}}


def dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=1))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-128ray", nargs="+", type=int, metavar="SEED", help="build the 128-ray caches for these seeds (GPU)")
    ap.add_argument("--context", type=int, default=8)
    a = ap.parse_args()
    for s in a.build_128ray or ():
        d = build_128ray(s, a.context)
        c = d["cols"][RAY128[0]]
        print(f"seed {s}: edit object {d['edit_object']}  origin {c['cont']['ghost_x']}  destination {c['cont']['target_x']}  "
              f"changed rays {len(c['cont']['differing_rays'])} (5-ray: {len(changed_rays(s, rw.FILTER_INST))})  "
              f"tile changes {c['cat']['changes_tile']}  arms cont {c['cont']['arms']}  cat {c['cat']['arms']}", flush=True)
