"""The scores.json BLOCK: which blocks a run gets, and the shape of one (moved verbatim from
master_eval cells [3] / [4], 2026-09-19). `probe_block` is the on-disk contract every reader of a
scores.json depends on (pim.figures.tables, the paper_ci ledger, scripts/score_prediction.py)."""
# [3] The discworld scorer — thin wiring over pim.environments.discworld.{bench,arms}
#     (the edit set in bench.py, probes/rollouts/arms in arms.py — the same split as
#     othello/{bench,arms}.py since 2026-09-07).
#     Fitted probes are STORED IN THE RUN DIR (runs/<topic>/<run>/probes/).
#     ⛔ ND is computed but NOT REPORTED for the REGRESSION target (2026-09-01): one fixed
#     direction with a swept scalar is coherent only for a CATEGORICAL target (Othello:
#     flip a tile, same change every case), never for 192 teleports of differing distance
#     and direction. Arms stay in scores.json as a record; the table omits them. On a
#     CATEGORICAL target (grid, appearance) the edit IS categorical (old cell -> empty, new
#     cell -> the object), so ND is a legitimate, reported editor there — as on Othello.
#     ⛔ The bench AND probe corpus come from the RUN'S OWN instance — a model is always
#     scored on the world it was trained in. Both are named by INSTANCE and resolved by
#     pim.environments.layout (layout v2, 2026-09-10): the bench is edits/v1, the probe
#     corpus probe/probe_<size>.h5, and the probe cache key is path-free.
#     A PROBE TARGET (a basis of the regression target, or a categorical target) is its own
#     block under `bases` and its own table ROW — never a column that exists for only some
#     runs. Every block says what it is: `target`, `basis`, `kind` (regression |
#     classification), `probe_recipe`, `alphas`, and — categorical — `bench_selection`
#     (teleports that change cell; a same-cell teleport is a no-op under that target).
#     Regression blocks: ONE probe set per basis (the FULL state); each editor is swept over
#     BOTH dim sets ("pos" = position read-outs only, "all" = the whole state) and the
#     BETTER arm is the reported one, with the winning dim set recorded on it. See SETTINGS
#     for why this loses nothing relative to the retired pos-only probes.
#     A categorical target whose probes are not in the run's probes/ yet is SKIPPED (the
#     scorer never fits them) and lands the next time this notebook runs.
import json

import numpy as np

from pim.environments.discworld.grid_target import categorical_target, snapped_target
from pim.metrics.decodability import probe_skill_from_stats
from pim.scoring.runs import REPO


def best_arm(recs, key="edit_index"):
    b = max(recs, key=lambda r: r[key])
    return {k: v for k, v in b.items() if np.isscalar(v)}

def by_point(probes):
    """A probe set's stats in RESIDUAL-POINT order (0..L) — never the dict's insertion
    order, which a merged cache blob need not share."""
    return [probes[ell][1] for ell in sorted(probes)]

def dw_bases_for(instance, s):
    """The regression bases an INSTANCE is scored in (2026-09-13: dw-8ray-obs5 → cartesian)."""
    return tuple(s.get("dw_bases_by_instance", {}).get(instance, s["dw_bases"]))

def instance_of(run_key):
    cfg = json.loads((REPO / "runs" / run_key / "config.json").read_text())
    return cfg.get("data", {}).get("instance", "dw-pn04")

def discworld_blocks(run_key, s):
    """[(block key, target, basis)] — every probe-target block a run gets: one per basis
    on the regression target, plus the run's extra targets (categorical targets are keyed by
    the target name; their probes live in the instance's first basis — frustum everywhere
    but dw-8ray-obs5)."""
    bases = dw_bases_for(instance_of(run_key), s)
    blocks = [(basis, s["dw_target"], basis) for basis in bases]
    blocks += [(t, t, bases[0]) for t in extra_targets_of(run_key, s)]
    return blocks

def extra_targets_of(run_key, s):
    """A run's extra probe targets: its own SETTINGS entry, or — for a SEED REPLICATE
    (config.json carries `replicate: {"of": "<topic>/<parent>", ...}`, 2026-09-14) — its
    parent's entry, so a replicate is scored on exactly the targets its parent was."""
    if run_key in s["dw_extra_targets"]:
        return tuple(s["dw_extra_targets"][run_key])
    cfg_p = REPO / "runs" / run_key / "config.json"
    if cfg_p.exists():
        rep = json.loads(cfg_p.read_text()).get("replicate")
        if rep and rep.get("of"):
            return tuple(s["dw_extra_targets"].get(rep["of"], ()))
    return ()

def dw_block_setup(target, s):
    """(categorical target or None, dim sets, (alpha_nd, alpha_pi, alpha_gs)) for a target."""
    cat = categorical_target(target)
    if cat is not None:
        return cat, ("all",), (s["dw_grid_alpha_nd"], s["dw_grid_alpha_pi"], s["dw_grid_alpha_gs"])
    snap = snapped_target(target)
    if snap is not None and snap.base == "pos":
        # a 4-output snapped POSITION target: "pos" and "all" are the same read-outs
        return None, ("all",), (s["dw_alpha_nd"], s["dw_alpha_pi"], s["dw_alpha_gs"])
    return None, s["dw_edit_dims"], (s["dw_alpha_nd"], s["dw_alpha_pi"], s["dw_alpha_gs"])

# The editors every block scores (2026-09-15): PI / ND / GS through the probes, IM (the inverse-map
# overwrite, pim.editors.inverse) and IM-NN (its retrieval form) through the state. The tables show
# PI, GS, IM (pim.figures.tables.EDITORS); ND and IM-NN stay in scores.json.
EDITORS_SCORED = ("PI", "ND", "GS", "IM", "IM-NN")
IM_VERSION = "2026-09-15.1"

def cat_inverse_in_scope(instance: str, target: str, s) -> bool:
    """Does this categorical discworld block get an inverse-map arm? Only where SETTINGS ``dw_cat_im``
    says so — ``{"instances": (...), "targets": (...)}`` (2026-09-20: the ray family and
    ``appearance-fac``; each map is a 200k-sequence streamed fit, ~30 min a block). Everywhere else a
    categorical block carries NO IM arm, and its table cell is blank — never the continuous map."""
    c = s.get("dw_cat_im") or {}
    return instance in c.get("instances", ()) and target in c.get("targets", ())


def attach_inverse(blocks: dict, arms_by_key: dict, stats: dict, ei_key: str = "edit_index") -> None:
    """Append a run's IM / IM-NN arms to each block (scalars only), set best['IM'] / best['IM-NN']
    (and per dim set), and record the inverse map's fit."""
    from pim.probes.inverse import INVERSE_EPOCHS, INVERSE_HIDDEN, RETRIEVAL_K
    for key, recs in arms_by_key.items():
        blk = blocks[key]
        blk["arms"] = [r for r in blk["arms"] if r["editor"] not in ("IM", "IM-NN")]
        blk["arms"] += [{k: v for k, v in r.items() if np.isscalar(v)} for r in recs]
        for ed in ("IM", "IM-NN"):
            sub = [r for r in blk["arms"] if r["editor"] == ed]
            blk["best"][ed] = best_arm(sub, ei_key) if sub else None
            for d in blk.get("best_by_dims", {}):
                blk["best_by_dims"][d][ed] = blk["best"][ed]
        blk["inverse_map"] = {"g_r2": stats["g_r2"], "g_rmse": stats["g_rmse"], "hidden": INVERSE_HIDDEN,
                              "epochs": INVERSE_EPOCHS, "k": RETRIEVAL_K, "version": IM_VERSION}

def probe_block(lin, mlp, sanity, u, arms, dimsets, *, target, basis, kind, n_classes, recipe,
                alphas, selection, ei_key="edit_index", extra=None):
    """ONE probe-target block, the same shape in every scorer (discworld frames, discworld
    tokens, Othello): what the target is, Probe Skill per residual point, the tripwire
    report, the unedited floor, the best arm per editor (and per dim set) and every arm."""
    def pick(ed, dims=None):
        # exact editor match ("PI" also owns "PI[zspace]", "GS" owns "GS@L0"; "IM" must NOT own "IM-NN")
        sub = [r for r in arms if (r["editor"] == ed or r["editor"].startswith(ed + "[") or r["editor"].startswith(ed + "@"))
               and (dims is None or r.get("dims", "all") == dims)]
        return best_arm(sub, ei_key) if sub else None
    return {
        "target": target, "basis": basis, "kind": kind, "n_classes": n_classes,
        "probe_recipe": {k: (str(v.relative_to(REPO)) if hasattr(v, "relative_to") else v)
                         for k, v in (recipe or {}).items()},
        "alphas": {"ND": list(alphas[0]), "PI": list(alphas[1]), "GS": list(alphas[2])},
        "bench_selection": selection,                 # None = the first n cases
        "unedited": {k: v for k, v in u.items() if np.isscalar(v)},
        # Probe Skill per residual point — R² on a regression target, 1 − err/majority on
        # a classification one (pim.metrics.probe_skill_from_stats)
        "probe_skill_linear": [probe_skill_from_stats(st) for st in by_point(lin)],
        "probe_skill_mlp": [probe_skill_from_stats(st) for st in by_point(mlp)],
        "probe_perdim_linear": [st.get("per_dim_r2") for st in by_point(lin)],
        "probe_perdim_mlp": [st.get("per_dim_r2") for st in by_point(mlp)],
        "probe_sanity": sanity,
        "best": {ed: pick(ed) for ed in EDITORS_SCORED},
        "best_by_dims": {d: {ed: pick(ed, d) for ed in EDITORS_SCORED} for d in dimsets},
        "arms": [{k: v for k, v in r.items() if np.isscalar(v)} for r in arms],
        **(extra or {}),
    }

def othello_blocks(s):
    """The Othello block keys: the canonical categorical block (implicit, top level) plus
    the extra targets, each a block under `bases`."""
    return ["mine/theirs"] + list(s["oth_extra_targets"])
