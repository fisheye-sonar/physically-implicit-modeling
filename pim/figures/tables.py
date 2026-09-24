"""pim.figures.tables — the master tables, as figures (2026-09-12).

The table notebooks (``notebooks/build_paper_tables_and_figs.ipynb``, the paper's tables and figures — the
canonical replication notebook shipped with the public code; ``build_appendix_tables_and_figs.ipynb``;
``build_full_tables.ipynb``, the long list) are thin callers of this module: they set the run lists and call
one function per table. Everything here reads the run directories' ``scores.json`` (written by
``master_eval.ipynb`` and the fold-in scripts) and the instance files under ``runs/_baselines/`` — NOTHING
under ``experiments/`` (2026-09-19).

What this module does and does not decide. It DRAWS, and it assembles rows. Every number's definition and
every rule that picks which number a cell shows is imported from ``pim.metrics``:
    selection.best_arm / best_point   the editor arm and the residual point a cell reports
                                      (best Edit Index INSIDE the fidelity guard, else the lowest fidelity ratio;
                                      ``ARM_GUARD`` below)
    replicates.pool_replicates        every ± (SD over seed replicates at a matched budget) and the t-based CI
    decodability.insample_gap_from_stats, prediction.*   the overfit gap; loss / floor / excess
What remains here are DISPLAY POLICIES, each a named constant or a short documented branch: the editors shown
(``EDITORS``), the regression basis (``set_basis`` / ``BASIS_BY_INSTANCE``), Othello's headline index
construction (``OTH_EI``), ND left blank on discworld regression rows, per-component skill at each
component's own best point, and Tables 1d/1e = skill minus the random-init floor in the same basis.

Conventions (Sevan, 2026-09-12): rows follow the run lists as given, Othello before
discworld with a heavy rule between the environments; Othello's Edit Index is the
SYMMETRIC-DIFFERENCE construction (the union stays in scores.json); every per-cell
decodability number is that cell's own optimum over residual points; headers are short.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

from pim.metrics.decodability import insample_gap_from_stats
from pim.metrics.replicates import ci95_halfwidth, t975  # noqa: F401  (re-exported: tests and the CI ledger import them here)
from pim.metrics.replicates import pool_replicates as _pool_replicates
from pim.metrics.edit_index import FIDELITY_GUARD, fidelity
from pim.metrics.selection import GUARD, best_arm, best_arm_by_fidelity, best_point

REPO = Path(__file__).resolve().parents[2]
# The editors SHOWN in every editability table (Sevan, 2026-09-15): PI, GS, IM — the inverse-map
# overwrite is canonical, ND is computed into scores.json but no longer tabulated by default.
EDITORS = ("PI", "GS", "IM")
EDITORS_ALL = ("PI", "ND", "GS", "IM", "IM-NN")     # every editor a row carries (ND / IM-NN on hand)
COMPONENTS = ("o1·x", "o1·y", "o2·x", "o2·y", "o1·vx", "o1·vy", "o2·vx", "o2·vy")
CANONICAL = {"discworld": "frustum", "othello": "mine/theirs"}
REG_BASES = ("frustum", "cartesian")
# Titles on the drawn tables and figures (2026-09-15, Sevan): a notebook sets ``T.SHOW_TITLES = False`` at its
# top to draw every panel without its "(a) …" title and "Table n — …" suptitle (the paper's captions carry them).
SHOW_TITLES = True


def _title(ax, text, **kw):
    if SHOW_TITLES and text:
        ax.set_title(text, **kw)


def _suptitle(fig, text, **kw):
    if SHOW_TITLES and text:
        fig.suptitle(text, **kw)


def set_basis(basis: str) -> None:
    """Switch EVERY discworld table to one regression basis (2026-09-15, Sevan): decodability,
    floors, PI / GS / IM all read the ``basis`` block; the other basis is never read. Call it
    at the top of a table notebook (``T.set_basis(BASIS)``). Grid-defined targets stay as they
    are in either basis. A row with no block in the requested basis is left BLANK, except the
    instances whose basis is fixed by construction (``BASIS_BY_INSTANCE``: dw-8ray-obs5 has no
    frustum block), which show in the basis they have with an asterisk."""
    if basis not in REG_BASES:
        raise ValueError(f"basis must be one of {REG_BASES}, got {basis!r}")
    CANONICAL["discworld"] = basis


# Instances that CANNOT carry every basis, and the one they do — mirrors master_eval SETTINGS
# ["dw_bases_by_instance"]. dw-8ray-obs5 has five observers, so the frustum basis (observer 0's
# lateral fraction + depth) is not defined for it; Sevan (2026-09-15) asked that it still show,
# marked with an asterisk, when frustum is requested.
BASIS_BY_INSTANCE = {"dw-8ray-obs5": ("cartesian",)}


def reg_key(bases: dict, instance: str | None = None) -> "str | None":
    """The REGRESSION block a discworld row is drawn from, or None to leave the row BLANK.

    The requested basis (``set_basis``) when the row has it. Otherwise None — a row with no block in
    the requested basis is blank, never the other basis's numbers (Sevan, 2026-09-15: "if they don't
    have results in the basis I specify then just leave them blank"). The one exception is an
    instance whose basis is fixed by construction (``BASIS_BY_INSTANCE``)."""
    want = CANONICAL["discworld"]
    if want in bases:
        return want
    for b in BASIS_BY_INSTANCE.get(instance or "", ()):
        if b in bases:
            return b
    return None


def basis_star(bases: dict, instance: str | None = None) -> str:
    """'*' when the row is drawn from a basis other than the requested one — only ever the
    ``BASIS_BY_INSTANCE`` exception, since every other mismatch is blank."""
    rk = reg_key(bases, instance)
    return "*" if rk is not None and rk != CANONICAL["discworld"] else ""


def _blank_regression_row(base: dict, key: str) -> dict:
    """A discworld row with no block in the requested basis: labelled, every cell empty."""
    row = {**base, "basis": key, "kind": "regression", "canonical": True,
           "skill_LIN": np.nan, "skill_MLP": np.nan, "tripwire": 0, "unedited": np.nan,
           "gap_LIN": np.nan, "gap_MLP": np.nan}
    for ed in EDITORS_ALL:
        row[f"{ed} EI"] = np.nan
        row[f"{ed} fid"] = np.nan
        row[f"{ed} arm"] = "—"
    return row
OTH_EI = "edit_index_symdiff"            # the Othello headline construction (2026-09-12)
RULE, RULE_ENV = "#172239", "#000000"
ARCH_LABEL = {"transformer_l": "L", "transformer_s": "S", "recurrent_l": "R",
              "transformer_l_tokens": "L·tok", "transformer_s_tokens": "S·tok"}

sns.set_theme(style="white", font_scale=0.95)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300})


# ── collection ───────────────────────────────────────────────────────────────


def find_run(name: str, root: Path = REPO / "runs") -> Path | None:
    """runs/<topic>/<name>/scores.json for a run NAME, outside archive/ and _-prefixed paths."""
    hits = [p for p in root.glob(f"*/{name}/scores.json")
            if p.parts[-3] != "archive" and not p.parts[-3].startswith("_")]
    return hits[0] if hits else None


# THE ARM-SELECTION RULE lives in pim.metrics.selection (2026-09-19): the best Edit Index among the arms
# INSIDE the fidelity guard, the lowest-fidelity-ratio arm only when an editor has no arm inside it (since 2026-09-23;
# the highest-index arm before). ``ARM_GUARD = None``
# restores the old unguarded argmax for every table at once.
ARM_GUARD: "float | None" = GUARD


ARM_SELECT = "index"          # "index" (the rule above) | "fidelity" (highest fidelity = lowest stored ratio) — set per collect() call


def _best_by(arms: list[dict], editor: str, key: str) -> dict | None:
    if ARM_SELECT == "fidelity":
        return best_arm_by_fidelity(arms, editor, key)
    return best_arm(arms, editor, key, guard=ARM_GUARD)


def _arm_str(a: dict | None) -> str:
    if not a:
        return "—"
    return (f"{a['dims']}·" if "dims" in a else "") + f"pt{a['point']}·α{a['alpha']:g}"


def _block_row(base: dict, key: str, T: dict, ei_key: str, kind: str, canonical: bool | None = None) -> dict:
    row = {**base, "basis": key, "kind": kind,
           "canonical": (key == CANONICAL[base["env"]]) if canonical is None else canonical,
           "skill_LIN": best_point(T["probe_skill_linear"])[0], "skill_MLP": best_point(T["probe_skill_mlp"])[0],
           "tripwire": T.get("probe_sanity", {}).get("n_violations", 0),
           "unedited": T["unedited"].get(ei_key, T["unedited"].get("edit_index", np.nan))}
    for fam, k in (("linear", "LIN"), ("mlp", "MLP")):
        ps = {r["point"]: r for r in T.get("probe_sanity", {}).get("rows", [])}
        bp = best_point(T[f"probe_skill_{fam}"])[1]
        row[f"gap_{k}"] = ps.get(bp, {}).get(f"insample_gap_{fam}", np.nan)
    arms = T.get("arms", [])
    inv = T.get("inverse_map") or {}
    im_best = _best_by(arms, "IM", ei_key) if arms else T["best"].get("IM")
    pt = int(im_best["point"]) if im_best else None
    for name, key in (("g_r2", "g_r2"), ("nn_r2", "nn_r2")):
        vals = inv.get(key)
        row[f"{name}@IM"] = (vals[pt] if vals and pt is not None and pt < len(vals) else np.nan)
        row[f"{name} max"] = best_point(vals)[0] if vals else np.nan
        row[f"{name} argmax"] = best_point(vals)[1] if vals else -1
    row["IM point"] = pt if pt is not None else -1
    for ed in EDITORS_ALL:
        b = _best_by(arms, ed, ei_key) if arms else T["best"].get(ed)
        if ed == "ND" and base["env"] == "discworld" and kind == "regression":
            b = None                     # one fixed direction cannot serve 1000 teleports (registry)
        row[f"{ed} EI"] = b.get(ei_key, np.nan) if b else np.nan
        row[f"{ed} fid"] = fidelity(b["fidelity_ratio"]) if b else np.nan     # REPORTED fidelity = 1 - the stored ratio (2026-09-22)
        row[f"{ed} arm"] = _arm_str(b)
        row[f"{ed} guarded"] = bool(b.get("within_guard", True)) if b else False   # False = no arm inside the guard
    return row


@dataclass
class Frames:
    label: str
    runs_oth: list[str]
    runs_dw: list[str]
    df: pd.DataFrame                     # one row per (run, target block)
    perdim: pd.DataFrame                 # discworld per-component skill, per-cell optimum
    base: dict                           # instance -> baselines.json
    frame_set: set = field(default_factory=set)
    rep_sd: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)

    @property
    def canonical(self) -> pd.DataFrame:
        return self.df[self.df["canonical"]].reset_index(drop=True)

    @property
    def extra(self) -> pd.DataFrame:
        return self.df[~self.df["canonical"]].reset_index(drop=True)

    @property
    def archs(self) -> dict:
        """{instance: set of architectures among the listed runs}."""
        out: dict[str, set] = {}
        for r in self.df.itertuples():
            out.setdefault(r.instance, set()).add(r.arch)
        return out


def pool_replicates(rep_rows: list[dict], *, pool_budgets: bool = False, budget_tolerance: float = 0.10) -> dict:
    """``pim.metrics.replicates.pool_replicates`` over the columns a table row carries: both probe skills, the
    unedited index, and every editor's Edit Index and guard (ND / IM-NN included)."""
    cols = ["skill_LIN", "skill_MLP", "unedited"] + [f"{e} EI" for e in EDITORS_ALL] + [f"{e} fid" for e in EDITORS_ALL]
    return _pool_replicates(rep_rows, cols, pool_budgets=pool_budgets, budget_tolerance=budget_tolerance)


def collect(runs_oth: list[str], runs_dw: list[str], label: str = "tables", *,
            pool_budgets: bool = False, budget_tolerance: float = 0.10, select: str = "index") -> Frames:
    """Every listed run's rows + its seed replicates' spread (see ``pool_replicates`` for
    the budget guard and its override). ``select`` = which arm an editor is reported at
    (``pim.metrics.selection``): ``"index"`` — the best Edit Index inside the fidelity guard (the
    tables' rule); ``"fidelity"`` — the highest fidelity, i.e. the lowest stored ratio (the appendix's alternative reading).
    The replicates are read by the same rule, so the ± follows it."""
    global ARM_SELECT
    if select not in ("index", "fidelity"):
        raise ValueError(f"select must be 'index' or 'fidelity', got {select!r}")
    prev, ARM_SELECT = ARM_SELECT, select
    try:
        return _collect(runs_oth, runs_dw, label, pool_budgets=pool_budgets, budget_tolerance=budget_tolerance)
    finally:
        ARM_SELECT = prev


def _collect(runs_oth: list[str], runs_dw: list[str], label: str, *, pool_budgets: bool, budget_tolerance: float) -> Frames:
    rows, perdim, frame_set, missing = [], [], set(), []
    rep_rows = []
    for env, names in (("othello", runs_oth), ("discworld", runs_dw)):
        for name in names:
            sp = find_run(name)
            if sp is None:
                missing.append(name)
                continue
            s = json.loads(sp.read_text())
            base = {"topic": sp.parts[-3], "run": name, "env": env, "instance": s["instance"],
                    "arch": s["arch"], "val": s["val_loss"]}
            if s.get("ei_construction") == "frame-set":
                frame_set.add(name)
            if env == "othello":
                T = {"probe_skill_linear": s["probe_skill"].get("mine|linear|sequence", [np.nan]),
                     "probe_skill_mlp": s["probe_skill"].get("mine|mlp|sequence", [np.nan]),
                     "unedited": s["unedited"], "best": s["best"], "arms": s["arms"],
                     "probe_sanity": {"n_violations": 0, "rows": []},
                     "inverse_map": s.get("inverse_map")}
                row = _block_row(base, "mine/theirs", T, OTH_EI, "classification")
                for fam, k in (("linear", "LIN"), ("mlp", "MLP")):
                    sub = [x for x in s["probe_stats"] if x["target"] == "mine" and x["split"] == "sequence"
                           and x["family"] == fam]
                    b = min(sub, key=lambda x: x["error_rate"]) if sub else None
                    row[f"gap_{k}"] = insample_gap_from_stats(b) if b else np.nan
                row["legal_mass"] = s["gates"]["legal_mass"]
                rows.append(row)
                for key, T in s.get("bases", {}).items():
                    rows.append(_block_row(base, key, T, OTH_EI, T.get("kind", "regression")))
            else:
                # ONE regression basis per table (set_basis): the requested block, the other basis
                # never read; a run with no block in that basis gets a BLANK regression row
                rk = reg_key(s["bases"], s["instance"])
                base = {**base, "star": basis_star(s["bases"], s["instance"])}
                if rk is None:
                    rows.append(_blank_regression_row(base, CANONICAL["discworld"]))
                for key, T in s["bases"].items():
                    if key in REG_BASES and key != rk:
                        continue
                    kind = T.get("kind", "regression")
                    rows.append(_block_row(base, key, T, "edit_index", kind, canonical=(key == rk)))
                    if kind == "regression" and key == rk:
                        for fam, k in (("linear", "LIN"), ("mlp", "MLP")):
                            pp = np.array([p[:len(COMPONENTS)] for p in T[f"probe_perdim_{fam}"]], float)
                            perdim.append({"run": name, "env": env, "arch": s["arch"], "instance": s["instance"],
                                           "basis": key, "probe": k, "point": "per cell",
                                           **dict(zip(COMPONENTS, pp.max(0)))})
            # seed replicates of this run (<name>__seed*) → ± columns
            for rp in sorted((REPO / "runs").glob(f"*/{name}__seed*/scores.json")):
                rs = json.loads(rp.read_text())
                rcfg = (json.loads((rp.parent / "config.json").read_text()).get("replicate", {})
                        if (rp.parent / "config.json").exists() else {})
                blocks = rs.get("bases", {}) if env == "discworld" else {"mine/theirs": {
                    "probe_skill_linear": rs["probe_skill"].get("mine|linear|sequence", [np.nan]),
                    "probe_skill_mlp": rs["probe_skill"].get("mine|mlp|sequence", [np.nan]),
                    "unedited": rs["unedited"], "best": rs["best"], "arms": rs["arms"]}}
                rk_rep = reg_key(blocks, rs.get("instance")) if env == "discworld" else None
                for key, T in blocks.items():
                    if key in REG_BASES and key != rk_rep:
                        continue
                    r = _block_row({**base, "run": rp.parts[-2]}, key, T,
                                   "edit_index" if env == "discworld" else OTH_EI, T.get("kind", "classification"))
                    r["parent"] = name
                    r["steps"] = rcfg.get("steps", np.nan)
                    r["seed"] = rcfg.get("seed", -1)
                    rep_rows.append(r)
    df = pd.DataFrame(rows)
    rep_sd = pool_replicates(rep_rows, pool_budgets=pool_budgets, budget_tolerance=budget_tolerance)
    base_json = {}
    for bp in (REPO / "runs" / "_baselines").glob("*/baselines.json"):
        b = json.loads(bp.read_text())
        base_json[b["instance"]] = b
    return Frames(label, list(runs_oth), list(runs_dw), df, pd.DataFrame(perdim), base_json,
                  frame_set, rep_sd, missing)


# ── drawing primitives ───────────────────────────────────────────────────────


def heat(ax, data, xt, yt, *, fmt, cmap, norm=None, vmin=None, vmax=None, cbar_label, title, annot_text=None):
    sns.heatmap(data, annot=annot_text if annot_text is not None else True,
                fmt="" if annot_text is not None else fmt, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                xticklabels=xt, yticklabels=yt, linewidths=0,
                cbar_kws=dict(label=cbar_label, shrink=0.75, pad=0.02), ax=ax,
                annot_kws=dict(fontsize=9 if annot_text is None else 7.5))
    for j in range(1, np.shape(data)[1]):
        ax.axvline(j, color="white", lw=1.2)
    _title(ax, title, fontsize=10.5, loc="left", pad=8)
    ax.tick_params(labelsize=9, length=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.setp(ax.get_xticklabels(), rotation=0)


def groups_of(labels) -> list:
    out = []
    for v in labels:
        if out and out[-1][0] == v:
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return out


def block_rules(ax, ys, texts=(), heavy=()):
    x0 = 0.0
    if texts:
        fig = ax.figure
        fig.canvas.draw()
        r, inv = fig.canvas.get_renderer(), ax.get_yaxis_transform().inverted()
        x0 = min(inv.transform((t.get_window_extent(r).x0, 0))[0] for t in texts) - 0.01
    for y in ys:
        ax.plot([x0, 1], [y, y], transform=ax.get_yaxis_transform(), clip_on=False,
                color=RULE_ENV if y in heavy else RULE, lw=4.0 if y in heavy else 1.6, solid_capstyle="butt")


def group_rows(ax, groups, x=-0.36, env_break: int | None = None):
    """Two-level row labels + rules between blocks; a HEAVY rule at `env_break` (the row
    index where Othello ends and discworld begins)."""
    y, ys, texts = 0, [], []
    for i, (lab, n) in enumerate(groups):
        if i:
            ys.append(y)
        texts.append(ax.text(x, y + n / 2, lab, transform=ax.get_yaxis_transform(),
                             ha="right", va="center", fontsize=9, color=RULE))
        y += n
    heavy = (env_break,) if env_break is not None and env_break in ys else ()
    block_rules(ax, ys, texts, heavy=heavy)


def env_break_of(df: pd.DataFrame) -> int | None:
    e = list(df["env"])
    return e.index("discworld") if "othello" in e and "discworld" in e else None


def side_rules(ax, groups, env_break=None):
    y = 0
    for _, n in groups[:-1]:
        y += n
        ax.axhline(y, color=RULE_ENV if y == env_break else RULE, lw=4.0 if y == env_break else 1.6)


def _star(r) -> str:
    """The row's basis-fallback mark — '' unless the collector set a string (Othello rows carry NaN)."""
    v = getattr(r, "star", "")
    return v if isinstance(v, str) else ""


def run_groups(df, frame_set=()):
    return groups_of([f"{r.env} · {r.run}{' †' if r.run in frame_set else ''}{_star(r)}" for r in df.itertuples()])


def image_table(df: pd.DataFrame, title: str, col_width: float = 1.1, fontsize: float = 8.5, index=True):
    """A DataFrame drawn as a figure (every table in the notebooks is an image)."""
    d = df.reset_index() if index else df
    n_rows, n_cols = d.shape
    cells = d.astype(object).where(d.notna(), "—").values
    # column widths from the longest string in each column (header included), so run names
    # are never clipped; the figure width follows
    lens = [max(len(str(c)), *(len(str(v)) for v in d[c])) for c in d.columns]
    widths = [0.11 * max(n_ch, 4) for n_ch in lens]
    fig, ax = plt.subplots(figsize=(max(6.0, sum(widths) * fontsize / 8.5 * 0.9), 0.34 * n_rows + 1.0))
    ax.axis("off")
    t = ax.table(cellText=cells, colLabels=list(d.columns), loc="center", cellLoc="center",
                 colWidths=[w / sum(widths) for w in widths])
    t.auto_set_font_size(False)
    t.set_fontsize(fontsize)
    t.scale(1, 1.25)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor("#d8d7d0")
        if i == 0:
            c.set_text_props(weight="bold", color=RULE)
            c.set_facecolor("#f1f0eb")
    _title(ax, title, fontsize=10.5, loc="left", pad=6)
    return fig


def _fmt(v, f):
    return "" if (v is None or (isinstance(v, float) and np.isnan(v))) else format(v, f)


def _annot(F: Frames, df, col, fmt):
    out = np.empty(len(df), dtype=object)
    for i, r in enumerate(df.to_dict("records")):
        v = r[col]
        sd = F.rep_sd.get((r["run"], r["basis"]), {}).get(col, np.nan)
        out[i] = _fmt(v, fmt) + ("" if np.isnan(sd) else f"\n±{sd:.3f}")
    return out


def _mark(F: Frames, names):
    return [f"{n} †" if n in F.frame_set else n for n in names]


# ── Table 1: decodability with its floors ────────────────────────────────────


def _obs_rows(A0: dict, env: str) -> list:
    """The right-aligned observation floor (large corpus preferred) — one row per instance."""
    unit = "games" if env == "othello" else "seq"
    for key, tag in (("observation_right_large", "large"), ("observation_right", "matched")):
        d = A0.get(key, {})
        if d.get("mlp"):
            n = d["mlp"].get("n_seq")
            lab = f"observation · {n // 1000}k {unit}" if n else "observation · matched corpus"
            return [(lab, {"skill_LIN": d["linear"]["skill"], "skill_MLP": d["mlp"]["skill"],
                           "gap_LIN": d["linear"]["insample_gap"], "gap_MLP": d["mlp"]["insample_gap"]})]
    return []


def table_decodability(F: Frames, tag: str = "1"):
    """Table 1 — Probe Skill of every listed run beside its two floors (observation, right-aligned;
    random-init of the same architecture on the same instance), canonical targets only."""
    C = F.canonical
    out = []
    for env in ("othello", "discworld"):
        for inst in dict.fromkeys(C[C["env"] == env]["instance"]):
            b = F.base.get(inst)
            _bb = (b or {}).get("archs", {}).get(next(iter((b or {}).get("archs", {})), ""), {}).get("bases", {})
            bkey = CANONICAL[env] if env == "othello" else reg_key(_bb, inst)
            # The FLOORS follow the same rule as the rows: an instance whose baselines were never
            # fitted in the requested basis (dw-smooth, dw-16ray in cartesian, 2026-09-15) gets BLANK
            # floor rows, not the other basis's numbers — a run can carry a cartesian block while its
            # baselines are frustum-only, which is how frustum floors once read as cartesian.
            fstar = "" if env == "othello" else basis_star(_bb, inst)
            _NAN = {"skill_LIN": np.nan, "skill_MLP": np.nan, "gap_LIN": np.nan, "gap_MLP": np.nan}
            archs = [a for a in dict.fromkeys(C[C["instance"] == inst]["arch"])]
            block = []
            if b and archs and bkey in b["archs"].get(archs[0], {}).get("bases", {}):
                block += [(lab + fstar, cells) for lab, cells in _obs_rows(b["archs"][archs[0]]["bases"][bkey], env)]
            elif b and archs:
                block.append((f"observation · not fitted in {CANONICAL[env]}", dict(_NAN)))
            for a in archs:
                if b and bkey in b["archs"].get(a, {}).get("bases", {}):
                    R = b["archs"][a]["bases"][bkey]["random_init"]
                    block.append((f"random-init · {ARCH_LABEL.get(a, a)}{fstar}",
                                  {"skill_LIN": R["linear"]["skill"], "skill_MLP": R["mlp"]["skill"],
                                   "gap_LIN": R["linear"]["insample_gap"], "gap_MLP": R["mlp"]["insample_gap"]}))
                elif b:
                    block.append((f"random-init · {ARCH_LABEL.get(a, a)}", dict(_NAN)))
                for r in C[(C["instance"] == inst) & (C["arch"] == a)].itertuples():
                    block.append((f"trained · {r.run}{' †' if r.run in F.frame_set else ''}{_star(r)}",
                                  {"skill_LIN": r.skill_LIN, "skill_MLP": r.skill_MLP,
                                   "gap_LIN": r.gap_LIN, "gap_MLP": r.gap_MLP}))
            if not b:
                block.insert(0, ("(no baselines yet)", {"skill_LIN": np.nan, "skill_MLP": np.nan,
                                                        "gap_LIN": np.nan, "gap_MLP": np.nan}))
            for src, cells in block:
                out.append({"env": env, "instance": inst, "source": src, **cells})
    B = pd.DataFrame(out)
    if not len(B):
        return None
    groups = groups_of([f"{r.env} · {r.instance}" for r in B.itertuples()])
    eb = env_break_of(B)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 0.5 * len(B) + 2.0), gridspec_kw=dict(width_ratios=[1, 1], wspace=0.62))
    heat(axes[0], B[["skill_LIN", "skill_MLP"]].values, ["LIN", "MLP"], list(B["source"]), fmt="+.3f",
         cmap="Greens", vmin=0.0, vmax=1.0, cbar_label="Probe Skill", title="(a) Probe Skill, best point")
    group_rows(axes[0], groups, x=-0.72, env_break=eb)
    heat(axes[1], B[["gap_LIN", "gap_MLP"]].values, ["LIN", "MLP"], [""] * len(B), fmt="+.3f",
         cmap="Oranges", vmin=0.0, vmax=0.5, cbar_label="in-sample − held-out", title="(b) overfit check")
    side_rules(axes[1], groups, eb)
    _suptitle(fig, f"Table {tag} — decodability against its floors", fontsize=12, y=1.0)
    return fig


# ── Table 1b: the inverse map's fit beside the retrieval bank's ─────────────


def table_inverse_r2(F: Frames, tag: str = "1b"):
    """Table 1b — held-out R² of the two state→residual instruments on every listed run's
    canonical block (2026-09-16): the inverse map g (MLP-128, `inverse_map.g_r2`) and the
    k-nearest-state retrieval mean (`inverse_map.nn_r2`), each at the point the IM editor's
    best arm writes and at its own best point. Both R² are against the training-mean baseline
    on the same held-out rows (``RetrievalBank.r2``). A run scored before the retrieval R² was
    recorded shows a blank retrieval cell until the scorer folds it in."""
    C = F.canonical
    if not len(C):
        return None
    rows = []
    for r, t in zip(C.to_dict("records"), C.itertuples()):     # column names carry spaces / '@'
        rows.append({"env": r["env"], "instance": r["instance"],
                     "source": f"{r['run']}{' †' if r['run'] in F.frame_set else ''}{_star(t)}",
                     "pt": r["IM point"],
                     "g@IM": r["g_r2@IM"], "g max": r["g_r2 max"], "g arg": r["g_r2 argmax"],
                     "nn@IM": r["nn_r2@IM"], "nn max": r["nn_r2 max"], "nn arg": r["nn_r2 argmax"]})
    B = pd.DataFrame(rows)
    groups = groups_of([f"{r.env} · {r.instance}" for r in B.itertuples()])
    eb = env_break_of(B)
    cols = ["g@IM", "g max", "nn@IM", "nn max"]
    xt = ["inverse map\n@ IM point", "inverse map\nbest point", "retrieval\n@ IM point", "retrieval\nbest point"]
    annot = np.empty((len(B), 4), dtype=object)
    for i, r in enumerate(B.to_dict("records")):
        annot[i] = [_fmt(r["g@IM"], "+.3f") + (f"\n(pt {r['pt']})" if r["pt"] >= 0 else ""),
                    _fmt(r["g max"], "+.3f") + (f"\n(pt {r['g arg']})" if r["g arg"] >= 0 else ""),
                    _fmt(r["nn@IM"], "+.3f"),
                    _fmt(r["nn max"], "+.3f") + (f"\n(pt {r['nn arg']})" if r["nn arg"] >= 0 else "")]
    fig, ax = plt.subplots(1, 1, figsize=(8.6, 0.5 * len(B) + 2.0))
    heat(ax, B[cols].values, xt, list(B["source"]), fmt="+.3f", cmap="Greens", vmin=0.0, vmax=1.0,
         cbar_label="held-out R² (vs training mean)", title="state → residual: held-out R²", annot_text=annot)
    group_rows(ax, groups, x=-0.72, env_break=eb)
    _suptitle(fig, f"Table {tag} — the inverse map beside nearest-state retrieval", fontsize=12, y=1.0)
    return fig


# ── Tables 1b–1e: discworld per-component, per-cell optimum ──────────────────


def _rand_perdim(F: Frames, instance: str, arch: str, fam: str, basis: str) -> np.ndarray:
    """Random-init per-component skill at each COMPONENT'S OWN best point, from the cached
    baseline probes (runs/_baselines/<instance>/probes/*.pt) fitted in the SAME regression basis
    as the row (until 2026-09-19 this read the frustum caches whatever the row's basis — a
    cartesian row had frustum floors subtracted). Where a pre-layout-v2 cache of the same fit
    survives beside the current one, the current (logical ``data`` key) is used."""
    import torch

    pdir = REPO / "runs" / "_baselines" / instance / "probes"
    want_arch = "recurrent_l" if str(arch).startswith("recurrent") else (
        "transformer_l_tokens" if str(arch).endswith("_tokens") else "transformer_l")
    found = []
    for pt in sorted(pdir.glob("probes_*.pt")) if pdir.exists() else []:
        blob = torch.load(pt, map_location="cpu", weights_only=False)
        prov = blob["provenance"]
        if prov.get("model", "none") == "none" or prov.get("target") != "full" or prov.get("family") != fam \
                or prov.get("basis") != basis:
            continue
        parch = "recurrent_l" if int(prov.get("span", 39)) > 100 else (
            "transformer_l_tokens" if prov.get("encoder") else "transformer_l")
        if parch != want_arch:
            continue
        pp = np.array([blob["probes"][p][1]["per_dim_r2"][:len(COMPONENTS)] for p in sorted(blob["probes"])], float)
        found.append((str(prov.get("data", "")).startswith("/"), pp.max(0)))      # legacy path-keyed caches sort last
    return min(found, key=lambda t: t[0])[1] if found else np.full(len(COMPONENTS), np.nan)


def tables_components(F: Frames, above_floor: bool = False):
    """1b/1c (per-cell optimum) or 1d/1e (minus the random-init per-cell optimum)."""
    figs = []
    for tag, fam in ((("1d", "LIN"), ("1e", "MLP")) if above_floor else (("1b", "LIN"), ("1c", "MLP"))):
        P = F.perdim[F.perdim["probe"] == fam].reset_index(drop=True) if len(F.perdim) else F.perdim
        if not len(P):
            continue
        D = P[list(COMPONENTS)].values.astype(float)
        if above_floor:
            famkey = {"LIN": "linear", "MLP": "mlp"}[fam]          # the probe family as the cache names it
            D = D - np.stack([_rand_perdim(F, r.instance, r.arch, famkey, r.basis) for r in P.itertuples()])
        fig, ax = plt.subplots(figsize=(8.6, 0.52 * len(P) + 1.5))
        heat(ax, D, list(COMPONENTS), _mark(F, P["run"]), fmt="+.3f",
             cmap="RdYlGn" if above_floor else "Greens", vmin=-1.0 if above_floor else 0.0, vmax=1.0,
             cbar_label="Probe Skill − random-init" if above_floor else "Probe Skill",
             title=f"Table {tag} — {fam} by component, per-cell best point"
                   + (", above the random-init floor" if above_floor else ""))
        ax.axvline(4, color=RULE, lw=2)
        n = len(P)
        ax.text(2, n + 0.62, "position", ha="center", fontsize=9.5, color="0.3")
        ax.text(6, n + 0.62, "velocity", ha="center", fontsize=9.5, color="0.3")
        figs.append(fig)
    return figs


# ── Table 2: editability ─────────────────────────────────────────────────────


def table_editability(F: Frames, tag: str = "2", note: str = ""):
    """(a) Edit Index with the unedited floor, (b) fidelity = 1 − the RMSE ratio — each editor at the arm ``collect`` selected
    (``note`` names a non-default selection in the title)."""
    C = F.canonical
    if not len(C):
        return None
    ei_cols = ["unedited"] + [f"{e} EI" for e in EDITORS]
    fid_cols = [f"{e} fid" for e in EDITORS]
    groups, eb = run_groups(C, F.frame_set), env_break_of(C)
    rh = 0.7 if F.rep_sd else 0.52
    fig, axes = plt.subplots(1, 2, figsize=(12.6, rh * len(C) + 2.0), gridspec_kw=dict(width_ratios=[5, 4], wspace=0.35))
    A = np.stack([_annot(F, C, c, "+.3f") for c in ei_cols], 1) if F.rep_sd else None
    Af = np.stack([_annot(F, C, c, "+.2f") for c in fid_cols], 1) if F.rep_sd else None
    heat(axes[0], C[ei_cols].values, ["unedited"] + list(EDITORS), list(C["basis"]), fmt="+.3f", cmap="RdYlGn",
         vmin=-1.0, vmax=1.0, cbar_label="Edit Index", annot_text=A, title="(a) Edit Index")
    group_rows(axes[0], groups, x=-0.22, env_break=eb)
    axes[0].axvline(1, color=RULE, lw=2)
    heat(axes[1], C[fid_cols].values, list(EDITORS), [""] * len(C), fmt="+.2f", cmap="RdYlGn",
         norm=TwoSlopeNorm(vmin=-2.0, vcenter=FIDELITY_GUARD, vmax=1.0), cbar_label="fidelity", annot_text=Af,
         title="(b) fidelity  (1 = perfect; 0 = unedited; <0 degraded)")
    side_rules(axes[1], groups, eb)
    _suptitle(fig, f"Table {tag} — editability" + (f" · {note}" if note else "")
              + ("   ± = SD over seed replicates" if F.rep_sd else ""), fontsize=12, y=1.0)
    return fig


def table_arms(F: Frames, tag: str = "2b"):
    C = F.canonical
    if not len(C):
        return None
    d = C.set_index(["env", "run", "basis"])[[f"{e} arm" for e in EDITORS]]
    return image_table(d, f"Table {tag} — best arm per cell (dims·point·α)", col_width=1.5)


def table_gridified(F: Frames, tag: str = "2c"):
    """Only the discworld GRIDIFIED targets (categorical partitions, factorised, snapped), only the
    runs that have them, coarse → fine by cell count within a run."""
    import h5py
    from pim.environments import layout
    from pim.environments.discworld.grid_target import target_cells

    X = F.extra
    X = X[X["env"] == "discworld"].copy()
    if not len(X):
        return None
    sims = {}
    for inst in set(X["instance"]):
        with h5py.File(layout.edits_file("discworld", inst)) as f:
            sims[inst] = json.loads(f.attrs["config_json"])["dataset"]["sim"]
    X["cells"] = [target_cells(b, sims[i]) for b, i in zip(X["basis"], X["instance"])]
    parts = []
    for run in dict.fromkeys(X["run"]):
        d = X[X["run"] == run].sort_values(["cells", "basis"])
        parts.append(d)
    T = pd.concat(parts, ignore_index=True)
    T["label"] = [f"{b} · {c}" for b, c in zip(T["basis"], T["cells"])]
    groups = run_groups(T, F.frame_set)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 0.5 * len(T) + 2.2), gridspec_kw=dict(width_ratios=[2.6, 5, 3.4], wspace=0.5))
    heat(axes[0], T[["skill_LIN", "skill_MLP"]].values, ["LIN", "MLP"], list(T["label"]), fmt="+.3f",
         cmap="Greens", vmin=0.0, vmax=1.0, cbar_label="Probe Skill", title="(a) Probe Skill")
    group_rows(axes[0], groups, x=-1.0)
    heat(axes[1], T[["unedited"] + [f"{e} EI" for e in EDITORS]].values, ["unedited"] + list(EDITORS), [""] * len(T),
         fmt="+.3f", cmap="RdYlGn", vmin=-1.0, vmax=1.0, cbar_label="Edit Index", title="(b) Edit Index")
    axes[1].axvline(1, color=RULE, lw=2)
    heat(axes[2], T[[f"{e} fid" for e in EDITORS]].values, list(EDITORS), [""] * len(T), fmt="+.2f", cmap="RdYlGn",
         norm=TwoSlopeNorm(vmin=-2.0, vcenter=FIDELITY_GUARD, vmax=1.0), cbar_label="fidelity", title="(c) fidelity")
    for ax in axes[1:]:
        side_rules(ax, groups)
    _suptitle(fig, f"Table {tag} — gridified discworld targets (label · cells), coarse → fine", fontsize=12, y=1.0)
    return fig


# ── Appendix: predictive loss beside the Bayes floor (2026-09-19; supersedes table_bayes) ──


def _replicate_losses(run: str, reading: str, rep_sd: dict) -> list[float]:
    """The pooled replicate set's losses for one reading: the members ``pool_replicates`` chose
    (matched budget), read from their own ``prediction`` blocks."""
    steps = next((set(v["steps"]) for (p, _), v in rep_sd.items() if p == run), None)
    out = []
    for rp in sorted((REPO / "runs").glob(f"*/{run}__seed*/scores.json")):
        cfg = rp.parent / "config.json"
        st = json.loads(cfg.read_text()).get("replicate", {}).get("steps") if cfg.exists() else None
        rd = json.loads(rp.read_text()).get("prediction", {}).get("readings", {}).get(reading)
        if rd and (steps is None or st in steps):
            out.append(float(rd["loss_paired"]))
    return out


def prediction_rows(runs_oth: list[str], runs_dw: list[str], rep_sd: dict | None = None) -> pd.DataFrame:
    """One row per (run, reading): the run's held-out loss (``scores.json`` → ``prediction``,
    written by ``scripts/score_prediction.py``), its instance's Bayes floor
    (``runs/_baselines/<instance>/bayes_floor.json``, ``scripts/bayes_floor.py``) and the
    excess — numbers only; ``table_prediction`` draws them. The loss shown is the one PAIRED with
    the floor (the same held-out sequences). The floor is ``value ± pm``: an exact floor (Othello)
    has pm = 0; a sampled floor is the midpoint of its bracket ± (half-width + one SE) —
    ``pim.metrics.prediction.floor_estimate``. The raw bracket ends stay in the frame."""
    from pim.metrics.prediction import excess_estimate, floor_bracket, floor_estimate, gap_closed

    rows = []
    for env, names in (("othello", runs_oth), ("discworld", runs_dw)):
        for name in names:
            sp = find_run(name)
            if sp is None:
                continue
            s = json.loads(sp.read_text())
            fp = REPO / "runs" / "_baselines" / s["instance"] / "bayes_floor.json"
            floor = json.loads(fp.read_text()) if fp.exists() else None
            readings = s.get("prediction", {}).get("readings", {})
            for key, rd in (readings or {None: None}).items():
                est = floor_estimate(floor, rd["objective"]) if rd else None
                br = floor_bracket(floor, rd["objective"]) if rd else None
                ex = excess_estimate(rd["loss_paired"], est) if rd else None
                reps = _replicate_losses(name, key, rep_sd or {}) if rd else []
                tv = ((floor or {}).get("trivial") or {}).get(rd["objective"]) if rd else None
                rows.append({"env": env, "run": name, "instance": s["instance"], "reading": key,
                             "objective": rd["objective"] if rd else None, "unit": rd["unit"] if rd else None,
                             "trivial": tv["value"] if tv else np.nan,
                             "gap_closed": gap_closed(rd["loss_paired"], tv["value"], est[0]) if (tv and est) else np.nan,
                             "loss": rd["loss_paired"] if rd else np.nan,
                             "loss_sd": float(np.std(reps, ddof=1)) if len(reps) >= 2 else np.nan, "n_rep": len(reps),
                             "floor": est[0] if est else np.nan, "floor_pm": est[1] if est else np.nan,
                             "floor_lo": br[0] if br else np.nan, "floor_hi": br[1] if br else np.nan,
                             "excess": ex[0] if ex else np.nan, "excess_pm": ex[1] if ex else np.nan,
                             "excess_rel": ex[2] if ex else np.nan,
                             "n_paired": rd["n_paired"] if rd else np.nan})
    return pd.DataFrame(rows)


READING_LABEL = {"moves": "", "frames": "", "tokens": " · token CE", "expected-frame": " · mean frame MSE"}


def table_prediction(runs_oth: list[str], runs_dw: list[str], tag: str = "A1", rep_sd: dict | None = None):
    """Table A1 as a figure, in the master tables' style: (a) the held-out loss beside the Bayes
    floor, (b) the excess, coloured by its size RELATIVE to the floor (the one scale that is
    comparable between Othello's nats and discworld's squared intensities); the TRIVIAL predictor
    (the best history-blind constant, from the instance's floor file) is the point of comparison at
    the other end — ``gap_closed`` in ``prediction_rows`` places the loss between the two. ``x ± y`` on the loss
    is the SD over seed replicates; on a floor or an excess it is the floor's estimation
    uncertainty — absent where the floor is exact (Othello)."""
    P = prediction_rows(runs_oth, runs_dw, rep_sd)
    if P.empty:
        return None

    def pm(v, d, signed=False):
        if pd.isna(v):
            return "—"
        return format(v, "+.5f" if signed else ".5f") + (
            "" if pd.isna(d) or d == 0 else ("\n± < 0.00001" if d < 5e-6 else f"\n± {d:.5f}"))

    A = np.array([[pm(r.trivial, np.nan), pm(r.loss, r.loss_sd), pm(r.floor, r.floor_pm)] for r in P.itertuples()], dtype=object)
    def ex(r):
        if pd.isna(r.excess):
            return "—"
        return (f"{r.excess:+.5f}   ({r.excess_rel:+.1%})"
                + ("" if pd.isna(r.excess_pm) or r.excess_pm == 0 else f"\n± {r.excess_pm:.5f}"))

    B = np.array([[ex(r)] for r in P.itertuples()], dtype=object)
    ylab = [f"{r.run}{READING_LABEL.get(r.reading, '')}" for r in P.itertuples()]
    groups = groups_of([f"{r.env} · {r.instance}" for r in P.itertuples()])
    eb = env_break_of(P)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 0.62 * len(P) + 2.0),
                             gridspec_kw=dict(width_ratios=[3.0, 1.5], wspace=0.06))
    # (a) numbers in different units per environment: a neutral ground, no colour scale
    sns.heatmap(np.zeros((len(P), 3)), annot=A, fmt="", cmap=["#f4f3ee"], cbar=False, linewidths=0,
                xticklabels=["trivial predictor", "held-out loss", "Bayes floor"], yticklabels=ylab, ax=axes[0],
                annot_kws=dict(fontsize=8.5))
    for j in (1, 2):
        axes[0].axvline(j, color="white", lw=1.2)
    _title(axes[0], "(a) trivial predictor · held-out loss · Bayes floor", fontsize=10.5, loc="left", pad=8)
    axes[0].tick_params(labelsize=9, length=0)
    plt.setp(axes[0].get_yticklabels(), rotation=0)
    group_rows(axes[0], groups, x=-0.42, env_break=eb)
    heat(axes[1], P[["excess_rel"]].to_numpy(float) * 100, ["excess  (loss − floor)"], [""] * len(P), fmt="", cmap="YlOrRd",
         vmin=0.0, vmax=25.0, cbar_label="excess as % of the floor", annot_text=B, title="(b) excess")
    side_rules(axes[0], groups, eb)
    side_rules(axes[1], groups, eb)
    _suptitle(fig, f"Table {tag} — held-out predictive loss beside the Bayes floor\n"
              "Othello: CE, nats / move · discworld: MSE, intensity² / ray (token CE: nats / frame) · ± on the loss = SD over "
              "seed replicates · ± on a floor or excess = the floor's estimation uncertainty (none where exact)\n"
              "trivial predictor = the best history-blind constant (mean frame / move or pattern frequencies), fitted on the probe corpus",
              fontsize=10.5, y=1.07)
    return fig


# ── Table 5: seed variance ───────────────────────────────────────────────────


SEED_VARIANCE_COLS = ("skill_LIN", "skill_MLP", "unedited",
                      "PI EI", "PI fid", "ND EI", "ND fid", "GS EI", "GS fid", "IM EI", "IM fid")


def table_seed_variance(F: Frames, tag: str = "5", which: str = "sd"):
    """Table 5 — the replicate spread per (run, basis) at a matched budget. ``which="sd"``: mean ± SD
    (the readout, 2026-09-18 Sevan); ``which="ci"``: the t-based 95% interval of the mean, [lo, hi],
    the secondary readout. Index AND guard for every editor a row carries."""
    if not F.rep_sd:
        return None
    rows = []
    mixed = any(v.get("pooled_budgets") and len(v["steps"]) > 1 for v in F.rep_sd.values())
    for (run, basis), v in sorted(F.rep_sd.items()):
        row = {"run": run, "basis": basis, "n": v["n"],
               "steps": " / ".join(f"{x // 1000}k" for x in v["steps"]) + (" (mixed budgets)" if len(v["steps"]) > 1 and v.get("pooled_budgets") else ""),
               "seeds": ",".join(str(x) for x in v.get("seeds", [])),
               "not pooled": " / ".join(f"{x // 1000}k" for x in v.get("dropped_steps", [])) or "—"}
        for c in SEED_VARIANCE_COLS:
            if c not in v:
                row[c] = "—"
            elif which == "sd":
                row[c] = f"{v[f'{c}_mean']:+.3f} ± {v[c]:.3f}"
            else:
                h = v.get(f"{c}_ci95", np.nan)
                row[c] = f"[{v[f'{c}_mean'] - h:+.3f}, {v[f'{c}_mean'] + h:+.3f}]" if np.isfinite(h) else "—"
        rows.append(row)
    what = "mean ± SD over the replicate set" if which == "sd" else "95% CI of the mean (t, n − 1) over the replicate set"
    title = (f"Table {tag} — seed replicates: {what}"
             + (" · budgets POOLED (override)" if mixed else " · matched training budget (±10%)"))
    return image_table(pd.DataFrame(rows).set_index(["run", "basis"]), title, col_width=1.25)


# ── Figures ──────────────────────────────────────────────────────────────────


def fig_training_curve(sources: list[str], tag: str = "1"):
    """Fig 1 — decodability, editability and the guard vs training step for the named canonical
    runs (their runs/training_curve/<run>_s<step>/ checkpoints), in the order given (Othello first).
    Othello reads its canonical mine/theirs block only (the signed-regression block would double
    every step); discworld its frustum block."""
    import re

    from matplotlib.lines import Line2D

    from pim.figures.theme import PALETTE

    rows = []
    for sp in sorted((REPO / "runs" / "training_curve").glob("*/scores.json")):
        name = sp.parent.name
        src = re.sub(r"_s\d+$", "", name)
        if src not in sources:
            continue
        s = json.loads(sp.read_text())
        step = int(re.search(r"_s(\d+)$", name).group(1))
        if s["env"] == "othello":
            T = {"probe_skill_linear": s["probe_skill"].get("mine|linear|sequence", [np.nan]),
                 "probe_skill_mlp": s["probe_skill"].get("mine|mlp|sequence", [np.nan]),
                 "unedited": s["unedited"], "arms": s["arms"], "best": s["best"]}
            k = OTH_EI
        else:
            rk = reg_key(s["bases"], s.get("instance"))
            if rk is None:                     # no block in the requested basis — nothing to plot
                continue
            T = s["bases"][rk]
            k = "edit_index"
        row = {"source": src, "env": s["env"], "instance": s["instance"], "arch": s["arch"], "step": step, "run": name,
               "skill_LIN": best_point(T["probe_skill_linear"])[0], "skill_MLP": best_point(T["probe_skill_mlp"])[0],
               "unedited": T["unedited"].get(k, T["unedited"].get("edit_index")), "_arms": T["arms"], "_k": k}
        rows.append(row)
    if not rows:
        return None
    C = pd.DataFrame(rows).sort_values(["source", "step"])
    C["source"] = pd.Categorical(C["source"], [s for s in sources if s in set(C["source"])], ordered=True)
    C = C.sort_values(["source", "step"]).reset_index(drop=True)
    def key_of(a):
        return (a["editor"], a["point"], a["alpha"], a.get("dims"))

    def hexc(i):
        return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in PALETTE[i])

    ENT = {"LIN": (hexc(0), "o"), "MLP": (hexc(1), "s"), "PI": (hexc(3), "o"), "ND": (hexc(2), "s"), "GS": (hexc(4), "^")}
    INK, REF, GRID = "#52514e", "#898781", "#e1e0d9"
    def kfmt(s):
        return f"{s // 1000}k" if s >= 1000 else str(s)

    HOLLOW = Line2D([], [], ls="none", marker="o", mfc="white", mec=INK, mew=1.4, label="hollow: that checkpoint's argmax")
    srcs = list(dict.fromkeys(C["source"]))
    fig, axes = plt.subplots(len(srcs), 3, figsize=(15, 4.1 * len(srcs)), squeeze=False, gridspec_kw=dict(wspace=0.32, hspace=0.6))

    def dress(ax, steps, title, ylabel, extra=(), loc="best"):
        ax.set_xscale("log")
        ax.set_xticks(steps)
        ax.set_xticklabels([kfmt(s) for s in steps], fontsize=8, rotation=35, ha="right")
        ax.minorticks_off()
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp_ in ax.spines.values():
            sp_.set_edgecolor("#c3c2b7")
        _title(ax, title, fontsize=10, loc="left", pad=6)
        ax.set_xlabel("training step", fontsize=9, color=INK)
        ax.set_ylabel(ylabel, fontsize=9, color=INK)
        ax.tick_params(labelsize=8, colors=INK)
        h, _ = ax.get_legend_handles_labels()
        ax.legend(handles=h + list(extra), fontsize=7.5, frameon=False, handlelength=2.4, loc=loc)

    for i, src in enumerate(srcs):
        d = C[C["source"] == src]
        x, steps = d["step"].values, list(d["step"])
        env = d.iloc[0]["env"]
        a, b_, c = axes[i]
        # (a) decodability + random-init floor
        bp = REPO / "runs" / "_baselines" / d.iloc[0]["instance"] / "baselines.json"
        if bp.exists():
            A = json.loads(bp.read_text()).get("archs", {}).get(d.iloc[0]["arch"])
            bkey = CANONICAL[env]
            if A and bkey in A["bases"]:
                R = A["bases"][bkey]["random_init"]
                a.axhline(R["linear"]["skill"], color=REF, lw=1.2, label="random-init · LIN")
                a.axhline(R["mlp"]["skill"], color=REF, lw=1.2, ls=":", label="random-init · MLP")
        for name, col in (("LIN", "skill_LIN"), ("MLP", "skill_MLP")):
            cc, m = ENT[name]
            a.plot(x, d[col].values, color=cc, lw=2, marker=m, ms=6, mec="white", mew=1.2, label=name)
        a.set_ylim(0, 1.02)
        dress(a, steps, f"(a) {src} — Probe Skill, best point", "Probe Skill")
        # (b)/(c): the arm that wins at the LAST checkpoint, read at every checkpoint; hollow = that checkpoint's argmax
        final_arms = d.iloc[-1]["_arms"]
        k = d.iloc[0]["_k"]
        b_.plot(x, d["unedited"].values, color=REF, lw=1.2, label="unedited")
        c.axhline(1.0, color=REF, lw=1.2, label="1.0 = unedited")
        any_h = False
        for ed in EDITORS:
            if ed == "ND" and env == "discworld":
                continue
            fb = _best_by(final_arms, ed, k)
            if not fb:
                continue
            tracked_ei, tracked_fid, own_ei, own_fid, differs = [], [], [], [], []
            for _, r in d.iterrows():
                by = {key_of(aa): aa for aa in r["_arms"]}
                t = by.get(key_of(fb))
                o = _best_by(r["_arms"], ed, k)
                tracked_ei.append(t[k] if t else np.nan)
                tracked_fid.append(fidelity(t["fidelity_ratio"]) if t else np.nan)
                own_ei.append(o[k] if o else np.nan)
                own_fid.append(fidelity(o["fidelity_ratio"]) if o else np.nan)
                differs.append(bool(o) and bool(t) and key_of(o) != key_of(t))
            cc, m = ENT[ed]
            lab = f"{ed} · {_arm_str(fb)}"
            b_.plot(x, tracked_ei, color=cc, lw=2, marker=m, ms=6, mec="white", mew=1.2, label=lab)
            c.plot(x, tracked_fid, color=cc, lw=2, marker=m, ms=6, mec="white", mew=1.2, label=lab)
            dm = np.array(differs)
            if dm.any():
                any_h = True
                b_.plot(x[dm], np.array(own_ei)[dm], ls="none", marker=m, ms=7, mfc="white", mec=cc, mew=1.6, label="_nolegend_")
                c.plot(x[dm], np.array(own_fid)[dm], ls="none", marker=m, ms=7, mfc="white", mec=cc, mew=1.6, label="_nolegend_")
        b_.set_ylim(-1, 1)
        b_.axhline(0, color="#c3c2b7", lw=0.8)
        dress(b_, steps, f"(b) {src} — Edit Index, tracked arm", "Edit Index", extra=[HOLLOW] if any_h else [])
        c.set_ylim(-2.0, 1.0)
        c.axhline(FIDELITY_GUARD, color="#c3c2b7", lw=0.8)
        dress(c, steps, f"(c) {src} — fidelity, tracked arm", "fidelity (1 − ratio)", extra=[HOLLOW] if any_h else [], loc="lower left")
    _suptitle(fig, f"Fig {tag} — training curve: decodability, editability, guard (the final checkpoint's arm, read at every step)",
                 fontsize=11.5, y=1.02)
    return fig
