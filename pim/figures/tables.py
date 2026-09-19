"""pim.figures.tables — the master tables, as figures (2026-09-12).

The two table notebooks (``notebooks/build_paper_tables_and_figs.ipynb``, the paper's tables and
figures — the canonical replication notebook shipped with the public code; ``notebooks/build_full_tables.ipynb``,
the long list) are thin callers of this module: they
set the run lists and call one function per table. Everything here reads the run
directories' ``scores.json`` (written by ``master_eval.ipynb``), the baselines under
``runs/_baselines/``, and two experiment score files (Table 3, Table 4); nothing is
recomputed and no metric is defined here — selection and drawing only.

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


def _best_by(arms: list[dict], editor: str, key: str) -> dict | None:
    """The arm with the highest `key` among an editor's arms. Arm labels are "PI[zspace]",
    "ND", "GS@L0" on discworld and "PI" / "ND" / "GS" on Othello."""
    sub = [a for a in arms if a["editor"] == editor or a["editor"].startswith(editor + "[")
           or a["editor"].startswith(editor + "@")]
    sub = [a for a in sub if a.get(key) is not None and not (isinstance(a.get(key), float) and np.isnan(a[key]))]
    return max(sub, key=lambda a: a[key]) if sub else None


def _arm_str(a: dict | None) -> str:
    if not a:
        return "—"
    return (f"{a['dims']}·" if "dims" in a else "") + f"pt{a['point']}·α{a['alpha']:g}"


def _block_row(base: dict, key: str, T: dict, ei_key: str, kind: str, canonical: bool | None = None) -> dict:
    row = {**base, "basis": key, "kind": kind,
           "canonical": (key == CANONICAL[base["env"]]) if canonical is None else canonical,
           "skill_LIN": max(T["probe_skill_linear"]), "skill_MLP": max(T["probe_skill_mlp"]),
           "tripwire": T.get("probe_sanity", {}).get("n_violations", 0),
           "unedited": T["unedited"].get(ei_key, T["unedited"].get("edit_index", np.nan))}
    for fam, k in (("linear", "LIN"), ("mlp", "MLP")):
        ps = {r["point"]: r for r in T.get("probe_sanity", {}).get("rows", [])}
        bp = int(np.argmax(T[f"probe_skill_{fam}"]))
        row[f"gap_{k}"] = ps.get(bp, {}).get(f"insample_gap_{fam}", np.nan)
    arms = T.get("arms", [])
    inv = T.get("inverse_map") or {}
    im_best = _best_by(arms, "IM", ei_key) if arms else T["best"].get("IM")
    pt = int(im_best["point"]) if im_best else None
    for name, key in (("g_r2", "g_r2"), ("nn_r2", "nn_r2")):
        vals = inv.get(key)
        row[f"{name}@IM"] = (vals[pt] if vals and pt is not None and pt < len(vals) else np.nan)
        row[f"{name} max"] = max(vals) if vals else np.nan
        row[f"{name} argmax"] = int(np.argmax(vals)) if vals else -1
    row["IM point"] = pt if pt is not None else -1
    for ed in EDITORS_ALL:
        b = _best_by(arms, ed, ei_key) if arms else T["best"].get(ed)
        if ed == "ND" and base["env"] == "discworld" and kind == "regression":
            b = None                     # one fixed direction cannot serve 1000 teleports (registry)
        row[f"{ed} EI"] = b.get(ei_key, np.nan) if b else np.nan
        row[f"{ed} fid"] = b["fidelity_ratio"] if b else np.nan
        row[f"{ed} arm"] = _arm_str(b)
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


def pool_replicates(rep_rows: list[dict], *, pool_budgets: bool = False,
                    budget_tolerance: float = 0.10) -> dict:
    """The replicate spread per (parent run, basis): ``{"n", "steps", "seeds", "dropped_steps",
    <col>: SD (ddof 1), <col>_mean}`` over the pooled replicate set.

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
    # every editor a row carries (ND / IM-NN included) and its guard: the SD is the readout,
    # the t-based 95% half-width (``<col>_ci95``) the secondary one, and ``<col>_values`` the
    # members themselves in seed order (2026-09-18, Sevan)
    cols = ["skill_LIN", "skill_MLP", "unedited"] + [f"{e} EI" for e in EDITORS_ALL] + [f"{e} fid" for e in EDITORS_ALL]
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


# Student-t 0.975 quantiles by degrees of freedom (n − 1), for the small replicate sets the
# tables pool; beyond 30 the normal quantile is used. A 95% interval on the MEAN of n seeds is
# mean ± t · SD / √n — at n = 3 that is 2.48 SD, at n = 5 1.24 SD.
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


def collect(runs_oth: list[str], runs_dw: list[str], label: str = "tables", *,
            pool_budgets: bool = False, budget_tolerance: float = 0.10) -> Frames:
    """Every listed run's rows + its seed replicates' spread (see ``pool_replicates`` for
    the budget guard and its override)."""
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
                    row[f"gap_{k}"] = ((b["error_rate"] - b["error_rate_insample"]) / b["majority_class_error_rate"]
                                      if b else np.nan)
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


def _rand_perdim(F: Frames, instance: str, arch: str, fam: str) -> np.ndarray:
    """Random-init per-component skill at each COMPONENT'S OWN best point, from the cached
    baseline probes (runs/_baselines/<instance>/probes/*.pt)."""
    import torch

    pdir = REPO / "runs" / "_baselines" / instance / "probes"
    want_arch = "recurrent_l" if str(arch).startswith("recurrent") else (
        "transformer_l_tokens" if str(arch).endswith("_tokens") else "transformer_l")
    for pt in sorted(pdir.glob("probes_*.pt")) if pdir.exists() else []:
        blob = torch.load(pt, map_location="cpu", weights_only=False)
        prov = blob["provenance"]
        if prov.get("model", "none") == "none" or prov.get("target") != "full" or prov.get("family") != fam \
                or prov.get("basis") != "frustum":
            continue
        parch = "recurrent_l" if int(prov.get("span", 39)) > 100 else (
            "transformer_l_tokens" if prov.get("encoder") else "transformer_l")
        if parch != want_arch:
            continue
        pp = np.array([blob["probes"][p][1]["per_dim_r2"][:len(COMPONENTS)] for p in sorted(blob["probes"])], float)
        return pp.max(0)
    return np.full(len(COMPONENTS), np.nan)


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
            D = D - np.stack([_rand_perdim(F, r.instance, r.arch, famkey) for r in P.itertuples()])
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


def table_editability(F: Frames, tag: str = "2"):
    C = F.canonical
    if not len(C):
        return None
    ei_cols = ["unedited"] + [f"{e} EI" for e in EDITORS]
    fid_cols = [f"{e} fid" for e in EDITORS]
    groups, eb = run_groups(C, F.frame_set), env_break_of(C)
    rh = 0.7 if F.rep_sd else 0.52
    fig, axes = plt.subplots(1, 2, figsize=(12.6, rh * len(C) + 2.0), gridspec_kw=dict(width_ratios=[5, 4], wspace=0.35))
    A = np.stack([_annot(F, C, c, "+.3f") for c in ei_cols], 1) if F.rep_sd else None
    Af = np.stack([_annot(F, C, c, ".2f") for c in fid_cols], 1) if F.rep_sd else None
    heat(axes[0], C[ei_cols].values, ["unedited"] + list(EDITORS), list(C["basis"]), fmt="+.3f", cmap="RdYlGn",
         vmin=-1.0, vmax=1.0, cbar_label="Edit Index", annot_text=A, title="(a) Edit Index")
    group_rows(axes[0], groups, x=-0.22, env_break=eb)
    axes[0].axvline(1, color=RULE, lw=2)
    heat(axes[1], C[fid_cols].values, list(EDITORS), [""] * len(C), fmt=".2f", cmap="RdYlGn_r",
         norm=TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=3.0), cbar_label="fidelity ratio", annot_text=Af,
         title="(b) fidelity ratio  (1 = unedited; >1 degraded)")
    side_rules(axes[1], groups, eb)
    _suptitle(fig, f"Table {tag} — editability" + ("   ± = SD over seed replicates" if F.rep_sd else ""), fontsize=12, y=1.0)
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
    heat(axes[2], T[[f"{e} fid" for e in EDITORS]].values, list(EDITORS), [""] * len(T), fmt=".2f", cmap="RdYlGn_r",
         norm=TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=3.0), cbar_label="fidelity ratio", title="(c) fidelity ratio")
    for ax in axes[1:]:
        side_rules(ax, groups)
    _suptitle(fig, f"Table {tag} — gridified discworld targets (label · cells), coarse → fine", fontsize=12, y=1.0)
    return fig


# ── Table 3: edit-direction alignment ────────────────────────────────────────


def table_alignment(F: Frames, tag: str = "3",
                    path: Path = REPO / "experiments" / "edit_direction_alignment" / "scores" / "table3_alignment.json",
                    haufe_path: Path = REPO / "experiments" / "edit_direction_alignment" / "scores" / "table3_haufe_edit.json"):
    """Rows: every listed run × (canonical target, appearance-fac). Alignment columns from the
    experiment's JSON; PI before Haufe from scores.json; PI after Haufe from the (queued) Haufe file."""
    A = {(r["run"].split("/")[-1], r["target"]): r for r in json.loads(path.read_text())} if path.exists() else {}
    H = {(r["run"].split("/")[-1], r["target"]): r for r in json.loads(haufe_path.read_text())} if haufe_path.exists() else {}
    rows = []
    for r in F.df.to_dict("records"):
        if not r["canonical"] and r["basis"] != "appearance-fac":
            continue
        a, h = A.get((r["run"], r["basis"]), {}), H.get((r["run"], r["basis"]), {})
        rows.append({"env": r["env"], "run": r["run"], "target": r["basis"], "pt": a.get("point", "—"),
                     "n": a.get("n_cases", "—"),
                     "rows": a.get("rows_frac"), "rows rnd": a.get("rows_generic"), "× rnd": a.get("rows_ratio"),
                     "Haufe": a.get("haufe_frac"), "Haufe rnd": a.get("haufe_generic"), "× rnd (H)": a.get("haufe_ratio"),
                     "PI EI": r["PI EI"], "PI fid": r["PI fid"],
                     "PI EI (H)": h.get("pi_ei"), "PI fid (H)": h.get("pi_fid")})
    if not rows:
        return None
    T = pd.DataFrame(rows)
    for c in ("rows", "rows rnd", "Haufe", "Haufe rnd"):
        T[c] = T[c].map(lambda v: _fmt(v, ".3f") if v is not None else "—")
    for c in ("× rnd", "× rnd (H)"):
        T[c] = T[c].map(lambda v: _fmt(v, ".1f") if v is not None else "—")
    for c in ("PI EI", "PI EI (H)"):
        T[c] = T[c].map(lambda v: _fmt(v, "+.3f") if v is not None else "—")
    for c in ("PI fid", "PI fid (H)"):
        T[c] = T[c].map(lambda v: _fmt(v, ".2f") if v is not None else "—")
    d = T.set_index(["env", "run", "target"])
    return image_table(d, f"Table {tag} — true edit direction vs the probe row space (at the best PI point); "
                          f"(H) = Haufe-corrected", col_width=0.95, fontsize=8)


# ── Table 4: Bayes floor vs test loss ────────────────────────────────────────


def table_bayes(F: Frames, tag: str = "4", path: Path = REPO / "experiments" / "bayes_floor" / "scores" / "test_loss.json"):
    L = json.loads(path.read_text()) if path.exists() else {}
    rows = []
    seen = set()
    for r in F.df.itertuples():
        if r.run in seen:
            continue
        seen.add(r.run)
        key = next((k for k in L if k.split("/")[-1] == r.run), None)
        e = L.get(key, {})
        rows.append({"env": r.env, "run": r.run, "instance": r.instance, "unit": e.get("unit", "—"),
                     "test loss": _fmt(e.get("test_loss"), ".5f") if e else "—",
                     "Bayes floor": _fmt(e.get("bayes_floor"), ".5f") if e.get("bayes_floor") is not None else "—",
                     "excess": _fmt(e.get("excess"), ".5f") if e.get("excess") is not None else "—"})
    if not rows:
        return None
    return image_table(pd.DataFrame(rows).set_index(["env", "run"]), f"Table {tag} — test loss vs the estimated Bayes floor",
                       col_width=1.25)


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
               "skill_LIN": max(T["probe_skill_linear"]), "skill_MLP": max(T["probe_skill_mlp"]),
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
                tracked_fid.append(t["fidelity_ratio"] if t else np.nan)
                own_ei.append(o[k] if o else np.nan)
                own_fid.append(o["fidelity_ratio"] if o else np.nan)
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
        c.set_ylim(0, 3.0)
        dress(c, steps, f"(c) {src} — fidelity ratio, tracked arm", "fidelity ratio", extra=[HOLLOW] if any_h else [], loc="upper left")
    _suptitle(fig, f"Fig {tag} — training curve: decodability, editability, guard (the final checkpoint's arm, read at every step)",
                 fontsize=11.5, y=1.02)
    return fig


def fig_capacity(tag: str = "2"):
    from pim.figures.probe_capacity import capacity_figure

    cap = REPO / "experiments" / "probe_capacity" / "scores"
    files = [cap / f"probe_capacity_{e}.json" for e in ("discworld", "othello")]
    if not any(p.exists() for p in files):
        return None
    fig = capacity_figure(files)
    _suptitle(fig, f"Fig {tag} — probe-capacity sweep", fontsize=11.5, y=1.02)
    return fig
