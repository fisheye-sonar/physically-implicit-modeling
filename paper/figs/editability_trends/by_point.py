"""Edit Index against residual point, per editor (appendix "Inverse Mapping Editability Trends Across
Residual Points").

Every number is READ from a run's scores.json. At each residual point the editor's reported arm is
``pim.metrics.selection.best_arm`` over the arms AT that point (the tables' rule: the best Edit Index
inside the fidelity guard, the unguarded best only when no arm passes; ``within_guard`` says which). GS
arms carry their start layer as their point and write from it onward, so GS is plotted at its start
point. The second row (A2) reads the inverse map's R² and the MLP probe skill per point. Nothing is
computed here. Outputs land beside this script.

    .pim/bin/python paper/figs/editability_trends/by_point.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))            # paper/figs
import paper_style as ps                                                # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.figures import tables as T                                     # noqa: E402  find_run, OTH_EI
from pim.figures.theme import style_ax                                  # noqa: E402
from pim.metrics.selection import arms_of, best_arm                     # noqa: E402

matplotlib.rcdefaults()                     # importing pim.figures.tables applies seaborn's theme; undo it
ps.apply()
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.lines import Line2D                                     # noqa: E402

OUT = Path(__file__).resolve().parent
PIECES = OUT / "pieces"
PAD = 0.2      # ps.save's tight bbox adds 0.1 in on every side: figsize = the SAVED page size minus this
EDITORS = ("PI", "GS", "IM")
MARK = {"PI": "o", "GS": "s", "IM": "^"}
GREY, ZERO = "#7f7f7f", "#c8c8c8"
# panel label -> run name (find_run locates the topic directory)
RUNS = {"Othello": "L-oth-20m", "Rayworld": "L-dw-noiseless-20m",
        "Othello, adjacent-flip": "L-oth-adjacent-flip-20m", "Rayworld, 8-ray": "L-dw-8ray-20m"}
SLUG = {"Othello": "othello", "Rayworld": "rayworld",
        "Othello, adjacent-flip": "othello_adjacent_flip", "Rayworld, 8-ray": "rayworld_8ray"}
STANDARD, EXTRA = ("Othello", "Rayworld"), ("Othello, adjacent-flip", "Rayworld, 8-ray")


def load(name: str) -> dict:
    s = json.loads(T.find_run(name).read_text())
    if s["env"] == "othello":
        d = dict(arms=s["arms"], key=T.OTH_EI, skill=s["probe_skill"]["mine|mlp|sequence"],
                 g_r2=s["inverse_map"]["g_r2"])
    else:
        b = s["bases"]["cartesian"]
        d = dict(arms=b["arms"], key="edit_index", skill=b["probe_skill_mlp"], g_r2=b["inverse_map"]["g_r2"])
    d["run"], d["n_points"], d["best"] = name, int(s["n_points"]), {}
    for ed in EDITORS:                          # the reported arm at each point: best_arm over that point's arms
        rows = []
        for p in range(d["n_points"]):
            b = best_arm([a for a in arms_of(d["arms"], ed) if a["point"] == p], ed, d["key"])
            if b is not None:
                rows.append((p, b))
        d["best"][ed] = rows
    return d


DATA = {lab: load(name) for lab, name in RUNS.items()}


# ── drawing ──────────────────────────────────────────────────────────────────


def _ax(ax):
    style_ax(ax)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8, labelcolor=ps.TEXT, length=2.5)
    ax.set_xticks(range(9))
    ax.set_xlim(-0.35, 8.35)


def draw_ei(ax, d: dict) -> bool:
    """Edit Index of the reported arm at each point; hollow = outside the guard. Returns whether any is hollow."""
    ax.axhline(0, color=ZERO, lw=0.6, zorder=0)
    any_hollow = False
    for ed in EDITORS:
        pts, c = d["best"][ed], ps.EDITOR_COLORS[ed]
        ax.plot([p for p, _ in pts], [b[d["key"]] for _, b in pts], color=c, marker=MARK[ed], ms=3.2, lw=1.0, zorder=3)
        hollow = [(p, b[d["key"]]) for p, b in pts if not b["within_guard"]]
        if hollow:
            any_hollow = True
            ax.plot(*zip(*hollow), ls="none", marker=MARK[ed], ms=3.2, mfc="white", mec=c, zorder=4)
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    _ax(ax)
    return any_hollow


def draw_skill(ax, d: dict) -> None:
    x = range(d["n_points"])
    ax.plot(x, d["g_r2"], color=ps.EDITOR_COLORS["IM"], marker="d", ms=2.6, lw=0.9, zorder=3, clip_on=False)
    ax.plot(x, d["skill"], color=GREY, marker="o", ms=2.6, lw=0.9, zorder=3, clip_on=False)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    _ax(ax)


def keys(two_row: bool, hollow: bool) -> list:
    h = [Line2D([], [], color=ps.EDITOR_COLORS[e], marker=MARK[e], ms=3.2, lw=1.0, label=e) for e in EDITORS]
    if hollow:
        h.append(Line2D([], [], ls="none", marker="o", ms=3.2, mfc="white", mec=ps.TEXT, label="outside fidelity guard"))
    if two_row:
        h += [Line2D([], [], color=ps.EDITOR_COLORS["IM"], marker="d", ms=2.6, lw=0.9, label="inverse map R²"),
              Line2D([], [], color=GREY, marker="o", ms=2.6, lw=0.9, label="MLP probe skill")]
    return h


LEGEND_KW = dict(handlelength=1.6, columnspacing=1.0, handletextpad=0.4)


def compose(labels, two_row: bool, stem: Path) -> None:
    n = len(labels)
    fig, axes = plt.subplots(2 if two_row else 1, n, squeeze=False, sharex=True, sharey="row",
                             figsize=(ps.TEXT_WIDTH_IN - PAD, (3.2 if two_row else 2.2) - PAD), layout="constrained",
                             gridspec_kw=dict(height_ratios=[1.75, 1] if two_row else [1]))
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.06)
    hollow = False
    for j, lab in enumerate(labels):
        hollow |= draw_ei(axes[0, j], DATA[lab])
        axes[0, j].set_title(lab, pad=3)
        if two_row:
            draw_skill(axes[1, j], DATA[lab])
        axes[-1, j].set_xlabel("Residual point")
    axes[0, 0].set_ylabel("Edit Index")
    if two_row:
        axes[1, 0].set_ylabel("Skill")
    h = keys(two_row, hollow)
    fig.legend(handles=h, loc="outside lower center", ncol=len(h), **LEGEND_KW)
    ps.save(fig, stem)
    plt.close(fig)


def piece(draw, d: dict, stem: Path, size, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    draw(ax, d)
    ax.set_xlabel("Residual point")
    ax.set_ylabel(ylabel)
    ps.save(fig, stem)
    plt.close(fig)


def legend_piece(handles, stem: Path) -> None:
    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, 0.3))
    fig.legend(handles=handles, loc="center", ncol=len(handles), **LEGEND_KW)
    ps.save(fig, stem)
    plt.close(fig)


# ── the printed table (the README carries it) ────────────────────────────────


def table() -> str:
    L = ["| run | point | PI | GS | IM | inverse map R² | MLP probe skill |", "|---|---|---|---|---|---|---|"]
    for lab, d in DATA.items():
        at = {ed: {p: b for p, b in d["best"][ed]} for ed in EDITORS}
        for p in range(d["n_points"]):
            cells = []
            for ed in EDITORS:
                b = at[ed].get(p)
                cells.append("" if b is None else
                             f"{b[d['key']]:+.3f} ({b['fidelity_ratio']:.2f}, α{b['alpha']:g}){'' if b['within_guard'] else '*'}")
            L.append(f"| {d['run']} | {p} | " + " | ".join(cells) + f" | {d['g_r2'][p]:.3f} | {d['skill'][p]:.3f} |")
    L.append("")
    L.append("Cell = Edit Index of the reported arm (fidelity ratio, step size α); * = that arm is outside the "
             "fidelity guard (no arm at that point has fidelity ratio ≤ 1). Blank = no arm at that point.")
    return "\n".join(L)


if __name__ == "__main__":
    PIECES.mkdir(exist_ok=True)
    compose(STANDARD, False, OUT / "by_point_A1")
    compose(STANDARD, True, OUT / "by_point_A2")
    compose(EXTRA, False, OUT / "by_point_A1_extra")
    compose(EXTRA, True, OUT / "by_point_A2_extra")
    compose(STANDARD + EXTRA, False, OUT / "by_point_A1_all")
    compose(STANDARD + EXTRA, True, OUT / "by_point_A2_all")
    for lab, d in DATA.items():
        piece(draw_ei, d, PIECES / f"A_ei_{SLUG[lab]}", (ps.HALF_WIDTH_IN - PAD, 1.9), "Edit Index")
        piece(draw_skill, d, PIECES / f"A_skill_{SLUG[lab]}", (ps.HALF_WIDTH_IN - PAD, 1.25), "Skill")
    legend_piece(keys(False, True), PIECES / "A1_legend")
    legend_piece(keys(True, True), PIECES / "A2_legend")
    md = table()
    print(md)
    (OUT / "by_point_values.md").write_text(md + "\n")
    (OUT / "by_point_values.json").write_text(json.dumps(
        {lab: {"run": d["run"], "edit_index_key": d["key"], "n_points": d["n_points"],
               "reported_arm": {ed: [{"point": p, **{k: b.get(k) for k in (d["key"], "fidelity_ratio", "alpha", "within_guard")}}
                                     for p, b in d["best"][ed]] for ed in EDITORS},
               "inverse_map_g_r2": d["g_r2"], "mlp_probe_skill": d["skill"]} for lab, d in DATA.items()}, indent=1))
