#!/usr/bin/env python
"""Figures + validation table from the saved per-case arrays (no model, no GPU).

One figure per editor found under scores/<label>_arms_<editor>.npz:
  (a) Edit Index vs move number — the best arm AT each move (bold), the arm that is best
      over all cases pooled read at each move (thin; the honest line — a per-move argmax
      over 63–81 arms with ~256 cases carries roughly +0.05 of selection optimism),
      the unedited floor per move, and — with ``--compare li`` — the shipped benchmark's
      own per-move best as hollow markers at moves 5–30;
  (b) the fidelity ratio of those two arms per move (1.0 = doing nothing; > 1 degraded);
  (c) the winning residual point per move.
With ``--compare`` a validation table (moves both sets cover) is printed and saved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.figures.theme import PALETTE  # noqa: E402

EXP = REPO / "experiments" / "othello_edit_by_step"
INK2, GRID, REF, FAINT = "#52514e", "#e1e0d9", "#898781", "#b9b7ae"


def hexc(i: int) -> str:
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in PALETTE[i])


COLOR = {"PI": hexc(3), "ND": hexc(2), "GS": hexc(4)}       # Fig 1's entity colours
MARK = {"PI": "o", "ND": "s", "GS": "^"}


def load_summary(label: str, editor: str) -> dict | None:
    p = EXP / "scores" / f"{label}_summary_{editor}.json"
    return json.loads(p.read_text()) if p.exists() else None


def figure(label: str, editor: str, cmp: str | None, out: Path) -> None:
    s = load_summary(label, editor)
    c = COLOR[editor]
    by = s["by_move"]
    m = np.array([r["move"] for r in by])
    best = np.array([r["best"]["ei"] for r in by])
    glob = np.array([r["global"]["ei"] for r in by])
    floor = np.array([r["unedited_ei"] for r in by])
    fid_b = np.array([r["best"]["fid"] for r in by])
    fid_g = np.array([r["global"]["fid"] for r in by])
    pt = np.array([r["best"]["point"] for r in by])
    g = s["global_arm"]
    fig, (a, b, d) = plt.subplots(3, 1, figsize=(9.6, 7.4), sharex=True,
                                  gridspec_kw=dict(height_ratios=[4, 1.6, 1], hspace=0.14))
    a.plot(m, floor, color=REF, lw=1.2, label="unedited (floor at that move)")
    a.plot(m, glob, color=c, lw=1.3, ls="-", alpha=0.75,
           label=f"pooled-best arm, read per move (pt{g['point']} · α{g['alpha']:g})")
    a.plot(m, best, color=c, lw=2.2, marker=MARK[editor], ms=3.8, mec="white", mew=0.8,
           label=f"best arm at each move ({editor})", zorder=3)
    if cmp:
        sc = load_summary(cmp, editor)
        if sc:
            mc = [r["move"] for r in sc["by_move"]]
            a.plot(mc, [r["best"]["ei"] for r in sc["by_move"]], ls="none", marker=MARK[editor],
                   ms=6.5, mfc="white", mec=c, mew=1.4, zorder=4,
                   label=f"shipped benchmark ({cmp}), best arm at each move")
            a.plot(mc, [r["unedited_ei"] for r in sc["by_move"]], ls="none", marker="o",
                   ms=4.5, mfc="white", mec=REF, mew=1.2, label="shipped benchmark, unedited")
    a.axhline(0, color="#c3c2b7", lw=0.8)
    a.set_ylim(-1.0, 1.0)
    a.set_ylabel("Edit Index (union support)", fontsize=9, color=INK2)
    a.set_title(f"{s.get('run', 'L-oth-20m')} — {editor} editability by move number "
                f"({label} cases; n per move in the table)", fontsize=10.5, loc="left", pad=8)
    a.legend(fontsize=7.5, frameon=False, loc="center right")   # between curves and floor
    b.axhline(1.0, color=REF, lw=1.2, label="1.0 = doing nothing")
    b.plot(m, fid_g, color=c, lw=1.3, alpha=0.75, label="pooled-best arm")
    b.plot(m, fid_b, color=c, lw=2.0, marker=MARK[editor], ms=3.2, mec="white", mew=0.7,
           label="best arm at each move")
    b.set_ylim(0, max(1.5, float(np.nanmax(np.r_[fid_b, fid_g])) * 1.15))
    b.set_ylabel("fidelity ratio", fontsize=9, color=INK2)
    b.legend(fontsize=7.5, frameon=False, loc="upper right", ncol=3)
    d.step(m, pt, where="mid", color=c, lw=1.6)
    d.set_ylabel("best point", fontsize=9, color=INK2)
    d.set_yticks(range(0, int(pt.max()) + 1, 2))
    d.set_ylim(-0.5, int(pt.max()) + 0.5)
    d.set_xlabel("move number", fontsize=9, color=INK2)
    d.set_xticks(list(range(1, int(m.max()) + 1, 4)) + [int(m.max())])
    for ax in (a, b, d):
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8, colors=INK2)
        for sp in ax.spines.values():
            sp.set_edgecolor("#c3c2b7")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def validation(label: str, cmp: str, editor: str) -> str:
    s, sc = load_summary(label, editor), load_summary(cmp, editor)
    if not (s and sc):
        return ""
    A = {r["move"]: r for r in s["by_move"]}
    B = {r["move"]: r for r in sc["by_move"]}
    shared = sorted(set(A) & set(B))
    lines = [f"## {editor}: {label} vs {cmp}, moves both cover", "",
             f"| move | n {label} | n {cmp} | unedited {label} | unedited {cmp} | best "
             f"{label} | best {cmp} | Δ best |", "|---|---|---|---|---|---|---|---|"]
    d = []
    for t in shared:
        d.append(A[t]["best"]["ei"] - B[t]["best"]["ei"])
        lines.append(f"| {t} | {A[t]['n']} | {B[t]['n']} | {A[t]['unedited_ei']:+.3f} | "
                     f"{B[t]['unedited_ei']:+.3f} | {A[t]['best']['ei']:+.3f} | "
                     f"{B[t]['best']['ei']:+.3f} | {d[-1]:+.3f} |")
    lines += ["", f"Δ best over {len(shared)} moves: mean {np.mean(d):+.3f}, "
              f"sd {np.std(d):.3f}, max |Δ| {np.max(np.abs(d)):.3f}",
              f"pooled-best arm: {label} pt{s['global_arm']['point']} α{s['global_arm']['alpha']:g}"
              f" (EI {s['global_arm']['pooled_ei']:+.3f}) · {cmp} pt{sc['global_arm']['point']} "
              f"α{sc['global_arm']['alpha']:g} (EI {sc['global_arm']['pooled_ei']:+.3f})", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", default="synth")
    ap.add_argument("--editors", nargs="+", default=["PI", "ND", "GS"])
    ap.add_argument("--compare", default=None, help="a second label, e.g. li")
    a = ap.parse_args()
    (EXP / "outputs").mkdir(parents=True, exist_ok=True)
    report = []
    for ed in a.editors:
        if load_summary(a.label, ed) is None:
            continue
        out = EXP / "outputs" / f"edit_by_step_{ed}.png"
        figure(a.label, ed, a.compare, out)
        print("wrote", out)
        if a.compare:
            report.append(validation(a.label, a.compare, ed))
    if report:
        txt = "\n".join(report)
        (EXP / "outputs" / f"validation_{a.label}_vs_{a.compare}.md").write_text(txt)
        print(txt)


if __name__ == "__main__":
    main()
