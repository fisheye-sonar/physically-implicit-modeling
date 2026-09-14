#!/usr/bin/env python3
"""Compare INLP colour cascades across runs (the dropout ablation figure, N runs).

    python experiments/dropout_ablation/scripts/inlp_compare.py --tag dropout \
        "dropout 0.1, 780k=experiments/adjacent_flip_ablation/scores/inlp_othello_L-oth-adjacent-20m.json" \
        "dropout 0, 390k=experiments/dropout_ablation/scores/inlp_othello_L-oth-adjacent-nodrop-390k.json" ...

Reads the JSON written by experiments/adjacent_flip_ablation/scripts/inlp_othello.py (no recomputation —
numbers are quoted from those files), writes two figures to experiments/dropout_ablation/outputs/ and prints
the tables the figures are built from: copies per tile by residual point, guarded (fid <= 1.1) Edit Index by K.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "experiments/dropout_ablation/outputs"
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]   # fixed categorical order
MARKERS = ["o", "s", "D", "^", "v", "P"]
KS = [1, 2, 4, 8, 16, 32, 64, 128]


def guarded(d, point, K, mode="shrink", fid_max=1.1):
    arms = [r for r in d["edits"]["arms"] if r["point"] == point and r["K"] == K and r["mode"] == mode and r["fid"] <= fid_max]
    return max(arms, key=lambda r: r["ei"]) if arms else None


def best_guarded(d, fid_max=1.1):
    arms = [r for r in d["edits"]["arms"] if r["fid"] <= fid_max]
    return max(arms, key=lambda r: r["ei"]) if arms else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help='"label=path.json"')
    ap.add_argument("--tag", required=True, help="suffix for the output files")
    ap.add_argument("--panels", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--title", default="oth-adjacent, INLP colour cascades")
    a = ap.parse_args()
    runs = []
    for spec in a.runs:
        label, path = spec.split("=", 1)
        runs.append((label, json.loads((REPO / path).read_text())))
    pts = list(range(9))

    # ── table 1: copies per tile by point ─────────────────────────────────────
    print("copies per tile (deflations to exhaustion, mean over 64 tiles), residual points 0-8")
    for label, d in runs:
        print(f"  {label:28s} " + " ".join(f"{d['points'][str(p)]['k_exhaust_mean']:6.1f}" for p in pts))
    print("half-R² iteration (first deflation at which mean R² falls below half its initial value), points 1-8")
    for label, d in runs:
        print(f"  {label:28s} " + " ".join(f"{d['points'][str(p)]['k_half_mean_curve']:6d}" for p in pts[1:]))

    # ── table 2: guarded edit index by K, shrink, points 1 and 2; best guarded overall ─────
    for p in (1, 2):
        print(f"guarded (fid<=1.1) Edit Index by K, shrink, point {p}; unedited in brackets")
        for label, d in runs:
            cells = []
            for K in KS:
                g = guarded(d, p, K)
                cells.append(f"{g['ei']:+.2f}/{g['fid']:.2f}" if g else "   —     ")
            print(f"  {label:28s} [{d['edits']['unedited']:+.2f}] " + " ".join(f"K{K:<3d}{c}" for K, c in zip(KS, cells)))
    print("best guarded arm overall")
    for label, d in runs:
        b = best_guarded(d)
        print(f"  {label:28s} {b['ei']:+.3f} / fid {b['fid']:.2f}  (pt {b['point']}, {b['mode']}, K={b['K']}, α{b['alpha']:g})" if b else f"  {label}: none")

    # ── figure 1: copies by point + best guarded by point ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    for i, (label, d) in enumerate(runs):
        ax.plot(pts[1:], [d["points"][str(p)]["k_exhaust_mean"] for p in pts[1:]], marker=MARKERS[i], color=COLORS[i], lw=2, ms=6, label=label)
    ax.set_xlabel("residual point"); ax.set_ylabel("copies per tile (deflations to exhaustion, mean over 64 tiles)")
    ax.set_title("orthogonal colour copies by residual point", loc="left"); ax.set_ylim(0, None); ax.grid(alpha=.3); ax.legend(frameon=False)
    ax = axes[1]
    for i, (label, d) in enumerate(runs):
        ys = []
        for K in KS:
            gs = [guarded(d, p, K) for p in (1, 2, 3, 4, 5)]
            gs = [g for g in gs if g]
            ys.append(max(g["ei"] for g in gs) if gs else np.nan)
        ax.plot(KS, ys, marker=MARKERS[i], color=COLORS[i], lw=2, ms=6, label=label)
        ax.axhline(d["edits"]["unedited"], color=COLORS[i], lw=1, ls=":", alpha=.7)
    ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("K copies written at once (shrink mode)"); ax.set_ylabel("best guarded Edit Index over points 1–5 (fid ≤ 1.1)")
    ax.set_title("K-copy edits: how many copies must be written before the edit lands\n(dotted = unedited floor)", loc="left"); ax.grid(alpha=.3); ax.legend(frameon=False)
    fig.suptitle(a.title + " — copies and K-copy editability across runs", y=1.02)
    fig.tight_layout(); f1 = OUT / f"inlp_compare_copies_{a.tag}.png"; fig.savefig(f1, dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── figure 2: R² by iteration overlay, one panel per point ───────────────
    n = len(a.panels); fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5), sharey=True)
    for ax, p in zip(np.atleast_1d(axes), a.panels):
        for i, (label, d) in enumerate(runs):
            c = d["points"][str(p)]["mean_r2_curve"]; x = np.arange(1, len(c) + 1)
            ax.plot(x, c, color=COLORS[i], lw=2, label=label if p == a.panels[0] else None)
            rc = d["points"][str(p)].get("mean_r2_random_curve")
            if rc: ax.plot(np.arange(1, len(rc) + 1), rc, color=COLORS[i], lw=1, ls=":", alpha=.8, label=f"random-direction control, {label}" if p == a.panels[0] else None)
        ax.set_xscale("log", base=2); ax.set_xticks([1, 4, 16, 64, 256]); ax.set_xticklabels(["1", "4", "16", "64", "256"])
        ax.set_title(f"residual point {p}\ncopies per tile: " + " / ".join(f"{d['points'][str(p)]['k_exhaust_mean']:.0f}" for _, d in runs), loc="left", fontsize=10)
        ax.set_xlabel("orthogonal deflation iteration (log scale)"); ax.set_ylim(0, 1); ax.grid(alpha=.3)
    np.atleast_1d(axes)[0].set_ylabel("held-out R² of a tile's colour\n(±1, occupied rows; mean over 64 tiles)")
    fig.legend(loc="upper center", ncol=len(runs), frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=9)
    fig.suptitle(a.title + " (copies listed in run order)", y=1.1)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); f2 = OUT / f"inlp_compare_r2_{a.tag}.png"; fig.savefig(f2, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", f1.relative_to(REPO)); print("wrote", f2.relative_to(REPO))


if __name__ == "__main__":
    main()
