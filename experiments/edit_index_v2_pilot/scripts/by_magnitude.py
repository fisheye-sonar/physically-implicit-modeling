#!/usr/bin/env python
"""Editability vs edit magnitude (tiles changed) on standard Othello, from the all-cases pilot
scores (scores/all_cases/L-oth-20m.json): per bin the mean Edit Index per editor under both
constructions, the guard, and the ceiling. Writes outputs/oth_uniform_by_magnitude.png."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[1]
d = json.load(open(EXP / "scores" / "all_cases" / "L-oth-20m.json"))
nc = np.array(d["per_case"]["n_changed"]); ceil = np.array(d["per_case"]["ceiling_v1"])
eds = [e for e in ("PI", "ND", "GS") if e in d["editors"]]
bins = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 99)]
labels = ["2", "3", "4", "5", "6", "7+"]
rows = []
print(f"standard Othello (L-oth-20m), {len(nc)} valid paired cases, canonical best arm per editor\n")
print(f"{'tiles':>5} {'n':>4} {'ceil_v1':>8} | " + " ".join(f"{e+'_v2':>7} {e+'_v1':>7} {'g2':>5} |" for e in eds))
for (lo, hi), lab in zip(bins, labels):
    m = (nc >= lo) & (nc <= hi)
    if m.sum() == 0:
        continue
    row = {"tiles": lab, "n": int(m.sum()), "ceiling_v1": float(np.nanmean(ceil[m]))}
    line = f"{lab:>5} {m.sum():>4} {row['ceiling_v1']:>+8.3f} | "
    for e in eds:
        v2 = np.array(d["editors"][e]["_per_case"]["ei_v2"])[m]; v1 = np.array(d["editors"][e]["_per_case"]["ei_v1"])[m]
        g2 = np.array(d["per_case"]["guard_v2_per_case"][e])[m]
        row[e] = {"ei_v2": float(np.nanmean(v2)), "ei_v1": float(np.nanmean(v1)), "guard_v2_median": float(np.median(g2))}
        line += f"{row[e]['ei_v2']:>+7.3f} {row[e]['ei_v1']:>+7.3f} {row[e]['guard_v2_median']:>5.2f} |"
    print(line); rows.append(row)
print("\n(g2 = median per-case guard, RMSE(edited, p_B)/RMSE(p_A, p_B); >1 degraded)")
(EXP / "scores" / "all_cases" / "L-oth-20m_by_magnitude.json").write_text(json.dumps(rows, indent=1))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
x = np.arange(len(rows))
for ax, key, title in zip(axes, ("ei_v2", "ei_v1"), ("model-referenced (v2)", "canonical references (v1), same cases")):
    for i, e in enumerate(eds):
        ax.plot(x, [r[e][key] for r in rows], marker="o", color=f"C{i}", label=e)
    if key == "ei_v1":
        ax.plot(x, [r["ceiling_v1"] for r in rows], color="k", ls="--", lw=1.2, label="ceiling (p_B)")
    ax.axhline(0, color="#bbb", lw=0.8); ax.axhline(1 if key == "ei_v2" else np.nan, color="k", ls=":", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{r['tiles']}\n(n={r['n']})" for r in rows])
    ax.set_ylim(-1.05, 1.05); ax.set_xlabel("tiles changed by the paired edit"); ax.set_title(title, loc="left", fontsize=10)
    ax.grid(True, axis="y", color="#eee"); ax.legend(fontsize=8, frameon=False, loc="lower left")
axes[0].set_ylabel("Edit Index (mean per bin)")
fig.suptitle("Standard Othello — editability vs edit magnitude, paired counterfactual edits, canonical single-flip arms (2026-09-11)", fontsize=10)
out = EXP / "outputs" / "oth_uniform_by_magnitude.png"; fig.savefig(out, dpi=130, bbox_inches="tight"); print(f"\nfigure -> {out.relative_to(EXP.parents[1])}")
