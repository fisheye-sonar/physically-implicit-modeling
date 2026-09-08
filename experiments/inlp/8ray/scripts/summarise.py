#!/usr/bin/env python
"""inlp_8ray -> summary.md + perdim_profile.png from the two result JSONs (no model, no GPU).

Per basis and residual point: cascade size, rank, aggregate R² of probes 1/2/4/8, the
per-component R² of probe 1 (must match Table 1b's row) and of the 8th probe, and the
best multi-probe edit by K (Edit Index / fidelity) against the canonical single-probe
arm. The figure shows, for one mid-stack point per basis, how each state component's R²
decays along the cascade — the by-component size of the linear code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[2]
sys.path.insert(0, str(REPO))
from pim.figures.theme import PALETTE  # noqa: E402

COMP = ("o1·x", "o1·y", "o2·x", "o2·y", "o1·vx", "o1·vy", "o2·vx", "o2·vy")
RUN = "L-dw-8ray-20m"


def hexc(i):
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in PALETTE[i % len(PALETTE)])


def main() -> None:
    res = {b: json.loads((EXP / "scores" / f"inlp_{RUN}_{b}.json").read_text())
           for b in ("frustum", "cartesian") if (EXP / "scores" / f"inlp_{RUN}_{b}.json").exists()}
    canon = json.loads((REPO / "runs" / "ray_ablation" / RUN / "scores.json").read_text())
    lines = [f"# INLP on {RUN} — summary", ""]
    for basis, r in res.items():
        T = canon["bases"][basis]
        pi = T["best"]["PI"]
        lines += [f"## {basis}  (unedited {r['unedited']['edit_index']:+.3f}; canonical PI best "
                  f"{pi['edit_index']:+.3f} / fid {pi['fidelity_ratio']:.2f} at pt{pi['point']} α{pi['alpha']:g})", "",
                  "| point | probes | rank | R² probe 1 / 2 / 4 / 8 | probe-1 per-component R² | "
                  "best K arm: EI / fid (α) | K=1 arm | all-K arm | wiring |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for p in sorted(r["points"], key=int):
            P = r["points"][p]
            prof = P["r2_profile"]
            def pick(i, prof=prof):
                return f"{prof[i]:.3f}" if i < len(prof) else "—"

            best_all = max(P["best"].values(), key=lambda a: a["edit_index"])
            k1 = P["best"]["K1u"]
            kall = P["best"][sorted(P["best"], key=lambda k: int(k[1:-1]))[-1]]
            pd1 = " ".join(f"{v:.2f}" for v in P["perdim_profile"][0])
            lines.append(f"| {p} | {P['n_probes']} | {P['total_rank']} | {pick(0)} / {pick(1)} / {pick(3)} / {pick(7)} | "
                         f"{pd1} | K{best_all['K']}{'s' if best_all['shrink'] else 'u'}: {best_all['edit_index']:+.3f} / "
                         f"{best_all['fidelity_ratio']:.2f} (α{best_all['alpha']:g}) | {k1['edit_index']:+.3f} / {k1['fidelity_ratio']:.2f} | "
                         f"K{kall['K']}: {kall['edit_index']:+.3f} / {kall['fidelity_ratio']:.2f} | {P['k1_vs_canonical_pi_reldiff']:.3f} |")
        lines.append("")
    (EXP / "scores" / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))

    # figure: per-component R² along the cascade, one mid-stack point per basis
    (EXP / "outputs").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, len(res), figsize=(6.4 * len(res), 4.4), squeeze=False)
    for ax, (basis, r) in zip(axes[0], res.items()):
        pt = "4"
        prof = np.array(r["points"][pt]["perdim_profile"])          # (n_probes, 8)
        for j, c in enumerate(COMP):
            ax.plot(np.arange(1, len(prof) + 1), prof[:, j], color=hexc(j), lw=1.8,
                    ls="-" if j < 4 else "--", marker="o" if j < 4 else "s", ms=3, label=c)
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        ax.set_title(f"{basis} · point {pt}", fontsize=10, loc="left")
        ax.set_xlabel("probe k in the nullspace cascade", fontsize=9)
        ax.set_ylabel("R² (train-mean baseline)", fontsize=9)
        ax.set_ylim(-0.1, 1.0)
        ax.grid(True, color="#e1e0d9", lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7.5, frameon=False, ncol=2)
    fig.suptitle(f"{RUN} — per-component held-out R² of the k-th orthogonal probe in the "
                 "nullspace cascade (solid: position, dashed: velocity)", fontsize=10.5, y=1.02)
    fig.savefig(EXP / "outputs" / "perdim_profile.png", dpi=170, bbox_inches="tight")
    print("wrote", EXP / "outputs" / "perdim_profile.png")


if __name__ == "__main__":
    main()
