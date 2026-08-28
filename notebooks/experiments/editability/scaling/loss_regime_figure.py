"""Draw the data-volume regime figure from the training logs on disk.

Orchestrator script (CLAUDE.md invariant 1): it loads metrics, computes the power-law fit, and
hands arrays to `pim.figures.scaling.loss_curves`. Cheap and idempotent — rerun any time the
20M run has written more val passes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from pim.figures.scaling import loss_curves  # noqa: E402

# E[log |legal|] for the uniform-over-legal generator. Not a fitted quantity.
BAYES = 2.0092
FIT_FROM = 25_000          # drop the warmup-contaminated head before fitting

CURVES = [
    ("20M games (Transformer L)",   "runs/scaling/BIG20M_othello_L",        "#1f6f8b"),
    ("900k games (Transformer L)",  "runs/othello_arch/L90_theirs_othello", "#c2185b"),
]


def load(rel: str):
    rows = [json.loads(l) for l in (REPO / rel / "metrics.jsonl").open()]
    return (np.array([r["step"] for r in rows], float),
            np.array([r["val_loss"] for r in rows], float),
            np.array([r.get("train_loss", np.nan) for r in rows], float))


def main() -> None:
    curves = []
    fit = None
    for label, rel, color in CURVES:
        s, v, t = load(rel)
        curves.append({"label": label, "steps": s, "val": v, "train": t, "color": color})
        e = v - BAYES
        m = (s >= FIT_FROM) & (e > 0)
        b, a = np.polyfit(np.log(s[m]), np.log(e[m]), 1)
        resid = np.log(e[m]) - (a + b * np.log(s[m]))
        r2 = 1 - resid.var() / np.log(e[m]).var()
        print(f"{label:<32} n={m.sum():>3}  excess = {np.exp(a):8.3f}·step^{b:+.4f}   logR2 {r2:.4f}")
        if "20M" in label:
            fit = (float(np.exp(a)), float(b))
            fit_label = f"20M fit: {np.exp(a):.1f}·step$^{{{b:.3f}}}$  (log-log R²={r2:.4f})"

    fig = loss_curves(curves, BAYES, fit=fit, fit_label=fit_label,
                      suptitle="Othello, same architecture and optimiser — only the pool size differs")
    out = REPO / "runs" / "scaling" / "loss_regime.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
