#!/usr/bin/env python3
"""Effective dimensionality of the residual stream per point, for the dropout arms.

For each run and residual point: harvest activations on the first N probe games (pim's
harvest_point), standardise per dimension exactly as inlp_othello.py does (canonical z-space),
and report the participation ratio PR = (tr Σ)² / tr(Σ²) of the covariance — the number of
"equally loaded" directions — plus the number of eigen-directions needed for 50 / 90 / 99 % of
the variance. Prints a table; writes scores/covariance_dim.json. Read-only on the runs.

    python experiments/dropout_ablation/scripts/covariance_dim.py "label=runs/topic/run" ... [--points 1 2 4 8] [--n-games 5000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.environments.othello import corpus as oc  # noqa: E402
from pim.environments.othello.data import canonical_vocab, harvest_point, tokens_and_labels  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402

DEV = "cuda"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--points", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--n-games", type=int, default=5000); a = ap.parse_args()
    out = {}
    print(f"{'run':20s} {'pt':>3s} {'PR':>7s} {'n50':>5s} {'n90':>5s} {'n99':>5s} {'top1 %':>7s}")
    for spec in a.runs:
        label, path = spec.split("=", 1); run_dir = REPO / path
        inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]; rules = oc.rules_of(inst)
        model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV); model.eval()
        itos = {v: k for k, v in canonical_vocab().items()}
        ptok, pln = oc.load(oc.build(only=("probe",), instance=inst, log=lambda s: None)["probe"])
        data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(ptok[:a.n_games], pln[:a.n_games])], **rules)
        out[label] = {}
        for p in a.points:
            R = harvest_point(model, data.tokens, p); X = R.reshape(-1, R.shape[-1])[data.mask.reshape(-1)]; del R
            xm, xs = X.mean(0), X.std(0); xs = np.maximum(xs, 1e-2 * np.median(xs)) + 1e-8
            Z = torch.tensor((X - xm) / xs, dtype=torch.float64, device=DEV); del X
            C = (Z.T @ Z) / (Z.shape[0] - 1); ev = torch.linalg.eigvalsh(C).flip(0).clamp_min(0); del Z, C
            pr = float(ev.sum() ** 2 / (ev ** 2).sum()); cum = torch.cumsum(ev, 0) / ev.sum()
            n = lambda q: int((cum < q).sum().item()) + 1
            out[label][p] = {"participation_ratio": pr, "n50": n(.5), "n90": n(.9), "n99": n(.99), "top1_share": float(ev[0] / ev.sum()), "d": int(ev.numel())}
            print(f"{label:20s} {p:3d} {pr:7.1f} {n(.5):5d} {n(.9):5d} {n(.99):5d} {100 * float(ev[0] / ev.sum()):7.1f}", flush=True)
        del model; torch.cuda.empty_cache()
    (REPO / "experiments/dropout_ablation/scores/covariance_dim.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
