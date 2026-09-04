#!/usr/bin/env python
"""Probe accuracy and confidence on squares the game no longer depends on.

Reads the relevance flags (`label_relevance.py`), harvests the residual stream of the
same games at every residual point, applies the run's cached canonical probes (LIN and
MLP-128 mine/theirs — a cache miss aborts, nothing is fitted) and records, per position
and square: argmax correctness, 3-class softmax entropy, and the probability on the true
class. Then, over OCCUPIED squares only (empties are trivially decodable from the
history), compares four disjoint categories:

    alive_frontier   the legal set depends on the disc; it touches an empty square
    alive_interior   the legal set depends on the disc; no empty neighbour
    irrelevant       flipping the disc leaves the legal set unchanged, but it lies on a
                     ray from some empty square (may matter later)
    dead             on no ray from any empty square — can never matter again

— by move number (never pooled across moves: dead squares live late, when everything is
harder), as paired within-position differences with bootstrap CIs, and stratified by the
controls (moves since the disc last changed colour, number of flips).

Outputs (scores/): probe_by_relevance_<run>.json (every aggregate, every point, both
families); per_square_pt<k>_<family>.npz for the focus points (canonical best per family
plus the late-game points 7 and 8); outputs/relevance_by_move_pt<k>.png.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pim.environments.othello import corpus as oc  # noqa: E402
from pim.environments.othello.data import (  # noqa: E402
    canonical_vocab, harvest_point, tokens_and_labels)
from pim.figures.theme import PALETTE  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402
from pim.probes.cache import ProbeCache  # noqa: E402

EXP = REPO / "experiments" / "othello_causal_state"
FAMS = ("linear", "mlp")
FAM_LABEL = {"linear": "LIN", "mlp": "MLP-128"}
# Occupied squares, disjoint, in order of decreasing relevance. "read" = on a gap-free run
# from some empty square (a legality/flip computation can traverse it); "alive" = read AND
# the legal set changes when it is flipped; "read_inert" = read but a single flip leaves the
# legal set unchanged (redundant brackets, or a run-interior disc) — the class the
# single-flip test wrongly called irrelevant; "dead" = on no ray at all (can never matter).
# NB "unread now" (on no gap-free run from any empty square) COINCIDES with dead — the
# nearest empty on any line always has a gap-free run to the disc — so it is not a separate
# category; the `traversable` flag survives only as the consistency check below.
CATS = ("alive_frontier", "alive_interior", "read_inert", "dead")
SINCE_BINS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 99)]
FLIP_BINS = [(0, 0), (1, 1), (2, 2), (3, 99)]
MOVE_BINS = [(1, 20), (21, 40), (41, 58)]


def probe_games(n: int):
    tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("probe",))["probe"])
    itos = {v: k for k, v in canonical_vocab().items()}
    return tokens_and_labels([[itos[int(t)] for t in row[:L]]
                              for row, L in zip(tok[:n], ln[:n])])


def cached_grid(model, data, cache_dir: Path, dev: str):
    store = ProbeCache(cache_dir)
    fname, prov = store.key(
        model, kind="othello_grid", targets=["mine"], families=["linear", "mlp"],
        splits=["sequence"], holdout=0.2, epochs=200, batch=4096, lr=1e-3, seed=0,
        n_seq=int(len(data.tokens)), n_rows=int(data.mask.sum()),
        n_points=model.n_layers + 1)
    blob = store.load(fname, prov, device=dev)
    if blob is None:
        sys.exit(f"no cached canonical probe grid under {cache_dir} ({fname}) — refusing to refit")
    print(f"probe grid cache HIT ({fname})", flush=True)
    return blob["probes"], fname


def categories(R) -> dict[str, np.ndarray]:
    mask = R["mask"][:, :, None]
    occ = R["occupied"].astype(bool) & mask
    dead, irr, trav, fr = (R[k].astype(bool) & mask
                           for k in ("dead", "irrelevant", "traversable", "frontier"))
    assert not (occ & ~trav & ~dead).any(), "an unread disc that is not dead: impossible"
    return {"dead": dead, "read_inert": occ & trav & irr,
            "alive_frontier": occ & trav & ~irr & fr, "alive_interior": occ & trav & ~irr & ~fr,
            "occupied": occ, "empty": mask & ~occ}


@torch.no_grad()
def probe_outputs(probe, acts: np.ndarray, mine: np.ndarray, dev: str, batch: int = 4096):
    """acts (G, T, d), mine (G, T, 64) -> correct (bool), entropy, p_true — all (G, T, 64)."""
    G, T, d = acts.shape
    X, Y = acts.reshape(-1, d), mine.reshape(-1, 64).astype(np.int64)
    correct = np.zeros((G * T, 64), bool)
    ent = np.zeros((G * T, 64), np.float32)
    ptrue = np.zeros((G * T, 64), np.float32)
    for i in range(0, G * T, batch):
        x = torch.from_numpy(X[i : i + batch]).to(dev)
        y = torch.from_numpy(Y[i : i + batch]).to(dev)
        p = torch.softmax(probe(x).float(), -1)                    # (B, 64, 3)
        correct[i : i + batch] = (p.argmax(-1) == y).cpu().numpy()
        ent[i : i + batch] = (-(p * torch.log(p.clamp_min(1e-12))).sum(-1)).cpu().numpy()
        ptrue[i : i + batch] = p.gather(-1, y[..., None])[..., 0].cpu().numpy()
    return correct.reshape(G, T, 64), ent.reshape(G, T, 64), ptrue.reshape(G, T, 64)


def by_move(stat: np.ndarray, cats: dict, names=CATS + ("occupied", "empty")) -> dict:
    out = {}
    for name in names:
        m = cats[name]
        n = m.sum((0, 2))
        s = np.where(m, stat, 0).sum((0, 2), dtype=np.float64)
        out[name] = {"n": n.tolist(),
                     "mean": [float(v) if k > 0 else None for v, k in zip(s / np.maximum(n, 1), n)]}
    return out


def paired(correct, ent, cats, a: str, b: str, seed: int = 0) -> dict:
    """Within-position mean(a) − mean(b) over positions holding both; bootstrap CI over positions."""
    A, B = cats[a], cats[b]
    has = A.any(2) & B.any(2)
    gs, ts = np.where(has)
    da, de, mv = [], [], []
    for g, t in zip(gs, ts):
        da.append(correct[g, t][A[g, t]].mean() - correct[g, t][B[g, t]].mean())
        de.append(ent[g, t][A[g, t]].mean() - ent[g, t][B[g, t]].mean())
        mv.append(t + 1)
    da, de, mv = np.array(da), np.array(de), np.array(mv)
    rng = np.random.default_rng(seed)

    def ci(v):
        if len(v) < 2:
            return [None, None]
        bs = np.array([rng.choice(v, len(v)).mean() for _ in range(1000)])
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    out = {"a": a, "b": b, "n_positions": int(len(da)),
           "acc_diff_mean": float(da.mean()) if len(da) else None, "acc_diff_ci95": ci(da),
           "entropy_diff_mean": float(de.mean()) if len(de) else None,
           "entropy_diff_ci95": ci(de), "by_move_bin": {}}
    for lo, hi in MOVE_BINS:
        m = (mv >= lo) & (mv <= hi)
        out["by_move_bin"][f"{lo}-{hi}"] = {
            "n_positions": int(m.sum()),
            "acc_diff_mean": float(da[m].mean()) if m.any() else None,
            "acc_diff_ci95": ci(da[m]), "entropy_diff_mean": float(de[m].mean()) if m.any() else None}
    return out


def stratified(correct, ent, cats, control: np.ndarray, bins, mask) -> dict:
    """Accuracy / entropy by category x control bin, per move bin — the confound check."""
    out = {}
    for lo_m, hi_m in MOVE_BINS:
        T = correct.shape[1]
        mmove = np.zeros_like(mask)
        mmove[:, lo_m - 1 : min(hi_m, T)] = True
        mmove &= mask
        blk = {}
        for name in CATS:
            row = {}
            for lo, hi in bins:
                m = cats[name] & mmove[:, :, None] & (control >= lo) & (control <= hi)
                n = int(m.sum())
                row[f"{lo}-{hi}"] = {"n": n, "acc": float(correct[m].mean()) if n else None,
                                     "entropy": float(ent[m].mean()) if n else None}
            blk[name] = row
        out[f"moves {lo_m}-{hi_m}"] = blk
    return out


def figure(res: dict, pt: int, out: Path) -> None:
    def hexc(i):
        return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in PALETTE[i])
    INK2, GRID = "#52514e", "#e1e0d9"
    COL = {"alive_frontier": hexc(0), "alive_interior": hexc(1), "read_inert": hexc(2),
           "dead": hexc(3)}
    MK = {"alive_frontier": "o", "alive_interior": "s", "read_inert": "^", "dead": "D"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), sharex=True,
                             gridspec_kw=dict(hspace=0.3, wspace=0.22))
    for i, fam in enumerate(FAMS):
        P = res["points"][f"{pt}|{fam}"]
        for j, (key, ylab) in enumerate((("acc_by_move", "argmax accuracy on occupied squares"),
                                         ("entropy_by_move", "mean 3-class entropy (nats)"))):
            ax = axes[i, j]
            for name in CATS:
                d = P[key][name]
                x = np.arange(1, len(d["mean"]) + 1)
                y = np.array([np.nan if v is None else v for v in d["mean"]])
                n = np.array(d["n"])
                ok = n >= 20                                    # hide bins with < 20 squares
                ax.plot(x[ok], y[ok], color=COL[name], lw=2 if name == "dead" else 1.5,
                        marker=MK[name], ms=3.6 if name == "dead" else 3, mec="white", mew=0.6,
                        label=f"{name.replace('_', ' ')}  (n = {int(n.sum()):,})")
            ax.set_title(f"{FAM_LABEL[fam]} · point {pt} — {ylab}", fontsize=10, loc="left", pad=6)
            ax.grid(True, color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=8, colors=INK2)
            for sp in ax.spines.values():
                sp.set_edgecolor("#c3c2b7")
            if j == 0:
                ax.set_ylim(0.5, 1.01)
            if i == 1:
                ax.set_xlabel("move number", fontsize=9, color=INK2)
            ax.legend(fontsize=7.5, frameon=False, loc="lower left" if j == 0 else "upper left")
    fig.suptitle(f"{res['run']} — board decodability by causal relevance of the square "
                 f"({res['n_games']:,} held-out games; bins with < 20 squares hidden)",
                 fontsize=11.5, y=0.99)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="runs/initial_othello_comparison/L-oth-20m")
    ap.add_argument("--relevance", default="scores/relevance_test5000.npz")
    ap.add_argument("--focus-extra", nargs="*", type=int, default=[7, 8])
    a = ap.parse_args()
    t0 = time.time()
    run_dir = (REPO / a.run).resolve()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    S = json.loads((run_dir / "scores.json").read_text())
    best = {fam: int(np.argmax(S["probe_skill"][f"mine|{fam}|sequence"])) for fam in FAMS}
    model, info = load_checkpoint(run_dir / "best_model.pt", device=dev)
    model.eval()
    (EXP / "outputs").mkdir(parents=True, exist_ok=True)

    R = np.load(EXP / a.relevance if not Path(a.relevance).is_absolute() else a.relevance)
    T = R["mask"].shape[1]                                   # the model sees 59 moves
    tokens, mask, mine = R["tokens"][:, :T].astype(np.int64), R["mask"], R["mine"]
    cats = categories(R)
    G = len(tokens)
    print(f"{G} games, {int(mask.sum())} positions; occupied {int(cats['occupied'].sum()):,}: "
          + ", ".join(f"{c} {int(cats[c].sum()):,}" for c in CATS), flush=True)
    probes, grid_file = cached_grid(model, probe_games(S["settings"]["oth_probe_games"]),
                                    run_dir / "probes", dev)
    n_points = model.n_layers + 1
    focus = sorted(set(best.values()) | set(a.focus_extra))
    res = {"run": run_dir.name, "arch": info.arch, "n_games": G, "positions": int(mask.sum()),
           "probe_grid": grid_file, "best_point": best, "focus_points": focus,
           "category_counts": {k: int(v.sum()) for k, v in cats.items()}, "points": {}}
    for pt in range(n_points):
        t1 = time.time()
        acts = harvest_point(model, tokens, pt)                   # (G, 59, d)
        for fam in FAMS:
            probe = probes[("mine", fam, "sequence", pt)].to(dev).eval()
            correct, ent, ptrue = probe_outputs(probe, acts, mine, dev)
            P = {"acc_by_move": by_move(correct, cats), "entropy_by_move": by_move(ent, cats),
                 "ptrue_by_move": by_move(ptrue, cats),
                 "paired": {f"{x}_vs_{y}": paired(correct, ent, cats, x, y) for x, y in
                            (("dead", "alive_interior"), ("read_inert", "alive_interior"),
                             ("dead", "read_inert"), ("alive_interior", "alive_frontier"))},
                 "controls": {"since_flip": stratified(correct, ent, cats, R["since_flip"],
                                                       SINCE_BINS, mask),
                              "n_flips": stratified(correct, ent, cats, R["n_flips"],
                                                    FLIP_BINS, mask)}}
            res["points"][f"{pt}|{fam}"] = P
            if pt in focus:
                np.savez_compressed(EXP / "scores" / f"per_square_pt{pt}_{fam}.npz",
                                    correct=correct, entropy=ent.astype(np.float16),
                                    p_true=ptrue.astype(np.float16))
            late = slice(39, 58)
            acc = {c: np.nanmean([v for v in P["acc_by_move"][c]["mean"][late] if v is not None])
                   for c in CATS}
            pd_ = P["paired"]["dead_vs_alive_interior"]
            print(f"point {pt} {FAM_LABEL[fam]:7s} acc moves 40-58: " +
                  "  ".join(f"{c} {acc[c]:.3f}" for c in CATS) +
                  f"  | paired dead−alive_interior {pd_['acc_diff_mean']:+.3f} "
                  f"CI {pd_['acc_diff_ci95']} (n {pd_['n_positions']})", flush=True)
        del acts
        if dev == "cuda":
            torch.cuda.empty_cache()
        print(f"  [{time.time() - t1:.0f}s]", flush=True)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    (EXP / "scores" / f"probe_by_relevance_{run_dir.name}.json").write_text(
        json.dumps(res, indent=1, default=float))
    for pt in focus:
        figure(res, pt, EXP / "outputs" / f"relevance_by_move_pt{pt}.png")
    print(f"done  [{res['minutes']} min]")


if __name__ == "__main__":
    main()
