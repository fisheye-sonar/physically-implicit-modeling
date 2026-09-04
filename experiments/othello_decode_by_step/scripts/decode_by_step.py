#!/usr/bin/env python
"""Othello decodability BY MOVE NUMBER — the canonical probes, read out per game step.

Question (Sevan, 2026-09-02): how does Probe Skill vary with where we are in the game?
For L-oth-20m, the canonical LIN and MLP-128 probes (the run's cached grid — NOTHING is
fitted here; a cache miss aborts rather than refits) are evaluated on the held-out games
one position at a time, at every residual point, and the best point per step is
reported — the same "best point" rule Table 1 uses, applied per move number.

Definitions (all the canonical ones, restricted to one step):
  * error_t      tile error rate of the probe's argmax read-out on held-out rows at
                 position t (the SAME seeded 80/20 split by game the grid was fitted with)
  * majority_t   the trivial baseline at position t: the majority class over all tiles of
                 the TRAIN rows at that position (the canonical majority baseline is the
                 same count pooled over positions — kept in the output as `majority_all`)
  * skill_t      1 − error_t / majority_t          (Probe Skill, per-step baseline)
  * skill_all_t  1 − error_t / majority_all        (same errors against the pooled baseline)
Move number = position index + 1 (the model sees the first 59 of 60 moves, so 1…59).

Outputs (experiments/othello_decode_by_step/): scores/decode_by_step_<run>.json and
outputs/decode_by_step_{linear,mlp}.png — one figure per probe family: skill against
move number for every residual point (faint) and the best point per step (bold), with the
best point's index underneath.
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

from pim.environments.othello import arms as oa  # noqa: E402
from pim.environments.othello import corpus as oc  # noqa: E402
from pim.environments.othello.data import (  # noqa: E402
    N_CLASSES, canonical_vocab, harvest_point, tokens_and_labels)
from pim.figures.theme import PALETTE  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402
from pim.probes.cache import ProbeCache  # noqa: E402
from pim.probes.mlp import CANONICAL_HIDDEN  # noqa: E402

FAMILIES = ("linear", "mlp")
FAM_LABEL = {"linear": "LIN", "mlp": f"MLP-{CANONICAL_HIDDEN}"}


def probe_games(n: int):
    """The scorer's `_probe_games`, verbatim: the first n games of the probe split."""
    tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("probe",))["probe"])
    itos = {v: k for k, v in canonical_vocab().items()}
    return tokens_and_labels([[itos[int(t)] for t in row[:L]]
                              for row, L in zip(tok[:n], ln[:n])])


def cached_grid(model, data, cache_dir: Path, dev: str):
    """The run's canonical probe grid, loaded with fit_probe_grid's own key — and a hard
    stop on a miss: refitting is a 20 GB job and never part of this experiment."""
    store = ProbeCache(cache_dir)
    fname, prov = store.key(
        model, kind="othello_grid", targets=["mine"], families=list(FAMILIES),
        splits=["sequence"], holdout=0.2, epochs=200, batch=4096, lr=1e-3, seed=0,
        n_seq=int(len(data.tokens)), n_rows=int(data.mask.sum()),
        n_points=model.n_layers + 1)
    blob = store.load(fname, prov, device=dev)
    if blob is None:
        sys.exit(f"no cached canonical probe grid under {cache_dir} ({fname}) — "
                 "refusing to refit; score the run with master_eval first")
    print(f"probe grid cache HIT ({fname})", flush=True)
    return blob["probes"]


@torch.no_grad()
def per_step_errors(probe, acts: np.ndarray, mine: np.ndarray, mask: np.ndarray,
                    dev: str, batch: int = 4096) -> np.ndarray:
    """(T,) tile error rate in % of one probe's argmax read-out at every position."""
    T = acts.shape[1]
    err = np.full(T, np.nan)
    for t in range(T):
        rows = np.where(mask[:, t])[0]
        if not len(rows):
            continue
        wrong, n = 0, 0
        for i in range(0, len(rows), batch):
            r = rows[i : i + batch]
            x = torch.from_numpy(acts[r, t]).to(dev)
            hat = probe(x).argmax(-1).cpu().numpy()          # (B, 64)
            y = mine[r, t].astype(np.int64)
            wrong += int((hat != y).sum())
            n += hat.size
        err[t] = 100.0 * wrong / n
    return err


def majority_errors(mine: np.ndarray, mask: np.ndarray):
    """Per-step and pooled majority-class error (%) over all tiles of the given rows."""
    T = mine.shape[1]
    per = np.full(T, np.nan)
    for t in range(T):
        lab = mine[mask[:, t], t].reshape(-1)
        if len(lab):
            per[t] = 100.0 * (1.0 - np.bincount(lab, minlength=N_CLASSES).max() / len(lab))
    lab = mine[mask].reshape(-1)
    pooled = 100.0 * (1.0 - np.bincount(lab, minlength=N_CLASSES).max() / len(lab))
    return per, pooled


def figure(res: dict, fam: str, out: Path):
    def hexc(i):
        return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in PALETTE[i])

    INK2, GRID, FAINT = "#52514e", "#e1e0d9", "#b9b7ae"
    col = hexc(0) if fam == "linear" else hexc(1)
    F = res["families"][fam]
    moves = np.array(res["move_number"])
    skill = np.array(F["skill"])                    # (n_points, T)
    best = np.array(F["best_skill"])
    best_pt = np.array(F["best_point"])
    fig, (a, b) = plt.subplots(2, 1, figsize=(9.0, 5.6), sharex=True,
                               gridspec_kw=dict(height_ratios=[4, 1], hspace=0.12))
    for p in range(skill.shape[0]):
        a.plot(moves, skill[p], color=FAINT, lw=1.0, zorder=1,
               label="every residual point" if p == 0 else "_nolegend_")
    a.plot(moves, best, color=col, lw=2.2, marker="o", ms=3.5, mec="white", mew=0.8,
           zorder=3, label=f"best point per step ({FAM_LABEL[fam]})")
    a.axhline(0, color="#c3c2b7", lw=0.8)
    a.set_ylim(min(-0.05, float(np.nanmin(skill)) - 0.02), 1.02)
    a.set_ylabel("Probe Skill  (1 − err / majority err, per step)", fontsize=9, color=INK2)
    a.set_title(f"{res['run']} — {FAM_LABEL[fam]} decodability of the board (mine/theirs) "
                f"by move number, held-out games", fontsize=10.5, loc="left", pad=8)
    a.legend(fontsize=8, frameon=False, loc="lower right")
    b.step(moves, best_pt, where="mid", color=col, lw=1.6)
    b.set_ylabel("best point", fontsize=9, color=INK2)
    b.set_yticks(range(0, skill.shape[0], 2))
    b.set_ylim(-0.5, skill.shape[0] - 0.5)
    b.set_xlabel("move number", fontsize=9, color=INK2)
    b.set_xticks(list(range(1, int(moves.max()) + 1, 4)) + [int(moves.max())])
    for ax in (a, b):
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8, colors=INK2)
        for sp in ax.spines.values():
            sp.set_edgecolor("#c3c2b7")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="runs/initial_othello_comparison/L-oth-20m")
    ap.add_argument("--n-games", type=int, default=20_000, help="probe corpus (canonical 20k)")
    a = ap.parse_args()
    t0 = time.time()
    run_dir = (REPO / a.run).resolve()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, info = load_checkpoint(run_dir / "best_model.pt", device=dev)
    model.eval()
    exp = REPO / "experiments" / "othello_decode_by_step"
    (exp / "scores").mkdir(parents=True, exist_ok=True)
    (exp / "outputs").mkdir(parents=True, exist_ok=True)

    data = probe_games(a.n_games)
    print(f"probe data: {len(data.tokens)} games, {int(data.mask.sum())} rows "
          f"[{time.time() - t0:.0f}s]", flush=True)
    probes = cached_grid(model, data, run_dir / "probes", dev)

    # the grid's own split, by game: evaluate on the held-out games only
    N, T = data.tokens.shape
    seq_of_row = np.repeat(np.arange(N)[:, None], T, 1)[data.mask]
    tr, te = oa._split(N, seq_of_row, "sequence", 0.2, 0)
    te_games = np.unique(seq_of_row[te])
    tr_games = np.unique(seq_of_row[tr])
    assert not set(te_games) & set(tr_games)
    mask_te, mine_te = data.mask[te_games], data.mine[te_games]
    maj_t, maj_all = majority_errors(data.mine[tr_games], data.mask[tr_games])
    n_rows_t = mask_te.sum(0)
    print(f"held-out games {len(te_games)} · train games {len(tr_games)} · pooled majority "
          f"err {maj_all:.2f}% · per-step majority err {np.nanmin(maj_t):.1f}–"
          f"{np.nanmax(maj_t):.1f}%", flush=True)

    n_points = model.n_layers + 1
    err = {fam: np.full((n_points, T), np.nan) for fam in FAMILIES}
    for point in range(n_points):
        t1 = time.time()
        acts = harvest_point(model, data.tokens[te_games], point)   # (n_te, 59, d)
        for fam in FAMILIES:
            pr = probes[("mine", fam, "sequence", point)].to(dev).eval()
            err[fam][point] = per_step_errors(pr, acts, mine_te, mask_te, dev)
        del acts
        if dev == "cuda":
            torch.cuda.empty_cache()
        print(f"point {point}: " + "  ".join(
            f"{FAM_LABEL[f]} err {np.nanmean(err[f][point]):.2f}% "
            f"(move 1 {err[f][point][0]:.1f}%, move {T} {err[f][point][T - 1]:.1f}%)"
            for f in FAMILIES) + f"  [{time.time() - t1:.0f}s]", flush=True)

    res = {"run": run_dir.name, "arch": info.arch, "instance": "oth-uniform",
           "n_games": int(N), "held_out_games": int(len(te_games)),
           "move_number": [t + 1 for t in range(T)], "n_rows_per_step": n_rows_t.tolist(),
           "majority_err_per_step": maj_t.tolist(), "majority_err_pooled": float(maj_all),
           "families": {}}
    for fam in FAMILIES:
        e = err[fam]
        skill = 1.0 - e / maj_t[None, :]
        skill_all = 1.0 - e / maj_all
        best_pt = np.nanargmax(skill, axis=0)
        res["families"][fam] = {
            "error_pct": e.tolist(), "skill": skill.tolist(), "skill_pooled_baseline":
            skill_all.tolist(), "best_point": best_pt.tolist(),
            "best_skill": skill[best_pt, np.arange(T)].tolist(),
            "best_skill_pooled_baseline": skill_all[best_pt, np.arange(T)].tolist(),
            "pooled_error_pct_per_point": np.nansum(e * n_rows_t, 1).__truediv__(
                n_rows_t.sum()).tolist()}
        figure(res, fam, exp / "outputs" / f"decode_by_step_{fam}.png")
    res["minutes"] = round((time.time() - t0) / 60, 1)
    out = exp / "scores" / f"decode_by_step_{run_dir.name}.json"
    out.write_text(json.dumps(res, indent=1, default=float))

    print("\nmove  " + "  ".join(f"{FAM_LABEL[f]:>22s}" for f in FAMILIES)
          + "      (best skill · point · err%)")
    for t in range(T):
        print(f"{t + 1:>4d}  " + "  ".join(
            f"{res['families'][f]['best_skill'][t]:+.3f} · pt{res['families'][f]['best_point'][t]}"
            f" · {err[f][res['families'][f]['best_point'][t], t]:5.1f}%".rjust(22)
            for f in FAMILIES) + f"   majority {maj_t[t]:.1f}%  n={n_rows_t[t]}")
    print(f"\ndone  {out}  [{res['minutes']} min]")


if __name__ == "__main__":
    main()
