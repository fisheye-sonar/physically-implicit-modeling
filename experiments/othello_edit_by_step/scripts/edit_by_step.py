#!/usr/bin/env python
"""Othello editability BY MOVE NUMBER — the canonical editor sweeps, scored per case.

The master_eval Othello loop, verbatim (same cached probes, same arms, same α grids,
same scorecard), run on a case set and kept PER CASE, so the Edit Index and the fidelity
guard can be read by game position. Case sets: the shipped 1001 (``--cases li``, moves
5–30) and the synthesised full-game set from ``synth_cases.py`` (moves 1–59).

Nothing is fitted: the probes are the run's cached canonical grid (LIN for PI/ND, MLP-128
for GS), loaded with the scorer's own key — a cache miss aborts. The grid's file name
is recorded with every result.

Saved once per case set (scores/<label>_cases.npz + .json): move number, square, current
and target label, legal sets, unsteered probabilities, unedited per-case Edit Index.
Saved per editor (scores/<label>_arms_<editor>.npz + .json): every arm's per-case Edit
Index, fidelity ratio and Li error, with the arm list. A later editor (``--editors GS``,
the MLP-probe replication) reuses everything above and adds only its own arm file.

  edit_by_step.py --cases cases/synth_seed0_n256.pkl --label synth --editors PI ND
  edit_by_step.py --cases li --label li --editors PI ND
  edit_by_step.py --cases cases/synth_seed0_n256.pkl --label synth --editors GS   # later
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pim.environments.othello import arms as oa  # noqa: E402
from pim.environments.othello import corpus as oc  # noqa: E402
from pim.environments.othello.bench import (  # noqa: E402
    benchmark_from_cases, case_targets, load_benchmark)
from pim.environments.othello.data import canonical_vocab, tokens_and_labels  # noqa: E402
from pim.metrics.othello_moves import (  # noqa: E402
    move_fidelity_ratio, move_fidelity_ratio_per_case, move_scorecard)
from pim.models import load_checkpoint, n_points  # noqa: E402
from pim.probes.cache import ProbeCache  # noqa: E402

EXP = REPO / "experiments" / "othello_edit_by_step"
EDITOR_MODE = {"ND": "add_sub", "PI": "pinv"}       # the scorer's two linear-probe arms


def probe_games(n: int):
    """The scorer's `_probe_games`, verbatim: the first n games of the probe split."""
    tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("probe",))["probe"])
    itos = {v: k for k, v in canonical_vocab().items()}
    return tokens_and_labels([[itos[int(t)] for t in row[:L]]
                              for row, L in zip(tok[:n], ln[:n])])


def cached_grid(model, data, cache_dir: Path, dev: str):
    """The run's canonical probe grid under fit_probe_grid's own key; abort on a miss."""
    store = ProbeCache(cache_dir)
    fname, prov = store.key(
        model, kind="othello_grid", targets=["mine"], families=["linear", "mlp"],
        splits=["sequence"], holdout=0.2, epochs=200, batch=4096, lr=1e-3, seed=0,
        n_seq=int(len(data.tokens)), n_rows=int(data.mask.sum()),
        n_points=model.n_layers + 1)
    blob = store.load(fname, prov, device=dev)
    if blob is None:
        sys.exit(f"no cached canonical probe grid under {cache_dir} ({fname}) — "
                 "refusing to refit; score the run with master_eval first")
    print(f"probe grid cache HIT ({fname})", flush=True)
    return blob["probes"], fname


def move_numbers(bench) -> np.ndarray:
    m = np.zeros(bench.n_cases, int)
    for toks, ids in zip(bench.tokens, bench.case_ids):
        m[ids] = toks.shape[1]
    return m


def summarise(label: str, editor: str) -> dict:
    """Per move number: the unedited floor, the best arm AT that move, and the arm that is
    best over all cases pooled, read at that move — written beside the arrays."""
    C = np.load(EXP / "scores" / f"{label}_cases.npz")
    A = np.load(EXP / "scores" / f"{label}_arms_{editor}.npz")
    meta = json.loads((EXP / "scores" / f"{label}_arms_{editor}.json").read_text())
    move, u_ei = C["move"], C["unedited_ei"]
    ei, fid = A["ei"], A["fid"]                       # (n_arms, n_cases)
    pooled = np.nanmean(ei, 1)
    g = int(np.nanargmax(pooled))
    out = {"label": label, "editor": editor, "global_arm": {**meta["arms"][g], "index": g,
           "pooled_ei": float(pooled[g])}, "by_move": []}
    for t in sorted(set(move.tolist())):
        rows = np.where(move == t)[0]
        per_arm = np.nanmean(ei[:, rows], 1)
        b = int(np.nanargmax(per_arm))
        out["by_move"].append({
            "move": int(t), "n": int(len(rows)), "unedited_ei": float(np.nanmean(u_ei[rows])),
            "best": {**meta["arms"][b], "index": b, "ei": float(per_arm[b]),
                     "fid": float(np.nanmean(fid[b, rows]))},
            "global": {"index": g, "ei": float(per_arm[g]),
                       "fid": float(np.nanmean(fid[g, rows]))}})
    (EXP / "scores" / f"{label}_summary_{editor}.json").write_text(
        json.dumps(out, indent=1, default=float))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="runs/initial_othello_comparison/L-oth-20m")
    ap.add_argument("--cases", required=True, help="'li' or a cases .pkl path")
    ap.add_argument("--label", required=True, help="result-file prefix, e.g. synth / li")
    ap.add_argument("--editors", nargs="+", default=["PI", "ND"],
                    choices=["PI", "ND", "GS"])
    a = ap.parse_args()
    t0 = time.time()
    run_dir = (REPO / a.run).resolve()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    S = json.loads((run_dir / "scores.json").read_text())["settings"]     # canonical grids
    model, info = load_checkpoint(run_dir / "best_model.pt", device=dev)
    model.eval()
    (EXP / "scores").mkdir(parents=True, exist_ok=True)

    # cases -> Benchmark, through the SAME construction as the shipped 1001
    if a.cases == "li":
        bench, src = load_benchmark(), "vendor/intervention_benchmark.pkl"
    else:
        p = (REPO / a.cases).resolve() if not Path(a.cases).is_absolute() else Path(a.cases)
        with open(p, "rb") as f:
            raw = pickle.load(f)
        bench, src = benchmark_from_cases(raw), str(p.relative_to(REPO))
        src += f" (md5 {hashlib.md5(p.read_bytes()).hexdigest()[:12]})"
    cur, tgt = case_targets(bench)
    move = move_numbers(bench)
    print(f"{a.label}: {bench.n_cases} cases, moves {move.min()}–{move.max()}, "
          f"{len(bench.tokens)} length buckets  [{time.time() - t0:.0f}s]", flush=True)

    data = probe_games(S["oth_probe_games"])
    probes, grid_file = cached_grid(model, data, run_dir / "probes", dev)
    npnt = n_points(model)
    lin_mine = {p: probes[("mine", "linear", "sequence", p)] for p in range(npnt)}
    mlp_mine = {p: probes[("mine", "mlp", "sequence", p)] for p in range(npnt)}

    # the unedited baseline, once per case set (the guard's denominator lives here)
    uns_probs = oa.unsteered_probs(model, bench)
    u = move_scorecard(uns_probs, bench.legal_pre, bench.legal_post)
    cases_npz = EXP / "scores" / f"{a.label}_cases.npz"
    np.savez_compressed(cases_npz, move=move, pos_int=bench.pos_int,
                        new_class=bench.new_class, cur_lab=cur, tgt_lab=tgt,
                        unsteered_probs=uns_probs.astype(np.float32),
                        unedited_ei=np.array(u["edit_index_union_per_case"], np.float32),
                        unedited_li_post=np.array(u["li_error_vs_post_per_case"], np.float32))
    (EXP / "scores" / f"{a.label}_cases.json").write_text(json.dumps({
        "label": a.label, "run": run_dir.name, "arch": info.arch, "source": src,
        "n_cases": int(bench.n_cases), "legal_pre": bench.legal_pre,
        "legal_post": bench.legal_post, "probe_grid": grid_file,
        "unedited": {k: v for k, v in u.items() if isinstance(v, (int, float))}},
        indent=None))
    print(f"unedited: EI {u['edit_index_union']:+.3f}  (n {u['n_scored']})", flush=True)

    for editor in a.editors:
        t1, arms, EI, FID, LI = time.time(), [], [], [], []
        if editor in EDITOR_MODE:
            mode, alphas = EDITOR_MODE[editor], S["oth_alpha_pi" if editor == "PI" else "oth_alpha_nd"]
            grid = [(ell, al) for ell in range(npnt) for al in alphas]
        else:
            grid = [(ls, al) for ls in S["oth_gs_layers"] for al in S["oth_alpha_gs"]]
        for ell, al in grid:
            if editor in EDITOR_MODE:
                pr, card = oa.linear_arm(model, bench, lin_mine, tgt, cur, mode=mode,
                                         alpha=al, points={ell})
            else:
                pr, card = oa.grad_steer_arm(model, bench, mlp_mine, ell, alpha=al,
                                             n_steps=S["oth_gs_steps"], beta=S["oth_gs_beta"],
                                             target_labels=tgt)
            EI.append(card["edit_index_union_per_case"])
            LI.append(card["li_error_vs_post_per_case"])
            FID.append(move_fidelity_ratio_per_case(pr, uns_probs, bench.legal_post))
            arms.append({"editor": editor, "point": int(ell), "alpha": float(al),
                         "write_ratio": card.get("write_ratio"),
                         "edit_index_union": card["edit_index_union"],
                         "fidelity_ratio": move_fidelity_ratio(pr, uns_probs, bench.legal_post),
                         "li_error_vs_post": card["li_error_vs_post"]})
            print(f"  {editor} pt{ell} α{al:g}: EI {card['edit_index_union']:+.3f}  "
                  f"fid {arms[-1]['fidelity_ratio']:.2f}", flush=True)
        np.savez_compressed(EXP / "scores" / f"{a.label}_arms_{editor}.npz",
                            ei=np.array(EI, np.float32), fid=np.array(FID, np.float32),
                            li_post=np.array(LI, np.float32))
        (EXP / "scores" / f"{a.label}_arms_{editor}.json").write_text(json.dumps({
            "label": a.label, "editor": editor, "run": run_dir.name, "probe_grid": grid_file,
            "probe_family": "mlp" if editor == "GS" else "linear",
            "settings": {k: v for k, v in S.items() if k.startswith("oth_")},
            "arms": arms, "minutes": round((time.time() - t1) / 60, 1)}, indent=1))
        s = summarise(a.label, editor)
        g = s["global_arm"]
        print(f"{editor}: {len(arms)} arms in {(time.time() - t1) / 60:.1f} min · pooled best "
              f"pt{g['point']} α{g['alpha']:g} EI {g['pooled_ei']:+.3f}", flush=True)
        for r in s["by_move"]:
            print(f"   move {r['move']:2d} n={r['n']:4d}  unedited {r['unedited_ei']:+.3f}  "
                  f"best pt{r['best']['point']} α{r['best']['alpha']:g} EI {r['best']['ei']:+.3f}"
                  f" fid {r['best']['fid']:.2f}  | global EI {r['global']['ei']:+.3f} "
                  f"fid {r['global']['fid']:.2f}")
    print(f"done  {a.label} {a.editors}  [{(time.time() - t0) / 60:.1f} min]")


if __name__ == "__main__":
    main()
