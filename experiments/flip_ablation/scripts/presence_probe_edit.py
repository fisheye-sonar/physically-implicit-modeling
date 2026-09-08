#!/usr/bin/env python
"""A DEDICATED binary presence probe (occupied vs empty per square), fitted fresh, and PI edits
through it — the clean version of the presence pilot (whose probe was the 3-way colour probe's
blank class, entangled with colour).

Per model (no-flip trained / flip trained / random-init no-flip) and residual point: harvest the
run's canonical 20k probe games at that point, fit the canonical classification linear probe
with 2 classes per tile (`pim.probes.base.fit_probe(n_classes=2)`, held out by sequence, seed 0),
persist it (ProbeCache under experiments/flip_ablation/probes), report its held-out error, then
PI-edit the presence bench (remove a disc / add a parity-consistent disc): the probe's
occupied/blank logits at the tile are swapped and the residual re-solved in z-space
(`pim.editors.pinv.inject_state`), exactly `arms.linear_arm`'s pinv branch with 2 classes.
Scored with the canonical scorecard and guard. Output: scores/presence_probe_<model>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from presence_pilot import bench_from, presence_cases  # noqa: E402

from pim.editors.pinv import inject_state  # noqa: E402
from pim.environments.othello import arms as oa  # noqa: E402
from pim.environments.othello import corpus as oc  # noqa: E402
from pim.environments.othello.bench import shipped_length_distribution  # noqa: E402
from pim.environments.othello.data import (BLANK, N_TILES, canonical_vocab, harvest_point,  # noqa: E402
                                           board_probs, tokens_and_labels)
from pim.metrics.set_editability import move_fidelity_ratio, move_scorecard  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402
from pim.probes.base import fit_probe  # noqa: E402
from pim.probes.cache import ProbeCache  # noqa: E402

EXP = REPO / "experiments" / "flip_ablation"
DEV = "cuda"
OCC = 1                                   # presence classes: 0 = empty, 1 = occupied


def fit_presence_probe(model, data, point, store, log=print):
    fname, prov = store.key(model, kind="presence_probe", point=int(point), n_seq=int(len(data.tokens)),
                            n_rows=int(data.mask.sum()), holdout=0.2, split="sequence", seed=0, family="linear")
    hit = store.load(fname, prov, device=DEV)
    if hit is not None:
        return hit
    R = harvest_point(model, data.tokens, point)                       # residuals at one point
    R = R.reshape(-1, R.shape[-1]) if R.ndim == 3 else R
    seq_of_row = np.repeat(np.arange(len(data.tokens)), data.tokens.shape[1])[data.mask.reshape(-1)]
    X = R[data.mask.reshape(-1)] if len(R) == data.mask.size else R
    y = (data.mine[data.mask] != BLANK).astype(np.int64)             # (rows, 64) presence
    tr, te = oa._split(len(data.tokens), seq_of_row, "sequence", 0.2, 0)
    probe, st = fit_probe(X[tr], y[tr], X[te], y[te], hidden=None, n_classes=2, device=DEV, seed=0)
    store.store(fname, prov, (probe, st))
    log(f"    point {point}: presence error {st['error_rate']:.3f}% (in-sample {st['error_rate_insample']:.3f}%, majority {st['majority_class_error_rate']:.1f}%)  [fit]")
    return probe, st


@torch.no_grad()
def presence_pinv_arm(model, bench, probe, point, alpha):
    """PI through the 2-class presence probe: swap the tile's (empty, occupied) logits, re-solve."""
    A = probe.net.weight.detach()
    Ap, bv = torch.linalg.pinv(A), probe.net.bias.detach()
    probs = np.zeros((len(bench.pos_int), N_TILES), np.float32)
    landed, ratios = [], []
    for toks, ids in zip(bench.tokens, bench.case_ids):
        idx = torch.from_numpy(toks).to(DEV)
        bsz = len(ids)
        sq = torch.from_numpy(bench.pos_int[ids]).to(DEV)
        want_occ = torch.from_numpy((bench.tgt_lab[ids] != BLANK).astype(np.int64)).to(DEV)   # target presence
        rec = {}

        def hook(layer, x, _rec=rec):
            if layer != point:
                return x
            cur = x[:, -1]
            z = (cur - probe.x_mean) / probe.x_std
            lg = probe.net(z).view(bsz, N_TILES, 2).clone()
            sel = lg[torch.arange(bsz), sq]
            new = sel.flip(-1)                                          # swap empty <-> occupied
            lg[torch.arange(bsz), sq] = new
            z_new = inject_state(z, lg.view(bsz, -1), A, Ap, bv)
            delta = alpha * (z_new - z) * probe.x_std
            out = x.clone()
            out[:, -1] = cur + delta
            _rec["ratio"] = float((delta.norm(dim=1) / cur.norm(dim=1)).mean())
            with torch.no_grad():                                      # did the read-out land?
                lg2 = probe(out[:, -1]).view(bsz, N_TILES, 2)
                _rec["landed"] = float((lg2[torch.arange(bsz), sq].argmax(-1) == want_occ).float().mean())
            return out

        probs[ids] = board_probs(model.decode(idx, edit=hook), getattr(model, "output_kind", "logits"))
        ratios.append(rec.get("ratio", 0.0))
        landed.append(rec.get("landed", float("nan")))
    card = move_scorecard(probs, bench.legal_pre, bench.legal_post)
    card["write_ratio"] = float(np.mean(ratios))
    card["readout_landed"] = float(np.nanmean(landed))
    return probs, card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/flip_ablation/L-oth-noflip-20m")
    ap.add_argument("--instance", default="oth-noflip")
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--points", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0, 5.0])
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()
    t0 = time.time()
    flip = oc.flip_of(args.instance)
    if args.random_init:
        from pim.probes.baselines import random_init_model
        model = random_init_model("transformer_l_tokens", {"vocab": 61, "block_size": 59}, seed=0, device=DEV).eval()
        label = f"random-init on {args.instance}"
    else:
        model, _ = load_checkpoint(REPO / args.run / "best_model.pt", device=DEV)
        model.eval()
        label = Path(args.run).name
    print("===", label, "===", flush=True)
    itos = {v: k for k, v in canonical_vocab().items()}
    ptok, pln = oc.load(oc.build(only=("probe",), instance=args.instance, log=lambda s: None)["probe"])
    data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(ptok[:20_000], pln[:20_000])], flip=flip)
    store = ProbeCache(EXP / "probes")
    tok, ln = oc.load(oc.build(only=("test",), instance=args.instance, log=lambda s: None)["test"])
    hist = [[int(itos[int(t)]) for t in row[:L]] for row, L in zip(tok, ln)]
    cases, rej = presence_cases(hist, args.n, shipped_length_distribution(), 0, flip)
    bench = bench_from(cases)
    is_rm = np.array([c["op"] == "remove" for c in cases])
    uns_probs = oa.unsteered_probs(model, bench)
    u = oa.unsteered(model, bench)
    print(f"{len(cases)} cases; unedited EI {u['edit_index_union']:+.3f} li_post {u['li_error_vs_post']:.2f}", flush=True)
    out = {"model": label, "n_cases": len(cases), "unedited": {k: v for k, v in u.items() if isinstance(v, (int, float))},
           "probes": {}, "arms": []}
    print(f"\n{'point':>5} {'alpha':>6} | {'EI':>7} {'fid':>6} {'li_post':>8} {'li_pre':>7} {'legal':>6} {'write':>6} {'landed':>7} | remove EI   add EI")
    for p in args.points:
        probe, st = fit_presence_probe(model, data, p, store)
        out["probes"][p] = {k: v for k, v in st.items() if isinstance(v, (int, float))}
        for a in args.alphas:
            pr, card = presence_pinv_arm(model, bench, probe, p, a)
            fid = move_fidelity_ratio(pr, uns_probs, bench.legal_post)
            ei = np.array(card["edit_index_union_per_case"], float)
            rec = {"point": p, "alpha": a, "edit_index_union": card["edit_index_union"], "fidelity_ratio": fid,
                   "li_error_vs_post": card["li_error_vs_post"], "li_error_vs_pre": card["li_error_vs_pre"],
                   "legal_mass": card["legal_mass"], "write_ratio": card["write_ratio"], "readout_landed": card["readout_landed"],
                   "ei_remove": float(np.nanmean(ei[is_rm])), "ei_add": float(np.nanmean(ei[~is_rm]))}
            out["arms"].append(rec)
            print(f"{p:>5} {a:>6g} | {rec['edit_index_union']:>+7.3f} {fid:>6.2f} {rec['li_error_vs_post']:>8.2f} {rec['li_error_vs_pre']:>7.2f} "
                  f"{rec['legal_mass']:>6.3f} {rec['write_ratio']:>6.2f} {rec['readout_landed']:>7.2f} | {rec['ei_remove']:>+8.3f}  {rec['ei_add']:>+8.3f}", flush=True)
    out["minutes"] = round((time.time() - t0) / 60, 1)
    (EXP / "scores" / f"presence_probe_{label.replace(' ', '_')}.json").write_text(json.dumps(out, indent=1, default=float))
    print("done", out["minutes"], "min")


if __name__ == "__main__":
    main()
