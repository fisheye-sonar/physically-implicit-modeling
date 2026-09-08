#!/usr/bin/env python
"""Pilot: decode and PI-edit PRESENCE (occupied vs empty) on the no-flip model.

Colour is causally irrelevant on oth-noflip (checkerboard theorem), so a colour edit had
nothing to steer. Occupancy is what legality depends on. Presence is also perfectly readable
from the moves (a square is occupied iff its token has appeared) — so this asks the paper's
question directly on the trained model, with no retraining: a state variable that is trivially
decodable from the input AND causally necessary for the output — is it editable?

Uses the run's OWN cached 3-way probes (blank / mine / theirs; presence is the blank class),
the canonical `linear_arm` (PI swaps the current and target class logits of one tile) and the
canonical scorecard. Cases from the no-flip TEST split with Li's prefix-length mix: half REMOVE
an occupied non-centre disc (→ blank), half ADD a disc on an empty square with its
parity-consistent colour (the board stays one the world can reach); rejected if the legal set
is unchanged or empty. Output: scores/presence_pilot_<run>.json + a printed table.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.environments.othello import arms as oa  # noqa: E402
from pim.environments.othello import corpus as oc  # noqa: E402
from pim.environments.othello.bench import Benchmark, shipped_length_distribution  # noqa: E402
from pim.environments.othello.data import BLANK, CENTRE, MINE, THEIRS, canonical_vocab, tokens_and_labels  # noqa: E402
from pim.environments.othello.vendor.othello import OthelloBoardState  # noqa: E402
from pim.metrics.set_editability import move_fidelity_ratio  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402
from pim.probes.cache import ProbeCache  # noqa: E402

EXP = REPO / "experiments" / "flip_ablation"
DEV = "cuda"
PARITY_COLOUR = np.array([[1 if (r + c) % 2 == 1 else -1 for c in range(8)] for r in range(8)])  # black on odd


def presence_cases(hist, n, length_counts, seed, flip):
    rng = np.random.default_rng(seed)
    tot = sum(length_counts.values())
    quota = {L: int(round(n * c / tot)) for L, c in sorted(length_counts.items())}
    cases, rej = [], {"same": 0, "empty": 0}
    for L, want in quota.items():
        pool = np.array([i for i, h in enumerate(hist) if len(h) > L])
        got = 0
        for g in rng.permutation(pool):
            if got >= want:
                break
            h = list(hist[g][:L])
            b = OthelloBoardState(flip=flip)
            b.update(h, prt=False)
            pre = sorted(b.get_valid_moves())
            if not pre:
                continue
            op = "remove" if got % 2 == 0 else "add"
            if op == "remove":
                cand = [sq for sq in range(64) if b.state[sq // 8, sq % 8] != 0 and sq not in CENTRE]
            else:
                cand = [sq for sq in range(64) if b.state[sq // 8, sq % 8] == 0]
            for sq in rng.permutation(cand):
                sq = int(sq)
                post = OthelloBoardState(flip=flip)
                post.update(h, prt=False)
                post.state[sq // 8, sq % 8] = 0 if op == "remove" else int(PARITY_COLOUR[sq // 8, sq % 8])
                lp = sorted(post.get_valid_moves())
                if not lp:
                    rej["empty"] += 1
                    continue
                if lp == pre:
                    rej["same"] += 1
                    continue
                nxt = b.next_hand_color
                colour = b.state[sq // 8, sq % 8] if op == "remove" else PARITY_COLOUR[sq // 8, sq % 8]
                occ_lab = MINE if colour == nxt else THEIRS
                cur, tgt = (occ_lab, BLANK) if op == "remove" else (BLANK, occ_lab)
                cases.append({"history": h, "pos_int": sq, "op": op, "cur": cur, "tgt": tgt,
                              "legal_pre": pre, "legal_post": lp, "game": int(g)})
                got += 1
                break
    return cases, rej


def bench_from(cases):
    stoi = canonical_vocab()
    by_len = {}
    for i, c in enumerate(cases):
        by_len.setdefault(len(c["history"]), []).append(i)
    toks, ids = [], []
    for L in sorted(by_len):
        m = np.array(by_len[L], int)
        ids.append(m)
        toks.append(np.array([[stoi[s] for s in cases[i]["history"]] for i in m], np.int64))
    return Benchmark(toks, ids, np.array([c["pos_int"] for c in cases], int), np.zeros(len(cases), int),
                     [c["legal_pre"] for c in cases], [c["legal_post"] for c in cases],
                     np.array([c["cur"] for c in cases], np.int64), np.array([c["tgt"] for c in cases], np.int64))


def cached_grid(model, data, cache_dir):
    store = ProbeCache(cache_dir)
    fname, prov = store.key(model, kind="othello_grid", targets=["mine"], families=["linear", "mlp"], splits=["sequence"],
                            holdout=0.2, epochs=200, batch=4096, lr=1e-3, seed=0, n_seq=int(len(data.tokens)),
                            n_rows=int(data.mask.sum()), n_points=model.n_layers + 1)
    blob = store.load(fname, prov, device=DEV)
    if blob is None:
        sys.exit(f"no cached probe grid under {cache_dir}")
    return blob["probes"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/flip_ablation/L-oth-noflip-20m")
    ap.add_argument("--instance", default="oth-noflip")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--random-init", action="store_true", help="the same architecture at random init (probes from runs/_baselines)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t0 = time.time()
    run = REPO / args.run
    inst, flip = args.instance, oc.flip_of(args.instance)
    n_cases = args.n
    points = (2, 4, 6)
    alphas = (0.5, 1.0, 2.0, 3.0, 5.0)
    if args.random_init:
        from pim.probes.baselines import random_init_model
        model = random_init_model("transformer_l_tokens", {"vocab": 61, "block_size": 59}, seed=0, device=DEV).eval()
        probe_dir = REPO / "runs" / "_baselines" / inst / "probes"
        label = f"random-init on {inst}"
    else:
        model, info = load_checkpoint(run / "best_model.pt", device=DEV)
        model.eval()
        probe_dir = run / "probes"
        label = run.name
    print("===", label, "===")
    itos = {v: k for k, v in canonical_vocab().items()}
    tok, ln = oc.load(oc.build(only=("test",), instance=inst, log=lambda s: None)["test"])
    hist = [[int(itos[int(t)]) for t in row[:L]] for row, L in zip(tok, ln)]
    cases, rej = presence_cases(hist, n_cases, shipped_length_distribution(), 0, flip)
    bench = bench_from(cases)
    print(f"{len(cases)} presence cases ({sum(c['op'] == 'remove' for c in cases)} remove / {sum(c['op'] == 'add' for c in cases)} add); rejected {rej}", flush=True)
    # the run's own cached probes (fitted by master_eval on the 20k probe games)
    ptok, pln = oc.load(oc.build(only=("probe",), instance=inst, log=lambda s: None)["probe"])
    npg = 20_000
    data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(ptok[:npg], pln[:npg])], flip=flip)
    probes = cached_grid(model, data, probe_dir)
    lin = {p: probes[("mine", "linear", "sequence", p)] for p in range(n_points(model))}
    # presence decodability at each point: blank-vs-occupied accuracy of the 3-way linear probe, 500 test games
    with torch.no_grad():
        x = torch.from_numpy(tok[:500, :59]).long().to(DEV)
        rs = model.residual_stack(x)
        lab = tokens_and_labels(hist[:500], flip=flip)
        occ_true = torch.from_numpy(lab.mine[:, :59] != BLANK).to(DEV)       # (500, 59, 64)
        mask = torch.from_numpy(lab.mask[:, :59]).to(DEV)
        pres = {}
        for p in points:
            pr = lin[p](rs[p].reshape(-1, rs[p].shape[-1])).argmax(-1).reshape(500, 59, 64) != BLANK
            pres[p] = float((pr == occ_true)[mask].float().mean())
    print("presence accuracy of the linear probe (blank vs occupied):", {p: round(v, 4) for p, v in pres.items()}, flush=True)
    uns_probs = oa.unsteered_probs(model, bench)
    u = oa.unsteered(model, bench)
    print(f"unedited: EI {u['edit_index_union']:+.3f}  li_post {u['li_error_vs_post']:.2f}  legal_mass(post) {u['legal_mass']:.3f}", flush=True)
    out = {"n_cases": len(cases), "rejected": rej, "presence_accuracy": pres, "unedited": {k: v for k, v in u.items() if isinstance(v, (int, float))}, "arms": []}
    print(f"\n{'point':>5} {'alpha':>6} | {'EI':>7} {'fid':>6} {'li_post':>8} {'li_pre':>7} {'legal':>6} {'write':>6} | remove-only EI  add-only EI")
    is_rm = np.array([c["op"] == "remove" for c in cases])
    for p in points:
        for a in alphas:
            pr, card = oa.linear_arm(model, bench, lin, bench.tgt_lab, bench.cur_lab, mode="pinv", alpha=a, points={p})
            fid = move_fidelity_ratio(pr, uns_probs, bench.legal_post)
            ei_case = np.array(card["edit_index_union_per_case"], float)
            rec = {"point": p, "alpha": a, "edit_index_union": card["edit_index_union"], "fidelity_ratio": fid,
                   "li_error_vs_post": card["li_error_vs_post"], "li_error_vs_pre": card["li_error_vs_pre"],
                   "legal_mass": card["legal_mass"], "write_ratio": card["write_ratio"],
                   "ei_remove": float(np.nanmean(ei_case[is_rm])), "ei_add": float(np.nanmean(ei_case[~is_rm]))}
            out["arms"].append(rec)
            print(f"{p:>5} {a:>6g} | {rec['edit_index_union']:>+7.3f} {fid:>6.2f} {rec['li_error_vs_post']:>8.2f} {rec['li_error_vs_pre']:>7.2f} "
                  f"{rec['legal_mass']:>6.3f} {rec['write_ratio']:>6.2f} | {rec['ei_remove']:>+8.3f}      {rec['ei_add']:>+8.3f}", flush=True)
    out["minutes"] = round((time.time() - t0) / 60, 1)
    (EXP / "scores").mkdir(exist_ok=True)
    (EXP / "scores" / f"presence_pilot_{label.replace(" ", "_")}{args.tag}.json").write_text(json.dumps(out, indent=1, default=float))
    print("done", out["minutes"], "min")


if __name__ == "__main__":
    main()
