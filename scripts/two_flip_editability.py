#!/usr/bin/env python
"""Balanced two-disc edits on adjacent-noflip: is the model editable once the target board is legal?

    .pim/bin/python scripts/two_flip_editability.py                    # 40 cases, the paper's table
    .pim/bin/python scripts/two_flip_editability.py --n 100

On ``oth-adjacent`` (the paper's adjacent-noflip) no single flipped disc is reachable by a legal
game: discs never change colour, so each colour's count is fixed by the number of moves
(``scripts/reachability_table.py``). Flipping one black and one white disc together keeps the
counts, and some such pairs are reachable. This script asks whether the inverse map (IM) edits
those.

Cases. Bench cases are taken in order. Each keeps its own flipped square s and tries partners t,
occupied squares of the opposite colour, in an order seeded by the case index. The first partner
whose two-disc board is REACHABLE and the first whose board is UNREACHABLE are kept
(``pim.environments.othello.reachability.search``, exact; a partner whose search is undecided is
skipped). The first ``--n`` cases with both are used, so every case contributes one single flip
(its bench edit), one legal pair and one illegal pair.

Editing. The canonical ``pim.environments.othello.arms.inverse_arms`` writes each group's post-edit
board (``post_boards``) at every residual point, with the run's cached inverse maps. The Edit Index
(symmetric difference) and the Edit Fidelity (1 - ``move_fidelity_ratio``) are computed per group.

Checks. Every legal pair's witness game replays to its target through the vendored engine, every
pair keeps both colour counts, and the single-flip boards equal the ones ``inverse_arms`` builds
itself for the bench edit. Output: runs/adjacency_ablation/L-oth-adjacent-20m/two_flip_editability.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from pim.environments.othello import corpus as oc  # noqa: E402
from pim.environments.othello.bench import benchmark_from_cases, cases_path  # noqa: E402
from pim.environments.othello.counterfactual import replay  # noqa: E402
from pim.environments.othello.reachability import search, state_of  # noqa: E402
from pim.environments.othello.vendor.othello import OthelloBoardState  # noqa: E402

VERSION = "2026-09-23.1"
RUN = "adjacency_ablation/L-oth-adjacent-20m"
GROUPS = ("single", "legal", "illegal")


def find_pairs(cases: list[dict], rules: dict, n: int, budget: int) -> list[dict]:
    """The first ``n`` bench cases with both a reachable and an unreachable balanced partner."""
    out = []
    for i, c in enumerate(cases):
        h, s = [int(x) for x in c["history"]], int(c["pos_int"])
        b, w, mover = state_of(replay(h, rules))
        opp = w if (b >> s) & 1 else b                          # partners: the other colour's discs
        partners = [t for t in range(64) if (opp >> t) & 1]
        rng = np.random.default_rng(i)
        legal = illegal = None
        for t in [partners[j] for j in rng.permutation(len(partners))]:
            tgt = (b ^ (1 << s) ^ (1 << t), w ^ (1 << s) ^ (1 << t), mover)
            assert bin(tgt[0]).count("1") == bin(b).count("1") and bin(tgt[1]).count("1") == bin(w).count("1")
            v = search(tgt, len(h), rules, order={m: k for k, m in enumerate(h)}, budget=budget)
            if v.status == "reachable" and legal is None:
                wb = replay(v.witness, rules)
                assert wb is not None and state_of(wb) == tgt, "witness failed the vendored engine's replay"
                legal = {"t": t, "witness": v.witness}
            elif v.status == "unreachable" and illegal is None:
                illegal = {"t": t}
            if legal and illegal:
                break
        if legal and illegal:
            out.append({"case": i, "s": s, "legal": legal, "illegal": illegal})
            if len(out) >= n:
                break
    return out


def main() -> None:
    import torch

    from pim.environments.othello import arms as oa
    from pim.environments.othello import case_targets
    from pim.environments.othello.data import MINE, THEIRS, canonical_vocab, tokens_and_labels
    from pim.metrics.set_editability import edit_index_legal, move_fidelity_ratio
    from pim.models import load_checkpoint
    from pim.scoring.othello import _probe_games

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=40, help="cases (each gives one single flip, one legal and one illegal pair)")
    ap.add_argument("--budget", type=int, default=2_000_000, help="search nodes per partner before it is skipped")
    a = ap.parse_args()
    t0 = time.time()
    run_dir = _REPO / "runs" / RUN
    S = json.loads((run_dir / "scores.json").read_text())
    inst, settings = S["instance"], S["settings"]
    rules = oc.rules_of(inst)
    cases = pickle.load(open(cases_path(inst), "rb"))
    pairs = find_pairs(cases, rules, a.n, a.budget)
    print(f"{len(pairs)} cases with a legal and an illegal balanced pair, from the first {pairs[-1]['case'] + 1} bench cases "
          f"[{(time.time() - t0) / 60:.1f} min]", flush=True)

    # one bench holding every group: the same histories three times, the edit differing per group
    sub = [cases[p["case"]] for p in pairs]
    bench1 = benchmark_from_cases(sub, **rules)
    n = len(sub)
    itos = {v: k for k, v in canonical_vocab().items()}
    hist = [[int(x) for x in c["history"]] for c in sub]
    bd = tokens_and_labels([[itos[canonical_vocab()[m]] for m in h] for h in hist], **rules)
    s_pre = np.stack([bd.mine[i, len(hist[i]) - 1] for i in range(n)])
    cur, tgt = case_targets(bench1)

    def swap(board, sq):
        board[sq] = THEIRS if board[sq] == MINE else MINE

    def legal_set(h, squares):
        b = OthelloBoardState(**rules)
        b.update(h, prt=False)
        for q in squares:
            b.state[q // 8, q % 8] *= -1
        return sorted(b.get_valid_moves())

    boards, legal_post = {}, {}
    for g in GROUPS:
        bb = s_pre.copy()
        sets = []
        for i, p in enumerate(pairs):
            sq = [p["s"]] if g == "single" else [p["s"], p[g]["t"]]
            for q in sq:
                swap(bb[i], q)
            sets.append(legal_set(hist[i], sq))
        boards[g], legal_post[g] = bb, sets
    ref = s_pre.copy()
    ref[np.arange(n), bench1.pos_int] = tgt
    assert (boards["single"] == ref).all(), "single-flip boards differ from inverse_arms' own construction"
    assert legal_post["single"] == [list(x) for x in bench1.legal_post], "single-flip legal sets differ from the bench's"

    all_cases = sub * 3
    bench = benchmark_from_cases(all_cases, **rules)
    bench = dataclasses.replace(bench, legal_post=legal_post["single"] + legal_post["legal"] + legal_post["illegal"])
    post_all = np.concatenate([boards[g] for g in GROUPS])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=dev)
    model.eval()
    uns = oa.unsteered_probs(model, bench)
    data = _probe_games(settings["oth_probe_games"], inst)
    _, stats, probs_by = oa.inverse_arms(model, bench, data, rules=rules, cache_dir=run_dir / "probes",
                                         n_games=int(settings["oth_probe_games"]), uns_probs=uns, log=None,
                                         return_probs=True, post_boards=post_all)
    pre_all = bench.legal_pre
    res = {g: {"unedited_index": None, "points": []} for g in GROUPS}
    for k, g in enumerate(GROUPS):
        sl = slice(k * n, (k + 1) * n)
        pre_g, post_g = pre_all[sl], legal_post[g]
        res[g]["unedited_index"] = float(np.nanmean(edit_index_legal(uns[sl], pre_g, post_g, "symdiff")))
        for ell in sorted({pt for (ed, pt) in probs_by if ed == "IM"}):
            pr = probs_by[("IM", ell)][sl]
            ei = np.asarray(edit_index_legal(pr, pre_g, post_g, "symdiff"), float)
            res[g]["points"].append({"point": int(ell), "index": float(np.nanmean(ei)),
                                     "index_se": float(np.nanstd(ei, ddof=1) / np.sqrt(np.isfinite(ei).sum())),
                                     "fidelity": 1.0 - float(move_fidelity_ratio(pr, uns[sl], post_g))})
    out = {"run": RUN, "instance": inst, "version": VERSION, "created": time.strftime("%Y-%m-%d %H:%M"),
           "n_cases": n, "budget": a.budget, "editor": "IM (canonical inverse_arms, post_boards)",
           "edit_index": "symmetric difference (legal-set)", "groups": res, "g_r2": stats["g_r2"],
           "pairs": pairs, "minutes": round((time.time() - t0) / 60, 1)}
    path = run_dir / "two_flip_editability.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1, default=float))
    os.replace(tmp, path)
    print(f"wrote {path.relative_to(_REPO)}")
    print(f"\nIM, n = {n} cases per group, Edit Index (symdiff) ± SE / Edit Fidelity")
    print("point |       single flip        |    legal two-disc flip   |   illegal two-disc flip")
    print("unedited " + "  ".join(f"| {res[g]['unedited_index']:+.3f}{'':18}" for g in GROUPS))
    for j in range(len(res["single"]["points"])):
        print(f"{res['single']['points'][j]['point']:>5}    " + "  ".join(
            f"| {res[g]['points'][j]['index']:+.3f} ± {res[g]['points'][j]['index_se']:.3f} / {res[g]['points'][j]['fidelity']:+.2f}"
            for g in GROUPS))


if __name__ == "__main__":
    main()
