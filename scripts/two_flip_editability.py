#!/usr/bin/env python
"""Balanced two-disc edits on adjacent-noflip: is the model editable once the target board is legal?

    .pim/bin/python scripts/two_flip_editability.py                    # 40 cases, the paper's table
    .pim/bin/python scripts/two_flip_editability.py --n 100
    .pim/bin/python scripts/two_flip_editability.py --run flip_ablation/L-oth-noflip-20m --no-legal   # standard-noflip
    .pim/bin/python scripts/two_flip_editability.py --run <topic>/<run> --n 1000 --out two_flip_editability_full.json
                                                   # the full bench: every one of the 1000 cases is tried

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
(its bench edit), one legal pair and one illegal pair. With ``--n`` at or above the bench size every
case is tried, and the output records why each dropped case was dropped (no reachable partner, no
unreachable partner, or only undecided searches), so the kept n can be checked. The pair search is
cached beside the output (``<out>.pairs.json``) and reused when its settings match. While it runs it
is logged case by case (``<out>.pairs.partial.jsonl``), so an interrupted search resumes.

``--no-legal`` is for a variant with no legal pair at all (standard-noflip, where every disc's colour
equals its square parity, so any flip is off parity): EVERY partner of each case is searched, the
reachable ones are counted (the claim is that there are none), and the first unreachable partner
is kept. The groups are then the single flip and the illegal pair.

Editing. The canonical ``pim.environments.othello.arms.inverse_arms`` writes each group's post-edit
board (``post_boards``) at every residual point, with the run's cached inverse maps. PI and GS
(``linear_arm`` / ``grad_steer_arm`` with their ``second`` tile) are swept over the run's own
settings grid, and each group is reported at the setting Table 2's rule picks within that group
(``pim.metrics.selection.best_arm``), and also at the run's Table 2 setting. The Edit Index
(symmetric difference) and the Edit Fidelity (1 - ``move_fidelity_ratio``) are computed per group.

Checks. Every legal pair's witness game replays to its target through the vendored engine, every
pair keeps both colour counts, the single-flip boards equal the ones ``inverse_arms`` builds itself
for the bench edit, and PI / GS at the run's Table 2 setting reproduce scores.json on the full bench.
Output: runs/<run>/<--out> (default two_flip_editability.json, the n = 40 pilot the paper table was built from).
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

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from pim.environments.othello import corpus as oc  # noqa: E402
from pim.environments.othello.bench import benchmark_from_cases, cases_path  # noqa: E402
from pim.environments.othello.counterfactual import replay  # noqa: E402
from pim.environments.othello.reachability import search, state_of  # noqa: E402
from pim.environments.othello.vendor.othello import OthelloBoardState  # noqa: E402

VERSION = "2026-09-24.1"
RUN = "adjacency_ablation/L-oth-adjacent-20m"
GROUPS = ("single", "legal", "illegal")


def _case_pairs(args: tuple) -> dict | None:
    """One case: its first reachable and first unreachable balanced partner, in an order seeded by the
    case index (or, with ``require_legal=False``, every partner searched and counted)."""
    i, c, rules, budget, require_legal = args
    h, s = [int(x) for x in c["history"]], int(c["pos_int"])
    b, w, mover = state_of(replay(h, rules))
    opp = w if (b >> s) & 1 else b                              # partners: the other colour's discs
    partners = [t for t in range(64) if (opp >> t) & 1]
    rng = np.random.default_rng(i)
    legal = illegal = None
    counts = {"reachable": 0, "unreachable": 0, "undecided": 0}
    for t in [partners[j] for j in rng.permutation(len(partners))]:
        tgt = (b ^ (1 << s) ^ (1 << t), w ^ (1 << s) ^ (1 << t), mover)
        assert bin(tgt[0]).count("1") == bin(b).count("1") and bin(tgt[1]).count("1") == bin(w).count("1")
        v = search(tgt, len(h), rules, order={m: k for k, m in enumerate(h)}, budget=budget)
        counts[v.status] += 1
        if v.status == "reachable" and legal is None:
            wb = replay(v.witness, rules)
            assert wb is not None and state_of(wb) == tgt, "witness failed the vendored engine's replay"
            legal = {"t": t, "witness": v.witness}
        elif v.status == "unreachable" and illegal is None:
            illegal = {"t": t}
        if require_legal and legal and illegal:
            break
    kept = bool(illegal and (legal or not require_legal))
    why = None if kept else ("no unreachable partner" if not illegal else "no reachable partner")
    if not kept and not legal and counts["undecided"]:
        why += " (some searches undecided)"
    return {"case": i, "s": s, "legal": legal, "illegal": illegal, "partners_searched": counts,
            "kept": kept, "dropped_because": why}


def _compatible(k1: dict, k2: dict) -> bool:
    """Two search keys whose per-case results are interchangeable: the target n and the size of the
    case pool do not change any case's result (pools extend the bench, checked at load)."""
    drop = ("n", "pool")
    return {k: v for k, v in k1.items() if k not in drop} == {k: v for k, v in k2.items() if k not in drop}


def find_pairs(cases: list[dict], rules: dict, n: int, budget: int, require_legal: bool = True,
               workers: int = 1, log_every: int = 50, partial: Path | None = None,
               key: dict | None = None) -> tuple[list[dict], list[dict]]:
    """The first ``n`` bench cases (in bench order) that qualify, and every case tried (kept or
    dropped, with the reason). Cases are independent, so a pool gives exactly the serial result:
    results are consumed in bench order and the pool stops at ``n``.

    ``partial``: a JSON-lines log (first line ``key``, then one tried case per line, in bench order),
    appended as each case finishes, so an interrupted search resumes where it stopped. Records are
    reused only under the same ``key``. Workers are recycled every few cases so the memory of a hard
    search is returned to the system."""
    tried: list[dict] = []
    if partial is not None and partial.exists():
        lines = partial.read_text().splitlines()
        if lines and _compatible(json.loads(lines[0]), key):
            for ln in lines[1:]:
                try:
                    tried.append(json.loads(ln))
                except json.JSONDecodeError:          # a line cut short by the interruption
                    break
            assert [r["case"] for r in tried] == list(range(len(tried))), "partial log out of order"
            assert all(r["s"] == int(cases[r["case"]]["pos_int"]) for r in tried), "partial log is for other cases"
            print(f"  pair search: resuming after {len(tried)} cases from {partial}", flush=True)
        else:
            tried = []
    out = [r for r in tried if r["kept"]][:n]
    if partial is not None:
        with open(partial, "w") as f:
            f.write(json.dumps(key) + "\n")
            for r in tried:
                f.write(json.dumps(r, default=int) + "\n")
    log = open(partial, "a") if partial is not None else None
    args = [(i, c, rules, budget, require_legal) for i, c in enumerate(cases)][len(tried):]
    t0 = time.time()
    if len(out) < n and args:
        # results are consumed strictly in bench order from a sliding window of submitted cases; a worker
        # that dies (e.g. killed at a memory cap) raises BrokenProcessPool here instead of hanging, and a
        # rerun resumes from the partial log
        ex = ProcessPoolExecutor(workers, mp_context=get_context("spawn"), max_tasks_per_child=8)
        window, nxt = deque(), 0
        try:
            while len(out) < n and (window or nxt < len(args)):
                while nxt < len(args) and len(window) < 4 * workers:
                    window.append(ex.submit(_case_pairs, args[nxt]))
                    nxt += 1
                r = window.popleft().result()
                tried.append(r)
                if log is not None:
                    log.write(json.dumps(r, default=int) + "\n")
                    log.flush()
                if r["kept"]:
                    out.append(r)
                if log_every and len(tried) % log_every == 0:
                    print(f"  pair search: {len(tried)} cases tried, {len(out)} kept [{(time.time() - t0) / 60:.1f} min]", flush=True)
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    if log is not None:
        log.close()
    return out, tried


def main() -> None:
    import torch

    from pim.environments.othello import arms as oa
    from pim.environments.othello import case_targets
    from pim.environments.othello.data import MINE, THEIRS, canonical_vocab, tokens_and_labels
    from pim.environments.othello.bench import load_benchmark
    from pim.metrics.selection import GUARD, best_arm
    from pim.metrics.set_editability import edit_index_legal, move_fidelity_ratio
    from pim.models import load_checkpoint, n_points
    from pim.scoring.othello import _probe_games

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=40, help="cases (each gives one single flip, one legal and one illegal pair)")
    ap.add_argument("--budget", type=int, default=2_000_000, help="search nodes per partner before it is skipped")
    ap.add_argument("--run", default=RUN, help="<topic>/<run> of an Othello run")
    ap.add_argument("--no-legal", action="store_true", help="the variant has no legal pair: single flip vs illegal pair only")
    ap.add_argument("--workers", type=int, default=16, help="processes for the pair search (the result does not depend on it)")
    ap.add_argument("--out", default="two_flip_editability.json", help="output file name inside runs/<run>/")
    ap.add_argument("--pool", type=int, default=1000,
                    help="cases to draw from: the 1000-case bench, or a larger set from scripts/make_othello_edits.py "
                         "--n <pool> (same recipe and seed, so its first 1000 cases ARE the bench; checked)")
    a = ap.parse_args()
    groups = ("single", "illegal") if a.no_legal else GROUPS
    two = [g for g in groups if g != "single"]
    t0 = time.time()
    run_dir = _REPO / "runs" / a.run
    S = json.loads((run_dir / "scores.json").read_text())
    inst, settings = S["instance"], S["settings"]
    rules = oc.rules_of(inst)
    cases = pickle.load(open(cases_path(inst), "rb"))
    if a.pool != len(cases):
        from pim.environments.layout import othello_cases_file

        pool = pickle.load(open(othello_cases_file(inst, a.pool), "rb"))
        assert len(pool) == a.pool and all(x["history"] == y["history"] and x["pos_int"] == y["pos_int"]
                                           for x, y in zip(cases, pool)), "the pool does not begin with the bench"
        cases = pool
    path = run_dir / a.out
    cache = path.with_suffix(".pairs.json")
    key = {"version": VERSION, "instance": inst, "n": a.n, "pool": len(cases), "budget": a.budget,
           "require_legal": not a.no_legal}
    partial = path.with_suffix(".pairs.partial.jsonl")
    c_ = json.loads(cache.read_text()) if cache.exists() else None
    if c_ is not None and c_.get("key") == key:
        pairs, tried = c_["pairs"], c_["tried"]
        print(f"pair search reused from {cache}", flush=True)
    else:
        if c_ is not None and _compatible(c_["key"], key):
            done = sum(1 for _ in open(partial)) - 1 if partial.exists() else 0
            if len(c_["tried"]) > done:                    # an earlier, smaller search seeds this one
                with open(partial, "w") as f:
                    f.write(json.dumps(key) + "\n")
                    for r in c_["tried"]:
                        f.write(json.dumps(r, default=int) + "\n")
        pairs, tried = find_pairs(cases, rules, a.n, a.budget, require_legal=not a.no_legal, workers=a.workers,
                                  partial=partial, key=key)
        tmp_ = cache.with_suffix(".json.tmp")
        tmp_.write_text(json.dumps({"key": key, "pairs": pairs, "tried": tried}, default=int))
        os.replace(tmp_, cache)
    searched = {k: sum(p_["partners_searched"][k] for p_ in pairs) for k in ("reachable", "unreachable", "undecided")}
    dropped = {}
    for r in tried:
        if not r["kept"]:
            dropped[r["dropped_because"]] = dropped.get(r["dropped_because"], 0) + 1
    print(f"{len(pairs)} cases kept of {len(tried)} bench cases tried; dropped {dropped}; partners searched (kept cases) "
          f"{searched} [{(time.time() - t0) / 60:.1f} min]", flush=True)
    if a.no_legal and searched["reachable"]:
        print(f"  WARNING: --no-legal, but {searched['reachable']} reachable balanced pairs exist on this variant", flush=True)

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
    for g in groups:
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

    all_cases = sub * len(groups)
    bench = benchmark_from_cases(all_cases, **rules)
    bench = dataclasses.replace(bench, legal_post=[x for g in groups for x in legal_post[g]])
    post_all = np.concatenate([boards[g] for g in groups])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=dev)
    model.eval()
    uns = oa.unsteered_probs(model, bench)
    data = _probe_games(settings["oth_probe_games"], inst)
    _, stats, probs_by = oa.inverse_arms(model, bench, data, rules=rules, cache_dir=run_dir / "probes",
                                         n_games=int(settings["oth_probe_games"]), uns_probs=uns, log=None,
                                         return_probs=True, post_boards=post_all)
    pre_all = bench.legal_pre
    res = {g: {"unedited_index": None, "points": []} for g in groups}
    for k, g in enumerate(groups):
        sl = slice(k * n, (k + 1) * n)
        pre_g, post_g = pre_all[sl], legal_post[g]
        res[g]["unedited_index"] = float(np.nanmean(edit_index_legal(uns[sl], pre_g, post_g, "symdiff")))
        for ell in sorted({pt for (ed, pt) in probs_by if ed == "IM"}):
            pr = probs_by[("IM", ell)][sl]
            ei = np.asarray(edit_index_legal(pr, pre_g, post_g, "symdiff"), float)
            res[g]["points"].append({"point": int(ell), "index": float(np.nanmean(ei)),
                                     "index_se": float(np.nanstd(ei, ddof=1) / np.sqrt(np.isfinite(ei).sum())),
                                     "fidelity": 1.0 - float(move_fidelity_ratio(pr, uns[sl], post_g))})
    # ---- PI and GS on the same cases: every setting of the run's grid, per group
    grid = oa.fit_probe_grid(model, data, cache_dir=run_dir / "probes", log=None)
    npnt = n_points(model)
    lin = {q: grid.probes[("mine", "linear", "sequence", q)] for q in range(npnt)}
    mlp = {q: grid.probes[("mine", "mlp", "sequence", q)] for q in range(npnt)}
    steps, beta = int(settings["oth_gs_steps"]), float(settings["oth_gs_beta"])
    t2 = {ed: best_arm(S["arms"], ed, "edit_index_symdiff", guard=GUARD) for ed in ("PI", "GS", "IM")}
    full = load_benchmark(inst)                     # the single-tile path must still reproduce scores.json
    fcur, ftgt = case_targets(full)
    funs = oa.unsteered_probs(model, full)
    for ed in ("PI", "GS"):
        pt, al = int(t2[ed]["point"]), float(t2[ed]["alpha"])
        pr = (oa.linear_arm(model, full, lin, ftgt, fcur, mode="pinv", alpha=al, points={pt})[0] if ed == "PI" else
              oa.grad_steer_arm(model, full, mlp, pt, alpha=al, n_steps=steps, beta=beta, target_labels=ftgt)[0])
        e = float(np.nanmean(edit_index_legal(pr, full.legal_pre, full.legal_post, "symdiff")))
        r = float(move_fidelity_ratio(pr, funs, full.legal_post))
        assert abs(e - t2[ed]["edit_index_symdiff"]) < 0.01 and abs(r - t2[ed]["fidelity_ratio"]) < 0.02, \
            f"{ed}: the single-tile path no longer reproduces scores.json ({e:+.4f} vs {t2[ed]['edit_index_symdiff']:+.4f})"
        print(f"  check {ed} at Table 2 setting (pt {pt}, a {al:g}) on the full bench: {e:+.4f} / ratio {r:.4f} = scores.json", flush=True)
    b1 = benchmark_from_cases(sub, **rules)
    b2 = dataclasses.replace(benchmark_from_cases(sub * len(two), **rules), legal_post=[x for g in two for x in legal_post[g]])
    cur1, tgt1 = case_targets(b1)
    cur2, tgt2 = case_targets(b2)
    t_sq = np.array([pr_[g]["t"] for g in two for pr_ in pairs], dtype=np.int64)
    cur_t = np.concatenate([s_pre] * len(two))[np.arange(len(two) * n), t_sq]
    assert set(np.unique(cur_t)) <= {MINE, THEIRS}, "a partner square is empty"
    tgt_t = np.where(cur_t == MINE, THEIRS, MINE)
    uns1, uns2 = oa.unsteered_probs(model, b1), oa.unsteered_probs(model, b2)
    parts = {"single": (slice(0, n), uns1, b1.legal_pre, legal_post["single"])}
    for k, g in enumerate(two):
        sl = slice(k * n, (k + 1) * n)
        parts[g] = (sl, uns2[sl], b2.legal_pre[sl], legal_post[g])
    arms = {g: [] for g in groups}

    def add(ed, pt, al, pr1, pr2):
        for g, (sl, u, pre_g, post_g) in parts.items():
            pr = pr1 if g == "single" else pr2[sl]
            ei = np.asarray(edit_index_legal(pr, pre_g, post_g, "symdiff"), float)
            arms[g].append({"editor": ed, "point": int(pt), "alpha": float(al), "edit_index_symdiff": float(np.nanmean(ei)),
                            "index_se": float(np.nanstd(ei, ddof=1) / np.sqrt(np.isfinite(ei).sum())),
                            "fidelity_ratio": float(move_fidelity_ratio(pr, u, post_g))})
    for pt in range(npnt):
        for al in settings["oth_alpha_pi"]:
            add("PI", pt, al, oa.linear_arm(model, b1, lin, tgt1, cur1, mode="pinv", alpha=al, points={pt})[0],
                oa.linear_arm(model, b2, lin, tgt2, cur2, mode="pinv", alpha=al, points={pt}, second=(t_sq, cur_t, tgt_t))[0])
    print(f"  PI swept [{(time.time() - t0) / 60:.1f} min]", flush=True)
    for ls in settings["oth_gs_layers"]:
        for al in settings["oth_alpha_gs"]:
            add("GS", ls, al, oa.grad_steer_arm(model, b1, mlp, ls, alpha=al, n_steps=steps, beta=beta, target_labels=tgt1)[0],
                oa.grad_steer_arm(model, b2, mlp, ls, alpha=al, n_steps=steps, beta=beta, target_labels=tgt2,
                                  second=(t_sq, tgt_t))[0])
    print(f"  GS swept [{(time.time() - t0) / 60:.1f} min]", flush=True)
    for g in groups:
        for q in res[g]["points"]:
            arms[g].append({"editor": "IM", "point": q["point"], "alpha": 1.0, "edit_index_symdiff": q["index"],
                            "index_se": q["index_se"], "fidelity_ratio": 1.0 - q["fidelity"]})
    reported = {g: {ed: best_arm(arms[g], ed, "edit_index_symdiff", guard=GUARD) for ed in ("PI", "GS", "IM")} for g in groups}
    at_t2 = {g: {ed: next(x for x in arms[g] if x["editor"] == ed and x["point"] == int(t2[ed]["point"])
                          and x["alpha"] == float(t2[ed]["alpha"])) for ed in ("PI", "GS", "IM")} for g in groups}

    out = {"run": a.run, "instance": inst, "groups_run": list(groups), "partners_searched": searched, "version": VERSION, "created": time.strftime("%Y-%m-%d %H:%M"),
           "n_cases": n, "case_pool": len(cases), "cases_tried": len(tried), "cases_dropped": dropped, "budget": a.budget, "editor": "IM (canonical inverse_arms, post_boards)",
           "reported": reported, "at_table2_setting": at_t2,
           "table2_setting": {ed: {"point": int(t2[ed]["point"]), "alpha": float(t2[ed]["alpha"])} for ed in ("PI", "GS", "IM")},
           "arms_by_group": arms,
           "edit_index": "symmetric difference (legal-set)", "groups": res, "g_r2": stats["g_r2"],
           "pairs": pairs, "minutes": round((time.time() - t0) / 60, 1)}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1, default=float))
    os.replace(tmp, path)
    print(f"wrote {path.relative_to(_REPO) if path.is_relative_to(_REPO) else path}")
    print(f"\nIM, n = {n} cases per group, Edit Index (symdiff) ± SE / Edit Fidelity")
    print("point | " + " | ".join(f"{g:^24}" for g in groups))
    print("unedited " + "  ".join(f"| {res[g]['unedited_index']:+.3f}{'':18}" for g in groups))
    for j in range(len(res["single"]["points"])):
        print(f"{res['single']['points'][j]['point']:>5}    " + "  ".join(
            f"| {res[g]['points'][j]['index']:+.3f} ± {res[g]['points'][j]['index_se']:.3f} / {res[g]['points'][j]['fidelity']:+.2f}"
            for g in groups))

    for title, tab in (("each group at the setting Table 2's rule picks within that group", reported),
                       ("each group at the run's Table 2 setting", at_t2)):
        print(f"\n{title}: index ± SE / Edit Fidelity (point, alpha)")
        for g in groups:
            print(f"  {g:8} " + "   ".join(f"{ed} {tab[g][ed]['edit_index_symdiff']:+.3f} ± {tab[g][ed]['index_se']:.3f} / "
                                        f"{1 - tab[g][ed]['fidelity_ratio']:+.2f} (pt {tab[g][ed]['point']}, a {tab[g][ed]['alpha']:g})"
                                        for ed in ("PI", "GS", "IM")))


if __name__ == "__main__":
    # exit explicitly: the process pool's manager thread can block interpreter shutdown after an early stop,
    # which would hang a finished run and, worse, turn a crash into a hang that the driver's retry never sees
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
