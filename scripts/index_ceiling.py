#!/usr/bin/env python
"""The Edit Index CEILING of a run → runs/<topic>/<run>/index_ceiling.json.

    .pim/bin/python scripts/index_ceiling.py --run noise_ablation/L-dw-noiseless-20m
    .pim/bin/python scripts/index_ceiling.py --run initial_othello_comparison/L-oth-20m --reachability

What the Edit Index gives when NO editor is involved and the model is simply run on a history that
genuinely produces the edited world — the sanity check on the index itself (paper, Metrics): an
index well below +1 here is the model's own prediction error, not an editor's failure, and every
editor's index is read against this number, not against +1.

discworld   the OVERWRITE ORACLE (``pim.environments.discworld.arms.overwrite_oracle_rollout``): the
            whole input window re-rendered with the edited disc displaced by its teleport vector,
            scored on the run's canonical edit bench (the instance's 1000-case selection) exactly
            like an editor's arm. dw-blink is refused: the bench does not carry the blackout
            schedule the counterfactual window would need.
othello     a REAL game whose board equals the flipped board (``othello.counterfactual.search_cf``),
            kept only when it is EXACT and ORDINARY — the model's error on it within the
            ``--pct`` percentile of held-out prefixes of the same length, which screens the
            move-swap histories the model handles badly. Both index constructions are reported
            (symmetric difference = the headline; union). Few cases survive (13–32 of 900), so the
            ceiling carries a percentile-bootstrap 95% interval over cases.
            ``--editors``       the canonical best PI / ND / GS arms re-scored on the SAME cases
            ``--reachability``  the inverse-map editor on those REACHABLE cases and on a matched
                                sample of cases with NO exact counterfactual board (legal vs
                                illegal target states)

Moved 2026-09-19 from ``experiments/adjacent_flip_ablation/scripts/ceiling_symdiff.py`` (Othello;
same seed, cases and filter — it reproduces those files) and from the 2026-09-07 oracle runs
(discworld, then on the old 192-case bench). Forward passes only: minutes per run.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

VERSION = "2026-09-19.1"


def _write(run_dir: Path, res: dict) -> None:
    out = run_dir / "index_ceiling.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=1, default=float))
    os.replace(tmp, out)
    print(f"  wrote {out.relative_to(_REPO)}", flush=True)


def discworld(run_dir: Path, S: dict, n: int) -> dict:
    from pim.environments.discworld import arms as dwa
    from pim.environments.discworld import bench as dwb
    from pim.models import load_checkpoint

    inst = S["instance"]
    sim_blink = float(json.loads((run_dir / "config.json").read_text()).get("data", {}).get("blink_prob", 0) or 0)
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=dwb.DEV)
    model.eval()
    basis = next(iter(S["bases"]))                      # the frames scored do not depend on the read-out basis
    b = dwb.load_bench(model, n=n, target="full", basis_name=basis if basis in ("frustum", "cartesian") else "cartesian",
                       instance=inst)
    if sim_blink > 0 or float(b.sim.get("blink_prob", 0) or 0) > 0:
        raise SystemExit(f"{inst}: the overwrite oracle re-renders the window WITHOUT the blackout schedule — "
                         "not a valid ceiling on a blink instance")
    uns = dwa.unsteered(model, b)
    card = dwa.score(model, b, dwa.overwrite_oracle_rollout(model, b), uns)
    keep = lambda c: {k: v for k, v in c.items() if np.isscalar(v)}                       # noqa: E731
    return {"env": "discworld", "instance": inst, "n_cases": int(b.n), "construction": "ray-zone",
            "method": "overwrite oracle: the input window re-rendered with the edited disc displaced throughout",
            "unedited": keep(uns), "ceiling": keep(card)}


def _boot(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> list[float]:
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def othello(run_dir: Path, S: dict, n: int, pct: float, editors: bool, reachability: bool) -> dict:
    from pim.environments.othello import arms as oa
    from pim.environments.othello import case_targets
    from pim.environments.othello import corpus as oc
    from pim.environments.othello.bench import benchmark_from_cases, cases_path
    from pim.environments.othello.counterfactual import replay, search_cf
    from pim.environments.othello.data import board_probs, canonical_vocab, tokens_and_labels
    from pim.metrics.set_editability import edit_index_legal, move_fidelity_ratio, move_rmse_per_case
    from pim.models import load_checkpoint, n_points

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inst, best, settings = S["instance"], S["best"], S["settings"]
    rules = oc.rules_of(inst)
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=dev)
    model.eval()
    stoi = canonical_vocab()
    itos = {v: k for k, v in stoi.items()}

    def pred(hists):
        out = np.zeros((len(hists), 64), np.float32)
        by: dict[int, list[int]] = {}
        for i, h in enumerate(hists):
            by.setdefault(len(h), []).append(i)
        with torch.no_grad():
            for ids in by.values():
                idx = torch.from_numpy(np.array([[stoi[x] for x in hists[i]] for i in ids])).to(dev)
                out[ids] = board_probs(model.decode(idx), getattr(model, "output_kind", "logits"))
        return out

    def own_legal(hs):
        return [sorted(replay(h, rules).get_valid_moves()) for h in hs]

    # ORDINARY = the model's error on a history is within the pct-th percentile of held-out prefixes of that length
    tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("test",), instance=inst)["test"])
    ho, r = {}, 0
    for L in range(5, 31):
        hs = []
        while len(hs) < 60 and r < len(tok):
            h = [int(itos[int(t)]) for t in tok[r][: int(ln[r])]]
            r += 1
            if len(h) > L:
                hs.append(h[:L])
        ho[L] = move_rmse_per_case(pred(hs), own_legal(hs))
    thr = {L: float(np.percentile(v, pct)) for L, v in ho.items()}

    cases = pickle.load(open(cases_path(inst), "rb"))
    rng = np.random.default_rng(0)
    raw = []
    for i in rng.permutation(len(cases))[:n]:
        c = cases[i]
        h, s = [int(x) for x in c["history"]], int(c["pos_int"])
        bst, d, _ = search_cf(h, s, rules)
        if bst is not None and d == 0:
            raw.append({"h": h, "hh": bst[0], "s": s, "ori": c["ori_color"]})
    p_cf_all = pred([f["hh"] for f in raw])
    rm = move_rmse_per_case(p_cf_all, own_legal([f["hh"] for f in raw]))
    ordn = np.array([x <= thr[len(f["hh"])] for f, x in zip(raw, rm)])
    found = [raw[i] for i in np.where(ordn)[0]]
    p_cf = p_cf_all[ordn]
    print(f"  {len(raw)} exact counterfactual boards of {n} cases searched, {len(found)} ordinary", flush=True)

    if not found:                          # oth-noflip: colour is fixed by the square — no game has the flipped board
        return {"env": "othello", "instance": inst, "n_searched": int(n), "n_exact": len(raw), "n_cases": 0,
                "method": "no exact, ordinary counterfactual game exists for any searched case",
                "unedited": {"symdiff": float("nan")}, "ceiling": {"symdiff": float("nan")}}

    def bench_of(sub):
        return benchmark_from_cases([{"history": f["h"], "pos_int": f["s"], "ori_color": f["ori"]} for f in sub], **rules)

    bench = bench_of(found)
    pre, post = bench.legal_pre, bench.legal_post
    p_un = pred([f["h"] for f in found])

    def both(probs, ci=False):
        sd, un = edit_index_legal(probs, pre, post, "symdiff"), edit_index_legal(probs, pre, post, "union")
        out = {"symdiff": float(np.nanmean(sd)), "union": float(np.nanmean(un)),
               "fidelity": float(move_fidelity_ratio(probs, p_un, post))}
        if ci:
            out.update({"symdiff_ci95": _boot(np.asarray(sd, float)), "union_ci95": _boot(np.asarray(un, float)),
                        "symdiff_by_case": [float(x) for x in sd]})
        return out

    res = {"env": "othello", "instance": inst, "n_searched": int(n), "n_exact": len(raw), "n_cases": len(found),
           "ordinary_percentile": pct, "construction": "legal-set (symdiff = the headline; union)",
           "method": "a real game whose board equals the flipped board, exact and ordinary",
           "unedited": both(p_un), "ceiling": both(p_cf, ci=True)}

    if editors or reachability:
        ptok, pln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("probe",), instance=inst)["probe"])
        data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(ptok[:20000], pln[:20000])], **rules)
    if editors:
        cur, tgt = case_targets(bench)
        NP = n_points(model)
        grid = oa.fit_probe_grid(model, data, cache_dir=run_dir / "probes", log=None)
        lin = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(NP)}
        mlp = {p: grid.probes[("mine", "mlp", "sequence", p)] for p in range(NP)}
        res["editors"] = {}
        for ed, mode in (("PI", "pinv"), ("ND", "add_sub")):
            bb = best[ed]
            pr, _ = oa.linear_arm(model, bench, lin, tgt, cur, mode=mode, alpha=float(bb["alpha"]), points={int(bb["point"])})
            res["editors"][ed] = {"point": bb["point"], "alpha": bb["alpha"], **both(pr)}
        bb = best["GS"]
        pr, _ = oa.grad_steer_arm(model, bench, mlp, int(bb["point"]), alpha=float(bb["alpha"]),
                                  n_steps=int(settings["oth_gs_steps"]), beta=float(settings["oth_gs_beta"]), target_labels=tgt)
        res["editors"]["GS"] = {"point": bb["point"], "alpha": bb["alpha"], **both(pr)}
    if reachability:
        def im_best(sub, tag):
            sb = bench_of(sub)
            pu = pred([f["h"] for f in sub])
            recs, _ = oa.inverse_arms(model, sb, data, rules=rules, cache_dir=run_dir / "probes",
                                      n_games=int(settings["oth_probe_games"]), uns_probs=pu, log=None)
            im = [x for x in recs if x["editor"] == "IM"]
            bs = max(im, key=lambda x: x["edit_index_symdiff"])
            print(f"  IM [{tag}] n={len(sub)}: best pt{bs['point']} symdiff {bs['edit_index_symdiff']:+.3f} fid {bs['fidelity_ratio']:.2f}", flush=True)
            return {"n": len(sub), "unedited_symdiff": float(np.nanmean(edit_index_legal(pu, sb.legal_pre, sb.legal_post, "symdiff"))),
                    "best_point": bs["point"], "symdiff": bs["edit_index_symdiff"], "union": bs["edit_index_union"],
                    "fidelity": bs["fidelity_ratio"], "by_point_symdiff": {x["point"]: round(x["edit_index_symdiff"], 3) for x in im}}

        unreach = []
        for i in rng.permutation(len(cases))[n: n + 400]:
            c = cases[i]
            h, s = [int(x) for x in c["history"]], int(c["pos_int"])
            bst, d, _ = search_cf(h, s, rules)
            if bst is None or d > 0:
                unreach.append({"h": h, "hh": None, "s": s, "ori": c["ori_color"]})
            if len(unreach) >= max(len(found), 30):
                break
        res["IM_by_reachability"] = {"reachable (legal target board)": im_best(found, "reachable"),
                                     "unreachable (no game produces the target board)": im_best(unreach, "unreachable")}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="<topic>/<run> under runs/")
    ap.add_argument("--n", type=int, default=None, help="discworld: bench cases (default 1000); othello: cases searched (default 900)")
    ap.add_argument("--pct", type=float, default=95, help="othello: the ordinariness percentile")
    ap.add_argument("--editors", action="store_true")
    ap.add_argument("--reachability", action="store_true")
    ap.add_argument("--dry", action="store_true", help="print, write nothing (smoke runs at a small --n)")
    a = ap.parse_args()
    run_dir = _REPO / "runs" / a.run
    S = json.loads((run_dir / "scores.json").read_text())
    t0 = time.time()
    print(f"{a.run} ({S['env']} / {S['instance']})", flush=True)
    res = (othello(run_dir, S, a.n or 900, a.pct, a.editors, a.reachability) if S["env"] == "othello"
           else discworld(run_dir, S, a.n or 1000))
    res = {"run": a.run, "version": VERSION, "created": time.strftime("%Y-%m-%d %H:%M"), **res,
           "minutes": round((time.time() - t0) / 60, 1)}
    u, c = res["unedited"], res["ceiling"]
    key = "symdiff" if S["env"] == "othello" else "edit_index"
    print(f"  unedited {u[key]:+.3f}   CEILING {c[key]:+.3f}" + (f"  95% CI {c['symdiff_ci95']}" if "symdiff_ci95" in c else "")
          + f"   ({res['n_cases']} cases)", flush=True)
    if not a.dry:
        _write(run_dir, res)


if __name__ == "__main__":
    main()
