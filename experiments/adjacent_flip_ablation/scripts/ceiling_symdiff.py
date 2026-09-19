"""The Edit-Index CEILING under BOTH constructions (2026-09-15).

`honesty_check_v2.py` (2026-09-11) established the honest ceiling — the model run on a true
counterfactual history whose board equals the flipped board, kept only when the model treats
that history as ordinary — but recorded the UNION Edit Index only. The Othello headline has
been the SYMMETRIC DIFFERENCE since 2026-09-12, so the paper needs the ceiling on that axis.

Same cases as v2 (same seed, same n, same ordinariness filter), same canonical best arms read
from each run's `scores.json`; reports union AND symdiff for the unedited model, the true
counterfactual (the ceiling), PI, ND and GS on the SAME cases. No training, no canonical change.

    python experiments/adjacent_flip_ablation/scripts/ceiling_symdiff.py --run runs/initial_othello_comparison/L-oth-20m
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/edit_direction_alignment/scripts"))
from othello_alignment import replay, search_cf  # noqa: E402
from pim.environments.othello import arms as oa, corpus as oc  # noqa: E402
from pim.environments.othello import case_targets  # noqa: E402
from pim.environments.othello.bench import benchmark_from_cases, cases_path  # noqa: E402
from pim.environments.othello.data import board_probs, canonical_vocab, tokens_and_labels  # noqa: E402
from pim.metrics.set_editability import edit_index_legal, move_fidelity_ratio, move_rmse_per_case  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402

DEV = "cuda"
EXP = REPO / "experiments/adjacent_flip_ablation"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--pct", type=float, default=95)
    a = ap.parse_args()
    run_dir = REPO / a.run
    inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
    rules = oc.rules_of(inst)
    S = json.loads((run_dir / "scores.json").read_text())
    best, settings = S["best"], S["settings"]
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV)
    NP = n_points(model)
    stoi = canonical_vocab()
    itos = {v: k for k, v in stoi.items()}

    def pred(hists):
        out = np.zeros((len(hists), 64), np.float32)
        by: dict[int, list[int]] = {}
        for i, h in enumerate(hists):
            by.setdefault(len(h), []).append(i)
        with torch.no_grad():
            for _, ids in by.items():
                idx = torch.from_numpy(np.array([[stoi[x] for x in hists[i]] for i in ids])).to(DEV)
                out[ids] = board_probs(model.decode(idx), getattr(model, "output_kind", "logits"))
        return out

    def own_legal(hs):
        return [sorted(replay(h, rules).get_valid_moves()) for h in hs]

    # ordinariness threshold by prefix length, exactly as honesty_check_v2
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
    thr = {L: float(np.percentile(v, a.pct)) for L, v in ho.items()}

    cases = pickle.load(open(cases_path(inst), "rb"))
    rng = np.random.default_rng(0)
    raw = []
    for i in rng.permutation(len(cases))[: a.n]:
        c = cases[i]
        h = [int(x) for x in c["history"]]
        s = int(c["pos_int"])
        bst, d, _ = search_cf(h, s, rules)
        if bst is not None and d == 0:
            raw.append({"h": h, "hh": bst[0], "s": s, "ori": c["ori_color"]})
    p_cf_all = pred([f["hh"] for f in raw])
    rm = move_rmse_per_case(p_cf_all, own_legal([f["hh"] for f in raw]))
    ordn = np.array([x <= thr[len(f["hh"])] for f, x in zip(raw, rm)])
    found = [raw[i] for i in np.where(ordn)[0]]
    p_cf = p_cf_all[ordn]
    print(f"{a.run}: {len(raw)} exact counterfactual boards, {len(found)} ordinary", flush=True)

    bench = benchmark_from_cases([{"history": f["h"], "pos_int": f["s"], "ori_color": f["ori"]} for f in found], **rules)
    pre, post = bench.legal_pre, bench.legal_post
    cur, tgt = case_targets(bench)
    p_un = pred([f["h"] for f in found])

    def both(probs):
        return {"union": float(np.nanmean(edit_index_legal(probs, pre, post, "union"))),
                "symdiff": float(np.nanmean(edit_index_legal(probs, pre, post, "symdiff"))),
                "fidelity": float(move_fidelity_ratio(probs, p_un, post))}

    res = {"run": a.run, "instance": inst, "n_exact": len(raw), "n_ordinary": len(found),
           "unedited": both(p_un), "ceiling_true_counterfactual": both(p_cf), "editors": {}}

    ptok, pln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("probe",), instance=inst)["probe"])
    data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(ptok[:20000], pln[:20000])], **rules)
    grid = oa.fit_probe_grid(model, data, cache_dir=run_dir / "probes", log=None)
    lin = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(NP)}
    mlp = {p: grid.probes[("mine", "mlp", "sequence", p)] for p in range(NP)}
    for ed, mode in (("PI", "pinv"), ("ND", "add_sub")):
        b = best[ed]
        pr, _ = oa.linear_arm(model, bench, lin, tgt, cur, mode=mode, alpha=float(b["alpha"]), points={int(b["point"])})
        res["editors"][ed] = {"point": b["point"], "alpha": b["alpha"], **both(pr)}
    b = best["GS"]
    pr, _ = oa.grad_steer_arm(model, bench, mlp, int(b["point"]), alpha=float(b["alpha"]),
                              n_steps=int(settings["oth_gs_steps"]), beta=float(settings["oth_gs_beta"]),
                              target_labels=tgt)
    res["editors"]["GS"] = {"point": b["point"], "alpha": b["alpha"], **both(pr)}

    # IM on the SAME ordinary reachable cases, and on a matched sample of cases with NO exact
    # counterfactual board (the case-level test of whether IM's failure tracks reachability).
    ptok_, pln_ = ptok, pln
    def im_best(sub_cases, tag):
        b = benchmark_from_cases([{"history": f["h"], "pos_int": f["s"], "ori_color": f["ori"]} for f in sub_cases], **rules)
        pu = pred([f["h"] for f in sub_cases])
        recs, _ = oa.inverse_arms(model, b, data, rules=rules, cache_dir=run_dir / "probes",
                                  n_games=int(settings["oth_probe_games"]), uns_probs=pu, log=None)
        im = [r for r in recs if r["editor"] == "IM"]
        bst = max(im, key=lambda r: r["edit_index_symdiff"])
        out = {"n": len(sub_cases), "unedited_symdiff": float(np.nanmean(edit_index_legal(pu, b.legal_pre, b.legal_post, "symdiff"))),
               "best_point": bst["point"], "symdiff": bst["edit_index_symdiff"], "union": bst["edit_index_union"],
               "fidelity": bst["fidelity_ratio"],
               "by_point_symdiff": {r["point"]: round(r["edit_index_symdiff"], 3) for r in im}}
        print(f"  IM [{tag}] n={len(sub_cases)}: unedited {out['unedited_symdiff']:+.3f}  best pt{bst['point']} symdiff {bst['edit_index_symdiff']:+.3f} union {bst['edit_index_union']:+.3f} fid {bst['fidelity_ratio']:.2f}", flush=True)
        return out
    res["IM_ordinary_reachable"] = im_best(found, "ordinary reachable")
    unreach = []
    for i in rng.permutation(len(cases))[a.n: a.n + 400]:
        c = cases[i]; h = [int(x) for x in c["history"]]; s = int(c["pos_int"])
        bst, d, _ = search_cf(h, s, rules)
        if bst is None or d > 0:
            unreach.append({"h": h, "hh": None, "s": s, "ori": c["ori_color"]})
        if len(unreach) >= max(len(found), 30):
            break
    res["IM_no_counterfactual"] = im_best(unreach, "no exact counterfactual")

    for k in ("unedited", "ceiling_true_counterfactual"):
        v = res[k]
        print(f"  {k:28s} union {v['union']:+.3f}  symdiff {v['symdiff']:+.3f}  fid {v['fidelity']:.2f}", flush=True)
    for ed, v in res["editors"].items():
        frac = v["symdiff"] / res["ceiling_true_counterfactual"]["symdiff"]
        print(f"  {ed + ' (pt' + str(v['point']) + ' a' + str(v['alpha']) + ')':28s} union {v['union']:+.3f}  "
              f"symdiff {v['symdiff']:+.3f}  fid {v['fidelity']:.2f}  symdiff/ceiling {frac:.2f}", flush=True)
    out = EXP / "scores" / f"ceiling_symdiff_{Path(a.run).name}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"  wrote {out.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
