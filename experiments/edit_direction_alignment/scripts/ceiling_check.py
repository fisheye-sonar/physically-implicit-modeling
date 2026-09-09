"""Diagnostic: what is the CEILING for the Othello interpolation, and does a patch at the
FINAL residual point reach it?

(a) ceiling = the model run on the counterfactual history itself, scored against the same
    legal_pre / legal_post — if this is low, the counterfactual is not a good stand-in for
    the edited board and the interpolation measures nothing.
(b) how close is the counterfactual board's own legal set to legal_post (Jaccard, exact match)?
(c) patch at EVERY residual point 0..8 at t = 1 (splice h_cf into the original context) —
    at the last point the head reads the spliced vector directly, so it must equal (a).
"""
import pickle, sys, json
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from othello_alignment import replay, search_cf, mine_board
from pim.environments.othello import corpus as oc
from pim.environments.othello.bench import benchmark_from_cases, cases_path
from pim.environments.othello.data import board_probs, canonical_vocab
from pim.metrics.set_editability import move_scorecard
from pim.models import load_checkpoint, n_points
DEV = "cuda"; stoi = canonical_vocab()
out = {}
for run, inst in (("runs/initial_othello_comparison/L-oth-20m", "oth-uniform"),
                  ("runs/adjacency_ablation/L-oth-adjacent-20m", "oth-adjacent")):
    model, _ = load_checkpoint(REPO / run / "best_model.pt", device=DEV); rules = oc.rules_of(inst)
    NP = n_points(model)
    cases = pickle.load(open(cases_path(inst), "rb")); rng = np.random.default_rng(0)
    found = []
    for i in rng.permutation(len(cases))[:300]:
        c = cases[i]; h = [int(x) for x in c["history"]]; s = int(c["pos_int"])
        best, d, _ = search_cf(h, s, rules)
        if best is not None and d <= 2:
            found.append({"h": h, "hh": best[0], "s": s, "ori": c["ori_color"], "ham": d})
    bench = benchmark_from_cases([{"history": f["h"], "pos_int": f["s"], "ori_color": f["ori"]} for f in found], **rules)
    # (b) the counterfactual board's own legal set vs legal_post
    jac, exact = [], 0
    for i, f in enumerate(found):
        b = replay(f["hh"], rules); lc = set(b.get_valid_moves()); lp = set(bench.legal_post[i])
        jac.append(len(lc & lp) / max(1, len(lc | lp))); exact += int(lc == lp)
    # (a) ceiling: the model on the counterfactual histories
    probs_cf = np.zeros((bench.n_cases, 64), np.float32)
    probs_or = np.zeros((bench.n_cases, 64), np.float32)
    with torch.no_grad():
        for toks, ids in zip(bench.tokens, bench.case_ids):
            idx_cf = torch.from_numpy(np.array([[stoi[x] for x in found[i]["hh"]] for i in ids])).to(DEV)
            probs_cf[ids] = board_probs(model.decode(idx_cf), getattr(model, "output_kind", "logits"))
            probs_or[ids] = board_probs(model.decode(torch.from_numpy(toks).to(DEV)), getattr(model, "output_kind", "logits"))
    card_cf = move_scorecard(probs_cf, bench.legal_pre, bench.legal_post)
    ceil = card_cf["edit_index_union"]
    uned = move_scorecard(probs_or, bench.legal_pre, bench.legal_post)["edit_index_union"]
    ham = np.array([f["ham"] for f in found]); ex = ham == 0
    per_cf = np.array(card_cf["edit_index_union_per_case"], float)
    per_or = np.array(move_scorecard(probs_or, bench.legal_pre, bench.legal_post)["edit_index_union_per_case"], float)
    print(f"  EXACT-counterfactual subset (n={int(ex.sum())}): ceiling {np.nanmean(per_cf[ex]):+.3f}  unedited {np.nanmean(per_or[ex]):+.3f}"
          f"   | non-exact (n={int((~ex).sum())}): ceiling {np.nanmean(per_cf[~ex]):+.3f}", flush=True)
    # (c) splice at every point
    prof = {}
    hcf = {}
    with torch.no_grad():
        for toks, ids in zip(bench.tokens, bench.case_ids):
            idx_cf = torch.from_numpy(np.array([[stoi[x] for x in found[i]["hh"]] for i in ids])).to(DEV)
            rs = model.residual_stack(idx_cf)[:, :, -1]
            for j, i in enumerate(ids): hcf[i] = rs[:, j]
    for ell in range(NP):
        probs = np.zeros((bench.n_cases, 64), np.float32)
        for toks, ids in zip(bench.tokens, bench.case_ids):
            idx = torch.from_numpy(toks).to(DEV); D = torch.stack([hcf[i][ell] for i in ids])
            def hook(layer, x, D=D):
                if layer != ell: return x
                o = x.clone(); o[:, -1] = D; return o
            with torch.no_grad():
                probs[ids] = board_probs(model.decode(idx, edit=hook), getattr(model, "output_kind", "logits"))
        c_ = move_scorecard(probs, bench.legal_pre, bench.legal_post)
        pc = np.array(c_["edit_index_union_per_case"], float)
        prof[ell] = [round(c_["edit_index_union"], 3), round(float(np.nanmean(pc[ex])), 3)]
    out[inst] = {"n": len(found), "unedited": round(uned, 3), "ceiling_cf_history": round(ceil, 3),
                 "legal_jaccard_cf_vs_post": round(float(np.mean(jac)), 3), "legal_exact": exact,
                 "hamming_mean": round(float(np.mean([f["ham"] for f in found])), 2), "splice_by_point": prof}
    print(f"{inst}: n={len(found)} unedited {uned:+.3f} | CEILING (model on cf history) {ceil:+.3f} | "
          f"cf legal set vs legal_post: Jaccard {np.mean(jac):.3f}, exact {exact}/{len(found)} | Hamming {np.mean([f['ham'] for f in found]):.2f}", flush=True)
    print(f"  splice h_cf at point (all / exact-only): " + " ".join(f"{p}:{v[0]:+.2f}/{v[1]:+.2f}" for p, v in prof.items()), flush=True)
    del model
json.dump(out, open(REPO / "experiments/edit_direction_alignment/scores/ceiling_check.json", "w"), indent=1)
