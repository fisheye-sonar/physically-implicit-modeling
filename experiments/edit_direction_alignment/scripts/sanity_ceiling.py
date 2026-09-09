"""Three checks, no training, no canonical changes:
(1) is the model really ~uniform-over-legal on ordinary held-out games?
(2) are the counterfactual histories ordinary by that same measure (vs their OWN legal set)?
(3) head-to-head on the SAME cases: the canonical editor's Edit Index vs the true
    counterfactual history's Edit Index, per case.
"""
import pickle, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from othello_alignment import replay, search_cf
from pim.environments.othello import arms as oa, corpus as oc
from pim.environments.othello import case_targets
from pim.environments.othello.bench import benchmark_from_cases, cases_path
from pim.environments.othello.data import board_probs, canonical_vocab, tokens_and_labels
from pim.metrics.set_editability import move_scorecard, uniform_over_legal
from pim.models import load_checkpoint, n_points
DEV = "cuda"; stoi = canonical_vocab(); itos = {v: k for k, v in stoi.items()}
inst = "oth-uniform"; rules = oc.rules_of(inst)
model, _ = load_checkpoint(REPO / "runs/initial_othello_comparison/L-oth-20m/best_model.pt", device=DEV)

def pred_last(hists):
    """model's next-move distribution at the end of each history (bucketed by length)."""
    out = np.zeros((len(hists), 64), np.float32)
    by = {}
    for i, h in enumerate(hists): by.setdefault(len(h), []).append(i)
    with torch.no_grad():
        for L, ids in by.items():
            idx = torch.from_numpy(np.array([[stoi[x] for x in hists[i]] for i in ids])).to(DEV)
            out[ids] = board_probs(model.decode(idx), getattr(model, "output_kind", "logits"))
    return out

def quality(probs, legals):
    mass = np.array([probs[i, L].sum() for i, L in enumerate(legals)])
    rmse = np.array([np.sqrt(((probs[i] - uniform_over_legal(L, 64)) ** 2)[sorted(L)].mean()) for i, L in enumerate(legals)])
    return mass.mean(), rmse.mean()

# (1) ordinary held-out games, prefixes matching the bench's length mix (5..30)
tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("test",), instance=inst)["test"])
rng = np.random.default_rng(0)
hs = []
for r in range(400):
    L = int(rng.integers(5, 31))
    h = [int(itos[int(t)]) for t in tok[r][: int(ln[r])]]
    if len(h) > L: hs.append(h[:L])
legal_own = [sorted(replay(h, rules).get_valid_moves()) for h in hs]
m, e = quality(pred_last(hs), legal_own)
print(f"(1) held-out real prefixes  n={len(hs)}: legal mass {m:.4f}  rmse to uniform-over-legal {e:.5f}", flush=True)

# (2) the counterfactual histories, against their OWN legal sets
cases = pickle.load(open(cases_path(inst), "rb")); rng = np.random.default_rng(0)
found = []
for i in rng.permutation(len(cases))[:300]:
    c = cases[i]; h = [int(x) for x in c["history"]]; s = int(c["pos_int"])
    best, d, _ = search_cf(h, s, rules)
    if best is not None and d <= 2:
        found.append({"h": h, "hh": best[0], "s": s, "ori": c["ori_color"], "ham": d, "i": int(i)})
cf_legal = [sorted(replay(f["hh"], rules).get_valid_moves()) for f in found]
probs_cf = pred_last([f["hh"] for f in found])
m2, e2 = quality(probs_cf, cf_legal)
print(f"(2) counterfactual histories n={len(found)}: legal mass {m2:.4f}  rmse to their OWN uniform {e2:.5f}"
      f"   -> {'ordinary' if abs(e2 - e) < 0.01 else 'ANOMALOUS'}", flush=True)

# (3) head-to-head on the same cases
bench = benchmark_from_cases([{"history": f["h"], "pos_int": f["s"], "ori_color": f["ori"]} for f in found], **rules)
cur, tgt = case_targets(bench)
data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(tok[:20000], ln[:20000])], **rules)
grid = oa.fit_probe_grid(model, data, cache_dir=REPO / "runs/initial_othello_comparison/L-oth-20m/probes", log=None)
lin = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(n_points(model))}
pr_pi, card_pi = oa.linear_arm(model, bench, lin, tgt, cur, mode="pinv", alpha=3.0, points={4})
pr_nd, card_nd = oa.linear_arm(model, bench, lin, tgt, cur, mode="add_sub", alpha=0.35, points={4})
c_cf = move_scorecard(probs_cf, bench.legal_pre, bench.legal_post)
c_un = move_scorecard(pred_last([f["h"] for f in found]), bench.legal_pre, bench.legal_post)
per = lambda c: np.array(c["edit_index_union_per_case"], float)
print(f"(3) SAME {len(found)} cases: unedited {c_un['edit_index_union']:+.3f} | true counterfactual {c_cf['edit_index_union']:+.3f} | "
      f"PI(pt4 a3) {card_pi['edit_index_union']:+.3f} | ND(pt4 a0.35) {card_nd['edit_index_union']:+.3f}", flush=True)
setex = np.array([set(cf_legal[i]) == set(bench.legal_post[i]) for i in range(len(found))])
print(f"    on the {int(setex.sum())} legal-set-exact cases: counterfactual {np.nanmean(per(c_cf)[setex]):+.3f} | "
      f"PI {np.nanmean(per(card_pi)[setex]):+.3f} | ND {np.nanmean(per(card_nd)[setex]):+.3f}", flush=True)
# what do the distributions look like on those cases?
for nm, P in (("counterfactual", probs_cf), ("PI", pr_pi), ("ND", pr_nd)):
    mm = np.array([P[i, bench.legal_post[i]].sum() for i in np.where(setex)[0]])
    rr = np.array([np.sqrt(((P[i] - uniform_over_legal(bench.legal_post[i], 64)) ** 2)[sorted(set(bench.legal_pre[i]) | set(bench.legal_post[i]))].mean()) for i in np.where(setex)[0]])
    rp = np.array([np.sqrt(((P[i] - uniform_over_legal(bench.legal_pre[i], 64)) ** 2)[sorted(set(bench.legal_pre[i]) | set(bench.legal_post[i]))].mean()) for i in np.where(setex)[0]])
    print(f"    {nm:14s}: mass on legal_post {mm.mean():.3f} | rmse to unif_post {rr.mean():.5f} | to unif_pre {rp.mean():.5f}", flush=True)
