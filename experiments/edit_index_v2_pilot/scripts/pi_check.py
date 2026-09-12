#!/usr/bin/env python
"""Is PI's flat +0.1 on paired Othello edits real or a wiring bug? (1) the pilot's multi-tile PI/ND
hooks vs the canonical linear_arm on Li's single-flip cases — must agree to numerical precision;
(2) on the 2-tile paired cases, a (point x alpha) sweep for PI with the read-out LANDING rate."""
import json, sys
from pathlib import Path
import numpy as np, torch
_REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(Path(__file__).parent))
import pilot as P
from pim.editors.nanda import addition_delta
from pim.editors.pinv import pinv_step, swap_class_logits
from pim.environments.othello import arms as oa, corpus as oc
from pim.environments.othello.bench import load_benchmark
from pim.environments.othello.data import N_CLASSES, N_TILES, board_probs, canonical_vocab
from pim.metrics.edit_index import edit_index_per_case
from pim.metrics.set_editability import edit_index_legal, uniform_over_legal
from pim.models import load_checkpoint
DEV = P.DEV
run_dir = _REPO / "runs/initial_othello_comparison/L-oth-20m"; inst = "oth-uniform"; rules = oc.rules_of(inst)
model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV); model.eval()
kind = getattr(model, "output_kind", "logits")
paths = oc.build(only=("probe",), instance=inst, log=lambda s: None)
grid = oa.fit_probe_grid(model, oc.probe_data(paths["probe"], 20_000, **rules), cache_dir=run_dir / "probes", log=None)
npts = model.n_layers + 1
lin = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(npts)}
stoi = canonical_vocab(); itos = {v: k for k, v in stoi.items()}

def multi_hook(pt, a, ed, probe, tiles, c_cls, t_cls, bsz, M):
    def hook(layer, x):
        if layer != pt: return x
        cur = x[:, -1]
        if ed == "ND":
            W = probe.net.weight.detach() / probe.x_std
            d = (W[tiles * N_CLASSES + t_cls] - W[tiles * N_CLASSES + c_cls]).sum(1)
            d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            delta = addition_delta(cur, d, a)
        else:
            lg = probe(cur)
            for m in range(M): lg = swap_class_logits(lg, tiles[:, m], c_cls[:, m], t_cls[:, m])
            delta = a * pinv_step(cur, lg.reshape(bsz, -1), probe, space="zspace")
        out = x.clone(); out[:, -1] = cur + delta; return out
    return hook

# ── (1) single-flip cases: pilot hook vs canonical linear_arm ──────────────────────────
bench = load_benchmark(inst)
for ed, mode, pt, a in (("PI", "pinv", 4, 3.0), ("ND", "add_sub", 4, 0.35)):
    can, card = oa.linear_arm(model, bench, lin, bench.tgt_lab, bench.cur_lab, mode=mode, alpha=a, points={pt})
    mine = np.zeros_like(can)
    for toks, ids in zip(bench.tokens, bench.case_ids):
        idx = torch.from_numpy(toks).to(DEV); bsz = len(ids)
        tiles = torch.from_numpy(bench.pos_int[ids]).to(DEV)[:, None]
        c = torch.from_numpy(bench.cur_lab[ids]).to(DEV)[:, None]; t = torch.from_numpy(bench.tgt_lab[ids]).to(DEV)[:, None]
        with torch.no_grad():
            mine[ids] = board_probs(model.decode(idx, edit=multi_hook(pt, a, ed, lin[pt], tiles, c, t, bsz, 1)), kind)
    print(f"(1) {ed} pt{pt} α{a}: canonical EI_union {card['edit_index_union']:+.4f} | pilot hook "
          f"{np.nanmean(edit_index_legal(mine, bench.legal_pre, bench.legal_post, 'union')):+.4f} | max|Δprob| {np.abs(can - mine).max():.2e}")

# ── (2) the 2-tile paired cases: PI sweep with landing ─────────────────────────────────
hists = [[itos[int(t)] for t in row] for toks, ids in zip(bench.tokens, bench.case_ids) for row in toks]
pairs, _ = P.make_pairs(hists, rules, P.K_BACK, P.POOL_OTH, np.random.default_rng(P.SEED))
two = [p for p in pairs if p["n_changed"] == 2]
print(f"\n(2) {len(two)} two-tile pairs (of {len(pairs)})")
by_len = {}
for i, p in enumerate(two): by_len.setdefault(len(p["hist_a"]), []).append(i)
n = len(two)
la = [p["legal_a"] for p in two]; lb = [p["legal_b"] for p in two]
supp = np.zeros((n, N_TILES), bool)
for i in range(n): supp[i, sorted(set(la[i]) | set(lb[i]))] = True
p_a = np.zeros((n, N_TILES), np.float32); p_b = np.zeros((n, N_TILES), np.float32)
groups = []
for L, members in sorted(by_len.items()):
    ids = np.array(members)
    ta = torch.tensor([[stoi[s] for s in two[i]["hist_a"]] for i in ids], device=DEV)
    tb = torch.tensor([[stoi[s] for s in two[i]["hist_b"]] for i in ids], device=DEV)
    with torch.no_grad():
        p_a[ids] = board_probs(model.decode(ta), kind); p_b[ids] = board_probs(model.decode(tb), kind)
    cur_b = torch.tensor(np.stack([two[i]["mine_a"] for i in ids]), device=DEV)
    tgt_b = torch.tensor(np.stack([two[i]["mine_b"] for i in ids]), device=DEV)
    chg = torch.tensor(np.stack([two[i]["changed"] for i in ids]), device=DEV)
    tiles = torch.stack([torch.where(chg[r])[0] for r in range(len(ids))])         # (B, 2) exactly two
    groups.append((ids, ta, tiles, cur_b.gather(1, tiles), tgt_b.gather(1, tiles), cur_b, tgt_b))
# what kinds of class change are the two tiles? (vacate: X->BLANK, occupy: BLANK->X)
from pim.environments.othello.data import BLANK, MINE, THEIRS
kinds = {}
for p in two:
    for t in np.where(p["changed"])[0]:
        kinds[(int(p["mine_a"][t]), int(p["mine_b"][t]))] = kinds.get((int(p["mine_a"][t]), int(p["mine_b"][t])), 0) + 1
nm = {BLANK: "BLANK", MINE: "MINE", THEIRS: "THEIRS"}
print("   tile transitions:", {f"{nm[a]}->{nm[b]}": c for (a, b), c in kinds.items()})
print(f"   unedited: p_A legal mass on legal_A {np.mean([p_a[i, la[i]].sum() for i in range(n)]):.3f}; ceiling_v1 "
      f"{np.nanmean(edit_index_legal(p_b, la, lb, 'union')):+.3f}")

def run(ed, pt, a):
    pr = np.zeros((n, N_TILES), np.float32); land = []
    for ids, ta, tiles, c, t, cur_b, tgt_b in groups:
        probe = lin[pt]; bsz = len(ids)
        h = multi_hook(pt, a, ed, probe, tiles, c, t, bsz, 2)
        with torch.no_grad():
            rs = model.residual_stack(ta); x = rs[pt][:, -1]
            xe = h(pt, rs[pt].clone())[:, -1]
            lab = probe(xe).argmax(-1)                                                    # (B, 64)
            land.append((lab.gather(1, tiles) == t).all(1).float().cpu().numpy())
            pr[ids] = board_probs(model.decode(ta, edit=h), kind)
    ei2 = np.nanmean(edit_index_per_case(pr, p_b, p_a, supp)); ei1 = np.nanmean(edit_index_legal(pr, la, lb, "union"))
    g = np.sqrt(((pr - p_b) ** 2).mean(1)).mean() / np.sqrt(((p_a - p_b) ** 2).mean(1)).mean()
    return ei2, ei1, g, float(np.concatenate(land).mean())

print(f"\n   {'editor':>6} {'pt':>3} {'α':>5} | {'EI_v2':>7} {'EI_v1':>7} {'guard':>6} {'landed':>7}")
for pt in (2, 3, 4, 5, 6):
    for a in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
        e2, e1, g, ld = run("PI", pt, a)
        print(f"   {'PI':>6} {pt:>3} {a:>5} | {e2:>+7.3f} {e1:>+7.3f} {g:>6.2f} {ld:>7.2f}")
for pt, a in ((4, 0.35), (4, 0.7), (3, 0.35), (5, 0.35)):
    e2, e1, g, ld = run("ND", pt, a)
    print(f"   {'ND':>6} {pt:>3} {a:>5} | {e2:>+7.3f} {e1:>+7.3f} {g:>6.2f} {ld:>7.2f}")
