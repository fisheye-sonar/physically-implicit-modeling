"""Extended α sweep for oth-adjacent's linear editors (ND, PI pinned at the canonical grid's
edge) plus a read-out landing check, through the canonical Othello arms and probes."""
import json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from pim.environments.othello import arms as oa, corpus as oc
from pim.environments.othello import case_targets, load_benchmark
from pim.environments.othello.data import N_CLASSES, N_TILES, canonical_vocab, tokens_and_labels
from pim.metrics.set_editability import move_fidelity_ratio
from pim.models import load_checkpoint, n_points
DEV = "cuda"
run = REPO / "runs/adjacency_ablation/L-oth-adjacent-20m"
inst = "oth-adjacent"
model, _ = load_checkpoint(run / "best_model.pt", device=DEV)
tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("probe",), instance=inst)["probe"])
itos = {v: k for k, v in canonical_vocab().items()}
data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(tok[:20000], ln[:20000])], **oc.rules_of(inst))
grid = oa.fit_probe_grid(model, data, cache_dir=run / "probes", log=None)      # cache hits
bench = load_benchmark(inst); cur, tgt = case_targets(bench)
uns = oa.unsteered_probs(model, bench); u = oa.unsteered(model, bench)
NP = n_points(model)
lin = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(NP)}
out = {"unedited": {k: v for k, v in u.items() if isinstance(v, (int, float))}, "arms": []}
# read-out landing: does the probe's own reading of the edited residual show the target class at the tile?
def landed(p, alpha, mode):
    hits = []
    for toks, ids in zip(bench.tokens, bench.case_ids):
        idx = torch.from_numpy(toks).to(DEV); bsz = len(ids)
        sq = torch.from_numpy(bench.pos_int[ids]).to(DEV); td = torch.from_numpy(tgt[ids]).to(DEV); cd = torch.from_numpy(cur[ids]).to(DEV)
        probe = lin[p]
        def hook(layer, x):
            if layer != p: return x
            from pim.editors.nanda import probe_direction, addition_delta
            from pim.editors.pinv import pinv_step
            c = x[:, -1]
            if mode == "add_sub":
                d = probe_direction(probe, sq * N_CLASSES + td, per_sample=True, subtract_rows=sq * N_CLASSES + cd)
                delta = addition_delta(c, d, alpha)
            else:
                lg = probe(c).clone(); ar = torch.arange(bsz, device=DEV); sel = lg[ar, sq]; new = sel.clone()
                new[ar, td] = sel[ar, cd]; new[ar, cd] = sel[ar, td]; lg[ar, sq] = new
                delta = alpha * pinv_step(c, lg.view(bsz, -1), probe, space="zspace")
            o = x.clone(); o[:, -1] = c + delta
            with torch.no_grad():
                hits.append((probe(o[:, -1]).argmax(-1)[torch.arange(bsz), sq] == td).float().mean().item())
            return o
        with torch.no_grad(): model.decode(idx, edit=hook)
    return float(np.mean(hits))
for mode, label, alphas in (("add_sub", "ND", (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)), ("pinv", "PI", (1.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0))):
    for p in range(NP):
        for a in alphas:
            pr, card = oa.linear_arm(model, bench, lin, tgt, cur, mode=mode, alpha=a, points={p})
            rec = {"editor": label, "point": p, "alpha": a, "fidelity_ratio": move_fidelity_ratio(pr, uns, bench.legal_post),
                   "readout_landed": landed(p, a, mode), **{k: v for k, v in card.items() if isinstance(v, (int, float))}}
            out["arms"].append(rec)
        best = max((r for r in out["arms"] if r["editor"] == label and r["point"] == p), key=lambda r: r["edit_index_union"])
        print(f"{label} pt{p}: best EI {best['edit_index_union']:+.3f} fid {best['fidelity_ratio']:.2f} α {best['alpha']} landed {best['readout_landed']:.2f} | "
              + " ".join(f"a{r['alpha']:g}:{r['edit_index_union']:+.2f}/{r['fidelity_ratio']:.1f}/L{r['readout_landed']:.2f}" for r in out['arms'] if r['editor']==label and r['point']==p), flush=True)
Path(REPO / "experiments/adjacency_ablation/scores").mkdir(exist_ok=True)
json.dump(out, open(REPO / "experiments/adjacency_ablation/scores/extended_alpha.json", "w"), indent=1)
