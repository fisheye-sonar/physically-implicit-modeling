"""How much of the oracle Δ does the output need? Patch h + t·Δ at the last position for
t in [0, 1.5] and score the canonical Edit Index — discworld (rollout) and Othello (legal-set
index via decode with an edit hook). A wide basin (the index saturating at small t) means many
latent states map to the edited environment state; a narrow one needs the whole Δ.
"""
import json, pickle, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from othello_alignment import search_cf, replay
from pim.environments.discworld import arms as dwa, bench as dwb
from pim.environments.discworld.bench import EF, K_ROLL, N_OBJ
from pim.environments.discworld.renderer import render_frame
from pim.environments.discworld.sim import fully_in_frustum
from pim.metrics.zone_editability import sim_config_from
from pim.environments.othello import corpus as oc
from pim.environments.othello.bench import benchmark_from_cases, cases_path
from pim.environments.othello.data import board_probs, canonical_vocab
from pim.metrics.set_editability import move_scorecard
from pim.models import load_checkpoint, n_points
from pim.probes.base import collect_residuals
DEV = "cuda"; TS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
out = {}
# ── discworld ──
model, _ = load_checkpoint(REPO / "runs/noise_ablation/L-dw-noiseless-20m/best_model.pt", device=DEV)
arr = dwb.bench_arrays(n=192, target="full", basis_name="frustum", data_dir=REPO / "datasets/discworld/dw-noiseless/eval")
sim = arr["sim"]; cfg = sim_config_from(sim, N_OBJ); pos, vel, eobj = arr["pos"], arr["vel"], arr["edit_object"]
n = len(eobj); ar = np.arange(n); dt = float(sim["dt"])
delta = pos[ar, EF, eobj] - (pos[ar, EF - 1, eobj] + vel[ar, EF - 1, eobj] * dt)
cf = pos[:, :EF].copy(); cf[ar, :, eobj] += delta[:, None, :]
ms = cfg.collision_margin * 2 * cfg.radius
keep = np.array([i for i in range(n) if fully_in_frustum(cf[i], cfg.radius, cfg) and (np.linalg.norm(cf[i, :, 0] - cf[i, :, 1], axis=-1) >= ms).all()])
refl = np.linspace(sim["refl_min"], sim["refl_max"], 2).astype(np.float32); rad = np.full(2, sim["radius"], np.float32)
ocf = np.stack([np.stack([render_frame(cf[i, f].astype(np.float32), rad, refl, cfg)[2] for f in range(EF)]) for i in keep]).astype(np.float32)
R = collect_residuals(model, arr["obs"][keep, :EF], batch=64)[:, :, -1]; Rcf = collect_residuals(model, ocf, batch=64)[:, :, -1]
b = dwb.load_bench(model, n=192, target="full", basis_name="frustum", data_dir=REPO / "datasets/discworld/dw-noiseless/eval", select=keep)
u = dwa.unsteered(model, b)
out["discworld"] = {}
for ell in (2, 4, 6):
    dwa.as_activations(model, ell); h0 = model.flat_state(b.state)
    D = torch.from_numpy(Rcf[ell] - R[ell]).to(DEV)
    row = {}
    for t in TS:
        roll = model.rollout_with_edit(b.state, ell, h0 + t * D, K_ROLL).cpu().numpy()
        c = dwa.score(model, b, roll, u); row[t] = (round(c["edit_index"], 3), round(c["fidelity_ratio"], 2))
    out["discworld"][ell] = row
    print("dw  pt", ell, " ".join(f"t{t}:{v[0]:+.2f}/{v[1]:.2f}" for t, v in row.items()), flush=True)
del model
# ── othello (both) ──
stoi = canonical_vocab()
for run, inst in (("runs/initial_othello_comparison/L-oth-20m", "oth-uniform"), ("runs/adjacency_ablation/L-oth-adjacent-20m", "oth-adjacent")):
    model, _ = load_checkpoint(REPO / run / "best_model.pt", device=DEV); rules = oc.rules_of(inst)
    cases = pickle.load(open(cases_path(inst), "rb")); rng = np.random.default_rng(0)
    found = []
    for i in rng.permutation(len(cases))[:300]:
        c = cases[i]; h = [int(x) for x in c["history"]]; s = int(c["pos_int"])
        best, d, _ = search_cf(h, s, rules)
        if best is not None and d <= 2:                      # near-exact counterfactuals only
            found.append((h, best[0], s, c["ori_color"]))
    bench = benchmark_from_cases([{"history": h, "pos_int": s, "ori_color": oc_} for h, _, s, oc_ in found], **rules)
    hcf = {}
    with torch.no_grad():
        for toks, ids in zip(bench.tokens, bench.case_ids):
            idx = torch.from_numpy(np.array([[stoi[x] for x in found[i][1]] for i in ids])).to(DEV)
            rs = model.residual_stack(idx)[:, :, -1]
            for j, i in enumerate(ids): hcf[i] = rs[:, j]
    out[inst] = {"n": len(found)}
    for ell in (2, 4, 6):
        row = {}
        for t in TS:
            probs = np.zeros((bench.n_cases, 64), np.float32)
            for toks, ids in zip(bench.tokens, bench.case_ids):
                idx = torch.from_numpy(toks).to(DEV)
                D = torch.stack([hcf[i][ell] for i in ids])
                def hook(layer, x, D=D):
                    if layer != ell: return x
                    o = x.clone(); o[:, -1] = x[:, -1] + t * (D - x[:, -1]); return o
                with torch.no_grad():
                    probs[ids] = board_probs(model.decode(idx, edit=hook), getattr(model, "output_kind", "logits"))
            card = move_scorecard(probs, bench.legal_pre, bench.legal_post); row[t] = round(card["edit_index_union"], 3)
        out[inst][ell] = row
        print(f"{inst:13s} pt {ell} (n={len(found)}) " + " ".join(f"t{t}:{v:+.2f}" for t, v in row.items()), flush=True)
    del model
json.dump(out, open(REPO / "experiments/edit_direction_alignment/scores/interpolation.json", "w"), indent=1, default=str)
