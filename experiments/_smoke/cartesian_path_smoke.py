"""Smoke of the CARTESIAN scoring path on one run (tiny n_seq, scratch probe cache — never the run's cache):
bench in cartesian, linear + MLP probe fits, PI at two α, one GS arm, observation floor probe. Prints timings."""
import json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from pim.environments.discworld import arms as dwa, bench as dwb
from pim.models import load_checkpoint, n_points
run = REPO / "runs/ray_ablation/L-dw-8ray-20m"; inst = json.loads((run / "config.json").read_text())["data"]["instance"]
S = json.loads((run / "scores.json").read_text())["settings"]
model, _ = load_checkpoint(run / "best_model.pt", device=dwb.DEV); model.eval()
cache = REPO / ".scratch/cartesian_smoke_probes"; cache.mkdir(parents=True, exist_ok=True)
t0 = time.time()
b = dwb.load_bench(model, n=50, target="full", basis_name="cartesian", instance=inst); u = dwa.unsteered(model, b)
print(f"bench cartesian: {b.n} cases, tgt {tuple(b.tgt.shape)}, unedited {u['edit_index']:+.3f}  [{time.time()-t0:.0f}s]", flush=True)
recipe = dwa.probe_recipe("full", inst, n_seq=2000)
t1 = time.time(); lin = dwa.fit_probes(model, target="full", family="linear", basis_name="cartesian", cache_dir=cache, log=None, **recipe)
print(f"linear probes (9 pts, 2000 seq): {time.time()-t1:.0f}s; skill pt4 R² {lin[4][1].get('r2', lin[4][1].get('r2_test'))}", flush=True)
t1 = time.time(); mlp = dwa.fit_probes(model, target="full", family="mlp", basis_name="cartesian", cache_dir=cache, log=None, **recipe)
print(f"mlp probes (9 pts, 2000 seq): {time.time()-t1:.0f}s", flush=True)
t1 = time.time(); rows = dwa.pinv_arm(model, b, lin, [1.0, 60.0], space="zspace", dims="all")
for r in rows: r["fidelity_ratio"] = dwa.fidelity_ratio(r, u)
best = max(rows, key=lambda r: r["edit_index"]); print(f"PI 2 α × 9 pts: {time.time()-t1:.0f}s; best {best['edit_index']:+.3f}/{best['fidelity_ratio']:.2f}", flush=True)
t1 = time.time(); g = dwa.grad_steer_arm(model, b, mlp, [4], [0.05], n_steps=S["dw_gs_steps"], beta=S["dw_gs_beta"], dims="all")
print(f"GS 1 arm: {time.time()-t1:.0f}s; {g[0]['edit_index']:+.3f}", flush=True)
from pim.environments import layout
t1 = time.time(); _, st = dwa.observation_probes(target="full", n_seq=2000, family="linear", basis_name="cartesian", span=int(getattr(model, "state_span", 39)), probe={"instance": inst, "size": "120k"}, cache_dir=cache, log=None)
print(f"observation floor probe (linear, 2000 seq): {time.time()-t1:.0f}s; keys {list(st.keys())[:5]}", flush=True)
print(f"SMOKE OK [{time.time()-t0:.0f}s total]")
