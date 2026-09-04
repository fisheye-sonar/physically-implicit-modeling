"""Can ONE full-state probe set replace the pos-only set? (Sevan's question, 2026-09-01)

If a probe fitted on the full 8-dim state can drive position alone as well as a probe
fitted on position alone, we halve the probe budget: 4 fits per (model, basis) become 2.

Three arms per editor, same bench, same alpha grid, same residual points:
  pos-probe        the current pos-only probe, driving its 4 dims          (baseline)
  full-probe[pos]  the FULL probe, PI restricted to its 4 position dims    (the test)
  full-probe[all]  the FULL probe driving all 8 dims                       (reference)

GS gets the analogous treatment through its change_mask: driving position only leaves
the velocity read-outs as hold-the-rest constraints rather than targets.
"""
import json, sys
import numpy as np
import torch
from pim.models import load_checkpoint
from pim.environments.discworld import bench as dwb

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/initial_othello_comparison/L-dw-20m"
BASIS = sys.argv[2] if len(sys.argv) > 2 else "cartesian"
N_SEQ = 30_000
A_PI = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0, 100.0, 175.0)
A_GS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5)

inst = json.load(open(f"{RUN}/config.json"))["data"]["instance"]
root = f"datasets/discworld/{inst}"
m, info = load_checkpoint(f"{RUN}/best_model.pt", device=dwb.DEV)
NP = m.n_layers + 1
print(f"{RUN}  basis={BASIS}  instance={inst}  val {info.val_loss:.5f}\n", flush=True)

probes, benches = {}, {}
for tgt in ("pos", "full"):
    benches[tgt] = dwb.load_bench(m, n=192, target=tgt, basis_name=BASIS,
                                  data_dir=f"{root}/eval")
    probes[tgt] = {
        fam: dwb.fit_probes(m, target=tgt, n_seq=N_SEQ, family=fam, basis_name=BASIS,
                            data_dir=f"{root}/probe", cache_dir=f"{RUN}/probes", log=None)
        for fam in ("linear", "mlp")}
u = dwb.unsteered(m, benches["pos"])
POS_DIMS = list(range(4))          # the 4 position read-outs inside the 8-dim full target

def best(recs):
    b = max(recs, key=lambda r: r["edit_index"])
    return b["edit_index"], dwb.fidelity_ratio(b, u), b["point"], b["alpha"]

rows = {}
# --- PI ---------------------------------------------------------------------
recs = dwb.pinv_arm(m, benches["pos"], probes["pos"]["linear"], A_PI)
rows["PI  pos-probe"] = best(recs)
recs = []
for ell, (pr, _) in probes["full"]["linear"].items():
    dwb.as_activations(m, ell)
    h0 = m.flat_state(benches["full"].state)
    from pim.editors.pinv import pinv_step
    step = pinv_step(h0, benches["full"].tgt, pr, space="zspace", dims=POS_DIMS)
    for a in A_PI:
        roll = m.rollout_with_edit(benches["full"].state, ell, h0 + a * step,
                                   dwb.K_ROLL).cpu().numpy()
        recs.append({"editor": "PI", "point": ell, "alpha": a,
                     **dwb.score(m, benches["pos"], roll)})   # scored on the SAME bench
rows["PI  full-probe[pos dims]"] = best(recs)
rows["PI  full-probe[all dims]"] = best(dwb.pinv_arm(m, benches["full"],
                                                     probes["full"]["linear"], A_PI))
# --- GS ---------------------------------------------------------------------
rows["GS  pos-probe"] = best(dwb.grad_steer_arm(m, benches["pos"], probes["pos"]["mlp"],
                                                range(NP), A_GS))
bf = benches["full"]
cm_pos = bf.change_mask.clone(); cm_pos[:, 4:] = False     # velocity -> hold-the-rest
# zones and gt_roll are WORLD-space ray masks, identical across targets and bases, so
# every arm below is scored against exactly the same ground truth.
bench_fp = dwb.Bench(bf.obs, bf.gt_roll, bf.zones, bf.tgt, cm_pos, bf.out_dims,
                     bf.state, bf.n)
rows["GS  full-probe[pos dims]"] = best(
    dwb.grad_steer_arm(m, bench_fp, probes["full"]["mlp"], range(NP), A_GS))
rows["GS  full-probe[all dims]"] = best(dwb.grad_steer_arm(m, bf, probes["full"]["mlp"],
                                                           range(NP), A_GS))

print(f"unedited EI {u['edit_index']:+.4f}\n")
print(f"{'arm':<26}{'EI':>9}{'fid':>7}{'pt':>4}{'alpha':>8}")
for k, (ei, fid, pt, a) in rows.items():
    print(f"{k:<26}{ei:>+9.4f}{fid:>7.2f}{pt:>4}{a:>8g}")
json.dump({k: dict(ei=v[0], fid=v[1], point=v[2], alpha=v[3]) for k, v in rows.items()},
          open(f"experiments/full_vs_pos/scores/full_vs_pos_{RUN.split('/')[-1]}_{BASIS}.json", "w"), indent=1)
print(f"\nwrote experiments/full_vs_pos/scores/full_vs_pos_{RUN.split('/')[-1]}_{BASIS}.json")
