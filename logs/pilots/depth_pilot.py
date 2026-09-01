"""Depth-coordinate pilot on L-dw-20m: which frustum depth decodes best?

LINEAR probes only (fast), target='full' so VELOCITY is visible — the whole point, since
the frustum hypothesis is specifically that image-plane motion (u̇, with its 1/y² weight)
is what the model encodes while Cartesian ẋ is the wrong target. Reports per-component
skill, and separates position from velocity rather than quoting the variance-weighted
aggregate (position outweighs velocity ~1000:1).
"""
import json, sys, numpy as np
from pim.models import load_checkpoint
from pim.environments.discworld import bench as dwb

RUN = 'runs/initial_othello_comparison/L-dw-20m'
INST = 'datasets/discworld/dw-pn04'
BASES = ["cartesian", "y", "rho", "inv_y", "inv_rho", "width"]
# ⛔ MEMORY: `collect_residuals` materialises ALL residual points at once —
# n_seq x 39 frames x 512 dims x 9 points x 4 B. At the canonical 30,000 that is
# 21.6 GB, and looping six bases stacked allocations until the OOM killer took the
# session down (2026-09-01). 8,000 is 5.8 GB and still 609 rows per linear-probe
# parameter, far more than enough for a RELATIVE comparison between bases.
N_SEQ = 8_000
import gc
m, info = load_checkpoint(f'{RUN}/best_model.pt', device=dwb.DEV)
print(f"{RUN}  val {info.val_loss:.5f}  |  n_seq={N_SEQ:,} "
      f"({N_SEQ*39*512*9*4/1e9:.1f} GB per basis)\n", flush=True)

rows = {}
for basis in BASES:
    lin = dwb.fit_probes(m, target="full", n_seq=N_SEQ, family="linear",
                         basis_name=basis, data_dir=f'{INST}/probe',
                         cache_dir=f'{RUN}/probes', log=None)
    # best residual point by POSITION skill and by VELOCITY skill, reported separately
    per_pt = [(p, np.array(s["per_dim_r2"])) for p, (_, s) in lin.items()]
    pos_best = max(per_pt, key=lambda t: t[1][:4].mean())
    vel_best = max(per_pt, key=lambda t: t[1][4:].mean())
    del lin; gc.collect()
    rows[basis] = dict(pos=pos_best[1][:4].mean(), pos_pt=pos_best[0],
                       vel=vel_best[1][4:].mean(), vel_pt=vel_best[0],
                       perdim=pos_best[1])
    r = rows[basis]
    print(f"{basis:<10} pos {r['pos']:+.4f} (pt{r['pos_pt']})   "
          f"vel {r['vel']:+.4f} (pt{r['vel_pt']})   "
          f"per-dim@pos-best " + " ".join(f"{v:+.3f}" for v in r['perdim']), flush=True)

print(f"\n{'':<10}{'position':>10}{'velocity':>10}   (mean per-dim Probe Skill, best point)")
for b, r in sorted(rows.items(), key=lambda kv: -kv[1]["vel"]):
    print(f"{b:<10}{r['pos']:>+10.4f}{r['vel']:>+10.4f}")
json.dump({b: {k: (v.tolist() if hasattr(v,'tolist') else v) for k,v in r.items()}
           for b, r in rows.items()},
          open('logs/pilots/depth_pilot_L-dw-20m.json','w'), indent=1)
print("\nwrote logs/pilots/depth_pilot_L-dw-20m.json")
