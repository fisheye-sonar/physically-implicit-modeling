"""Grid-target editing with HAUFE-corrected write directions (2026-09-09).

Moved here 2026-09-09 from experiments/grid_target_control/scripts/ when the grid target
was canonicalised (`pim.environments.discworld.grid_target`; its probes live in the run's
own probes/ and its bench in `pim.environments.discworld.bench`): this script belongs to the
alignment question, not to the target.

The grid (classification) probes are the only discworld read-out whose row space sits ABOVE
chance against the true edit direction (2.1–2.3× generic, vs 0.6–0.8× for the regression
probes — `grid_probe_alignment.py`). Their Haufe-corrected subspace, by contrast, sits AT
chance (0.9–1.0×). So this is the sharp test of which quantity editing follows: absolute
overlap (Haufe is larger in Othello) or overlap relative to baseline (raw rows win here).

PI-haufe: Δz = α Pᵀ δy, with P the pattern matrix (W Pᵀ = I, so the read-out lands on the same
target as the pseudo-inverse; only the null-space component differs).
ND-haufe: direction = P[(cell B, class)] − P[(cell A, class)].
GS is unchanged (it steers the MLP probes; Haufe is a linear-probe correction).

Result (scores/haufe_edit_grid.json, run 2026-09-09 from the original location, numbers
unchanged by the move): ND +0.373 → +0.048, PI +0.126 → +0.115 — the correction HURTS here.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import haufe_patterns  # noqa: E402

from pim.editors.pinv import swap_class_logits  # noqa: E402
from pim.environments.discworld import arms as dwa, bench as dwb  # noqa: E402
from pim.environments.discworld.bench import K_ROLL  # noqa: E402
from pim.environments.discworld.grid_target import CANONICAL as GRID_T  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402
from pim.probes.base import collect_residuals  # noqa: E402

DEV = "cuda"; EXP = REPO / "experiments/edit_direction_alignment"
RUN = "runs/noise_ablation/L-dw-noiseless-20m"
ALPHA_PI = (0.25, 1.0, 3.0, 8.0, 20.0, 60.0, 100.0, 175.0)
ALPHA_ND = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0)
t0 = time.time()
run_dir = REPO / RUN
inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
root = REPO / "datasets/discworld" / inst
model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV); NP = n_points(model)
lin = {ell: pr for ell, (pr, _) in dwa.fit_probes(
    model, target=GRID_T.name, family="linear", basis_name="frustum", cache_dir=run_dir / "probes",
    log=None, require_cached=True, **dwa.probe_recipe(GRID_T.name, root)).items()}
import h5py
with h5py.File(root / "probe/test.h5", "r") as f:
    obs_cov = f["obs_intensity"][30_000:31_500, :39].astype(np.float32)
Rc = collect_residuals(model, obs_cov, batch=64)

# the canonical grid bench: the first 192 cell-changing cases, with the per-case move
b = dwb.load_bench(model, n=192, target=GRID_T.name, basis_name="frustum", data_dir=root / "eval")
n = b.n; ar = np.arange(n)
A_t, B_t, cls_t = b.cells["A"], b.cells["B"], b.cells["cls"]
zero = torch.zeros_like(cls_t)
u = dwa.unsteered(model, b)
print(f"{n} cases (cell-changing) · unedited {u['edit_index']:+.3f}", flush=True)
arms = []
for ell in range(NP):
    probe = lin[ell]
    C = probe.n_classes
    z_all = (torch.from_numpy(Rc[ell].reshape(-1, Rc.shape[-1])).to(DEV) - probe.x_mean) / probe.x_std
    P = haufe_patterns(probe.net.weight.detach(), torch.cov(z_all.T))
    dwa.as_activations(model, ell)
    h0 = model.flat_state(b.state); z0 = (h0 - probe.x_mean) / probe.x_std
    lg0 = probe.net(z0)
    lg = swap_class_logits(probe(h0), A_t, zero, cls_t)         # empty <-> the object class,
    lg = swap_class_logits(lg, B_t, zero, cls_t)                # at the old and the new cell
    d_y = lg.reshape(n, -1) - lg0
    dz = d_y @ P
    dvec = P[B_t * C + cls_t] - P[A_t * C + cls_t]
    for a in ALPHA_PI:
        roll = model.rollout_with_edit(b.state, ell, h0 + a * dz * probe.x_std, K_ROLL).cpu().numpy()
        c = dwa.score(model, b, roll, u)
        arms.append({"editor": "PI-haufe", "point": ell, "alpha": a,
                     **{k: v for k, v in c.items() if np.isscalar(v)}})
    for a in ALPHA_ND:
        d = a * h0.norm(dim=-1, keepdim=True) * dvec / dvec.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        roll = model.rollout_with_edit(b.state, ell, h0 + d * probe.x_std, K_ROLL).cpu().numpy()
        c = dwa.score(model, b, roll, u)
        arms.append({"editor": "ND-haufe", "point": ell, "alpha": a,
                     **{k: v for k, v in c.items() if np.isscalar(v)}})
    bb = max((r for r in arms if r["point"] == ell), key=lambda r: r["edit_index"])
    print(f"  pt {ell}: best {bb['editor']} {bb['edit_index']:+.3f}/fid {bb['fidelity_ratio']:.2f} (α {bb['alpha']})"
          f"  [{(time.time()-t0)/60:.1f} min]", flush=True)
out = {"run": RUN, "n_cases": n, "bench_selection": b.selection,
       "unedited": {k: v for k, v in u.items() if np.isscalar(v)}, "arms": arms}
for ed in ("PI-haufe", "ND-haufe"):
    sub = [r for r in arms if r["editor"] == ed]
    bb = max(sub, key=lambda r: r["edit_index"])
    g = [r for r in sub if r["fidelity_ratio"] <= 1.1]
    bg = max(g, key=lambda r: r["edit_index"]) if g else None
    out[ed] = {"best": bb, "guarded": bg}
    print(f"{ed}: BEST {bb['edit_index']:+.3f}/fid {bb['fidelity_ratio']:.2f} (pt{bb['point']} α{bb['alpha']})"
          + (f" | guarded {bg['edit_index']:+.3f}/{bg['fidelity_ratio']:.2f} (pt{bg['point']} α{bg['alpha']})" if bg else ""), flush=True)
(EXP / "scores/haufe_edit_grid.json").write_text(json.dumps(out, indent=1, default=float))
print(f"total {(time.time()-t0)/60:.1f} min", flush=True)
