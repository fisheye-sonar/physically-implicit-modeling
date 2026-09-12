"""Alignment of the true edit direction with the GRID (classification) probes — the control
for "Othello reads a classification, discworld a regression" — the grid probe target,
`pim.environments.discworld.grid_target` (canonicalised 2026-09-09 from experiments/grid_target_control).

Same model (L-dw-noiseless-20m), same oracle counterfactual Δ, same cases as
`discworld_alignment.py`; only the probe target differs:

    regression   4 rows  — the position read-outs (u, 1/y per object), frustum basis
    grid 3-way   6 rows  — the {empty, obj0, obj1} rows of the TWO cells that change
                           between the real and counterfactual worlds at the last context
                           frame (cell of the edited object before vs after the shift) —
                           the exact analogue of Othello's "rows of the changed tiles"

Cases where the shift does not change the object's cell are dropped: there is no cell edit
to align with. Reports raw / generic / Haufe / Haufe-generic for both targets.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import frac_in, haufe_patterns, orth, zspace  # noqa: E402

from pim.environments.discworld import arms as dwa, bench as dwb  # noqa: E402
from pim.environments.discworld.bench import EF, N_OBJ  # noqa: E402
from pim.environments.discworld.grid_target import CANONICAL as GRID_T, N_CLASSES  # noqa: E402
from pim.environments.discworld.renderer import render_frame  # noqa: E402
from pim.environments.discworld.sim import fully_in_frustum  # noqa: E402
from pim.metrics.zone_editability import sim_config_from  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402
from pim.probes.base import collect_residuals  # noqa: E402

DEV = "cuda"
RUN = "runs/noise_ablation/L-dw-noiseless-20m"
run_dir = REPO / RUN
inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
root = REPO / "datasets/discworld" / inst
model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV); NP = n_points(model)

# ── the counterfactual, identical to discworld_alignment.py ──────────────────
arr = dwb.bench_arrays(n=192, target="full", basis_name="frustum", data_dir=root / "eval")
sim = arr["sim"]; cfg = sim_config_from(sim, N_OBJ)
pos, vel, eobj = arr["pos"], arr["vel"], arr["edit_object"]
n = len(eobj); ar = np.arange(n); dt = float(sim["dt"])
delta = pos[ar, EF, eobj] - (pos[ar, EF - 1, eobj] + vel[ar, EF - 1, eobj] * dt)
cf_pos = pos[:, :EF].copy(); cf_pos[ar, :, eobj] += delta[:, None, :]
ms = cfg.collision_margin * 2.0 * cfg.radius
keep = np.array([i for i in range(n) if fully_in_frustum(cf_pos[i], cfg.radius, cfg)
                 and (np.linalg.norm(cf_pos[i, :, 0] - cf_pos[i, :, 1], axis=-1) >= ms).all()])
refl = np.linspace(sim["refl_min"], sim["refl_max"], N_OBJ).astype(np.float32)
rad = np.full(N_OBJ, sim["radius"], np.float32)
obs_cf = np.stack([np.stack([render_frame(cf_pos[i, f].astype(np.float32), rad, refl, cfg)[2]
                             for f in range(EF)]) for i in keep]).astype(np.float32)
obs = arr["obs"][keep, :EF]
# the cells that change at the LAST CONTEXT FRAME (what the last-position residual represents)
j = eobj[keep]
cell_real = GRID_T.cell_of(pos[keep, EF - 1, :][np.arange(len(keep)), j], sim)
cell_cf = GRID_T.cell_of(cf_pos[keep, EF - 1, :][np.arange(len(keep)), j], sim)
moved = cell_real != cell_cf
print(f"{len(keep)} valid counterfactuals; the shift changes the object's CELL in {int(moved.sum())} of them "
      f"({moved.mean():.0%}) — the rest have no cell edit to align with", flush=True)
sel = np.where(moved)[0]
R = collect_residuals(model, obs, batch=64)[:, :, -1]
Rcf = collect_residuals(model, obs_cf, batch=64)[:, :, -1]

# ── probes: the canonical regression set and the grid classification set ─────
lin_reg = dwa.fit_probes(model, target="full", n_seq=30_000, family="linear", basis_name="frustum",
                         data_dir=root / "probe", cache_dir=run_dir / "probes", log=None)
# the grid probes live in the run's own probes/ (re-keyed 2026-09-09) and are never refitted here
_grid = dwa.fit_probes(model, target=GRID_T.name, family="linear", basis_name="frustum",
                       cache_dir=run_dir / "probes", log=None, require_cached=True,
                       **dwa.probe_recipe(GRID_T.name, root))
grid_lin = {ell: pr for ell, (pr, _) in _grid.items()}
print(f"grid probes loaded for points {sorted(grid_lin)} (skill "
      f"{min(st['skill'] for _, st in _grid.values()):.3f}–"
      f"{max(st['skill'] for _, st in _grid.values()):.3f})", flush=True)

import h5py
with h5py.File(root / "probe/test.h5", "r") as f:
    obs_cov = f["obs_intensity"][30_000:31_500, :39].astype(np.float32)
Rc = collect_residuals(model, obs_cov, batch=64)
rng = np.random.default_rng(0)
perm = rng.permutation(len(sel))
out = {"run": RUN, "n_valid_cf": int(len(keep)), "n_cell_changing": int(moved.sum()), "layers": {}}
print(f"\n{'pt':>3} | {'grid rows':>9} {'gen':>7} {'×':>5} | {'grid Haufe':>10} {'genH':>7} {'×':>5} "
      f"| {'reg rows':>8} {'gen':>7} {'×':>5} | {'reg Haufe':>9} {'genH':>7} {'×':>5}", flush=True)
for ell in range(NP):
    row = {}
    for name, probe, kind in (("grid", grid_lin.get(ell), "grid"), ("reg", lin_reg[ell][0], "reg")):
        if probe is None: continue
        z = zspace(probe, torch.from_numpy(R[ell]).to(DEV))
        zcf = zspace(probe, torch.from_numpy(Rcf[ell]).to(DEV))
        dz = (zcf - z)[torch.from_numpy(sel).to(DEV)]
        dgen = (z[torch.from_numpy(sel[perm]).to(DEV)] - z[torch.from_numpy(sel).to(DEV)])
        cov = torch.cov(zspace(probe, torch.from_numpy(Rc[ell].reshape(-1, Rc.shape[-1])).to(DEV)).T)
        W = probe.net.weight.detach()
        P = haufe_patterns(W, cov)
        if kind == "reg":
            rows_idx = [list(range(4))] * len(sel)                     # the position read-outs
        else:
            rows_idx = [[int(cell_real[i]) * N_CLASSES + k for k in range(N_CLASSES)]
                        + [int(cell_cf[i]) * N_CLASSES + k for k in range(N_CLASSES)] for i in sel]
        fr, fh, gr, gh = [], [], [], []
        for m, ri in enumerate(rows_idx):
            Br, Bh = orth(W[ri]), orth(P[ri])
            fr.append(float(frac_in(dz[m:m + 1], Br))); fh.append(float(frac_in(dz[m:m + 1], Bh)))
            gr.append(float(frac_in(dgen[m:m + 1], Br))); gh.append(float(frac_in(dgen[m:m + 1], Bh)))
        row[name] = {"rows": float(np.mean(fr)), "generic_rows": float(np.mean(gr)),
                     "haufe": float(np.mean(fh)), "generic_haufe": float(np.mean(gh)),
                     "n_rows": len(rows_idx[0])}
    out["layers"][str(ell)] = row
    g, r = row.get("grid"), row["reg"]
    if g:
        print(f"{ell:>3} | {g['rows']:>9.3f} {g['generic_rows']:>7.3f} {g['rows']/max(g['generic_rows'],1e-9):>4.1f}× "
              f"| {g['haufe']:>10.3f} {g['generic_haufe']:>7.3f} {g['haufe']/max(g['generic_haufe'],1e-9):>4.1f}× "
              f"| {r['rows']:>8.3f} {r['generic_rows']:>7.3f} {r['rows']/max(r['generic_rows'],1e-9):>4.1f}× "
              f"| {r['haufe']:>9.3f} {r['generic_haufe']:>7.3f} {r['haufe']/max(r['generic_haufe'],1e-9):>4.1f}×", flush=True)
(REPO / "experiments/edit_direction_alignment/scores/grid_probe_alignment.json").write_text(json.dumps(out, indent=1))
