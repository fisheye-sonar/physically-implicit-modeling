"""Alignment of the true edit direction with the FACTORISED categorical probes (2026-09-10).

`appearance-fac` on L-dw-8ray-20m — per object one softmax over run centre (15) and one over
run length (5) — is the categorical read-out that survives factorisation (findings/
probe-target-type.md). Same oracle counterfactual as `grid_probe_alignment.py` (the edited
object's trajectory shifted by its teleport and re-rendered; Δ = h_cf − h at the last context
position), on the 8-ray instance; three probe read-outs on the same model, same cases:

    fac         the rows of the edited object's tiles whose class CHANGES between the real and
                the counterfactual world at the last context frame: (tile, old) and (tile, new)
                — 2 rows per moved factor, 2–4 rows; ND's direction is their contrast summed
    joint cell  the 6 rows of `appearance` ({empty, obj0, obj1} × the old and the new cell)
    regression  the 4 position rows of the canonical `full` probe (frustum basis)

Reported per residual point: the fraction of Δ inside each subspace (raw rows / Haufe
patterns), the same for a GENERIC displacement (an unrelated case's Δ-sized move: the
experiment's usual baseline) and for a RANDOM subspace of the same rank (Gaussian rows — the
chance floor, ≈ rank / d); and for fac and the joint cell the cos² between Δ and the single
ND write direction (raw and Haufe), against a random direction (≈ 1 / d). Contained: reads
cached probes, writes scores/fac_probe_alignment.json, changes nothing canonical.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cos, frac_in, haufe_patterns, orth, zspace  # noqa: E402

from pim.environments import layout  # noqa: E402
from pim.environments.discworld import arms as dwa, bench as dwb  # noqa: E402
from pim.environments.discworld.bench import EF, N_OBJ  # noqa: E402
from pim.environments.discworld.grid_target import N_CLASSES, categorical_target  # noqa: E402
from pim.environments.discworld.renderer import render_frame  # noqa: E402
from pim.environments.discworld.sim import fully_in_frustum  # noqa: E402
from pim.metrics.zone_editability import sim_config_from  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402
from pim.probes.base import collect_residuals  # noqa: E402

DEV = "cuda"
RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/ray_ablation/L-dw-8ray-20m"
FAC = sys.argv[2] if len(sys.argv) > 2 else "appearance-fac"
t0 = time.time()
run_dir = REPO / RUN
inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV); NP = n_points(model)
T_fac = categorical_target(FAC); T_cell = T_fac.cat

# ── the oracle counterfactual (identical construction to grid_probe_alignment.py) ────
arr = dwb.bench_arrays(n=192, target="full", basis_name="frustum", instance=inst)
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
j = eobj[keep]; kk = np.arange(len(keep))
cell_real = T_cell.cell_of(pos[keep, EF - 1][kk, j], sim)
cell_cf = T_cell.cell_of(cf_pos[keep, EF - 1][kk, j], sim)
moved = cell_real != cell_cf
sel = np.where(moved)[0]
# the factorised move at the last context frame: object j's tiles, classes real → cf
F = T_fac.n_factors
lab_real = T_fac.factor_labels(pos[keep, EF - 1], sim).reshape(len(keep), N_OBJ, F)[kk, j]   # (K, F)
lab_cf = T_fac.factor_labels(cf_pos[keep, EF - 1], sim).reshape(len(keep), N_OBJ, F)[kk, j]
tiles = j[:, None] * F + np.arange(F)[None, :]
print(f"{len(keep)} valid counterfactuals of 192; the shift changes the object's cell in {int(moved.sum())} "
      f"(centre changes in {int((lab_real[sel, 0] != lab_cf[sel, 0]).sum())}, length in "
      f"{int((lab_real[sel, 1] != lab_cf[sel, 1]).sum())}, both in "
      f"{int(((lab_real[sel] != lab_cf[sel]).sum(1) == 2).sum())})", flush=True)
R = collect_residuals(model, obs, batch=64)[:, :, -1]
Rcf = collect_residuals(model, obs_cf, batch=64)[:, :, -1]

# ── the three probe sets (cached; nothing is refitted) ───────────────────────────────
def _lin(target, **kw):
    fits = dwa.fit_probes(model, target=target, family="linear", basis_name="frustum",
                          cache_dir=run_dir / "probes", log=None, **kw)
    return {ell: pr for ell, (pr, _) in fits.items()}
lin_fac = _lin(FAC, require_cached=True, **dwa.probe_recipe(FAC, inst))
lin_cell = _lin(T_cell.name, require_cached=True, **dwa.probe_recipe(T_cell.name, inst))
lin_reg = _lin("full", **dwa.probe_recipe("full", inst, n_seq=30_000))
C_fac, C_cell = lin_fac[0].n_classes, lin_cell[0].n_classes
print(f"probes: {FAC} ({T_fac.n_tiles} tiles × {C_fac}), {T_cell.name} ({T_cell.n_cells(sim)} × {C_cell}), full (8)",
      flush=True)

import h5py
with h5py.File(layout.probe_file("discworld", inst, "120k"), "r") as f:
    obs_cov = f["obs_intensity"][30_000:31_500, :39].astype(np.float32)     # held out of the 30k fit
Rc = collect_residuals(model, obs_cov, batch=64)
rng = np.random.default_rng(0)
perm = rng.permutation(len(sel))
g = torch.Generator(device="cpu").manual_seed(0)

def rows_fac(i):
    r = []
    for f in range(F):
        if lab_real[i, f] != lab_cf[i, f]:
            r += [int(tiles[i, f]) * C_fac + int(lab_real[i, f]), int(tiles[i, f]) * C_fac + int(lab_cf[i, f])]
    return r

def nd_dir(P, i, kind):
    """ND's write direction from a weight/pattern matrix P: the target − current contrast."""
    if kind == "fac":
        d = sum(P[int(tiles[i, f]) * C_fac + int(lab_cf[i, f])] - P[int(tiles[i, f]) * C_fac + int(lab_real[i, f])]
                for f in range(F) if lab_real[i, f] != lab_cf[i, f])
    else:
        c = int(j[i]) + 1
        d = P[int(cell_cf[i]) * C_cell + c] - P[int(cell_real[i]) * C_cell + c]
    return d

out = {"run": RUN, "fac_target": FAC, "n_valid_cf": int(len(keep)), "n_cell_changing": int(len(sel)),
       "d_model": int(R.shape[-1]), "layers": {}}
hdr = (f"{'pt':>3} | {'fac rows':>8} {'gen':>6} {'rand':>6} | {'fac Haufe':>9} {'genH':>6} {'randH':>6} "
       f"| {'cell rows':>9} {'gen':>6} | {'cellH':>6} {'genH':>6} | {'reg rows':>8} {'gen':>6} | {'regH':>6} {'genH':>6} "
       f"|| {'cos² ND fac':>11} {'H':>6} {'rand':>6} | {'cos² ND cell':>12} {'H':>6}")
print("\nfraction of Δ inside the subspace (mean over cases); gen = an unrelated case's displacement; "
      "rand = a random subspace / direction of the same rank\n" + hdr, flush=True)
for ell in range(NP):
    row = {}
    for name, probe, kind in (("fac", lin_fac[ell], "fac"), ("cell", lin_cell[ell], "cell"), ("reg", lin_reg[ell], "reg")):
        z = zspace(probe, torch.from_numpy(R[ell]).to(DEV))
        zcf = zspace(probe, torch.from_numpy(Rcf[ell]).to(DEV))
        S = torch.from_numpy(sel).to(DEV)
        dz = (zcf - z)[S]
        dgen = z[torch.from_numpy(sel[perm]).to(DEV)] - z[S]
        cov = torch.cov(zspace(probe, torch.from_numpy(Rc[ell].reshape(-1, Rc.shape[-1])).to(DEV)).T)
        W = probe.net.weight.detach()
        P = haufe_patterns(W, cov)
        if kind == "reg":
            rows_idx = [list(range(4))] * len(sel)
        elif kind == "cell":
            rows_idx = [[int(cell_real[i]) * C_cell + k for k in range(N_CLASSES)]
                        + [int(cell_cf[i]) * C_cell + k for k in range(N_CLASSES)] for i in sel]
        else:
            rows_idx = [rows_fac(i) for i in sel]
        fr, fh, gr, gh, rr, rh, cn, ch, cr = ([] for _ in range(9))
        for m, (i, ri) in enumerate(zip(sel, rows_idx)):
            Br, Bh = orth(W[ri]), orth(P[ri])
            fr.append(float(frac_in(dz[m:m + 1], Br))); fh.append(float(frac_in(dz[m:m + 1], Bh)))
            gr.append(float(frac_in(dgen[m:m + 1], Br))); gh.append(float(frac_in(dgen[m:m + 1], Bh)))
            Wr = torch.randn(len(ri), W.shape[1], generator=g).to(DEV)         # random rows, same rank
            rr.append(float(frac_in(dz[m:m + 1], orth(Wr))))
            rh.append(float(frac_in(dz[m:m + 1], orth(haufe_patterns(Wr, cov)))))
            if kind != "reg":
                cn.append(float(cos(dz[m:m + 1], nd_dir(W, i, kind)[None]) ** 2))
                ch.append(float(cos(dz[m:m + 1], nd_dir(P, i, kind)[None]) ** 2))
                cr.append(float(cos(dz[m:m + 1], torch.randn(1, W.shape[1], generator=g).to(DEV)) ** 2))
        row[name] = {"rows": float(np.mean(fr)), "generic_rows": float(np.mean(gr)), "random_rows": float(np.mean(rr)),
                     "haufe": float(np.mean(fh)), "generic_haufe": float(np.mean(gh)), "random_haufe": float(np.mean(rh)),
                     "n_rows_mean": float(np.mean([len(r) for r in rows_idx]))}
        if kind != "reg":
            row[name].update({"cos2_nd": float(np.mean(cn)), "cos2_nd_haufe": float(np.mean(ch)),
                              "cos2_random_dir": float(np.mean(cr))})
    out["layers"][str(ell)] = row
    f_, c_, r_ = row["fac"], row["cell"], row["reg"]
    print(f"{ell:>3} | {f_['rows']:>8.3f} {f_['generic_rows']:>6.3f} {f_['random_rows']:>6.3f} "
          f"| {f_['haufe']:>9.3f} {f_['generic_haufe']:>6.3f} {f_['random_haufe']:>6.3f} "
          f"| {c_['rows']:>9.3f} {c_['generic_rows']:>6.3f} | {c_['haufe']:>6.3f} {c_['generic_haufe']:>6.3f} "
          f"| {r_['rows']:>8.3f} {r_['generic_rows']:>6.3f} | {r_['haufe']:>6.3f} {r_['generic_haufe']:>6.3f} "
          f"|| {f_['cos2_nd']:>11.4f} {f_['cos2_nd_haufe']:>6.4f} {f_['cos2_random_dir']:>6.4f} "
          f"| {c_['cos2_nd']:>12.4f} {c_['cos2_nd_haufe']:>6.4f}", flush=True)
tag = RUN.split("/")[-1]
(REPO / f"experiments/edit_direction_alignment/scores/fac_probe_alignment_{tag}.json").write_text(
    json.dumps(out, indent=1))
print(f"total {(time.time() - t0) / 60:.1f} min", flush=True)
