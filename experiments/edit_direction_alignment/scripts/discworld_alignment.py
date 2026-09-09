"""Discworld: the true edit direction from an ORACLE counterfactual history.

For each canonical edit case (frustum basis, L-dw-noiseless-20m), the counterfactual history
shifts the edited object's whole pre-edit trajectory by the teleport vector and re-renders
it with the canonical renderer (noiseless, so obs == clean). Its next frame is, by
construction, the edited world's frame EF, so the residual it produces at the last
position is the model's own representation of "the object IS there": Δ = h_cf − h at every
residual point. Then: how much of Δ lies in the position probe's writable subspaces
(common.py), how linear Δ is in the teleport vector (an oracle linear editor's ceiling), and
— the causal check — what the canonical Edit Index does when h is patched with Δ, with its
row-space part only, and with its complement only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cos, frac_in, orth, subspace_fracs, zspace  # noqa: E402

from pim.environments.discworld import arms as dwa  # noqa: E402
from pim.environments.discworld import bench as dwb  # noqa: E402
from pim.environments.discworld.bench import EF, K_ROLL, N_OBJ  # noqa: E402
from pim.environments.discworld.renderer import render_frame  # noqa: E402
from pim.environments.discworld.sim import fully_in_frustum  # noqa: E402
from pim.metrics.zone_editability import sim_config_from  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402
from pim.probes.base import collect_residuals  # noqa: E402

EXP = REPO / "experiments" / "edit_direction_alignment"
DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/noise_ablation/L-dw-noiseless-20m")
    ap.add_argument("--patch", action="store_true", help="also run the causal patch (slower)")
    a_ = ap.parse_args()
    RUN = a_.run
    run_dir = REPO / RUN
    inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
    root = REPO / "datasets/discworld" / inst
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV)
    NP = n_points(model)
    arr = dwb.bench_arrays(n=192, target="full", basis_name="frustum", data_dir=root / "eval")
    sim = arr["sim"]; cfg = sim_config_from(sim, N_OBJ)
    pos, vel, eobj = arr["pos"], arr["vel"], arr["edit_object"]
    n = len(eobj); ar = np.arange(n)
    dt = float(sim["dt"])
    delta = pos[ar, EF, eobj] - (pos[ar, EF - 1, eobj] + vel[ar, EF - 1, eobj] * dt)   # the teleport
    cf_pos = pos[:, :EF].copy()
    cf_pos[ar, :, eobj] += delta[:, None, :]
    # validity of the counterfactual history: fully in frustum and collision-free at every frame
    min_sep = cfg.collision_margin * 2.0 * cfg.radius
    ok = np.array([fully_in_frustum(cf_pos[i], cfg.radius, cfg)
                   and (np.linalg.norm(cf_pos[i, :, 0] - cf_pos[i, :, 1], axis=-1) >= min_sep).all()
                   for i in range(n)])
    keep = np.where(ok)[0]
    print(f"{n} cases, {len(keep)} with a valid counterfactual history", flush=True)
    refl = np.linspace(sim["refl_min"], sim["refl_max"], N_OBJ).astype(np.float32)
    rad = np.full(N_OBJ, sim["radius"], np.float32)
    blink = arr.get("blink_visible")                       # dw-blink: the schedule travels with the case
    if blink is not None:
        from pim.environments.discworld.blink import paint_markers
    obs_cf = np.zeros((len(keep), EF, arr["obs"].shape[-1]), np.float32)
    for a, i in enumerate(keep):
        v = None if blink is None else blink[i]
        for f in range(EF):
            d_, ids_, inten = render_frame(cf_pos[i, f].astype(np.float32), rad, refl, cfg,
                                           visible=None if v is None else v[f])
            if v is not None:
                paint_markers(ids_, inten, v[f], v[f + 1] if f + 1 < v.shape[0] else None)
            obs_cf[a, f] = inten
    # ⛔ the counterfactual history must match the instance's OBSERVATION NOISE, or the model
    # sees clean frames where it was trained on noisy ones and Δ is a distribution shift
    ns = float(sim.get("obs_noise_std", 0.0))
    if ns > 0:
        obs_cf = np.clip(obs_cf + np.random.default_rng(0).normal(0, ns, obs_cf.shape), 0, 1).astype(np.float32)
    obs = arr["obs"][keep, :EF]
    # sanity: the cf history's next frame IS the edited world's frame EF
    with torch.no_grad():
        pred_cf = model.decode(torch.from_numpy(obs_cf).to(DEV)).cpu().numpy()
    gt_ef = arr["gt_roll"][keep, 0]
    rmse_cf = float(np.sqrt(((pred_cf - gt_ef) ** 2).mean()))
    with torch.no_grad():
        rmse_un = float(np.sqrt(((model.decode(torch.from_numpy(obs).to(DEV)).cpu().numpy() - gt_ef) ** 2).mean()))
    print(f"  VALIDITY: model on cf history vs GT edited frame rmse {rmse_cf:.4f} "
          f"(unedited history {rmse_un:.4f}; ratio {rmse_cf / rmse_un:.2f} — small = the cf history really is the edited world)", flush=True)
    R = collect_residuals(model, obs, batch=64)[:, :, -1]          # (NP, n, d) at the last position
    Rcf = collect_residuals(model, obs_cf, batch=64)[:, :, -1]
    # probes (cached in the run dir) and the residual covariance for Haufe / PCA
    lin = dwa.fit_probes(model, target="full", n_seq=30_000, family="linear", basis_name="frustum",
                         data_dir=root / "probe", cache_dir=run_dir / "probes", log=None)
    import h5py
    with h5py.File(root / "probe/test.h5", "r") as f:
        obs_cov = f["obs_intensity"][30_000:31_500, :39].astype(np.float32)      # held-out sequences
    Rcov = collect_residuals(model, obs_cov, batch=64)                            # (NP, 1500, 39, d)
    # bench for the causal check
    b = dwb.load_bench(model, n=192, target="full", basis_name="frustum", data_dir=root / "eval",
                       select=keep)
    u = dwa.unsteered(model, b)
    dz_teleport = np.stack([delta[keep, 0], delta[keep, 1]], 1)                  # (m, 2) world units
    out = {"run": RUN, "instance": inst, "n_cases": int(len(keep)), "n_total": int(n),
           "unedited_edit_index": u["edit_index"], "cf_rmse": rmse_cf, "unedited_rmse": rmse_un,
           "obs_noise_std": ns, "layers": {}}
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(keep)); tr, te = perm[: len(keep) // 2], perm[len(keep) // 2:]
    for ell in range(NP):
        probe = lin[ell][0]
        h = torch.from_numpy(R[ell]).to(DEV); hcf = torch.from_numpy(Rcf[ell]).to(DEV)
        z, zcf = zspace(probe, h), zspace(probe, hcf)
        dz = zcf - z
        Zc = zspace(probe, torch.from_numpy(Rcov[ell].reshape(-1, Rcov.shape[-1])).to(DEV))
        cov_z = torch.cov(Zc.T)
        W = probe.net.weight.detach()                                            # (8, d) full state
        res = {}
        for name, rows in (("pos", list(range(4))), ("full", list(range(8)))):
            fr = subspace_fracs(dz, W[rows], cov_z)
            res[name] = {k: [float(v.mean()), float(v.std())] for k, v in fr.items()}
        # generic baseline: displacement to ANOTHER case's residual (same layer)
        dgen = z[torch.from_numpy(rng.permutation(len(keep))).to(DEV)] - z
        res["generic_pos_rows"] = [float(frac_in(dgen, orth(W[:4])).mean()), 0.0]
        res["generic_pos_haufe"] = float(subspace_fracs(dgen, W[:4], cov_z)["haufe"].mean())
        # is Δ linear in the teleport? oracle linear editor, half/half
        X = np.concatenate([dz_teleport, np.ones((len(keep), 1))], 1)
        Y = dz.cpu().numpy()
        coef, *_ = np.linalg.lstsq(X[tr], Y[tr], rcond=None)
        pred = X[te] @ coef
        r2 = 1 - ((Y[te] - pred) ** 2).sum() / ((Y[te] - Y[tr].mean(0)) ** 2).sum()
        res["oracle_linear_r2"] = float(r2)
        res["oracle_linear_cos"] = float(cos(torch.from_numpy(pred).to(DEV), dz[torch.from_numpy(te).to(DEV)]).mean())
        # rank structure of Δ across cases
        s = np.linalg.svd(Y - Y.mean(0), compute_uv=False); ev = s ** 2 / (s ** 2).sum()
        res["delta_rank_for_90pct"] = int(np.searchsorted(np.cumsum(ev), 0.9) + 1)
        res["delta_norm_over_h_norm"] = float((dz.norm(dim=1) / z.norm(dim=1)).mean())
        # the causal check: patch the last-position residual and roll out (opt-in: slower)
        cards = {}
        if a_.patch:
            dwa.as_activations(model, ell)
            h0 = model.flat_state(b.state)
            Bp = orth(W[:4])
            d_par = ((dz @ Bp) @ Bp.T) * probe.x_std
            d_all = dz * probe.x_std
            for nm, d in (("full", d_all), ("rows_only", d_par), ("complement", d_all - d_par)):
                roll = model.rollout_with_edit(b.state, ell, h0 + d, K_ROLL).cpu().numpy()
                c = dwa.score(model, b, roll, u)
                cards[nm] = {"edit_index": c["edit_index"], "fidelity_ratio": c["fidelity_ratio"]}
        res["patch"] = cards
        out["layers"][str(ell)] = res
        p = res["pos"]
        msg = (f"pt {ell}: rows {p['rows'][0]:.3f}/gen {res['generic_pos_rows'][0]:.3f}  haufe {p['haufe'][0]:.3f}/gen "
               f"{res['generic_pos_haufe']:.3f}  | rank90 {res['delta_rank_for_90pct']}  |Δ|/|h| {res['delta_norm_over_h_norm']:.2f}"
               f"  oracle-lin R² {r2:+.2f}")
        if cards:
            msg += (f"  | patch full {cards['full']['edit_index']:+.2f} rows {cards['rows_only']['edit_index']:+.2f}"
                    f" compl {cards['complement']['edit_index']:+.2f}")
        print(msg, flush=True)
    (EXP / "scores" / f"discworld_alignment_{inst}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
