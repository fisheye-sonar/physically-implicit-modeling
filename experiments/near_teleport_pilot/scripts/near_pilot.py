"""Near-teleport pilot (2026-09-15): does the inverse-map editor fail when the teleport target
is INSIDE the generator's exclusion zone?

The generator rejects trajectories in which two discs come within collision_margin × 2r
(1.6 units at r = 0.5), so no training frame has the discs closer than that. This pilot
rebuilds the canonical edit cases of one run with the edited disc teleported to a point
1.0 < d < 1.6 from the other disc (no overlap, but never seen), re-renders the post-edit
frames, and scores PI / GS / IM on those cases and on the SAME cases with their canonical
(far) teleports. Everything downstream is canonical: bench_arrays on a pilot edits.h5, the
cached probes and inverse maps, the canonical arms and scorecards.

    .pim/bin/python experiments/near_teleport_pilot/scripts/near_pilot.py \
        --run runs/noise_ablation/L-dw-noiseless-20m [--n 192] [--d-lo 1.0 --d-hi 1.6]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.environments import layout  # noqa: E402
from pim.environments.discworld import arms as dwa  # noqa: E402
from pim.environments.discworld.bench import EF, K_ROLL, load_bench  # noqa: E402
from pim.environments.discworld.renderer import render_frame  # noqa: E402
from pim.environments.discworld.sim import fully_in_frustum  # noqa: E402
from pim.metrics.zone_editability import sim_config_from  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402

EXP = REPO / "experiments/near_teleport_pilot"
DEV = "cuda"


def build_near_h5(inst: str, select: np.ndarray, out: Path, d_lo: float, d_hi: float,
                  seed: int = 0, tries: int = 500) -> tuple[np.ndarray, dict]:
    """Copy the selected cases into a pilot edits.h5 with the edited disc teleported to
    d ∈ (d_lo, d_hi) from the other disc at EF; post-edit frames re-rendered. Returns the
    indices (into ``select``) that found a valid target, and stats."""
    src = layout.edits_file("discworld", inst)
    rng = np.random.default_rng(seed)
    with h5py.File(src, "r") as f:
        cfg_json = f.attrs["config_json"]
        sim = json.loads(cfg_json)["dataset"]["sim"]
        keep, data = [], {k: [] for k in f.keys()}
        cfg = sim_config_from(sim, 2)
        r = float(sim["radius"])
        T = f["positions"].shape[1]
        dists = []
        for j, i in enumerate(select):
            i = int(i)
            pos = f["positions"][i].astype(np.float64)          # (T, 2, 2), post-edit trajectory
            vel = f["velocities"][i].astype(np.float64)
            k = int(f["edit_object"][i]); o = 1 - k
            other = pos[EF, o]
            found = None
            for _ in range(tries):
                d = rng.uniform(d_lo, d_hi); th = rng.uniform(0, 2 * np.pi)
                t = other + d * np.array([np.cos(th), np.sin(th)])
                # edited disc from EF on: t + v·s; must stay in the frustum and never overlap
                traj = pos.copy()
                for s in range(EF, T):
                    traj[s, k] = t + vel[EF - 1, k] * (s - EF) * float(sim["dt"])
                ok = fully_in_frustum(traj[EF:, k][:, None, :], r, cfg)
                ok = ok and bool((np.linalg.norm(traj[EF:, k] - traj[EF:, o], axis=1) > 2 * r + 1e-3).all())
                if ok:
                    found = traj; break
            if found is None:
                continue
            keep.append(j); dists.append(float(np.linalg.norm(found[EF, k] - other)))
            rad = f["radii"][i]; refl = f["reflectivities"][i]
            inten = f["obs_intensity"][i].copy(); ids = f["obs_id"][i].copy(); dep = f["obs_depth"][i].copy()
            for s in range(EF, T):
                dd, ii, it = render_frame(found[s].astype(np.float32), rad, refl, cfg)
                dep[s], ids[s], inten[s] = dd, ii, it
            for key in f.keys():
                if key == "positions": data[key].append(found.astype(np.float32))
                elif key == "obs_intensity": data[key].append(inten)
                elif key == "obs_id": data[key].append(ids)
                elif key == "obs_depth": data[key].append(dep)
                elif key == "edit_value": data[key].append(found[EF, k].astype(np.float32))
                else: data[key].append(f[key][i])
        out.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out, "w") as g:
            for key, rows in data.items():
                g.create_dataset(key, data=np.stack(rows))
            g.attrs["config_json"] = cfg_json
    return np.array(keep), {"n_requested": int(len(select)), "n_built": len(keep),
                            "target_dist_mean": float(np.mean(dists)), "d_lo": d_lo, "d_hi": d_hi}


def best_arms(arms: list[dict]) -> dict:
    out = {}
    for ed in ("PI", "GS", "IM", "IM-NN"):
        sub = [a for a in arms if a["editor"].startswith(ed) and not (ed == "IM" and a["editor"] == "IM-NN")]
        if sub:
            b = max(sub, key=lambda a: a["edit_index"])
            out[ed] = {k: b[k] for k in ("editor", "point", "alpha", "edit_index", "fidelity_ratio",
                                         "collateral_rmse", "target_rmse", "ghost_rmse") if k in b}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--d-lo", type=float, default=1.0); ap.add_argument("--d-hi", type=float, default=1.6)
    a = ap.parse_args()
    run_dir = REPO / a.run
    inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
    S = json.loads((run_dir / "scores.json").read_text()); st = S["settings"]
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV)
    model.eval()
    select = np.asarray(json.loads(layout.edits_selection("discworld", inst).read_text())["select"], int)[: a.n]

    pilot_dir = EXP / "data" / f"{inst}_near_{a.d_lo}_{a.d_hi}"
    keep, bstats = build_near_h5(inst, select, pilot_dir / "edits.h5", a.d_lo, a.d_hi)
    print(f"{inst}: built {bstats['n_built']}/{bstats['n_requested']} near cases, mean target distance {bstats['target_dist_mean']:.2f}", flush=True)

    basis = "frustum"
    recipe = dwa.probe_recipe("full", inst, n_seq=int(st["dw_probe_seqs"]))
    probes_dir = run_dir / "probes"
    lin = dwa.fit_probes(model, target="full", family="linear", basis_name=basis, cache_dir=probes_dir, log=None, **recipe)
    mlp = dwa.fit_probes(model, target="full", family="mlp", basis_name=basis, cache_dir=probes_dir, log=None, **recipe)

    b_near = load_bench(model, n=len(keep), target="full", basis_name=basis, data_dir=pilot_dir, use_selection=False)
    b_far = load_bench(model, n=len(keep), target="full", basis_name=basis, instance=inst, select=select[keep])
    benches = {"near": b_near, "far": b_far}
    uns = {k: dwa.unsteered(model, b) for k, b in benches.items()}
    arms = {k: [] for k in benches}
    for k, b in benches.items():
        arms[k] += [{**r, "fidelity_ratio": dwa.fidelity_ratio(r, uns[k])} for r in dwa.pinv_arm(model, b, lin, st["dw_alpha_pi"], dims="all")]
        arms[k] += [{**r, "fidelity_ratio": dwa.fidelity_ratio(r, uns[k])} for r in dwa.grad_steer_arm(
            model, b, mlp, range(model.n_layers + 1), st["dw_alpha_gs"], n_steps=int(st["dw_gs_steps"]), beta=float(st["dw_gs_beta"]), dims="all")]
    im_arms, im_stats = dwa.inverse_arms(model, benches, basis_name=basis, unsteered_cards=uns, cache_dir=probes_dir, log=None, **recipe)
    for k in benches:
        arms[k] += im_arms[k]

    res = {"run": a.run, "instance": inst, "build": bstats, "g_r2": im_stats["g_r2"], "benches": {}}
    for k, b in benches.items():
        bb = best_arms(arms[k])
        res["benches"][k] = {"n": int(b.n), "unedited": uns[k]["edit_index"], "collateral_unedited": uns[k]["collateral_rmse"], "best": bb,
                             "im_by_point": {r["point"]: round(r["edit_index"], 3) for r in arms[k] if r["editor"] == "IM"}}
        print(f"  [{k}] n={b.n} unedited {uns[k]['edit_index']:+.3f}  " + "  ".join(
            f"{ed} {v['edit_index']:+.3f}/{v['fidelity_ratio']:.2f} (pt{v['point']} a{v['alpha']})" for ed, v in bb.items()), flush=True)
        print(f"       IM by point: {res['benches'][k]['im_by_point']}", flush=True)
    out = EXP / "scores" / f"near_pilot_{Path(a.run).name}_{a.d_lo}_{a.d_hi}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"  wrote {out.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
