"""Near-teleport pilot, categorical target (2026-09-15): the same near-target edits.h5 as
``near_pilot.py``, scored through the ``appearance-fac`` probes (PI / ND / GS on the factorised
categorical read-out, IM as the state write), paired with the canonical far targets on the same
cases. Everything canonical: bench_arrays on the pilot file, cached probes, protocol grids.

    .pim/bin/python experiments/near_teleport_pilot/scripts/near_pilot_fac.py --run runs/noise_ablation/L-dw-noiseless-20m
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.environments import layout  # noqa: E402
from pim.environments.discworld import arms as dwa  # noqa: E402
from pim.environments.discworld.bench import bench_arrays, load_bench  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402

EXP = REPO / "experiments/near_teleport_pilot"
DEV = "cuda"
TARGET = "appearance-fac"


def best_arms(arms):
    out = {}
    for ed in ("PI", "ND", "GS", "IM", "IM-NN"):
        sub = [a for a in arms if (a["editor"] == ed) or (ed in ("PI", "GS") and a["editor"].startswith(ed))]
        if sub:
            b = max(sub, key=lambda a: a["edit_index"])
            out[ed] = {k: b[k] for k in ("editor", "point", "alpha", "edit_index", "fidelity_ratio", "collateral_rmse") if k in b}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--d-lo", type=float, default=1.0); ap.add_argument("--d-hi", type=float, default=1.6)
    a = ap.parse_args()
    run_dir = REPO / a.run
    inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
    st = json.loads((run_dir / "scores.json").read_text())["settings"]
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV); model.eval()
    pilot_dir = EXP / "data" / f"{inst}_near_{a.d_lo}_{a.d_hi}"
    assert (pilot_dir / "edits.h5").exists(), "run near_pilot.py first (it builds the pilot edits.h5)"
    # the pilot file holds the near-teleported versions of the canonical selection's first n cases;
    # rebuild the mapping pilot row -> canonical case index the same way near_pilot did
    select = np.asarray(json.loads(layout.edits_selection("discworld", inst).read_text())["select"], int)[: a.n]
    build = json.loads((EXP / "scores" / f"near_pilot_{Path(a.run).name}_{a.d_lo}_{a.d_hi}.json").read_text())["build"]
    n_built = int(build["n_built"])
    # near_pilot keeps cases in order and drops those with no valid target; recover which by re-deriving
    # is not stored, so require the common case (at most one dropped) and align by edit_value distance
    import h5py
    with h5py.File(pilot_dir / "edits.h5") as f:
        near_seeds = f["seeds"][:]
    with h5py.File(layout.edits_file("discworld", inst)) as f:
        all_seeds = f["seeds"][:]
    canon_idx = np.array([int(np.where(all_seeds == s_)[0][0]) for s_ in near_seeds])
    assert len(canon_idx) == n_built and set(canon_idx) <= set(select.tolist())

    basis = "frustum"
    # the near bench: first n cell-changing cases in the pilot file (grid_selection inside bench_arrays)
    near_arr = bench_arrays(n=n_built, target=TARGET, basis_name=basis, data_dir=pilot_dir, use_selection=False)
    # bench_arrays applied grid_selection on the pilot file; recover the chosen pilot rows to pair the far bench
    from pim.environments.discworld.bench import grid_selection
    from pim.environments.discworld.grid_target import selection_target
    sel_near, _ = grid_selection(pilot_dir, n_built, selection_target(TARGET))
    paired_canon = canon_idx[sel_near]
    b_near = load_bench(model, n=len(sel_near), target=TARGET, basis_name=basis, data_dir=pilot_dir, use_selection=False)
    b_far = load_bench(model, n=len(sel_near), target=TARGET, basis_name=basis, instance=inst, select=paired_canon)
    print(f"{inst} [{TARGET}]: near bench n={b_near.n}, far bench n={b_far.n} (same canonical cases)", flush=True)

    recipe = dwa.probe_recipe(TARGET, inst)
    probes_dir = run_dir / "probes"
    lin = dwa.fit_probes(model, target=TARGET, family="linear", basis_name=basis, cache_dir=probes_dir, log=None, require_cached=True, **recipe)
    mlp = dwa.fit_probes(model, target=TARGET, family="mlp", basis_name=basis, cache_dir=probes_dir, log=None, require_cached=True, **recipe)
    benches = {"near": b_near, "far": b_far}
    uns = {k: dwa.unsteered(model, b) for k, b in benches.items()}
    arms = {k: [] for k in benches}
    for k, b in benches.items():
        arms[k] += [{**r, "fidelity_ratio": dwa.fidelity_ratio(r, uns[k])} for r in dwa.pinv_arm(model, b, lin, st["dw_grid_alpha_pi"], dims="all")]
        for ell, (probe, _) in lin.items():
            arms[k] += [{**r, "fidelity_ratio": dwa.fidelity_ratio(r, uns[k])} for r in dwa.nanda_arm(model, b, probe, ell, st["dw_grid_alpha_nd"], dims="all")]
        arms[k] += [{**r, "fidelity_ratio": dwa.fidelity_ratio(r, uns[k])} for r in dwa.grad_steer_arm(
            model, b, mlp, range(model.n_layers + 1), st["dw_grid_alpha_gs"], n_steps=int(st["dw_gs_steps"]), beta=float(st["dw_gs_beta"]), dims="all")]
    full_recipe = dwa.probe_recipe("full", inst, n_seq=int(st["dw_probe_seqs"]))
    im_arms, im_stats = dwa.inverse_arms(model, benches, basis_name=basis, unsteered_cards=uns, cache_dir=probes_dir, log=None, **full_recipe)
    for k in benches:
        arms[k] += im_arms[k]
    res = {"run": a.run, "instance": inst, "target": TARGET, "n": int(b_near.n), "benches": {}}
    for k, b in benches.items():
        bb = best_arms(arms[k])
        res["benches"][k] = {"unedited": uns[k]["edit_index"], "collateral_unedited": uns[k]["collateral_rmse"], "best": bb}
        print(f"  [{k}] unedited {uns[k]['edit_index']:+.3f}  " + "  ".join(
            f"{ed} {v['edit_index']:+.3f}/{v['fidelity_ratio']:.2f} (pt{v['point']} a{v['alpha']})" for ed, v in bb.items()), flush=True)
    out = EXP / "scores" / f"near_pilot_fac_{Path(a.run).name}_{a.d_lo}_{a.d_hi}.json"
    out.write_text(json.dumps(res, indent=1)); print(f"  wrote {out.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
