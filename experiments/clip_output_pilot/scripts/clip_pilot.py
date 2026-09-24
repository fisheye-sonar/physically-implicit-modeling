"""Does clipping the predicted frame to [0, 1] change discworld editability? (2026-09-17)

The observation is a reflectivity scan in [0, 1] by construction (0 on a miss, 0.4 / 0.8 on a
disc), but the regression head is unconstrained, so a destructive write can be scored on values
the environment can never produce. This pilot re-scores the CANONICAL best arm of each editor
with the rollout clipped to [0, 1], at the residual point and step size already selected in
``scores.json`` — no probe refitting, no sweep, no canonical change.

Both references are clean renders in [0, 1], so clipping can only move the prediction toward the
feasible set. The unsteered rollout is clipped too, so the Fidelity Ratio's denominator is the
same post-processing (the raw-denominator variant is reported beside it).

    .pim/bin/python experiments/clip_output_pilot/scripts/clip_pilot.py \
        --run runs/noise_ablation/L-dw-noiseless-20m [--basis cartesian]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.editors.inverse import inverse_overwrite, retrieval_overwrite  # noqa: E402
from pim.environments.discworld import arms as dwa  # noqa: E402
from pim.environments.discworld.bench import K_ROLL, full_state_pair, load_bench  # noqa: E402
from pim.metrics.zone_editability import edit_scorecard  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402

DEV = "cuda"
EXP = REPO / "experiments/clip_output_pilot"
KEEP = ("edit_index", "target_rmse", "ghost_rmse", "collateral_rmse", "edit_frame_rmse")


def card(b, roll, uns_card=None):
    c = edit_scorecard(roll, b.zones, b.gt_roll)
    if uns_card is not None:
        c["fidelity_ratio"] = dwa.fidelity_ratio(c, uns_card)
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--basis", default="cartesian")
    ap.add_argument("--target", default="full")
    ap.add_argument("--block", default=None, help="scores.json bases key (default: --basis); the\n                    categorical targets are fitted in the frustum basis but reported under their own block")
    ap.add_argument("--no-im", action="store_true", help="skip IM / IM-NN (target-independent: they write the full state)")
    a = ap.parse_args()
    run_dir = REPO / a.run
    inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
    S = json.loads((run_dir / "scores.json").read_text())
    st = S["settings"]
    block = a.block or a.basis
    blk = S["bases"][block]
    best = blk["best"]
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV); model.eval()
    probes_dir = run_dir / "probes"

    b = load_bench(model, n=int(st["dw_bench_n"]), target=a.target, basis_name=a.basis, instance=inst)
    recipe = dwa.probe_recipe(a.target, inst, n_seq=int(st["dw_probe_seqs"]))
    lin = dwa.fit_probes(model, target=a.target, family="linear", basis_name=a.basis,
                         cache_dir=probes_dir, log=None, require_cached=True, **recipe)
    mlp = dwa.fit_probes(model, target=a.target, family="mlp", basis_name=a.basis,
                         cache_dir=probes_dir, log=None, require_cached=True, **recipe)

    rolls = {}
    uns = dwa.unsteered_rollout(model, b)
    rolls["unedited"] = uns
    for ed in ("PI", "ND", "GS"):
        if ed not in best or best[ed] is None:
            continue
        pt, al = int(best[ed]["point"]), float(best[ed]["alpha"])
        if ed == "PI":
            rolls[ed] = dwa.pinv_rollout(model, b, lin[pt][0], pt, al, dims="all")
        elif ed == "ND":
            rolls[ed] = dwa.nanda_rollout(model, b, lin[pt][0], pt, al, dims="all")
        else:
            rolls[ed] = dwa.grad_steer_rollout(model, b, mlp, pt, al,
                                               n_steps=int(st["dw_gs_steps"]),
                                               beta=float(st["dw_gs_beta"]), dims="all")
    # IM / IM-NN at their canonical points, from the cached inverse maps
    want = {} if a.no_im else {ed: int(best[ed]["point"]) for ed in ("IM", "IM-NN") if ed in best and best[ed]}
    if want:
        _s_pre, s_post = full_state_pair(b.pos, b.vel, b.edit_object, b.sim, a.basis)
        import torch
        s_post_t = torch.from_numpy(s_post).to(DEV)
        im_recipe = dwa.probe_recipe("full", inst, n_seq=int(st["dw_probe_seqs"]))
        for ell, g, bank, _stats in dwa.iter_inverse_maps(
                model, basis_name=a.basis, cache_dir=probes_dir, log=None,
                points=sorted(set(want.values())), **im_recipe):
            dwa.as_activations(model, ell)
            for ed, h_new in (("IM", inverse_overwrite(g, s_post_t)),
                              ("IM-NN", retrieval_overwrite(bank, s_post_t))):
                if want.get(ed) == ell:
                    rolls[ed] = model.rollout_with_edit(b.state, ell, h_new, K_ROLL).cpu().numpy()

    frac_out = {k: float(((r[:, 0] < 0) | (r[:, 0] > 1)).mean()) for k, r in rolls.items()}
    lo = {k: float(r[:, 0].min()) for k, r in rolls.items()}
    hi = {k: float(r[:, 0].max()) for k, r in rolls.items()}
    uns_raw = card(b, uns)
    uns_clip = card(b, np.clip(uns, 0.0, 1.0))

    res = {"run": a.run, "instance": inst, "basis": a.basis, "block": block, "target": a.target,
            "n": int(b.n), "arms": []}
    print(f"{inst} [block {block} | basis {a.basis} | target {a.target}] n={b.n}", flush=True)
    for ed, roll in rolls.items():
        raw = card(b, roll, None if ed == "unedited" else uns_raw)
        clp = card(b, np.clip(roll, 0.0, 1.0), None if ed == "unedited" else uns_clip)
        clp_rawden = card(b, np.clip(roll, 0.0, 1.0), None if ed == "unedited" else uns_raw)
        rec = {"editor": ed,
               "point": None if ed == "unedited" else int(best[ed]["point"]),
               "alpha": None if ed == "unedited" else float(best[ed]["alpha"]),
               "frac_rays_out_of_range": frac_out[ed], "min": lo[ed], "max": hi[ed],
               "raw": {k: raw[k] for k in KEEP if k in raw},
               "clipped": {k: clp[k] for k in KEEP if k in clp}}
        if ed != "unedited":
            rec["raw"]["fidelity_ratio"] = raw["fidelity_ratio"]
            rec["clipped"]["fidelity_ratio"] = clp["fidelity_ratio"]
            rec["clipped_raw_denominator_fidelity"] = clp_rawden["fidelity_ratio"]
        res["arms"].append(rec)
        f_raw = raw.get("fidelity_ratio", 1.0); f_clp = clp.get("fidelity_ratio", 1.0)
        print(f"  {ed:9s} pt{str(rec['point']):>4s} a{str(rec['alpha'] or ''):>6s} | "
              f"out-of-range {frac_out[ed]:6.2%} [{lo[ed]:+.2f},{hi[ed]:+.2f}] | "
              f"index {raw['edit_index']:+.3f} -> {clp['edit_index']:+.3f} | "
              f"fid {f_raw:5.2f} -> {f_clp:5.2f}", flush=True)
    out = EXP / "scores" / f"clip_{Path(a.run).name}_{block}_{a.target}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"  wrote {out.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
