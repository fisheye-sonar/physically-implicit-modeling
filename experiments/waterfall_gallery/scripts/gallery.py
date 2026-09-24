"""A few edit waterfalls for review (2026-09-16, Sevan): the scored best arm of one editor on one
run, drawn through the canonical helper. Cells: GS on appearance-fac (dw-8ray, dw-5ray) and IM on
the cartesian regression block (dw-noiseless, dw-blink, dw-16ray, dw-5ray). Arms come from each
run's scores.json, rollouts from pim.environments.discworld.arms, drawing from
pim.figures.waterfall_grid (research/specs/WATERFALL_SPEC.md). Outputs in ../outputs/."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from pim.models import load_checkpoint
from pim.environments.discworld import arms as dwa, bench as dwb
from pim.environments.discworld.bench import full_state_pair
from pim.editors.inverse import inverse_overwrite
from pim.figures import waterfall_grid

OUT = REPO / "experiments/waterfall_gallery/outputs"; OUT.mkdir(exist_ok=True)
N_ROWS, N_CTX, N_BENCH = 4, 6, 32
CELLS = [("ray_ablation/L-dw-8ray-20m", "appearance-fac", "GS"),
         ("ray_ablation/L-dw-5ray-20m", "appearance-fac", "GS"),
         ("noise_ablation/L-dw-noiseless-20m", "cartesian", "IM"),
         ("blink_ablation/L-dw-blink-20m", "cartesian", "IM"),
         ("ray_ablation/L-dw-16ray-20m", "cartesian", "IM"),
         ("ray_ablation/L-dw-5ray-20m", "cartesian", "IM")]

def _cx(mask):
    i = np.where(mask)[0]; return i.mean() if i.size else np.nan

for run, block, ed in CELLS:
    run_dir = REPO / "runs" / run
    inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
    best = json.loads((run_dir / "scores.json").read_text())["bases"][block]["best"][ed]
    pt, alpha, dims = int(best["point"]), float(best["alpha"]), best.get("dims", "all")
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=dwb.DEV)
    cat = block == "appearance-fac"
    target, basis = (block, "frustum") if cat else ("full", block)
    b = dwb.load_bench(model, n=N_BENCH, target=target, basis_name=basis, instance=inst)
    recipe = dwa.probe_recipe(target, inst, n_seq=30_000)
    with torch.no_grad():
        uns = dwa.unsteered_rollout(model, b)
        if ed == "GS":
            mlp = dwa.fit_probes(model, target=target, family="mlp", basis_name=basis,
                                 cache_dir=run_dir / "probes", log=None, require_cached=True, **recipe)
            roll = dwa.grad_steer_rollout(model, b, mlp, pt, alpha, dims=dims)
        else:
            _, s_post = full_state_pair(b.pos, b.vel, b.edit_object, b.sim, basis)
            s_post = torch.from_numpy(s_post).to(dwb.DEV)
            g = next(gg for ell, gg, _, _ in dwa.iter_inverse_maps(
                model, basis_name=basis, points=[pt], cache_dir=run_dir / "probes", log=None, **recipe))
            dwa.as_activations(model, pt)
            roll = model.rollout_with_edit(b.state, pt, inverse_overwrite(g, s_post), dwb.K_ROLL).cpu().numpy()
    cu, ce = dwa.score(model, b, uns), dwa.score(model, b, roll)
    fid = dwa.fidelity_ratio(ce, cu)
    lab = f"{ed} (pt{pt} α{alpha:g}, fid {fid:.2f})"
    fig = waterfall_grid(
        columns={"unsteered": uns[:N_ROWS], lab: roll[:N_ROWS]},
        context=b.obs[:N_ROWS, dwb.EF - N_CTX: dwb.EF], gt=b.gt_roll[:N_ROWS],
        title=f"{run} — {ed} at its scored best arm ({block}; EF={dwb.EF}, {dwb.K_ROLL}-step rollout, {inst})",
        sample_idx=range(N_ROWS),
        target_x=np.array([_cx(b.zones.target[i]) for i in range(N_ROWS)]),
        ghost_x=np.array([_cx(b.zones.ghost[i]) for i in range(N_ROWS)]),
        metrics={"unsteered": cu["edit_index"], lab: ce["edit_index"]})
    name = f"{run_dir.name}_{block}_{ed}.png"
    fig.savefig(OUT / name, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"{name}: EI {ce['edit_index']:+.3f} (scored {best['edit_index']:+.3f}) fid {fid:.2f} "
          f"(scored {best['fidelity_ratio']:.2f}) on the first {N_BENCH} cases", flush=True)
    del model; torch.cuda.empty_cache()
