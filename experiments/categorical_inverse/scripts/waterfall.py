"""Waterfall for the categorical inverse map (2026-09-20): unsteered | the OLD write (the basis's continuous
full-state map, which is what the categorical block's "IM" was) | the NEW write (g of the block's own one-hot
labels + Cartesian velocity), on the first cases of the run's categorical bench, through the canonical
helper (pim.figures.waterfall_grid, research/specs/WATERFALL_SPEC.md).

    PYTHONPATH=$PWD .pim/bin/python experiments/categorical_inverse/scripts/waterfall.py ray_ablation/L-dw-8ray-20m

Needs the preview to have run for that parent (the categorical maps are read from this experiment's probes/
cache; the point is the preview's reported arm). Reads the run; writes only ../outputs/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.editors.inverse import inverse_overwrite  # noqa: E402
from pim.environments.discworld import arms as dwa, bench as dwb  # noqa: E402
from pim.environments.discworld.bench import full_state_pair  # noqa: E402
from pim.figures import waterfall_grid  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402
from pim.probes.inverse import encode_categorical_state  # noqa: E402

EXP = REPO / "experiments" / "categorical_inverse"
OUT = EXP / "outputs"
OUT.mkdir(exist_ok=True)
N_ROWS, N_CTX, N_BENCH, TARGET = 4, 6, 32, "appearance-fac"


def _cx(mask):
    i = np.where(mask)[0]
    return i.mean() if i.size else np.nan


for run in sys.argv[1:]:
    run_dir = REPO / "runs" / run
    prev = json.loads((EXP / "scores" / f"preview_{run_dir.name}_{TARGET}.json").read_text())
    scores = json.loads((run_dir / "scores.json").read_text())
    inst, basis = prev["instance"], prev["basis"]
    pt_new = int(prev["new"]["reported (best inside the guard)"]["point"])
    old = prev["old_continuous_map_on_this_bench"]["reported (best inside the guard)"]
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=dwb.DEV)
    model.eval()
    b = dwb.load_bench(model, n=N_BENCH, target=TARGET, basis_name=basis, instance=inst)
    cols, metrics = {}, {}
    with torch.no_grad():
        uns = dwa.unsteered_rollout(model, b)
        cu = dwa.score(model, b, uns)
        cols["unsteered"], metrics["unsteered"] = uns[:N_ROWS], cu["edit_index"]
        if old:                                   # the continuous map of the block's basis (the run's own cache: a HIT)
            pt_old = int(old["point"])
            rec = dwa.probe_recipe("full", inst, n_seq=scores["settings"]["dw_probe_seqs"])
            s_post = torch.from_numpy(full_state_pair(b.pos, b.vel, b.edit_object, b.sim, basis)[1]).to(dwb.DEV)
            g = next(gg for _, gg, _, _ in dwa.iter_inverse_maps(model, basis_name=basis, points=[pt_old],
                                                                 cache_dir=run_dir / "probes", log=None, **rec))
            dwa.as_activations(model, pt_old)
            roll = model.rollout_with_edit(b.state, pt_old, inverse_overwrite(g, s_post), dwb.K_ROLL).cpu().numpy()
            c = dwa.score(model, b, roll, cu)
            lab = f"OLD continuous · pt{pt_old} · fid {c['fidelity_ratio']:.2f}"          # short: long titles collide (STYLE legibility)
            cols[lab], metrics[lab] = roll[:N_ROWS], c["edit_index"]
        rec = dict(prev["recipe"])
        velc = torch.from_numpy(full_state_pair(b.pos, b.vel, b.edit_object, b.sim, "cartesian")[1]).to(dwb.DEV)[:, -4:]
        _, g, _, gst = next(iter(dwa.iter_inverse_maps(model, basis_name=basis, target=TARGET, points=[pt_new],
                                                       cache_dir=EXP / "probes", log=None, **rec)))
        dwa.as_activations(model, pt_new)
        h_new = inverse_overwrite(g, encode_categorical_state(b.tgt, velc, gst["n_classes"]))
        roll = model.rollout_with_edit(b.state, pt_new, h_new, dwb.K_ROLL).cpu().numpy()
        c = dwa.score(model, b, roll, cu)
        lab = f"NEW categorical · pt{pt_new} · fid {c['fidelity_ratio']:.2f}"
        cols[lab], metrics[lab] = roll[:N_ROWS], c["edit_index"]
    fig = waterfall_grid(
        columns=cols, context=b.obs[:N_ROWS, dwb.EF - N_CTX: dwb.EF], gt=b.gt_roll[:N_ROWS],
        title=f"{run} — inverse-map write on the {TARGET} bench: the old continuous-state map vs the categorical-state map "
              f"(EF={dwb.EF}, {dwb.K_ROLL}-step rollout, {inst}; first {N_BENCH} cases scored)",
        sample_idx=range(N_ROWS),
        target_x=np.array([_cx(b.zones.target[i]) for i in range(N_ROWS)]),
        ghost_x=np.array([_cx(b.zones.ghost[i]) for i in range(N_ROWS)]),
        metrics=metrics)
    name = f"waterfall_{run_dir.name}_{TARGET}_categorical_IM.png"
    fig.savefig(OUT / name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"{name}: " + " · ".join(f"{k.split(' · ')[0]} EI {v:+.3f}" for k, v in metrics.items()), flush=True)
    del model
    torch.cuda.empty_cache()
