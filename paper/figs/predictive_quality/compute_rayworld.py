"""Predictive-quality arrays for the Blink and 5-ray panels of rayworld.pdf (2026-09-21, round 5). GPU.

For each (run, instance) the SAME three arrays the Standard panel takes from
``.scratch/history_rewrite_arrays.npz``, obtained exactly as ``paper/figs/history_rewrite/make_figure.py``
obtains them for dw-noiseless: the canonical bench (``dwb.load_bench``: the first ``N_BENCH`` selected cases,
full-state target, Cartesian basis), the no-edit free-run (``dwa.unsteered_rollout``), the unedited world's clean
render rolled forward (``b.zones.gt_unedited_traj``, the scorer's ghost trajectory; on dw-blink it is rendered
under the same blackout schedule and markers the observations carry) and the observed history (``b.obs[:, :EF]``).
One model at a time on the shared GPU, freed after. Cached to ``.scratch/predictive_quality_<instance>.npz``;
no metric, no other logic.

    .pim/bin/python paper/figs/predictive_quality/compute_rayworld.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.environments.discworld import arms as dwa, bench as dwb  # noqa: E402
from pim.environments.discworld.bench import EF  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402

RUNS = [("blink_ablation/L-dw-blink-20m", "dw-blink"), ("ray_ablation/L-dw-5ray-20m", "dw-5ray")]
N_BENCH = 32          # the Standard panel's pool: history_rewrite_arrays.npz holds the first 32 selected cases

if __name__ == "__main__":
    for run, inst in RUNS:
        run_dir = REPO / "runs" / run
        assert json.loads((run_dir / "config.json").read_text())["data"]["instance"] == inst, (run, inst)
        model, _ = load_checkpoint(run_dir / "best_model.pt", device=dwb.DEV)
        b = dwb.load_bench(model, n=N_BENCH, target="full", basis_name="cartesian", instance=inst)
        with torch.no_grad():
            roll = dwa.unsteered_rollout(model, b)
        out = REPO / ".scratch" / f"predictive_quality_{inst}.npz"
        np.savez_compressed(out, obs_hist=b.obs[:, :EF], gt_unedited_roll=b.zones.gt_unedited_traj,
                            roll_unsteered=roll, meta_json=json.dumps({"run": run, "instance": inst, "n": int(b.n)}))
        print(f"{inst}: obs {tuple(b.obs[:, :EF].shape)} gt {tuple(b.zones.gt_unedited_traj.shape)} "
              f"roll {tuple(roll.shape)} -> {out.relative_to(REPO)}")
        del model, b
        torch.cuda.empty_cache()
