"""Render the probe-target resolution sweep from the runs' canonical score blocks.

Numbers come from ``runs/<run>/scores.json["bases"]`` (written by master_eval); cell counts
from the target itself on the instance's sim config. Re-run any time a block lands.
"""
import json
import sys
from pathlib import Path

import h5py

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.environments.discworld.grid_target import categorical_target  # noqa: E402
from pim.figures import sweep_figure  # noqa: E402

RUNS = [("ray_ablation/L-dw-8ray-20m", "dw-8ray", "Transformer-L, frames (L-dw-8ray-20m)",
         "Edit Index (ray-zone)"),
        ("interface_ablation/L-dw-8ray-tok-20m", "dw-8ray", "Transformer-L, tokens (L-dw-8ray-tok-20m)",
         "Edit Index † (frame-set)"),
        ("noise_ablation/L-dw-noiseless-20m", "dw-noiseless", "Transformer-L, frames (L-dw-noiseless-20m)",
         "Edit Index (ray-zone)")]
OUT = REPO / "experiments/probe_targets/outputs/probe_target_sweep.png"

rows = []
for run, inst, label, ei in RUNS:
    sp = REPO / "runs" / run / "scores.json"
    if not sp.exists():
        continue
    with h5py.File(REPO / "datasets/discworld" / inst / "eval/edits.h5", "r") as f:
        sim = json.loads(f.attrs["config_json"])["dataset"]["sim"]
    blocks = json.loads(sp.read_text())["bases"]
    targets = {}
    for key, b in blocks.items():
        t = categorical_target(key)
        if t is not None and all(b["best"].get(ed) for ed in ("PI", "ND", "GS")):
            targets[key] = (t.n_cells(sim), b)
    if targets:
        rows.append(dict(label=label, ei_name=ei, targets=targets))
        print(run, {k: c for k, (c, _) in sorted(targets.items(), key=lambda kv: kv[1][0])})
fig = sweep_figure(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
