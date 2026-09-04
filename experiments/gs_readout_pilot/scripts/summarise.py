#!/usr/bin/env python
"""Read gs_readout_pilot_<run>.json -> the landing table (stdout, markdown).

One row per (basis, dims, start layer, alpha); the three descent lengths side by side, so
"did more steps change anything?" is a glance. `cov` = fraction of the requested teleport
the probe read-out covers after the write (1.0 = lands exactly), at the first intervened
layer and at the last (layer 8, after the whole sequential write); EI/fid = the canonical
Edit Index and fidelity ratio of the rollout the same write produced.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else
         Path(__file__).resolve().parents[1] / "scores" / "gs_readout_pilot_L-dw-20m.json")
R = json.loads(p.read_text())
steps = sorted({c["n_steps"] for c in R["configs"]})
rows = defaultdict(dict)
for c in R["configs"]:
    rows[(c["basis"], c["dims"], c["start_layer"], c["alpha"])][c["n_steps"]] = c

print(f"# {R['run']} · {R['instance']} · beta {R['beta']} · {R['n_edits']} edits · "
      f"{len(R['configs'])} configs · {R.get('minutes', '?')} min\n")
hdr = "| basis | dims | L | α | " + " | ".join(
    f"{s} steps: cov first / last · EI · fid" for s in steps) + " |"
print(hdr)
print("|" + "---|" * (4 + len(steps)))
for (basis, dims, ls, alpha), by in sorted(rows.items()):
    cells = []
    for s in steps:
        c = by.get(s)
        cells.append("—" if c is None else
                     f"{c['first_frac_covered_mean']:.3f} / {c['last_frac_covered_mean']:.3f}"
                     f" · {c['edit_index']:+.3f} · {c['fidelity_ratio']:.2f}")
    print(f"| {basis} | {dims} | {ls} | {alpha:g} | " + " | ".join(cells) + " |")

# the two summary numbers the question needs
cov = [c["last_frac_covered_mean"] for c in R["configs"]]
ei = [c["edit_index"] for c in R["configs"]]
print(f"\nlast-layer coverage over all configs: min {min(cov):.3f}, median "
      f"{sorted(cov)[len(cov) // 2]:.3f}, max {max(cov):.3f}")
print(f"Edit Index over all configs: min {min(ei):+.3f}, max {max(ei):+.3f}")
canon = [c for c in R["configs"] if c["n_steps"] == 100]
long_ = {(c["basis"], c["dims"], c["start_layer"], c["alpha"]): c
         for c in R["configs"] if c["n_steps"] == max(steps)}
d = [long_[(c["basis"], c["dims"], c["start_layer"], c["alpha"])]["edit_index"] - c["edit_index"]
     for c in canon if (c["basis"], c["dims"], c["start_layer"], c["alpha"]) in long_]
if d:
    print(f"EI change from 100 -> {max(steps)} steps: min {min(d):+.3f}, max {max(d):+.3f}, "
          f"mean {sum(d) / len(d):+.3f}")
