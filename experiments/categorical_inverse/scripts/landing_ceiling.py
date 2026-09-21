"""The ceiling of the LANDING check (2026-09-20): how often does a run's own categorical probe read the TRUE
labels of the edited object off a NATURAL (unedited) residual? `readout_landed` demands an exact match on
every tile of the edited object (run centre AND run length); a probe that cannot do that on natural
residuals cannot be expected to do it on a written one, so the landing rate of a write is read against this.

    PYTHONPATH=$PWD .pim/bin/python experiments/categorical_inverse/scripts/landing_ceiling.py <topic>/<run> [...]

Read-only: the run's checkpoint and cached probes, the bench. Appends `landing_ceiling` to the run's preview JSON
in this experiment's scores/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.environments.discworld import arms as dwa, bench as dwb  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402

EXP, TARGET = REPO / "experiments" / "categorical_inverse", "appearance-fac"

for run in sys.argv[1:]:
    run_dir = REPO / "runs" / run
    pp = EXP / "scores" / f"preview_{run_dir.name}_{TARGET}.json"
    prev = json.loads(pp.read_text())
    S = json.loads((run_dir / "scores.json").read_text())["settings"]
    inst, basis = prev["instance"], prev["basis"]
    pt = int(prev["new"]["reported (best inside the guard)"]["point"])
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=dwb.DEV)
    model.eval()
    b = dwb.load_bench(model, n=S["dw_bench_n"], target=TARGET, basis_name=basis, instance=inst)
    rec = dwa.probe_recipe(TARGET, inst, n_seq=S["dw_probe_seqs"])
    out = {"point": pt, "criterion": "every tile of the edited object reads its label exactly"}
    with torch.no_grad():
        dwa.as_activations(model, pt)
        h0 = model.flat_state(b.state)
        for fam in ("linear", "mlp"):
            fits = dwa.fit_probes(model, target=TARGET, family=fam, basis_name=basis, cache_dir=run_dir / "probes",
                                  log=None, require_cached=True, **rec)
            lab = fits[pt][0](h0).argmax(-1)
            t = b.moves["tile"]
            out[fam] = {
                # the NATURAL residual read against the labels it actually has (the object's pre-edit cells)
                "natural_reads_true_labels": float((lab.gather(1, t) == b.moves["old"]).all(1).float().mean()),
                "per_tile_natural": float((lab.gather(1, t) == b.moves["old"]).float().mean()),
                "written_reads_target_labels": prev["landing"].get(fam, {}).get("written"),
            }
    prev["landing_ceiling"] = out
    pp.write_text(json.dumps(prev, indent=1, default=float))
    print(f"{run_dir.name} pt{pt}: " + " · ".join(
        f"{fam}: natural {out[fam]['natural_reads_true_labels']:.3f} (per tile {out[fam]['per_tile_natural']:.3f}) "
        f"vs written→target {out[fam]['written_reads_target_labels']:.3f}" for fam in ("linear", "mlp")), flush=True)
    del model
    torch.cuda.empty_cache()
