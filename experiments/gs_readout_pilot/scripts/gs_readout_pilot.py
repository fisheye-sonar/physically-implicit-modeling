#!/usr/bin/env python
"""GS read-out landing pilot — does gradient steering actually REACH its target?

Question (Sevan, 2026-09-02): after the GS descent, how close is the probe's read-out to
the requested target state? If the descent never lands, GS's negative Edit Index on
discworld could just mean "not run for long enough"; if it lands and the generation
still ignores it, the write is inert to the dynamics — a different finding entirely.

What this measures, per configuration (basis x dims x start layer x alpha x n_steps):
  * at EVERY intervened layer, the read-out on the CHANGED dims (the teleported
    object's position, or position+velocity for `all`) before and after that layer's
    write — RMSE in the probe's own basis units, and the fraction of the requested
    teleport covered per edit (1 - |after| / |before|, mean and median over the bench);
  * the drift of the HELD dims (everything the edit was told to leave alone);
  * the canonical hook's own record (edit loss before/after, write ratio);
  * the Edit Index and fidelity ratio of the rollout the very same write produces.
The descent length is swept (100 = the canonical setting, 500, 2000) to separate
"under-converged" from "converged but ignored".

Everything is the canonical path: the run's cached MLP-128 probes (a cache HIT, no refit
— a miss would be a 20 GB job, so this runs under a capped unit), the same
`make_intervention_hook`/`_descend` the scorer uses, the same bench, the same scorecard.
Nothing here is a new editor; it is instrumentation wrapped around the existing one.

Results: experiments/gs_readout_pilot/scores/gs_readout_pilot_<run>.json — rewritten
atomically after every configuration, so a killed run keeps what it measured.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pim.editors.grad_steer import build_edit_spec, make_intervention_hook  # noqa: E402
from pim.environments.discworld import bench as dwb  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402


def rmse(t: torch.Tensor) -> float:
    return float(torch.sqrt((t.float() ** 2).mean())) if t.numel() else float("nan")


def landing_hook(pts, specs, start_layer, alpha, n_steps, rec, land):
    """The canonical GS hook, with a read-out measurement after each layer's write."""
    inner = make_intervention_hook(pts, specs, start_layer, alpha=alpha, n_steps=n_steps,
                                   record=rec)

    def hook(layer, x):
        out = inner(layer, x)
        if layer >= start_layer and layer in pts:
            with torch.no_grad():
                pr, sp = pts[layer], specs[layer]
                chg = sp.weight >= 1.0                 # the dims the edit drives
                held = (sp.weight > 0) & ~chg          # the dims it is told to hold
                d0, d1 = pr(x[:, -1]) - sp.values, pr(out[:, -1]) - sp.values
                n0 = torch.sqrt(((d0 * chg) ** 2).sum(1))   # per-edit distance to target
                n1 = torch.sqrt(((d1 * chg) ** 2).sum(1))
                frac = 1.0 - n1 / n0.clamp_min(1e-9)
                land[layer] = {
                    "changed_rmse_before": rmse(d0[chg]),
                    "changed_rmse_after": rmse(d1[chg]),
                    "held_rmse_before": rmse(d0[held]),
                    "held_drift_after": rmse(d1[held]),
                    "frac_covered_mean": float(frac.mean()),
                    "frac_covered_median": float(frac.median()),
                    "teleport_size_mean": float(n0.mean()),
                }
        return out

    return hook


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="runs/initial_othello_comparison/L-dw-20m")
    ap.add_argument("--bases", nargs="+", default=["frustum", "cartesian"])
    ap.add_argument("--dims", nargs="+", default=["pos", "all"])
    ap.add_argument("--start-layers", nargs="+", type=int, default=[0, 4, 8])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.2, 0.5, 1.0, 2.0])
    ap.add_argument("--steps", nargs="+", type=int, default=[100, 500, 2000])
    ap.add_argument("--beta", type=float, default=0.2, help="hold-the-rest weight (canonical 0.2)")
    ap.add_argument("--n", type=int, default=192, help="bench edits (canonical 192)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    run_dir = (REPO / a.run).resolve()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, info = load_checkpoint(run_dir / "best_model.pt", device=dev)
    model.eval()
    inst = (json.loads((run_dir / "config.json").read_text())
            .get("data", {}).get("instance", "dw-pn04"))
    inst_root = REPO / "datasets" / "discworld" / inst
    out_path = (Path(a.out) if a.out else
                REPO / "experiments" / "gs_readout_pilot" / "scores"
                / f"gs_readout_pilot_{run_dir.name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res = {"run": run_dir.name, "instance": inst, "arch": info.arch, "beta": a.beta,
           "n_edits": a.n, "device": dev, "sweep": vars(a), "configs": []}

    def save():
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1, default=float))
        os.replace(tmp, out_path)

    t0 = time.time()
    for basis in a.bases:
        b = dwb.load_bench(model, n=a.n, target="full", basis_name=basis,
                           data_dir=inst_root / "eval")
        mlp = dwb.fit_probes(model, target="full", n_seq=30_000, family="mlp",
                             basis_name=basis, data_dir=inst_root / "probe",
                             cache_dir=run_dir / "probes", log=print)
        u = dwb.unsteered(model, b)
        for dims in a.dims:
            cm = dwb.restrict_mask(b.change_mask, dims)
            for ls in a.start_layers:
                pts = {e: mlp[e][0] for e in mlp if e >= ls}
                specs = {}
                for e, pr in pts.items():
                    dwb.as_activations(model, e)
                    specs[e] = build_edit_spec(pr, model.flat_state(b.state), cm, b.tgt,
                                               beta=a.beta)
                for alpha in a.alphas:
                    for n_steps in a.steps:
                        t1, rec, land = time.time(), {}, {}
                        hook = landing_hook(pts, specs, ls, alpha, n_steps, rec, land)
                        roll = dwb._roll_hook(model, b.state, hook)
                        sc = dwb.score(model, b, roll, u)
                        last = max(land)
                        cfg = {"basis": basis, "dims": dims, "start_layer": ls,
                               "alpha": alpha, "n_steps": n_steps,
                               "edit_index": sc["edit_index"],
                               "fidelity_ratio": sc["fidelity_ratio"],
                               "first_frac_covered_mean": land[ls]["frac_covered_mean"],
                               "last_layer": last,
                               **{f"last_{k}": v for k, v in land[last].items()},
                               "per_layer": {str(e): {**land[e], **rec.get(e, {})}
                                             for e in sorted(land)},
                               "seconds": round(time.time() - t1, 1)}
                        res["configs"].append(cfg)
                        save()
                        L = land[last]
                        print(f"{basis:9s} {dims:3s} L{ls} a={alpha:<4g} steps={n_steps:<5d}"
                              f" EI {sc['edit_index']:+.3f} fid {sc['fidelity_ratio']:.2f}"
                              f" | covered L{ls} {land[ls]['frac_covered_mean'] * 100:5.1f}%"
                              f"  L{last} {L['frac_covered_mean'] * 100:5.1f}%"
                              f" (rmse {L['changed_rmse_before']:.3f}->"
                              f"{L['changed_rmse_after']:.3f}, held drift "
                              f"{L['held_drift_after']:.3f})  [{cfg['seconds']}s]", flush=True)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    save()
    print(f"done  {out_path}  [{res['minutes']} min]")


if __name__ == "__main__":
    main()
