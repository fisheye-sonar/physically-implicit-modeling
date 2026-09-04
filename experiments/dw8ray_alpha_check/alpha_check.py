#!/usr/bin/env python
"""L-dw-8ray-20m: the canonical PI sweep is pinned at its top alpha (175) — extend it.

The canonical alpha grid (0.1 … 175) was set on 128-ray instances. On dw-8ray the best PI
arm is the LAST grid value with the index still rising (frustum: alpha 100 -> +0.264,
175 -> +0.297; fidelity 0.90 -> 1.11), so the reported +0.297 is a lower bound. This
re-runs the identical canonical arm (`bench.pinv_arm`, z-space, cached LIN probes — a
cache hit or nothing) at larger alphas on both bases and every point, and records where
the index peaks and where the fidelity guard crosses 1. Diagnostic only: the canonical
scores.json is NOT touched.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pim.environments.discworld import bench as dwb  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402

RUN = REPO / "runs" / "ray_ablation" / "L-dw-8ray-20m"
INST = REPO / "datasets" / "discworld" / "dw-8ray"
ALPHAS = [100.0, 175.0, 250.0, 350.0, 500.0, 700.0, 1000.0, 1500.0]
OUT = Path(__file__).resolve().parent / "alpha_check_L-dw-8ray-20m.json"


def main() -> None:
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_checkpoint(RUN / "best_model.pt", device=dev)
    model.eval()
    res = {"run": RUN.name, "alphas": ALPHAS, "bases": {}}
    for basis in ("cartesian", "frustum"):
        b = dwb.load_bench(model, n=192, target="full", basis_name=basis, data_dir=INST / "eval")
        lin = dwb.fit_probes(model, target="full", n_seq=30_000, family="linear",
                             basis_name=basis, data_dir=INST / "probe",
                             cache_dir=RUN / "probes", log=print)     # cache HIT expected
        u = dwb.unsteered(model, b)
        rows = []
        for dims in ("pos", "all"):
            for r in dwb.pinv_arm(model, b, lin, ALPHAS, space="zspace", dims=dims):
                r["fidelity_ratio"] = dwb.fidelity_ratio(r, u)
                rows.append({k: v for k, v in r.items() if np.isscalar(v)})
        best = max(rows, key=lambda r: r["edit_index"])
        best_ok = max((r for r in rows if r["fidelity_ratio"] <= 1.0), key=lambda r: r["edit_index"],
                      default=None)
        res["bases"][basis] = {"unedited": u["edit_index"], "arms": rows,
                               "best": best, "best_fid_le_1": best_ok}
        print(f"{basis}: best {best['dims']}·pt{best['point']}·a{best['alpha']:g} EI "
              f"{best['edit_index']:+.3f} fid {best['fidelity_ratio']:.2f}; best with fid<=1: "
              + (f"{best_ok['dims']}·pt{best_ok['point']}·a{best_ok['alpha']:g} EI "
                 f"{best_ok['edit_index']:+.3f} fid {best_ok['fidelity_ratio']:.2f}" if best_ok else "none"),
              flush=True)
        for pt in range(len(lin)):
            line = [(a["alpha"], round(a["edit_index"], 3), round(a["fidelity_ratio"], 2))
                    for a in rows if a["dims"] == "pos" and a["point"] == pt]
            print(f"   pos pt{pt}: " + " ".join(f"a{al:g}:{ei:+.2f}/{f:.2f}" for al, ei, f in line))
    res["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"done  {OUT}  [{res['minutes']} min]")


if __name__ == "__main__":
    main()
