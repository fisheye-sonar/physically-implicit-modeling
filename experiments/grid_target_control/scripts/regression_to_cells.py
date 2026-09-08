"""The regression probes scored ON THE GRID AXIS: map the canonical (frustum, full-state)
probes' continuous read-out to cells and compute the same per-cell error / skill the
grid classification probes report. This is the ceiling the classification probes should
approach — if the two agree, the grid metric is simply harsher (bin edges); if the
regression-to-cells number is far higher, the classification fits are under-optimised.
Held-out sequences: probe_250k[200000:], seen by neither fit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid import G, N_CLASSES, N_OBJ, TAG, cell_of, cell_of_frustum, label_frames, labels_from_cells  # noqa: E402

from pim.environments.discworld import arms as dwa  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402
from pim.probes.base import collect_residuals  # noqa: E402
from pim.probes.cache import ProbeCache  # noqa: E402

EXP = REPO / "experiments" / "grid_target_control"
DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/noise_ablation/L-dw-noiseless-20m")
    ap.add_argument("--points", type=int, nargs="+", default=[2, 6, 8])
    ap.add_argument("--n-seq", type=int, default=4000)
    ap.add_argument("--start", type=int, default=200_000)
    ap.add_argument("--grid-json", default=str(EXP / f"scores/grid_probes{TAG}.json"))
    a = ap.parse_args()
    run_dir = REPO / a.run
    inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
    root = REPO / "datasets/discworld" / inst
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV)
    with h5py.File(root / "probe_250k/test.h5", "r") as f:
        obs = f["obs_intensity"][a.start:a.start + a.n_seq].astype(np.float32)
        pos = f["positions"][a.start:a.start + a.n_seq, :, :N_OBJ, :].astype(np.float32)
    sim = json.load(open(root / "probe_250k/dataset.json"))["sim"]
    T = getattr(model, "state_span", obs.shape[1])
    obs, pos = obs[:, :T], pos[:, :T]
    y_true, _ = label_frames(pos, sim)                                  # (N, T, G)
    true_cells = cell_of(pos, sim)                                      # (N, T, 2)
    maj_err = float((y_true > 0).mean())                                # majority = empty
    grid_res = json.loads(Path(a.grid_json).read_text()) if Path(a.grid_json).exists() else {"points": {}}
    store = ProbeCache(EXP / "probes")
    out = {"run": a.run, "n_seq": a.n_seq, "held_out_from": a.start, "majority_error": maj_err, "points": {}}
    for ell in a.points:
        R = collect_residuals(model, obs, batch=64, points=[ell])[0]   # (N, T, d)
        X = torch.from_numpy(R.reshape(-1, R.shape[-1])).to(DEV)
        row = {}
        for fam in ("linear", "mlp"):
            probes = dwa.fit_probes(model, target="full", n_seq=30_000, family=fam, basis_name="frustum",
                                    data_dir=root / "probe", cache_dir=run_dir / "probes", log=None)
            probe = probes[ell][0]
            with torch.no_grad():
                pred = torch.cat([probe(X[i:i + 65536]) for i in range(0, len(X), 65536)]).cpu().numpy()
            pred = pred.reshape(a.n_seq, T, -1)                          # (u0, iy0, u1, iy1, v...)
            u = pred[..., [0, 2]]
            iy = pred[..., [1, 3]]
            cells = cell_of_frustum(u, iy, sim)                          # (N, T, 2)
            lab = labels_from_cells(cells, 1.0 / np.maximum(iy, 1e-6))
            err = float((lab != y_true).mean())
            obj_ok = float((cells == true_cells).mean())
            row[f"regression_{fam}"] = {"cell_error_rate": err * 100, "skill": 1 - err / maj_err,
                                        "object_in_right_cell": obj_ok}
            g = grid_res["points"].get(str(ell), {}).get(fam)
            if g:
                p2, _ = store.load(g["cache"]["file"], g["cache"]["prov"], device=DEV)
                with torch.no_grad():
                    lg = torch.cat([p2(X[i:i + 65536]) for i in range(0, len(X), 65536)])   # (rows, G, 3)
                    # the object's predicted cell = the cell where its class logit margin is largest
                    margin = (lg[..., 1:] - lg[..., :1])                                    # (rows, G, 2)
                    pc = margin.argmax(1).cpu().numpy().reshape(a.n_seq, T, 2)
                    hat = lg.argmax(-1).cpu().numpy().reshape(a.n_seq, T, G)
                errc = float((hat != y_true).mean())
                row[f"grid_{fam}"] = {"cell_error_rate": errc * 100, "skill": 1 - errc / maj_err,
                                      "object_in_right_cell": float((pc == true_cells).mean()),
                                      "fit_reported_skill": g["skill"]}
        out["points"][str(ell)] = row
        print(f"point {ell}:")
        for k, v in row.items():
            print(f"   {k:18s} cell-err {v['cell_error_rate']:.3f}%  skill {v['skill']:+.3f}  object in right cell {v['object_in_right_cell']:.3f}")
        del R, X
    (EXP / f"scores/regression_to_cells{TAG}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
