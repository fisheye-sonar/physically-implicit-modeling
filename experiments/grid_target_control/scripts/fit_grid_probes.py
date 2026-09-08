"""Stage A — fit the grid-cell classification probes on a trained discworld model.

One LIN and one MLP-128 probe per residual point, 3-way per cell (`grid.py`), on the
instance's LARGE probe split (default the first 200k of probe_250k), held out 80/20 BY
SEQUENCE (seed 0, the canonical split rule). Fitted through the canonical streamed
fitter (`pim.probes.baselines.fit_probe_stream` over `MemmapRows`): the residual stack of
one point is 200k × 39 × 512 × 4 B = 16 GB, collected to a memmap on the nvme
(`.scratch/`, the same place `fit_probes` uses) and streamed by sequence block, so
host and GPU memory stay small. Every fit is persisted (ProbeCache under
experiments/grid_target_control/probes/) BEFORE anything reads it.

Epochs default to 50 — the project's precedent for large-corpus fits (master_eval b3
observation floors on 250k sequences; the probe-capacity sweep), and 50 epochs over
200k sequences is 1.65× the gradient steps of the canonical 200 epochs over 30k.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid import G, N_CLASSES, N_OBJ, NU, ND, TAG, label_frames  # noqa: E402

from pim.models import load_checkpoint, n_points  # noqa: E402
from pim.probes.base import collect_residuals  # noqa: E402
from pim.probes.baselines import MemmapRows, fit_probe_stream, random_init_model  # noqa: E402
from pim.probes.cache import ProbeCache  # noqa: E402

EXP = REPO / "experiments" / "grid_target_control"
DEV = "cuda"
SEED = 0


def load_probe_split(split_dir: Path, n_seq: int):
    with h5py.File(split_dir / "test.h5", "r") as f:
        obs = f["obs_intensity"][:n_seq].astype(np.float32)
        pos = f["positions"][:n_seq, :, :N_OBJ, :].astype(np.float32)
    sim = json.load(open(split_dir / "dataset.json"))["sim"]
    return obs, pos, sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/noise_ablation/L-dw-noiseless-20m")
    ap.add_argument("--probe-split", default=None, help="default: <instance>/probe_250k")
    ap.add_argument("--n-seq", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--points", type=int, nargs="*", default=None, help="default: all")
    ap.add_argument("--families", nargs="+", default=["linear", "mlp"])
    ap.add_argument("--random-init", action="store_true",
                    help="the same architecture at random initialisation (the floor)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    run_dir = REPO / a.run
    cfg = json.loads((run_dir / "config.json").read_text())
    inst = cfg["data"]["instance"]
    split_dir = Path(a.probe_split) if a.probe_split else REPO / "datasets/discworld" / inst / "probe_250k"
    if a.random_init:
        _, info = load_checkpoint(run_dir / "best_model.pt", device="cpu")
        model = random_init_model(info.arch, info.model_config, seed=0, device=DEV).eval()
        label = f"random-init {info.arch}"
    else:
        model, info = load_checkpoint(run_dir / "best_model.pt", device=DEV)
        label = a.run
    NP = n_points(model)
    points = a.points if a.points else list(range(NP))
    out_path = EXP / "scores" / (a.out or (f"grid_probes_random_init{TAG}.json" if a.random_init else f"grid_probes{TAG}.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{label} · instance {inst} · probe split {split_dir.name} · n_seq {a.n_seq:,} · "
          f"grid {NU}x{ND}={G} cells x {N_CLASSES} classes · epochs {a.epochs}", flush=True)

    obs, pos, sim = load_probe_split(split_dir, a.n_seq)
    span = getattr(model, "state_span", obs.shape[1])
    obs = obs[:, :span]
    y_np, conflicts = label_frames(pos[:, :span], sim)              # (N, T, G) uint8
    N, T = y_np.shape[:2]
    frac_nonempty = float((y_np > 0).mean())
    print(f"  labels: {N:,} seq x {T} frames; non-empty cells {frac_nonempty:.4%} of entries; "
          f"shared-cell conflicts {conflicts:,} of {N * T:,} frames ({conflicts / (N * T):.3%})", flush=True)
    y = torch.from_numpy(y_np).to(DEV)                              # 1 GB at 200k
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N)
    tr_seq, te_seq = perm[: int(0.8 * N)], perm[int(0.8 * N):]

    store = ProbeCache(EXP / "probes")
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results.update({"run": label, "instance": inst, "probe_split": str(split_dir.relative_to(REPO)),
                    "n_seq": N, "n_frames": T, "grid": {"NU": NU, "ND": ND, "G": G, "n_classes": N_CLASSES},
                    "epochs": a.epochs, "frac_nonempty": frac_nonempty, "conflict_frac": conflicts / (N * T)})
    results.setdefault("points", {})
    sdir = REPO / ".scratch"
    sdir.mkdir(exist_ok=True)
    for ell in points:
        t0 = time.time()
        keys = {}
        todo = []
        for fam in a.families:
            fname, prov = store.key(model, kind="grid3", grid=f"{NU}x{ND}", point=int(ell),
                                    n_seq=int(N), epochs=int(a.epochs), family=fam, seed=SEED,
                                    holdout=0.2, split="sequence", data=str(split_dir.resolve()))
            keys[fam] = (fname, prov)
            hit = store.load(fname, prov, device=DEV)
            if hit is None:
                todo.append(fam)
            else:
                print(f"  point {ell} {fam}: cache HIT {fname}", flush=True)
                results["points"].setdefault(str(ell), {})[fam] = {**hit[1], "cache": {"file": fname, "prov": prov}}
        if todo:
            tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False, dir=sdir)
            tmp.close()
            try:
                Rm = collect_residuals(model, obs, batch=64, memmap=tmp.name, points=[ell])   # (1, N, T, d)
                print(f"  point {ell}: residuals collected in {time.time() - t0:.0f}s "
                      f"({Rm.shape[1]:,} x {Rm.shape[2]} x {Rm.shape[3]} -> memmap)", flush=True)
                hist = MemmapRows(Rm[0], device=DEV)
                for fam in todo:
                    t1 = time.time()
                    probe, st = fit_probe_stream(hist, y, tr_seq, te_seq,
                                                 hidden=None if fam == "linear" else 128,
                                                 n_classes=N_CLASSES, seed=SEED, epochs=a.epochs,
                                                 batch=4096, log=None)
                    st["skill"] = 1.0 - st["error_rate"] / st["majority_class_error_rate"]
                    st["minutes"] = (time.time() - t1) / 60
                    store.store(*keys[fam], (probe, st))                 # persisted FIRST
                    st["cache"] = {"file": keys[fam][0], "prov": keys[fam][1]}
                    results["points"].setdefault(str(ell), {})[fam] = st
                    print(f"  point {ell} {fam}: err {st['error_rate']:.3f}% (in-sample {st['error_rate_insample']:.3f}%, "
                          f"majority {st['majority_class_error_rate']:.3f}%) skill {st['skill']:+.4f}  "
                          f"[{st['minutes']:.1f} min]", flush=True)
                    out_path.write_text(json.dumps(results, indent=1))
                del hist, Rm
            finally:
                os.unlink(tmp.name)
        out_path.write_text(json.dumps(results, indent=1))
        print(f"  point {ell} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    best = {fam: max(results["points"], key=lambda p: results["points"][p].get(fam, {}).get("skill", -9))
            for fam in a.families}
    results["best_point"] = best
    out_path.write_text(json.dumps(results, indent=1))
    print("best points:", {f: (p, round(results["points"][p][f]["skill"], 4)) for f, p in best.items()}, flush=True)


if __name__ == "__main__":
    main()
