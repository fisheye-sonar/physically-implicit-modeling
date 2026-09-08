"""Stage C (part) — the OBSERVATION floor for the grid target: the same 3-way-per-cell
probes fitted to the causal input history (right-aligned, block 0 = the current frame —
the layout under which a linear probe can express a current-frame lookup; master_eval
b4) instead of the residual stream, on the same 200k sequences and split. What a shallow
read of the input already gives on this target; the random-init floor is
`fit_grid_probes.py --random-init`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_grid_probes import DEV, EXP, SEED, load_probe_split  # noqa: E402
from grid import G, N_CLASSES, ND, NU, TAG, label_frames  # noqa: E402

from pim.probes.baselines import CausalHistory, fit_probe_stream  # noqa: E402
from pim.probes.cache import ProbeCache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="dw-noiseless")
    ap.add_argument("--probe-split", default=None)
    ap.add_argument("--n-seq", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--span", type=int, default=39, help="frames per sequence the model sees")
    ap.add_argument("--families", nargs="+", default=["linear", "mlp"])
    a = ap.parse_args()
    split_dir = Path(a.probe_split) if a.probe_split else REPO / "datasets/discworld" / a.instance / "probe_250k"
    obs, pos, sim = load_probe_split(split_dir, a.n_seq)
    obs = obs[:, :a.span]
    y_np, _ = label_frames(pos[:, :a.span], sim)
    N, T = y_np.shape[:2]
    y = torch.from_numpy(y_np).to(DEV)
    src = torch.from_numpy(obs).to(DEV)                       # 4 GB at 200k
    hist = CausalHistory(src, kind="dense", align="right")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N)
    tr_seq, te_seq = perm[: int(0.8 * N)], perm[int(0.8 * N):]
    store = ProbeCache(EXP / "probes")
    out_path = EXP / "scores" / f"grid_probes_observation{TAG}.json"
    res = {"instance": a.instance, "probe_split": str(split_dir.relative_to(REPO)), "n_seq": N,
           "n_frames": T, "grid": {"NU": NU, "ND": ND, "G": G, "n_classes": N_CLASSES},
           "epochs": a.epochs, "layout": "right-aligned causal history", "families": {}}
    print(f"observation floor · {a.instance} · {N:,} seq x {T} · d_in {hist.dim} · epochs {a.epochs}", flush=True)
    for fam in a.families:
        t0 = time.time()
        fname, prov = store.key(None, kind="grid3_observation", grid=f"{NU}x{ND}", align="right",
                                n_seq=int(N), epochs=int(a.epochs), family=fam, seed=SEED,
                                holdout=0.2, split="sequence", data=str(split_dir.resolve()), span=T)
        hit = store.load(fname, prov, device=DEV)
        if hit is None:
            probe, st = fit_probe_stream(hist, y, tr_seq, te_seq, hidden=None if fam == "linear" else 128,
                                         n_classes=N_CLASSES, seed=SEED, epochs=a.epochs, batch=4096, log=None)
            st["skill"] = 1.0 - st["error_rate"] / st["majority_class_error_rate"]
            st["minutes"] = (time.time() - t0) / 60
            store.store(fname, prov, (probe, st))
        else:
            st = hit[1]
        res["families"][fam] = {**st, "cache": {"file": fname, "prov": prov}}
        out_path.write_text(json.dumps(res, indent=1))
        print(f"  {fam}: err {st['error_rate']:.3f}% (in-sample {st['error_rate_insample']:.3f}%, majority "
              f"{st['majority_class_error_rate']:.3f}%) skill {st['skill']:+.4f} [{st.get('minutes', 0):.1f} min]", flush=True)


if __name__ == "__main__":
    main()
