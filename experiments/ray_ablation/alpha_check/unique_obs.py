#!/usr/bin/env python
"""How many distinct observations does the 20M-sequence dw-8ray corpus contain?

Counts, in one chunked pass over the flat memmap (N, 40, 8) float32:
  * distinct FRAMES (8-ray intensity vectors) — with noise off every ray is one of the
    background value and the two fixed reflectivities, so at most 3^8 = 6561 patterns
  * distinct SEQUENCES (whole 40 x 8 observation histories), via a 64-bit hash
  * the frame-pattern entropy (bits), and the most common patterns
Output: experiments/ray_ablation/alpha_check/unique_obs_<instance>.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.environments.discworld import bigcorpus as bc  # noqa: E402

INST = sys.argv[1] if len(sys.argv) > 1 else "dw-8ray"
CHUNK = 250_000


def main() -> None:
    t0 = time.time()
    bc.use_instance(INST)
    obs = bc.open_obs("r")                       # (N, 40, R)
    N, T, R = obs.shape
    levels = None
    counts = None
    seq_hashes = np.empty(N, np.uint64)
    n_seen = 0
    for i in range(0, N, CHUNK):
        x = np.asarray(obs[i : i + CHUNK])                       # (n, T, R) float32
        if levels is None:
            levels = np.unique(x[: min(len(x), 20_000)])
            print(f"{INST}: N={N:,} T={T} R={R}; intensity levels in the first 20k: "
                  f"{levels.tolist()}", flush=True)
            counts = np.zeros(len(levels) ** R, np.int64)
        codes = np.searchsorted(levels, x.reshape(-1, R))        # (n*T, R) level index
        assert (levels[codes] == x.reshape(-1, R)).all(), "unexpected intensity value"
        flat = (codes * (len(levels) ** np.arange(R))[None, :]).sum(1)
        counts += np.bincount(flat, minlength=len(counts))
        raw = np.ascontiguousarray(x).view(np.uint8).reshape(len(x), -1)
        for j in range(len(x)):
            seq_hashes[i + j] = int.from_bytes(hashlib.blake2b(raw[j].tobytes(), digest_size=8).digest(), "little")
        n_seen += len(x)
        if (i // CHUNK) % 8 == 0:
            print(f"  {n_seen:,} sequences  [{time.time() - t0:.0f}s]", flush=True)
    frames_total = N * T
    nz = counts[counts > 0]
    p = nz / frames_total
    ent = float(-(p * np.log2(p)).sum())
    top = np.argsort(-counts)[:10]
    uniq_seq = int(len(np.unique(seq_hashes)))
    res = {"instance": INST, "n_sequences": int(N), "frames_per_sequence": int(T), "rays": int(R),
           "intensity_levels": levels.tolist(), "possible_frame_patterns": int(len(levels) ** R),
           "distinct_frames": int(len(nz)), "total_frames": int(frames_total),
           "frame_pattern_entropy_bits": ent, "top10_frames": [
               {"pattern": [levels[(int(c) // len(levels) ** k) % len(levels)] for k in range(R)],
                "share": float(counts[c] / frames_total)} for c in top],
           "distinct_sequences": uniq_seq, "duplicate_sequences": int(N - uniq_seq),
           "minutes": round((time.time() - t0) / 60, 1)}
    out = Path(__file__).resolve().parent / f"unique_obs_{INST}.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "top10_frames"}, indent=1))
    print("top frames:", [(t["pattern"], round(t["share"], 4)) for t in res["top10_frames"][:5]])
    print(f"done  {out}")


if __name__ == "__main__":
    main()
