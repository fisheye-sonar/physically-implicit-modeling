"""The 20M-sequence discworld corpus: generation, stripping, and the streaming source.

Ported verbatim 2026-08-31 from ``discworld_scale/corpus.py`` (path resolution updated
for the new home; nothing else changed — the seed law below is load-bearing).

WHY THIS EXISTS
---------------
The Othello 20M corpus is 1.2 GB of int8 tokens and lives in RAM. A discworld sequence is
40 x 128 float32 = 20 KB, so 20M of them is 410 GB and must be streamed. This module generates
the corpus in shards, strips each shard to the fields training actually needs, and appends them
to one flat memmap the trainer reads directly.

DATA-GENERATING PROCEDURE — matched to `datasets/4_fixed_refl_inview`
---------------------------------------------------------------------
`position_noise_std=0.04` (NOT dset 17's 0.0), `obs_noise_std=0.2`, `fixed_reflectivities`,
`always_in_frustum`, `boundary=open`, 2 objects, 40 frames, 128 rays. Verified 2026-08-24 that
`position_noise_std` is the ONLY substantive difference between dsets 4 and 17 — the `soft_*`
and `omni2d` fields differ only because they postdate dset 4, and dset 17 carries their defaults
(`soft_edge=0.0`, `omni2d=False`, documented bit-for-bit identical).

SEEDS — must not overlap dset 4, or the evaluation on dset 4 is worthless
-------------------------------------------------------------------------
    dset 4      0 – 120,000
    dset 17     3,000,000 – 3,950,000
    THIS        10,000,000 + k * 500,000,000   for shard k

⛔ Spacing is 500M, not the shard size, because of `pim/simulator/dataset.py:111`:

        cfg = dataclasses.replace(cfg, seed=int(seed) + attempt * 1_000_000)

  A rejected sample RETRIES AT A DIFFERENT SEED, up to `max_gen_attempts=300` attempts, i.e. up
  to +299,000,000. Two consequences, both handled here:
    1. shard stride (500M) exceeds the max retry offset, so shards cannot collide;
    2. shard size (500k) is BELOW the 1M retry quantum, so within a shard a retry can never land
       on another sample's base seed. A 1M+ shard would silently duplicate sequences.
  Measured retry rate on 300k samples at these settings: **0.0000%** — but the structure above
  makes it safe rather than lucky. `verify()` asserts it after the fact.

STORAGE — 437 GB of the 500 GB budget
-------------------------------------
Per 1M samples the full suite is 47.5 GB, of which `obs_depth` (20.5) and `obs_id` (5.1) are dead
weight: training uses the NOISY observation, and `clean_obs` reconstruction is only needed on
dset 4, which already exists. Keeping obs + positions + velocities:

    obs.f32     (20M, 40, 128) float32   410 GB   flat memmap, uncompressed for streaming
    meta.h5     positions, velocities, seeds        26 GB
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[3]
SHARD_N, N_SHARDS = 500_000, 40
SEED_STRIDE = 500_000_000
N_TOTAL = SHARD_N * N_SHARDS                 # 20,000,000
FRAMES = 40
OBS_RES = 128   # observation WIDTH of the current instance (rays kept) — rebound by use_instance()
VAL_N = 2_000_000                            # matches Othello's val_fraction 0.1 on 20M
TRAIN_N = N_TOTAL - VAL_N

_COMMON_FLAGS = ["--n-objects", "2", "--frames", str(FRAMES),
                 "--boundary", "open", "--fixed-reflectivities", "--always-in-frustum"]
_RAYS_128 = ["--obs-res", "128"]
# dw-8ray (2026-09-03): 10 rays CAST, the two wall-aligned ones dropped -> 8 kept; radius 1.0
_RAYS_8 = ["--obs-res", "10", "--drop-edge-rays", "--radius", "1.0",
           "--max-edit-attempts", "2000"]   # radius-1 teleports need more tries (shards make 100 edits)
_NEW_RANGES = [(60_000_000_000, 80_000_000_000, "dw-8ray train"),
               (85_000_000_000, 85_400_000_000, "dw-8ray eval suite"),
               (980_000_000_000, 981_000_000_000, "dw-8ray probe suite"),
               (990_000_000_000, 991_000_000_000, "dw-8ray probe_large")]

# ── the instance registry ─────────────────────────────────────────────────────
# One entry per environment instance that owns a 20M streaming corpus. `forbidden`
# lists every seed range OTHER data lives in — verify() hard-fails on any overlap.
# ⛔ Cross-instance seed sharing is deliberate-only: dw-noiseless uses FRESH seeds
# because pairing with dw-pn04 is impossible (always_in_frustum accepts ICs by
# simulating the trajectory forward, consuming noise draws in the acceptance loop —
# measured 2026-08-31: same seed, noise off => unrelated worlds, drift ~5 units).
INSTANCES = {
    "dw-pn04": {
        "base_seed": 10_000_000,
        "obs_dim": 128,
        "sim_flags": _COMMON_FLAGS + _RAYS_128 + ["--position-noise", "0.04", "--obs-noise-std", "0.2"],
        "forbidden": [(0, 120_000, "dset4-era eval"), (3_000_000, 3_950_000, "dset17"),
                      (960_000_000_000, 961_000_000_000, "dw-pn04 probe_large (capacity sweep)"),
                      (970_000_000_000, 971_000_000_000, "dw-noiseless probe_large")] + _NEW_RANGES,
    },
    "dw-noiseless": {  # dw-pn04 with ALL noise off; everything else identical
        "base_seed": 30_000_000_000,
        "obs_dim": 128,
        "sim_flags": _COMMON_FLAGS + _RAYS_128 + ["--position-noise", "0.0", "--obs-noise-std", "0.0"],
        "forbidden": [(0, 120_000, "dset4-era eval"), (3_000_000, 3_950_000, "dset17"),
                      (10_000_000, 19_800_000_000, "dw-pn04 train"),
                      (52_000_000_000, 52_400_000_000, "dw-noiseless eval suite"),
                      (900_000_000_000, 901_000_000_000, "dw-pn04 probe suite"),
                      (950_000_000_000, 951_000_000_000, "dw-noiseless probe suite"),
                      (960_000_000_000, 961_000_000_000, "dw-pn04 probe_large (capacity sweep)"),
                      (970_000_000_000, 971_000_000_000, "dw-noiseless probe_large")] + _NEW_RANGES,
    },
    "dw-8ray": {  # dw-noiseless with disc radius 1.0 and 8 usable rays (10 cast, wall rays dropped)
        "base_seed": 60_000_000_000,
        "obs_dim": 8,
        "sim_flags": _COMMON_FLAGS + _RAYS_8 + ["--position-noise", "0.0", "--obs-noise-std", "0.0"],
        "forbidden": [(0, 120_000, "dset4-era eval"), (3_000_000, 3_950_000, "dset17"),
                      (10_000_000, 19_800_000_000, "dw-pn04 train"),
                      (30_000_000_000, 50_000_000_000, "dw-noiseless train"),
                      (52_000_000_000, 52_400_000_000, "dw-noiseless eval suite"),
                      (85_000_000_000, 85_400_000_000, "dw-8ray eval suite"),
                      (900_000_000_000, 901_000_000_000, "dw-pn04 probe suite"),
                      (950_000_000_000, 951_000_000_000, "dw-noiseless probe suite"),
                      (960_000_000_000, 961_000_000_000, "dw-pn04 probe_large (capacity sweep)"),
                      (970_000_000_000, 971_000_000_000, "dw-noiseless probe_large"),
                      (980_000_000_000, 981_000_000_000, "dw-8ray probe suite"),
                      (990_000_000_000, 991_000_000_000, "dw-8ray probe_large")],
    },
}


def instance_dir(inst: str) -> Path:
    return REPO / "datasets" / "discworld" / inst


def train_dir(inst: str) -> Path:
    d = instance_dir(inst) / "train"
    if inst == "dw-pn04" and not d.exists():          # pre-move legacy location
        return REPO / "datasets" / "20_dwscale_20m"
    return d


def _spec(inst: str) -> dict:
    if inst not in INSTANCES:
        raise KeyError(f"unknown instance {inst!r}; registered: {sorted(INSTANCES)}")
    return INSTANCES[inst]


# module-level default: the original canonical instance (back-compat for existing callers)
INSTANCE = "dw-pn04"
OUT = train_dir(INSTANCE)
BASE_SEED = INSTANCES[INSTANCE]["base_seed"]
SIM_FLAGS = INSTANCES[INSTANCE]["sim_flags"]
FORBIDDEN = INSTANCES[INSTANCE]["forbidden"]


def use_instance(inst: str) -> None:
    """Point every module function at one instance's paths, seeds, and flags.

    The generation/verify functions below deliberately kept their original (verified)
    bodies over module globals; this is the one switch that rebinds them. Call it
    FIRST — the __main__ entry does, from its argv."""
    global INSTANCE, OUT, BASE_SEED, SIM_FLAGS, FORBIDDEN, OBS_RES
    spec = _spec(inst)
    INSTANCE = inst
    OBS_RES = int(spec.get("obs_dim", 128))
    OUT = train_dir(inst)
    BASE_SEED = spec["base_seed"]
    SIM_FLAGS = spec["sim_flags"]
    FORBIDDEN = spec["forbidden"]


def shard_seed(k: int) -> int:
    return BASE_SEED + k * SEED_STRIDE


def obs_path() -> Path:
    return OUT / "obs.f32"


def _shard_dir(k: int) -> Path:
    return OUT / f"_shard_{k:03d}"


def generate_shard(k: int, workers: int = 8, log=print) -> None:
    """One shard via the shared generator. Idempotent: a stripped shard is skipped."""
    if (OUT / f"_done_{k:03d}").exists():
        log(f"  shard {k:03d} already done")
        return
    d = _shard_dir(k)
    if d.exists():
        subprocess.run(["rm", "-rf", str(d)], check=True)
    cmd = [sys.executable, str(REPO / "scripts" / "generate_dataset.py"), str(d),
           "--n-train", str(SHARD_N), "--n-val", "100", "--n-test", "100", "--n-edits", "100",
           *SIM_FLAGS, "--seed", str(shard_seed(k)), "--n-workers", str(workers),
           "--compression-level", "0"]
    env = {"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"}
    subprocess.run(cmd, check=True, env=env, cwd=str(REPO),
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def strip_shard(k: int, log=print) -> None:
    """Append this shard's obs to the flat memmap and its meta to meta.h5, then delete the shard.

    Dropping `obs_depth` and `obs_id` here is what keeps the corpus inside the disk budget.
    """
    if (OUT / f"_done_{k:03d}").exists():
        return
    src = _shard_dir(k) / "train.h5"
    lo = k * SHARD_N
    obs = np.memmap(obs_path(), dtype=np.float32, mode="r+",
                    shape=(N_TOTAL, FRAMES, OBS_RES))
    with h5py.File(src, "r") as f:
        n = f["obs_intensity"].shape[0]
        assert n == SHARD_N, f"shard {k} has {n} != {SHARD_N}"
        for i in range(0, n, 50_000):                       # chunked so RAM stays flat
            j = min(i + 50_000, n)
            obs[lo + i: lo + j] = f["obs_intensity"][i:j]
        meta = {kk: f[kk][:] for kk in ("positions", "velocities", "seeds",
                                        "reflectivities", "radii")}
    obs.flush()
    del obs
    with h5py.File(OUT / "meta.h5", "a") as g:
        for kk, v in meta.items():
            if kk not in g:
                g.create_dataset(kk, shape=(N_TOTAL, *v.shape[1:]), dtype=v.dtype,
                                 chunks=(min(4096, N_TOTAL), *v.shape[1:]))
            g[kk][lo: lo + SHARD_N] = v
    subprocess.run(["rm", "-rf", str(_shard_dir(k))], check=True)
    (OUT / f"_done_{k:03d}").touch()
    log(f"  shard {k:03d} stripped  (seeds {shard_seed(k):,}..{shard_seed(k) + SHARD_N:,})")


def verify(log=print) -> dict:
    """Seed disjointness from dset 4/17, no duplicates, and the obs file fully written."""
    with h5py.File(OUT / "meta.h5", "r") as g:
        s = g["seeds"][:]
    assert len(s) == N_TOTAL, f"{len(s):,} seeds, expected {N_TOTAL:,}"
    dups = len(s) - len(np.unique(s))
    assert dups == 0, f"{dups:,} DUPLICATE seeds — retry collision, corpus is not i.i.d."
    for lo, hi, name in FORBIDDEN:
        n_bad = int(((s >= lo) & (s < hi)).sum())
        assert n_bad == 0, f"{n_bad:,} seeds collide with {name} [{lo:,},{hi:,}) — evaluation void"
    sz = obs_path().stat().st_size
    want = N_TOTAL * FRAMES * OBS_RES * 4
    assert sz == want, f"obs.f32 is {sz:,} bytes, expected {want:,}"
    out = {"n": int(N_TOTAL), "duplicates": 0, "seed_min": int(s.min()), "seed_max": int(s.max()),
           "obs_bytes": sz, "disjoint_from": [n for _, _, n in FORBIDDEN]}
    log(f"  VERIFIED {N_TOTAL:,} sequences, seeds {s.min():,}..{s.max():,}, "
        f"0 duplicates, disjoint from dset4 + dset17, obs.f32 {sz / 1e9:.0f} GB")
    return out


def open_obs(mode: str = "r") -> np.memmap:
    return np.memmap(obs_path(), dtype=np.float32, mode=mode,
                     shape=(N_TOTAL, FRAMES, OBS_RES))


if __name__ == "__main__":
    import concurrent.futures as cf

    if len(sys.argv) > 1:
        use_instance(sys.argv[1])
    print(f"instance {INSTANCE}: corpus -> {OUT}, base seed {BASE_SEED:,}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if not obs_path().exists():                             # sparse allocate, filled per shard
        np.memmap(obs_path(), dtype=np.float32, mode="w+",
                  shape=(N_TOTAL, FRAMES, OBS_RES)).flush()
    todo = [k for k in range(N_SHARDS) if not (OUT / f"_done_{k:03d}").exists()]
    print(f"{len(todo)} shards to do (of {N_SHARDS}), 4 concurrent x 8 workers", flush=True)
    # 4 concurrent generators: one process saturates ~8 cores (measured 8 and 16 workers both
    # ~2,220 samples/s at ~810-880% CPU), so parallelism has to come from processes.
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(generate_shard, k): k for k in todo}
        for fut in cf.as_completed(futs):
            k = futs[fut]
            fut.result()
            strip_shard(k)
    info = verify()
    (OUT / "corpus.json").write_text(json.dumps(
        {**info, "shard_n": SHARD_N, "n_shards": N_SHARDS, "base_seed": BASE_SEED,
         "seed_stride": SEED_STRIDE, "sim_flags": SIM_FLAGS,
         "train_n": TRAIN_N, "val_n": VAL_N}, indent=1))
    print("corpus complete", flush=True)
