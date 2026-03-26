"""Dataset generation.

Generates a large collection of (1D observation sequence, latent state) pairs
and writes them to a compressed HDF5 file.  A companion JSON config file is
written alongside for human-readable reference.

HDF5 schema
-----------
All arrays are padded to ``max_objects`` along the object axis so that every
sample has a uniform shape, regardless of how many objects were actually
spawned.  The ``n_objects`` array records the true count per sample.

  obs_intensity  (N, n_frames, obs_res)          float32  reflectivity of first hit; 0=miss/bg
  obs_depth      (N, n_frames, obs_res)          float32  depth of first hit per ray; 0=miss
  obs_id         (N, n_frames, obs_res)          int8     object index of first hit; -1=miss
  is_visible     (N, n_frames, max_objects)      bool     True if object overlaps frustum
  positions      (N, n_frames, max_objects, 2)   float32  (x, y) per object per frame
  velocities     (N, n_frames, max_objects, 2)   float32  velocity per object per frame
  colors         (N, max_objects, 3)             float32  RGB colour, padded with zeros
  reflectivities (N, max_objects)                float32  per-object reflectivity, padded 0
  n_objects      (N,)                            uint8    actual object count per sample
  seeds          (N,)                            int64    RNG seed used for each sample
"""

from __future__ import annotations

import dataclasses
import json
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from pim.config import SimConfig
from pim.renderer import render_scene
from pim.sim import compute_visibility, simulate

# ── DatasetConfig ─────────────────────────────────────────────────────────────


@dataclass
class DatasetConfig:
    """Top-level configuration for one dataset generation run."""

    n_samples: int = 100_000
    sim: SimConfig = field(default_factory=SimConfig)
    base_seed: int = 0
    # parallelism — set n_workers=0 to run single-process (useful for debugging)
    n_workers: int = 4
    # how many samples to accumulate in RAM before flushing to HDF5
    write_batch: int = 512
    # chunk size along the sample axis inside HDF5 (affects random-access speed)
    hdf5_chunk: int = 64
    compression: str = "gzip"
    compression_level: int = 4


# ── Worker (module-level so multiprocessing can pickle it) ────────────────────


def _generate_one(args: tuple[int, SimConfig, int]) -> dict:
    """Generate one sample.  Runs in worker processes.

    Returns a dict of numpy arrays padded to ``max_obj`` along the object axis.
    On rare rejection-sampler failures the seed is offset and retried.
    """
    seed, base_cfg, max_obj = args
    cfg = dataclasses.replace(base_cfg, seed=int(seed))

    for attempt in range(10):
        try:
            if attempt:
                cfg = dataclasses.replace(cfg, seed=int(seed) + attempt * 1_000_000)
            scene = simulate(cfg)
            break
        except RuntimeError:
            if attempt == 9:
                raise

    obs_depth, obs_id, obs_intensity = render_scene(scene)
    vis = compute_visibility(scene)  # (n_frames, n)
    n = scene.positions.shape[1]

    pos_out = np.zeros((cfg.n_frames, max_obj, 2), dtype=np.float32)
    vel_out = np.zeros((cfg.n_frames, max_obj, 2), dtype=np.float32)
    col_out = np.zeros((max_obj, 3), dtype=np.float32)
    refl_out = np.zeros((max_obj,), dtype=np.float32)
    vis_out = np.zeros((cfg.n_frames, max_obj), dtype=bool)

    pos_out[:, :n] = scene.positions.astype(np.float32)
    vel_out[:, :n] = scene.velocities.astype(np.float32)
    col_out[:n] = scene.colors.astype(np.float32)
    refl_out[:n] = scene.reflectivities.astype(np.float32)
    vis_out[:, :n] = vis

    return {
        "obs_intensity": obs_intensity.astype(np.float32),
        "obs_depth": obs_depth.astype(np.float32),
        "obs_id": obs_id.astype(np.int8),
        "is_visible": vis_out,
        "positions": pos_out,
        "velocities": vel_out,
        "colors": col_out,
        "reflectivities": refl_out,
        "n_objects": np.uint8(n),
        "seed": np.int64(cfg.seed),
    }


# ── HDF5 helpers ──────────────────────────────────────────────────────────────


def _create_datasets(hf: h5py.File, dcfg: DatasetConfig, max_obj: int) -> None:
    N, F, R = dcfg.n_samples, dcfg.sim.n_frames, dcfg.sim.obs_res
    C = dcfg.hdf5_chunk
    kw = dict(compression=dcfg.compression, compression_opts=dcfg.compression_level)

    hf.create_dataset(
        "obs_intensity", (N, F, R), dtype="float32", chunks=(C, F, R), **kw
    )
    hf.create_dataset("obs_depth", (N, F, R), dtype="float32", chunks=(C, F, R), **kw)
    hf.create_dataset("obs_id", (N, F, R), dtype="int8", chunks=(C, F, R), **kw)
    hf.create_dataset(
        "is_visible", (N, F, max_obj), dtype="bool", chunks=(C, F, max_obj), **kw
    )
    hf.create_dataset(
        "positions",
        (N, F, max_obj, 2),
        dtype="float32",
        chunks=(C, F, max_obj, 2),
        **kw,
    )
    hf.create_dataset(
        "velocities",
        (N, F, max_obj, 2),
        dtype="float32",
        chunks=(C, F, max_obj, 2),
        **kw,
    )
    hf.create_dataset(
        "colors", (N, max_obj, 3), dtype="float32", chunks=(C, max_obj, 3), **kw
    )
    hf.create_dataset(
        "reflectivities", (N, max_obj), dtype="float32", chunks=(C, max_obj), **kw
    )
    hf.create_dataset("n_objects", (N,), dtype="uint8", chunks=(min(C * F, N),), **kw)
    hf.create_dataset("seeds", (N,), dtype="int64", chunks=(min(C * F, N),), **kw)


def _write_batch(hf: h5py.File, batch: list[dict], start: int) -> None:
    end = start + len(batch)
    hf["obs_intensity"][start:end] = np.stack([s["obs_intensity"] for s in batch])
    hf["obs_depth"][start:end] = np.stack([s["obs_depth"] for s in batch])
    hf["obs_id"][start:end] = np.stack([s["obs_id"] for s in batch])
    hf["is_visible"][start:end] = np.stack([s["is_visible"] for s in batch])
    hf["positions"][start:end] = np.stack([s["positions"] for s in batch])
    hf["velocities"][start:end] = np.stack([s["velocities"] for s in batch])
    hf["colors"][start:end] = np.stack([s["colors"] for s in batch])
    hf["reflectivities"][start:end] = np.stack([s["reflectivities"] for s in batch])
    hf["n_objects"][start:end] = np.array(
        [s["n_objects"] for s in batch], dtype=np.uint8
    )
    hf["seeds"][start:end] = np.array([s["seed"] for s in batch], dtype=np.int64)


# ── Main entry point ──────────────────────────────────────────────────────────


def generate_dataset(dcfg: DatasetConfig, output_dir: str | Path) -> None:
    """Generate a dataset and write it into ``output_dir``.

    The directory is created if it does not exist.  If it exists and is
    non-empty the function prints an error and returns without writing anything.

    Files written:
        <output_dir>/dataset.h5    — HDF5 data
        <output_dir>/dataset.json  — human-readable config + schema
    """
    output_dir = Path(output_dir)

    if output_dir.exists():
        contents = list(output_dir.iterdir())
        if contents:
            print(
                f"Error: output directory '{output_dir}' already exists and is not empty "
                f"({len(contents)} item(s) found).  Halting to avoid overwriting data."
            )
            return
    else:
        output_dir.mkdir(parents=True)

    output_path = output_dir / "dataset.h5"
    json_path = output_dir / "dataset.json"

    max_obj = (
        dcfg.sim.n_objects if dcfg.sim.n_objects is not None else dcfg.sim.n_objects_max
    )

    # ── Config JSON ───────────────────────────────────────────────────────
    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": dataclasses.asdict(dcfg),
        "schema": {
            "obs_intensity": f"float32  (N, n_frames={dcfg.sim.n_frames}, obs_res={dcfg.sim.obs_res})  — intensity in [0,1]; 0=background",
            "obs_depth": "float32  (N, n_frames, obs_res)  — depth of first hit; 0=miss",
            "obs_id": "int8     (N, n_frames, obs_res)  — object index, -1=miss",
            "is_visible": f"bool     (N, n_frames, max_objects={max_obj})  — partial frustum overlap per object",
            "positions": f"float32  (N, n_frames, max_objects={max_obj}, 2)  — (x, y)",
            "velocities": "float32  (N, n_frames, max_objects, 2)  — (vx, vy)",
            "colors": "float32  (N, max_objects, 3)  — RGB, zero-padded",
            "reflectivities": f"float32  (N, max_objects={max_obj})  — per-object reflectivity, zero-padded",
            "n_objects": "uint8    (N,)  — true object count per sample",
            "seeds": "int64    (N,)  — RNG seed per sample",
        },
    }
    config_json = json.dumps(meta, indent=2)
    json_path.write_text(config_json)

    # ── Worker args: one per sample, each with its own seed ──────────────
    seeds = dcfg.base_seed + np.arange(dcfg.n_samples, dtype=np.int64)
    args = [(int(s), dcfg.sim, max_obj) for s in seeds]

    chunksize = max(1, dcfg.write_batch // max(1, dcfg.n_workers))

    pool = mp.Pool(dcfg.n_workers) if dcfg.n_workers > 0 else None
    try:
        iterator = (
            pool.imap(_generate_one, args, chunksize=chunksize)
            if pool is not None
            else map(_generate_one, args)
        )

        written = 0
        batch: list[dict] = []

        with h5py.File(output_path, "w") as hf:
            hf.attrs["config_json"] = config_json
            _create_datasets(hf, dcfg, max_obj)

            t0 = time.perf_counter()
            with tqdm(
                total=dcfg.n_samples,
                unit="sample",
                dynamic_ncols=True,
                desc="generating",
            ) as pbar:
                for sample in iterator:
                    batch.append(sample)
                    pbar.update(1)

                    if len(batch) >= dcfg.write_batch:
                        _write_batch(hf, batch, written)
                        written += len(batch)
                        batch = []

                if batch:
                    _write_batch(hf, batch, written)
                    written += len(batch)

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    elapsed = time.perf_counter() - t0
    size_mb = output_path.stat().st_size / 1e6
    print(
        f"\n{dcfg.n_samples:,} samples  |  "
        f"{elapsed:.1f}s  ({dcfg.n_samples / elapsed:.0f} samples/s)  |  "
        f"{size_mb:.1f} MB  →  {output_path}"
    )
    print(f"config     →  {json_path}")
