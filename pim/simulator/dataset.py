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
  radii          (N, max_objects)                float32  per-object radius, padded 0
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

from .config import SimConfig
from .renderer import render_scene
from .sim import Scene, compute_visibility, simulate

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


# ── Sample loader ─────────────────────────────────────────────────────────────


def load_sample(
    path: str, idx: int
) -> tuple[Scene, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct a Scene and stored observations from one HDF5 row.

    Returns
    -------
    scene        : Scene with positions, velocities, etc. unpadded to true n_objects
    obs_depth    : (T, R) float32
    obs_id       : (T, R) int8
    obs_intensity: (T, R) float32
    """
    with h5py.File(path, "r") as f:
        cfg = SimConfig(**json.loads(f.attrs["config_json"])["dataset"]["sim"])
        n = int(f["n_objects"][idx])
        positions      = f["positions"][idx, :, :n, :].astype(np.float64)   # (T, n, 2)
        velocities     = f["velocities"][idx, :, :n, :].astype(np.float64)  # (T, n, 2)
        colors         = f["colors"][idx, :n, :].astype(np.float64)          # (n, 3)
        reflectivities = f["reflectivities"][idx, :n].astype(np.float64)     # (n,)
        radii          = f["radii"][idx, :n].astype(np.float64)              # (n,)
        obs_depth      = f["obs_depth"][idx].astype(np.float32)             # (T, R)
        obs_id         = f["obs_id"][idx]                                    # (T, R)
        obs_intensity  = f["obs_intensity"][idx].astype(np.float32)         # (T, R)
    scene = Scene(
        positions=positions, velocities=velocities, radii=radii,
        colors=colors, reflectivities=reflectivities, config=cfg,
    )
    return scene, obs_depth, obs_id, obs_intensity


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
    # `reconstruct_clean_obs` recovers the noiseless render from (obs_id, reflectivities),
    # which is exact ONLY for the flat renderer where intensity == reflectivity. Under soft
    # rendering (antialiasing / shading / blur) it is not recoverable, so store it.
    obs_clean = None
    if soft_enabled(cfg):
        obs_clean = render_scene(
            dataclasses.replace(scene, config=dataclasses.replace(cfg, obs_noise_std=0.0))
        )[2].astype(np.float32)
    vis = compute_visibility(scene)  # (n_frames, n)
    n = scene.positions.shape[1]

    pos_out = np.zeros((cfg.n_frames, max_obj, 2), dtype=np.float32)
    vel_out = np.zeros((cfg.n_frames, max_obj, 2), dtype=np.float32)
    col_out = np.zeros((max_obj, 3), dtype=np.float32)
    radii_out = np.zeros((max_obj,), dtype=np.float32)
    refl_out = np.zeros((max_obj,), dtype=np.float32)
    vis_out = np.zeros((cfg.n_frames, max_obj), dtype=bool)

    pos_out[:, :n] = scene.positions.astype(np.float32)
    vel_out[:, :n] = scene.velocities.astype(np.float32)
    col_out[:n] = scene.colors.astype(np.float32)
    radii_out[:n] = scene.radii.astype(np.float32)
    refl_out[:n] = scene.reflectivities.astype(np.float32)
    vis_out[:, :n] = vis

    return {
        "obs_intensity": obs_intensity.astype(np.float32),
        **({"obs_clean": obs_clean} if obs_clean is not None else {}),
        "obs_depth": obs_depth.astype(np.float32),
        "obs_id": obs_id.astype(np.int8),
        "is_visible": vis_out,
        "positions": pos_out,
        "velocities": vel_out,
        "colors": col_out,
        "radii": radii_out,
        "reflectivities": refl_out,
        "n_objects": np.uint8(n),
        "seed": np.int64(cfg.seed),
    }


# ── HDF5 helpers ──────────────────────────────────────────────────────────────


from pim.simulator.soft_render import soft_enabled  # noqa: E402


def _create_datasets(hf: h5py.File, dcfg: DatasetConfig, max_obj: int) -> None:
    N, F, R = dcfg.n_samples, dcfg.sim.n_frames, dcfg.sim.obs_res
    C = dcfg.hdf5_chunk
    kw = dict(compression=dcfg.compression, compression_opts=dcfg.compression_level)

    hf.create_dataset(
        "obs_intensity", (N, F, R), dtype="float32", chunks=(C, F, R), **kw
    )
    if soft_enabled(dcfg.sim):
        hf.create_dataset(
            "obs_clean", (N, F, R), dtype="float32", chunks=(C, F, R), **kw
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
        "radii", (N, max_obj), dtype="float32", chunks=(C, max_obj), **kw
    )
    hf.create_dataset(
        "reflectivities", (N, max_obj), dtype="float32", chunks=(C, max_obj), **kw
    )
    hf.create_dataset("n_objects", (N,), dtype="uint8", chunks=(min(C * F, N),), **kw)
    hf.create_dataset("seeds", (N,), dtype="int64", chunks=(min(C * F, N),), **kw)


def _write_batch(hf: h5py.File, batch: list[dict], start: int) -> None:
    end = start + len(batch)
    hf["obs_intensity"][start:end] = np.stack([s["obs_intensity"] for s in batch])
    if "obs_clean" in batch[0]:
        hf["obs_clean"][start:end] = np.stack([s["obs_clean"] for s in batch])
    hf["obs_depth"][start:end] = np.stack([s["obs_depth"] for s in batch])
    hf["obs_id"][start:end] = np.stack([s["obs_id"] for s in batch])
    hf["is_visible"][start:end] = np.stack([s["is_visible"] for s in batch])
    hf["positions"][start:end] = np.stack([s["positions"] for s in batch])
    hf["velocities"][start:end] = np.stack([s["velocities"] for s in batch])
    hf["colors"][start:end] = np.stack([s["colors"] for s in batch])
    hf["radii"][start:end] = np.stack([s["radii"] for s in batch])
    hf["reflectivities"][start:end] = np.stack([s["reflectivities"] for s in batch])
    hf["n_objects"][start:end] = np.array(
        [s["n_objects"] for s in batch], dtype=np.uint8
    )
    hf["seeds"][start:end] = np.array([s["seed"] for s in batch], dtype=np.int64)


# ── Main entry point ──────────────────────────────────────────────────────────


def reconstruct_clean_obs(
    obs_id: np.ndarray,
    reflectivities: np.ndarray,
) -> np.ndarray:
    """Reconstruct noiseless observation intensities from stored obs_id and reflectivities.

    The clean intensity at each ray is determined entirely by which object (if any)
    the ray hits — no re-simulation needed.  Since ``obs_id`` and ``reflectivities``
    are already stored in every HDF5 file, this requires zero extra storage.

    Formula: ``clean[..., t, r] = reflectivities[..., obs_id[..., t, r]]``
    if ``obs_id[..., t, r] >= 0`` (object hit), else ``0.0`` (background/miss).

    Parameters
    ----------
    obs_id        : (T, R) or (N, T, R) int8 — stored object-hit index, -1=miss
    reflectivities: (max_obj,) or (N, max_obj) float32 — per-object reflectivity

    Returns
    -------
    clean_obs : same leading shape as obs_id, float32
    """
    clean = np.zeros(obs_id.shape, dtype=np.float32)
    hit = obs_id >= 0
    if obs_id.ndim == 2:  # single sample (T, R)
        clean[hit] = reflectivities[obs_id[hit].astype(np.intp)]
    else:  # batched (N, T, R)
        n_idx = np.broadcast_to(
            np.arange(obs_id.shape[0], dtype=np.intp)[:, None, None], obs_id.shape
        )
        clean[hit] = reflectivities[n_idx[hit], obs_id[hit].astype(np.intp)]
    return clean


def generate_dataset(dcfg: DatasetConfig, h5_path: str | Path) -> dict:
    """Generate a dataset and write it to ``h5_path``.

    The parent directory is created if it does not exist.  If the file already
    exists the function raises ``FileExistsError``.

    The HDF5 file's ``config_json`` attribute stores full metadata so that
    ``load_sample`` and other readers remain self-contained.  The returned
    metadata dict can be incorporated into a suite-level JSON by the caller.

    Parameters
    ----------
    dcfg    : DatasetConfig
    h5_path : path to the ``.h5`` file to create

    Returns
    -------
    meta : dict — the metadata written into the HDF5 attrs
    """
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    if h5_path.exists():
        raise FileExistsError(f"{h5_path} already exists — refusing to overwrite.")

    max_obj = (
        dcfg.sim.n_objects if dcfg.sim.n_objects is not None else dcfg.sim.n_objects_max
    )

    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": dataclasses.asdict(dcfg),
        "schema": {
            "obs_intensity": f"float32  (N, n_frames={dcfg.sim.n_frames}, obs_res={dcfg.sim.obs_res})  — noisy intensity in [0,1]; 0=background",
            "obs_depth":      "float32  (N, n_frames, obs_res)  — depth of first hit; 0=miss",
            "obs_id":         "int8     (N, n_frames, obs_res)  — object index, -1=miss",
            "is_visible":     f"bool     (N, n_frames, max_objects={max_obj})  — partial frustum overlap per object",
            "positions":      f"float32  (N, n_frames, max_objects={max_obj}, 2)  — (x, y)",
            "velocities":     "float32  (N, n_frames, max_objects, 2)  — (vx, vy)",
            "colors":         "float32  (N, max_objects, 3)  — RGB, zero-padded",
            "radii":          f"float32  (N, max_objects={max_obj})  — per-object radius, zero-padded",
            "reflectivities": f"float32  (N, max_objects={max_obj})  — per-object reflectivity, zero-padded",
            "n_objects":      "uint8    (N,)  — true object count per sample",
            "seeds":          "int64    (N,)  — RNG seed per sample",
            "_clean_obs_note": "Clean (noiseless) obs can be reconstructed via reconstruct_clean_obs(obs_id, reflectivities) — no extra storage needed.",
        },
    }
    config_json = json.dumps(meta, indent=2)

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

        with h5py.File(h5_path, "w") as hf:
            hf.attrs["config_json"] = config_json
            _create_datasets(hf, dcfg, max_obj)

            t0 = time.perf_counter()
            with tqdm(
                total=dcfg.n_samples,
                unit="sample",
                dynamic_ncols=True,
                desc=f"generating → {h5_path.name}",
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
    size_mb = h5_path.stat().st_size / 1e6
    print(
        f"  {dcfg.n_samples:,} samples  |  "
        f"{elapsed:.1f}s  ({dcfg.n_samples / elapsed:.0f} samples/s)  |  "
        f"{size_mb:.1f} MB  →  {h5_path}"
    )
    return meta
