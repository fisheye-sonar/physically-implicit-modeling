"""Edits dataset generation.

Extends the normal dataset with a single position edit applied mid-sequence:
one object is teleported to a new location at ``edit_frame``, and the scene is
re-rendered from that point.  All downstream arrays (positions, observations,
visibility) reflect the edited trajectory.

HDF5 schema
-----------
All fields from the normal dataset, plus five edit-metadata fields:

  edit_frame   (N,)     int32    frame at which the edit is applied
  edit_object  (N,)     int8     index of the moved object
  edit_op      (N,)     uint8    operation code; 0 = set_position
  edit_value   (N, 2)   float32  new (x, y) position
  n_edits      (N,)     uint8    number of edits per sample (always 1 for now)

The ``positions`` array stores the **post-edit** trajectory.  The ``velocities``
array stores the original velocities unchanged — the object continues with the
same velocity from its new location after the edit.
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
from pim.sim import Scene, compute_visibility, frustum_half_width, simulate

# Operation codes
OP_SET_POSITION: int = 0


# ── EditDatasetConfig ──────────────────────────────────────────────────────────


@dataclass
class EditDatasetConfig:
    """Top-level configuration for one edits-dataset generation run."""

    n_samples: int = 10_000
    sim: SimConfig = field(default_factory=SimConfig)
    base_seed: int = 0
    n_workers: int = 4
    write_batch: int = 512
    hdf5_chunk: int = 64
    compression: str = "gzip"
    compression_level: int = 4

    # ── edit parameters ───────────────────────────────────────────────────
    # Frame at which the position edit is applied.  -1 means n_frames // 2.
    edit_frame: int = -1
    # If True, reject edits where the modified object leaves the frustum at
    # any frame in [edit_frame, n_frames).
    edit_always_in_frustum: bool = True
    # Maximum attempts to find a collision-free edit position before giving up.
    max_edit_attempts: int = 50


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sample_in_frustum(
    rng: np.random.Generator, cfg: SimConfig, margin: float
) -> np.ndarray:
    """Sample a random (x, y) strictly inside the frustum with the given margin.

    The margin is subtracted from every boundary, so a circle of radius
    ``margin`` centred at the returned point will be fully contained.
    """
    y_lo = cfg.y_near + margin
    y_hi = cfg.y_far - margin
    if y_lo >= y_hi:
        raise RuntimeError("Frustum too small for given margin — cannot sample position.")

    for _ in range(1_000):
        y = float(rng.uniform(y_lo, y_hi))
        x_lim = float(frustum_half_width(y, cfg)) - margin
        if x_lim <= 0:
            continue
        x = float(rng.uniform(-x_lim, x_lim))
        return np.array([x, y], dtype=np.float32)

    raise RuntimeError("Could not sample a valid position inside the frustum.")


# ── Worker (module-level so multiprocessing can pickle it) ────────────────────


def _generate_one_edit(
    args: tuple[int, SimConfig, int, int, bool, int],
) -> dict:
    """Generate one edited sample.  Runs in worker processes.

    Parameters (passed as a tuple for multiprocessing compatibility)
    ---------------------------------------------------------------
    seed, base_cfg, max_obj, edit_frame, edit_always_in_frustum, max_edit_attempts
    """
    seed, base_cfg, max_obj, edit_frame, edit_always_in_frustum, max_edit_attempts = args
    cfg = dataclasses.replace(base_cfg, seed=int(seed))

    # ── Generate base scene (same retry pattern as _generate_one) ─────────
    for attempt in range(10):
        try:
            if attempt:
                cfg = dataclasses.replace(cfg, seed=int(seed) + attempt * 1_000_000)
            scene = simulate(cfg)
            break
        except RuntimeError:
            if attempt == 9:
                raise

    T = cfg.n_frames
    n = scene.positions.shape[1]
    eff_edit_frame = edit_frame if edit_frame >= 0 else T // 2

    # ── Find a valid edit position ─────────────────────────────────────────
    # RNG isolated from simulation RNG to keep seeds independent.
    rng = np.random.default_rng(int(seed) + 2_000_000)
    obj_idx = int(rng.integers(0, n))

    new_pos = None
    for _ in range(max_edit_attempts):
        candidate = _sample_in_frustum(rng, cfg, margin=cfg.radius)
        delta = candidate - scene.positions[eff_edit_frame, obj_idx]

        ok = True
        for f in range(eff_edit_frame, T):
            edited_pos = scene.positions[f, obj_idx] + delta

            # Collision check with every other object at this frame
            for j in range(n):
                if j == obj_idx:
                    continue
                dist = float(np.linalg.norm(edited_pos - scene.positions[f, j]))
                if dist < cfg.collision_margin * 2.0 * cfg.radius:
                    ok = False
                    break
            if not ok:
                break

            # Optional frustum containment check
            if edit_always_in_frustum:
                x, y = float(edited_pos[0]), float(edited_pos[1])
                r = cfg.radius
                x_lim = float(frustum_half_width(y, cfg))
                if not (
                    y - r >= cfg.y_near
                    and y + r <= cfg.y_far
                    and abs(x) + r <= x_lim
                ):
                    ok = False
                    break

        if ok:
            new_pos = candidate
            break

    if new_pos is None:
        raise RuntimeError(
            f"Could not find a collision-free edit position for seed {seed} "
            f"after {max_edit_attempts} attempts."
        )

    # ── Apply edit: shift positions[edit_frame:, obj] by delta ────────────
    new_positions = scene.positions.copy()
    delta = new_pos - scene.positions[eff_edit_frame, obj_idx]
    new_positions[eff_edit_frame:, obj_idx] += delta

    modified_scene = Scene(
        positions=new_positions,
        velocities=scene.velocities,  # original velocities preserved
        radii=scene.radii,
        colors=scene.colors,
        reflectivities=scene.reflectivities,
        config=scene.config,
    )

    # ── Render and compute visibility with modified scene ─────────────────
    obs_depth, obs_id, obs_intensity = render_scene(modified_scene)
    vis = compute_visibility(modified_scene)

    # ── Pack output (same padding pattern as _generate_one) ───────────────
    pos_out  = np.zeros((T, max_obj, 2), dtype=np.float32)
    vel_out  = np.zeros((T, max_obj, 2), dtype=np.float32)
    col_out  = np.zeros((max_obj, 3),    dtype=np.float32)
    rad_out  = np.zeros((max_obj,),      dtype=np.float32)
    refl_out = np.zeros((max_obj,),      dtype=np.float32)
    vis_out  = np.zeros((T, max_obj),    dtype=bool)

    pos_out[:, :n]  = new_positions.astype(np.float32)
    vel_out[:, :n]  = modified_scene.velocities.astype(np.float32)
    col_out[:n]     = modified_scene.colors.astype(np.float32)
    rad_out[:n]     = modified_scene.radii.astype(np.float32)
    refl_out[:n]    = modified_scene.reflectivities.astype(np.float32)
    vis_out[:, :n]  = vis

    return {
        "obs_intensity":  obs_intensity.astype(np.float32),
        "obs_depth":      obs_depth.astype(np.float32),
        "obs_id":         obs_id.astype(np.int8),
        "is_visible":     vis_out,
        "positions":      pos_out,
        "velocities":     vel_out,
        "colors":         col_out,
        "radii":          rad_out,
        "reflectivities": refl_out,
        "n_objects":      np.uint8(n),
        "seed":           np.int64(cfg.seed),
        # edit metadata
        "edit_frame":     np.int32(eff_edit_frame),
        "edit_object":    np.int8(obj_idx),
        "edit_op":        np.uint8(OP_SET_POSITION),
        "edit_value":     new_pos.astype(np.float32),
        "n_edits":        np.uint8(1),
    }


# ── HDF5 helpers ──────────────────────────────────────────────────────────────


def _create_datasets(hf: h5py.File, dcfg: EditDatasetConfig, max_obj: int) -> None:
    N, F, R = dcfg.n_samples, dcfg.sim.n_frames, dcfg.sim.obs_res
    C = min(dcfg.hdf5_chunk, N)
    kw = dict(compression=dcfg.compression, compression_opts=dcfg.compression_level)

    hf.create_dataset("obs_intensity",  (N, F, R),          dtype="float32", chunks=(C, F, R),          **kw)
    hf.create_dataset("obs_depth",      (N, F, R),          dtype="float32", chunks=(C, F, R),          **kw)
    hf.create_dataset("obs_id",         (N, F, R),          dtype="int8",    chunks=(C, F, R),          **kw)
    hf.create_dataset("is_visible",     (N, F, max_obj),    dtype="bool",    chunks=(C, F, max_obj),    **kw)
    hf.create_dataset("positions",      (N, F, max_obj, 2), dtype="float32", chunks=(C, F, max_obj, 2), **kw)
    hf.create_dataset("velocities",     (N, F, max_obj, 2), dtype="float32", chunks=(C, F, max_obj, 2), **kw)
    hf.create_dataset("colors",         (N, max_obj, 3),    dtype="float32", chunks=(C, max_obj, 3),    **kw)
    hf.create_dataset("radii",          (N, max_obj),       dtype="float32", chunks=(C, max_obj),       **kw)
    hf.create_dataset("reflectivities", (N, max_obj),       dtype="float32", chunks=(C, max_obj),       **kw)
    hf.create_dataset("n_objects",      (N,),               dtype="uint8",   chunks=(min(C * F, N),),   **kw)
    hf.create_dataset("seeds",          (N,),               dtype="int64",   chunks=(min(C * F, N),),   **kw)
    # edit metadata
    hf.create_dataset("edit_frame",     (N,),    dtype="int32",   chunks=(min(C * F, N),), **kw)
    hf.create_dataset("edit_object",    (N,),    dtype="int8",    chunks=(min(C * F, N),), **kw)
    hf.create_dataset("edit_op",        (N,),    dtype="uint8",   chunks=(min(C * F, N),), **kw)
    hf.create_dataset("edit_value",     (N, 2),  dtype="float32", chunks=(min(C * F, N), 2), **kw)
    hf.create_dataset("n_edits",        (N,),    dtype="uint8",   chunks=(min(C * F, N),), **kw)


def _write_batch(hf: h5py.File, batch: list[dict], start: int) -> None:
    end = start + len(batch)
    hf["obs_intensity"][start:end]  = np.stack([s["obs_intensity"]  for s in batch])
    hf["obs_depth"][start:end]      = np.stack([s["obs_depth"]      for s in batch])
    hf["obs_id"][start:end]         = np.stack([s["obs_id"]         for s in batch])
    hf["is_visible"][start:end]     = np.stack([s["is_visible"]     for s in batch])
    hf["positions"][start:end]      = np.stack([s["positions"]      for s in batch])
    hf["velocities"][start:end]     = np.stack([s["velocities"]     for s in batch])
    hf["colors"][start:end]         = np.stack([s["colors"]         for s in batch])
    hf["radii"][start:end]          = np.stack([s["radii"]          for s in batch])
    hf["reflectivities"][start:end] = np.stack([s["reflectivities"] for s in batch])
    hf["n_objects"][start:end]      = np.array([s["n_objects"]      for s in batch], dtype=np.uint8)
    hf["seeds"][start:end]          = np.array([s["seed"]           for s in batch], dtype=np.int64)
    hf["edit_frame"][start:end]     = np.array([s["edit_frame"]     for s in batch], dtype=np.int32)
    hf["edit_object"][start:end]    = np.array([s["edit_object"]    for s in batch], dtype=np.int8)
    hf["edit_op"][start:end]        = np.array([s["edit_op"]        for s in batch], dtype=np.uint8)
    hf["edit_value"][start:end]     = np.stack([s["edit_value"]     for s in batch])
    hf["n_edits"][start:end]        = np.array([s["n_edits"]        for s in batch], dtype=np.uint8)


# ── Main entry point ──────────────────────────────────────────────────────────


def generate_edits_dataset(dcfg: EditDatasetConfig, output_dir: str | Path) -> None:
    """Generate an edits dataset and write it into ``output_dir``.

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
    json_path   = output_dir / "dataset.json"

    max_obj = (
        dcfg.sim.n_objects if dcfg.sim.n_objects is not None else dcfg.sim.n_objects_max
    )

    eff_edit_frame = (
        dcfg.edit_frame if dcfg.edit_frame >= 0 else dcfg.sim.n_frames // 2
    )

    # ── Config JSON ───────────────────────────────────────────────────────
    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": dataclasses.asdict(dcfg),
        "schema": {
            "obs_intensity":  f"float32  (N, n_frames={dcfg.sim.n_frames}, obs_res={dcfg.sim.obs_res})  — post-edit intensity; 0=background",
            "obs_depth":      "float32  (N, n_frames, obs_res)  — depth of first hit; 0=miss",
            "obs_id":         "int8     (N, n_frames, obs_res)  — object index, -1=miss",
            "is_visible":     f"bool     (N, n_frames, max_objects={max_obj})  — partial frustum overlap per object",
            "positions":      f"float32  (N, n_frames, max_objects={max_obj}, 2)  — post-edit (x, y)",
            "velocities":     "float32  (N, n_frames, max_objects, 2)  — original (vx, vy); unchanged by edit",
            "colors":         "float32  (N, max_objects, 3)  — RGB, zero-padded",
            "radii":          f"float32  (N, max_objects={max_obj})  — per-object radius, zero-padded",
            "reflectivities": f"float32  (N, max_objects={max_obj})  — per-object reflectivity, zero-padded",
            "n_objects":      "uint8    (N,)  — true object count per sample",
            "seeds":          "int64    (N,)  — RNG seed per sample",
            "edit_frame":     f"int32    (N,)  — frame where edit is applied (here: {eff_edit_frame})",
            "edit_object":    "int8     (N,)  — index of the moved object",
            "edit_op":        "uint8    (N,)  — operation code: 0 = set_position",
            "edit_value":     "float32  (N, 2)  — new (x, y) position after edit",
            "n_edits":        "uint8    (N,)  — number of edits per sample (always 1)",
        },
    }
    config_json = json.dumps(meta, indent=2)
    json_path.write_text(config_json)

    # ── Worker args ───────────────────────────────────────────────────────
    seeds = dcfg.base_seed + np.arange(dcfg.n_samples, dtype=np.int64)
    args = [
        (int(s), dcfg.sim, max_obj, dcfg.edit_frame,
         dcfg.edit_always_in_frustum, dcfg.max_edit_attempts)
        for s in seeds
    ]

    chunksize = max(1, dcfg.write_batch // max(1, dcfg.n_workers))

    pool = mp.Pool(dcfg.n_workers) if dcfg.n_workers > 0 else None
    try:
        iterator = (
            pool.imap(_generate_one_edit, args, chunksize=chunksize)
            if pool is not None
            else map(_generate_one_edit, args)
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
