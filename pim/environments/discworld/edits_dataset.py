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
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import SimConfig, obs_dim
from .dataset import common_layout, generate_h5, pack_sample, simulate_with_retry
from .sim import Scene, frustum_half_width, fully_in_frustum

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
    scene, cfg = simulate_with_retry(base_cfg, seed)    # the shared seed law

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

            # Optional frustum containment check — the simulator's own rule
            if edit_always_in_frustum and not fully_in_frustum(
                edited_pos[None, None, :], cfg.radius, cfg
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

    # ── Render + pack (the shared packer), then the edit metadata ─────────
    return {
        **pack_sample(modified_scene, cfg, max_obj),
        "edit_frame":     np.int32(eff_edit_frame),
        "edit_object":    np.int8(obj_idx),
        "edit_op":        np.uint8(OP_SET_POSITION),
        "edit_value":     new_pos.astype(np.float32),
        "n_edits":        np.uint8(1),
    }


EDIT_LAYOUT = [                       # the five extra fields, after the common ones
    ("edit_frame", (), "int32", "scalar"),
    ("edit_object", (), "int8", "scalar"),
    ("edit_op", (), "uint8", "scalar"),
    ("edit_value", (2,), "float32", "scalar"),
    ("n_edits", (), "uint8", "scalar"),
]


# ── Main entry point ──────────────────────────────────────────────────────────


def generate_edits_dataset(dcfg: EditDatasetConfig, h5_path: str | Path) -> dict:
    """Generate an edits dataset and write it to ``h5_path``.

    The parent directory is created if it does not exist.  If the file already
    exists the function raises ``FileExistsError``.

    Parameters
    ----------
    dcfg    : EditDatasetConfig
    h5_path : path to the ``.h5`` file to create

    Returns
    -------
    meta : dict — the metadata written into the HDF5 attrs
    """
    h5_path = Path(h5_path)
    if h5_path.exists():
        raise FileExistsError(f"{h5_path} already exists — refusing to overwrite.")

    from pim.environments.discworld.render2d import validate

    validate(dcfg.sim)

    max_obj = (
        dcfg.sim.n_objects if dcfg.sim.n_objects is not None else dcfg.sim.n_objects_max
    )
    R = obs_dim(dcfg.sim)
    eff_edit_frame = (
        dcfg.edit_frame if dcfg.edit_frame >= 0 else dcfg.sim.n_frames // 2
    )

    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": dataclasses.asdict(dcfg),
        "schema": {
            "obs_intensity":  f"float32  (N, n_frames={dcfg.sim.n_frames}, R={R})  — post-edit noisy intensity; 0=background",
            "obs_depth":      f"float32  (N, n_frames, R={R})  — depth of first hit; 0=miss",
            "obs_id":         f"int8     (N, n_frames, R={R})  — object index, -1=miss",
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
            "_clean_obs_note": "Clean (noiseless) obs can be reconstructed via reconstruct_clean_obs(obs_id, reflectivities).",
        },
    }
    seeds = dcfg.base_seed + np.arange(dcfg.n_samples, dtype=np.int64)
    generate_h5(h5_path, n_samples=dcfg.n_samples, n_workers=dcfg.n_workers,
                write_batch_size=dcfg.write_batch, config_json=json.dumps(meta, indent=2),
                layout=common_layout(dcfg.sim, max_obj) + EDIT_LAYOUT,
                n_frames=dcfg.sim.n_frames, chunk=dcfg.hdf5_chunk,
                compression=dcfg.compression, compression_level=dcfg.compression_level,
                worker=_generate_one_edit,
                args=[(int(s), dcfg.sim, max_obj, dcfg.edit_frame,
                       dcfg.edit_always_in_frustum, dcfg.max_edit_attempts) for s in seeds])
    return meta
