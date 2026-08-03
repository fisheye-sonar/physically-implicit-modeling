"""Continuous-action (large affordance) simulation and dataset generation.

NEW module — does not modify existing simulator paths.  Mirrors
``pim/simulator/actions.py`` but replaces the discrete 0.7-unit nudge token with
a **continuous, large** affordance whose *type* is the independent variable.  A
single generator with ``mode ∈ {"dxdy", "teleport", "axis_x"}`` applies the
affordance as a **persistent per-object position change** from the event frame,
re-renders the (noisy) observation, and writes a continuous ``actions`` field.

Motivation (see research/directions/action-space-object-individuation.md): the
target is object *individuation* — does training on an interaction affordance
reorganize the PASSIVE latent into a separable, grabbable object HANDLE that
generalizes to interventions it was never trained on (a real object vs a trained
button)?  The action *type* is the independent variable:

- ``dxdy``     — relative displacement ``(dx, dy)``, large (uniform in ±M), forces
                 object-*tracking*.  Recorded action value = the displacement.
- ``teleport`` — absolute placement to a target ``(x', y')`` sampled in-frustum
                 (reuse ``edits_dataset._sample_in_frustum``); saturates the target
                 space; forces **ghost removal**.  Recorded value = the target.
- ``axis_x``   — relative displacement restricted to the **x-axis** (``(dx, 0)``);
                 the content-generalization probe (train x-only, test edits on y).

Action field
------------
``actions`` has shape ``(T, n_obj, 3)`` = per object ``[active, a1, a2]``:
    active ∈ {0, 1}  — 1 on the object acted at that transition (one object/event)
    a1, a2           — the recorded action value (per the mode above), **normalized**
                       to ≈[-1, 1] by the frustum extent so model inputs are O(1).
no-op = ``active = 0`` and ``a1 = a2 = 0``.  Every transition acts at most one
object; ~``p_action`` of transitions carry an action, the rest are genuine no-ops
(and rejected frustum/collision-guarded actions become no-ops that frame).

Alignment: ``actions[s]`` is the action driving the transition **into** frame
``s + 1`` (source ``s`` -> target ``s + 1``); the observation at ``s + 1`` reflects
the moved world.  ``actions[T-1] = no-op`` (no frame T to drive).  This matches a
next-step-prediction world model: at input step ``s`` (seeing ``obs[s]``) the model
is told action ``a_s`` and predicts ``obs[s + 1]``.

Normalization constants (per ``SimConfig``)
-------------------------------------------
    NX      = x_far                          (x half-extent of the frustum)
    Y_MID   = (y_near + y_far) / 2
    Y_HALF  = (y_far - y_near) / 2
Relative moves:  a1 = dx / NX,        a2 = dy / Y_HALF
Absolute (teleport): a1 = x' / NX,    a2 = (y' - Y_MID) / Y_HALF
"""

from __future__ import annotations

import dataclasses
import json
import multiprocessing as mp
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from .actions import _obj_in_frustum
from .config import SimConfig
from .edits_dataset import _sample_in_frustum
from .renderer import render_scene
from .sim import Scene, compute_visibility, simulate

MODES = ("dxdy", "teleport", "axis_x")


# ── Normalization ─────────────────────────────────────────────────────────────


def norm_consts(cfg: SimConfig) -> tuple[float, float, float]:
    """Return (NX, Y_MID, Y_HALF) used to normalize recorded action values."""
    nx = float(cfg.x_far)
    y_mid = float((cfg.y_near + cfg.y_far) / 2.0)
    y_half = float((cfg.y_far - cfg.y_near) / 2.0)
    return nx, y_mid, y_half


def normalize_action(mode: str, a1: float, a2: float, cfg: SimConfig) -> tuple[float, float]:
    """Normalize a raw recorded action value to ≈[-1, 1] by the frustum extent."""
    nx, y_mid, y_half = norm_consts(cfg)
    if mode == "teleport":
        return a1 / nx, (a2 - y_mid) / y_half
    return a1 / nx, a2 / y_half  # relative (dxdy, axis_x)


def denormalize_action(mode: str, a1n: float, a2n: float, cfg: SimConfig) -> tuple[float, float]:
    """Inverse of :func:`normalize_action` (recover the raw recorded value)."""
    nx, y_mid, y_half = norm_consts(cfg)
    if mode == "teleport":
        return a1n * nx, a2n * y_half + y_mid
    return a1n * nx, a2n * y_half


# ── Continuous-action simulation ───────────────────────────────────────────────


def simulate_with_continuous_actions(
    cfg: SimConfig,
    *,
    mode: str,
    move_scale: float = 4.0,
    p_action: float = 0.18,
    action_seed: int | None = None,
) -> tuple[Scene, np.ndarray]:
    """Generate a base scene, then apply per-frame random continuous actions.

    The base trajectory comes from ``simulate(cfg)`` (byte-identical to the
    passive dataset for the same seed).  Each transition, with probability
    ``p_action``, one random object is acted; accepted moves accumulate as a
    persistent per-object offset (teleport *replaces* the object's offset;
    dxdy / axis_x *add* to it).  Frustum + collision guarded at the effect frame.

    Parameters
    ----------
    mode        : "dxdy" | "teleport" | "axis_x"
    move_scale  : M for relative modes — dx, dy ~ Uniform(-M, +M) (world units).
                  Unused for teleport (targets come from the in-frustum sampler).

    Returns
    -------
    scene   : Scene with the *moved* positions (velocities unchanged).
    actions : (T, n_obj, 3) float32 — [active, a1, a2] per object per frame,
              normalized; actions[T-1] = 0.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    scene = simulate(cfg)
    T = cfg.n_frames
    n = scene.positions.shape[1]
    r = cfg.radius
    min_sep = cfg.collision_margin * 2.0 * r

    arng = np.random.default_rng(
        action_seed if action_seed is not None else int(cfg.seed) + 8_000_000
    )

    off = np.zeros((n, 2), dtype=np.float64)          # cumulative per-object offset
    actions = np.zeros((T, n, 3), dtype=np.float32)   # [active, a1, a2]
    new_positions = scene.positions.copy()

    for t in range(1, T):                              # target frame t; source s = t-1
        s = t - 1
        if arng.random() < p_action:
            obj = int(arng.integers(0, n))
            base_t = scene.positions[t]                # (n, 2) unperturbed base at t

            if mode == "teleport":
                target = _sample_in_frustum(arng, cfg, margin=r)  # (2,) in-frustum
                cand_obj = np.asarray(target, dtype=np.float64)
                cand_off_obj = cand_obj - base_t[obj]
                rec1, rec2 = float(target[0]), float(target[1])
            else:
                dx = float(arng.uniform(-move_scale, move_scale))
                dy = 0.0 if mode == "axis_x" else float(arng.uniform(-move_scale, move_scale))
                cand_off_obj = off[obj] + np.array([dx, dy], dtype=np.float64)
                cand_obj = base_t[obj] + cand_off_obj
                rec1, rec2 = dx, dy

            # candidate positions of all objects at frame t (only `obj` changes)
            cand_pos_t = base_t + off
            cand_pos_t[obj] = cand_obj

            ok = _obj_in_frustum(cand_pos_t[obj], r, cfg)
            if ok:
                for j in range(n):
                    if j == obj:
                        continue
                    if np.linalg.norm(cand_pos_t[obj] - cand_pos_t[j]) < min_sep:
                        ok = False
                        break
            if ok:
                off[obj] = cand_off_obj
                a1, a2 = normalize_action(mode, rec1, rec2, cfg)
                actions[s, obj] = (1.0, a1, a2)
        new_positions[t] = scene.positions[t] + off

    modified = Scene(
        positions=new_positions,
        velocities=scene.velocities,
        radii=scene.radii,
        colors=scene.colors,
        reflectivities=scene.reflectivities,
        config=cfg,
    )
    return modified, actions


# ── Dataset generation (mirrors actions.py, continuous `actions` field) ─────────


def _generate_one(args: tuple[int, SimConfig, int, str, float, float]) -> dict:
    """Generate one continuous-action sample.  Module-level for multiprocessing."""
    seed, base_cfg, max_obj, mode, move_scale, p_action = args
    cfg = dataclasses.replace(base_cfg, seed=int(seed))

    for attempt in range(10):
        try:
            if attempt:
                cfg = dataclasses.replace(cfg, seed=int(seed) + attempt * 1_000_000)
            scene, actions = simulate_with_continuous_actions(
                cfg, mode=mode, move_scale=move_scale, p_action=p_action
            )
            break
        except RuntimeError:
            if attempt == 9:
                raise

    _, _, obs_intensity = render_scene(scene)
    vis = compute_visibility(scene)
    n = scene.positions.shape[1]

    pos_out = np.zeros((cfg.n_frames, max_obj, 2), dtype=np.float32)
    vel_out = np.zeros((cfg.n_frames, max_obj, 2), dtype=np.float32)
    act_out = np.zeros((cfg.n_frames, max_obj, 3), dtype=np.float32)
    refl_out = np.zeros((max_obj,), dtype=np.float32)
    vis_out = np.zeros((cfg.n_frames, max_obj), dtype=bool)

    pos_out[:, :n] = scene.positions.astype(np.float32)
    vel_out[:, :n] = scene.velocities.astype(np.float32)
    act_out[:, :n] = actions
    refl_out[:n] = scene.reflectivities.astype(np.float32)
    vis_out[:, :n] = vis

    return {
        "obs_intensity": obs_intensity.astype(np.float32),
        "actions": act_out,
        "is_visible": vis_out,
        "positions": pos_out,
        "velocities": vel_out,
        "reflectivities": refl_out,
        "n_objects": np.uint8(n),
        "seed": np.int64(cfg.seed),
    }


def _create_datasets(hf: h5py.File, n_samples: int, cfg: SimConfig, max_obj: int,
                     chunk: int, compression: str, clevel: int) -> None:
    N, F, R = n_samples, cfg.n_frames, cfg.obs_res
    C = min(chunk, N)
    kw = dict(compression=compression, compression_opts=clevel)
    hf.create_dataset("obs_intensity", (N, F, R), dtype="float32", chunks=(C, F, R), **kw)
    hf.create_dataset("actions", (N, F, max_obj, 3), dtype="float32",
                      chunks=(C, F, max_obj, 3), **kw)
    hf.create_dataset("is_visible", (N, F, max_obj), dtype="bool", chunks=(C, F, max_obj), **kw)
    hf.create_dataset("positions", (N, F, max_obj, 2), dtype="float32", chunks=(C, F, max_obj, 2), **kw)
    hf.create_dataset("velocities", (N, F, max_obj, 2), dtype="float32", chunks=(C, F, max_obj, 2), **kw)
    hf.create_dataset("reflectivities", (N, max_obj), dtype="float32", chunks=(C, max_obj), **kw)
    hf.create_dataset("n_objects", (N,), dtype="uint8", chunks=(min(C * F, N),), **kw)
    hf.create_dataset("seeds", (N,), dtype="int64", chunks=(min(C * F, N),), **kw)


def _write_batch(hf: h5py.File, batch: list[dict], start: int) -> None:
    end = start + len(batch)
    hf["obs_intensity"][start:end] = np.stack([s["obs_intensity"] for s in batch])
    hf["actions"][start:end] = np.stack([s["actions"] for s in batch])
    hf["is_visible"][start:end] = np.stack([s["is_visible"] for s in batch])
    hf["positions"][start:end] = np.stack([s["positions"] for s in batch])
    hf["velocities"][start:end] = np.stack([s["velocities"] for s in batch])
    hf["reflectivities"][start:end] = np.stack([s["reflectivities"] for s in batch])
    hf["n_objects"][start:end] = np.array([s["n_objects"] for s in batch], dtype=np.uint8)
    hf["seeds"][start:end] = np.array([s["seed"] for s in batch], dtype=np.int64)


def generate_continuous_action_dataset(
    h5_path: str | Path,
    sim: SimConfig,
    *,
    mode: str,
    n_samples: int,
    base_seed: int = 0,
    move_scale: float = 4.0,
    p_action: float = 0.18,
    n_workers: int = 8,
    write_batch: int = 512,
    hdf5_chunk: int = 64,
    compression: str = "gzip",
    compression_level: int = 4,
) -> dict:
    """Generate a continuous-action dataset and write it to ``h5_path``.

    Stores obs_intensity (noisy) + a continuous ``actions`` (N, T, max_obj, 3)
    field + positions/velocities/visibility/reflectivities for verification.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    if h5_path.exists():
        raise FileExistsError(f"{h5_path} already exists — refusing to overwrite.")

    max_obj = sim.n_objects if sim.n_objects is not None else sim.n_objects_max
    nx, y_mid, y_half = norm_consts(sim)

    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kind": "continuous_action_augmented",
        "mode": mode,
        "n_samples": n_samples,
        "base_seed": base_seed,
        "move_scale": move_scale,
        "p_action": p_action,
        "norm_consts": {"NX": nx, "Y_MID": y_mid, "Y_HALF": y_half},
        "sim": dataclasses.asdict(sim),
        "schema": {
            "obs_intensity": f"float32 (N, {sim.n_frames}, {sim.obs_res}) — noisy moved intensity",
            "actions": f"float32 (N, {sim.n_frames}, {max_obj}, 3) — [active, a1, a2] per object; "
                       f"a1,a2 normalized; mode={mode}; action[s] drives s->s+1; actions[T-1]=0",
            "positions": f"float32 (N, {sim.n_frames}, {max_obj}, 2) — moved (x,y)",
            "velocities": f"float32 (N, {sim.n_frames}, {max_obj}, 2) — base (vx,vy), unchanged",
            "is_visible": f"bool (N, {sim.n_frames}, {max_obj})",
            "reflectivities": f"float32 (N, {max_obj})",
            "n_objects": "uint8 (N,)",
            "seeds": "int64 (N,)",
        },
    }
    config_json = json.dumps(meta, indent=2)

    seeds = base_seed + np.arange(n_samples, dtype=np.int64)
    args = [(int(s), sim, max_obj, mode, move_scale, p_action) for s in seeds]
    chunksize = max(1, write_batch // max(1, n_workers))

    pool = mp.Pool(n_workers) if n_workers > 0 else None
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
            _create_datasets(hf, n_samples, sim, max_obj, hdf5_chunk, compression, compression_level)
            t0 = time.perf_counter()
            with tqdm(total=n_samples, unit="sample", dynamic_ncols=True,
                      desc=f"generating [{mode}] → {h5_path.name}") as pbar:
                for sample in iterator:
                    batch.append(sample)
                    pbar.update(1)
                    if len(batch) >= write_batch:
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
    print(f"  [{mode}] {n_samples:,} samples | {elapsed:.1f}s "
          f"({n_samples / elapsed:.0f}/s) | {size_mb:.1f} MB → {h5_path}")
    return meta
