"""Action-augmented (nudge) simulation and dataset generation.

NEW module — does not modify existing simulator paths.  Extends the passive
simulator with a discrete action-token channel that applies small *persistent*
position nudges to individual objects, so a world model can be trained on
trajectories where random actions have a real causal effect on the world.

Motivation (see research/directions/action-conditioned-structure.md): test
whether merely *training on* causally-effective random actions induces
causal/editable latent structure that is then measurable on the PASSIVE
(no-op) model.

Action-token space (Sevan's call: discrete tokens), for ``n`` objects
-------------------------------------------------------------------------
    token 0            : no-op (dominant; genuine zero-effect token)
    tokens 1..4        : object 0  {+x, -x, +y, -y}
    tokens 5..8        : object 1  {+x, -x, +y, -y}
    ...                : 4 tokens per object, so N_TOKENS = 1 + 4 * n

A non-no-op token applies ``nudge`` world units to that object along the
chosen axis, as a persistent offset carried forward from the frame it takes
effect.  If the nudge would break frustum containment or cause a collision at
its effect frame, it is treated as a no-op that frame (token stored as 0).

Alignment: ``actions[s]`` is the token that perturbs the transition **into**
frame ``s + 1`` (i.e. source frame ``s`` -> target frame ``s + 1``); the
observation at ``s + 1`` reflects the nudged world.  ``actions[T-1] = 0`` (no
frame T to drive).  This aligns with a next-step-prediction world model: at
input step ``s`` (seeing ``obs[s]``) the model is told ``a_s = actions[s]`` and
predicts ``obs[s + 1]``.
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

from .config import SimConfig
from .renderer import render_scene
from .sim import Scene, compute_visibility, frustum_half_width, simulate

# ── Action-token geometry ─────────────────────────────────────────────────────

# unit direction per within-object axis index k: 0=+x, 1=-x, 2=+y, 3=-y
_AXIS_DIR = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])


def n_tokens(n_objects: int) -> int:
    """Number of discrete action tokens for ``n_objects`` (no-op + 4 per obj)."""
    return 1 + 4 * n_objects


def token_meaning(token: int) -> str:
    """Human-readable label for a token id."""
    if token == 0:
        return "no-op"
    obj = (token - 1) // 4
    k = (token - 1) % 4
    return f"obj{obj}{['+x', '-x', '+y', '-y'][k]}"


def _obj_in_frustum(p: np.ndarray, r: float, cfg: SimConfig) -> bool:
    """True if a circle of radius ``r`` centred at ``p`` is fully inside the frustum."""
    x, y = float(p[0]), float(p[1])
    if not (y - r >= cfg.y_near and y + r <= cfg.y_far):
        return False
    x_lim = float(frustum_half_width(y, cfg)) - r
    return abs(x) <= x_lim


# ── Nudge-augmented simulation ────────────────────────────────────────────────


def simulate_with_actions(
    cfg: SimConfig,
    *,
    nudge: float = 0.7,
    p_action: float = 0.15,
    action_seed: int | None = None,
) -> tuple[Scene, np.ndarray]:
    """Generate a valid base scene, then apply per-frame random action nudges.

    The base trajectory comes from ``simulate(cfg)`` (byte-identical to the
    passive dataset for the same seed).  A discrete token is sampled each
    transition; accepted nudges accumulate as a persistent per-object offset.

    Returns
    -------
    scene   : Scene with the *nudged* positions (velocities unchanged — the
              object continues with the same velocity from its shifted location,
              matching the edits-dataset semantics).
    actions : (n_frames,) int  — token driving each transition; actions[T-1]=0.
    """
    scene = simulate(cfg)
    T = cfg.n_frames
    n = scene.positions.shape[1]
    r = cfg.radius
    min_sep = cfg.collision_margin * 2.0 * r

    arng = np.random.default_rng(
        action_seed if action_seed is not None else int(cfg.seed) + 7_000_000
    )

    off = np.zeros((n, 2), dtype=np.float64)          # cumulative per-object offset
    actions = np.zeros(T, dtype=np.int64)
    new_positions = scene.positions.copy()

    max_tok = 4 * n
    for t in range(1, T):                              # target frame t; source s=t-1
        s = t - 1
        tok = 0
        if arng.random() < p_action:
            tok_cand = int(arng.integers(1, max_tok + 1))
            obj = (tok_cand - 1) // 4
            k = (tok_cand - 1) % 4
            dvec = _AXIS_DIR[k] * nudge

            cand_off = off.copy()
            cand_off[obj] = off[obj] + dvec
            cand_pos_t = scene.positions[t] + cand_off  # (n, 2) with candidate offsets

            ok = _obj_in_frustum(cand_pos_t[obj], r, cfg)
            if ok:
                for j in range(n):
                    if j == obj:
                        continue
                    if np.linalg.norm(cand_pos_t[obj] - cand_pos_t[j]) < min_sep:
                        ok = False
                        break
            if ok:
                off = cand_off
                tok = tok_cand
        actions[s] = tok
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


# ── Dataset generation (mirrors simulator/dataset.py, adds `actions`) ─────────


def _generate_one_action(args: tuple[int, SimConfig, int, float, float]) -> dict:
    """Generate one nudge-augmented sample.  Module-level for multiprocessing."""
    seed, base_cfg, max_obj, nudge, p_action = args
    cfg = dataclasses.replace(base_cfg, seed=int(seed))

    for attempt in range(10):
        try:
            if attempt:
                cfg = dataclasses.replace(cfg, seed=int(seed) + attempt * 1_000_000)
            scene, actions = simulate_with_actions(
                cfg, nudge=nudge, p_action=p_action
            )
            break
        except RuntimeError:
            if attempt == 9:
                raise

    obs_depth, obs_id, obs_intensity = render_scene(scene)
    vis = compute_visibility(scene)
    n = scene.positions.shape[1]

    pos_out = np.zeros((cfg.n_frames, max_obj, 2), dtype=np.float32)
    vel_out = np.zeros((cfg.n_frames, max_obj, 2), dtype=np.float32)
    refl_out = np.zeros((max_obj,), dtype=np.float32)
    vis_out = np.zeros((cfg.n_frames, max_obj), dtype=bool)

    pos_out[:, :n] = scene.positions.astype(np.float32)
    vel_out[:, :n] = scene.velocities.astype(np.float32)
    refl_out[:n] = scene.reflectivities.astype(np.float32)
    vis_out[:, :n] = vis

    return {
        "obs_intensity": obs_intensity.astype(np.float32),
        "actions": actions.astype(np.int8),
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
    hf.create_dataset("actions", (N, F), dtype="int8", chunks=(min(C * F, N), F), **kw)
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


def generate_action_dataset(
    h5_path: str | Path,
    sim: SimConfig,
    *,
    n_samples: int,
    base_seed: int = 0,
    nudge: float = 0.7,
    p_action: float = 0.15,
    n_workers: int = 8,
    write_batch: int = 512,
    hdf5_chunk: int = 64,
    compression: str = "gzip",
    compression_level: int = 4,
) -> dict:
    """Generate a nudge-augmented dataset and write it to ``h5_path``.

    Stores the standard passive schema (minus obs_depth/obs_id, which the
    action-model training does not use) plus an ``actions`` (N, T) int8 field.
    """
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    if h5_path.exists():
        raise FileExistsError(f"{h5_path} already exists — refusing to overwrite.")

    max_obj = sim.n_objects if sim.n_objects is not None else sim.n_objects_max

    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kind": "action_augmented",
        "n_samples": n_samples,
        "base_seed": base_seed,
        "nudge": nudge,
        "p_action": p_action,
        "n_tokens": n_tokens(max_obj),
        "sim": dataclasses.asdict(sim),
        "schema": {
            "obs_intensity": f"float32 (N, {sim.n_frames}, {sim.obs_res}) — noisy nudged intensity",
            "actions": f"int8 (N, {sim.n_frames}) — token driving transition s->s+1; 0=no-op; actions[T-1]=0",
            "positions": f"float32 (N, {sim.n_frames}, {max_obj}, 2) — nudged (x,y)",
            "velocities": f"float32 (N, {sim.n_frames}, {max_obj}, 2) — base (vx,vy), unchanged by nudge",
            "is_visible": f"bool (N, {sim.n_frames}, {max_obj})",
            "reflectivities": f"float32 (N, {max_obj})",
            "n_objects": "uint8 (N,)",
            "seeds": "int64 (N,)",
        },
    }
    config_json = json.dumps(meta, indent=2)

    seeds = base_seed + np.arange(n_samples, dtype=np.int64)
    args = [(int(s), sim, max_obj, nudge, p_action) for s in seeds]
    chunksize = max(1, write_batch // max(1, n_workers))

    pool = mp.Pool(n_workers) if n_workers > 0 else None
    try:
        iterator = (
            pool.imap(_generate_one_action, args, chunksize=chunksize)
            if pool is not None
            else map(_generate_one_action, args)
        )
        written = 0
        batch: list[dict] = []
        with h5py.File(h5_path, "w") as hf:
            hf.attrs["config_json"] = config_json
            _create_datasets(hf, n_samples, sim, max_obj, hdf5_chunk, compression, compression_level)
            t0 = time.perf_counter()
            with tqdm(total=n_samples, unit="sample", dynamic_ncols=True,
                      desc=f"generating → {h5_path.name}") as pbar:
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
    print(f"  {n_samples:,} samples | {elapsed:.1f}s "
          f"({n_samples / elapsed:.0f}/s) | {size_mb:.1f} MB → {h5_path}")
    return meta
