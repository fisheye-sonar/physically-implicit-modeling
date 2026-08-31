"""ORACLE editor 2 — freeze-time teacher-forced object interpolation.

Ported 2026-08-31 from ``multistep_steering.ipynb`` cells [6]/[11] (the arm that worked
on discworld with oracle access). The second of the two oracle arms defending the Edit
Index (see ``oracle_overwrite``), and the gentler one: instead of overwriting the state
outright, it shows the model a SHORT RENDERED SEQUENCE in which the edited object
glides from its pre-edit position to the teleport target while time is frozen (the
unedited object holds still), teacher-forces those N frames, then free-runs. The model
absorbs the edit through its own observation channel — the only write mechanism its
training distribution ever prepared it for.

N=1 is the one-frame teleport baseline; the N-sweep is the experiment. The 2026-08
finding on the GRU: clean frozen frames trade ghost removal against dynamics; the
noise-matched variant (frames rendered WITH the training obs noise) is the fair form —
pass ``obs_noise_std`` to get it.
"""

from __future__ import annotations

import numpy as np
import torch

from pim.environments.discworld.renderer import render_scene
from pim.environments.discworld.sim import Scene
from pim.metrics.editability import sim_config_from


def frozen_frames(
    sim: dict,
    pre_pos: np.ndarray,      # (n_obj, 2) all objects' positions at the edit frame
    tgt_pos: np.ndarray,      # (n_obj, 2) all objects' post-edit positions
    edit_object: int,
    n_frames: int,
    *,
    reflectivities: np.ndarray,
    obs_noise_std: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """(n_frames, obs_res) rendered frames: the edited object lerps pre→target while
    every other object holds at its (post-edit) position; time is frozen throughout."""
    n_obj = pre_pos.shape[0]
    cfg = sim_config_from(sim, n_obj)
    cfg = type(cfg)(**{**cfg.__dict__, "n_frames": n_frames,
                       "obs_noise_std": float(obs_noise_std), "seed": seed})
    fr = np.zeros((n_frames, n_obj, 2), np.float32)
    for j in range(n_frames):
        t = (j + 1) / n_frames
        fr[j] = tgt_pos
        fr[j, edit_object] = pre_pos[edit_object] + t * (tgt_pos[edit_object]
                                                         - pre_pos[edit_object])
    scene = Scene(positions=fr,
                  velocities=np.zeros((n_frames, n_obj, 2), np.float32),
                  radii=np.full(n_obj, sim["radius"], np.float32),
                  colors=np.ones((n_obj, 3), np.float32),
                  reflectivities=np.asarray(reflectivities, np.float32),
                  config=cfg)
    _, _, rint = render_scene(scene)
    return rint.astype(np.float32)


@torch.no_grad()
def freeze_time_rollout(model, state, frames: torch.Tensor, steps: int) -> torch.Tensor:
    """Teacher-force the frozen frames into ``state``, then free-run ``steps`` frames.

    frames : (B, N, obs_res) — from ``frozen_frames``, batched by the caller.
    Returns (B, steps, obs_res). Works on any protocol model with ``advance``.
    """
    s = state
    for j in range(frames.shape[1]):
        s = model.advance(s, frames[:, j])
    pred = model.decode(s)
    out, s = [pred], model.advance(s, pred)
    for _ in range(steps - 1):
        p, s = model.predict_step(s)
        out.append(p)
    return torch.stack(out, 1)
