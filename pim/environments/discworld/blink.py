"""Blink: object blackouts with a one-ray warning marker (the dw-blink instance, 2026-09-07).

Why. In every other discworld instance the current observation shows every object, so a
model can predict the next frame from the current frame alone and never has to CARRY a
position through time — the causal state of the task need not contain "where object j
is" as a remembered quantity. A blackout removes an object from the observation for a
run of frames while its physics continues unchanged; to predict its reappearance frame
the model MUST keep and advance its position. That makes position a guaranteed member
of the causal state, which is the pre-condition the editability question needs.

The schedule (``blink_schedule``) is a pure function of the sim config (seed included),
so the edits split — which re-renders the SAME seed with one object teleported — hides
exactly the same frames as the unedited world. Physics never sees the schedule.

Markers. So the model is told when a blackout starts and ends, the frame BEFORE a
blackout and the LAST frame of a blackout carry a ``MARK_VALUE`` (0.5) on one edge ray:
ray 0 for object 0, the last ray for object 1 (overriding whatever disc the ray hit).
Both markers are the same "toggle next frame" signal. In ``obs_id`` a marker ray is
stored as ``marker_id(j) = -2 - j`` so ``reconstruct_clean_obs`` recovers it exactly.

Rules, in ``blink_schedule``:
  * no blackout can begin before frame ``blink_warmup`` (3: frames 0-2 always visible),
    so the pre-marker of the earliest blackout sits on frame 2;
  * a blackout begins on a visible frame with probability ``blink_prob`` per object per
    frame, lasts ``min(blink_max, Geometric(1/blink_mean))`` frames (mean 6, cap 12), and
    may run to the end of the sequence (then there is no end marker);
  * at most ONE object is hidden at a time, and two blackouts of the same object are
    separated by at least one visible frame, so every marker is unambiguous.
"""
from __future__ import annotations

import numpy as np

from .config import SimConfig

MARK_VALUE = 0.5


def blink_enabled(cfg: SimConfig) -> bool:
    return float(getattr(cfg, "blink_prob", 0.0)) > 0.0


def marker_id(j: int) -> int:
    """The ``obs_id`` code of object ``j``'s marker ray (-2 for object 0, -3 for object 1)."""
    return -2 - int(j)


def marker_ray(j: int, n_rays: int) -> int:
    """Object 0 signals on the leftmost ray, object 1 on the rightmost."""
    return 0 if j == 0 else n_rays - 1


def blink_schedule(cfg: SimConfig, n_obj: int) -> np.ndarray | None:
    """(n_frames, n_obj) bool visibility, or None when blinking is off.

    Deterministic in ``cfg.seed`` (stream offset 7: the simulator uses ``seed`` and the
    renderer's noise ``seed + 1``).
    """
    if not blink_enabled(cfg):
        return None
    if n_obj > 2:
        raise ValueError("blink markers are defined for at most 2 objects (one edge ray each)")
    rng = np.random.default_rng(int(cfg.seed) + 7)
    F = int(cfg.n_frames)
    vis = np.ones((F, n_obj), dtype=bool)
    remaining = np.zeros(n_obj, dtype=int)          # hidden frames still to serve
    p, mean, cap, warm = (float(cfg.blink_prob), float(cfg.blink_mean),
                          int(cfg.blink_max), int(cfg.blink_warmup))
    for t in range(F):
        for j in range(n_obj):                       # first serve the running blackouts
            if remaining[j] > 0:
                vis[t, j] = False
                remaining[j] -= 1
        for j in range(n_obj):                       # then consider new ones
            if t < max(warm, 1) or not vis[t - 1, j]:   # warm-up, or just reappeared
                continue
            if not vis[t].all():                     # some object is hidden this frame
                continue
            if rng.random() < p:
                length = min(cap, int(rng.geometric(1.0 / mean)))
                vis[t, j] = False
                remaining[j] = length - 1
    return vis


def paint_markers(hit_id: np.ndarray, obs_intensity: np.ndarray,
                  vis_now: np.ndarray, vis_next: np.ndarray | None) -> None:
    """In place: put the toggle marker on object ``j``'s edge ray when its visibility
    changes between this frame and the next (no next frame => no marker)."""
    if vis_next is None:
        return
    R = obs_intensity.shape[0]
    for j in range(vis_now.shape[0]):
        if bool(vis_now[j]) != bool(vis_next[j]):
            r = marker_ray(j, R)
            obs_intensity[r] = MARK_VALUE
            hit_id[r] = marker_id(j)


def transitions(visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """From a (..., F, n_obj) visibility array: ``pre[..., t, j]`` = frame t is the last
    visible frame before a blackout of j; ``post[..., t, j]`` = frame t is the last hidden
    frame before j reappears. Both are False on the final frame."""
    v = np.asarray(visible, bool)
    pre = np.zeros_like(v)
    post = np.zeros_like(v)
    pre[..., :-1, :] = v[..., :-1, :] & ~v[..., 1:, :]
    post[..., :-1, :] = ~v[..., :-1, :] & v[..., 1:, :]
    return pre, post
