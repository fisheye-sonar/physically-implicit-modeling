"""
physically-implicit-modeling  (pim)
=====================================
Toy simulation for studying implicit vs explicit world representations.

Toy world
---------
The world is a 2D trapezoid (frustum cross-section).  Circles move through it
with approximately linear trajectories.  The observer sits at the origin,
outside the frustum, and only receives the 1D observation signal — never the
2D state directly.

Because the frustum is a proper perspective cone (x_max(y) = tan(half_fov)·y),
the near plane is proportionally narrower and the far plane is wider.

2D state
--------
``sim.Scene.positions`` — shape (n_frames, n_objects, 2) — is the explicit
latent state.  It represents everything a perfect world model would need to
track.  The 2D visualisation in ``viz`` renders this state for human inspection
only; it is not part of the observation pipeline.

1D observation
--------------
At each frame, ``renderer.render_frame`` casts rays analytically from the
observer through the scene and returns the depth of the first circle hit by
each ray.  Closer objects subtend more rays (appear larger) and move faster
across the scan (appear to move faster) — both depth-perspective effects arise
naturally from the geometry.  Optional Gaussian noise can be added.

Explicit interface points  (for future model comparisons)
---------------------------------------------------------
- Latent state       →  ``sim.Scene.positions``
- State update       →  ``sim.simulate`` / the inner stepping loop in sim.py
- Observation render →  ``renderer.render_frame`` / ``renderer.render_scene``
- Visual render      →  ``viz.animate_scene``  (human-facing only)
"""

from pim.config import SimConfig
from pim.sim import Scene, simulate
from pim.renderer import render_frame, render_scene
from pim.viz import animate_scene, save_animation
from pim.dataset import DatasetConfig, generate_dataset

__all__ = [
    "SimConfig",
    "Scene",
    "simulate",
    "render_frame",
    "render_scene",
    "animate_scene",
    "save_animation",
    "DatasetConfig",
    "generate_dataset",
]
