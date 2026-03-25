# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PhysicallyImplicitModeling — early-stage Python research project exploring implicit vs explicit world representations in a toy dynamical environment.

## Commands

```bash
# run all tests
poetry run pytest

# run a single test file / single test
poetry run pytest tests/test_sim.py
poetry run pytest tests/test_sim.py::test_no_collisions

# demo animation (interactive)
python scripts/demo.py
python scripts/demo.py --seed 7 --n-objects 4 --frames 80 --waterfall-mode human
python scripts/demo.py --save outputs/demo.gif

# generate dataset
python scripts/generate_dataset.py data/my_run          # creates data/my_run/dataset.h5 + .json
python scripts/generate_dataset.py data/my_run --n-samples 100000 --n-workers 8

# lint / format
poetry run ruff check pim tests
poetry run black pim tests scripts
```

## Architecture

Four independent layers so future implicit/explicit/hybrid models can substitute individual components:

| Layer | Module | Role |
|---|---|---|
| Latent state | `pim/sim.py` | `Scene` dataclass — positions, velocities, radii, colors, reflectivities |
| State update | `pim/sim.py` `simulate()` | Linear motion + noise + boundary handling; rejection sampling for collision avoidance |
| Observation render | `pim/renderer.py` | Analytical ray casting → `(obs_depth, obs_id, obs_intensity)`; only intersections within `[y_near, y_far]` are valid |
| Visual render | `pim/viz.py` | Matplotlib animation (2D world + waterfall); human-facing only |
| Dataset generation | `pim/dataset.py` | Multiprocessing HDF5 writer; `generate_dataset(dcfg, output_dir)` |

**World geometry.** 2D perspective frustum (trapezoid). Observer at origin; frustum at `y ∈ [y_near, y_far]`. Default: `y_near=3, y_far=12, x_near=1.5, x_far=6` (FOV_tan=0.5, ~1.3:1 aspect). `x_near/y_near == x_far/y_far` so the ray-caster's FOV covers the frustum exactly.

**1D observation.** `obs_res` rays fan out each frame. Returns three arrays: `obs_depth` (y of first hit), `obs_id` (object index, -1=miss), `obs_intensity` (reflectivity of hit object + optional additive Gaussian noise, clipped to [0,1]). Objects outside `[y_near, y_far]` are invisible and do not occlude.

**Scene fields.**
- `reflectivities` — per-object scalar in `[refl_min, refl_max]`; `refl_min_sep` enforces minimum pairwise separation (default 0.15)
- `compute_visibility(scene)` — returns `(n_frames, n_objects) bool` for frustum overlap

**Boundary modes.** `"bounce"` (reflect off frustum walls), `"open"` (drift freely, out-of-frustum objects invisible), `"wrap"` (toroidal in bounding rectangle).

**Waterfall viz modes.** `mode="model"` — grayscale from `obs_intensity` (what the model sees). `mode="human"` — color-coded by object identity, brightness by inverse depth. Reflectivity values shown as labels inside circles in the 2D panel.

**HDF5 schema** (per sample, padded to `max_obj` on object axis):
`obs_intensity (T,R)`, `obs_depth (T,R)`, `obs_id (T,R)`, `is_visible (T,max_obj)`, `positions (T,max_obj,2)`, `velocities (T,max_obj,2)`, `colors (max_obj,3)`, `reflectivities (max_obj,)`, `n_objects`, `seeds`.

**Notebook imports.** `from pim import SimConfig, simulate, render_scene, animate_scene`. Load samples with the `load_sample` helper in notebooks (reconstruct `Scene` including `reflectivities`, return 4-tuple `scene, obs_depth, obs_id, obs_intensity`).

## Environment

- Python 3.12 virtual environment lives in `.pim/`
- [direnv](https://direnv.net/) activates it automatically via `.envrc`; if not active, run `source .pim/bin/activate`
- Dependencies managed with Poetry (`pyproject.toml`)
