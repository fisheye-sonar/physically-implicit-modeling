# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PhysicallyImplicitModeling — early-stage Python research project.

## Commands

```bash
# run all tests
poetry run pytest

# run a single test file / single test
poetry run pytest tests/test_sim.py
poetry run pytest tests/test_sim.py::test_no_collisions

# demo animation (interactive)
python scripts/demo.py
python scripts/demo.py --seed 7 --n-objects 4 --frames 80
python scripts/demo.py --save outputs/demo.gif

# lint / format
poetry run ruff check pim tests
poetry run black pim tests scripts
```

## Architecture

The simulation is split into four independent layers, kept in separate modules so future implicit/explicit/hybrid models can substitute individual components:

| Layer | Module | Role |
|---|---|---|
| Latent state | `pim/sim.py` | `Scene.positions` (n_frames × n_objects × 2) — the ground truth |
| State update | `pim/sim.py` `simulate()` | Linear motion + noise + frustum-wall reflection; rejection sampling for collision avoidance |
| Observation render | `pim/renderer.py` | Analytical ray casting → 1D depth signal; no pixel rasterisation |
| Visual render | `pim/viz.py` | Matplotlib animation (2D world + waterfall); human-facing only |

**World geometry.** The scene is a 2D perspective frustum (trapezoid). The observer sits at the origin; the frustum occupies `y ∈ [y_near, y_far]`. Default config: `y_near=3, y_far=12, x_near=1.5, x_far=6`. Because `x_near/y_near == x_far/y_far == 0.50`, the frustum is a proper pinhole cone and the ray-caster's FOV naturally covers it exactly. The ~1.3:1 aspect ratio is intentional so the 2D panel looks balanced.

**1D observation.** Each frame, `obs_res` rays fan out from the observer. The first circle each ray hits contributes its y-depth to the signal. Closer objects subtend more rays (appear larger) and sweep the scan faster — both perspective effects arise from the geometry, not post-processing.

**Entry point for notebooks.** `from pim import SimConfig, simulate, render_scene, animate_scene` imports everything needed. `pim/__init__.py` has a full conceptual overview.

## Environment

- Python 3.12 virtual environment lives in `.pim/`
- [direnv](https://direnv.net/) activates it automatically via `.envrc`; if direnv is not active, run `source .pim/bin/activate` manually
- Dependencies managed with Poetry (`pyproject.toml`)
