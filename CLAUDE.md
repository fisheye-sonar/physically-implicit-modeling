# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PhysicallyImplicitModeling — early-stage Python research project exploring implicit vs explicit world representations in a toy dynamical environment.

## Commands

```bash
# run all tests
poetry run pytest

# demo animation (interactive)
python scripts/demo.py
python scripts/demo.py --seed 7 --n-objects 4 --waterfall-mode human
python scripts/demo.py --fixed-reflectivities --always-in-frustum

# generate dataset
python scripts/generate_dataset.py data/my_run --n-samples 100000 --n-workers 8
python scripts/generate_dataset.py data/my_run --fixed-reflectivities --always-in-frustum

# train GRU
python scripts/train_gru.py

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
| Visual render | `pim/viz.py` | Dark-theme matplotlib animation (2D world + waterfall); simulator aesthetic |
| Dataset generation | `pim/dataset.py` | Multiprocessing HDF5 writer; `generate_dataset(dcfg, output_dir)` |
| World model | `pim/models/gru.py` | GRU: `encoder (Linear+ReLU) → GRU → decoder (Linear)`; `model(obs) → (pred, h)`; `model.step(obs_t, h) → (pred_t, h_next)` for single-step AR rollout |

**World geometry.** 2D perspective frustum (trapezoid). Observer at origin; frustum at `y ∈ [y_near, y_far]`. Default: `y_near=3, y_far=12, x_near=1.5, x_far=6` (FOV_tan=0.5, ~1.3:1 aspect). `x_near/y_near == x_far/y_far` so the ray-caster's FOV covers the frustum exactly.

**1D observation.** `obs_res` rays fan out each frame. Returns three arrays: `obs_depth` (y of first hit), `obs_id` (object index, -1=miss), `obs_intensity` (reflectivity of hit object + optional additive Gaussian noise, clipped to [0,1]). Objects outside `[y_near, y_far]` are invisible and do not occlude.

**Scene fields.**
- `reflectivities` — per-object scalar in `[refl_min, refl_max]`; `refl_min_sep` enforces minimum pairwise separation (default 0.15)
- `fixed_reflectivities=True` — uniformly spaced in `[refl_min, refl_max]`, same order every sample (object 0 = min, object N-1 = max); use with `USE_HUNGARIAN=False` in probes
- `always_in_frustum=True` — rejection-sample until no object circle ever touches a frustum edge
- `compute_visibility(scene)` — returns `(n_frames, n_objects) bool` for frustum overlap

**Boundary modes.** `"bounce"` (reflect off frustum walls), `"open"` (drift freely, out-of-frustum objects invisible), `"wrap"` (toroidal in bounding rectangle).

**Waterfall viz modes.** `mode="model"` — grayscale from `obs_intensity` (what the model sees). `mode="human"` — color-coded by object identity, brightness by inverse depth. Reflectivity values shown as labels inside circles in the 2D panel.

**HDF5 schema** (per sample, padded to `max_obj` on object axis):
`obs_intensity (T,R)`, `obs_depth (T,R)`, `obs_id (T,R)`, `is_visible (T,max_obj)`, `positions (T,max_obj,2)`, `velocities (T,max_obj,2)`, `colors (max_obj,3)`, `reflectivities (max_obj,)`, `n_objects`, `seeds`.

**Notebook helpers** (`notebooks/helpers/`): import as `import helpers.nb_utils as nb_utils` / `import helpers.nb_viz as nb_viz`.
- `nb_utils.load_sample(path, idx)` → `(Scene, obs_depth, obs_id, obs_intensity)`
- `nb_utils.load_model(ckpt_path, device)` → `(model, ckpt_info)`
- `nb_utils.get_hidden_states(model, obs, device)` → `h: (B, T-1, H)` teacher-forcing
- `nb_utils.autoregressive_rollout(model, obs_np, n_context, device)` → `pred: (T-n_context, R)`
- `nb_viz.style_ax(ax, dark=False)`, `nb_viz.plot_color(scene_color)` (Okabe-Ito remap)
- `nb_viz.plot_waterfall_pair(..., dark=False)`, `nb_viz.animate_3panel(..., dark=False)`

**Probe notebooks** (`probe_combined.ipynb`): trains linear and MLP position probes on GRU hidden states. Key toggles: `USE_HUNGARIAN`, `USE_AUTOREGRESSIVE`, `NUM_OBS`. Uses exact lstsq for linear probe when `USE_HUNGARIAN=False`; gradient descent otherwise.

## Visual aesthetic policy

Two distinct aesthetics are used depending on context:

**Academic / light theme** — all result figures, plots, metrics, and diagrams intended to communicate findings (e.g. loss curves, bar charts, MSE vs context, probe trajectory plots). White background, Okabe-Ito colorblind-safe palette, clean spines, minimal decoration. Use `nb_viz.style_ax(ax)` and `nb_viz.plot_color()`.

**Simulator / dark theme** — any visualization that shows the simulator itself as an artifact: the 2D scene animation, waterfall panels, and any figure where the model is "running in" the simulator environment. Dark navy background (`#0a0a14`), the sim's original object colors, retrofuturistic aesthetic. In `pim/viz.py` this is the default. In `nb_viz` functions, pass `dark=True` (e.g. `plot_waterfall_pair(..., dark=True)`, `animate_3panel(..., dark=True)`).

When in doubt: metrics and analysis → light/academic; simulator output → dark.

## Environment

- Python 3.12 virtual environment lives in `.pim/`
- [direnv](https://direnv.net/) activates it automatically via `.envrc`; if not active, run `source .pim/bin/activate`
- Dependencies managed with Poetry (`pyproject.toml`)
