# Physically Implicit Modeling

A toy environment for studying implicit vs explicit world representations in sequential perception tasks.

## What it is

A 2D perspective frustum world with 1–5 circular objects moving under configurable dynamics. The "sensor" is a 1D ray-cast intensity scan — a deliberately impoverished observation that hides depth, occludes objects, and conflates position with reflectivity. The goal is to learn world models from this signal alone.

## Quick start

```bash
source .pim/bin/activate   # or let direnv do it
python scripts/demo.py
python scripts/demo.py --boundary open --n-objects 4 --waterfall-mode human
```

## Generate a dataset

```bash
python scripts/generate_dataset.py data/train --n-samples 100000 --n-workers 8
python scripts/generate_dataset.py data/small --n-samples 1000
```

Each run creates a directory with `dataset.h5` and `dataset.json`. The HDF5 contains observation sequences (`obs_intensity`, `obs_depth`, `obs_id`, `is_visible`) alongside ground-truth latent state (`positions`, `velocities`, `reflectivities`, `colors`).

## Key config knobs

| Parameter | Effect |
|---|---|
| `boundary` | `bounce` / `open` / `wrap` |
| `direction_noise_std`, `speed_noise_std`, `position_noise_std` | trajectory noise |
| `refl_min`, `refl_max`, `refl_min_sep` | per-object reflectivity range and minimum pairwise separation |
| `fixed_reflectivities` | uniformly space reflectivities between `refl_min`/`refl_max` (same order every sample) |
| `always_in_frustum` | reject trajectories where any object touches the frustum edge |
| `obs_noise_std` | additive Gaussian noise on the intensity scan (0 = clean) |

## Structure

```
pim/
  config.py       — SimConfig dataclass (all parameters)
  sim.py          — simulator, Scene, visibility
  renderer.py     — analytical ray casting
  viz.py          — dark-theme matplotlib animation (simulator aesthetic)
  dataset.py      — HDF5 dataset generation
  models/
    gru.py        — GRU world model
scripts/
  demo.py
  generate_dataset.py
  train_gru.py
notebooks/
  helpers/        — shared nb_utils.py and nb_viz.py
  gru_eval.ipynb  — model evaluation (next-step MSE, autoregressive rollout)
  probe_combined.ipynb — linear + MLP position probes
tests/
```
