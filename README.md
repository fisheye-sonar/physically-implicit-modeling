# Physically Implicit Modeling

A toy environment for studying what kinds of internal states world models form when trained on impoverished observations.

## Research question

When a system is trained on observations alone, what kind of internal state does it build? Specifically: does that state correspond to the underlying environment state, or merely to observation predictivity? Is it recoverable via probes? Is it stable over long rollouts? And is it a manipulable system — one where targeted edits to the latent produce coherent, predictable changes in behavior — or just a tangled compression of past inputs?

This project investigates those questions across architectures (GRU, RSSM) and dataset conditions using a controlled toy environment: a 2D scene with circular objects visible only through a 1D ray-cast intensity scan that conflates position with reflectivity and hides depth.

## Setup

Requires Python 3.12+ and [Poetry](https://python-poetry.org/).

```bash
python -m venv .pim
source .pim/bin/activate
pip install poetry
poetry install
```

[direnv](https://direnv.net/) will activate `.pim` automatically on entry if configured. After setup:

```bash
python scripts/demo.py
python scripts/demo.py --boundary open --n-objects 4 --waterfall-mode human
```

## Generate a dataset

```bash
python scripts/generate_dataset.py data/my_run
python scripts/generate_dataset.py data/my_run --n-train 100000 --n-workers 8
```

Each run generates all four splits into the output directory:

```
data/my_run/
  train.h5
  val.h5
  test.h5
  edits.h5
  dataset.json   ← shared sim config + split metadata
```

Each HDF5 contains observation sequences (`obs_intensity`, `obs_depth`, `obs_id`, `is_visible`) alongside ground-truth latent state (`positions`, `velocities`, `reflectivities`, `colors`). The edits split additionally records `edit_frame`, `edit_object`, and `edit_value`. Seeds are assigned non-overlappingly across splits.

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
  simulator/      — SimConfig, Scene, simulate(), renderer, viz, HDF5 dataset I/O
  world_models/
    gru/          — GRU world model + evaluation pipeline
    rssm/         — RSSM world model (Dreamer-style, det + stoch latent) + evaluation pipeline
  extractors/     — StateDefinition, LinearExtractor, MLPExtractor, matching losses
  editors/        — hidden-state steering via pseudoinverse
  eval/           — pure-function evaluation (prediction, recovery, rollout, controllability)
scripts/
  demo.py
  generate_dataset.py
  generate_edits_dataset.py
  train_gru.py / train_rssm.py
  gru_eval.py / rssm_eval.py
notebooks/
  helpers/        — shared nb_utils.py and nb_viz.py
  dataset_viz.ipynb
  gru_eval.ipynb
  rssm_eval.ipynb
tests/
```
