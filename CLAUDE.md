# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PhysicallyImplicitModeling — Python research project investigating what internal states world models form when trained on observations. Core questions: does the learned state correspond to the underlying environment state or just to observation predictivity? Is it recoverable via linear/nonlinear probes? Stable over long rollouts? And is it a manipulable system — where targeted latent edits produce coherent behavioral changes — or an opaque compression? Studied across architectures (GRU, RSSM) and dataset conditions in a controlled toy environment.

## Commands

```bash
# run all tests
poetry run pytest

# demo animation (interactive)
python scripts/demo.py
python scripts/demo.py --seed 7 --n-objects 4 --waterfall-mode human
python scripts/demo.py --fixed-reflectivities --always-in-frustum

# generate dataset (all splits: train/val/test/edits)
python scripts/generate_dataset.py data/my_run
python scripts/generate_dataset.py data/my_run --n-train 100000 --n-workers 8 --fixed-reflectivities

# train GRU / RSSM
python scripts/train_gru.py
python scripts/train_rssm.py

# lint / format
poetry run ruff check pim tests
poetry run black pim tests scripts
```

## Architecture

Five independent layers organized into clearly separated packages:

| Package | Module | Role |
|---|---|---|
| `pim/simulator/` | `sim.py` | `Scene` dataclass — positions, velocities, radii, colors, reflectivities |
| `pim/simulator/` | `sim.py simulate()` | Linear motion + noise + boundary handling; rejection sampling for collision avoidance |
| `pim/simulator/` | `renderer.py` | Analytical ray casting → `(obs_depth, obs_id, obs_intensity)`; only intersections within `[y_near, y_far]` are valid |
| `pim/simulator/` | `viz.py` | Dark-theme matplotlib animation (2D world + waterfall); simulator aesthetic |
| `pim/simulator/` | `dataset.py` | Multiprocessing HDF5 writer; `generate_dataset(dcfg, h5_path)`; `load_sample(path, idx)` |
| `pim/simulator/` | `edits_dataset.py` | Edits dataset: teleports one object at `edit_frame`; `generate_edits_dataset(cfg, h5_path)` |
| `pim/world_models/` | `protocol.py` | `WorldModel` + `HiddenStateModel` typing Protocols |
| `pim/world_models/` | `dataloader.py` | `ObservationDataset`, `build_dataloaders` |
| `pim/world_models/gru/` | `model.py` | `GRUModel`: encoder→GRU→decoder; `.step()`, `.get_hidden_states()`, `.hidden_size` |
| `pim/world_models/gru/` | `run_eval.py` | `EvalConfig`, `setup()`, `run_criterion{1-4}()`, `plot_criterion{1-4}()`, `run_all()`, `save_*()` |
| `pim/world_models/rssm/` | `model.py` | `RSSMModel`: det recurrent `h_t` (GRUCell) + stoch latent `s_t`; ELBO training; `.observe_step()`, `.imagine_step()`, `.decode()` |
| `pim/world_models/rssm/` | `run_eval.py` | Same interface as GRU eval; hidden states are `cat([h_t, s_t])`; AR rollout uses pure prior |
| `pim/extractors/` | `base.py` | `StateDefinition(name, state_shape, extract_fn)` — describes any physical quantity to recover |
| `pim/extractors/` | `linear.py` | `LinearExtractor(hidden_size, state_def)` — compatible with probe pseudoinverse |
| `pim/extractors/` | `mlp.py` | `MLPExtractor(hidden_size, state_def, hidden_dim)` — nonlinear probe |
| `pim/extractors/` | `matching.py` | `identity_mse()` / `hungarian_mse()` — permutation-invariant losses for extractor training |
| `pim/extractors/` | `training.py` | `fit_lstsq()` (exact), `train_extractor()` (gradient) |
| `pim/editors/` | `probe_steering.py` | `probe_decomposition()`, `inject_state()` — hidden-state steering via pseudoinverse |
| `pim/eval/` | `_helpers.py` | `run_autoregressive()`, `run_teacher_forcing()`, `collect_rollout()` — inference only |
| `pim/eval/` | `prediction.py` | `eval_single_step()`, `eval_horizon_mse()`, `eval_mse_by_context()` |
| `pim/eval/` | `recovery.py` | `eval_recovery()` — probe accuracy on test set |
| `pim/eval/` | `rollout.py` | `eval_observation_drift()`, `eval_trajectory_coherence()`, `rollout_coherence()` |
| `pim/eval/` | `controllability.py` | `eval_controllability()` — steered vs unsteered vs GT after probe injection |
| `pim/eval/` | `plotting.py` | Academic-aesthetic figures (light/Okabe-Ito); `style_ax()`, `plot_color()`, plot helpers |

**State-centric evaluation philosophy.** Eval functions are pure functions over pre-computed numpy arrays — they never call models directly. The notebook orchestrates three stages:
1. **Inference** (`pim/eval/_helpers.py`): model + obs → `(obs_rollout, internal_states)`
2. **Extraction** (`pim/extractors/`): internal_states + extractor → `decoded_env_states`
3. **Evaluation** (`pim/eval/*.py`): state sequences → metrics

**`StateDefinition`** is intentionally general: `state_shape` can be `(n_obj, 2)` for 2D positions, `(4,)` for a global attribute, etc. `output_dim = math.prod(state_shape)`.

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

**Edits HDF5 extras:** `edit_frame`, `edit_object`, `edit_op`, `edit_value (N,2)`, `n_edits`.

**Notebook helpers** (`notebooks/helpers/`): import as `import helpers.nb_utils as nb_utils` / `import helpers.nb_viz as nb_viz`.
- `nb_utils.load_model(ckpt_path, device)` → `(model, ckpt_info)`
- `nb_utils.build_loader(h5_path, indices, keys, batch_size)` → `DataLoader`
- `pim.simulator.dataset.load_sample(path, idx)` → `(Scene, obs_depth, obs_id, obs_intensity)`
- `nb_viz.style_ax(ax, dark=False)`, `nb_viz.plot_color(scene_color)` (Okabe-Ito remap)
- `nb_viz.plot_waterfall_pair(..., dark=False)`, `nb_viz.animate_3panel(..., dark=False)`
- `pim.eval._helpers`: `run_autoregressive()`, `run_teacher_forcing()`, `collect_rollout()`

**Evaluation scripts** (`scripts/gru_eval.py`, `scripts/rssm_eval.py`): CLI wrappers — run `run_all()` then save all figures + `metrics.json` + `eval_config.json` to a timestamped output directory. Supports `--criteria 1 2` to run a subset. All logic lives in `pim/world_models/{gru,rssm}/run_eval.py`.

**Evaluation notebooks** (`gru_eval.ipynb`, `rssm_eval.ipynb`): thin shells — each cell calls one `run_eval` function. Both cover the same four criteria:
1. Setup & Quick Validation — load model, training curves, dataset waterfall
2. Criterion 1 — Predictive Quality (next-step MSE, horizon sweep, MSE by context)
3. Criterion 2 — Recovery (LinearExtractor + MLPExtractor probe, bar chart, trajectory viz)
4. Criterion 3 — Rollout Consistency (observation drift, trajectory coherence, coherence distribution)
5. Criterion 4 — Counterfactual Controllability (probe-steered edits, steered vs unsteered vs GT)

**`dataset_viz.ipynb`**: standalone dataset exploration notebook.

**Notebook editing**: always use the `NotebookEdit` tool (and `Read`/`Grep` for inspection) when working with `.ipynb` files. Do not use Bash to manipulate notebook JSON directly.

## Visual aesthetic policy

Two distinct aesthetics are used depending on context:

**Academic / light theme** — all result figures, plots, metrics, and diagrams intended to communicate findings (e.g. loss curves, bar charts, MSE vs context, probe trajectory plots). White background, Okabe-Ito colorblind-safe palette, clean spines, minimal decoration. Use `pim.eval.plotting` functions or `nb_viz.style_ax(ax)` (light mode, no `dark=True`).

**Simulator / dark theme** — any visualization that shows the simulator itself as an artifact: the 2D scene animation, waterfall panels, and any figure where the model is "running in" the simulator environment. Dark navy background (`#0a0a14`), the sim's original object colors, retrofuturistic aesthetic. In `pim/simulator/viz.py` this is the default. In `nb_viz` functions, pass `dark=True` (e.g. `plot_waterfall_pair(..., dark=True)`, `animate_3panel(..., dark=True)`).

When in doubt: metrics and analysis → light/academic; simulator output → dark.

## Environment

- Python 3.12 virtual environment lives in `.pim/`
- [direnv](https://direnv.net/) activates it automatically via `.envrc`; if not active, run `source .pim/bin/activate`
- Dependencies managed with Poetry (`pyproject.toml`)
