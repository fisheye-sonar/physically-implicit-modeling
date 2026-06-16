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
| `pim/world_models/` | `protocol.py` | `WorldModel` + `HiddenStateModel` Protocols. HSM exposes `flat_state` / `state_from_flat` / `decode` / `observe_sequence` / `predict_step` so all eval is model-agnostic |
| `pim/world_models/` | `dataloader.py` | `ObservationDataset`, `build_dataloaders` |
| `pim/world_models/` | `loader.py` | `load_checkpoint(path)` auto-detects GRU vs RSSM; `load_dataset(data_dir)` reads test.h5 + edits.h5 + dataset.json |
| `pim/world_models/gru/` | `model.py` | `GRUModel`: encoder→GRU→decoder; implements full HiddenStateModel protocol |
| `pim/world_models/rssm/` | `model.py` | `RSSMModel`: deterministic `h` (GRUCell) + stochastic `s`; ELBO training; flat state = `cat([h, s])` |
| `pim/extractors/` | `base.py` | `StateDefinition(name, state_shape, extract_fn)` |
| `pim/extractors/` | `linear.py` | `LinearExtractor` — has `.fit()` (defaults to lstsq); compatible with probe pseudoinverse |
| `pim/extractors/` | `mlp.py` | `MLPExtractor` — has `.fit()` (gradient descent); not compatible with pseudoinverse |
| `pim/extractors/` | `spec.py` | `ProbeSpec` — minimal `(name, probe, marker, color_idx, linestyle)` wrapper for plot/eval pipelines |
| `pim/extractors/` | `matching.py` | `identity_mse` / `hungarian_mse` |
| `pim/extractors/` | `training.py` | Low-level `fit_lstsq` / `train_extractor`; usually called via extractor `.fit()` |
| `pim/editors/` | `probe_steering.py` | `probe_decomposition`, `inject_state` |
| `pim/eval/` | `_helpers.py` | `teacher_force`, `autoregressive_rollout(s)`, `collect_rollouts`, `decode_states_multi` — only module that touches models |
| `pim/eval/` | `baselines.py` | `compute_obs_baselines`, `compute_pos_baselines` |
| `pim/eval/` | `prediction.py` | `eval_single_step`, `eval_horizon_mse`, `eval_mse_by_context` |
| `pim/eval/` | `recovery.py` | `fit_probes(probes, …)`, `eval_recovery_multi(probes, …)` — both take a `list[ProbeSpec]` |
| `pim/eval/` | `rollout.py` | `eval_observation_drift`, `eval_position_drift`, `eval_trajectory_coherence`, `per_sample_coherence` |
| `pim/eval/` | `controllability.py` | `warm_up_to_edit`, `rollout_steered`, `rollout_unsteered`, `eval_controllability`, `eval_position_controllability` |
| `pim/figures/` | `theme.py` | `PALETTE`, `style_ax`, `style_ax_dark`, `plot_color` |
| `pim/figures/` | `setup.py` | `plot_training_curves`, `plot_dataset_overview` |
| `pim/figures/` | `prediction.py` | `plot_mse_by_context`, `plot_horizon_rmse` |
| `pim/figures/` | `recovery.py` | `plot_recovery_bars`, `plot_recovery_by_context`, `plot_recovery_trajectory` |
| `pim/figures/` | `rollout.py` | `plot_observation_drift`, `plot_position_drift`, `plot_coherence_bar/distribution`, `plot_rollout_trajectory/3panel` |
| `pim/figures/` | `controllability.py` | `plot_controllability_{obs,positions,trajectory,waterfalls}` |

**Strict separation of eval vs. figures.** `pim/eval/*` returns metrics arrays/scalars only — no matplotlib. `pim/figures/*` returns `matplotlib.Figure` objects, taking pre-computed arrays in — no model calls, no metric computation. The notebook/script is the orchestrator that wires them together explicitly. To change *how* a metric is computed, edit `pim/eval/`; to change *how* it's drawn, edit `pim/figures/`.

**Probe lists.** `ProbeSpec` lets any number of probes flow through `fit_probes` → `eval_recovery_multi` → figure builders uniformly. Add a new probe type by instantiating `ProbeSpec(name=..., probe=YourExtractor(...), marker=..., color_idx=..., linestyle=...)`. Probe hyperparameters live in the extractor constructor (notebook-controlled), never in `EvalConfig`.

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

**Notebook helpers** (`notebooks/helpers/nb_viz.py`): animation builders only (`animate_3panel`, `animate_gt_vs_predicted`, `animate_ar_gt_vs_predicted`, `plot_waterfall_pair`). Loading and inference now live in `pim.world_models.load_checkpoint` / `pim.world_models.load_dataset` and `pim.eval._helpers`.

**Evaluation CLI** (`scripts/run_eval.py`): single script that auto-detects model type from the checkpoint. Saves all figures + `metrics.json` + `eval_config.json` to a timestamped output directory. Mirrors the notebook pipeline step-by-step.

**Evaluation notebooks** (`gru_eval.ipynb`, `rssm_eval.ipynb`): explicit top-to-bottom pipelines. Each cell is one operation (load, teacher-force, fit probes, …) and passes named artifacts (`states_tf`, `probes`, `decoded_roll`, `steered`, `metrics_recovery`, …) to the next cell. Cover four eval sections:
1. Setup — `load_checkpoint`, `load_dataset`, baselines, training curves
2. Predictive Quality — `teacher_force` + `autoregressive_rollouts` + prediction metrics
3. Recovery — `fit_probes(probes, …)` + `eval_recovery_multi(probes, …)` (list of `ProbeSpec`)
4. Rollout Consistency — `collect_rollouts` + `decode_states_multi(probes, …)` + drift/coherence metrics
5. Counterfactual Controllability — `warm_up_to_edit` → `rollout_steered/unsteered` → `eval_controllability` / `eval_position_controllability`

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
