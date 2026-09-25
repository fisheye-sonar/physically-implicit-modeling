# Worker report: core

Owned: RELEASE `pim/models/**`, `pim/probes/**`, `pim/editors/**`, `pim/metrics/**`, `pim/training/**`, `pim/__init__.py`.
Result: pruned to scope, every docstring and comment rewritten to the SPEC rules, and equivalence against
PRIVATE verified bitwise on CPU. The five packages went from 5,355 to 2,748 lines.

Helper scripts, raw outputs and comparison logs are in `experiments/release/work/core/`.

## Deleted

| what | why |
|---|---|
| `models/transformer_s.py`, `models/recurrent.py`, `models/blocks.py` | Transformer-S / Recurrent-L (OUT) |
| `editors/nanda.py`, `editors/nullspace.py`, `editors/oracle_overwrite.py`, `editors/freeze_interpolation.py` | ND, nullspace and oracle editors (OUT) |
| `probes/nullspace.py` | nullspace cascade (OUT) |
| registry: `_build_s`, `_build_s_tokens`, `_build_recurrent`, their `BUILDERS` keys, legacy rules 1–3, the `transformer_s_tokens` vocab branch, `load_run` (unused) | S/R only. **Rule 4 (bare minGPT, no `arch` key) is kept**: the Othello standard checkpoint needs it |
| protocol: `WorldModel` (unused Protocol class) and the S/R documentation | `n_points` and `free_run` kept |
| transformer_l: `output_kind="raw"`, mixed input/head combinations, `INPUTS`/`HEADS`/`OUTPUT_KINDS`, `_win_mask` (unused) | raw head is OUT; no shipped config uses a mixed pair |
| pinv: the `"raw"` and `"legacy"` spaces | OUT; zspace numerics unchanged (it always recomputed `pinv` on its own A) |
| grad_steer: the `optimizer="gd"` branch | unused by any consumer; `optimizer` kwarg kept (Othello arms pass it), "adam" only |
| editors/inverse: `inverse_delta` | unused, non-default IM variant |
| metrics: `prediction.top1_accuracy`, `prediction.excess`, `set_editability.move_fidelity_ratio_per_case`, `zone_editability.object_constants`, `SCORECARD_COLUMNS`, `random_samples` | no in-scope consumer uses them (grepped all listed consumers in PRIVATE; also absent from RELEASE) |
| zone_editability `sim_config_from`: the three `omni2d*` kwargs | the SimConfig defaults (False/48/64) equal the values stored in all 8 shipped Rayworld datasets; SimConfig repr verified identical |
| training: `mse_next_move_onehot`, the `mse_onehot` branch of `token_source`, `othello_source` (unused wrapper) | OUT / unused; also removes training's import of the Othello data module |
| all `__pycache__` in owned dirs | they embed absolute paths; other workers' processes recreate them, so infra's `.gitignore` must exclude them |

## Kept, and why

- `probes/cache.py`: code byte-for-byte except docstrings, comments and the error/INDEX.md message strings (AST identical with strings blanked). `fingerprint`, key hashing, `VERSION = 2`, `span = getattr(model, "state_span", -1)`, provenance handling unchanged.
- `probes/base.py`, `linear.py`, `mlp.py`, `baselines.py`, `inverse.py`: code unchanged (AST identical, docstrings aside). `WorldStateProbe` stays at `pim.probes.base` with the same attributes: every cached probe pickles it. `CausalHistory` keeps both alignments (the scorer's floors use left and right). `MemmapRows` and `fit_probe_stream` are kept (categorical probes, categorical IM).
- Transformer-L parameter names, registration order and init are identical (fingerprints of all 43 shipped checkpoints and of the random-init models match); `_seq_mask` kept (`collect_residuals` calls it); `state_span` returns `block_size` as before. The attributes `input`, `head`, `output_kind` ("logits" for a token model, None for a frame model) are kept, because Othello arms read `getattr(model, "output_kind", "logits")` into `scores.json`.
- metrics: every in-scope function is code-identical (`edit_index`, zone/set editability, fidelity, `selection` with the current highest-fidelity fallback, `replicates`, `decodability` including `probe_skill_classification` / `trivial_error_rate`, which are part of the Probe Skill definition, and the prediction/floor helpers).
- training: `TrainConfig` fields and defaults are unchanged (including `lr_schedule` with its cosine branch, because the field is serialized into every config.json). `train()` and `rayworld_source` are code-identical.

## API changes (in-scope signatures)

1. `TransformerL.__init__(obs_res=None, block_size=39, n_layer=8, n_head=8, n_embd=512, dropout=0.1, *, vocab=None)`:
   - the `input`, `head` and `output_kind` kwargs are removed;
   - exactly one of `obs_res` / `vocab` must be given (raises otherwise).
2. `TransformerLTokens.__init__`: the `output_kind` kwarg is removed.
3. Registry builders now ignore `input` / `head` / `output_kind` config keys. No shipped config has them.
4. `pinv_maps(probe)` returns `{"zspace": PinvMap}` only. `pinv_step(..., space="zspace")` raises `KeyError` for any other space. The signature is unchanged.
5. `make_intervention_hook(..., optimizer="adam")`: any other value raises `ValueError`. The signature is unchanged.
6. `token_source(..., objective="ce")`: any other value raises `ValueError`. The signature is unchanged, and `"objective": "ce"` is still recorded in the meta.
7. Removed exports:
   - `pim.models`: `WorldModel`, `load_run`, `ModelConfig`, `TransformerS`, `TransformerSTokens`, `TransformerState`, `RecurrentConfig`, `RecurrentL`;
   - `pim.probes`: `NullspaceCascade`, `fit_nullspace_cascade`;
   - `pim.editors`: `probe_direction`, `addition_delta`, `addition_hook`, `inverse_delta`, `multiprobe_delta`, `counterfactual_state`, `overwrite_rollout`, `frozen_frames`, `freeze_time_rollout`;
   - `pim.metrics`: `SCORECARD_COLUMNS`, `object_constants`, `random_samples`;
   - `pim.training`: `othello_source`.
   - Grepping RELEASE shows no remaining importer of any of these.

## Requests to other owners

- **scripts, `scripts/train.py`:**
  - drop the `transformer_s` / `recurrent_l` arch choices and their `_model_config` branches (they now raise `KeyError` from `build`);
  - drop `--objective mse_onehot` and the `mc["output_kind"] = "raw"` line (`token_source` raises on it; the registry would ignore the key);
  - the `(2026-09-02)` comment and the docstring lines about `mse_onehot` go with them.
- **tables, `pim/figures/tables.py`:** `ARCH_LABEL` still lists `transformer_s`, `recurrent_l` and `transformer_s_tokens`, and there is a recurrent branch at about lines 561–570. These architectures no longer exist. Nothing breaks, but it is dead code.
- **scoring, `pim/scoring/baselines.py:47`:** the comment mentions `transformer_s`. Rewrite it to Transformer-L only.
- **export:**
  1. `runs/initial_othello_comparison/L-oth-20m/scores.json` is missing from PRIVATE: the directory mtime is 22:14:27 today, and the audits read the file earlier in the day. I used your recovered copy `work/export/recovered/L-oth-20m.scores.json` (read-only) for the selection check.
  2. Do not ship the 112 probe-cache files listed in `work/core/dead_probe_files.tsv`:
     - 18 files in `L-dw-8ray-20m/probes` pickle `pim.probes.nullspace.NullspaceCascade`. They cannot be unpickled in the release, and `ProbeCache.write_index` would list them as ERROR rows.
     - 84 files in the replicate dirs adjacent-flip `__seed1` (30), standard `__seed1` (30) and standard `__seed2` (24) are keyed on a model fingerprint that is not the shipped checkpoint (the pre-extension 390k weights).
     - 10 baseline files (`dw-noiseless` 4, `dw-8ray` 6) are keyed on random-init models of unshipped architectures.
     - No shipped code path can hit any of them.

## Verification (all CPU: `CUDA_VISIBLE_DEVICES=""`, 4 threads; PRIVATE and RELEASE in separate processes)

Each check is `verify_core.py <out> <sections>` run in both trees; `compare.py` then compares the saved arrays with `np.array_equal`, requiring equal dtype and shape, and the JSON exactly. The discworld→rayworld renames are normalized.

| check | result |
|---|---|
| **a. probe fitting, synthetic data**: `fit_linear` regression (lstsq) and classification; `fit_mlp` regression and classification; `check_probe_sanity`; `fit_inverse_map`; `fit_inverse_map_stream` (`CategoricalState` → on-disk `MemmapRows`); `fit_probe_stream` on `CausalHistory` (dense, left and right, closed form and Adam; one-hot classification with a row mask) and on `MemmapRows`; `RetrievalBank` (euclidean and onehot) `mean` and `r2`; `encode_categorical_state` | **IDENTICAL**: 89 arrays (all probe state dicts), 15 stats/report dicts |
| **b. editors on real models**, 8-ray checkpoint `L-dw-8ray-20m` (read-only), 200 real eval sequences: `collect_residuals` (all 9 points); linear, MLP and inverse map fitted on the spot at point 4; PI `pinv_step` full and `dims=[0,1]` plus `readout_error`; PI, IM and IM-NN `rollout_with_edit` (5 steps); GS `build_edit_spec` + `make_intervention_hook` → `decode` plus its record; `pinv_maps`; `flat_state` | **IDENTICAL** (23.4 M elements with c). PI lands: readout error 4e-6 / 2e-6 |
| b (cont.), Othello standard checkpoint (legacy rule-4 load), 150 real eval games: residuals; classification linear and MLP probes; PI with a `swap_class_logits` target → edited logits; two-point GS classification hook → logits plus its record | **IDENTICAL** |
| **c. metrics on random arrays**: `edit_index_per_case`, `masked_rmse_per_case`, `fidelity_ratio_from`, `fidelity`, `case_stats`, `ratio_ci95` (with and without root); zone `edit_scorecard`, `fidelity_ratio`, `fidelity_ci95`, `edit_index`, `zone_rmse`, `edit_index_by_step`; `build_edit_zones` with the real 8-ray sim config and renderer; `sim_config_from`; set `li_error`, `edit_index_legal` (union and symdiff), `uniform_over_legal`, `move_scorecard`, `move_rmse`, `move_fidelity_ratio`, `move_fidelity_ci95`; every decodability function; every prediction/floor helper; `pool_replicates` (matched budgets and `pool_budgets=True`), `t975`, `ci95_halfwidth` | **IDENTICAL**. `build_edit_zones` first failed in RELEASE on the rayworld-env worker's mid-edit import (`load_sample`); it was retried after their fix and passed. `SimConfig` reprs are equal |
| c (cont.), **selection on real arms**: Othello standard (recovered copy; keys `edit_index` and `edit_index_symdiff`), 8-ray `cartesian`, 8-ray `appearance-fac`. For each of PI, GS, IM, IM-NN and ND: `best_arm` (guarded and `guard=None`), `best_arm_by_fidelity`, `best_arm` per residual point, `arms_of` counts, `best_point` on the probe skills | **IDENTICAL** (this covers the fallback rule, e.g. Othello IM-NN has `within_guard` False) |
| **d. model loading**: the 13 in-scope checkpoints and all 30 shipped replicates (`__seed0_s512000`, `__seed1`, `__seed2` of the ten main runs), read-only | **IDENTICAL**: arch, `fingerprint`, state-dict key order, `state_span` (59 Othello, 39 Rayworld), `val_loss`, and a `ProbeCache.key` filename for a fixed provenance. On the 13, `forward`, `decode` and `residual_stack` on a fixed input are bitwise equal, and so are the `random_init_model(arch, config, seed=0)` fingerprint, span and cache key |
| d (extra), **on-disk probe cache, RELEASE code only**: for every `probes_*.pt` in the 43 shipped run dirs and the 12 in-scope `_baselines` dirs, rebuild the key from the stored provenance with the RELEASE-loaded model (or random-init model, or `None`), then `ProbeCache.load` it | **1,245 / 1,245** run files and **350 / 350** baseline files (266 observation, 84 random-init) regenerate their exact filename and load. The remainder are the dead files above |
| **e. training**: 30 steps of `pim.training.train` on CPU with seed 0 for a tiny Transformer-L on a synthetic frame memmap (`rayworld_source`, MSE) and a tiny token model (`token_source`, CE); every per-step loss recorded | **IDENTICAL**: all 30 losses per objective, metrics.jsonl, checkpoint schedule, final `best_model.pt` state dict, config.json (minus `commit_sha`) |
| **f. imports and lint**: all 28 owned modules import in RELEASE (`pim.__file__` is under RELEASE); every RELEASE `pim` module imports (0 failures at the last run); `ruff check --select F,E9` on the owned files | pass / pass / "All checks passed!" |
| **code identity**: AST comparison with docstrings stripped (and a second pass with string constants blanked) | the only code differences are the intended cuts listed above |
| **writing rules**: grep of owned files for dates, "Sevan", discworld/dw-/L-dw/L-oth, research/harness/findings/REGISTRY/PROGRESS/experiments/paper_ci, ⛔/⚠, absolute paths, history words, British spellings | clean |

## Open issues

- **`L-oth-20m/scores.json` is missing** from PRIVATE runs (see export above). I did not touch PRIVATE runs/.
- Training-time `train()` still writes a wall-clock `"at"` timestamp into `config.json["resumed"]` when a run is resumed. This is behavior, not shipped text, so I left it. Say if it should go.
- A GPU run was not done. Every check above ran on CPU, as the brief preferred.
