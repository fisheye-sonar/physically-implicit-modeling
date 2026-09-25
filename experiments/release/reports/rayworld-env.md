# rayworld-env worker report

Owned: `RELEASE/pim/environments/rayworld/**`. The package went from 6,697 to 4,346 lines. Every module imports, ruff is clean, and every equivalence check against PRIVATE is bitwise identical. The only differences are the planned renames and `FLOOR_VERSION` (now "1.0").

Helper scripts and outputs: `experiments/release/work/rayworld-env/` (`gen_private.py`, `gen_release.py`, `compare_gen.py`, `targets_check.py`, `bayes_check.py`, `cache_check.py`, `arms_check.py`, `tokens_check.py`, `demo_check.py`, `compare_pkl.py`, `style_check.py`; results in `out/`). The bootstrap copy of every file is in `work/rayworld-env/orig/`.

## Deleted

- **`render2d.py`**, and the omni2d branch in `renderer.render_frame`, `config.obs_dim`, `dataset.generate_dataset` (validate call and description) and `edits_dataset`. It is also gone from `bayes.unsupported`. Every `SimConfig` field is kept, `omni2d*` included, in the same order with the same defaults (checked).
- **`soft_render.py`**, reduced to the power-profile path:
  - removed: `render_frame_torch`, `blur_matrix`, `_sigmoid`, `_SQ_FLOOR`, `_profile`, and the edge, psf, occlusion-temperature and lambert branches;
  - kept: the float expressions of the path the smooth variant uses, unchanged;
  - an unsupported knob now raises through `config.check_supported`.
- **`loading.py`**: removed `Dataset`, `DatasetBundle`, `_load_h5_dataset` and `load_dataset`, which have no in-scope consumer. `EditsData` and `load_edits` stay.
- **`dataset.load_sample`**: no in-scope consumer; the qualitative figure builds `SimConfig(**stored)` itself.
- **`grid_target.py`**: removed `appearance-d<k>`, `appearance-lat`, all multi-view support (`_n_views`, `_view_poses`, `_single_view`, `_view_cells`, `_view_radix` and the V>1 factor paths), `grid-*-fac`, `full@` snapping, `CANONICAL`, `cell_of_frustum` and `cell_centre_world`. The post-hoc method assignment became normal methods.
- **`arms.py`**:
  - ND: `nanda_rollout`, `nanda_arm`, `categorical_direction`;
  - the oracles: `counterfactual_history`, `overwrite_oracle_rollout`, `freeze_oracle_rollout`, `oracle_arm`;
  - `free_rollout`;
  - recurrent-only branches: `rollout_with_hook` in `_roll_hook`, `state_view` in `as_activations`, and the constant-residual skips in both inverse-map iterators;
  - the path-based corpus lookup: the `data_dir` and `split` parameters and `layout.legacy_probe_key`.
- **`bench.py`**:
  - the `DATA` default and the `data_dir` path forms (`layout.legacy_edits_instance`);
  - the `"pos"` dim set;
  - `restrict_mask`, which was the identity for `"all"`.
- **`token_bench.py`**: `nanda_arm`, `_check_dims`, the raw and clipnorm output kinds, and the `data_dir` parameter.
- **`tokens.py`**: `decode` and `encode_h5` (unused), and the `layout.parse_dataset_path` fallback in `h5_splits`. The meta fields `created` and `layout` are gone; `sources` is now repo-relative.
- **`blink.transitions`** and **`observers.observer_of_ray`**: unused.
- **`bigcorpus.py`**:
  - removed the rw-pn04 instance and the `ensure_marker` call;
  - the default `INSTANCE` is now `"standard"`;
  - the forbidden-range bookkeeping is now one `SEED_RANGES` table of in-scope splits plus unnamed `RESERVED` ranges. For every instance, `forbidden(inst)` returns every range except its own training range;
  - the shard subprocess now inherits `os.environ` (plus PYTHONPATH) instead of a hardcoded PATH, and `rm -rf` became `shutil.rmtree`.
- **Every docstring and comment was rewritten** to the SPEC rules: module docstrings are at most 4 lines, function docstrings at most 3, comments one line; there are no dates, history, names, research-file references or banners, and prose uses American spelling. Every "Sevan" line from `scan_full.tsv` is gone.

## Kept, and why

- **`frustum.py`**: SPEC exception. Categorical probes are keyed under "frustum", and `pos@appearance` needs it.
- **`viz.py` and `interactive.py`**: the demos need them, and `viz.BG_HEX` / `TEXT_COLOR` are imported by `pim.figures.waterfall`.
- **`observers.py`**: obs5 data generation.
- **`sim.py` noise and boundary branches**: the noise draws only happen when std > 0; `bounce` and `wrap` are used by the demos.
- **`GridTarget` is generic**, serving grid-6x5, grid-10x3 and grid-16x8. Snapping is generic `pos@<categorical>`.
- **Arms kept for other consumers**:
  - `unsteered_rollout`, `pinv_rollout`, `grad_steer_rollout` (qualitative, history-rewriting and prediction figures);
  - `iter_inverse_maps` (history-rewriting and qualitative figures);
  - `score`, `as_activations`, `probe_recipe`, `GRID_PROBE_RECIPE`, `fidelity_ratio` and `collect_residuals`. `collect_residuals` stays a module global because the qualitative figure monkeypatches `rwa.collect_residuals`;
  - `pinv_target` and `readout_landed` (token bench);
  - `fit_probes(seed=)` (probe-seed replicates).
- **Bench functions kept for figures and scripts**: `bench_from_arrays`, `bench_of`, `full_state_pair`, `EF`, `N_OBJ`, `K_ROLL` (figures), and `bench_arrays(use_selection=False)` (`make_edit_selection`).
- **`edits_dataset._generate_one_edit`, `blink.blink_schedule`, `renderer.render_scene` / `render_frame`, `sim.Scene`**: the qualitative figure.
- **`bayes`**: `bayes_floor`, `trivial_predictors`, `N_FLOOR_SEQ`, `FLOOR_VERSION` (`scripts/bayes_floor.py`, `prediction.py`).
- **`tokens`**: `load_tokens`, `tokenize_instance`, `UNK`, `encode`, `FrameVocab` (train, token script, prediction, `make_edit_selection`).

## API changes (in-scope signatures)

- **`fit_probes(model, target, n_seq, family, log, basis_name, cache, cache_dir, encoder, encoder_tag, epochs, require_cached, probe, seed)`**:
  - lost `split` and `data_dir`;
  - no consumer passed either, and all consumers use keywords after `n_seq`.
- **`observation_probes` and `iter_inverse_maps`** also lost `split` and `data_dir`.
- **`probe_recipe(target, instance, n_seq)`**: the second parameter is renamed from `inst_root` (always passed positionally). It also accepts `"rayworld/<inst>"`.
- **Bench functions**:
  - `bench_arrays(n, target, basis_name, *, select, use_selection, instance)` and `load_bench(model, n, target, basis_name, *, select, use_selection, instance)`: `data_dir` is removed and the rest is keyword-only;
  - `load_token_bench` has the same change;
  - `grid_selection(n, grid, instance=None)`;
  - `selection_path(instance=None)`.
- **`tokens.h5_splits(instance)`** now takes an instance name.
- **Categorical targets and unsupported configs raise**:
  - a multi-observer sim now raises `ValueError` in the appearance targets and in cell-indexed labels. Before, the appearance targets supported several views; cell-indexed labels already refused them;
  - `SimConfig` options that are not implemented raise in `generate_dataset`, `generate_edits_dataset` and `render_frame_soft`: omni2d, lambert, soft_edge, psf, occlusion temperature. The new function is `config.check_supported`.
- **Written strings that changed**:
  - the `grid_selection` "rule" drops ", 2026-09-12";
  - `FLOOR_VERSION` is "1.0";
  - `AppearanceTarget.factor_names` is ("center", "length"). Nothing reads it.

## Requests to other owners

1. **scoring, `pim/scoring/rayworld.py`**: delete the ND loops that call `rwa.nanda_arm` (about L123) and `tkb.nanda_arm` (about L184). Both functions are gone.
2. **scoring, `pim/environments/layout.py`**:
   - set `DEFAULT_INSTANCE["rayworld"] = "standard"`;
   - keep these callable as used here: `REPO`, `DEFAULT_INSTANCE`, `instance_root(cls, inst)`, `train_dir(cls, inst)`, `eval_file(cls, inst)`, `edits_file(cls, inst)`, `edits_selection(cls, inst)`, `probe_file(cls, inst, size)`, `probe_manifest(cls, inst, size)`;
   - `probe_key(cls, inst, size)` must keep returning `("rayworld/<inst>", "probe_<size>")`;
   - rayworld no longer calls `legacy_probe_key`, `legacy_edits_instance`, `parse_dataset_path`, `ensure_marker` or `edits_dir`.
3. **scripts, `scripts/generate_dataset.py`**:
   - remove the flags `check_supported` now rejects: `--omni2d`, `--omni2d-h`, `--omni2d-w`, `--soft-edge`, `--soft-psf-sigma`, `--soft-occlusion-temp`, and the `lambert` choice of `--soft-shading`. Also remove the `render2d` references;
   - keep the positional-directory four-split mode and the flags `bigcorpus.generate_shard` passes: `--n-train --n-val --n-test --n-edits --seed --n-workers --compression-level`, plus the instance flags `--n-objects --frames --boundary --fixed-reflectivities --always-in-frustum --obs-res --drop-edge-rays --radius --max-edit-attempts --position-noise --obs-noise-std --blink-prob --blink-mean --blink-max --blink-warmup --soft-shading --soft-profile-power --n-observers --region`;
   - that path reads `meta_train["generated_at"]`, which `dataset.py` still writes.
4. **scripts, `scripts/train.py`**: `{"corpus": str(bc.OUT)}` writes an absolute path into `config.json`. Use `str(bc.OUT.relative_to(bc.REPO))`. Also set `CANONICAL_INSTANCE["rayworld"]` to `"standard"`.
5. **scripts, callers of the changed signatures** (see API changes). None in PRIVATE passed `data_dir` or `split`.
6. **tables, `pim/figures/tables.py`**: `target_cells` returns None for the out-of-scope targets (appearance-d2/-d3/-lat, grid-*-fac, full@…). It raises for obs5's `appearance-fac` (multi-observer). `table_gridified` should whitelist the in-scope targets, or those blocks must be stripped from the shipped `scores.json`.
7. **export**:
   - strip the out-of-scope categorical blocks: obs5 `appearance-fac`, appearance-d*/-lat, grid-4x2/8x4/32x16, 5/16-ray `appearance`;
   - freshly written values now read: `grid_selection` "rule" without ", 2026-09-12"; `bench_selection.file` = `datasets/rayworld/<inst>/<edits dir>/selection.json`; `bayes_floor.json` "version" "1.0".
8. **infra, `.gitignore`**: add `.scratch/`. `arms._scratch_dir()` memory-maps residual stacks there at the repo root.
9. **core, `pim/metrics/zone_editability.py`**: the docstring near L67 still names `render2d.py`. `sim_config_from` passing the omni2d fields is fine, since the fields are kept.

## Verification (all run; exact results)

- **(a) Data generation, 8 instances** (standard, blink, 128-ray, 16-ray, 8-ray, 5-ray, smooth, obs5):
  - Setup: PRIVATE built each training `SimConfig` through its own `generate_dataset.py` parser from the bigcorpus flags, and RELEASE rebuilt it from that dict. Both generated the first 3 sequences of shard 0 (`base_seed + i`) plus sample 0 of the eval, probe_120k and edits splits, using each split's stored config and seed.
  - Result: every packed field is bitwise identical, PRIVATE against RELEASE (49 arrays per instance, 53 for blink and smooth). The fields are obs_intensity, obs_depth, obs_id, positions, velocities, reflectivities, radii, colors, is_visible, seeds, n_objects, blink_visible, obs_clean and the edit fields.
  - Against the rows stored on disk (read-only), generated data is identical too: 44/44 fields for standard, 128-ray, 8-ray, 5-ray and obs5; 47/47 for blink and smooth; 38/38 for 16-ray, which has no local training corpus, so its train rows were compared PRIVATE against RELEASE only.
  - The bigcorpus registry is identical for all 8 instances: base_seed, obs_dim, sim_flags and the forbidden-range set.
- **(b) Categorical targets** on the first 400 sequences of each instance's probe_250k plus 20k random reachable positions:
  - 8-ray: appearance, appearance-fac, grid-6x5, grid-10x3, grid-16x8, pos@appearance;
  - appearance-fac on 128-ray, 16-ray and 5-ray; grid-16x8 on standard;
  - compared: labels, conflicts, `cell_of`, `n_cells` / `n_tiles_on` / `n_classes_on`, centroids, snap, `edit_cells` / `edit_moves`, `target_cells`, `selection_target`, `arms._targets` (the probe-fit targets), and the full / pos targets in cartesian and frustum;
  - 197 results: all identical except `probe_recipe` instance names ('dw-8ray' vs '8-ray'). The SimConfig field list (names, types, defaults) and `obs_dim` of all 8 instances are also identical.
- **(c) Bench and edit zones**: `bench_arrays` (n=24) on real PRIVATE edits, read-only; RELEASE had layout pointed at PRIVATE files.
  - 8-ray, 9 target/basis combinations (full cartesian/frustum, pos, appearance-fac, appearance, the three grids, pos@appearance);
  - blink, smooth and obs5 on full; 5-ray on appearance-fac and full; a first-n bench per instance;
  - compared: obs, pos, vel, clean, gt_roll, y, change_mask, out_dims, cells, moves, n, kind, every zones field, the selection record (minus the path and the dated rule text) and `full_state_pair`: all identical.
  - Token bench on 8-ray (full and appearance-fac): tokens, pre/post tokens, keep, tgt, change_mask, legal sets, the full scorecard and `expected_frame` are identical, as are `encode`, `frame_codes` and the token-encoder tag.
- **End-to-end arms** (8-ray frame model and 8-ray token model, from real PRIVATE caches; cache writes were monkeypatched to raise):
  - `unsteered`, `pinv_arm` (2 points × 3 alphas), `grad_steer_arm`, `pinv_rollout`, `unsteered_rollout`, and `inverse_arms` continuous IM + IM-NN (point 4, with the retrieval bank). For the token model: `unsteered`, `pinv_arm`, `grad_steer_arm` and `inverse_arms`;
  - 16 results, 0 differences.
- **Probe-cache keys**: RELEASE with the data field mapped to the old spelling got `require_cached` hits on all 20 lookups; PRIVATE got the same 20. The lookups were:
  - frame model: full cartesian/frustum, appearance-fac, appearance, grid-16x8 and pos@appearance, each linear and MLP;
  - token model: full cartesian/frustum and appearance-fac;
  - the categorical inverse-map lookups at points 0 and 4.
  - Stats are identical and no file under PRIVATE `runs/` was written (checked by mtime).
- **(d) Bayes floor**:
  - `unsupported` on all 8 eval configs, `marker_process` on blink's schedule and `observation_codes` are identical;
  - so is a tiny `bayes_floor` on 8-ray and blink (n_seq 3, 32 particles, CPU, `trivial_predictors` included);
  - only `FLOOR_VERSION` differs, as planned.
  - Tokens: `tokenize_instance` on a 5,000-sequence synthetic 8-ray instance gives an identical `train.i16`, an identical `vocab.npz` (byte-identical file) and identical meta, apart from the removed fields.
- **Demos**:
  - `simulate` + `render_scene` + `make_waterfall` in bounce, wrap and open, with all three noise terms on, are identical;
  - `animate_scene` builds and steps;
  - `InteractiveWorld`: 200 steps in force and in shift mode, with death and noise frames, are identical.
- **(e) Imports, lint and guards**:
  - `python -c import` of all 20 modules plus the package: OK, with `pim.__file__` under RELEASE;
  - `ruff check`: all checks passed, including F (undefined names) and E9;
  - guards: obs5 appearance targets raise; omni2d, lambert, soft_edge, psf and temp configs raise;
  - a grep finds none of the forbidden strings (names, dates, discworld, dw-, research files, pn04, render2d, ND, oracle, recurrent, legacy, absolute paths).

## Open issues

- **`bigcorpus.generate_shard`** depends on the four-split CLI of `scripts/generate_dataset.py` (request 3).
- **Defaults depend on layout**: the bench and probe defaults use `layout.DEFAULT_INSTANCE["rayworld"]` (request 2). `bigcorpus` itself defaults to "standard".
- **obs5 categorical targets now raise.** If any shipped table code calls `target_cells` on obs5's `appearance-fac` block, it will raise until the block is stripped or whitelisted (requests 6 and 7).
- **The SPEC does not fix eval/probe/edits seeds in code.** Each split's seeds live in its HDF5 `config_json` and manifest, and `SEED_RANGES` records the ranges.
