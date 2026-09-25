# Code-closure audit for the minimal public release (2026-09-23)

Read-only static audit. Nothing in the repo was changed. The helper scripts are in this scratchpad (`closure.py` does module-level imports; `symbols.py` and `permod.py` do top-level def/class/assign reachability that follows package re-exports and in-function imports; `deflines.py` computes line ranges). Line counts come from the working tree, which has uncommitted edits to `selection.py`, `tables.py` and all four notebooks. **Those uncommitted edits are what reproduce the paper's numbers** (the 2026-09-23 highest-fidelity fallback in `best_arm`, and `dw_cat_im` for the 8-ray grids). Cut the release from the working tree, or commit first.

## Headline

- **Module level: everything in `pim/` except `figures/probe_capacity.py` is imported by some listed entry point.** The eager package `__init__`s pull in every editor, every model and the nullspace probe. So "import closure" says nothing on its own. Every cut below also needs its `__init__` export removed.
- **Symbol level:**
  - **No listed entry point uses any symbol from:** `editors/{freeze_interpolation,oracle_overwrite,nullspace}`, `probes/nullspace`, `figures/{probe_capacity,probe_targets}`. `figures/theme` is used only by the unlisted `editability_trends` figures.
  - **Reached only through the model registry's `BUILDERS` dict:** `models/{transformer_s,recurrent,blocks}`.
  - **Reached only behind a config gate:** `discworld/render2d` (the omni2d branch in `renderer.render_frame`).
  - **Used only by the demos:** `discworld/{viz,interactive}`. `viz` also supplies two colour constants to `waterfall`.
- **Estimated shrink of `pim/`** (17,430 lines including vendor):
  - about 1,610 lines from whole-module cuts;
  - about 1,350 lines from function- and branch-level cuts;
  - plus 735 more if the demos are dropped, and about 170 more if the vendor files are trimmed.
  - That brings `pim/` to roughly 14.4k lines, or about 13.5k without the demos.
- **No cut listed here changes a reported number, as long as the release keeps its numbers coming from the existing `scores.json` files.** Selection is per editor: `arms_of` prefix-matches, and `ND` never matches `PI[...]`, `GS@...` or `IM`. There are no legacy or raw PI arms in any in-scope `scores.json`; every discworld PI arm is `PI[zspace]`. The couplings that could break things are listed at the end. The top ones:
  - `_infer_arch` rule 4 must stay: the headline `L-oth-20m` checkpoint has no `arch` key.
  - The `"frustum"` basis name is a cache key and a required basis.
  - `table_gridified` draws every extra block present in `scores.json`.
  - Several paper tables have no entry point at all.

## 1. Module reachability (symbol level)

Entry abbreviations:

| Abbreviation | Entry point |
|---|---|
| ME | `master_eval.ipynb` |
| PT | `build_paper_tables_and_figs.ipynb` |
| AT | `build_appendix_tables_and_figs.ipynb` |
| QRW | `qualitative_edits/make_figure.py` |
| QOTH | `qualitative_edits_othello/make_figure.py` |
| QM | `qualitative_main/*` (via `common.py`, which importlib-loads QRW and QOTH) |

Other names are the scripts. Special cases:
- `make_othello_corpus.py` reaches `othello.corpus` through `runpy` (its `__main__`).
- `layout_checkpoint_replicate.py` imports no `pim` code (torch only).
- `paper_style.py` imports no `pim` code.
- Package `__init__`s always execute on import.

```
editors.grad_steer / .inverse / .pinv    QOTH, QRW, ME, reach, 2flip
editors.nanda                            QOTH, ME, reach, 2flip          (ND only)
editors.freeze_interpolation/oracle_overwrite/nullspace    NOT USED
env.discworld.arms                       QRW, ME, fit_probes
env.discworld.bayes                      bayes_floor, score_pred
env.discworld.bench                      QRW, ME, fit_probes, mk_sel
env.discworld.bigcorpus                  train (and generation via its __main__)
env.discworld.blink                      QRW, ME, bayes_floor, demo, gen_ds, mk_sel
env.discworld.config/observers           everything discworld (incl. PT/AT via tables.table_gridified)
env.discworld.dataset                    QRW, ME, gen_ds, mk_sel
env.discworld.edits_dataset              QRW, gen_ds
env.discworld.frustum / grid_target      QRW, ME, PT, AT (table_gridified), fit_probes, mk_sel
env.discworld.loading                    ME, mk_sel
env.discworld.render2d                   only via renderer's omni2d gate (never taken in scope)
env.discworld.renderer / soft_render     QRW, ME, demo, play, gen_ds, mk_sel
env.discworld.sim                        QRW, demo, play, gen_ds
env.discworld.token_bench                ME, fit_probes
env.discworld.tokens                     ME, fit_probes, mk_tok, mk_sel, score_pred, train
env.discworld.viz                        QRW (2 constants via waterfall), demo
env.discworld.interactive                play
env.layout                               ~every entry
env.othello.arms                         QOTH, ME, bayes_floor, reach, 2flip
env.othello.bayes                        bayes_floor
env.othello.bench                        QOTH, ME, mk_oedits, reach, 2flip
env.othello.corpus / data / vendor.othello   QOTH, ME, bayes_floor, mk_oedits, ocstats, reach, train, 2flip
env.othello.counterfactual               ocstats, reach, 2flip   (replay, flips_per_move only)
env.othello.reachability                 reach, 2flip
env.othello.vendor.mingpt_model          every model loader (via transformer_l)
env.prediction                           score_pred
figures.tables                           PT, AT
figures.waterfall                        QRW (DARK_BG, DIFF_CMAP, GHOST_C, TARGET_C only; waterfall_grid unused by listed entries)
figures.theme / probe_targets / probe_capacity   NOT USED by listed entries
metrics.decodability                     QOTH, QRW, PT, AT, ME, fit_probes, reach, 2flip
metrics.edit_index                       QOTH, QM, PT, AT, ME, reach, 2flip
metrics.prediction                       AT, bayes_floor, score_pred
metrics.replicates                       PT, AT
metrics.selection                        QOTH, QRW, QM, PT, AT, reach, 2flip
metrics.set_editability                  QOTH, QM, ME, reach, 2flip
metrics.zone_editability                 QRW, ME, mk_sel
models.registry / transformer_l / protocol   QOTH, QRW, ME, fit_probes, reach, score_pred, train, 2flip
models.transformer_s / recurrent / blocks    only through registry.BUILDERS (+ models/__init__)
probes.base / cache                      QOTH, QRW, ME, fit_probes, reach, 2flip
probes.inverse                           QOTH, QRW, ME, reach, 2flip
probes.linear / mlp / baselines          QRW, ME, fit_probes
probes.nullspace                         NOT USED
scoring.*                                ME   (scoring.othello._probe_games also reach, 2flip)
training.*                               train
```

**Dynamic lookups checked by hand:**
- `models/registry.BUILDERS` is keyed by arch string. In-scope keys are `transformer_l` and `transformer_l_tokens`.
- `scoring.driver.scorer_for` dispatches on the arch suffix and the env.
- `grid_target.categorical_target` parses target names by regex.
- `common.py` loads QRW and QOTH through `importlib`.
- `make_othello_corpus` uses `runpy`.
- `pinv_step(space=...)` is a string switch.
- The `data_dir=` legacy forms have no in-scope caller. They are only used by `migrate_datasets.py`, which is being dropped.

**Paper artefacts made by scripts outside the listed entry points.** These also need to ship; the last column says what they pull in.

| Paper artefact | Script | What it additionally needs |
|---|---|---|
| Figure `editability_over_res_point`, table `tab:im_by_point` | `paper/figs/editability_trends/by_point.py` | `figures.theme`, `tables.find_run`, `selection` |
| `history_rewrite` figure | `paper/figs/history_rewrite/make_figure.py`, `draw_paper.py` | `figures.waterfall.waterfall_grid`, `discworld.viz`, `iter_inverse_maps` |
| `rayworld_predictions` / `othello_predictions` | `paper/figs/predictive_quality/*` | waterfall and viz constants |
| Overview figures | `paper/figs/environments_overview/{rayworld,othello}/make_figure.py` | vendor `permit_reverse`, `rows`, `columns` |
| Teaser / setup figures | static PDFs in `paper/figs/` | nothing |

## 2. Per-module verdicts

The "−N" figures are estimated line savings.

### Cut whole modules (about 1,612 lines)

| Module | Lines | Reason |
|---|---|---|
| `editors/freeze_interpolation.py` | 73 | oracle editor, not in the paper |
| `editors/oracle_overwrite.py` | 44 | oracle editor |
| `editors/nullspace.py` | 65 | nullspace editor |
| `probes/nullspace.py` | 139 | nullspace cascade |
| `editors/nanda.py` | 80 | ND; the only users are the ND arms, which are cut below |
| `models/transformer_s.py` | 452 | Transformer-S |
| `models/blocks.py` | 101 | used only by Transformer-S |
| `models/recurrent.py` | 265 | Recurrent-L |
| `figures/probe_capacity.py` | 83 | probe-width sweep |
| `figures/probe_targets.py` | 109 | probe-target sweep; remove `sweep_figure` from `figures/__init__` |
| `discworld/render2d.py` | 201 | omni2d; remove the gates in `renderer`, `dataset`, `edits_dataset`, `config.obs_dim`, `zone_editability.sim_config_from` |

**Conditional cuts:**
- `discworld/viz.py` (332) and `interactive.py` (403): keep only if `demos/demo.py` and `play.py` ship. If they go, inline `BG_HEX` and `TEXT_COLOR` into `waterfall.py`.
- `figures/theme.py` (34): keep only if the `editability_trends` figures ship.

### `pim/scoring/` — simplify (about −76)

- **`blocks.py`** (156): keep. Drop ND from `EDITORS_SCORED` and `probe_block`'s alphas (−5); the schema stays readable, since `alphas.ND` is never read. Change the `dw-pn04` fallback in `instance_of` to `dw-noiseless`. Keep `discworld_blocks`, `extra_targets_of`, `dw_block_setup` (the snapped branch serves `pos@appearance`), `cat_inverse_in_scope`, `attach_inverse`.
- **`discworld.py`** (203): drop the ND loops in `score_discworld` (L121–123) and `score_discworld_tokens` (L182–184) (−6). Keep `inverse_discworld`, both categorical and regression.
- **`othello.py`** (166):
  - drop the `add_sub` / ND mode in `othello_arms` and ND in `PROBE_SOURCES`;
  - drop the whole `oth_extra_targets` (mine_signed) block loop, L342–368 (−35);
  - keep `_probe_games`, which `reach` and `2flip` import.
- **`baselines.py`** (369):
  - drop the Othello extra-target (mine_signed) branch of `score_baseline_targets`, L185–208 (−24);
  - optionally drop the two left-aligned `OBS_KINDS`, since tables read only `observation_right(_large)`; the print lines need editing (−6);
  - keep `_dw_regression_floors`, which serves the `pos@appearance` floors;
  - keep `state_span` (`getattr(rand, "state_span", 39)` is TransformerL's `block_size`).
- **`driver.py`** (235): keep. The fold-in hooks (`PIM_ADD_CAT_IM`, `PIM_ADD_NN_R2`, `PIM_FORCE_RESCORE`, `missing_inverse`/`add_inverse`) only matter for incremental updates to existing files. They can be removed later, but they are harmless.
- **`runs.py`, `summary.py`**: keep.

**`master_eval` SETTINGS.** Keys that feed only out-of-scope blocks:
- `dw_alpha_nd`, `dw_grid_alpha_nd`, `oth_alpha_nd`;
- `oth_extra_targets` (set to `()`);
- `oth_reg_alpha_{pi,nd,gs}`;
- in `dw_extra_targets`:
  - noiseless: drop `grid-16x8`, `grid-8x4`, `appearance-lat`, `grid-32x16`;
  - 8-ray: drop `appearance-d2/-d3`, `appearance-lat`, `grid-8x4`, `grid-32x16`, `grid-4x2`;
  - token run: everything except `appearance-fac`;
  - 5-ray: `appearance`;
  - smooth and obs5: `appearance-fac` (not reported; `tab:additional_rw` is continuous only);
  - drop the `R-dw-8ray-20m` and `training_curve/*` entries entirely.

Keep `dw_bases=("frustum","cartesian")` with frustum first (coupling #4). With `EVAL_VERSION` unchanged, pruning SETTINGS triggers no rescoring. The code reads these keys unconditionally, so remove each read together with its key.

### `pim/environments/discworld/`

- **`arms.py`** (885) — simplify (−140):
  - cut the oracle section: `counterfactual_history`, `overwrite_oracle_rollout`, `freeze_oracle_rollout`, `oracle_arm` (L529–596);
  - cut `free_rollout`;
  - cut ND: `nanda_rollout`, `categorical_direction`, `nanda_arm`, and the imports of `freeze_time_rollout`, `frozen_frames`, `overwrite_rollout`, `addition_hook`, `probe_direction`, `object_constants`;
  - cut recurrent-only branches: `_roll_hook`'s `rollout_with_hook`, `as_activations`'s `state_view`, the constant-residual skip in `iter_inverse_maps`;
  - cut the legacy `data_dir` path in `_probe_corpus`;
  - keep the `space=` parameter defaulting to zspace.
- **`bench.py`** (381) — simplify (−15): legacy `data_dir` in `_edit_set`, the `DATA` default, and the `"pos"` entry of `DIM_SETS` (the `"pos"` dim set is retired; SETTINGS uses `("all",)`).
- **`grid_target.py`** (739) — simplify (−90):
  - drop the `appearance-d<k>` and `-lat` branches in `AppearanceTarget`;
  - drop multi-view support (`_n_views`, `_view_poses`, `_single_view`, `_view_cells`, `_view_radix`, and the V>1 paths in `_app_factors`/`_app_factor_sizes`), which serves only obs5's unreported `appearance-fac`;
  - drop the `grid-*-fac` factors (`_grid_factors`, `_grid_factor_sizes`), `full@…` snapping, and `CANONICAL`;
  - keep `GridTarget` (generic, and serves `grid-6x5`, `10x3`, `16x8`), plain `AppearanceTarget`, `FactorisedTarget`, `SnappedTarget`;
  - ⚠ this interacts with coupling #6.
- **`token_bench.py`** (344) — simplify (−30): cut `nanda_arm` and the ND import, and the raw/clipnorm branches in `frame_probs`. Keep `zone_edit_index_expected`, `fidelity_ratio_expected` and `mean_frame_rmse`: the paper's mean-frame rows read them.
- **`tokens.py`** (222): keep; `decode` and `encode_h5` are unused (−8).
- **`soft_render.py`** (291) — simplify (−140): keep `soft_enabled`, `render_frame_soft` and `_profile`'s power branch, all needed by dw-smooth. Cut `render_frame_torch` (66), `blur_matrix`, the psf / occlusion-temp / soft-edge / lambert branches, and `_SQ_FLOOR`. ⚠ The arithmetic must stay bit-identical (coupling #12).
- **`renderer.py`** (189): drop the omni2d gate (−6). Keep the multi-observer path (obs5 is in scope) and blink.
- **`sim.py`** (296):
  - the direction/speed/position-noise branches can go (−12); every draw is gated on std>0, so no RNG stream changes;
  - the `bounce`/`wrap` boundaries (−25) can go only if the demos go (`SimConfig` defaults to `bounce`; all instances use `open`).
- **`config.py`** (186): drop only omni2d in `obs_dim`. **Keep every dataclass field**, including dead ones (coupling #10).
- **`dataset.py`** (368): cut `load_sample` (28) and the omni2d description and validate call (−35). Keep the soft `obs_clean` path.
- **`edits_dataset.py`**: drop `render2d.validate` (−3).
- **`loading.py`** (207): cut `Dataset`, `DatasetBundle`, `_load_h5_dataset`, `load_dataset` (−100), and update the discworld `__init__`.
- **`bigcorpus.py`** (521): drop the `dw-pn04` entry and switch the module default `INSTANCE` to an in-scope instance; it is looked up at import time (−15). Keep the `forbidden` seed ranges (the disjointness proof).
- **`blink.py`**: keep; `transitions` is unused (−10).
- **`observers.py`**: keep (obs5); `observer_of_ray` is unused (−6).
- **`frustum.py`**: keep (the user's exception). It is also structurally required: grids are defined in frustum coordinates, `pos@appearance` needs it, and categorical probes are keyed under `"frustum"`.
- **`bayes.py`**: keep.

### `pim/environments/othello/`

- **`arms.py`** (559) — simplify (−50):
  - `linear_arm`: drop the `add` / `add_sub` modes and the regression-probe (mine_signed) branch;
  - `grad_steer_arm`: drop the regression branch;
  - `observation_probes`: drop the `REGRESSION_TARGETS` branch;
  - `fit_probe_grid`: drop the regression targets;
  - `_split`: drop `"frame"`;
  - `gates`: drop the raw-head stats (`out_sum_mean`, `out_neg_mass_mean`); tables read only `legal_mass`.
- **`data.py`** (217) — simplify (−20): `REGRESSION_TARGETS`, `signed_mine`, the mine_signed branch of `flatten_rows`, the raw/clipnorm branches of `move_probs`/`OUTPUT_KINDS`. `synthetic_games` is unused, but `_one_game` is used by the corpus.
- **`bench.py`** (197): cut `load_li_benchmark`, `shipped_length_distribution` and `BENCHMARK_PKL` (−16). Then `make_othello_edits.py` must require `--length` (the paper uses 20).
- **`corpus.py`** (268): the data-scale `LADDER` rungs M, L1, L2 and `rung()` are optional (−10).
- **`counterfactual.py`** (98): cut `mine_board` and `search_cf` (−48; `search_cf` is used only by the dropped `index_ceiling.py`). Keep `replay` and `flips_per_move`.
- **`reachability.py`, `bayes.py`**: keep.
- **`vendor/`**: see §3.

### `pim/environments/layout.py` and `prediction.py`

- **`layout.py`** (272) — simplify (−60): cut the v1-migration machinery (`_has_v1_files`, the refusal branch in `ensure_marker`, `parse_dataset_path`, `legacy_probe_key`, `legacy_edits_instance`, `unused_dir`, edits `v2`). Set `DEFAULT_INSTANCE["discworld"]` to `dw-noiseless`. **Keep `probe_key` byte-identical**: it is inside every discworld probe cache key.
- **`prediction.py`** (117): keep; `load_floor` is unused (−3).

### `pim/editors/`

- **`pinv.py`** (190): drop the raw and legacy spaces (`pinv_maps` → zspace only; the raw/legacy branch of `pinv_step`) (−35). Numerics are unchanged (coupling #13).
- **`grad_steer.py`** (203): drop `optimizer="gd"` (−8).
- **`inverse.py`** (40): `inverse_delta` is unused (−4).

### `pim/probes/`

- **Keep:** `base.py`, `cache.py`, `linear.py`, `mlp.py`, `inverse.py`.
- **`baselines.py`**: keep. `CausalHistory`'s left alignment can go if the scorer stops fitting the left floors (−10).
- **`probes/__init__`**: remove the nullspace export.

### `pim/models/`

- **`registry.py`** (146): drop the S and recurrent builders, and legacy rules 1–3 (−30). **Keep rule 4** (coupling #2). `load_run` is unused.
- **`transformer_l.py`** (298): drop `output_kind="raw"`, the mixed input/head combinations and the validation around them (−20). **Keep the constructor order, parameter names and the `_seq_mask` stub**; `collect_residuals` calls it (coupling #3).
- **`protocol.py`** (83): the `WorldModel` Protocol is unused; trim the S/R documentation (−15).
- **`models/__init__`**: remove the S and R exports.

### `pim/training/`

- **`train.py`** (317): `mse_next_move_onehot` (−19) and the cosine schedule are optional.
- **`sources.py`** (136): `othello_source` is unused, and the mse_onehot objective can go (−16).
- **`stream.py`**: keep.

### `pim/metrics/`

- **Keep:** `edit_index`, `replicates`, `selection` (working-tree version).
- **`decodability`**: `trivial_error_rate` and `probe_skill_classification` are used only by tests (−31, optional).
- **`prediction`**: `top1_accuracy` and `excess` are unused (−10).
- **`set_editability`**: `move_fidelity_ratio_per_case` is unused (−7).
- **`zone_editability`**: `object_constants` (oracle only), `random_samples`, `SCORECARD_COLUMNS`, and the omni2d kwargs in `sim_config_from` (use `.get`, which is safe) (−30).

### `pim/figures/`

- **`tables.py`** (958) — simplify (−190):
  - cut `fig_training_curve` (129; its training-curve runs are not in the paper);
  - cut `tables_components` and `_rand_perdim` (48, including recurrent logic), `table_arms`, `_mark`, the per-dim collection in `_collect`, and the S/R entries of `ARCH_LABEL`;
  - keep `image_table` and **`table_seed_variance`**, and add the latter to the appendix notebook with cartesian basis and no ND columns (coupling #16);
  - keep `set_basis`, `REG_BASES` and `BASIS_BY_INSTANCE` (coupling #5);
  - drop ND from `EDITORS_ALL` and `SEED_VARIANCE_COLS`.
- **`waterfall.py`**: keep (the user wants it; `history_rewrite` uses `waterfall_grid`).
- **`figures/__init__`**: remove `sweep_figure`.

### Scripts

- **Keep as they are:** `bayes_floor`, `score_prediction`, `reachability_table`, `two_flip_editability`, `othello_corpus_stats` (source of the paper's flips-per-move prose), `make_discworld_tokens`, `make_edit_selection`, `make_othello_corpus`, `make_othello_edits` (after the `--length` change).
- **`fit_probes.py` is REQUIRED.** The scorer never fits categorical probes or their floors (`require_cached`). Its per-run invocation list lived in the dropped `scripts/drivers/probe_target_fit.sh`, so document it.
- **`layout_checkpoint_replicate.py` is REQUIRED.** It builds every `__seed0_s512000` member from the parent's `ckpt/step_512000.pt`, which is on the 1000·2^k schedule.
- **`train.py`** (−40): drop the `transformer_s`/`recurrent_l` arch choices, `--objective mse_onehot`, `--dropout` (dropout ablation) and optionally `--limit`. **Change `CANONICAL_INSTANCE["discworld"]` from `dw-pn04`.**
- **`generate_dataset.py`** (−30): drop the omni2d, soft-edge, psf and occlusion flags. The noise flags are passed explicitly as 0.0 by every in-scope instance through `bigcorpus` sim_flags; if the flags are removed, the 0.04 defaults must go too.
- **Demos:** `demo.py` and `play.py` need `viz`, `interactive`, and the `sim` bounce boundary. Optional.

## 3. Vendored Othello code (`pim/environments/othello/vendor/`)

- **`LICENSE` is present** (MIT, Copyright (c) 2023 Kenneth Li). Both `.py` files have attribution headers.
- **`othello.py`: the header is wrong.** It says "byte-identical … Nothing kept here is modified", but git history shows local edits: the `flip` and `placement` parameters, the `_captures`/`_legal` refactor, and the adjacent-flip change (commit 57dcc21). Change the header to "modified". MIT allows modification but the attribution must be accurate.
  - **Used:** `OthelloBoardState` (`update`, `umpire`, `get_valid_moves`, `state`, `next_hand_color`, `tentative_move` in tests, `_captures`, `_legal`), `get_ood_game`, `eights` (tests), and `rows`/`columns`/`permit_reverse` (only the environments_overview figure).
  - **Unused (about 60 lines):** `mask`, `permit`, `start_hands`, `wanna_use`, `get_occupied`, `get_age`, `__print__`, `plot_hm`, `get_gt`.
  - ⚠ `get_ood_game` plus `get_valid_moves` ordering plus the global `random` must stay exact for corpus regeneration.
- **`mingpt_model.py`:**
  - **Used:** `GPT`, `GPTConfig`, `Block`, `CausalSelfAttention`. `GPT._init_weights` is load-bearing for the random-init floor.
  - **Unused:** `GPTforProbing`, `GPTforIntervention`, `GPTforProbeIA` (109 lines), plus `GPT.forward` and `configure_optimizers` (tests may reference `GPT.forward`).
  - **Attribution:** Li's `mingpt/model.py` derives from Karpathy's minGPT (MIT). Add that attribution; this needs checking against upstream.
- **`intervention_benchmark.pkl`** (Li's 1001 cases, 60 KB): not used for any paper number. It is read only by `load_li_benchmark`, `shipped_length_distribution` (`make_othello_edits --length 0`) and `tests/test_othello_bench.py`. It can be dropped.

## 4. `pyproject.toml` dependencies

- **Imported by the closure:** `numpy`, `torch`, `h5py`, `matplotlib`, `tqdm` (`dataset.py`), **`pandas`** (`tables.py`, `metrics/replicates.py`) and **`seaborn`** (`tables.py`).
- **`pandas` and `seaborn` are declared in neither `pyproject.toml` nor `poetry.lock`.** They exist only in the local venv (3.0.5 and 0.13.2), so a fresh install cannot import `pim.figures.tables`. Add both.
- **Notebook execution also needs `jupyter`/`nbconvert`**, which is undeclared (only `ipykernel` is in the dev group).
- **Can drop:** `torchvision` (never imported). `pillow` is not imported directly; it is a matplotlib dependency, and the demo GIF writer uses it only through matplotlib, so its explicit pin can go.
- **Dev group:** `ipywidgets` and `nbstripout` are not needed for the release; keep `pytest`.
- The odd Python pin (excluding 3.14.1) exists only because of torchvision and triton, so it can be simplified once `torchvision` is gone.

## 5. Tests — candidate shipped suite

Counts are test functions (parametrised tests counted once).

- **Keep unchanged: about 143 tests in 28 files.**
  - Discworld environment: `bench_target_alignment`, `blink`, `drop_edge_rays`, `renderer`, `sim`, `observers`, `grid_target`, `discworld_tokens`.
  - Othello environment: `corpus_provenance`, `othello_{adjacent,adjacent_flip,noflip,instances,reachability}`.
  - Metrics and tables: `case_variance`, `edit_index_shared`, `fidelity_guard`, `prediction_floor`, `selection_replicates`, `tables_replicates`, `tables_module`.
  - Probes: `probe_cache_canonical`, `probes_canonical`.
  - Training: `training_canonical`, `training_resume`.
  - Other: `token_bench`, `waterfall_grid`.
  - `categorical_inverse`: all but the clear-script test, which loads `experiments/categorical_inverse/scripts/clear_continuous_im.py`.
  - Data-dependent tests (`tables_module`, `token_bench`, parts of `bench_target_alignment`, `prediction_floor`, `probe_targets`) skip when runs or datasets are absent.
- **Keep after edits: about 61 tests.**
  - `editors_canonical`: drop nanda, nullspace, legacy and raw; about 13 of 20 remain.
  - `models_canonical`: drop the S parametrisations, S legacy rules and raw `output_kind`; about 7 of 12 remain.
  - `baselines`: swap RecurrentL for TransformerL in `test_collect_residuals_points_subset`; 11 remain.
  - `layout`: drop the legacy-key tests; 2–3 of 5 remain.
  - `scoring_package`: drop the mine_signed expectations; 6 remain.
  - `probe_targets`: drop `signed_mine`, the Othello regression arms, and the appearance variants; about 14 of 17 remain.
  - `soft_render`: keep defaults bit-identical, hard single-ray and the power-profile tests; drop torch, occlusion, psf, lambert and edge; about 5 of 11 remain.
  - `othello_bench`: the pkl tests need a synthetic replacement.
  - `othello_counterfactual`: drop the `search_cf` test.
- **Cut: about 63 tests in 7 files.**
  - `oracle_editors` (1), `recurrent` (13), `transformer` (12, Transformer-S), `render2d` (18), `mse_onehot_head` (3), `multiview_target` (4, only if multi-view is cut).
  - `interactive` (12): cut only if the demos go.
- **Result: a core suite of about 200 tests in about 37 files.** This was not executed; running pytest would write cache directories.

## 6. Risky couplings — where a cut could break scoring of the shipped runs or change a number

1. **Package `__init__`s import eagerly.** Every removed module must be dropped from `pim/{editors,models,probes,figures}/__init__.py` and from the `discworld`/`othello` `__init__` exports (`load_li_benchmark`, `Dataset`, `load_dataset`, `load_sample`). Otherwise `import pim.models` fails.
2. **`_infer_arch` legacy rule 4 must stay.** `runs/initial_othello_comparison/L-oth-20m/best_model.pt` (the headline Othello model) has no `arch` key: it is a bare minGPT state dict with `model_config={vocab_size, block_size, …, *_pdrop}` and a top-level `vocab`. Every other in-scope checkpoint, including all 32 in-scope `__seed*` replicates, carries `arch`, so rules 1–3 can go.
3. **Keep the TransformerL constructor exactly.** Probe and inverse-map caches are keyed by `fingerprint(state_dict)`. The random-init floor is `torch.manual_seed(0)` followed by `build(...)`, so parameter creation order and `GPT._init_weights` set its weights. Renaming or reordering modules (`gpt` first, `tok_emb`/`head` replaced by `Identity`, `encoder`, `_modules["decoder"]`) orphans the caches and changes the Table 1 random-init column on refit. `_seq_mask` is called by `probes.base.collect_residuals`.
4. **The `"frustum"` basis is load-bearing even though no frustum number is reported.**
   - `blocks.discworld_blocks` keys every categorical block's probes, inverse maps and floors under `dw_bases[0]`, which is `"frustum"`.
   - `pos@appearance` (`tab:categorical`) raises `ValueError` in any basis other than frustum.
   - QRW hardcodes `CAT=("appearance-fac","frustum")`.
   - Removing or reordering frustum in SETTINGS makes fresh scoring skip every categorical block (`require_cached` misses). Dropping only the frustum *regression* block needs a code change in `discworld_blocks`, not a SETTINGS change.
5. **`tables.REG_BASES` must keep `"frustum"`.** `_collect` skips the regression block that is not requested only because it is in `REG_BASES`. Remove it and every run's frustum block becomes an "extra" row that shows up in `table_gridified` (paper Table 3 / appendix A2b).
6. **`table_gridified` draws every extra discworld block present in `scores.json`.** That includes out-of-scope `grid-8x4`, `grid-32x16`, `grid-4x2`, `appearance-d2/-d3/-lat` and 5/16-ray `appearance`. It calls `target_cells(name)` on each. If the `-d<k>`/`-lat` parsing is cut, those rows still render with `cells=None`. Fix one of two ways: add a target whitelist to `table_gridified` (display policy only, no number changes), or strip those blocks from the shipped `scores.json` files. `table_gridified` also opens `datasets/discworld/<inst>/edits/v1/edits.h5` for the sim config, so **rendering the tables needs the edits datasets on disk**.
7. **ND and mine_signed arms and blocks can stay in the old `scores.json` files.** Selection is per editor (`arms_of`); Othello extra rows are filtered from every paper table; `pool_replicates` only gains unused ND columns. Removing the code changes nothing, but keep `EDITORS_ALL`, `pool_replicates` columns and `SEED_VARIANCE_COLS` consistent with each other.
8. **SETTINGS keys are read unconditionally** (`s["dw_alpha_nd"]`, `s["oth_reg_alpha_*"]`, `s["oth_extra_targets"]`, `s["dw_grid_alpha_nd"]`). Prune each key together with its reads. `EVAL_VERSION` stays as it is, so there is no forced rescore; `missing_blocks` asks only for what SETTINGS lists.
9. **Replicate pooling reads each member's `config.json` `replicate.steps` and `seed`.** Ship every `__seed*` directory. `__seed0_s421875` on adjflip and noiseless is only listed as dropped by the ±10% budget guard; it changes no SD.
10. **Keep the dead `SimConfig` fields.** Stored `config_json["dataset"]["sim"]` in every existing dataset carries all fields (omni2d*, soft_edge, soft_psf_sigma, soft_occlusion_temp, noise). `SimConfig(**stored)` is called at `paper/figs/qualitative_edits/make_figure.py:87` (and in `dataset.load_sample`). Deleting the fields makes the qualitative figure crash on existing data. Keep them, or filter unknown keys.
11. **Data regeneration must reproduce bit for bit.**
    - Discworld noise draws are gated on std>0, so cutting noise features is safe.
    - But the `SimConfig.obs_noise_std` and `generate_dataset --obs-noise-std` defaults are 0.04. They are safe only while the instances keep passing an explicit 0.0.
    - Othello games come from global `random.seed(seed*1_000_003+i)` followed by vendor `get_ood_game`/`get_valid_moves`. Do not touch those.
12. **dw-smooth renders through `render_frame_soft`** (power profile; edge, psf and temp all 0), and its datasets have `obs_clean` because `soft_enabled`. Trimming branches must leave the float expressions identical, e.g. `alpha = gate * (signed > 0)` and `front = (dt>0).astype(float)`. Keep `test_defaults_are_bit_identical` and `test_power_profile_*`.
13. **The PI raw/legacy cut is numerically safe.** The zspace path recomputes `pinv(A)` on its own (sliced) A and never uses the other maps.
14. **The fields the table builders read must survive.** Tables read these:
    - per block: `probe_skill_{linear,mlp}`, `probe_sanity.{n_violations,rows[].insample_gap_*}`, `unedited`, `arms[]{editor,point,alpha,dims,edit_index|edit_index_symdiff,fidelity_ratio}`, `inverse_map.{g_r2,nn_r2}`, `kind`;
    - top level: `instance`, `arch`, `val_loss`, `ei_construction`, and for Othello `probe_skill`, `probe_stats`, `gates.legal_mass`, `prediction.readings`;
    - `config.json` `replicate`; `baselines.json` `archs.<arch>.bases.<basis>.{observation_right(_large),random_init}`; `bayes_floor.json`.
    - Anything that only writes other fields (ND arms, `alphas.ND`, gates raw stats, left-aligned floors, `probe_perdim_*`) is safe to cut.
15. **Several paper numbers have no entry point, so the release cannot regenerate them from the listed notebooks:**
    - `app:metric_spread` seed SDs: `table_seed_variance` exists only in `build_full_tables` (frustum basis). The paper uses the cartesian `experiments/paper_ci` ledger.
    - `tab:im_vs_nn`: IM-NN EI and fidelity live only as df columns; there is no renderer.
    - `tab:additional_rw`: smooth and obs5 are in no shipped notebook's run list.
    - `tab:tokens_editability` mean-frame rows: this needs `best_arm` on `zone_edit_index_expected` with `fidelity_ratio_expected` as the guard. `best_arm` guards on `fidelity_ratio`, and no `pim` function does this.
    - `tab:categorical`: 8-ray grids plus `pos@appearance`, via `collect` outside the listed notebooks.
    - `tab:im_by_point`: `editability_trends/by_point.py`.
    - `tab:legal_illegal` / `tab:two_flip`: hand-typed from the scripts' JSON.
    - Add notebook cells, or ship those scripts.
16. **Reproducing the whole chain from scratch needs more than `master_eval`:**
    - `fit_probes.py` for every categorical target and floor (`require_cached`);
    - `layout_checkpoint_replicate.py` for the seed-0 members;
    - `score_prediction.py` and `bayes_floor.py` for Table A1;
    - `make_edit_selection.py` for `edits/v1/selection.json`.
    - `master_eval` scans only runs that have `best_model.pt`, so it is a no-op if checkpoints are not shipped.
    - The existing IM, categorical-IM and `nn_r2` values were folded in incrementally, and one experiment script cleared the continuous-map arms. A fresh `score_all` reproduces them only to GPU noise, about 1e-4 per the paper notes.
17. **The qualitative figures need more than `scores.json`.** They read `.scratch/qualitative_edits_catim_*.pkl` (gitignored; built by QRW), checkpoints, `datasets/*/edits`, and the runs' `probes/` caches. Without the caches, the categorical probes refit at 200k sequences.
18. **Change the `dw-pn04` defaults when its registry entry goes.** `layout.DEFAULT_INSTANCE`, `bigcorpus.INSTANCE` (looked up at import: `INSTANCES[INSTANCE]`), `train.py CANONICAL_INSTANCE` and the scoring fallbacks all default to it. Deleting `INSTANCES["dw-pn04"]` without changing them breaks `import bigcorpus`.

---
Note: `scripts/othello_corpus_stats.py` changed on disk during this audit (mtime 21:39 PT, +19/−5 lines). This audit did not make that change; every command it ran was read-only. The audit read the earlier version of that script; its imports (`othello.corpus`, `counterfactual.flips_per_move`) should be rechecked if the change touched them.
