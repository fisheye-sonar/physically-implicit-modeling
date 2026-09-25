# Worker report: scoring

Owned (RELEASE): `pim/scoring/**`, `pim/environments/layout.py`, `pim/environments/prediction.py`,
`pim/environments/__init__.py`, `notebooks/master_eval.ipynb` (new). The owned modules went from
1,692 to 1,218 lines. Helper scripts, logs and every comparison output are in
`experiments/release/work/scoring/` (`results/`). The throwaway trees were deleted after the runs.

**Headline.** The scorer runs on the shipped bundle as a no-op: all 43 runs and 12 baselines
skip, and the notebook executes against the read-only STAGING tree without error. Rescoring
Othello standard from scratch reproduces the shipped `scores.json`:
- PI, GS, gates, Probe Skill, probe stats and the unedited card are bit-identical.
- IM / IM-NN differ by at most 8.3e-7.
- The 10 refit probe files are bitwise equal to the shipped ones.

One open issue: GS on categorical blocks differs from the shipped files by up to 7.5e-3 in Edit
Index. PRIVATE's own code shows the same difference on this GPU, so the release did not cause
it (see Open issues).

## What changed and why

### `pim/environments/layout.py` (rewritten, 272 → 144 lines)
- **Flattened `edits/`.** `edits/v1/` is now `edits/`, matching exactly what STAGING ships:
  - Othello: `datasets/othello/<inst>/edits/cases_<n>.pkl`, next to the `edits_<n>.npz` games;
  - Rayworld: `datasets/rayworld/<inst>/edits/{edits.h5,edits.json,selection.json}`.
- **Deleted:**
  - the layout-v1 migration: `LAYOUT_VERSION`, `layout_file`, `read_layout`, `is_migrated`, `write_marker`, `_has_v1_files`, `ensure_marker`;
  - `EDITS_VERSIONS` and the `version` parameters;
  - `unused_dir` (no `_unused/` any more);
  - `parse_dataset_path`, `legacy_probe_key`, `legacy_edits_instance`.
- **Settings.** `DEFAULT_INSTANCE = {"rayworld": "standard", "othello": "standard"}`.
  - New `INSTANCES`: the SPEC instance names per environment. The path helpers still accept any name.
  - New `BASELINES`, plus a new helper `baselines_dir(env, inst)` → `runs/_baselines/<env>/<inst>/`.
- **Kept, with the same behavior:**
  - `REPO`, `DATASETS`, `CLASSES`, `RW_PROBE_SIZES`, `OTH_ROLE` (including `edits`);
  - `instance_root`, `train_dir`, `probe_dir`, `probe_file`, `probe_manifest`, `eval_dir`, `eval_file`, `eval_manifest`, `edits_dir`, `edits_file`, `edits_manifest`, `edits_selection`;
  - `othello_split_dir`, `othello_split_file`, `othello_cases_file(inst, n_cases=1000)`, `tokens_dir`;
  - `probe_key`, whose body is character-identical, so `("rayworld/<inst>", "probe_<size>")` is unchanged.

### `pim/scoring/**` (rewritten)
- **ND removed everywhere:**
  - the loops in `score_rayworld` and `score_rayworld_tokens`;
  - the `add_sub` / ND arm in `othello_arms`;
  - `PROBE_SOURCES["ND"]`, `EDITORS_SCORED`, `alphas.ND`, and the ND alpha settings.
- **Othello extra-target path removed:** the `mine_signed` block loop, `othello_blocks`, the baselines' Othello extra-target branch, and `oth_extra_targets` / `oth_reg_alpha_*`.
- **Research catch-up machinery removed:**
  - the env flags `PIM_ADD_CAT_IM`, `PIM_ADD_NN_R2`, `PIM_FORCE_RESCORE`, `PIM_ONLY_RUNS`, `PIM_SKIP_TOPICS`, `PIM_DW_BASES`;
  - the dated `scores_backup/` copies;
  - the `settings` dump, `commit_sha`, `blocks_added` and `inverse_added` records.
  - None of them is needed for a fresh score to produce every shipped block and arm.
- **"Incomplete" now means** (no flags): a current `scores.json` that lacks either
  - a block SETTINGS asks for, or
  - an IM arm on a block that must have one: every regression block, the Othello block, and each categorical block in the `rw_cat_im` scope.
  - Every shipped categorical block in that scope already has its IM arm, so nothing is owed. `nn_r2` is not required: it is written by every fresh score, but its absence does not make a run incomplete.
- **Versions** are all `"1.0"`: `EVAL_VERSION`, `EVAL_VERSION_BY_ENV = {"othello", "rayworld"}` (in `driver.py`, with `eval_version(r)`), `BASELINE_VERSION`, `IM_VERSION`, and `PRED_VERSION` (in `prediction.py`).
- **Repo-relative writes.** `probe_dir` is written as `runs/<id>/probes`. `scores.json` never receives an absolute path, a sha, a settings dump or a date; the shipped `minutes` field is kept.
- **Runs.** `scan_runs` walks `runs/<env>/<variant>/` explicitly (`iterdir`, which follows symlinked env dirs) and skips `_*` directories (`runs/_baselines`).
  - Row ids are `<env>/<variant>` (`r["id"]`), replacing `topic` / `run`.
  - The still-training guard is kept.
- **Replicates** take their parent's extra targets through `config.json` `replicate.of` (a run id), unless SETTINGS lists them explicitly (it does).
- **Baselines** are keyed by `(env, instance)` under `runs/_baselines/<env>/<inst>/`.
  - A new SETTINGS key, `rw_floor_targets`, names the extra targets that get floors: appearance-fac, appearance, pos@appearance. This is what makes the shipped files complete; the 8-ray grid targets never had floors and are not demanded.
  - The Rayworld body of `score_baselines_arch` now calls `score_baseline_bases` (the same per-basis loop `_rw_regression_floors` it inlined).
  - Othello floors are their own function.
- **Renames:** `best_arm` → `top_arm` in `blocks.py`, so it does not clash with `pim.metrics.selection.best_arm`; `_dw_*` → `_rw_*`.
- **Docstrings and comments** were rewritten to the SPEC rules. The core request is done: the Transformer-S comment in `baselines.py` is gone.

### `pim/environments/prediction.py`
- `PRED_VERSION = "1.0"`.
- The docstring was rewritten.
- `load_floor` was removed: it had no user, and its path was per-instance, not per `(env, instance)`.

### `pim/environments/__init__.py`
- The docstring was rewritten.

### `notebooks/master_eval.ipynb` (new, built minimal with Write; no outputs; kernelspec `python3`)
The cells are:
1. a title;
2. a scan cell that prints the run table;
3. ONE SETTINGS cell;
4. baselines, `score_all_baselines(RUNS, SETTINGS)`;
5. score, `score_all(RUNS, SETTINGS)`;
6. summaries, `print_summaries(RUNS)`;
with a short markdown line between cells.

SETTINGS holds:
- **Probe sizes:** `rw_probe_seqs` 30000, `oth_probe_games` 20000.
- **Bench:** `rw_bench_n` 1000, `rw_target` "full", `rw_edit_dims` ("all",).
- **Bases:** `rw_bases` ("frustum", "cartesian"), with frustum first; `rw_bases_by_instance` {"rayworld/obs5": ("cartesian",)}.
- **PI / GS alpha grids** for regression and categorical targets, Rayworld and Othello.
- **GS:** start points (0, 2, 4, 6, 8), 100 steps, beta 0.2.
- **`oth_gates_games`:** 10000.
- **`rw_extra_targets`:** the exact map from the export report, with all 12 N-ray replicates listed explicitly.
- **`rw_floor_targets`.**
- **`rw_cat_im`:** instances rayworld/{128,16,8,5}-ray; targets appearance-fac, appearance, grid-6x5, grid-10x3, grid-16x8.

All of these values equal what the shipped `scores.json` record: `check_settings.py` compares alphas, GS layers, n_seq, dims and target for all 43 runs and finds 0 mismatches. The bench n (1000) and the gates' n_games (10000) were checked separately against the stored values.

## API changes (in-scope signatures)
- **layout:**
  - `edits_file(cls, inst, *, n_cases=1000)`, `edits_manifest(cls, inst, *, n_cases=1000)`, `edits_selection(cls, inst)` and `edits_dir(cls, inst)` lost `version`. `n_cases` is keyword-only, so a stray positional `"v1"` raises `TypeError`.
  - `othello_cases_file(inst, n_cases=1000)` lost `version`.
  - `eval_manifest` raises for Othello.
  - The removed symbols are listed above.
  - Added: `INSTANCES`, `BASELINES`, `baselines_dir(env, inst)`.
- **`pim.scoring`:**
  - `scan_runs` rows carry `id` (no `topic` / `run`).
  - `score_all(runs, s, eval_version=eval_version, dry_run=False)`: `eval_version` is now optional.
  - `pim.scoring` also exports `eval_version` and `EVAL_VERSION_BY_ENV`.
  - `missing_inverse(r, prev, s)`: `s` is required.
  - `rw_block_setup` returns `(cat, dimsets, (PI alphas, GS alphas))`.
  - `probe_block(..., alphas=(pi, gs))`.
  - `othello_arms(..., alphas=(pi, gs), s)`.
  - `score_baselines_arch(runs, env, inst, arch, model_config, s)` and `score_baseline_targets(inst, arch, model_config, targets, s)` have reordered, env-keyed signatures.
  - `extra_targets_for` → `floor_targets_for(runs, env, inst, arch, s)`.
  - `blocks.best_arm` → `blocks.top_arm`; `blocks.instance_of` → `blocks.run_config`.
  - `pim.scoring.othello._probe_games(n, instance)` is unchanged, because three scripts import it.
- **`pim.environments.prediction`:** `load_floor` removed.

## Requests to other owners
All the Stage A layout requests from scripts / othello / rayworld are now satisfied on my side. A grep of RELEASE finds no remaining caller of any removed layout / scoring / prediction symbol. The scripts worker has already moved `generate_dataset.py`, `make_othello_edits.py`, `make_edit_selection.py`, `reachability_table.py` and `two_flip_editability.py` onto the flattened API and SETTINGS-matching constants. Remaining:
- **infra, `pyproject.toml`:** if it copies PRIVATE's config, drop the `"pim/scoring/*.py" = ["E702", "E731", "F541"]` per-file ignore and its comment. The rewritten package passes `ruff check --select E,W,F --ignore E501` without it.
- **scripts, optional:** `scripts/{bayes_floor,fit_probes,reachability_table,othello_flip_rates}.py` spell `REPO / "runs" / "_baselines" / env / inst`. `pim.environments.layout.baselines_dir(env, inst)` returns the same path; using it would keep a single place that builds that path.
- **tables, FYI:** `pim/figures/tables.py` already uses `layout.baselines_dir(env, instance)`. `layout.BASELINES.glob("*/*/baselines.json")` lists every baselines file.
- **core, FYI:** `pim/training/train.py` still writes `_commit_sha()` into a fresh run's `config.json`. The export strips it from the shipped configs; decide whether a fresh training run should record it.

## Verification (exact results)
- **(a) Dry runs on the STAGING-linked RELEASE tree.** `score_all(RUNS, S, dry_run=True)` and `score_all_baselines(RUNS, S, dry_run=True)`, with S exec'd from the notebook's SETTINGS cell:
  - 43/43 runs (27 Rayworld, 16 Othello) print `skip ... (scored at 1.0)`;
  - 12/12 baselines print `skip`;
  - there are 0 WOULD and 0 stale lines, and both to-do lists are `[]`.
  - Re-run on the final code: same result (`dry_run_staging_final.log`).
- **(b) Notebook execution** (nbconvert `--execute` of `notebooks/master_eval.ipynb`, output written to `work/scoring/master_eval_staging_executed_final.ipynb`, kernel cwd `notebooks/`, against STAGING):
  - exit 0, 0 error outputs;
  - the baselines cell prints 12 skips and the score cell 43 skips; 0 WOULD / stale / wrote lines;
  - the summaries render for all 43 runs;
  - run ids are the new names, so the RELEASE `pim` was imported.
  - STAGING is read-only, so any write attempt would have raised; none did. The RELEASE notebook is unchanged (6,220 bytes, no outputs).
- **(c) Othello from scratch**, in a throwaway tree with copies of STAGING `runs/othello/standard`, `runs/_baselines/othello/standard` and `datasets/othello/standard`; the notebook copy was executed in the tree.
  - **Run B (refit): `scores.json` and `probes/` deleted, every probe and inverse map refit.**
    - It took 14.1 min.
    - Bit-identical groups (max abs diff 0): gates (8 leaves), probe_skill (18), probe_stats (126), unedited (7), PI arms (1188), GS arms (300), best PI (11), best GS (10).
    - IM arms: max 8.34e-7 (131 of 216 leaves differ). IM-NN arms: max 9.23e-8. Best IM: 1.05e-7. Best IM-NN: 3.7e-8. inverse_map (g_r2, g_rmse, nn_r2): max 1.52e-10.
    - Overall, the 2,187 common numeric leaves have max abs diff **8.3e-7**.
    - The 10 refit probe files have exactly the shipped filenames, and they are **bitwise equal** to the shipped files (6,316 tensor / stat leaves, 0 differ).
  - **Run A (cache hits): `probes/` kept, `scores.json` deleted.**
    - It took 1.9 min and wrote no new probe file.
    - The per-group maxima are identical to run B's.
    - Run A and run B are identical to each other in every leaf except `minutes`.
    - IM itself was a cache hit, so the IM deltas are how the shipped IM arms were computed in PRIVATE (the fold-in path), not refit noise.
  - **Expected key differences (both runs):**
    - NEW adds the case-level spread fields (`*_case_sd`, `*_case_se`, `*_ci95_*`, `*_n_cases`, `fidelity_ci95_*`) to the PI / GS arms and `unedited`. The shipped standard file predates them; its replicates have them.
    - The shipped `gates.output_kind` / `out_sum_mean` / `out_neg_mass_mean` are not written: othello-env removed them.
    - The shipped `prediction` block is absent: it is written by `scripts/score_prediction.py`, not the scorer.
    - There are no non-numeric differences: bench text, rules, probe_dir and probe_sources are equal.
- **(c+) Rayworld checks** (throwaway trees; STAGING pieces symlinked read-only or copied):
  - **Floors.** Rebuilding `runs/_baselines/rayworld/8-ray/baselines.json` (both archs, 3 extra targets) from shipped caches: 899/899 leaves identical to STAGING, with no new probe files.
  - **Full rescore of `rayworld/obs5`**, cached probes:
    - PI (1728 leaves), GS (500), IM, probe skill / perdim / sanity and unedited are exact.
    - IM-NN: max 1.6e-6.
    - The fresh file adds the case-level fields and a `bench_selection` record. The shipped block has null there, yet every number matches, so the shipped numbers were computed on the same selected cases; only the metadata record was missing (PRIVATE's file is also null).
  - **Add-back on `rayworld/8-ray__seed0`** (appearance-fac block deleted, cartesian IM arms stripped):
    - the driver re-adds both;
    - the re-added cartesian IM / IM-NN match the shipped arms to 4.6e-8;
    - the appearance-fac IM, probe skill, sanity, bench selection and unedited card all match (unedited to 6e-8);
    - GS does not; see Open issues.
  - **Add-back on `rayworld/8-ray-tokens`:** the same pattern. The regression IM / IM-NN re-add is within 8e-8; the categorical GS differences are in Open issues.
- **(d) Imports and lint:**
  - all 11 owned modules import, with `pim.__file__` under RELEASE;
  - `ruff check --isolated --select F821,F822,F823,F401,F811,F841,E9` → "All checks passed!";
  - `--select E,W,F --ignore E501` → "All checks passed!";
  - nbformat validates the notebook: 10 cells, 0 outputs, `execution_count` null.
- **Writing rules.** Grep of the owned files for dates, names, discworld / dw-, research / harness / experiments references, banners, absolute paths, history words and British spellings: the only hits are false positives (`raise`, and the bench text "at a fixed 20-move prefix", which must match the shipped string).
- **Layout.** Every layout helper resolves to an existing STAGING file for all 12 instances: eval, edits, manifests, selection, probe_120k (+250k on the N-ray instances), Othello splits and cases, `tokens/vocab.npz`, `baselines.json`. `bench.cases_path("standard")` resolves to `datasets/othello/standard/edits/cases_1000.pkl`.
- **Cleanup.** `__pycache__` under RELEASE: none left. Every RELEASE run used `PYTHONDONTWRITEBYTECODE=1` after the first import check.

## Open issues
- **GS on categorical blocks does not reproduce the shipped values beyond about 1e-2.**
  - Edit Index and fidelity differ by up to 7.5e-3 / 3.9e-3 on 8-ray__seed0 appearance-fac, and 4.0e-3 / 3.3e-3 on the token model's appearance-fac. The largest difference is from GS start layer 0; from layer 8 it is 1e-4.
  - Top GS arm, 8-ray__seed0: 0.4626 / 0.5390 fresh against 0.4620 / 0.5408 shipped. Token model: 0.3514 / 0.4722 against 0.3493 / 0.4735. The chosen arm (point, alpha) is the same, and at two decimals the values are unchanged.
  - PI's `readout_landed` at alpha 0.5 differs too (up to 0.076). At alpha 0.5 the class-score swap stops halfway and leaves the two logits tied, so the argmax is decided by rounding. PI's Edit Index agrees to 8.5e-7.
  - **Cause, proven:**
    - RELEASE is bitwise deterministic run to run (two add-backs, 2,689 leaves, 0 differ).
    - RELEASE equals PRIVATE's own scorer run on this GPU on the same block, bitwise: 2,689 / 2,689 leaves (frames) and 2,820 / 2,820 (tokens). PRIVATE's ND arms ran first there, as they did originally.
    - Both differ from the shipped values identically. So the shipped categorical blocks come from a different compute environment (another host or GPU). The release code does not change them.
  - Regression blocks (Othello, obs5, the frustum / cartesian re-adds) reproduce exactly.
  - Anyone who rescores a categorical block should expect third-decimal GS shifts.
- **A full rescore drops the `prediction` block.** Run `scripts/score_prediction.py` afterwards. The notebook says so.
- **A from-scratch floor rebuild depends on what the bundle ships.** It needs `probe_250k` for the `_large` floors, which ships only for the N-ray instances. Categorical floors are cache-only (`scripts/fit_probes.py`). With the shipped `baselines.json` files nothing is refit.
