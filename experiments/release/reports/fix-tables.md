# Worker report: fix-tables

**Result: all 8 items are applied.** Every check passes, and no shipped number changed.
- **Tables:** the table gates give the same results as before, except for the two checks whose prints were removed.
- **Figures:** all 31 outputs (PNG, PDF and JSON) are byte-identical to the figures-recheck treeC renders.
- **Dry runs:** they skip everything on the STAGING-linked release.

Work dir: `experiments/release/work/fix-tables/`.
- `before/`: a snapshot of RELEASE code taken at the start.
- `removed/pim/figures/waterfall.py`: the moved-out module.
- `gate/`: the gate copies and their results.
- `nbtree2/`, `executed2/`: the notebook runs.
- `figtree2/`: the figure run.
- `mock_runs/`: the mock `im_reconstruction.json`.
- `attrcheck.py`: a check that every `module.attr` used exists.

## Changes (RELEASE)

1. **Waterfall module removed.**
   - `scripts/figures/style.py` now defines the palette itself: `DARK_BG` (= `viz.BG_HEX`), `EDIT_LINE`, `TARGET_C`, `GHOST_C`, and `DIFF_CMAP` (same name `pim_diff`, same colors). It no longer imports `pim.figures.waterfall`, which was the only import from that module.
   - `pim/figures/waterfall.py` was moved to `work/fix-tables/removed/pim/figures/`.
   - `pim/figures/__init__.py` is now just `"""The paper's tables (``tables``)."""`.
   - The `viz.py` comment has already been changed by its owner ("the dark palette the demos and figures share"), so no request is needed.
2. **`pim/scoring/summary.py`:**
   - The column header `fid` is now `ratio`.
   - The docstring says the arm shown is the top arm before the fidelity cutoff, with its fidelity ratio (1 minus Edit Fidelity).
   - `master_eval` Summaries markdown: "Each editor's top arm before the fidelity cutoff, with its fidelity ratio (`ratio`, 1 minus Edit Fidelity). The table notebooks apply the paper's selection rule."
3. **`pim/scoring/driver.py`:**
   - Deleted `_has_im`, `missing_inverse`, `add_inverse` and the missing-IM branch of `_complete`, plus their now-unused imports. `missing_blocks` is kept.
   - These functions only backfilled IM: `score_rayworld` / `score_othello` compute IM whenever they score a block, including under `only=`.
   - Updated the module and `score_all` docstrings, and the `master_eval` title and Score markdown ("lacks a block gets only that block added").
4. **`blocks.attach_inverse`:** writes `nn_r2` only when it is present and every value is finite.
   - Categorical blocks now get no `nn_r2` key, like the shipped `appearance-fac` blocks.
   - Regression blocks keep it, in the same key order.
5. **Categorical probe recipe from SETTINGS.**
   - New `blocks.rw_probe_recipe(target, instance, s)` calls `rwa.probe_recipe(target, instance, n_seq=s["rw_probe_seqs"], cat_n_seq=s["rw_cat_probe_seqs"], cat_epochs=s["rw_cat_probe_epochs"])`.
   - It is used at all four recipe sites in `scoring/rayworld.py` and in `baselines._rw_categorical_floors`, which now takes `s`.
   - fix-lib's keyword arguments have landed and I re-tested against them.
   - `master_eval` SETTINGS gains `"rw_cat_probe_seqs": 200_000` and `"rw_cat_probe_epochs": 50`, and `GS_LAYERS` is commented `# GS start points (residual points)`. The Settings markdown says to fit categorical probes with `fit_probes.py` at those sizes (its defaults).
6. **`pim/figures/tables.py` and the table notebooks.**
   - (a) Removed prints:
     - `tokens_numbers` no longer returns "categorical cases kept".
     - `two_flip_numbers` no longer returns `partners_searched`. It keeps `n_cases` and the max SE, now as integer and float columns via `from_dict`.
   - (a) Appendix headers:
     - steps-md: "IM's steps as the rays coarsen (in units of the combined seed SD of the two seed means; each at least six)".
     - tokens-numbers-md: "Trained probes within 0.013 of the frame model".
     - "lands" is defined ("An editor lands when its seed-mean Edit Index is at least 0.25.") in `paper_tables` spread-md and `appendix` maxima-md.
   - (b) New `tables.flip_rates(runs)`:
     - It reads `runs/_baselines/othello/<inst>/corpus_stats.json` through `layout.baselines_dir`, using the fields `flips_per_move`, `flips_per_game`, `n_games` and `n_moves`.
     - It has a `paper_tables` cell right after the run lists, with `FLIP_RATES = ["othello/standard", "othello/adjacent-flip"]`, which matches `othello_flip_rates.py`'s default instances.
     - Output: standard 2.2449, adjacent-flip 0.2687.
   - (b) New `tables.pi_landing(F, alpha=1.0)`:
     - It returns a `Table` (`.values` / `.text` / figure; format `.1e`), because pandas printed about 1e-6 values inconsistently.
     - It shows `readout_err_after` of the PI α = 1 arm per Rayworld run and residual point.
     - It has a `paper_tables` cell after the seed spread.
     - Point 0 is 1.6 to 2.6. Points 1 to 8 are 8e-7 to 4.5e-6 (the 5-ray maximum at point 1).
   - (c) New `tables.im_reconstruction(run="othello/standard")`:
     - It reads `runs/<run>/im_reconstruction.json` in the given format and returns `g error` / `model error` / `ratio` indexed by point.
     - It has an appendix cell after `tab:im_by_point`.
     - I tested it on a mock file with that format, built from the private reconstruction ratios: 9.5, 9.4, 8.0, 5.4, 1.95, 1.80, 6.2, 10.1, 15.7, which matches the paper's sentence.
     - The real file is written by fix-scripts' `scripts/im_reconstruction.py`, whose output format matches the reader.
   - (d) Both table-notebook titles say that each table object's `.values` and `.text` give its numbers as a DataFrame and as printed text. The appendix title also lists `im_reconstruction.json`, and the `paper_tables` title mentions the flip rates.
7. **`scripts/figures/editability_by_point.py`:** `main()` parses `argparse.ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)`, so `--help` prints usage and exits.
8. **`layout.py`:** `baselines_dir(env, inst)` and a single `tokens_dir(inst)` exist. No change was needed.
   - My files use them: `baselines.py` (`layout.baselines_dir`, `layout.tokens_dir`), `tables.py` (`layout.baselines_dir`), and `prediction.py` (`layout.eval_file`).
   - The scripts' hand-spelled `runs/_baselines` paths are already fixed by their owner. What remains is in `rayworld/tokens.py` (request 1).

**Extra fix needed to keep my files working:** another owner renamed `othello.corpus.LADDER["D"]` to `N_TRAIN_GAMES`.
- `pim/scoring/othello.py` (2 sites, including `_probe_games`, which scoring and four scripts use) and `scripts/figures/qualitative_othello.py` (1 site) still used the old name, and `qualitative_othello.py` / `qualitative_overview.py` crashed.
- I switched them to `oc.N_TRAIN_GAMES`. It is the same value, 20,000,000.

## Verification

- **ruff:** `ruff check --no-cache pim scripts notebooks` gives "All checks passed!".
- **attrcheck:** `attrcheck.py` over all my files finds 0 missing module attributes. It does catch the old `LADDER` uses in `before/`.
- **Dry runs** (RELEASE root, STAGING links, SETTINGS taken from the notebook cell): `score_all` gives 43/43 skip and `[]`; `score_all_baselines` gives 12/12 skip and `[]`.
- **Add path:** the dry run of `_complete` with a block dropped in memory reports `WOULD add to rayworld/8-ray: blocks ['grid-6x5']`.
- **Recipes:** `rw_probe_recipe` reproduces the stored `probe_recipe` of all 98 shipped Rayworld blocks, and the categorical IM's `n_seq` / `epochs`.
- **Cache keys:** recomputing the `8-ray` `appearance-fac` floors through `score_baseline_targets` (cached probes only) is identical to the shipped `baselines.json` block. Loading the `appearance-fac` / `grid-16x8` probes through `_fit` reproduces the shipped `probe_skill_linear` / `mlp` exactly.
- **`attach_inverse` unit test:** a finite list writes `nn_r2`; all-NaN, partly NaN or absent omits it; the `best` / `best_by_dims` entries are unchanged.
- **Notebooks** (fresh copies in `nbtree2`, STAGING links plus the mock `im_reconstruction.json`): all three execute with rc 0 and no error or stderr output. `master_eval` gives 12/12 baseline skips and 43/43 run skips; both return `[]`.
- **Table gate:**
  - I ran `work/tables/check_tables.py` itself from the RELEASE root. As expected, it stops at its removed-print check (`KeyError: 'unreachable'`) before writing anything.
  - An adapted copy (`gate/check_tables_after.py`) drops only the two removed prints (partners searched, 49 removed cases). Run from the RELEASE root, it gives 1165 / 855 / 29 checked, with reference 0, paper 2 (the accepted `tab:im_by_point` cells) and numbers 1 (the accepted 0.106).
  - That is identical to the untouched gate on the `before/` snapshot, except those two entries.
- **Verify-tables tool** (`work/tables/verifier_v1/verify_tables.py`, `verify_numbers.py`, copied into `gate/vt_*`): the result JSONs are byte-identical before and after (871 cells, text_mismatch 2, raw_mismatch 2, text_vs_raw 0).
- **Figures:** all six figure scripts ran in a fresh tree (`figtree2`, current RELEASE code, STAGING links), each with its own cache. All 31 outputs are byte-identical to `work/figures-recheck/treeC/outputs/figures`, with 0 differing pixels in all 15 PNGs. `figtree2` still matched RELEASE code at the end (`diff -rq` empty).
- **Nothing else was written:** no `__pycache__`, `outputs/` or checkpoints in RELEASE; `.ruff_cache` mtime is unchanged; no STAGING file is newer than my start marker.

**Rule slip:** I ran `rm -rf <figtree1>/outputs/cache` in the loop of my first figure run. It touched only a cache inside my own throwaway tree `work/fix-tables/figtree1`. Nothing in RELEASE, STAGING or PRIVATE was affected. Later runs used `mv` into `work/fix-tables/removed/` instead.

## Requests

1. **fix-lib** (`pim/environments/rayworld/tokens.py`):
   - Lines 82–83 define a second `tokens_dir(instance_dir)`. Delete it, and in `tokenize_instance` use `layout.tokens_dir(Path(instance_dir).name)`.
   - Lines 104 and 106 spell `train/corpus.json` and `train/obs.f32` by hand. Use `layout.train_dir("rayworld", inst) / "corpus.json"` and `/ "obs.f32"`.
2. **export:** ship `runs/othello/standard/im_reconstruction.json` (core bundle; MANIFEST / SHA256SUMS) once fix-scripts has written it. Until then, `appendix_tables.ipynb` stops at the reconstruction cell.
3. **infra (`README.md`):**
   - Step 5: add `python scripts/im_reconstruction.py`.
   - Lines 188–189: the flip rates are read by the main-text notebook (`paper_tables.ipynb`), and the reconstruction test by the appendix notebook.
4. **Whoever reruns the table gate:** `work/tables/check_tables.py` still checks the two removed prints. Use `work/fix-tables/gate/check_tables_after.py` (same checks minus those two), or drop them there.

## Open issues

- `layout.py` still has the unreferenced `probe_dir`, `eval_dir`, `othello_split_file` and `INSTANCES`, and error messages that point to `othello_split_file` (static finding on `pinv.py:41`). This was not in my list, so I left them.
- The appendix reconstruction cell has only been exercised on the mock file. It should be re-executed once the real `im_reconstruction.json` is in STAGING.
