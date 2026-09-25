# Worker report: scoring-settings

**Task.** The export now ships `bases["appearance-fac"]` for the main runs `rayworld/standard` and `rayworld/blink`. Make the scorer's SETTINGS ask for exactly that block:
- on those two run ids only;
- not on their replicates;
- no IM arm;
- no floors, unless the shipped baselines carry them.

Then re-verify the no-op dry runs and the notebook.

**Status.** Done.
- The SETTINGS cell now asks for the block on those two runs.
- Two small changes in `pim/scoring/` were needed. Without them, the SETTINGS entry alone made the scorer want 8 new things on the shipped bundle.
- Both dry runs on the staging-linked release tree skip everything: 43/43 runs and 12/12 baselines.
- The notebook executes with 0 errors and 0 write attempts.
- Scoring the block again in a throwaway tree gives the shipped block exactly (details below).

Helpers, logs and results are in `experiments/release/work/scoring-settings/`.

## Floors: checked
The shipped `runs/_baselines/rayworld/{standard,blink}/baselines.json` hold only `cartesian` and `frustum` for `transformer_l`.
- Their `probes/` folders (20 files each) are all `target: full`.
- So there are no appearance-fac floors for Standard or Blink, and the scorer must not ask for them.

## Why the code had to change
With only the SETTINGS entry added, a dry run on the shipped bundle printed 8 things the scorer would do (`dry_run_before_code_change.log`):
- **Replicates.** `WOULD add ... blocks ['appearance-fac']` on `rayworld/{standard,blink}__seed{0,1,2}`. `extra_targets_of` fell back to the parent's entry through `config.json` `replicate.of`.
- **Floors.** `WOULD fit baselines rayworld/{standard,blink}: extra targets {'transformer_l': ['appearance-fac']}`. `rw_floor_targets` was a flat tuple of targets, applied to every instance whose runs list that target.

## Changes

**`pim/scoring/blocks.py`**
- `extra_targets_of(run_id, s)` now returns only the run's own `rw_extra_targets` entry. A replicate no longer inherits its parent's targets.
  - SETTINGS already listed all 12 N-ray replicates under their own ids, so no shipped run changes.
- New `in_scope(scope, instance, target)` checks a SETTINGS scope of the form `{"instances": (...), "targets": (...)}`.
  - `cat_inverse_in_scope` now calls it. Its behavior and signature are unchanged.

**`pim/scoring/baselines.py`**
- `floor_targets_for` now keeps an extra target only if `in_scope(s["rw_floor_targets"], inst, t)` is true.

**`notebooks/master_eval.ipynb`** (SETTINGS cell only, edited with NotebookEdit)
- New constant `N_RAY = ("rayworld/128-ray", "rayworld/16-ray", "rayworld/8-ray", "rayworld/5-ray")`.
- `rw_extra_targets` gains `"rayworld/standard": APP_FAC, "rayworld/blink": APP_FAC`, with the comment "for the qualitative figures; no floors, no IM".
- The header comment now says that a seed replicate is listed under its own id.
- `rw_floor_targets` changes from a tuple to `{"instances": N_RAY, "targets": ("appearance-fac", "appearance", "pos@appearance")}`.
- `rw_cat_im` now uses `N_RAY`, with the same values as before. Standard and Blink stay out of it.
- The notebook still has no outputs, `execution_count` null and kernelspec `python3`. It passes `nbformat.validate`. No other cell's source changed.

**API changes to note**
- `rw_floor_targets` is now a scope dict, not a tuple. Its only reader is `floor_targets_for`.
- Replicates no longer inherit extra targets. This supersedes the `scoring.md` line "Replicates take their parent's extra targets through `config.json` `replicate.of`".

## Verification
1. **What the scorer asks for, old logic vs new** (`check_scope.py`).
   - The previous functions were rebuilt in the script and run on the previous SETTINGS; the new code ran on the new SETTINGS. All 27 Rayworld runs were compared.
   - Only `rayworld/standard` and `rayworld/blink` change: `[frustum, cartesian]` becomes `[frustum, cartesian, appearance-fac]`. Their IM list stays `[]` before and after.
   - For every Rayworld run, the new list of blocks equals the blocks in the shipped `scores.json`.
   - The floor-target sets are unchanged for all 13 (instance, arch) pairs: 128/16/5-ray get `[appearance-fac]`; 8-ray `transformer_l` gets `[appearance-fac, appearance, pos@appearance]`; 8-ray tokens gets `[appearance-fac]`; Standard, Blink, smooth, obs5 and every Othello pair get `[]`.
2. **Dry runs on the staging-linked release tree** (`dry_run_staging.log`; `pim` imported from RELEASE).
   - `score_all(..., dry_run=True)`: 43/43 runs print `skip ... (scored at 1.0)`.
   - `score_all_baselines(..., dry_run=True)`: 12/12 baselines print `skip`.
   - There are 0 WOULD, stale, wrote or error lines, and both to-do lists are `[]`.
3. **The scorer adds the block when it is missing** (throwaway tree; STAGING datasets and baselines linked read-only; run probes linked file by file).
   - The appearance-fac block was deleted from the Standard and Blink `scores.json`.
   - The dry run printed `WOULD add to rayworld/{standard,blink}: blocks ['appearance-fac']  IM on []`. `standard__seed0` and both baselines printed `skip`.
   - A real `score_all` then added the block back on each run:
     - every probe load hit the cache, and no probe file was written;
     - the scorer attached no IM arm and no `inverse_map`;
     - a follow-up dry run skipped both runs.
   - Leaf-by-leaf against the shipped block (`compare_{standard,blink}_appfac.txt`), with 0 differences on both runs:
     - PI arms: 1512 leaves; GS arms: 360;
     - Probe Skill, per-dim values and sanity: 84;
     - the unedited scorecard, `best`, `best_by_dims`, alphas, `probe_recipe`, `bench_selection`, `n_classes`, kind, target and basis.
     - The top arms match: Standard PI pt1 α100 +0.0074 / 1.935 and GS@L0 α0.35 +0.3258 / 0.706; Blink PI pt4 α100 +0.0059 / 2.259 and GS@L0 α1.5 +0.2599 / 1.148.
     - The only difference is the per-case spread fields (`*_case_sd/se`, `*_ci95_*`, `*_n_cases`), which a fresh score always adds. `scoring.md` reports the same for Othello and obs5.
     - GS matches exactly here, unlike the 8-ray and tokens categorical blocks in `scoring.md`. These two blocks were probably computed on this machine.
4. **Notebook** (`master_eval_staging_executed.ipynb`).
   - It was executed with nbconvert from `RELEASE/notebooks`, with the output written to the work dir.
   - Exit code 0, 0 error outputs, 0 stderr.
   - The baselines cell prints 12 skips and returns `[]`. The score cell prints 43 skips and returns `[]`. There are 0 WOULD, stale, wrote or `===` lines.
   - The summaries show the Standard appearance-fac block.
   - The RELEASE notebook's sha256 is the same before and after.
   - STAGING is read-only and no write was attempted: no STAGING file is newer than the start of this task.
5. **Lint.** `ruff check --isolated --select E,W,F --ignore E501 pim/scoring` → all checks passed.
6. **Cleanup.** 0 `__pycache__` under RELEASE (every run used `PYTHONDONTWRITEBYTECODE=1`). The throwaway tree is deleted; the scores it re-added are in `work/scoring-settings/results/`.

## Requests
- **scripts (the `stage_b_commands.md` / README command list) — needed to rebuild these two blocks from scratch.**
  - The scorer reads categorical probes from the cache only, and `probe_recipe("appearance-fac", inst)` uses `probe_250k`.
  - A from-scratch rebuild of the Standard and Blink blocks therefore needs, before `master_eval`:
    ```
    python scripts/generate_dataset.py --instance I --role probe --size 250k   # also for standard and blink
    python scripts/fit_probes.py --run rayworld/I --target appearance-fac      # I in standard blink (no --random-init / --observation)
    ```
  - Without them the scorer prints `appearance-fac: SKIPPED` and adds nothing; it does not crash.
  - The 250k corpus for Standard and Blink is also what the shipped `_large` observation floors of those instances were fitted on.
- **tables — FYI.** Nothing new from this change. The tables keep the Standard and Blink categorical rows out through `CAT_RUNS` (`export-fix.md` request 1), which does not depend on SETTINGS.

## Open issues
- None found in the scoring path.
