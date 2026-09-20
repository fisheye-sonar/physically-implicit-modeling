# Cut-over note — for the session operating the paper_ci queue (2026-09-19)

Branch **`master_eval_module`** (one commit on top of `c023401`), pushed to `origin`; the working
clone is `/home/sevan/research/PIM/pim-master-eval-refactor` (its `runs/` and `datasets/` are
symlinks into the live tree; nothing was written through them — see "What the gate touched").

## What changes on merge

| file | change |
|---|---|
| `notebooks/master_eval.ipynb` | 9 cells → 6; 1,251 code lines → 174. Cell [2] (SETTINGS, `EVAL_VERSION*`, `eval_version`) is **byte-identical** (sha256 of the cell source `f77f03837515e054` before and after; the 16-ray `dw_extra_targets` entry is in it). Cells [3]–[5] are one call each: `score_all_baselines(RUNS, SETTINGS)`, `score_all(RUNS, SETTINGS, eval_version)`, `print_summaries(RUNS)` |
| `pim/scoring/{__init__,runs,blocks,discworld,othello,baselines,driver,summary}.py` | NEW — the cells' code, assembled by slicing the cell text. 33 functions moved: 16 byte-identical, 17 differ only by `s=SETTINGS` → `s` in the signature and a threaded `s` / `runs` argument (21 changed lines). The two loops gained a `dry_run=False` parameter and nothing else |
| `tests/test_scoring_package.py` | NEW — 6 CPU tests of the queue's contracts |
| `experiments/master_eval_refactor/` | NEW — the gate (scripts, tracked results in `scores/`, README, this note) |
| `research/REGISTRY.md` | §Evaluation gains the module map; the owed Othello guard-CI line now points at `pim/scoring/othello.py::othello_arms` |
| `CLAUDE.md`, `pim/__init__.py` | the package list names `scoring/` |
| `pyproject.toml` | ruff per-file ignore for three STYLE codes in the verbatim-moved lines (lint count on `pim tests` unchanged: 47 before, 47 after, none in the new files) |

**Not changed:** `scripts/drivers/*`, `experiments/paper_ci/*`, every `EVAL_VERSION`, `BASELINE_VERSION`,
`IM_VERSION`, the `scores.json` schema, `best_arm` (unguarded max; Othello's `best` keyed on the union
index), `pim/figures/tables.py`. Entry point unchanged: `nbconvert --execute --inplace
notebooks/master_eval.ipynb`. The three environment hooks are honoured where they were:
`PIM_ONLY_RUNS` / `PIM_SKIP_TOPICS` in `pim/scoring/runs.py::scan_runs` (moved verbatim, unit-tested),
`PIM_DW_BASES` in notebook cell [2] (untouched).

## Two facts about the hosts that bear on timing

- The dispatcher's `push_inputs` sends only `experiments/paper_ci/{config.json,scripts/,queue/<id>.json}`
  and the job's `runs/…` inputs — never `notebooks/` or `pim/`. So on the 4090 the notebook and
  `pim/scoring/` arrive TOGETHER by `git pull` and cannot be skewed by a launch.
- The live tree strips notebook outputs with an `nbstripout` clean filter (local git config), which is
  why an executed `master_eval.ipynb` shows as unmodified; a merge replaces it cleanly. If the 4090's
  checkout has no such filter its executed notebook WILL show as modified: `git checkout
  notebooks/master_eval.ipynb` there before pulling (between executions, as you planned).

## The gate's results

See `README.md` §Results and `scores/`:
`decisions_check.json`, `<run>.diff.txt` for each fixture, `noop_check.json`.

## What the gate touched

Reads only under `runs/` and `datasets/`. Each fixture's `probes/` listing was compared before and
after scoring (reported in its `.diff.txt`); the live `scores.json` / `baselines.json` mtimes were
checked around the end-to-end `nbconvert` run. Scratch memmaps went to the clone's own `.scratch/`.
GPU: ~35 min alongside `rep_dw-5ray_s2`'s training, under a 30 GB unit (`me_refactor_gate`, peak < 8 GB).

## After cut-over — what I would check on the first scoring pass per host

1. The executed notebook's cell [4] output lists `skip … (scored at …)` for every current run and
   exactly ONE `=== scoring <new replicate> ===` (+ the seed-0 member if it was laid out) — no `stale`.
2. The same seven `adding blocks […] → nothing added (probes not fitted yet)` lines as before the change
   (3 × dw-noiseless members, 2 × dw-5ray, 2 × dw-8ray; listed in `scores/decisions_check.json`). More
   than those = something is off; fewer = a probe set got fitted.
3. The new member's `scores.json` has `bases` = frustum, cartesian, appearance-fac (+ IM / IM-NN arms in each),
   and its numbers sit beside its siblings' in the ledger.

Rollback: `git revert <merge>` restores the 9-cell notebook; `pim/scoring/` becomes dead code and is harmless.

## Owed afterwards (not part of this move)

- The two one-line edits now possible: `nn_r2` in `pim/scoring/blocks.py::attach_inverse` (+ the Othello
  `inverse_map` dicts in `othello.py::score_othello` / `driver.py::add_inverse`), and the guard's CI in
  `pim/scoring/othello.py::othello_arms`.
- After the queue drains: one arm-selection rule (`best_arm` ≡ `tables._best_by`, `guard=` parameter,
  Othello keyed on symdiff), and the style tidy that removes the ruff per-file ignore.
- Noticed, not touched: `score_pending.sh` greps `logs/<name>/master_eval.log` for
  `wrote|adding blocks|SKIPPED` to build its completion ping, but nbconvert sends cell stdout into the
  NOTEBOOK's outputs, not that log (it holds two `[NbConvertApp]` lines) — so that part of the ping has
  always been empty. Unchanged by this move.
- Cell [2]'s header comment still says "nothing else is recomputed — cell [6]"; it is now cell [4]. Left
  alone to keep the cell byte-identical through the cut-over.
