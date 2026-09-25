# Worker report: tables

Owned: RELEASE `pim/figures/**`, `notebooks/paper_tables.ipynb`, `notebooks/appendix_tables.ipynb`,
`scripts/figures/editability_by_point.py`. Helpers and outputs: `experiments/release/work/tables/`
(`check_tables.py`, `check_results.json`, `executed/`, `tree/` with the rendered figure).

**Result.** Every paper table is rebuilt from the shipped artifacts by one function. All 1165 compared values equal
`reference_tables.json` exactly; the only deltas are the two expected `dropped_steps` changes. 853 of the 855 printed
cells equal the paper. The 2 that differ are stale cells in the paper (see "Paper corrections"). 28 of the 29 in-text
numbers match; the 29th is a rounding overstatement in the paper.

## What changed and why

- **`pim/figures/tables.py`: rewritten, 958 → 885 lines.**
  - **Run layout.** Run ids are `<env>/<variant>`, read from `runs/<id>/scores.json`. Replicates are found at
    `runs/<env>/<variant>__seed*/` and pooled by `pim.metrics.replicates.pool_replicates`, with the budget taken
    from each member's `config.json` `replicate.steps`.
  - **Floors.** Decodability floors and Bayes floors are read from `layout.baselines_dir(env, instance)`, so they
    are keyed by (env, instance).
  - **Blocks.** Rayworld tables read only `REPORTED_BLOCKS`: cartesian, appearance-fac, appearance, grid-6x5,
    grid-10x3, grid-16x8 and pos@appearance. frustum is never read. The basis switch (`set_basis`, `REG_BASES`,
    `BASIS_BY_INSTANCE`, the star and blank rows), ND, `ARCH_LABEL` and its S/R branches are gone. So are the
    per-component tables, the training-curve figure, the overfit panel, `table_arms` and `image_table`.
  - **Rendering.** seaborn is dropped. One small renderer (`draw`) draws every table in the paper's layout: section
    and variant labels, a two-level header, and each cell heat-colored on its column's scale (skill Greens 0..1,
    index RdYlGn −1..1, fidelity RdYlGn centered at 0, SD Oranges, excess YlOrRd).
  - **Cell text.** Cells are printed exactly as the paper rounds them, so the check compares strings. A rounded
    zero prints as `0.00` / `+0.00`.
  - **Return value.** Each `table_*` returns a `Table`: `.values` holds the unrounded numbers plus run/block,
    `.text` the printed strings, `.fig` the figure. Its `_repr_png_` makes a notebook cell show the figure.
  - **Selection.** Every arm comes from `pim.metrics.selection`: `best_arm`, `best_arm_by_fidelity` or
    `arms_of`. The mean-frame rows run `best_arm` on arms whose `fidelity_ratio` is replaced by
    `fidelity_ratio_expected`, with key `zone_edit_index_expected`. No new metric math.
- **One function per paper table:**
  - main text: `table_decodability`, `table_editability` (daggers where the seed SD > 0.1);
  - appendix: `table_predictive_skill`, `table_seed_spread`, `table_legal_illegal`, `table_two_flip`,
    `table_tokens_decodability`, `table_tokens_editability` (frame-set* and mean-frame rows),
    `table_additional_rw` (two panels), `table_im_by_point`, `table_categorical`, `table_im_vs_nn`,
    `table_fidelity_selected`.
- **In-text number functions:**
  - `im_vs_nn_gain`;
  - `seed_sd_maxima`, which counts an editor as landing when its seed-mean Edit Index is ≥ `LANDS` = 0.25;
  - `gap_closed_min`, `ci_multiplier`;
  - `im_steps`: step / √(s1²+s2²) and step / the larger SD;
  - `dagger_cells`, `replicate_arms`, `seed_means_vs_main`, `fixed_setting_check`;
  - `probe_refit_spread`: from `variance.json`, using the linear probe's `best_skill.sd` and
    `PI@canonical_edit_best` Edit Index SD, and the inverse map's `IM` Edit Index SD;
  - `reachability_counts`, `two_flip_numbers`, `tokens_numbers`, `fidelity_rule_shift`.
  - `by_point` feeds both `tab:im_by_point` and the figure.
- **`pim/figures/waterfall.py`, 229 → 96 lines.**
  - Kept: `waterfall_grid` with the arguments the history-rewriting script passes (`columns`, `context`, `gt`,
    `title`, `sample_idx`, `target_x`, `ghost_x`, `metrics`, `metric_label`, `gt_label`), and the constants the
    figures import (`DARK_BG`, `DARK_TEXT`, `DIFF_CMAP`, `EDIT_LINE`, `GHOST_C`, `TARGET_C`).
  - Removed, because nothing in the release passes them: `diff_columns`, `diff_scale`, `leads_by_one`, `vmin`,
    `vmax`, `col_width` and `row_height` (the last two are now module constants).
  - Docs rewritten: no spec reference, no dates, no history. There were no scaling panels in this file.
- **Deleted:** `probe_capacity.py`, `probe_targets.py` and `theme.py`. `style_ax` was used only by the by-point
  figure, which now styles its axes itself.
- **`pim/figures/__init__.py`** exports only `waterfall_grid`. `tables` is imported as a module.
- **Notebooks.** Both were written as fresh nbformat-4.5 JSON, with kernelspec `python3`, no outputs and
  execution counts of None.
  - Layout: a title, one cell with the hard-coded run ids, then one call per cell under short headers that name
    the paper table or number.
  - `paper_tables.ipynb` has Table 1, Table 2, the 53.6% gain and the seed-SD maxima.
  - `appendix_tables.ipynb` has every appendix table in paper order, each followed by its numbers.
- **`scripts/figures/editability_by_point.py`.**
  - A simplified port of `paper/figs/editability_trends/by_point.py`, reading values through `tables.by_point`.
  - It imports the figures worker's `scripts/figures/style.py` for rcParams, `TEXT_WIDTH_IN`, `TEXT` and `save`,
    and writes `outputs/figures/editability_over_res_point.{pdf,png}`.
  - The hollow-marker legend now reads "Edit Fidelity < 0", the caption's wording.

## API changes (the old API was private to the table notebooks; nothing else in RELEASE imports it)

- `collect(runs_oth, runs_rw, label, *, pool_budgets, budget_tolerance, select)` becomes
  `collect(runs, select="index", *, pool_budgets=False, budget_tolerance=0.10) -> Frames`.
  - `runs` is one list of release ids; the env comes from each run's `scores.json`.
  - `Frames` has `df` (column `block`, which replaces `basis`), `rep_sd` keyed by (run, block), `reps`,
    `select` and `runs`.
- Table functions take `Frames` or run lists and return `Table`, not a `Figure` or None.
- Removed: `set_basis`, `reg_key`, `basis_star`, `find_run`, `SHOW_TITLES`, `ARM_GUARD`, `ARM_SELECT`, `CANONICAL`,
  `REG_BASES`, `BASIS_BY_INSTANCE`, `ARCH_LABEL`, `COMPONENTS`, the `t975` / `ci95_halfwidth` re-exports, the
  `pool_replicates` wrapper, `table_inverse_r2` (folded into `table_decodability` and `table_im_vs_nn`),
  `tables_components`, `table_arms`, `table_gridified` (now `table_categorical` plus the categorical rows of
  Table 2), `table_prediction` / `prediction_rows` (now `table_predictive_skill` / `prediction_values`),
  `table_seed_variance` (now `table_seed_spread`), `fig_training_curve` and `image_table`.
- `pim.figures` no longer exports `PALETTE`, `style_ax` or `sweep_figure`.
- `waterfall_grid` loses the keyword arguments listed above.

## Requests to other owners

- **figures** (`scripts/figures/**`):
  - `waterfall_grid` no longer accepts `diff_columns`, `diff_scale`, `leads_by_one`, `vmin`, `vmax`, `col_width`
    or `row_height`. The history-rewriting port must not pass them. PRIVATE's `make_figure.py` passes none of them.
  - Optional: add `EDITOR_COLORS = {"PI": "#0072B2", "GS": "#D55E00", "IM": "#009E73"}` to `style.py`.
    `editability_by_point.py` defines the same dict locally and can switch to `st.EDITOR_COLORS`.
  - `editability_by_point.py` relies on `style.TEXT`, `style.TEXT_WIDTH_IN` and `style.save(fig, rel_stem)`.
    Keep those names.
- **infra** (`pyproject.toml`, `.gitignore`):
  - `pim.figures.tables` needs `pandas`, `matplotlib` and `h5py`, but no longer `seaborn`. Drop seaborn unless
    another owner imports it.
  - The notebooks need `jupyter` / `nbconvert`.
  - `outputs/` must stay gitignored.

## Verification (exact results)

**a. Notebook execution.**
- Command: `nbconvert --to notebook --execute` on RELEASE `notebooks/{paper_tables,appendix_tables}.ipynb`, run
  against the STAGING-linked RELEASE, with output to `work/tables/executed/*.executed.ipynb`.
- Both exited 0.
  - paper: 6 code cells, 0 error or stderr cells, 2 table images.
  - appendix: 30 code cells, 0 error or stderr cells, 11 table images.
- Written with `PYTHONDONTWRITEBYTECODE=1`. Nothing was written into RELEASE; no `outputs/` directory was created.

**b1. Table diff against `reference_tables.json`** (`check_tables.py`, exact equality, NaN-aware).
- **1165 values, 0 mismatches.** Per table:

  | table | values compared | reference frame |
  |---|---|---|
  | tab:decodability | 80 (including floors) | paper |
  | tab:editability | 98 | paper |
  | tab:editability daggers | 84 | paper `rep_sd` > 0.1 |
  | tab:im_vs_nn | 60 | paper |
  | tab:categorical | 56 | paper |
  | tab:fidelity_selected | 98 | `appendix-A2 (select=fidelity)` |
  | tab:tokens_decodability | 32 | `tokens @ cartesian` |
  | tab:tokens_editability | 28 | `tokens @ cartesian` (the mean-frame rows have no reference row) |
  | tab:additional_rw | 24 + 21 | `additional_rw @ cartesian` |
  | tab:seed_spread | 476 | every SD, mean, member value, n, steps and seeds |
  | tab:predictive_skill | 108 | `prediction_rows` |

- **Expected deltas.**
  - `dropped_steps` changes from [421875] to [] for `othello/adjacent-flip` and `rayworld/standard`.
  - ND columns are not compared (gone).
  - There is no `rep_sd` for 8-ray pos@appearance; that row shows no ± and none is drawn.
- **Spread coverage:** 14 shipped reference spreads and 14 release spreads, with none missing and none extra.

**b2. Table diff against `paper/paper_draft.tex`.**
- Method: the tabulars behind each `\label{tab:*}` were parsed, and 855 printed cells compared as strings, daggers
  included.
- **2 mismatches**, both in `tab:im_by_point`, Othello IM Index:
  - point 3: paper +0.26, shipped 0.25492 → +0.25;
  - point 6: paper +0.39, shipped 0.38495 → +0.38.
- The paper is wrong; see "Paper corrections".
- Every other cell of all 14 table panels matches: decodability, editability with its 4 daggers, predictive
  (including every ± and the ×10⁻³ scaling), seed spread with its 4 daggers, legal/illegal (including `--` and n),
  two-flip, both token tables (including the mean-frame rows), additional_rw (a) and (b), categorical, im_vs_nn and
  fidelity_selected.

**c. In-text numbers: 29 checked, 28 match.**

| number | release | paper |
|---|---|---|
| IM vs IM-NN mean gain | 53.6% (per run 80.3 / 89.0 / 48.7 / 35.9 / 32.9 / 35.0) | 53.6% |
| Probe Skill seed SD, max | 0.0037, at 128-ray LIN | at most 0.004 |
| Edit Index seed SD of landing editors, max | 0.034 (IM, standard Othello) | below 0.04 / at most 0.034 |
| cells with seed SD > 0.1 | 4 | 4 |
| smallest share of the gap closed | 0.9914 (Othello standard) | at least 99% |
| CI multiplier | 2.484 | 2.48 |
| smallest IM step as the rays coarsen | 6.10 combined-SD units (16→8, continuous); 6.49 on the categorical target (8→5); 7.39 in larger-SD units | at least six |
| seed means vs 780k values | Edit Index max \|diff\| 0.045; Edit Fidelity 0.256 | within 0.05; up to 0.26 |
| Othello standard PI | seed-mean fidelity 0.48 vs 0.70; α 5 on every seed vs 3 | 0.48 vs 0.70, larger step |
| fixed-setting SDs | 0.071 / 0.050 / 0.317; adjflip PI per seed +0.28/+0.33/+0.19 at fidelity 0.17/0.13/−0.40 | 0.05 to 0.07, 0.32 |
| adjacent-noflip IM | 1 seed with no arm inside the cutoff | one seed |
| adjflip IM | 0.62 ± 0.03, fidelity 0.40–0.59, point 5 on every seed | same |
| three selection daggers | a different setting on each seed | same |
| probe refits | skill 0.0007 (max 0.00073), PI 0.016, IM 0.004 (standard) / 0.036 (adjflip), 6–10 seeds | same |
| reachability counts | 441/555/4, 335/614/51, 0/1000/0 twice | same |
| two-flip | max SE 0.11 (legal/illegal groups), 480 partners searched on standard-noflip | same |
| token model | probe gap 0.013, 49 cases removed | same |
| selection rule | PI/GS steps: 18 smaller, 9 same, 1 larger | mostly smaller |
| decodability paragraph | MLP min 0.883; random init within 0.03 except blink (0.109) | same |
| editability paragraph | +0.35, and +0.51 / +0.60 on the categorical 5-ray | same |
| im_vs_nn prose ranges | 0.32–0.58, 0.73–0.90, +0.27…+0.60, 0.22–0.54, 0.17–0.26 | same |
| **IM shift under the fidelity rule** | **0.106** (adjflip +0.664 → +0.558) | **"at most 0.10"** |

**d. By-point figure.**
- `scripts/figures/editability_by_point.py`, run in the throwaway tree, exited 0 and wrote
  `work/tables/tree/outputs/figures/editability_over_res_point.{pdf,png}`.
- I viewed the PNG. It is visually identical to `paper/figs/editability_trends/by_point.png`; only the
  hollow-marker legend text differs.
- 220 plotted values were compared with `paper/figs/editability_trends/by_point_values.json`:
  - 193 are identical;
  - 27 (the IM arms, Edit Index and fidelity ratio) differ by 1e-7 to 1e-9. The shipped scores come from a later
    re-attachment of the IM arms, and the difference is invisible in the plot.

**e. Imports and lint.**
- `pim.figures`, `pim.figures.tables` and `pim.figures.waterfall` import, with `pim.__file__` under RELEASE. The
  figure script also imports.
- `ruff --select F,E9` on the owned `.py` files and both notebooks: all checks passed.
- `ruff --select E,W,F --line-length 120` on the owned `.py` files: all checks passed.
- `waterfall_grid` smoke test: renders 9 axes, warns on fewer than 3 rows, and raises on empty columns.
- The writing-rule grep (dates, names, `discworld`/`dw-`, research files, history words, absolute paths, British
  spellings) is clean. `GREY` was renamed to `GRAY`.

**Cleanup.** Every `__pycache__` under RELEASE was deleted. Some of them came from my early runs before I switched
to `PYTHONDONTWRITEBYTECODE=1`; the others were created by concurrent workers, and deleting them is harmless.

## Paper corrections (the paper was not edited)

1. **`tab:im_by_point`, Othello IM Index at points 3 and 6 should read +0.25 and +0.38, not +0.26 and +0.39.**
   The shipped `scores.json` gives 0.25492 and 0.38495. The paper's own comment (line 832) already records
   "+0.2549 / +0.3849", and the figure's stored values agree to 1e-7. The printed cells predate the IM re-attach.
   The prose ("IM lands only at points 4 and 5") is unaffected.
2. **`app:edit_fidelity_choice`: "IM ... moves by at most 0.10 in Edit Index."** The true largest move is 0.106
   (adjacent-flip, +0.664 → +0.558). 0.10 is the difference of the two rounded table cells, +0.66 and +0.56. It
   should read "by about 0.1" or "at most 0.11".

## Open issues

- **"Lands" is not defined in the paper.** `seed_sd_maxima` uses a seed-mean Edit Index ≥ 0.25 (`LANDS`). The
  0.034 maximum (IM on standard Othello, mean +0.778) is the same for any threshold above 0.165 and up to 0.778
  (checked in 0.01 steps). At 0.165 or below, adjacent-flip PI (mean +0.165, SD 0.245) enters, and at 0.132 or
  below so does 5-ray continuous PI (SD 0.036).
- **The probe-refit set is encoded as `probe_refit_spread` defaults.** The linear probe uses othello/standard,
  othello/adjacent-flip__seed0/1/2 and rayworld/8-ray appearance-fac; the inverse map uses othello/standard and the
  three adjacent-flip replicates. Three `variance.json` files are shipped but not quoted by the paper:
  othello/adjacent-flip, rayworld/standard and the rayworld/8-ray `full` target. Adding them would not change the
  maxima.
- **Row labels are cosmetic.** Labels such as "Appearance (30)" and the variant notes follow the paper loosely. The
  check matches rows by order and values, not by labels.
