# Worker report: figures

**Owned:** RELEASE `scripts/figures/**`, except `editability_by_point.py` (the tables worker's).

**Result:** six self-contained scripts regenerate the paper's data-driven edit and prediction figures from the shipped artifacts alone. They total 1,092 lines; the PRIVATE sources they replace are 2,642 lines plus `paper_style.py`.
- Against the PRIVATE figure outputs, 4 of the 14 figures are **pixel-identical** PNGs: the main overview, the history rewrite and both prediction figures.
- The 5 Rayworld appendix grids differ **only** in the Standard/Blink categorical PI and GS cells, which STAGING does not ship (request 1). With those artifacts added they are pixel-identical too.
- The 5 Othello appendix grids differ **only** in the three cells whose selected setting changes under the current `best_arm`.
- Every editor frame, board distribution and number that does not depend on those two issues is **bitwise identical** to the PRIVATE caches.

Helper scripts, the throwaway trees and all outputs are in `experiments/release/work/figures/`. `tree/` is the release code on STAGING symlinks; `tree_ext/` adds the four re-keyed probes and two blocks of request 1.

## What I built (`scripts/figures/`)

| script | writes (under `outputs/figures/`, mirroring the paper's `figs/`) | source in PRIVATE `paper/figs/` | run time / peak RSS |
|---|---|---|---|
| `style.py` | nothing; shared helpers | `paper_style.py` | — |
| `qualitative_overview.py` | `qualitative_edits_overview.{pdf,png}` | `qualitative_main/` (`composite_final_sidebyside`) | 114 s / 9.8 GB (first run; cached afterwards) |
| `qualitative_rayworld.py` | `appendix/rayworld_qualitative/qualitative_edits_seed{5,7,9,10,12}_paired.*` | `qualitative_edits/make_figure.py` (paired mode) | 75 s / 6.7 GB |
| `qualitative_othello.py` | `appendix/othello_qualitative/othello_edits_seed{1..5}_cols.*` | `qualitative_edits_othello/make_figure.py` (cols layout) | 32 s / 7.6 GB |
| `history_rewrite.py` | `appendix/history_rewrite.{pdf,png,json}`, and prints the paper's numbers | `history_rewrite/{make_figure,draw_paper}.py` | 11 s / 6.1 GB |
| `predictions.py` | `appendix/rayworld_predictions.*` (six variants, two bands of three) and `appendix/othello_predictions.*` (four variants) | `predictive_quality/{compute_rayworld,rayworld,othello}.py` | 10 s / 2.2 GB |

**`style.py`** holds the shared style:
- fonts: Arial as TrueType, with Liberation Sans / Nimbus Sans fallbacks;
- the colors;
- `save`: PDF plus PNG under `outputs/figures/`, cropped with no padding, and without a `CreationDate`, so the PDFs are byte-reproducible;
- the drawing primitives: `strip`, `error`, `locators`, `waterfall`, `box`, `arrow`, `line_key`, `board`, `mark_key`, `colorbar`, and `EDITOR_COLORS`.

Importing it selects the Agg backend and puts the repository root on `sys.path`, so the scripts run as `python scripts/figures/<x>.py` with no `PYTHONPATH`. The tables worker's `editability_by_point.py` already imports it (`TEXT`, `TEXT_WIDTH_IN`, `save`), and it runs with my version.

**How the scripts are built:**
- Every number and write goes through `pim`:
  - selection: `pim.metrics.selection.best_arm`;
  - benches and scenarios: `rayworld.bench.bench_from_arrays` / `load_bench` and the edit-set generator;
  - editors: `rayworld.arms.{pinv_rollout, grad_steer_rollout, iter_inverse_maps, unsteered_rollout, score, fidelity_ratio}` and `othello.arms.{fit_probe_grid, linear_arm, grad_steer_arm, inverse_arms, unsteered_probs}`;
  - Othello references: `pim.metrics.set_editability.uniform_over_legal`.
- Probes and inverse maps are read from each run's cache: `require_cached=True` for the probes, and cache hits for the maps.
- The only caches are `outputs/cache/qualitative_{rayworld,othello}.pkl`. The overview reuses them; `--recompute` refreshes them. Nothing is written to `.scratch/`.
- **No alternate layouts, rounds, pieces or sidecar machinery.** Each script has a docstring and a `--help`. The only flags are `--seeds` (the two appendix grids) and `--recompute`.

**The paper's defaults are fixed in the code:**
- Rayworld scenario seeds: 0/2 (continuous) and 1 (categorical) for the overview; 5, 7, 9, 10, 12 for the appendix. I checked the documented rule: the passing seeds among 0–13 are exactly 0, 1, 2, 5, 7, 9, 10, 12.
- Othello overview cases: 342, 261, 39.
- Othello appendix picks: `default_rng(k)`. They match PRIVATE's sidecars for k = 1..5.
- History-rewrite cases: 14, 18, 25 (eligibility at ≥ 20 rays).
- Prediction cases: Rayworld 20 and 26 in every variant; Othello 511/636/848, 16/40/75, 503/648/911, 543/631/728.

**Which variant of each figure:**
- **Overview:** `composite_final_sidebyside` (the four-column A3 cut, 6.10 in).
  - Why: rounds 7 and 8 of the PRIVATE work (commits adf2c2a and 628da44) refined only the side-by-side.
  - Its (b) shows three Othello variants, which matches the caption's "across environment variants".
  - This is a judgement call; see open issue 1.
- **Rayworld appendix:** the `_paired` mode, as named in the `.tex`.
- **Othello appendix:** the `cols` layout, as named in the `.tex`.
- **Rayworld predictions:** the six-variant, two-band version (commit 25d0f49), matching the caption.
- **History rewrite:** `history_rewrite.pdf` (single write plus rewritten history), not `_histonly`.

## API changes
None to `pim`. These are new files only.

## Requests to other owners

**1. export: ship the Standard and Blink `appearance-fac` block and its two probes.**
- **Why.** The paper's appendix Rayworld figures (`fig:more_qualitative_edits_rayworld_1/2`, and the text "the categorical IM is blank for standard and blink") draw categorical PI and GS for Standard and Blink. STAGING stripped those blocks and probes as out of scope, so the release draws those 4 cells per figure blank and prints a note.
- **What to add:**
  - `bases["appearance-fac"]` in `runs/rayworld/standard/scores.json` and `runs/rayworld/blink/scores.json`, from PRIVATE `noise_ablation/L-dw-noiseless-20m` and `blink_ablation/L-dw-blink-20m`. Only the arms are read. There is no IM arm, as in PRIVATE.
  - Four seed-0 probe files, re-keyed as the export re-keys (only `data` changes):

    | run | family | PRIVATE file | re-keyed file |
    |---|---|---|---|
    | standard | linear | `probes_3201354ff714f717.pt` | `probes_56ff19304c766a12.pt` |
    | standard | mlp | `probes_474fa1d299d82fb0.pt` | `probes_c35d55fcb934f87d.pt` |
    | blink | linear | `probes_d0bf8c88ee6b18f7.pt` | `probes_27c7d687d087f562.pt` |
    | blink | mlp | `probes_22304240197eca90.pt` | `probes_2015e2ea7a793132.pt` |

    All are `appearance-fac`, 200000 sequences, `probe_250k`, frustum, 50 epochs.
- **Verified.** With exactly these additions (`work/figures/make_ext_tree.py` builds them), all five Rayworld appendix grids are **pixel-identical** to the PRIVATE figures. No `probe_250k` corpus is needed for Standard and Blink, because the probes are cache hits.
- **Knock-on for other owners.** The tables code must keep excluding these two blocks from Table 2c; rayworld-env's whitelist request #6 already covers this. SETTINGS does not list them, so the scorer does not refit them.
- **If this is declined:** the appendix paragraph must say that every categorical cell is blank for Standard and Blink.

**2. infra (`.gitignore`, `README.md`):**
- **`.gitignore`:** keep `outputs/` and `.scratch/` ignored. `pim.environments.rayworld.arms._scratch_dir()` creates an empty `.scratch/` at the repo root when the figures call `iter_inverse_maps`; its memory maps are temporary and removed.
- **README:**
  - The qualitative figures and the history rewrite need the **corpora** bundle: `iter_inverse_maps` reads `probe_120k` (continuous maps) and `probe_250k` (categorical maps) even on a cache hit, and the Othello figures read `probe_20000.npz`.
  - `predictions.py` needs only the core bundle.
  - All of them need a GPU, and up to about 10 GB of RAM (the 128-ray categorical map reads 200k sequences).

**3. tables (`scripts/figures/editability_by_point.py`), optional:**
- save as `"appendix/editability_over_res_point"`, so it lands beside the other appendix figures and mirrors the paper path `figs/appendix/editability_over_res_point.pdf`;
- use `style.EDITOR_COLORS` instead of the local copy (same values).

**4. rayworld-env (`pim/environments/rayworld/arms.py`), optional; stage A is closed:**
- `_iter_categorical_inverse_maps` could read the probe corpus only on a cache miss.
- `iter_inverse_maps` could skip the retrieval bank when only g is wanted.
- Both changes are numerics-free. They would let the figures run without the 9.35 GB corpora bundle and without the 200k-sequence read. Not needed for correctness.

## Verification (exact results)

Setup: everything ran in `work/figures/tree/`, a copy of RELEASE `pim/` and `scripts/` with `runs/` and `datasets/` symlinked read-only into STAGING. RELEASE `layout.py` was already flat, so no patch was applied. `pim.__file__` was under the tree and RELEASE. A clean run (outputs deleted) of all five scripts exited 0 each.

1. **Rayworld editor frames against the PRIVATE caches** (`.scratch/qualitative_edits_catim_*`; `compare_rayworld.py`):
   - Coverage: every column of seeds 0, 1, 2, 5, 7, 9, 10, 12 over Standard, Blink, 16/8/5-ray and 128-ray.
   - Compared: context, both GTs, locators, and PI/GS/IM on both blocks.
   - Result: **all `==` (bitwise)**; max |new − old| = 0.0. The drawn settings equal PRIVATE's.
   - Only exception: Standard/Blink categorical PI/GS are `None` (request 1).
2. **Othello distributions against the PRIVATE guarded cache** (`compare_othello.py`):
   - Boards, positions and legal sets are identical for all 4 variants.
   - PI/GS/IM are `==` on standard and adjacent-flip; PI and IM are `==` on adjacent-noflip; PI is `==` on standard-noflip.
   - Different, as expected: adjacent-noflip GS (max |Δp| 0.928), standard-noflip GS (1.0) and IM (0.874), whose selected settings changed.
3. **History rewrite:** the Edit Index, Fidelity, and step-1 and step-14 Edit Indices of all four rollouts, and both history RMSEs, are **bitwise equal** to PRIVATE `history_rewrite/scores.json`. Printed:

   | rollout | Edit Index | Edit Fidelity | step 1 | step 14 |
   |---|---|---|---|---|
   | unedited | −0.93 | 0.00 | −0.93 | −0.82 |
   | IM | +0.61 | 0.64 | −0.63 | −0.74 |
   | hist+IM | +0.63 | 0.64 | +0.59 | +0.23 |
   | hist | +0.65 | 0.52 | +0.57 | +0.21 |

   History RMSE: original 0.264, rewritten 0.114. IM is at point 6 (`best_arm`). These match the paper text.
4. **PNG pixel comparison against PRIVATE's PNGs** (`pixdiff.py`; same sizes in every case):

   | figure | result |
   |---|---|
   | overview vs `composite_final_sidebyside.png` | **0** differing pixels |
   | `history_rewrite` | **0** |
   | `rayworld_predictions` | **0** |
   | `othello_predictions` | **0** |
   | Rayworld seeds 5, 7, 9, 10, 12 (STAGING) | 238,032 px each, all inside the Standard/Blink categorical PI/GS box (rows 1088–1361, cols 276–1230); 0 elsewhere |
   | same five in `tree_ext` (request 1 applied) | **0** each |
   | Othello seeds 1–5 | differences only in rows 925–1447 and cols 756–1305, i.e. the GS/IM rows of the two no-flip columns (the three changed cells); 0 elsewhere |
5. **Viewed every PNG** with the Read tool: 1 overview, 5 Rayworld grids, 5 Othello grids, the history rewrite and 2 prediction figures. Layout, labels and colors match the PRIVATE figures.
6. **Lint and help:**
   - `ruff check --isolated` (defaults) and `--select E,W,F --line-length 120` on the six files: "All checks passed!"
   - `--help` works for all five scripts.
   - No font warnings; the PDFs embed ArialMT, Arial-BoldMT and Arial-ItalicMT; the PDF metadata holds only the Matplotlib Creator/Producer.
7. **Writing rules:** a grep for dates, names, discworld/dw-/L-dw/L-oth/oth-, research/harness/experiments/queue paths, absolute paths, ⛔/⚠, history words and British spellings found nothing, apart from "fixed" meaning constant.
8. **No writes outside the work dir:**
   - STAGING is read-only and no write was attempted. No PRIVATE runs, datasets, logs or outputs were touched.
   - No `__pycache__` was created under RELEASE (everything ran with `PYTHONDONTWRITEBYTECODE=1`).
   - One exception: my `ruff` runs refreshed the pre-existing, self-ignored `.ruff_cache/0.15.7` at the RELEASE root. Later runs used `--no-cache`.

## Open issues

1. **Overview variant (judgement call).** I reproduced `composite_final_sidebyside` (A3). If the paper uses the stacked `composite_final` or `composite_final_sidebyside_single` instead, the layout constants in `qualitative_overview.py` need changing; the data are the same.
2. **Othello appendix cells change under the current selection rule.** Where no setting is within the guard, the release draws Table 2's lowest-fidelity-ratio fallback:

   | cell | release draws | paper draws (highest Edit Index) |
   |---|---|---|
   | adjacent-noflip GS | pt 2, α 0.05 | pt 2, α 1.5 |
   | standard-noflip GS | pt 4, α 0.05 | pt 0, α 1.5 |
   | standard-noflip IM | pt 1 | pt 7 |

   The new writes are near-unedited distributions. The appendix sentence "Where no setting reaches an Edit Fidelity of 0 … the figures show the setting with the highest Edit Index rather than the fallback of Table 2", and its `% figure facts` comment, no longer describe the regenerated figures. The paper text needs an update (not a release file).
3. **Overview Othello case for adjacent-noflip.** The case is fixed at 39, the paper's case, and PI and IM are unchanged there, so the overview is pixel-identical. Re-running PRIVATE's "second-most typical" rule under the current GS setting would pick case 352 (39 is now third). The docstring describes the cases as fixed, typical-looking cases rather than claiming the rule.
4. **Request 1 is needed to match the Rayworld appendix grids.** Until then those grids run, but leave Standard/Blink categorical PI/GS blank, with a printed note.
5. **Corpora bundle.** The qualitative and history figures depend on the corpora bundle, because `iter_inverse_maps` reads the probe corpus even on a cache hit (requests 2 and 4).
