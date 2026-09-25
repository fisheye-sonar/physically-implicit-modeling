# Worker report: figures-recheck

**Task.** Two changes landed after stage B:
- the export added the Standard and Blink `appearance-fac` blocks;
- the core-ray-fix worker changed the inverse-map paths to read the probe corpora only on a cache miss.

I re-ran every figure script against the STAGING artifacts, pixel-compared the outputs with PRIVATE's renders, then hid the corpora and checked which scripts still run.

**Result.** Every expectation holds, and one fix to two of my files was needed.
- **Rayworld appendix grids:** all 5 are now pixel-identical to PRIVATE.
- **Other figures:** the overview, the history rewrite and both prediction figures are still pixel-identical.
- **Othello grids:** they differ from PRIVATE only in the three cells the stage-B report explained.
- **Hidden corpora, as shipped:**
  - `qualitative_overview.py`, `qualitative_rayworld.py` and `history_rewrite.py` still **failed**, on `datasets/rayworld/standard/probe/probe_120k.h5`. The core-ray-fix change only takes effect when the caller passes `bank=False`, and they did not.
  - `qualitative_othello.py` and `predictions.py` ran.
- **The fix:** I added `bank=False` to the two figure calls (core-ray-fix request 1). Now **all six figure scripts run from the core bundle alone**. Their outputs are byte-identical to the all-bundle run, before and after the fix.
- **One qualification for the Othello figures:** without the corpora they silently regenerate `datasets/othello/<variant>/probe/probe_20000.npz` in place, about 6 s each. The games are identical to the shipped ones.

Everything is in `experiments/release/work/figures-recheck/`:
- the trees (`treeA`–`treeD`);
- `logs/<run>/{summary.txt, *.log, *.err, *.strace}`;
- `pixdiff.py`, `make_core_tree.py`, `run_all.sh`, `run_trace.sh`.

## Setup

- **Code.** Each tree is a copy of RELEASE `pim/` and `scripts/`. `diff -r` against RELEASE was empty for every tree at run time, and `pim.__file__` resolved inside the tree.
  - I used copies because `style.save` writes to `<repo>/outputs/`; running in RELEASE itself would have written there.
- **Data.**
  - **A and C:** the same five STAGING symlinks RELEASE has, so all bundles are present.
  - **B and D:** one file symlink per MANIFEST entry with `bundle == "core"` (784 files). They have no `datasets/*/*/probe/` and no replicate weights. The corpora bundle is exactly the 32 `probe/` files.
- **Code state.**
  - **A and B:** the scripts as they were before my edit.
  - **C and D:** after the edit.
- **Runs.**
  - Each script ran with an empty `outputs/cache/`, so every script computed everything itself.
  - `PYTHONDONTWRITEBYTECODE=1` throughout.
  - One heavy job at a time on the GPU.

## 1. Pixel comparison against PRIVATE (`pixdiff.py`, the stage-B reference pairs)

The references are in PRIVATE `paper/figs/`:
- `qualitative_main/composite_final_sidebyside.png`
- `qualitative_edits/more_seeds/seed{k}/qualitative_edits_seed{k}_paired.png`
- `qualitative_edits_othello/more_seeds/seed{k}/othello_edits_seed{k}_cols.png`
- `history_rewrite/history_rewrite.png`
- `predictive_quality/{rayworld,othello}.png`

Results are the same in trees A, C and D; all sizes match.

| figure | differing pixels |
|---|---|
| overview | **0** |
| Rayworld seeds 5, 7, 9, 10, 12 | **0** each (was 238,032 each in stage B) |
| history rewrite | **0** |
| Rayworld predictions | **0** |
| Othello predictions | **0** |
| Othello seed 1 | 95,139, in exactly 3 clusters (below) |
| Othello seed 2 | 108,572, in exactly 3 clusters (below) |
| Othello seed 3 | 96,176, in exactly 3 clusters (below) |
| Othello seed 4 | 102,011, in exactly 3 clusters (below) |
| Othello seed 5 | 90,871, in exactly 3 clusters (below) |

**The Othello clusters.** They are identical in position in all five seeds (1306 × 1448 image; 4 variant columns × 5 condition rows):

| cluster (pixels) | cell | release draws (paper draws) |
|---|---|---|
| rows 925–1175, cols 756–1007 | GS row, Adjacent NoFlip | pt 2, α 0.05 (pt 2, α 1.5) |
| rows 925–1175, cols 1055–1305 | GS row, Standard NoFlip | pt 4, α 0.05 (pt 0, α 1.5) |
| rows 1196–1447, cols 1055–1305 | IM row, Standard NoFlip | pt 1 (pt 7) |

- The Adjacent NoFlip IM cell (rows 1196–1447, cols 756–1007) has 0 differing pixels.
- These are the three cells of stage-B open issue 2. The printed settings confirm them: adjacent-noflip GS (2, 0.05); standard-noflip GS (4, 0.05) and IM (1, 1.0).

**Rayworld settings.** The logs no longer print the "no appearance-fac block … cells are left blank" note. Standard and Blink get categorical settings:

| run | categorical PI | categorical GS | categorical IM |
|---|---|---|---|
| Standard | (4, 0.25) | (0, 0.35) | none |
| Blink | (3, 0.25) | (0, 0.7) | none |

**History numbers.** Printed and saved in `history_rewrite.json`:
- Edit Index −0.93 / +0.61 / +0.63 / +0.65, with Fidelity 0.00 / 0.64 / 0.64 / 0.52 (unedited / IM / hist+IM / hist).
- History RMSE 0.264 → 0.114; IM at point 6; drawn cases [14, 18, 25].
- These are **bitwise equal** to PRIVATE `paper/figs/history_rewrite/scores.json`: every Edit Index, step-1 and step-14 value, fidelity and both RMSEs.
- The JSON file is byte-identical to the stage-B and core-ray-fix renders.

**Viewed every PNG** (the overview, 5 Rayworld grids, 5 Othello grids, the history rewrite and both prediction figures).
- Standard and Blink now show categorical PI and GS predictions and errors. Only their categorical IM cells are blank, which matches the paper's "the categorical IM is blank for standard and blink".
- Layout, labels, colorbars and keys are intact everywhere.

## 2. The corpora hidden (core bundle only)

**Before my fix** (tree B; traced with `strace -e openat`):

| script | result | the corpus file it needs, and why |
|---|---|---|
| `qualitative_overview.py` | **FAIL** | `datasets/rayworld/standard/probe/probe_120k.h5` (FileNotFoundError): the path runs through `qualitative_rayworld.predict`, which calls `iter_inverse_maps` for the continuous IM |
| `qualitative_rayworld.py` | **FAIL** | the same file, same call |
| `history_rewrite.py` | **FAIL** | the same file, from `history_rewrite.py:75` |
| `qualitative_othello.py` | ran (rc 0) | regenerated `probe_20000.npz` for all 4 variants (see below) |
| `predictions.py` | ran (rc 0) | none |

**Why the three Rayworld failures happened.**
- `iter_inverse_maps` defaults to `bank=True`. For the continuous map it then reads the probe corpus and computes residuals to build the IM-NN retrieval bank, even on a cache hit of g (`arms.py:479`, `if bank or hit is None`).
- The figures discard the bank.
- The categorical path already read no corpus on a hit. With the corpora present, the trace of the unmodified scripts shows it opening only the `probe_250k.json` manifest, never the `.h5`.

**The fix** (my files only; this is core-ray-fix's request 1):
- `scripts/figures/qualitative_rayworld.py`, `inverse_map()`:
  - pass `bank=False`;
  - replace the "frees the retrieval bank" comment on `gen.close()` with the one-line comment `# no retrieval bank: a cached map then reads no probe corpus` above the call.
- `scripts/figures/history_rewrite.py:75`: pass `bank=False`.
- **Numerics-free:** neither script reads the bank or `nn_r2`.

**After the fix** (tree D, fresh, core only):

| script | rc | wall | peak RSS | `datasets/` files opened (strace) |
|---|---|---|---|---|
| `qualitative_overview.py` | 0 | 63 s* | 8.0 GB | Rayworld `edits/edits.h5` ×6; `eval/test.json` for 128-ray and 5-ray; Othello `edits/cases_1000.pkl` ×3; **Othello `probe/probe_20000.npz` ×3** |
| `qualitative_rayworld.py` | 0 | 26 s | 2.7 GB | `edits/edits.h5` ×5; `eval/test.json` for 16-, 8- and 5-ray (no `probe/`) |
| `qualitative_othello.py` | 0 | 65 s* | 7.9 GB | `edits/cases_1000.pkl` ×4; **`probe/probe_20000.npz` ×4** |
| `history_rewrite.py` | 0 | 2.6 s | 1.3 GB | `standard/edits/{edits.h5, selection.json}` |
| `predictions.py` | 0 | 8.7 s | 2.2 GB | `edits/{edits.h5, selection.json}` ×6; Othello `cases_1000.pkl` ×4 |
| `editability_by_point.py` (not mine) | 0 | 0.7 s | 0.14 GB | none; it reads `scores.json` only |

\* These wall times include the silent Othello regeneration. With the splits present (tree C) the overview takes 46 s and `qualitative_othello.py` 58 s.

**Other checks on tree D:**
- No `.scratch/` was created.
- No file was written under `runs/`, so there were no probe or inverse-map cache misses.
- The only files written outside `outputs/` are the 4 regenerated Othello splits.
- The categorical maps took their sim from `eval/test.json` (the `_corpus_sim` fallback). Their `d_in` guard passed.

**Outputs are identical across trees.** All 29 outputs (14 PNG, 14 PDF, `history_rewrite.json`) are **byte-identical** across:
- A (old code, all bundles);
- C (fixed code, all bundles);
- D (fixed code, core only).

With the corpora present, the fix also cuts the Rayworld grid from 66 s / 6.7 GB to 26 s / 2.7 GB, and the history rewrite from 9.3 s / 6.0 GB to 2.7 s / 1.3 GB.

**Why the Othello figures still touch the probe split** (`othello_qualitative.probe_games` → `pim.environments.othello.corpus.build(only=("probe",))`). They load the first 20,000 games of `datasets/othello/<variant>/probe/probe_20000.npz` for two reasons:
1. The probe-grid cache key of `oa.fit_probe_grid` includes `n_seq` and `n_rows` (the labelled-position count), computed from those games.
2. `oa.inverse_arms` harvests the model's residuals on those games to build the IM-NN retrieval bank on every call, even when g is a cache hit. Its key also uses `n_seq`.

When the file is absent, `corpus.build` regenerates it in place. The script passes `log=lambda s: None`, so this happens silently.
- Timed alone: 6.2 s for 20,000 games on 32 cores.
- `tokens`, `lengths`, `lo` and `seed` are **identical** to STAGING's file for all 4 variants.
- adjacent-flip and adjacent-noflip are byte-identical (same sha256).
- standard and standard-noflip gain the metadata keys the shipped files lack: `flip` / `placement` / `instance`, and `placement` respectively.
- The figures are byte-identical either way.

I left this unchanged. Removing the dependency would take a change to `pim/environments/othello/arms.py` (not mine), and the cost is small.

## 3. What the README must say about bundles (request to infra)

- **Every figure in `scripts/figures/` runs from the core bundle; none needs the corpora or replicates bundles.**
  - Rayworld qualitative grids, history rewrite, both prediction figures and editability-by-point: core only; nothing under `datasets/*/*/probe/` is read.
  - Othello qualitative grids and the overview's panel (b) read `datasets/othello/<variant>/probe/probe_20000.npz` (corpora bundle, 4 files, 4.9 MB). Without it, the script regenerates the file in place: about 6 s per variant on CPU, same games, written into `datasets/`.
- **Resources:**
  - a GPU (every run here used one; CPU was not tested) and up to about 8 GB of RAM (the overview and the Othello grids; the Rayworld grids need 2.7 GB);
  - times on one GPU: the overview about 45 s, Othello grids about 60 s, Rayworld grids about 26 s, the others under 10 s.
- **Outputs:**
  - `outputs/figures/` holds the figures;
  - `outputs/cache/qualitative_{rayworld,othello}.pkl` holds the prediction caches, which the overview reuses; `--recompute` refreshes them.
- **`.scratch/`:** keep it gitignored. The figure scripts no longer create it; scoring and probe fits still do.
- **Superseded advice.** This replaces the stage-B figures report's request 2 ("the qualitative figures and the history rewrite need the corpora bundle … up to about 10 GB"). It also supersedes open issues 4 and 5 there.

## Requests

1. **infra (`README.md`):** the bundle notes in section 3.
2. **Informational, for tables / style-fix (`scripts/figures/editability_by_point.py`):** it runs from the core bundle, but it is **not** pixel-identical to PRIVATE `paper/figs/editability_trends/by_point.png`.
   - 15,547 pixels differ, all in the legend row (rows 909–940).
   - The release key reads "Edit Fidelity < 0" where PRIVATE's reads "outside fidelity guard".
   - The plotted data are identical, and the render is byte-identical to style-fix's.
   - The label is presumably intended, but the owner should confirm it matches the caption.

## Verification

- `ruff check --no-cache --isolated`, and with `--select E,W,F --line-length 120`, on both edited files: "All checks passed!".
- The writing-rule grep (dates, old names, paths, banners, history words) on both files: no hits.
- **RELEASE is untouched apart from the two edits:**
  - no `__pycache__` (none created, none to delete);
  - no `outputs/`;
  - `.ruff_cache` mtime unchanged;
  - the empty `.scratch/` at the RELEASE root predates this task (mtime Sep 23 22:53) and was left alone.
- No writes to STAGING or PRIVATE `runs/` / `datasets/` / `logs/` / `outputs/`. No git operations.

## Open issues

1. **Othello appendix text (still open, as in stage-B open issue 2).** The regenerated Othello grids show the fallback settings for the three cells above. The appendix sentence saying those figures show "the setting with the highest Edit Index" needs changing in the paper, which is not a release file.
2. **The overview variant (`composite_final_sidebyside`) is still the stage-B judgement call.** It is pixel-identical to that PRIVATE render; I did not re-examine which variant the paper includes.
3. **CPU-only runs of the figure scripts were not tested.**
