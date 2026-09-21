# BRIEF, round 2 — the main-text qualitative figure: `composite_final` and `composite_final_sidebyside`,
# and the appendix qualitative figures regenerated in Arial with the current arms

Read `paper/figs/FIGURE_BRIEF_COMMON.md` first, then this file, then the round-1 `BRIEF.md` and `README.md` in this
folder (what exists: `common.py`, `rayworld_panel.py`, `othello_panel.py`, `composite.py`; Sevan picked
`composite_R3_T3_typical_2row` as the base). Folder: `paper/figs/qualitative_main/`. You may also edit the two appendix
scripts `paper/figs/qualitative_edits/make_figure.py` and `paper/figs/qualitative_edits_othello/make_figure.py`
(fonts and the categorical inverse map, below) and regenerate their outputs. GPU: yes, one model at a time, freed after.

## Global changes already made (just use them)
`paper/figs/paper_style.py`: all figure text is **Arial** now (`ps.apply()`), and `ps.save()` crops with **zero outer
padding**. The two appendix scripts still carry their own Times New Roman rcParams block: replace it with
`sys.path.insert(0, <paper/figs>)`, `import paper_style as ps`, `ps.apply()` and re-render (their own `savefig`
calls must also use `pad_inches=0`).

## Part A — the appendix figures, correct arms, Arial
The categorical inverse map was deployed 2026-09-20 (`experiments/categorical_inverse/README.md`, read the top):
on a categorical block (`appearance-fac`) the IM editor's map inverts the block's own labels + Cartesian velocity,
and it exists ONLY for dw-128ray / dw-16ray / dw-8ray / dw-5ray; dw-noiseless ("Standard") and dw-blink categorical
blocks carry **no IM arm** (their table cell is blank). The appendix Rayworld figure's `cat` IM row was drawn from the
OLD continuous map (cache), which is no longer a scored arm anywhere.
- In `paper/figs/qualitative_edits/make_figure.py::predictions`, on a categorical block get the IM write through the
  canonical categorical path: `dwa.iter_inverse_maps(model, basis_name=basis, target=target, points=[pt],
  cache_dir=run_dir / "probes", log=None, **<the categorical recipe>)` — mirror EXACTLY how
  `pim/scoring/discworld.py::inverse_discworld` calls it for a categorical target (recipe = the forward probe's,
  `arms.GRID_PROBE_RECIPE`), and build the write as `pim.environments.discworld.arms.inverse_arms(..., target=target)`
  does (g applied to the one-hot of `Bench.tgt` plus the discs' Cartesian velocity; `pim.probes.inverse.encode_categorical_state`).
  Cache hits only — never fit a map. Where `best_arm(scores, block, "IM")` is None (Standard, Blink categorical),
  the cell is BLANK (empty panel, thin frame, nothing drawn), mirroring the table; no text.
- New cache names (e.g. `_catim` suffix) so the old frames are not reused. Regenerate seeds 0–5 (`--seed k`, five
  models each; the caches then serve Part B) and re-render every appendix output (`qualitative_edits_seed0*` all
  modes, `more_seeds/`, the Othello `_cols`/`_rows` + `more_seeds/` from its cache — no GPU for Othello).
- Apply the Othello marking semantics of Part B to the appendix Othello figure as well (one helper, reused).

## Part B — `composite_final` (Sevan's spec, verbatim where it matters)
Base: `composite_R3_T3_typical_2row`, then:
1. **Rayworld panel (a): four columns** — "Standard (continuous)" × two scenarios and "Standard (categorical)" × the
   SAME two scenarios; rows Context, Unedited, Ground truth, PI, GS, IM with the paired signed-error strips; cyan /
   pink locators. Produce TWO data versions and recommend one:
   - **A1** Standard = dw-noiseless (`noise_ablation/L-dw-noiseless-20m`): the categorical IM cells are BLANK (no arm).
   - **A2** every column from the 128-ray model of the ray family (`ray_ablation/L-dw-128ray-20m`, instance
     `dw-128ray`, radius 1.0), where the categorical IM is the categorical inverse map and every cell is a scored
     arm (Table 2's categorical section starts with this model). Add it as a variant through the appendix script's
     machinery (an optional `variants` argument to `build()`; own cache).
   Column titles are short ("Standard (continuous)" over the pair, or per column); the caption names the model.
2. **Othello panel (b)**: columns Unedited, Ground truth, PI, **GS**, IM; rows Standard, Adjacent NoFlip; the 5×5
   zoom (T3). **New cases**: keep the `typical` rule but take the next-nearest-to-mean eligible case per variant
   (rank 2; rank 3 if the picture is unclear); state the rank in the README and JSON.
   **Marking**: in the Unedited column outline, in CYAN (`ps.ORIGIN_C`), the flipped tile AND every square whose
   legality switches (the symmetric difference); in every other column outline those same squares in PINK
   (`ps.DEST_C`). Nothing else is outlined. A small key (three or four words: "cyan pre-edit · pink post-edit",
   or two swatches) once in the figure. Boards about 10 % larger than round 1; the gap between the Ground truth row
   and the PI row a touch larger than the other row gaps (as the Rayworld panel has between its GT and edit rows).
3. Panel letters (a) (b) noticeably larger (about 11–12 pt bold). Size the Rayworld panel so (a) and (b) are
   vertically balanced at 5.5 in width. Arial, zero padding. Name: `composite_final` (+ `_A1` / `_A2` if both kept).
4. **`composite_final_sidebyside`**: (a) left, (b) right, the same changes EXCEPT: no GS in Othello; instead a third
   variant **Adjacent Flip** between Standard and Adjacent NoFlip — in this layout the variants run as COLUMNS
   (Standard, Adjacent Flip, Adjacent NoFlip) and the conditions as rows (Unedited, Ground truth, PI, IM); the marked
   squares' rims THINNER (the boards are smaller here); the GT→PI gap rule and the 10 % rule still apply.
5. Pieces for both (every strip, board, key) under `pieces/`, sidecar JSONs with cases / seeds / arms / ranks, README
   updated with the caption facts (arms, guard, selection rule, what cyan / pink mean, blank cells = no arm, which
   model the Rayworld panel uses in A1 vs A2, Table 2 numbers for the drawn cells via `best_arm`).

## Report
Files; which of A1 / A2 you recommend and why; the new Othello cases and their per-case indices vs the population
means; confirmation that the appendix figures were regenerated (which seeds, which cells are blank); anything not done.
