# predictive_quality — the trained models predict well before any edit (appendix)

Two figures, one per environment. Neither script loads a model: both read caches that other figure scripts
computed. Every PDF is vector (Arial embedded as TrueType; the only rasters are the Rayworld observation and
error strips, nearest interpolation at 600 ppi so all 128 rays survive), cropped to the content with no outer
padding (`paper_style.save`), with a 300-dpi PNG preview beside it. No number appears inside either figure;
the only text is the row labels, the axis words, the key, the colour-bar label and the Othello panel letters.

| file | what it shows |
|---|---|
| `rayworld.{pdf,png}` | Rayworld, `runs/noise_ablation/L-dw-noiseless-20m` (dw-noiseless): three rows, Ground truth (the simulator's clean render of the UNEDITED world continuing), Prediction (the model's free-run) and Difference (prediction − truth on the canonical signed-error map), by six held-out sequences as columns; in every panel the last 8 observed frames above the dashed free-run-start line and the 15 free-run steps below it, time downward, 128 rays wide; the colour bar once beside the difference row; 5.49 x 2.80 in |
| `rayworld.json` | sidecar: run, instance, the cases and their rule, what each row draws (the difference's definition and scale), the source arrays, the file list |
| `othello.{pdf,png}` | Othello, four rule-variant columns (a) standard, (b) adjacent-flip, (c) adjacent-noflip, (d) standard-noflip; three example blocks stacked vertically, each a (legal moves / model) row pair: the board tinted by its true legal set (uniform over the legal squares, the Bayes-optimal target) over the same board tinted by the trained model's unedited next-move distribution; a thin gap between the blocks; 5.55 x 7.90 in (a full-page appendix figure) |
| `othello.json` | sidecar: cache column, run, instance, the three bench case ids per variant with board length and legal-set size, tint parameters, the file list |
| `pieces/rayworld_case<idx>_{gt,pred,diff}.pdf` (18), `pieces/rayworld_key.pdf`, `pieces/rayworld_colorbar.pdf` | every Rayworld panel (0.72 x 0.78 in), the key and the colour bar |
| `pieces/othello_<variant>_<case>_{legal,model}.pdf` (24) | every Othello board at 2.0 in (PNG previews beside them are gitignored) |

`pieces/` holds only what the two figures use.

## Rayworld — `rayworld.py`

`.pim/bin/python paper/figs/predictive_quality/rayworld.py` (CPU, a few seconds). Reads the SAME array file as
`paper/figs/history_rewrite/draw_paper.py`: `.scratch/history_rewrite_arrays.npz`, written on the GPU by
`paper/figs/history_rewrite/make_figure.py` for the 32 canonical bench cases of the run (if the file is
missing, run that script first). Arrays used: `obs_hist` the observed frames 0..19, `gt_unedited_roll` the
simulator's clean unedited continuation over frames 20..34 (the scorer's ghost trajectory), `roll_unsteered`
the model's free-run from frame 20 with no edit. The Ground truth and Prediction panels come from
`draw_paper.py`'s panel code, imported by path (`gray` on the dark panel, fixed 0-1 range, nearest, thin
frame, white page); dw-noiseless has no observation noise, so the observed frames above the line equal the
clean render.

**Difference row.** prediction − truth per ray over the free-run steps, drawn with `error` and `_panel`
imported from `paper/figs/qualitative_edits/make_figure.py` (the prediction is clipped to [0, 1] first, that
module's default; `raw=True` would be the scorer's unclipped quantity) on `pim.figures.waterfall.DIFF_CMAP`:
under-prediction red, over-prediction green, zero error the dark background, fixed ±1 scale as the qualitative
figures' `_diff` variant uses. Above the line the row is left empty (dark): those frames are observations, not
predictions. The colour bar ("prediction − truth", ticks −1, 0, 1) sits once beside this row.

**Cases.** `numpy.random.default_rng(0).choice(32, 6, replace=False)`, sorted: bench cases **1, 8, 9, 14, 17,
22**. No edit is involved, so no displacement filter (the history-rewrite figure filters small edits; this one
does not need to).

**What it shows (for the caption).** The free-run tracks the clean unedited world over the 15 steps; the
difference row is dark except thin red or green slivers along disc edges, a ray or two where an edge is
predicted slightly early or late. As a descriptive check on the drawn arrays (not a canonical metric),
|prediction − truth| exceeds 0.1 on 0.8 to 1.8 % of the free-run rays of these six sequences (mean |error|
0.004 to 0.009, largest 0.55 to 0.80, at edges). The run's prediction-skill numbers for the text come from its
`scores.json` (`master_eval`), not from this figure.

## Othello — `othello.py`

`.pim/bin/python paper/figs/predictive_quality/othello.py` (CPU; `--seed`, default 0). Moved 2026-09-21 from
`paper/figs/environments_overview/othello/make_predictive.py`. Reads `.scratch/othello_edits_guarded_cache.pkl`,
the qualitative Othello figure's cache (guarded best arms, 2026-09-19): `board_pre`, `legal_pre`,
`probs["Unedited"]` per variant. Boards are the bench's 20-move positions; the "legal moves" row is
`pim.metrics.set_editability.uniform_over_legal(legal_pre)`, the "model" row the cached `probs["Unedited"]`.
Squares are tinted exactly as `draw_board` tints them (full at 0.02 probability mass and above,
`(p / 0.02) ** 0.6` below). The board drawing is loaded by path from
`paper/figs/environments_overview/othello/make_figure.py`, which itself loads `draw_board` from
`paper/figs/qualitative_edits_othello/make_figure.py`; the stacked page layout is this script's own.

**Cases.** One `numpy.random.default_rng(0)` shared across the variants in column order; per variant
`rng.choice(1000, 3, replace=False)`, sorted (the 1000 cached bench cases):

| panel | run | bench cases | legal squares |
|---|---|---|---|
| (a) standard | `initial_othello_comparison/L-oth-20m` | 511, 636, 848 | 12, 10, 13 |
| (b) adjacent-flip | `adjacent_flip_ablation/L-oth-adjacent-flip-20m` | 16, 40, 75 | 24, 17, 24 |
| (c) adjacent-noflip | `adjacency_ablation/L-oth-adjacent-20m` | 503, 648, 911 | 22, 22, 19 |
| (d) standard-noflip | `flip_ablation/L-oth-noflip-20m` | 543, 631, 728 | 7, 5, 6 |

Within each block the two rows look identical at the canonical tint saturation (0.02): a model that puts
1/|L| on every legal square looks exactly like the legal-set board, which is the intended message; the few
paler squares in the model rows are legal squares the model gives less than 0.02 of its mass.
