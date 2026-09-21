# history_rewrite — persistent transformer edits by rewriting the history (appendix)

Two scripts, compute and drawing kept apart. `make_figure.py` (GPU, about 15 s) computes everything and
writes `scores.json`, its three diagnostic waterfalls and the array file `.scratch/history_rewrite_arrays.npz`;
`draw_paper.py` (CPU, 3 s) reads that file and draws the paper figure, its one alternative, the frames piece
and the pieces. Both import only `pim.*` and `paper/figs/paper_style.py`; the run they read is
`runs/noise_ablation/L-dw-noiseless-20m` (its `scores.json` for the IM editor's scored point, its cached
inverse map) and the dw-noiseless edit bench. Moved here 2026-09-19 from `experiments/history_rewrite/`.

**Idea (Sevan, 2026-09-16).** A transformer's edit at the last position is forgotten: the next prediction
is recomputed from the unedited observation history. The post-edit target state carries the edited disc's
velocity, so its counterfactual trajectory can be integrated BACKWARDS over the whole teacher-forced
window. At every history step t the residual at the IM point is overwritten with g(s_cf[t]) — the inverse
map of the counterfactual state — and the model's own prediction becomes the rewritten frame t+1. The
rewritten history replaces the window the edit is launched from. Frame 0 is kept (nothing precedes it).

**Arms** (canonical bench, first 32 selected cases, Cartesian block, scored like any arm — Edit Index and
fidelity ratio from `pim`): `unsteered` · `IM` (canonical: original history, g(s_post) at EF−1) ·
`hist` (rewritten history, NO write at EF−1) · `hist+IM` (both). Recorded 2026-09-16 and reproduced to
six decimals on 2026-09-21: the rewritten history alone carries the edit (+0.65 / 0.48); with the write on
top +0.63 / 0.36 against the canonical +0.61 / 0.36.

## Compute — `make_figure.py`

`.pim/bin/python paper/figs/history_rewrite/make_figure.py` (GPU, one model, about 15 s; n = 32 cases, one
seed — an illustration, not a table number). Outputs beside the script: `scores.json` (every arm's scorecard
+ the history RMSEs) and its three diagnostic waterfalls `A_original_history_arms.png` ·
`B_rewritten_history_arms.png` · `C_history_vs_counterfactual_render.png` (through
`pim.figures.waterfall_grid`, dark page, verbose titles; the compute's own check, NOT paper figures). Since
2026-09-21 it also saves `.scratch/history_rewrite_arrays.npz`: `obs_hist` (32, 20, 128) the original
observed history · `obs_cf` the rewritten history · `cf_clean` the simulator's clean render of the
counterfactual history · `gt_roll` (32, 15, 128) the clean edited-world rollout · `gt_unedited_roll` the clean
unedited continuation (the scorer's ghost trajectory) · `roll_unsteered`, `roll_IM`, `roll_hist+IM`,
`roll_hist` the four free-runs · `target_x`, `ghost_x` the destination / origin ray centres per case ·
`edit_object`, `s_post` · `scores_json` (the text of `scores.json`).

## Paper figure — `draw_paper.py`

`.pim/bin/python paper/figs/history_rewrite/draw_paper.py` (CPU; needs the npz above). Every output is a
vector PDF (Arial embedded as TrueType, the observation strips the only rasters, nearest interpolation at
600 ppi so all 128 rays survive, cropped to the content with no outer padding) with a 300-dpi PNG preview
beside it.

| file | what it shows |
|---|---|
| `history_rewrite.{pdf,png}` | **the figure.** Three edits (rows) x Ground truth / Unedited / Single-point edit / History rewrite; the last 8 original observed frames above the edit-frame line in EVERY column, the 15 free-run steps below; 5.42 x 3.50 in at the printed size |
| `history_rewrite_histonly.{pdf,png}` | the one alternative, same cases: the fourth column is the free-run from the rewritten history with NO write at the edit frame, versus the default fourth column, which is the free-run from the rewritten history PLUS the inverse-map write at the edit frame |
| `history_frames.{pdf,png}` | appendix piece, the same three cases: all 20 history frames as Original frames (`obs_hist`) / Rewritten frames (`obs_cf`) / Counterfactual render (`cf_clean`); no edit-frame line (the edit frame is the frame after the last row); the first row of the rewritten column is the kept original frame 0; 5.42 x 3.12 in |
| `pieces/case<idx>_{gt,unedited,im,hist_im,hist,frames_original,frames_rewritten,frames_cfrender}.pdf` | one PDF per column per drawn case (main columns 1.24 x 0.97 in, frames columns 1.68 x 0.84 in); `pieces/legend_key.pdf` and `pieces/legend_key_positions.pdf` the two keys; 26 files |
| `history_rewrite_paper.json` | sidecar: run, instance, block, IM point, what each column draws, the selection rule with threshold, eligible cases and the drawn cases with their displacements, per-case locators, every arm's Edit Index, fidelity and per-step curve, the history RMSEs, the file list |

The Rayworld predictive-quality figure that used to be drawn here is `paper/figs/predictive_quality/rayworld.py`
(it reads the same npz and imports this folder's panel code).

**Columns.** Time runs downward in every panel. Above the dashed edit-frame line every column shows the same
frames: the ground-truth pre-edit history, the original observed frames 12..19 (`draw_paper.py` asserts the
four context arrays of a row are identical). Below the line each column shows its own free-run over frames
20..34: *Ground truth* the clean edited-world rollout; *Unedited* the model's free-run with no edit;
*Single-point edit* the canonical IM arm — one inverse-map overwrite at residual point 6 at the edit frame,
free-run from there; *History rewrite* the free-run from the rewritten history (the model's own predictions
with the counterfactual state written at every step) plus the same write at the edit frame (`hist+IM`); in
`history_rewrite_histonly` the write is dropped (`hist`). The cyan line is the edited disc's origin position,
the pink line its destination position, both as ray centres at the edit frame (the scorer's ghost / target
zones). Panels are drawn as `paper/figs/qualitative_edits/make_figure.py` draws them: `gray` on the dark
panel, fixed 0-1 range, nearest, thin frame, white page.

**Row selection.** Random among the edits large enough to see. Eligible = the bench cases whose teleport
displacement |target_x − ghost_x| is at least **20 rays** (the scorer's ray-zone centres at the edit frame;
the sensor is a pinhole fan, no wrap; the median of the 32 is 19.5 rays; case 29 has no visible origin and is
ineligible): **15 of 32** cases, 1, 2, 4, 6, 7, 10, 12, 14, 18, 19, 24, 25, 27, 28, 31. Then
`numpy.random.default_rng(0).choice(eligible, 3, replace=False)` over the eligible cases in ascending index
order, sorted: cases **14, 18, 25** with displacements **37.5, 29.5 and 39.0 rays**. The frames piece uses the
same cases. Stated here and in the sidecar.

**Caption facts** (all from `scores.json`, the scorer's own cards; ray-zone Edit Index,
`pim.metrics.zone_editability`, on `noise_ablation/L-dw-noiseless-20m`, dw-noiseless, Cartesian block, IM at
residual point 6 = the scorer's best IM arm, n = 32 cases, one seed):

| arm | Edit Index at the edit frame | fidelity ratio | Edit Index at step 1 | at step 14 | mean over 15 steps |
|---|---|---|---|---|---|
| unsteered | −0.93 | 1.00 | −0.93 | −0.82 | −0.87 |
| IM (single-point edit) | +0.61 | 0.36 | −0.63 | −0.74 | −0.65 |
| hist+IM (history rewrite) | +0.63 | 0.36 | +0.59 | +0.23 | +0.46 |
| hist (rewritten history, no write) | +0.65 | 0.48 | +0.57 | +0.21 | +0.43 |

The single-point edit lands (+0.61) and is forgotten one step later (−0.63); the rewritten history holds the
edit through the horizon (+0.59 at step 1, +0.23 at step 14) and the write on top adds little (+0.63 vs +0.65
without it at step 0; both decay alike). RMSE of the history frames 1..19 against the clean counterfactual
render (the `history_frames` figure): original observed frames 0.264, rewritten frames 0.114 (rewritten vs
original 0.255). Per-case standard errors of the step-0 Edit Index are 0.02 to 0.04 (in `scores.json`). One
seed, 32 cases: an illustration of the mechanism, not a table number.

**Caveats.** The rewritten frames are the model's own generations and are visibly blurrier than the clean
render (`history_frames`). The locators mark the edit-frame ray centres only; the discs drift with their
velocity along the rollout. PDFs are cropped to the content with no padding (`ps.save`), so the spacing around
a figure is set in LaTeX; the text is 8-9 pt at the printed size.
