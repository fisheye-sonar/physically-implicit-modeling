# history_rewrite — persistent transformer edits by rewriting the history (appendix)

Two scripts, compute and drawing kept apart. `make_figure.py` (GPU, about 15 s) computes everything and
writes `scores.json`, three internal waterfalls and the array file `.scratch/history_rewrite_arrays.npz`;
`draw_paper.py` (CPU, 3 s) reads that file and draws the paper figure, its variants and the pieces. Both
import only `pim.*` and `paper/figs/paper_style.py`; the run they read is
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
seed — an illustration, not a table number). Outputs beside the script: `A_original_history_arms.png` ·
`B_rewritten_history_arms.png` · `C_history_vs_counterfactual_render.png` (internal waterfalls through
`pim.figures.waterfall_grid`, dark page, verbose titles — NOT paper style) · `scores.json` (every arm's
scorecard + the history RMSEs). Since 2026-09-21 it also saves `.scratch/history_rewrite_arrays.npz`:
`obs_hist` (32, 20, 128) the original observed history · `obs_cf` the rewritten history · `cf_clean` the
simulator's clean render of the counterfactual history · `gt_roll` (32, 15, 128) the clean edited-world
rollout · `gt_unedited_roll` the clean unedited continuation (the scorer's ghost trajectory) · `roll_unsteered`,
`roll_IM`, `roll_hist+IM`, `roll_hist` the four free-runs · `target_x`, `ghost_x` the destination / origin ray
centres per case · `edit_object`, `s_post` · `scores_json` (the text of `scores.json`).

## Paper figure — `draw_paper.py`

`.pim/bin/python paper/figs/history_rewrite/draw_paper.py` (CPU; needs the npz above). Every output is a
vector PDF (Arial embedded as TrueType, the observation strips the only rasters, nearest interpolation at
600 ppi so all 128 rays survive, cropped to the content with no outer padding) with a 300-dpi PNG preview
beside it. Round 2 (2026-09-21, after Sevan's review): every column shows the real pre-edit history above
the line; the rewritten frames moved to their own appendix piece; legend wording "origin position" /
"destination position".

| file | what it shows |
|---|---|
| `history_rewrite_top3.{pdf,png}` | **recommended.** Three edits (rows) x four columns; the last 8 original observed frames above the edit-frame line in EVERY column, all 15 free-run steps below; 5.42 x 3.50 in at the printed size |
| `history_rewrite_top3_k10.{pdf,png}` | the same cropped to the first 10 free-run steps (5.42 x 2.87 in) |
| `history_rewrite_top3_histonly.{pdf,png}`, `..._k10_histonly` | the hist-alone variant: the fourth column is the free-run from the rewritten history with NO write at the edit frame, versus the default fourth column, which is the free-run from the rewritten history PLUS the inverse-map write at the edit frame |
| `history_rewrite_random3*.{pdf,png}` | the same four variants on three random cases (seed 0) — the honest control; one of them (case 16, 7.5 rays) is a barely visible edit |
| `history_frames_{top3,random3}.{pdf,png}` | appendix piece: all 20 history frames of the same cases as Original frames (`obs_hist`) / Rewritten frames (`obs_cf`) / Counterfactual render (`cf_clean`); no edit-frame line (the edit frame is the frame after the last row); the first row of the rewritten column is the kept original frame 0 |
| `prediction_quality.{pdf,png}` | stretch: half width, Ground truth (clean unedited world) against the model's free-run, no edit, on the random-3 cases; the dashed line marks where the free-run starts |
| `pieces/case<idx>_{gt,unedited,im,hist_im,hist}.pdf` | every main-figure panel of the six selected cases as its own PDF (8 history frames + 15 steps, 1.24 x 0.97 in) |
| `pieces/case<idx>_frames_{original,rewritten,cfrender}.pdf` | every history_frames panel (20 frames, 1.68 x 0.84 in) |
| `pieces/legend_key.pdf`, `pieces/legend_key_positions.pdf` | the keys of the main figure and of the frames figure |
| `history_rewrite_paper.json` | sidecar: run, instance, block, IM point, what each column draws, case indices per selection with the rule and the displacements, per-case locators, every arm's Edit Index, fidelity and per-step curve, the history RMSEs, the file list |

**Columns of the main figure.** Time runs downward in every panel. Above the dashed edit-frame line every
column shows the same frames: the ground-truth pre-edit history, the original observed frames 12..19
(`draw_paper.py` asserts the four context arrays of a row are identical). Below the line each column shows
its own free-run over frames 20..34: *Ground truth* the clean edited-world rollout; *Unedited* the model's
free-run with no edit; *Single-point edit* the canonical IM arm — one inverse-map overwrite at residual
point 6 at the edit frame, free-run from there; *History rewrite* the free-run from the rewritten history
(the model's own predictions with the counterfactual state written at every step) plus the same write at
the edit frame (`hist+IM`); in the `_histonly` variant the write is dropped (`hist`). The cyan line is the
edited disc's origin position, the pink line its destination position, both as ray centres at the edit frame
(the scorer's ghost / target zones). Panels are drawn as `paper/figs/qualitative_edits/make_figure.py` draws
them: `gray` on the dark panel, fixed 0-1 range, nearest, thin frame, white page.

**Row selection.** `top3`: the three of the 32 bench cases with the largest teleport displacement
|target_x − ghost_x| in rays (the sensor is a pinhole fan, no wrap): cases **4, 2, 1** with 91.5, 75.0 and
61.0 rays (median 19.5; case 29 is excluded because its origin is not visible at the edit frame). `random3`:
`numpy.random.default_rng(0).choice(32, 3, replace=False)`, sorted: cases **16, 19, 25** (7.5, 44.5, 39.0
rays). The frames figures use the same cases; the stretch figure uses the random-3 cases. An editorial rule
for the main figure, stated here and in the sidecar.

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
render (the `history_frames` figures): original observed frames 0.264, rewritten frames 0.114 (rewritten vs
original 0.255). Per-case standard errors of the step-0 Edit Index are 0.02 to 0.04 (in `scores.json`). One
seed, 32 cases: an illustration of the mechanism, not a table number.

**Caveats.** The rewritten frames are the model's own generations and are visibly blurrier than the clean
render (see `history_frames_top3`, where the bright disc of case 4 is wider than in the render). The locators
mark the edit-frame ray centres only; the discs drift with their velocity along the rollout. PDFs are cropped
to the content with no padding (`ps.save`), so the spacing around a figure is set in LaTeX; the text is
8-9 pt at the printed size.
