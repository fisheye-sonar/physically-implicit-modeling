# predictive_quality — the trained models predict well before any edit (appendix)

Two figures, one per environment. Neither script loads a model: both read caches that other figure scripts
computed. Every PDF is vector (Arial embedded as TrueType; the only rasters are the Rayworld observation and
error strips, nearest interpolation at 600 ppi so all 128 rays survive), cropped to the content with no outer
padding (`paper_style.save`), with a 300-dpi PNG preview beside it. No number appears inside either figure;
the only text is the row labels, the axis words, the key, the colour-bar label and the Othello panel letters.

| file | what it shows |
|---|---|
| `rayworld.{pdf,png}` | Rayworld, three panels of two held-out sequences each under bold letters: (a) Standard, `runs/noise_ablation/L-dw-noiseless-20m` on dw-noiseless; (b) Blink, `runs/blink_ablation/L-dw-blink-20m` on dw-blink; (c) 5-ray, `runs/ray_ablation/L-dw-5ray-20m` on dw-5ray. Three rows: Ground truth (the simulator's clean render of the UNEDITED world continuing), Prediction (the model's free-run from the observed frames) and Difference (prediction − truth on the canonical signed-error map); in every strip the last 8 observed frames above the dashed free-run-start line and the 15 free-run steps below it, time downward; the colour bar once beside the difference row; 5.49 x 2.96 in |
| `rayworld.json` | sidecar: per panel the run, instance, source arrays, pool size, ray count and cases; the rule; what each row draws (the difference's definition and scale); the file list |
| `compute_rayworld.py` | GPU step for panels (b) and (c): the canonical calls only, cached to `.scratch/predictive_quality_<instance>.npz` |
| `othello.{pdf,png}` | Othello, four rule-variant columns (a) standard, (b) adjacent-flip, (c) adjacent-noflip, (d) standard-noflip; three example blocks stacked vertically, each a (legal moves / model) row pair: the board tinted by its true legal set (uniform over the legal squares, the Bayes-optimal target) over the same board tinted by the trained model's unedited next-move distribution; a thin gap between the blocks; 5.55 x 7.90 in (a full-page appendix figure) |
| `othello.json` | sidecar: cache column, run, instance, the three bench case ids per variant with board length and legal-set size, tint parameters, the file list |
| `pieces/rayworld_<instance>_case<idx>_{gt,pred,diff}.pdf` (18), `pieces/rayworld_key.pdf`, `pieces/rayworld_colorbar.pdf` | every Rayworld strip (0.68 x 0.78 in), the key and the colour bar |
| `pieces/othello_<variant>_<case>_{legal,model}.pdf` (24) | every Othello board at 2.0 in (PNG previews beside them are gitignored) |

`pieces/` holds only what the two figures use.

## Rayworld — `compute_rayworld.py` and `rayworld.py`

`.pim/bin/python paper/figs/predictive_quality/compute_rayworld.py` (GPU, one model at a time, about 5 s), then
`.pim/bin/python paper/figs/predictive_quality/rayworld.py` (CPU, a few seconds).

**Arrays.** Panel (a) reads the SAME array file as `paper/figs/history_rewrite/draw_paper.py`:
`.scratch/history_rewrite_arrays.npz`, written on the GPU by `paper/figs/history_rewrite/make_figure.py` for
the 32 canonical bench cases of the Standard run (if the file is missing, run that script first). Panels (b)
and (c) read `.scratch/predictive_quality_dw-blink.npz` and `.scratch/predictive_quality_dw-5ray.npz`, written
by `compute_rayworld.py` through exactly the calls `make_figure.py` uses for dw-noiseless and nothing else:
`dwb.load_bench(model, n=32, target="full", basis_name="cartesian", instance=inst)` (the first 32 selected
bench cases), `dwa.unsteered_rollout(model, b)`, `b.zones.gt_unedited_traj`, `b.obs[:, :EF]`. In every panel:
`obs_hist` the observed frames 0..19, `gt_unedited_roll` the simulator's clean unedited continuation over frames
20..34 (the scorer's ghost trajectory), `roll_unsteered` the model's free-run from frame 20 with no edit. The
Ground truth and Prediction strips come from `draw_paper.py`'s panel code, imported by path (`gray` on the dark
panel, fixed 0-1 range, nearest, thin frame, white page). None of the three instances has observation noise,
so on (a) and (c) the observed frames above the line equal the clean render.

**Blink (b).** The observed context is drawn as the model saw it: an object vanishes for a run of frames and
the frame before a blackout and its last frame carry the 0.5 marker on one edge ray (ray 0 for object 0, the
last ray for object 1). The free-run starts from those observed frames. The ground truth is the unedited world
rendered under the same blackout schedule and markers (`b.zones.gt_unedited_traj` applies the schedule), so a
disc that is hidden over the horizon is dark in both rows and the difference row shows only mistakes.

**5-ray (c).** Every strip is 5 rays wide, stretched to the same panel width as the 128-ray strips (nearest
interpolation): five fat columns, disc radius 1.0 (the coarse-ray family's geometry).

**Difference row.** prediction − truth per ray over the free-run steps, drawn with `error` and `_panel`
imported from `paper/figs/qualitative_edits/make_figure.py` (the prediction is clipped to [0, 1] first, that
module's default; `raw=True` would be the scorer's unclipped quantity) on `pim.figures.waterfall.DIFF_CMAP`:
under-prediction red, over-prediction green, zero error the dark background, fixed ±1 scale as the qualitative
figures' `_diff` variant uses. Above the line the row is left empty (dark): those frames are observations, not
predictions. The colour bar ("prediction − truth", ticks −1, 0, 1) sits once beside this row.

**Cases.** Two per panel: one `numpy.random.default_rng(0)` per instance, `choice(32, 2, replace=False)`,
sorted. The pool size and seed are the same for the three instances, so the indices coincide: bench cases
**20, 26** of dw-noiseless for (a), **20, 26** of dw-blink for (b), **20, 26** of dw-5ray for (c) (different
sequences, different benches). No edit is involved, so no displacement filter (the history-rewrite figure
filters small edits; this one does not need to).

**What it shows (for the caption).** The free-run tracks the clean unedited world over the 15 steps; the
difference row is dark except thin red or green slivers along disc edges, a ray or two where an edge is
predicted slightly early or late. On Blink (b) the model carries a hidden disc through its blackout and
predicts its reappearance, so the difference stays dark there too. On 5-ray (c) the picture is honest about
the coarse sensor: a mistake is a whole fat ray. In case 26 the free-run never moves the bright disc onto the
ray it reaches at the free-run start (red on that ray over the whole horizon) and keeps it on the ray it has
left (green over the last half); in case 20 it keeps a disc lit after it has left its ray. As a descriptive check on the drawn arrays (not a
canonical metric), |prediction − truth| exceeds 0.1 on 0.8 / 1.3 % of the free-run rays for (a) cases 20 / 26,
1.6 / 0.5 % for (b), and 5.3 / 30.7 % for (c). The run's prediction-skill numbers for the text come from its
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
