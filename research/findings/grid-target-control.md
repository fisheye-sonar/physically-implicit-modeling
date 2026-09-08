# Grid-target control — a categorical discworld state is not what Othello has (2026-09-08)

**Question.** Othello's probe target is categorical (64 tiles × 3 classes); discworld's is
continuous. Is that the difference that makes Othello editable? Control: re-express the
discworld state as a 16 × 8 grid of cells, each {empty, centre of object 0, centre of
object 1} — Othello's shape of target, 384 logits — fit new probes of that shape on the
SAME trained model (`runs/noise_ablation/L-dw-noiseless-20m`), and run the canonical
editors through them. No new model. Prediction (Sevan): no meaningful change.

**Answer: mostly as predicted, with one real shift.** Editability does not come close to
Othello's, but the categorical target gives the gradient editors a better-conditioned
write: GS goes from −0.10 (regression target) to +0.29 at fidelity 0.87, and ND — a
categorical "move" edit that is legitimate on this target — reaches +0.37 at fidelity
0.91. Both are one-frame disturbances that revert by the next step, and the waterfall
shows no relocated disc. PI stays weak (+0.13 at fidelity 1.6). Decodability on the grid
axis is lower than on the regression axis because the metric is harsher, not because the
state is read worse: the direct grid probes beat the regression probes mapped onto cells.

## Setup

- Grid (`experiments/grid_target_control/scripts/grid.py`): uniform in the frustum basis
  (normalised ray coordinate u′, inverse depth 1/y) over the reachable region, so every
  cell is the same size in the observation's own coordinates; 1.56% of cell entries are
  non-empty; the nearer object wins a shared cell (0.05% of frames). The
  observation-exact partition would need ~2,000 cells; 128 keeps Othello's scale.
- Probes: LIN and MLP-128, 3-way per cell, all 9 residual points, on the first 200k
  sequences of `probe_250k`, 50 epochs (the large-corpus precedent; 1.65× the canonical
  gradient steps), held out by sequence; classification linear probes have no closed form,
  so both families are SGD. All persisted in the experiment's probe cache.
- Editors (`edit_grid.py`): PI = logit swap between the object's class and "empty" at the
  old and new cell, z-space re-solve; ND = the probe row of (new cell, class) minus the row
  of (old cell, class), per case; GS = cross-entropy steering with change mask on the two
  cells, β 0.2, 100 steps. Bench = the first 192 canonical cases whose teleport changes cell
  (2.1% dropped as no-ops). Same ray-zone Edit Index and fidelity guard.
- Extended α grids after the canonical ones pinned at their edges: PI to 100, ND to 12,
  GS to 5 (`scores/grid_edit_ext.json`).

## Results (frustum basis, `experiments/grid_target_control/scores/summary_ext.md`)

| target | skill LIN / MLP | unedited | PI | ND | GS |
|---|---|---|---|---|---|
| grid 3-way | 0.43 / 0.71 | −0.93 | +0.13 / fid 1.58 (pt 1, α 60) | **+0.37 / fid 0.91** (pt 3, α 8) | **+0.29 / fid 0.87** (pt 1, α 0.75) |
| grid, fid ≤ 1.1 | | | +0.08 / 1.10 (pt 1, α 20) | +0.37 / 0.91 | +0.29 / 0.87 |
| regression (canonical) | 0.96 / 1.00 | −0.92 | +0.23 / 1.95 (pt 1, α 175) | n/a | −0.10 / 0.99 |
| Othello L-oth-20m (reference) | | −0.82 | +0.61 | +0.62 | +0.65 |

The ND and GS optima are interior (ND α 6–8 plateau at +0.37, fidelity rising through 1
at α 12; GS α 0.5–1 plateau at +0.26–0.29). By-step index for every arm: +0.3 at step 0,
−0.3 to −0.5 at step 1, −0.6 from step 2 — the edit does not enter the carried history.

**Decodability on the grid axis** (skill = 1 − err / err(majority), majority = empty):
MLP 0.59 at point 0 → 0.71 plateau from point 4; LIN 0.11 → 0.43 at point 1 → 0.30. Per
object, the MLP puts a disc in its correct cell 84–86% of the time. The canonical
regression probes mapped onto the same cells (`regression_to_cells.py`, held-out
sequences): MLP 0.31–0.45 skill, 66–72% of objects in the right cell; LIN below majority.
So the grid probes read the discretised state BETTER than the R² 0.99 regression probes
do — the cells are simply finer than the model's positional precision (a depth bin is two
rays of apparent width), and the axis is harsh.

**Floors on the same target, same 200k sequences and split** (`grid_probes_random_init.json`,
`grid_probes_observation.json`):

| probe | trained (best pt) | random-init (best pt) | observation, right-aligned |
|---|---|---|---|
| LIN | 0.43 (pt 1) | 0.32 (pt 1) | 0.05 |
| MLP-128 | 0.71 (pt 7) | 0.58 (pt 1) | 0.54 |

Training adds +0.12 (MLP) / +0.11 (LIN) over the random-init reservoir and +0.17 over a
shallow read of the input — the same modest margins the regression axis shows (0.996 vs
0.94 / 0.91 for the MLP). The grid target changes the axis, not the picture.

## Reading

1. The categorical target is not the ingredient. Othello's editors land at +0.6–0.7 with
   the same probe shape; here the best guarded arm is +0.37 and it is a one-frame smear.
2. It does change the editors' conditioning: with a categorical target GS has a
   well-scaled loss and ND has a coherent per-case direction, and both stop being
   destructive (fidelity < 1) — the regression editors on this model only move the index at
   α ≥ 60 with fidelity ≈ 2. That is a statement about the EDITORS, not about the
   representation: a better-conditioned write finds a little more of whatever the
   regression write was finding, and no more.
3. Together with blink (carried state), tokens (interface), MSE Othello (objective), 8-ray
   (resolution) and no-flip (decodability ≠ use): the target's type joins the list of
   differences that do not explain the gap.

Caveats: the grid is one resolution (16 × 8); a coarser one (8 × 4, every cell ~4× larger
and closer to the model's precision) is one environment variable away
(`GRID=8x4 drivers/grid_target_control.sh`) and would test whether the +0.37 is limited by
cell precision. dw-pn04 (the noisy instance) was started first and stopped after five
probes; its partial state is parked in `_pn04_partial/`.

Assets: `experiments/grid_target_control/` (scripts, probes, scores, the waterfall
`outputs/waterfall_grid_edits.png`); logs `logs/grid_target_control/`.
