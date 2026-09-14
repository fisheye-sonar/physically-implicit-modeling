# 2026-09-12 — Discworld rescored under the PRE-dynamics write target: nothing moves

**What changed.** The write target every discworld editor aims at is now the pre-dynamics
state — `pos[EF] − v·dt` for the edited object, the current state (frame EF−1) for the other
— instead of the post-dynamics state `pos[EF]` (GOTCHAS 2026-09-12; commit b7f741f; test
`tests/test_bench_target_alignment.py`). References, zones, probes, Othello untouched. All 19
discworld runs rescored by `master_eval` under `EVAL_VERSION_BY_ENV["discworld"] =
"2026-09-12.1"` (unit `rescore_dw_alignment`, 13:15 → 14:20 PT, 65 min), tables rebuilt.
Pre-fix scores beside each run as `scores.pre-alignment-2026-09-12.json`.

**Result: every number within noise of its predecessor.** Unedited floors identical (the
references did not change). Best-arm Edit Index per editor moved by ≤ 0.06 in every block
(median |Δ| ≈ 0.005); guards by ≤ 0.05 except two categorical blocks on dw-noiseless where GS
crossed 1.0 downward (grid-8x4 1.11 → 0.95, appearance-lat 1.08 → 0.87). No conclusion, no
ordering, no reported headline changes.

**Why so little, when a perfect edit had been capped at +0.70.** The cap bound a PERFECT
write. The actual writes land far from either reference (PI +0.2…+0.3 on the regression
target, guards ≈ 2), and a one-step offset of 0.077 world units — 3 % of a 2.6-unit teleport,
0.15 radii — is invisible against errors of that size. The offset would have mattered for an
editor near the ceiling; none is. The fix is still required for the metric to mean what the
paper says (edit the pre-dynamics state, compare the post-dynamics rendering), and it is
what makes the discworld and Othello constructions the same statement.

## Before → after, best arm per editor (Edit Index; guard in parentheses), frustum-basis blocks

| run | block | PI | ND | GS |
|---|---|---|---|---|
| L-dw-20m | frustum | +0.199→+0.198 (1.98→2.00) | −0.017→−0.017 (2.97) | −0.221→−0.222 (0.94) |
| L-dw-noiseless-20m | frustum | +0.233→+0.229 (1.95→1.97) | −0.028→−0.028 (3.51) | −0.099→−0.092 (0.99→1.00) |
| L-dw-noiseless-20m | appearance-fac | +0.014→+0.017 (1.95→1.90) | +0.628→+0.614 (0.78→0.79) | +0.352→+0.330 (0.68→0.70) |
| L-dw-noiseless-20m | grid-16x8 | +0.126→+0.111 (1.58) | +0.373→+0.367 (0.91→0.92) | +0.289→+0.294 (0.87→0.88) |
| L-dw-8ray-20m | frustum | +0.282→+0.284 (0.90) | −0.148→−0.148 (1.15) | −0.064→−0.067 (0.82) |
| L-dw-8ray-20m | appearance | +0.429→+0.421 (0.76→0.77) | +0.429→+0.411 (0.91→0.93) | +0.614→+0.586 (0.39→0.43) |
| L-dw-8ray-20m | appearance-fac | +0.414→+0.408 (0.71→0.72) | +0.504→+0.507 (0.86) | +0.464→+0.460 (0.53→0.54) |
| L-dw-8ray-tok-20m | frustum | +0.006→+0.006 (0.74) | — | −0.102→−0.091 (0.77→0.78) |
| L-dw-8ray-tok-20m | appearance | +0.260→+0.231 (0.53→0.56) | +0.439→+0.403 (0.41→0.45) | +0.575→+0.541 (0.31→0.35) |
| L-dw-8ray-tok-20m | appearance-fac | +0.235→+0.227 (0.54→0.55) | +0.298→+0.287 (0.50→0.52) | +0.403→+0.379 (0.43→0.45) |
| L-dw-blink-20m | frustum | +0.215→+0.216 (1.78→1.79) | −0.030→−0.030 (3.50) | −0.088→−0.104 (1.00→1.02) |
| L-dw-blink-20m | appearance-fac | +0.023→+0.013 (2.44→2.52) | +0.531→+0.525 (0.94) | +0.332→+0.293 (0.95→0.98) |
| L-dw-5ray-20m | appearance | +0.536→+0.531 (0.78) | +0.474→+0.476 (0.85) | +0.638→+0.623 (0.40→0.43) |
| R-dw-8ray-20m | frustum | +0.168→+0.171 (1.10→1.09) | −0.278→−0.278 (0.93) | −0.605→−0.621 (0.91→0.92) |
| training curve (8 checkpoints) | frustum | all within ±0.005 | all within ±0.001 | all within ±0.03 |
| noiseless seeds (×3) | frustum / appearance-fac | within ±0.02 | within ±0.02 | within ±0.03 |

Every other categorical / resolution-sweep block (appearance-d2/-d3/-lat, grid-4x2 … 32x16,
pos@appearance) sits in the same band; the largest single move is GS on the token model's
appearance-d2 block, +0.585 → +0.530. Two floors shifted in the third decimal where the
cell-changing case list changed by a case or two under the new target cell (grid-6x5, grid-10x3).

## What the record does with this

- Every finding quoting a discworld editor number carries a one-line banner: rescored
  2026-09-12, numbers within ≤ 0.06, `scores.json` is the current value. No superseding
  entries per number — there is nothing to supersede.
- `edit-direction-alignment.md`'s Haufe-edit and patch numbers come from experiment scripts
  that used the old target; they are NOT rescored and the banner says so. The effect there
  would be of the same size.
- The prediction made before the run ("128-ray editor numbers up modestly") was wrong, and
  wrong for a reason worth keeping: the target offset only matters near the ceiling.
