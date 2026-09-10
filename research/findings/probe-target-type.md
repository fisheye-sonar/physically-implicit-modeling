# Probe-target type — regression vs categorical, both directions (2026-09-09 → 10, overnight)

**Status: `in progress` — the chain is running (`scripts/drivers/probe_targets.sh`, unit
`probe_targets`, logs `logs/probe_targets/`). Numbers below are placeholders until it lands.**

**Question.** The grid-target control (`grid-target-control.md`) asked whether Othello's
CATEGORICAL probe target is what makes it editable, by giving discworld a categorical target
on `L-dw-noiseless-20m` (a 16 × 8 grid, coarser than the observation resolves). Two gaps
remained: (i) the other direction — Othello read by a REGRESSION probe — and (ii) a discworld
categorical target that is exactly what the observation resolves, on an instance where that
is cheap: dw-8ray, whose renderer produces exactly **30 distinct single-disc appearances**
(runs of lit rays), i.e. the observation-exact partition IS a 30-cell target.

## Design

- **Othello, regression target `mine_signed`** — +1 mine, 0 blank, −1 theirs, one value per
  tile, 64-output probes (LIN closed-form, MLP-128), the SAME information and frame as the
  canonical 3-way target. Editors through the regression machinery: PI drives the tile's
  read-out to ±1 (z-space, y-affine), ND adds the tile's probe row with the sign of the flip
  (constant magnitude, hence sound), GS the MSE spec on the tile. Same bench, guard and
  scorecard. All three Othello runs; random-init and observation floors on all three
  instances. Row `mine_signed` in the master table.
- **dw-8ray, appearance target** — cell = (first ray, last ray) of the disc's lit run; 30
  realisable cells (lengths 1–5; the empty appearance never occurs); ray-based by
  construction; NOT a product grid (a two-ray run spans depths ~6–11). Every teleport of the
  filtered 192-case bench changes cell. Both 8-ray models: the frame model
  (`L-dw-8ray-20m`, ray-zone rollout scoring) and the token model (`L-dw-8ray-tok-20m`,
  frame-set step-0 scoring), probes on 200k sequences × 50 epochs (`GRID_PROBE_RECIPE`),
  random-init floors for both architectures and the right-aligned observation floor.
- **Resolution sweep** (one probe set per model, no floors): `appearance-d2`, `-d3` (each run
  split into depth bands the frame does NOT resolve), `grid-16x8` (the noiseless bridge);
  if time allowed before 06:00 PT: `appearance-lat` (coarser than the frame: runs merged by
  centre, 15 cells), `grid-8x4`, `grid-32x16`.

## Results

_(filled in when the chain lands: `runs/<run>/scores.json["bases"][<target>]`, the master
tables, `logs/probe_targets/headline_*.txt`)_

| run | target | skill LIN / MLP | unedited | PI | ND | GS |
|---|---|---|---|---|---|---|
| L-oth-20m | mine_signed | | | | | |
| L-oth-20m-mse | mine_signed | | | | | |
| L-oth-noflip-20m | mine_signed | | | | | |
| L-oth-adjacent-20m | mine_signed | | | | | |
| L-dw-8ray-20m | appearance | | | | | |
| L-dw-8ray-tok-20m † | appearance | | | | | |

## Reading

_(pending)_

## Provenance

Code: `pim/environments/discworld/grid_target.py` (`AppearanceTarget`, `covered_rays`),
`pim/environments/othello/data.py::signed_mine`, the regression branches in
`pim/environments/othello/arms.py`, the categorical branch of
`pim/environments/discworld/token_bench.py`, `scripts/fit_probes.py`, `master_eval`
SETTINGS `dw_extra_targets` / `oth_extra_targets`. Tests: `tests/test_probe_targets.py`.
Geometry check: 30 appearances on dw-8ray vs 2,883 on the 128-ray noiseless instance.
