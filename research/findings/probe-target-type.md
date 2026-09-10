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

**Othello — landed 2026-09-09 19:50 PT** (EI / fidelity; the canonical 3-way numbers in
brackets; skill is R² here vs 1 − err/majority there, so the two skill columns are NOT on
one formula):

| run | target | skill LIN / MLP | unedited | PI | ND | GS |
|---|---|---|---|---|---|---|
| L-oth-20m | mine_signed | 0.839 / 0.892 | −0.713 | **+0.618 / 0.23** (+0.608) | **+0.625 / 0.23** (+0.622) | +0.418 / 0.45 (+0.647) |
| L-oth-20m-mse | mine_signed | 0.803 / 0.846 | −0.817 | **+0.729 / 0.19** (+0.684) | **+0.724 / 0.18** (+0.739) | +0.479 / 0.35 (+0.730) |
| L-oth-noflip-20m | mine_signed | 0.957 / 0.997 | −0.823 | +0.042 / 2.08 (−0.001) | +0.055 / 2.62 (+0.086) | −0.053 / 2.02 (+0.020) |
| L-oth-adjacent-20m | mine_signed | 0.895 / 0.937 | −0.680 | −0.015 / 6.00 (−0.053) | +0.004 / 2.56 (+0.118) | −0.000 / 5.27 (+0.002) |
| L-dw-8ray-20m | appearance | | | | | |
| L-dw-8ray-tok-20m † | appearance | | | | | |

Best arms (regression): PI pt 5 α 3 / pt 4 α 3; ND pt 4 α 0.5 / pt 4 α 0.1; GS pt 0 α 0.05
on both editable runs — GS's best sits one step from the small end of its grid (0.02 … 0.5),
so its lower value may be grid-limited; PI and ND are interior. Tripwire clean everywhere.

## Reading

1. **Othello read by a REGRESSION probe is exactly as editable as Othello read by a
   categorical one.** PI and ND reproduce the canonical Edit Index to within a few
   hundredths on both editable runs, at the same fidelity (0.18–0.23); the two
   non-editable runs stay non-editable with the same failed guards. The probe target's
   type is therefore excluded in THIS direction as well as the discworld direction
   (`grid-target-control.md`): the same information, read out in either form, edits (or
   fails to edit) the same way. Decodability on the R² axis is lower (0.84–0.90 vs 0.96–0.98
   skill) because a linear map has to place three levels on one line — a statement about the
   axis, not the representation.
2. GS through the regression MLP is weaker (+0.42 / +0.48 vs +0.65 / +0.73). Its best arm is
   near the grid edge; extend the grid downward before reading anything into that number.

_(dw-8ray appearance results pending)_

## Provenance

Code: `pim/environments/discworld/grid_target.py` (`AppearanceTarget`, `covered_rays`),
`pim/environments/othello/data.py::signed_mine`, the regression branches in
`pim/environments/othello/arms.py`, the categorical branch of
`pim/environments/discworld/token_bench.py`, `scripts/fit_probes.py`, `master_eval`
SETTINGS `dw_extra_targets` / `oth_extra_targets`. Tests: `tests/test_probe_targets.py`.
Geometry check: 30 appearances on dw-8ray vs 2,883 on the 128-ray noiseless instance.
