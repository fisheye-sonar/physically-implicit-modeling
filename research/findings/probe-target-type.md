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
| L-dw-8ray-20m | **appearance** | 0.893 / 0.904 | −0.909 | **+0.429 / 0.76** (pt 3 α 20) | **+0.429 / 0.91** (pt 5 α 12) | **+0.614 / 0.39** (pt 0 α 0.5) |
| L-dw-8ray-20m | frustum regression (canonical) | 0.950 / 0.981 | −0.911 | +0.282 / 0.90 | n/a (−0.148 / 1.15) | −0.064 / 0.82 |
| L-dw-8ray-tok-20m † | **appearance** | 0.900 / 0.907 | −0.763 | **+0.260 / 0.53** (pt 3) | **+0.439 / 0.41** (pt 3) | **+0.575 / 0.31** (pt 2) |
| L-dw-8ray-tok-20m † | frustum regression (canonical) | 0.968 / 0.980 | −0.779 | +0.006 / 0.74 | n/a | −0.102 / 0.77 |

Best arms (Othello regression): PI pt 5 α 3 / pt 4 α 3; ND pt 4 α 0.5 / pt 4 α 0.1; GS pt 0
α 0.05 on both editable runs — GS's best sits one step from the small end of its grid
(0.02 … 0.5), so its lower value may be grid-limited; PI and ND are interior. Tripwire
clean everywhere.

**dw-8ray, appearance — landed 2026-09-09 22:59 PT.** Floors: random-init LIN 0.877 / MLP
0.888 (frame arch), 0.870 / 0.873 (token arch); observation (right-aligned, 200k) LIN 0.540
/ MLP 0.887. Trained models sit ~0.02 above random-init: the appearance of a disc is close
to a read of the current frame, as the regression axis on this instance already showed.

**It is not a one-frame effect, and it is not one residual point.** By-step Edit Index of
the frame model's best arms (unedited −0.91 → −0.53 over the 15 steps): GS +0.61, +0.48,
+0.34, +0.24, +0.17, +0.11, … +0.06; PI +0.43, +0.36, +0.26, +0.18, … +0.08; ND +0.43,
+0.30, +0.17, +0.08, then ≈ −0.05 — the edit decays but stays on the edited side for the
whole rollout (the 16 × 8 grid on the noiseless run flipped to −0.3 at step 1). Per residual
point, appearance vs regression on the SAME model, same α grids, same bench:

| editor | appearance, points 0…8 (EI / fid) | regression, points 0…8 |
|---|---|---|
| PI | +0.03/0.77 · +0.33/0.90 · +0.40/0.86 · **+0.43/0.76** · +0.41/0.83 · +0.40/0.67 · +0.39/0.74 · +0.35/0.88 · +0.26/1.47 | −0.91 · +0.04 · +0.28/0.94 · +0.28/0.90 · +0.26 · +0.25/1.14 · +0.25/1.32 · +0.22 · +0.21/1.66 |
| ND | +0.31/0.91 · +0.19 · +0.35 · +0.39 · +0.41 · **+0.43/0.91** · +0.41 · +0.41 · +0.42/0.96 | −0.72 · −0.72 · −0.39 · −0.32 · −0.23 · −0.21 · −0.16 · −0.16 · −0.15 |
| GS | **+0.61/0.39** · +0.51/0.52 · +0.47/0.52 · +0.36/0.78 · +0.35/0.68 · +0.37/0.86 · +0.33/1.02 · +0.35/1.13 · +0.36/1.04 | −0.06 · −0.33 · −0.39 · −0.47 · −0.47 · −0.52 · −0.52 · −0.59 · −0.61 |

Every editor lands at essentially every point through the appearance probes, with the
guard below 1 through point 5; through the regression probes on the same residual stream,
GS and ND are negative everywhere and PI never exceeds +0.28. The token model repeats the
pattern (GS +0.54 … +0.58 at points 0–3, fid 0.31–0.33; ND +0.44 at points 3–4; PI +0.26 at
point 3; regression: PI ≈ 0, GS negative at every point).

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

3. **dw-8ray becomes editable through the observation-exact categorical target — the
   first discworld setting where the canonical editors relocate a disc and the model
   carries it** (status `observed`: one instance, one seed per model, two models). GS
   reaches +0.61 at fidelity 0.39 on the frame model and +0.58 / 0.31 on the token model,
   PI and ND +0.43 with the guard below 1; through the regression probes on the SAME
   residual stream, same α grids and bench, GS and ND are negative at every point. The
   effect is not a one-frame smear (by-step stays positive for 15 steps; the waterfall
   `runs/ray_ablation/L-dw-8ray-20m/figures/waterfall_edits_appearance.png` shows the disc
   at the target ray from the edit frame on, carried down the rollout in most rows, with
   grey smearing in the PI/ND columns) and not a point-0 artefact (every point lands).
4. **What changed relative to the grid control** (`grid-target-control.md`, ND +0.37 / GS
   +0.29, one-frame, reverting): the target's resolution. The 16 × 8 grid on the 128-ray
   instance was both coarser than the frame (laterally) and finer than the model's
   precision (in depth); the appearance partition is exactly the set of states the frame
   distinguishes, so the categorical write asks for a change the observation can express
   and nothing more. The resolution sweep (`appearance-d2/-d3` finer than the frame,
   `appearance-lat` coarser, the product grids) is the direct test of this reading.
5. **Caveats to carry.** (a) On dw-8ray the appearance IS nearly a read of the current
   frame (random-init floor 0.88 vs trained 0.90), so what is edited is close to the
   observation code; whether a deeper, carried state was moved is what the by-step decay
   (+0.61 → +0.06) and the smearing say only in part — the discrete-state discworld
   instance proposed on 2026-09-09 remains the environment-side test. (b) The 8-ray world
   is coarse: a 2–3-ray run is a large fraction of the frame, and radius 1.0 discs make
   every teleport of the filtered bench a cell change. (c) One seed.

## Provenance

Code: `pim/environments/discworld/grid_target.py` (`AppearanceTarget`, `covered_rays`),
`pim/environments/othello/data.py::signed_mine`, the regression branches in
`pim/environments/othello/arms.py`, the categorical branch of
`pim/environments/discworld/token_bench.py`, `scripts/fit_probes.py`, `master_eval`
SETTINGS `dw_extra_targets` / `oth_extra_targets`. Tests: `tests/test_probe_targets.py`.
Geometry check: 30 appearances on dw-8ray vs 2,883 on the 128-ray noiseless instance.
