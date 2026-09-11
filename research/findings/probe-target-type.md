# Probe-target type — regression vs categorical, both directions (2026-09-09 → 10, overnight)

**Status: `observed` (2026-09-10 04:47 PT — the chain `scripts/drivers/probe_targets.sh`,
unit `probe_targets`, logs `logs/probe_targets/`, completed every stage except the last
sweep variant, `grid-32x16`, on which it was OOM-killed at the 45 GB cap; nothing partial
was written). One seed per model; one instance per environment for the new targets.**

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
**The resolution sweep (dw-8ray, 2026-09-10 00:06 → )** — one probe set per model, no
floors; EI / fidelity at each editor's best arm:

| target | cells | frame model: skill LIN / MLP · GS · PI · ND | token model †: skill LIN / MLP · GS · PI · ND |
|---|---|---|---|
| appearance (observation-exact) | 30 | 0.89 / 0.90 · **+0.61 / 0.39** · +0.43 / 0.76 · +0.43 / 0.91 | 0.90 / 0.91 · **+0.58 / 0.31** · +0.26 / 0.53 · +0.44 / 0.41 |
| appearance-d2 (finer: 2 depth bands) | 60 | 0.66 / 0.68 · +0.57 / 0.41 · +0.48 / 0.67 · +0.39 / 0.89 | 0.67 / 0.69 · +0.59 / 0.29 · +0.30 / 0.50 · +0.30 / 0.51 |
| appearance-d3 (finer: 3 depth bands) | 90 | 0.46 / 0.51 · +0.51 / 0.53 · +0.41 / 0.85 · +0.38 / 0.93 | 0.48 / 0.52 · +0.54 / 0.32 · +0.11 / 0.65 · +0.24 / 0.54 |
| grid-16x8 (product grid, the noiseless control's target) | 128 | 0.16 / 0.21 · +0.37 / 0.66 · +0.10 / 1.43 · +0.31 / 0.99 | 0.17 / 0.23 · +0.23 / 0.54 · +0.01 / 0.75 · +0.11 / 0.66 |
| appearance-lat (coarser: runs merged by centre) | 15 | 0.89 / 0.92 · +0.42 / 0.61 · +0.33 / 0.81 · +0.42 / 0.89 | 0.91 / 0.92 · +0.29 / 0.52 · +0.06 / 0.71 · +0.16 / 0.61 |
| grid-8x4 (product grid, coarse) | 32 | 0.46 / 0.51 · +0.35 / 0.80 · +0.31 / 0.82 · +0.31 / 0.94 | 0.47 / 0.53 · +0.20 / 0.59 · +0.06 / 0.69 · +0.10 / 0.66 |
| grid-32x16 (product grid, fine) | 512 | NOT RUN — the unit was OOM-killed at the 45 GB cap on this target's first fit (04:47 PT; 1,536 logits, label arrays 4× the next target's). Nothing partial was written. | |

Alignment, not cell count: `appearance-d3` (90 cells, run-aligned) and `grid-8x4` (32
cells, product) decode equally (LIN 0.46) but edit at GS +0.51 vs +0.35 on the frame model
and +0.54 vs +0.20 on the token model.

Reference: the SAME 16 × 8 grid on `L-dw-noiseless-20m` gave PI +0.13 / 1.58, ND +0.37 /
0.91, GS +0.29 / 0.87 (`grid-target-control.md`). **So the grid control's weak, reverting
result was a property of the TARGET (a product grid misaligned with what the frame
resolves), not of the instance**: on dw-8ray the grid reproduces the noiseless numbers
while the observation-exact partition edits at Othello's level, and editability degrades
monotonically as the target over-resolves the frame (GS +0.61 → +0.57 → +0.51 → +0.37 as
decodability falls 0.89 → 0.66 → 0.46 → 0.16).

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

## The resolution sweep on dw-8ray (2026-09-10, chains 1–2; `observed`, one seed)

Every categorical target below is fitted with one recipe (200k sequences × 50 epochs) and
scored with the same editors, α grids and 192-case bench; EI / fidelity at each editor's best
arm. The product grids (`grid-<nu>x<nd>`) are uniform in (u′, 1/y); `appearance` is the
frame's own partition (the runs of lit rays), `-d<k>` splits each run into k depth bands,
`-lat` merges runs by centre. `grid-6x5` and `grid-10x3` are Sevan's MISALIGNED controls:
the appearance partition's cell count with a structure that is not the runs of rays.

**Frame model `L-dw-8ray-20m`** (regression target, canonical frustum row, for reference:
skill 0.95 / 0.98, PI +0.28 / 0.90, ND n/a, GS −0.06 / 0.82):

| target | cells | skill LIN / MLP | PI | ND | GS |
|---|---|---|---|---|---|
| grid-4x2 | 8 | 0.75 / 0.78 | +0.41 / 0.73 | +0.24 / 0.92 | +0.31 / 0.85 |
| appearance-lat | 15 | 0.89 / 0.92 | +0.33 / 0.81 | +0.42 / 0.89 | +0.42 / 0.61 |
| **appearance** | **30** | 0.89 / 0.90 | **+0.43 / 0.76** | **+0.43 / 0.91** | **+0.61 / 0.39** |
| grid-6x5 (misaligned) | 30 | 0.47 / 0.54 | +0.32 / 0.87 | +0.29 / 0.92 | +0.34 / 0.66 |
| grid-10x3 (misaligned) | 30 | 0.44 / 0.50 | +0.36 / 0.85 | +0.34 / 0.95 | +0.34 / 0.63 |
| grid-8x4 | 32 | 0.46 / 0.51 | +0.31 / 0.82 | +0.31 / 0.94 | +0.35 / 0.80 |
| appearance-d2 | 60 | 0.66 / 0.68 | +0.48 / 0.67 | +0.39 / 0.89 | +0.57 / 0.41 |
| appearance-d3 | 90 | 0.46 / 0.51 | +0.41 / 0.85 | +0.38 / 0.93 | +0.51 / 0.53 |
| grid-16x8 | 128 | 0.16 / 0.21 | +0.10 / 1.43 | +0.31 / 0.99 | +0.37 / 0.66 |
| grid-32x16 | 512 | 0.01 / 0.04 | +0.14 / 1.22 | +0.31 / 0.97 | +0.42 / 0.68 |
| grid-64x32 | 2048 | LIN at the majority error (skill 0.00, points 0–2); fit stopped as uninformative (~8 h) | | | |

**Token model `L-dw-8ray-tok-20m` †** (frame-set index, step 0; regression row: skill 0.97 /
0.98, PI +0.01 / 0.74, GS −0.10 / 0.77):

| target | cells | skill LIN / MLP | PI | ND | GS |
|---|---|---|---|---|---|
| grid-4x2 | 8 | 0.75 / 0.79 | +0.03 / 0.74 | +0.05 / 0.72 | +0.09 / 0.71 |
| appearance-lat | 15 | 0.91 / 0.92 | +0.06 / 0.71 | +0.16 / 0.61 | +0.29 / 0.52 |
| **appearance** | **30** | 0.90 / 0.91 | **+0.26 / 0.53** | **+0.44 / 0.41** | **+0.58 / 0.31** |
| grid-6x5 (misaligned) | 30 | 0.49 / 0.55 | +0.03 / 0.72 | +0.07 / 0.69 | +0.19 / 0.60 |
| grid-10x3 (misaligned) | 30 | 0.45 / 0.51 | +0.05 / 0.70 | +0.11 / 0.65 | +0.23 / 0.57 |
| grid-8x4 | 32 | 0.47 / 0.53 | +0.06 / 0.69 | +0.10 / 0.66 | +0.19 / 0.59 |
| appearance-d2 | 60 | 0.67 / 0.69 | +0.30 / 0.50 | +0.30 / 0.51 | +0.58 / 0.29 |
| appearance-d3 | 90 | 0.48 / 0.52 | +0.11 / 0.65 | +0.24 / 0.54 | +0.54 / 0.32 |
| grid-16x8 | 128 | 0.17 / 0.23 | +0.01 / 0.75 | +0.11 / 0.66 | +0.23 / 0.54 |
| grid-32x16 | 512 | 0.01 / 0.05 | +0.02 / 0.75 | +0.10 / 0.66 | +0.18 / 0.58 |

**Figure** — `experiments/probe_targets/outputs/probe_target_sweep.png` (rendered by
`experiments/probe_targets/scripts/sweep_figure.py` through `pim.figures.sweep_figure`):
one row per run, Probe Skill then the Edit Index at each editor's best arm against cell
count, the three target families in colour, guard-failing arms hollow.

### Reading the sweep

1. **The observation-exact partition is a peak, on both models and for every editor.**
   Coarser than the frame (`appearance-lat`, 15 cells) is just as DECODABLE (0.89 vs 0.89)
   but edits worse (GS +0.42 vs +0.61; token +0.29 vs +0.58): a lateral-only cell leaves
   the run length, hence the depth, unspecified, so the write underdetermines the frame it
   has to produce. Finer than the frame (`-d2`, `-d3`) loses decodability quickly (0.89 →
   0.66 → 0.46) and editability gradually (GS +0.61 → +0.57 → +0.51 on the frame model;
   the token model holds GS at +0.58 → +0.54 while PI and ND fall).
2. **Alignment, not cell count** (Sevan's control). Two independent 30-cell product grids
   decode at half the skill and edit at about half the level of the 30-cell appearance
   partition (frame model GS +0.34 / +0.34 vs +0.61; token model +0.19 / +0.23 vs +0.58, PI
   and ND near zero). The whole product-grid family, from 8 cells to 128, sits in a flat
   band (frame model GS +0.31 … +0.37) regardless of resolution: a grid uniform in the
   frustum basis cuts through the appearance cells everywhere, so at any resolution most
   of its cells ask for a change the frame cannot express, or express only partly.
3. **Sevan's hypothesis holds on the fine side**, with a shape: past the frame's resolution
   the probe stops reading the target long before the editors stop landing (at 2048 cells
   the linear probe is at the majority baseline). The peak is at the frame's own partition.
4. **Under the product grids, editability is insensitive to decodability.** At 512 cells
   the probes read almost nothing (skill 0.01 / 0.04, one step above the majority baseline)
   and GS still lands +0.42 / 0.68 on the frame model, ND +0.31 / 0.97 — the same band as
   the 8-cell grid the probe reads at 0.75. A near-majority probe's logits still carry a
   position-dependent direction per cell, and GS follows it; what it cannot do is single out
   the frame's own partition. So Probe Skill is not what predicts editability here;
   alignment of the target with the observation is.
5. The token model's PI is fragile (+0.26 at the peak, near zero everywhere else) while its
   GS is robust to depth-splitting (+0.58 → +0.54) — same shape as on the frame model, with
   the categorical head amplifying the difference between a target the frame expresses and
   one it does not.

## The same gradient on dw-noiseless (2026-09-10, chains 4–5; `L-dw-noiseless-20m`, one seed)

Sevan's prediction: the 128-ray noiseless model benefits from a FINER grid than the 8-ray
model does, because its frame resolves position much more finely (its appearance partition
has 2,889 cells vs 30). Same recipe, same editors and α grids, canonical bench filtered to
cell-changing teleports. Regression row for reference: skill 0.97 / 0.99, PI +0.30 / 0.98,
GS −0.03 / 1.01 (unedited −0.94).

| target | cells | skill LIN / MLP | PI | ND | GS |
|---|---|---|---|---|---|
| grid-8x4 | 32 | 0.71 / 0.89 | +0.27 / 1.05 | +0.34 / 0.92 | +0.27 / 1.11 |
| grid-16x8 | 128 | 0.43 / 0.71 | +0.13 / 1.58 | +0.37 / 0.91 | +0.29 / 0.87 |
| appearance-lat | 233 | 0.04 / 0.63 | +0.04 / 1.68 | **+0.51 / 0.79** | +0.30 / 1.08 |
| grid-32x16 | 512 | 0.20 / 0.44 | +0.17 / 1.13 | +0.34 / 0.92 | +0.29 / 0.90 |
| grid-64x32 | 2048 | LIN at the majority error (points 0–2); fit stopped 14:20 PT at Sevan's call (~6 h for LIN + MLP at 6,144 logits) | | | |

### Reading the noiseless gradient

1. **Decodability does survive finer grids here, as Sevan predicted.** The MLP reads
   grid-32x16 at 0.44 (the 8-ray models: 0.04) and grid-8x4 at 0.89; the 128-ray frame
   resolves position finely enough that a 512-cell grid is still a readable target.
2. **Editability does not follow it.** ND sits at +0.34 … +0.37 and GS at +0.27 … +0.30 across
   32, 128 and 512 cells — the same flat product-grid band as on dw-8ray, one level lower —
   and PI degrades monotonically as the grid gets finer (+0.27 / 1.05 → +0.13 / 1.58 →
   +0.17 / 1.13, the guard failing at every resolution). A finer grid buys this model
   nothing an editor can use; the model does not "prefer" a resolution.
3. **The outlier is `appearance-lat`** (233 cells, depth dropped): linearly unreadable (LIN
   0.00–0.04 at every point) yet MLP 0.63, and the best ND on any discworld target so far
   (**+0.51 / 0.79**). The lateral partition is the one target here that is aligned with the
   frame (a run's centre is a ray index), so this is the dw-8ray alignment result reappearing
   through the one editor whose write is a fixed direction per cell — PI and GS are at or over
   the guard on it. The full appearance partition (2,889 cells, 22 GB label tensor) remains
   the untested case that would close the comparison with dw-8ray.
4. Where the two instances differ: on dw-8ray every product grid is misaligned with a coarse
   frame, so the whole grid family is flat AND weakly decodable; on noiseless the grids are
   decodable but equally flat. Editability tracks alignment with the observation, not the
   probe's skill — the same conclusion (§4) from a second instance.

Notes so far. (i) The LINEAR probe is at the majority on `appearance-lat` (233 narrow lateral
bands; in-sample gap 0.000, so not overfitting): a bounded interval of one coordinate is not
a half-space, and the finer the band the less a linear read-out can carve it — the same
collapse as `grid-32x16` on dw-8ray, and why PI (which needs the linear probe) is dead there
while the MLP reads it at 0.63. (ii) ND on the 233-cell aligned partition is the best editor
result on this run so far (+0.51 / 0.79 vs +0.37 / 0.91 on `grid-16x8`, +0.34 on `grid-8x4`),
in the direction of Sevan's prediction that the 128-ray model wants a finer,
observation-aligned partition; GS does not follow (+0.30, guard failing on every categorical
target here). The bench under `appearance-lat` drops only 2 same-cell teleports (21 under
`grid-8x4`). (iii) With `grid-32x16` in, the product grids on noiseless are FLAT from 32 to
512 cells — ND +0.34 / +0.37 / +0.34, GS +0.27 / +0.29 / +0.29 — exactly the 8-ray pattern,
while decodability at 512 cells holds up far better here (MLP 0.44 vs 0.04): the finer
frame carries the finer grid, but a finer PRODUCT grid still does not edit better. The one
lift so far is the observation-aligned `appearance-lat` (ND +0.51). Bench drops 1 same-cell
teleport at 32x16.

Provenance: `runs/{ray_ablation/L-dw-8ray-20m,interface_ablation/L-dw-8ray-tok-20m}/scores.json["bases"]`,
`logs/probe_targets*/headline_*.txt`, drivers `scripts/drivers/probe_targets{,_2,_3,_4,_5}.sh`.
Pending: `grid-64x32` on `L-dw-noiseless-20m` (chain 5, started 13:20 PT).

## The snapped regression target: is it the target or the read-out? (2026-09-10, `pos@appearance` on `L-dw-8ray-20m`)

Sevan's question after the sweep: does the categorical appearance row edit well because
its TARGET is the frame's own partition (alignment), or because a categorical READ-OUT (a
softmax over cells, edited by swapping two class logits) is a cleaner thing to write than a
coordinate? `pos@appearance` separates the two: the same 30 cells, but every position is
replaced by the centre of its cell (frustum basis) and read as the ordinary 4-output
regression — the regression probes, PI (pseudo-inverse), GS (MSE spec), the same 192
cell-changing cases as the appearance row, the canonical 30k recipe, floors inline.

| target (dw-8ray frame model) | kind | skill LIN / MLP | in-sample gap | PI | ND | GS |
|---|---|---|---|---|---|---|
| frustum (canonical) | regression, 8 outputs | 0.95 / 0.98 | 0.000 / 0.001 | +0.28 / 0.90 (α 100) | n/a | −0.06 / 0.82 |
| **pos@appearance** | **regression, 4 outputs (snapped)** | 0.96 / 0.99 | 0.000 / 0.001 | +0.34 / 1.15 (α 175); **+0.34 / 0.92 at α 100** | n/a | −0.13 / 0.84 |
| appearance | classification, 30 × 3 | 0.89 / 0.90 | 0.002 / 0.005 | +0.43 / 0.76 (α 20) | +0.43 / 0.91 | **+0.61 / 0.39** |

Floors for `pos@appearance` (dw-8ray, Transformer-L): observation right-aligned LIN 0.40 /
MLP 0.98; random-init 0.97 / 0.99 — the same picture as the canonical target (0.39 / 0.95;
0.96 / 0.98). No overfit: the in-sample gap is ≤ 0.001 at every point (Sevan's condition on
the 30k recipe).

**Reading.** Snapping the regression target changes nothing the editors can use. The
snapped row IS the regression row: PI lands only at the same enormous step (α ≈ 100, write
ratio 4–7, the read-out overshooting by 45 units) and GS is still negative. The gain of the
categorical row therefore does not come from the target being frame-expressible; it comes
from the categorical read-out and its edit — swapping two cells' logits — which the
regression pipeline cannot express even when its target values are the very same cell
centres. Two reasons this is unsurprising in hindsight, both worth recording:
1. The snap moves a position by 0.04 in u and 0.015 in 1/y on average against target
   standard deviations of 0.41 and 0.043 — a few percent of the variance. A linear probe of
   a continuous representation reads straight through to the underlying position (skill
   0.96, up from 0.95 only because the snapped target has slightly less variance to
   explain), so the fitted map, and hence the pseudo-inverse write, is the canonical one.
2. A categorical edit is a DIRECTION with a magnitude the class margin sets (Othello's
   flip); a regression edit is a displacement the pseudo-inverse must realise through a map
   whose null space is most of the residual stream. The categorical rows' α grid sits at
   ~20 with the read-out landing on 92% of cases; the regression α has to reach 100 to move
   the output at all, and by then it has degraded it.
The alignment result (§ sweep: exact partition ≫ misaligned 30-cell grids) still stands —
but it is a result about which categorical target to use, not a route to editability for a
regression probe. The next test on this axis is the converse: a categorical read-out on a
target that is NOT frame-aligned already exists (the product grids, GS ≈ +0.35), so the
remaining unknown is a categorical head on the snapped centres' cells with a coarser or
finer partition than the frame — i.e. the sweep itself. The regression side is closed.

Provenance: `runs/ray_ablation/L-dw-8ray-20m/scores.json["bases"]["pos@appearance"]`,
`runs/_baselines/dw-8ray/baselines.json` (`pos@appearance`), unit `snapped_appearance`
(`logs/snapped_appearance/`, 15 min), driver `scripts/drivers/score_pending.sh`; the target
is `grid_target.SnappedTarget` and applies to any partition on any instance by name.

## The factorised categorical target: does the read-out's gain need the joint cell? (2026-09-10, `appearance-fac` on `L-dw-8ray-20m`)

Sevan's follow-up to the snapped result: if the gain lives in the categorical read-out, does
it survive FACTORISING it? `appearance-fac` reads the same 30-cell partition object-wise —
per object one softmax over the run's CENTRE (15 classes: which pixel the disc's centre is
in) and one over its LENGTH (5 classes: how many rays wide) — 4 tiles × 20 classes instead
of 30 cells × 3. An edit is a per-tile class change on the edited object's tiles (PI swaps
old ↔ new at each moved tile, ND adds the summed row contrast, GS asks CE toward the new
labels); same 192 cell-changing cases, same recipe, same floors.

| target (dw-8ray frame model) | logits | skill LIN / MLP | floors: rand-init / obs-right | PI | ND | GS |
|---|---|---|---|---|---|---|
| appearance (cell-indexed, 30 × 3) | 90 | 0.89 / 0.90 | 0.88 · 0.89 / 0.54 · 0.89 | +0.43 / 0.76 (pt 3, α 20; landed 92%) | +0.43 / 0.91 (pt 5, α 12) | **+0.61 / 0.39** (pt 0, α 0.5) |
| **appearance-fac** (object-indexed, 4 × 20) | 80 | **0.94 / 0.94** | 0.92 · 0.94 / 0.48 · 0.94 | +0.41 / 0.71 (pt 1, α 35; landed 95%) | **+0.50 / 0.86** (pt 0, α 12) | +0.46 / 0.53 (pt 0, α 0.5) |
| pos@appearance (regression, snapped) | 4 | 0.96 / 0.99 | 0.97 · 0.99 / 0.40 · 0.98 | +0.34 / 0.92 | n/a | −0.13 / 0.84 |
| frustum (regression, canonical) | 8 | 0.95 / 0.98 | 0.96 · 0.98 / 0.39 · 0.95 | +0.28 / 0.90 | n/a | −0.06 / 0.82 |

**Reading.**
1. **The categorical gain survives factorisation.** PI is unchanged (+0.41 vs +0.43, both
   guard-passing, the read-out landing on 95% of cases), ND is the best ND on any discworld
   target so far (+0.50 / 0.86), GS drops from +0.61 to +0.46 but stays far above the
   regression rows (−0.06, −0.13) and above the product-grid band (+0.31 … +0.37). The
   factorised read-out is therefore the form to scale: on 128 rays it is 2 × (233 + lengths)
   classes against 2,889 × 3 for the joint cell.
2. **What the factorisation costs is GS's joint move.** GS on the joint cell can push one
   logit pair per cell and let the softmax settle the object's whole (centre, length) at once;
   factorised, it descends two separate cross-entropies whose optimum is reached at a smaller
   step (α 0.35–0.5, GS falling past α 0.75) — a coupling the joint cell gave for free.
3. **Decodability is higher, not lower, than the joint cell's** (0.94 vs 0.89): fifteen centre
   classes and five length classes are each easier than thirty joint cells, and the random-init
   floor rises with it (0.92 / 0.94 vs 0.88 / 0.89) — as everywhere on discworld, the position
   read is not what training adds. The observation LIN floor is lower (0.48 vs 0.54): a linear
   map of the frame to (centre, length) is harder than to the run's one-hot.
4. Together with the snapped result: target ALIGNMENT (the sweep), a CATEGORICAL read-out (the
   snapped control) and a per-tile class-swap EDIT (this row: the joint cell is not required)
   are what make the discworld model edit like Othello. The cheapest form that keeps all three
   is `<partition>-fac`.

Provenance: `runs/ray_ablation/L-dw-8ray-20m/scores.json["bases"]["appearance-fac"]`,
`runs/_baselines/dw-8ray/baselines.json` (`appearance-fac`), unit `appearance_fac`
(`logs/appearance_fac/`, 65 min: 15 + 16 min model probes, 15 + 16 min random-init, 1 min
observation, 1 min scoring), driver `scripts/drivers/probe_target_fit.sh`.

## The factorised target on the 128-ray model (2026-09-10 evening, `appearance-fac` on `L-dw-noiseless-20m`)

The scaled test the factorisation was built for: the noiseless instance's appearance partition
has 2,889 cells (8,667 logits as a joint cell — never fitted); factorised it is 233 centre
classes + 29 length classes per object, 4 tiles × 262 = 1,048 logits. Same recipe, floors and
192 cell-changing cases as every categorical row.

| target (noiseless frame model) | logits | skill LIN / MLP | floors: rand-init / obs-right | PI | ND | GS |
|---|---|---|---|---|---|---|
| frustum (regression, canonical) | 8 | 0.96 / 1.00 | 0.96 · 0.99 / 0.39 · 0.95 | +0.23 / 1.95 | n/a | −0.10 / 0.99 |
| grid-16x8 (joint cell) | 384 | 0.43 / 0.71 | — | +0.13 / 1.58 | +0.37 / 0.91 | +0.29 / 0.87 |
| appearance-lat (joint cell, 233) | 699 | 0.04 / 0.63 | — | +0.04 / 1.68 | +0.51 / 0.79 | +0.30 / 1.08 |
| **appearance-fac** (centre × length) | 1,048 | **0.43 / 0.79** | 0.30 · 0.59 / 0.12 · 0.52 | +0.01 / 1.95 | **+0.63 / 0.78** | **+0.35 / 0.68** |
| *Othello `L-oth-20m`, mine/theirs, for scale* | 192 | 0.98 / 0.98 | | +0.61 / 0.24 | +0.62 / 0.23 | +0.65 / 0.21 |

**Reading.**
1. **ND on the 128-ray model now matches Othello's** (+0.63 / 0.78 vs +0.62 / 0.23 — the
   guard is looser, the index the same), from a run whose regression rows have never edited
   (PI +0.23 at guard 1.95, GS −0.10). Sevan's prediction that noiseless wants a finer
   partition than 8-ray holds in this form: the observation-exact partition, factorised, is
   the target this model edits along. ND's α curve is a clean peak (α 1.5 → 6 → 12: +0.31 →
   +0.63 → +0.55, guard 0.58 → 0.78 → 0.93).
2. **Decodability here is what training added.** LIN 0.43 / MLP 0.79 against a random-init
   floor of 0.30 / 0.59 and an observation floor of 0.12 / 0.52 — the first discworld target
   on which the trained model sits clearly above BOTH floors (on 8-ray and on every regression
   row the random-init floor is within 0.02 of the trained probe). 233 lateral classes at
   128 rays are a hard linear read; the skill peaks at point 1 and drifts down, unlike 8-ray's
   flat profile.
3. **PI collapses** (+0.01, guard 1.95, write ratio 1,076): the pseudo-inverse of a
   1,048-row read-out is ill-conditioned, so the exact-jump write explodes. This is the
   pipeline's PI, unchanged; it says the factorised target at this width is ND's and GS's
   territory. A ridge / z-space-clipped PI would be an instrument change and is not made.
4. **GS is at the product-grid band's ceiling** (+0.35 / 0.68, α 0.35), consistent with the
   8-ray reading that factorisation costs GS the joint move.
5. Combined with the 8-ray rows: the factorised categorical read-out gives the best ND on
   both discworld instances (+0.50, +0.63), scales to the full 128-ray partition at a tenth
   of the joint cell's logits, and its raw rows are the best-aligned discworld subspace
   with the true edit direction (`edit-direction-alignment.md` Result 6). It is the target to
   carry forward — and the one to try on `L-dw-blink-20m`, where position is a carried
   variable but only the regression target has ever been probed.

Provenance: `runs/noise_ablation/L-dw-noiseless-20m/scores.json["bases"]["appearance-fac"]`,
`runs/_baselines/dw-noiseless/baselines.json`, unit `appearance_fac_noiseless`
(`logs/appearance_fac_noiseless/`, 72 min), driver `scripts/drivers/probe_target_fit.sh`.

## The factorised target on the blink model (2026-09-10 night, `appearance-fac` on `L-dw-blink-20m`)

The test of the "integration pressure" account of discworld editability with the working target
held fixed: dw-blink is the one instance where position is a CARRIED variable (an object leaves
the observation for ~5 frames at a time; the trained model reads its position at 0.96–0.98 ten
frames into a blackout where the observation and random-init floors have decayed to 0.46–0.78 —
`blink-ablation.md`), and it has never been probed with anything but the regression target,
which edits nowhere. Same partition, classes, recipe, floors and bench construction as the
noiseless row (dw-blink shares dw-noiseless's geometry).

| run, `appearance-fac` | skill LIN / MLP | floors: rand-init / obs-right | PI | ND | GS |
|---|---|---|---|---|---|
| `L-dw-noiseless-20m` (position always visible) | 0.43 / 0.79 | 0.30 · 0.59 / 0.12 · 0.52 | +0.01 / 1.95 | **+0.63 / 0.78** (pt 2, α 6) | +0.35 / 0.68 |
| `L-dw-blink-20m` (position carried through blackouts) | 0.46 / 0.68 | 0.22 · 0.45 / 0.07 · 0.39 | +0.02 / 2.44 | **+0.53 / 0.94** (pt 3, α 8) | +0.33 / 0.95 |
| `L-dw-blink-20m`, regression (canonical), for reference | 0.90 / 0.99 | 0.72 · 0.94 / 0.35 · 0.91 | +0.22 / 1.78 | n/a | −0.09 / 1.00 |

**Reading.** Blink edits under the factorised target — its first positive editability result of
any kind — at ND +0.53 with the guard just passing (0.94; +0.51 / 0.87 at α 6), GS +0.33 at the
guard's edge. It does NOT edit better than noiseless, whose position is never hidden: ND +0.53
vs +0.63, GS at the same level with a worse guard. The training margin over the floors is as
wide on blink as on noiseless (MLP +0.23 / +0.29 vs +0.20 / +0.27), and the extra pressure to
carry position bought nothing an editor can use. So the account "the environment must make the
model COMPUTE position, and then position edits" is not supported on discworld: blink makes the
model compute it in the strongest sense we can measure, and its editability under the target
that works is at or below the always-visible instance's. What decides editability across the
three discworld instances is the read-out (categorical, factorised, on the observation-exact
partition), and within that read-out the instances rank noiseless > blink > 8-ray on ND
(+0.63, +0.53, +0.50) — a narrow band, with the run whose frame is finest on top.

Provenance: `runs/blink_ablation/L-dw-blink-20m/scores.json["bases"]["appearance-fac"]`,
`runs/_baselines/dw-blink/baselines.json`, unit `appearance_fac_blink` (`logs/appearance_fac_blink/`,
72 min), driver `scripts/drivers/probe_target_fit.sh`. The canonical blink bench (first 192
cases, mostly visible at the edit frame) — the blink subsets under this target are not scored.

## The factorised target on the token model (2026-09-10 night, `appearance-fac` on `L-dw-8ray-tok-20m`)

| `L-dw-8ray-tok-20m` (frame-set Edit Index †) | skill LIN / MLP | floors rand-init / obs-right | PI | ND | GS |
|---|---|---|---|---|---|
| appearance (joint cell, 30 × 3) | 0.90 / 0.91 | 0.87 · 0.87 / 0.54 · 0.89 | +0.26 / 0.53 | **+0.44 / 0.41** | **+0.58 / 0.31** |
| appearance-fac (centre × length, 4 × 20) | 0.94 / 0.94 | 0.92 · 0.93 / 0.48 · 0.94 | +0.24 / 0.54 | +0.30 / 0.50 | +0.40 / 0.43 |
| *frame model `L-dw-8ray-20m`, appearance-fac, for comparison* | 0.94 / 0.94 | 0.92 · 0.93 / 0.48 · 0.93 | +0.41 / 0.71 | +0.50 / 0.86 | +0.46 / 0.53 |

**Reading.** On the token model the factorisation COSTS editability: ND +0.44 → +0.30 and GS
+0.58 → +0.40 (PI unchanged at +0.24), where on the frame model ND rose (+0.43 → +0.50) and
only GS fell. Decodability is identical across the two models (0.94 / 0.94, floors alike) —
the difference is entirely in what the editors can do with the write. A plausible reason:
the token model's output is a softmax over whole FRAMES, one token per (centre, length) run,
so the joint-cell write — one class swap per cell, the cell being the frame's own token — is
the write most nearly aligned with its output head; splitting it into a centre move and a
length move asks the head to combine two partial writes. The frame model renders rays and
has no such preference. The ranking across the four factorised rows is therefore
frame-8-ray (+0.41 / +0.50 / +0.46) > noiseless (— / +0.63 / +0.35) ≈ blink (— / +0.53 / +0.33)
> tokens-8-ray (+0.24 / +0.30 / +0.40) — with 8-ray's frame model the only run where every
editor lands, which is Sevan's ranking (2026-09-10 22:25).

Provenance: `runs/interface_ablation/L-dw-8ray-tok-20m/scores.json["bases"]["appearance-fac"]`,
`runs/_baselines/dw-8ray/baselines.json` (`transformer_l_tokens` / `appearance-fac`), unit
`appearance_fac_tok` (`logs/appearance_fac_tok/`, 63 min).

## The quantisation push: dw-5ray (2026-09-11, `L-dw-5ray-20m`, trained overnight)

Sevan's environment toggle, pushed one step further: dw-8ray's geometry with **5 usable rays**
(7 cast, wall rays dropped; radius 1.0, the floor for this ray count — at 4 kept rays a
reachable disc can light no ray). 20M sequences, the matched Transformer-L recipe, 780k steps
(best val MSE 0.00694 at 450k vs 0.00574 on 8-ray — the coarser frame is intrinsically less
predictable). Appearance partition 14 cells (run lengths 1–4, 9 centres); factorised 9 + 4 = 13
classes per object, 52 logits. Bets on record (2026-09-11 09:10): Sevan — a slight bump over
8-ray; me — level with 8-ray and a noisier index.

**Regression rows** (canonical; first 192 cases — dw-5ray has no filtered selection file, so
identical-frame teleports are included and the unedited floor sits at −0.86 rather than −0.91):

| run, frustum regression | skill LIN / MLP | rand-init | PI | ND | GS |
|---|---|---|---|---|---|
| L-dw-5ray-20m | 0.94 / 0.97 | 0.95 / 0.96 | +0.30 / 1.12 (α 175) | n/a | −0.11 / 1.02 |
| L-dw-8ray-20m | 0.95 / 0.98 | 0.96 / 0.97 | +0.28 / 0.90 (α 100) | n/a | −0.06 / 0.82 |

Inert, like every discworld regression row: PI only at the top of the α grid and over the
guard, GS negative, the random-init floor equal to the trained probe.

**The factorised categorical target** (same recipe, floors, cell-changing bench — 51 of the
first 243 teleports stay in their cell at 14 cells vs 34 of 226 at 30):

| run, `appearance-fac` | logits | skill LIN / MLP | floors rand-init / obs-right | PI | ND | GS |
|---|---|---|---|---|---|---|
| **L-dw-5ray-20m** | 52 | 0.93 / 0.93 | 0.92 · 0.92 / 0.52 · 0.92 | **+0.49 / 0.68** (pt 1, α 20; landed 95%) | **+0.56 / 0.90** (pt 0, α 12) | **+0.58 / 0.47** (pt 0, α 0.35) |
| L-dw-8ray-20m | 80 | 0.94 / 0.94 | 0.92 · 0.93 / 0.48 · 0.93 | +0.41 / 0.71 (pt 1, α 35) | +0.50 / 0.86 (pt 0, α 12) | +0.46 / 0.53 (pt 0, α 0.5) |
| L-dw-noiseless-20m | 1,048 | 0.43 / 0.79 | 0.30 · 0.59 / 0.12 · 0.52 | +0.01 / 1.95 | +0.63 / 0.78 | +0.35 / 0.68 |
| L-dw-blink-20m | 1,048 | 0.46 / 0.68 | 0.22 · 0.45 / 0.07 · 0.39 | +0.02 / 2.44 | +0.53 / 0.94 | +0.33 / 0.95 |

**Reading.** Sevan's bet holds: every editor is higher on 5-ray than on 8-ray — PI +0.07,
ND +0.05, GS +0.12 — with the guards alike or better, and PI lands at a smaller step (α 20 vs
35, write ratio 3.4 vs 7.5). Under the factorised target 5-ray is now the best-edited discworld
run on PI and GS and second on ND, and the first discworld row where all three editors clear
+0.48. The bump is small against one seed's noise (a categorical Edit Index over 192 cases
moves by ~±0.03 between α neighbours), but it is on all three editors in the same direction.
Decodability is unchanged (0.93 vs 0.94; the random-init floor equal to the trained probe, as
on 8-ray) — the environment toggle moved editability without moving decodability, which is
what a genuine environment effect on the write side should look like. GS at point 0 (the input
embedding) dominates as on 8-ray; deeper points fall off faster here (guarded GS +0.58 → +0.38 →
+0.28 at points 0–2). So along the ray axis 128 → 8 → 5, under the read-out that works, PI and
GS rise monotonically (PI —, +0.41, +0.49; GS +0.35, +0.46, +0.58) while ND is flat-to-down
(+0.63, +0.50, +0.56): coarser frames make the model's state more WRITABLE by the exact and
gradient editors, at no cost in how readable it is. The next rung down does not exist at this
radius (4 rays leaves blind positions); a larger radius at 4 rays, or a smaller one at 5 (discs
between rays — a natural blink), are the remaining moves on this axis.

Provenance: `runs/ray_ablation/L-dw-5ray-20m/scores.json` (canonical + `appearance-fac`),
`runs/_baselines/dw-5ray/baselines.json`, unit `dw_5ray` (`scripts/drivers/dw_5ray.sh`,
`logs/ray_ablation/dw_5ray/`: generate 2 h 07, train 7 h 53, score 29 min, fac 65 min), joint-cell
`appearance` row pending (unit `dw_5ray_appearance`).
