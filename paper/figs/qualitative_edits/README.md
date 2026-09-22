# qualitative_edits — one matched scenario, every ray-world variant, next-step edits

`make_figure.py --seed <k>` → `qualitative_edits_seed<k>.{pdf,png,json}` beside it; `make_figure.py --set` regenerates the
whole set (the primary figure beside the script, the `more_seeds/seed<k>/` entries under it).

**Columns** are the variants in `VARIANTS` (display name, instance, run): Standard, Blink, 16-ray,
8-ray, 5-ray by default. **The scenario is the same world in every column**: one two-disc
trajectory with a teleport at the edit frame, generated from the seed by the edit-set generator
(`edits_dataset._generate_one_edit`) under the tightest geometry among the listed variants (disc
radius 1.0, the coarse-ray family's), then rendered under each variant's own renderer — so the
Standard and Blink columns show the same positions at their radius 0.5, the ray columns at 1.0.
The Blink column carries the blackout schedule the seed implies for that instance, forced visible
from frame EF−1 on so the edit is observable.

**How scenarios are matched across columns.** One scenario = one world (positions, velocities, teleport) generated under the
tightest geometry (radius 1.0) and rendered under EACH instance's own renderer (radius 0.5 / 128 rays for dw-noiseless and
dw-blink; radius 1.0 / N rays for the ray family). Positions are identical across columns; only the rendering differs.

**Rows** per column: the last `--context` (8) observed frames above the edit, time downward, as
fed to the model; then single NEXT-STEP frames (not rollouts): unedited, the clean ground truth,
and PI / GS / IM at each run's scored best arm (`scores.json`) on the continuous target (full
state, Cartesian block) and on the categorical target (factorised appearance, frustum block).
As everywhere in the project the write targets the PRE-dynamics state (`bench.full_state_pair`),
and the bench is built by `bench.bench_from_arrays`, the scorer's own construction, so what is
drawn is what is scored.

## 2026-09-21, rounds 4-5: the scenario filter and the re-seeded set

- **Filter (Sevan: "only show examples which change for all of them").** A seed is drawn only if its teleport VISIBLY changes
  the 5-ray observation: the scenario is rendered under the dw-5ray config and the clean post-edit frame at the edit frame must
  differ from the clean unedited frame **on at least 2 rays (`MIN_RAYS`), at least 2 of them (`MIN_STRONG`) by at least 0.2
  (`MIN_DELTA`) in intensity**. Round 4 asked only for a non-empty difference; that passed scenarios changing a single ray,
  which read as no change at all in the drawing (seed 8 changes only 5-ray ray 1), so round 5 raised the bar. Both quantities
  come from the scorer's own zone construction (`pim.metrics.zone_editability.build_edit_zones` inside
  `bench.bench_from_arrays`): the `differing` mask is the support of the Edit Index, the magnitudes are the gap between its two
  clean reference renders. CPU, no model: `change_at(seed, inst)` returns (changed rays, |Δ| on them), `passes(seed)` applies
  the rule, `passing_seeds(n, start, exclude=...)` walks the seed line. `--find` advances the seed until it passes this filter
  (the older condition, "the teleport changes a categorical tile on every variant", is still checked and printed as a warning;
  every drawn seed satisfies it too).
- **Pass rates.** Seeds 0-59: **29 pass**, 31 fail. Fifteen of the failures (8, 16, 18, 19, 20, 22, 23, 35, 36, 38, 45, 46, 55,
  56, 59) passed the round-4 rule on a single ray; the other sixteen change no 5-ray ray at all. On this generator the intensity
  clause never binds on its own: the discs' fixed reflectivities make every changed ray differ by 0.4 or 0.8, so the ray-count
  clause decides every case in 0-59. The clause stays as the stated guard against a faint change on a future instance.
- **The set (round 5).** The **primary figure is the first passing seed, 0**, unchanged, so the paper's
  `figs/qualitative_edits/qualitative_edits_seed0_paired.pdf` keeps resolving. `more_seeds/` holds the **first five passing
  seeds the main-text figure does not draw**: since that figure's scenarios are seeds 0, 1, 2, the set is **5, 7, 9, 10, 12**
  (`appendix_seeds()`; folders are named by the generator seed). Round 4's set was the first six passing seeds (0, 1, 2, 5, 7,
  8); Sevan caught that `more_seeds/seed2` redrew the main figure's 5-ray column, so seeds 1 and 2 were dropped as duplicates
  and seed 8 as a filter failure, and 9, 10, 12 took their places. **Note:** the primary figure's Standard column is still the
  same world as the main figure's Example 1 (both seed 0); deliberate, and Sevan's call whether to move it.

  | slot | seed | 5-ray changed rays | delta intensity | 128-ray Standard | 16-ray | 8-ray |
  |---|---|---|---|---|---|---|
  | primary (beside the script) | 0 | 2: rays 0, 4 | 0.4, 0.4 | 28 | 7 | 3 |
  | more_seeds/seed5 | 5 | 3: rays 0, 3, 4 | 0.4 each | 28 | 6 | 3 |
  | more_seeds/seed7 | 7 | 3: rays 0, 1, 3 | 0.4 each | 53 | 10 | 6 |
  | more_seeds/seed9 | 9 | 2: rays 2, 3 | 0.4, 0.4 | 27 | 5 | 2 |
  | more_seeds/seed10 | 10 | 2: rays 1, 3 | 0.8, 0.4 | 35 | 6 | 2 |
  | more_seeds/seed12 | 12 | 2: rays 0, 4 | 0.8, 0.8 | 29 | 7 | 4 |

  All six change a categorical tile on every variant. As drawn, every column of every figure shows the change: the Ground truth
  row differs from the Unedited Pred row on at least 2 rays by at least 0.2 (checked per column, coarsest included).
- **Caches and sidecars.** Each column of a cache carries `cont["differing_rays"]`, its own renderer's changed rays. The sidecar
  JSON records `filter` (instance, changed rays, `delta_intensity`, `n_strong`, `passes`, the rule) and `differing_rays` per
  variant. `bench_arrays_for` is the model-free half of `bench_for`, so the filter builds the scorer's bench without loading a
  model. `build(seed, context, variants=None)` still accepts another variant list (the main-text figure's 128-ray member) and
  always generates the scenario under the appendix's base geometry (`base_config`).

Drawing follows the canonical waterfall (`pim/figures/waterfall.py`): `gray` on the dark panel
background, fixed 0–1 range, nearest interpolation; the page is white, the text black; no gridlines;
every strip stretched so a one-pixel-tall observation is readable; the ground-truth row's label is bold. Two more files per seed: `…_diff` draws the six edit rows as prediction −
ground truth on the canonical signed-error map (red = under-prediction, green = over, zero = background),
and `…_overlay` keeps each prediction's own grey and tints every ray red / green in proportion to its
error, so a perfect ray is simply drawn (`--diff-scale`, default ±1.0, the true worst case, sets both);
the bar for either sits at the right, spanning the six edit rows. `…_paired` stacks, for each edit row, the plain prediction with its `_diff` strip directly beneath
(no gap). `…_abs` is the overlay with a
single hue — red by absolute error, no bar (for a caption that explains it). In all three the error is
taken on the prediction CLIPPED to [0, 1] — what the grey panel shows — so a raw output of −0.85 on an
empty ray, drawn black like the truth, is not tinted; `--raw-error` uses the raw value instead, which is
the scorer's quantity (`zone_rmse` never clips, so those negatives do count in the Edit Index).
`--tint-gamma` shapes the overlay tint (> 1 mutes small errors). Cyan / pink lines on the eight
single-frame rows mark the edited disc's origin (its unedited rays at the edit frame) and destination
(its target rays); `--no-locators` drops them. Predictions are cached in `.scratch/` per seed so `--redraw`
re-renders without reloading the five models. Nothing on the figure names the environment.

## 2026-09-21: Arial, the categorical inverse map, blank cells

- Fonts and page come from `paper/figs/paper_style.py` (`ps.apply()`: Arial, TrueType embedding, white page,
  no outer padding on save) instead of the script's own Times New Roman block.
- **The categorical rows' IM is the categorical inverse map** (deployed 2026-09-20,
  `experiments/categorical_inverse/README.md`): on the `appearance-fac` block the write is g(one-hot of the
  block's own post-edit labels, the discs' Cartesian velocity) at the block's guarded arm, obtained exactly as
  `pim/scoring/discworld.py::inverse_discworld` obtains it (`arms.iter_inverse_maps(..., target=target,
  **probe_recipe(target))`, the forward probe's 200k / 50-epoch recipe, `pim.probes.inverse.encode_categorical_state`
  for the state). The map is read from the run's `probes/` cache only (`_cache_hits_only`: a miss raises instead of
  starting a 30-minute fit). Until this date the row showed the continuous full-state map on the categorical bench,
  which is no longer a scored arm anywhere.
- **Blank cells.** A block with no arm for an editor (`best_arm` None) leaves its spot fully empty — no panel, no
  frame, no strip; the row label stays (Sevan, round 3): today the categorical IM on Standard (dw-noiseless) and Blink
  (dw-blink), whose table cells are blank; the ray family (16 / 8 / 5-ray) carries the categorical arm (points 6 / 6 / 5).
- Caches are `.scratch/qualitative_edits_catim_seed<k>_ctx8.pkl` (the drawn seeds 0, 5, 7, 9, 10, 12; seeds 1 and 2 are read
  by the main-text figure, 3, 4 and 8 remain on disk unread); the `_guarded` caches hold the old categorical IM frames and are
  not read any more.

## Which arm is drawn (2026-09-19)

Each editor is drawn at the arm the TABLES report — `pim.metrics.selection.best_arm`: the best Edit Index among the
arms inside the fidelity guard (ratio ≤ 1), the unguarded best only where an editor has none. Until 2026-09-19 the
figure used the scorer's unguarded `best`, which for PI on the fine-ray instances and on the adjacency Othello
variants is a different (more destructive) write than the one whose numbers the paper quotes. The cached writes in
`.scratch/` carry `_guarded` in their names, so the old caches are not reused.

## more_seeds/

`more_seeds/seed<k>/` holds the same figure for the other passing seeds (k = 5, 7, 9, 10, 12: written by `--set`, or one at a
time by `--seed <k> --out-dir more_seeds/seed<k>`; file names `qualitative_edits_seed<k>`), to see how much the picture depends
on the drawn scenario. Same arms, same models; only the scenario changes. The set is disjoint from the main-text figure's
scenarios (seeds 0, 1, 2) apart from the primary figure above, so no entry here repeats a column of that figure.
