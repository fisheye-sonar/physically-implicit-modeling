# qualitative_edits — one matched scenario, every ray-world variant, next-step edits

`make_figure.py --seed <k>` → `qualitative_edits_seed<k>.{pdf,png,json}` beside it; `make_figure.py --passing 6` regenerates
the whole set (the first passing seed beside the script, the next five under `more_seeds/seed<k>/`).

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

## 2026-09-21, round 4: the scenario filter and the re-seeded set

- **Filter (Sevan: "only show examples which change for all of them").** A seed is drawn only if its teleport VISIBLY changes
  the 5-ray observation: the scenario is rendered under the dw-5ray config and the clean post-edit frame at the edit frame must
  differ from the clean unedited frame on at least one ray. The quantity is the scorer's own differing-ray zone
  (`pim.metrics.zone_editability.build_edit_zones` inside `bench.bench_from_arrays`, the support of the Edit Index) under the
  dw-5ray renderer: `visible_change(seed)` (CPU, no model; `differing_rays` for any instance, `passing_seeds(n)` for the first
  `n` passing seeds, `FILTER_INST = "dw-5ray"`). `--find` now advances the seed until it passes this filter (until this date it
  advanced until the teleport changed a categorical tile on every variant; that condition is still checked and printed as a
  warning, and every drawn seed satisfies it too). Of seeds 0-39, 27 pass; the 13 that fail (3, 4, 6, 11, 13, 14, 15, 26, 28,
  30, 31, 32, 33) are teleports that stay inside the disc's own 5-ray ray(s).
- **The set is the first six passing seeds: 0, 1, 2, 5, 7, 8.** Slot → generator seed: beside the script (the paper's
  `qualitative_edits_seed0_paired.pdf`) → seed 0 (unchanged); `more_seeds/` slots 1-5 → seeds 1, 2, 5, 7, 8 (folders are named
  by the generator seed). Until this date the slots held seeds 0-5; seeds 3 and 4 fail the filter (their disc stays inside
  one 5-ray ray) and their folders were removed, seeds 7 and 8 take their places. Rays on which the clean edited and unedited
  frames differ, per seed: seed 0: 5-ray 2 (rays 0, 4), 128-ray Standard 28; seed 1: 2 (0, 3), 26; seed 2: 4 (1, 2, 3, 4), 44;
  seed 5: 3 (0, 3, 4), 28; seed 7: 3 (0, 1, 3), 53; seed 8: 1 (ray 1), 7. All six change a categorical tile on every variant.
  The main-text figure (`paper/figs/qualitative_main/`) draws seeds 0, 1, 2 of this set.
- **Caches and sidecars.** All six `_catim` caches were rebuilt (identical predictions; each column now also carries
  `cont["differing_rays"]`, its own renderer's changed rays). The sidecar JSON gained `filter` (instance, changed 5-ray rays,
  the rule) and `differing_rays` per variant. `bench_arrays_for` is the model-free half of `bench_for`, so the filter builds the
  scorer's bench without loading a model. `build(seed, context, variants=None)` still accepts another variant list (the
  main-text figure's 128-ray member) and always generates the scenario under the appendix's base geometry (`base_config`).

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
- Caches are `.scratch/qualitative_edits_catim_seed<k>_ctx8.pkl` (seeds 0, 1, 2, 5, 7, 8 since round 4; 3 and 4 remain on
  disk but are not read); the `_guarded` caches hold the old categorical IM frames and are not read any more.

## Which arm is drawn (2026-09-19)

Each editor is drawn at the arm the TABLES report — `pim.metrics.selection.best_arm`: the best Edit Index among the
arms inside the fidelity guard (ratio ≤ 1), the unguarded best only where an editor has none. Until 2026-09-19 the
figure used the scorer's unguarded `best`, which for PI on the fine-ray instances and on the adjacency Othello
variants is a different (more destructive) write than the one whose numbers the paper quotes. The cached writes in
`.scratch/` carry `_guarded` in their names, so the old caches are not reused.

## more_seeds/

`more_seeds/seed<k>/` holds the same figure for the other passing seeds (k = 1, 2, 5, 7, 8: written by `--passing 6`, or one
at a time by `--seed <k> --out-dir more_seeds/seed<k>`; file names `qualitative_edits_seed<k>`), to see how much the picture
depends on the drawn scenario. Same arms, same models; only the scenario changes.
