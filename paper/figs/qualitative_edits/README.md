# qualitative_edits — one matched scenario, every ray-world variant, next-step edits

`make_figure.py --seed <k>` → `qualitative_edits_seed<k>.{pdf,png,json}` beside it.

**Columns** are the variants in `VARIANTS` (display name, instance, run): Standard, Blink, 16-ray,
8-ray, 5-ray by default. **The scenario is the same world in every column**: one two-disc
trajectory with a teleport at the edit frame, generated from the seed by the edit-set generator
(`edits_dataset._generate_one_edit`) under the tightest geometry among the listed variants (disc
radius 1.0, the coarse-ray family's), then rendered under each variant's own renderer — so the
Standard and Blink columns show the same positions at their radius 0.5, the ray columns at 1.0.
The Blink column carries the blackout schedule the seed implies for that instance, forced visible
from frame EF−1 on so the edit is observable.

**Rows** per column: the last `--context` (8) observed frames above the edit, time downward, as
fed to the model; then single NEXT-STEP frames (not rollouts): unedited, the clean ground truth,
and PI / GS / IM at each run's scored best arm (`scores.json`) on the continuous target (full
state, Cartesian block) and on the categorical target (factorised appearance, frustum block).
As everywhere in the project the write targets the PRE-dynamics state (`bench.full_state_pair`),
and the bench is built by `bench.bench_from_arrays`, the scorer's own construction, so what is
drawn is what is scored. `--find` advances the seed until the teleport changes a factorised tile
on every variant (otherwise a categorical row asks for no change there; the sidecar JSON records
which arms were used and whether each variant's tile changed).

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

## Which arm is drawn (2026-09-19)

Each editor is drawn at the arm the TABLES report — `pim.metrics.selection.best_arm`: the best Edit Index among the
arms inside the fidelity guard (ratio ≤ 1), the unguarded best only where an editor has none. Until 2026-09-19 the
figure used the scorer's unguarded `best`, which for PI on the fine-ray instances and on the adjacency Othello
variants is a different (more destructive) write than the one whose numbers the paper quotes. The cached writes in
`.scratch/` carry `_guarded` in their names, so the old caches are not reused.

## more_seeds/

`more_seeds/seed<k>/` holds the same figure for other seeds (k = 1…5: `--seed <k> --out-dir more_seeds/seed<k>`,
file names `qualitative_edits_seed<k>`), to see how much the picture depends on the drawn scenario. Same arms, same models — only the
scenario (discworld) or the sampled bench cases (Othello) change.
