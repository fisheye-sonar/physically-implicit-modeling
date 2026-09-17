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
every strip stretched so a one-pixel-tall observation is readable; the ground-truth row is framed in
the locator green. A second file, `…_diff`, draws the six edit rows as prediction − ground truth on
the canonical signed-error map (red = under-prediction, green = over, zero = background; `--diff-scale`,
default ±1.0, the true worst case). Predictions are cached in `.scratch/` per seed so `--redraw`
re-renders without reloading the five models. Nothing on the figure names the environment.
