# qualitative_main — the main-text editability figure: (a) Rayworld, (b) Othello

Two deliverables (round 5, 2026-09-21): **`composite_final`** and **`composite_final_sidebyside`**, Sevan's spec
(`paper/figs/briefs/qualitative_main_round4.md` plus round 5's three fixes, on top of rounds 2-3). Panel (a) has **four columns
from three models**: Standard (continuous) on Examples 1 and 2 (dw-noiseless, `noise_ablation/L-dw-noiseless-20m`, Cartesian
block), 128-ray (categorical) on Example 3 (dw-128ray, `ray_ablation/L-dw-128ray-20m`, appearance-fac block) and 5-ray
(categorical) on the SAME Example 3 (dw-5ray, `ray_ablation/L-dw-5ray-20m`, appearance-fac block). Every drawn scenario passes
the 5-ray visibility filter (below). Panel (b) is round 3's, unchanged. Everything is drawn from caches by the appendix scripts' own helpers
(`paper/figs/qualitative_edits/make_figure.py`: `_panel`, `_blank`, `error`, and its `build` / `predictions` / filter
machinery; `paper/figs/qualitative_edits_othello/make_figure.py`: `draw_board`, `marked_squares`, `mark`, `mark_key`), style by
`paper/figs/paper_style.py` (Arial as TrueType, white page, zero outer padding). No metric is computed here except the canonical
per-case Othello Edit Index (`pim.metrics.set_editability.edit_index_legal`) behind the typical-case rule and the sidecars; no
write is recomputed. The Rayworld caches were built on the GPU (one model at a time, freed after) through the appendix
script's `build()`: probes and inverse maps from each run's `probes/` cache, never fitted; the categorical blocks' IM through the
categorical inverse map deployed 2026-09-20 (`_catim` caches). The Othello cache is the appendix's
`.scratch/othello_edits_guarded_cache.pkl` (2026-09-19 guarded arms).

## How scenarios are matched across columns

One scenario = one world (positions, velocities, teleport) generated under the tightest geometry (radius 1.0) and rendered under
EACH instance's own renderer (radius 0.5 / 128 rays for dw-noiseless and dw-blink; radius 1.0 / N rays for the ray family).
Positions are identical across columns; only the rendering differs.

## Regenerate

    .pim/bin/python paper/figs/qualitative_edits/make_figure.py --set             # the appendix caches and figures (GPU; --redraw reuses the caches)
    .pim/bin/python paper/figs/qualitative_main/common.py --build-128ray 1         # the 128-ray cache of Example 3's seed (GPU, one model)
    .pim/bin/python paper/figs/qualitative_main/composite_final.py                 # both figures + pieces + sidecars (CPU)
    # options kept in the scripts, not delivered: rayworld_panel.py --options R1 R2 R3 R4 (round 1, seeds 0 / 0-2 unfiltered);
    # othello_panel.py [--rule typical --rank k] [--gs]; composite.py (round 1's composites). Rounds 2-3's A1 / A2 cuts were retired.

## Files

| file | what |
|---|---|
| `composite_final.{pdf,png}` | **the main-text figure.** (a) four columns: Standard (continuous) x Examples 1, 2; 128-ray (categorical) x Example 3; 5-ray (categorical) x Example 3; rows Context, Unedited Pred, Ground truth, PI, GS, IM, a signed-error strip under each edit row; a group title over each model's columns, an empty 0.18-column spacer between models; the pre-edit / post-edit key (dots) top right above the colour bar. (b) Othello Standard / Adjacent NoFlip (rows) x Unedited Pred, Ground truth, PI, GS, IM (columns), 5 x 5 zoom, typical cases of rank 2, cyan / pink marks. **5.72 x 4.62 in** ((a) 2.23 in, (b) 2.23 in, gap 0.16 in; strips 0.98 in wide, strip unit 0.142 in; boards 0.98 in). Include at `\linewidth`: LaTeX scales 5.72 to 5.5 in (x 0.96), 8 pt text prints at 7.7 pt, the 7.5 pt key at 7.2 pt |
| `composite_final_sidebyside.{pdf,png}` | (a) left (3.27 in), (b) right (2.45 in): variants Standard / Adjacent Flip / Adjacent NoFlip as columns, Unedited Pred / Ground truth / PI / IM as rows (no GS), boards 0.60 in, rims 1.0 pt. **5.72 x 2.93 in**; (a)'s strips are 0.46 x 0.19 in with 7 pt example titles and no spacers; its group titles are two lines: the model over its own columns ("Standard", "128-ray", "5-ray"), the block qualifier once over the neighbouring columns that share it ("(continuous)" over Examples 1-2, "(categorical)" over the two Example 3 columns), because a 7 pt "(categorical)" is 0.54 in wide, wider than a 0.46 in column |
| `pieces/composite_final/` | every element as its own PDF (+ PNG preview): the 36 strips of (a), named `col<k>_<instance>_seed<s>_<row>` with rows `context`, `unedited`, `ground_truth`, `<block>_<editor>_{prediction,error}` (block `cont` / `cat`; e.g. `col3_dw-128ray_seed2_cat_IM_prediction`), `key_error_scale` (the ±1 bar), `key_locators` (cyan / pink lines); the ten boards of (b) at 1.4 in (`<Variant>_case<i>_<condition>`), a faded full-board `_thumbnail` per variant with the 5 x 5 window, `key_marks` (the dots), `key_tint` |
| `pieces/composite_final_sidebyside/` | the twelve boards of the side-by-side (b) (rims 1.0 pt), thumbnails, keys; its strips are `pieces/composite_final/`'s |
| `composite_final.json`, `composite_final_sidebyside.json` | sidecars: geometry (inches); the Rayworld columns (title, group, variant, instance, run, seed, cache, block, edit object, locator ray centres, `n_changed_rays_5ray`, `changed_rays` per renderer (dw-5ray, the column's own, dw-noiseless, dw-128ray), `delta_intensity_5ray`, arms drawn for both blocks, the guarded Table 2 cells), the selection and matching rules; the Othello cases (run, case id, rank, flipped tile, marked squares, legal sets, window, per-case Edit Index of every condition, population means, arms drawn, Table 2 cells) |
| `common.py`, `rayworld_panel.py`, `othello_panel.py`, `composite_final.py`, `composite.py` | the scripts (caches, the filter and selection; panel (a), option `A3` = the final cut, `CAT_SLOT` = which scenario the categorical pair takes; panel (b); the two composites; round 1's composites, kept only as a script) |

## Selection rules

- **Scenario filter (Sevan: "only show examples which change for all of them"; tightened in round 5).** A seed is eligible only
  if its teleport VISIBLY changes the 5-ray observation: the scenario is rendered under the dw-5ray config and the clean
  post-edit frame at the edit frame must differ from the clean unedited frame **on at least 2 rays, at least 2 of them by at
  least 0.2 in intensity**. Round 4 asked only for a non-empty difference, which passed scenarios changing a single ray; those
  read as no change at all in the drawing (seed 8 changes only 5-ray ray 1), so round 5 raised the bar. Both quantities come
  from the scorer's own zone construction (`pim.metrics.zone_editability.build_edit_zones` through `bench.bench_from_arrays`):
  the `differing` mask is the support the Edit Index is scored over, the magnitudes are the gap between its two clean reference
  renders (`make_figure.passes` / `change_at`, CPU, no model). **Seeds 0-59: 29 pass**, 31 fail; 15 of the failures (8, 16, 18,
  19, 20, 22, 23, 35, 36, 38, 45, 46, 55, 56, 59) passed the round-4 rule on a single ray, the rest change no ray at all. On
  this generator the intensity clause never binds on its own: the discs' fixed reflectivities make every changed ray differ by
  0.4 or 0.8, so the ray-count clause decides every case in 0-59.
- **The three scenarios are the first three passing seeds: 0, 1, 2.** Which one the categorical pair takes is Sevan's editorial
  choice (round 5, `rayworld_panel.CAT_SLOT`): **the second, seed 1**, whose teleport crosses the frame; the other two are the
  continuous examples in seed order. So **Example 1 = seed 0, Example 2 = seed 2, Example 3 = seed 1** (one world drawn at two
  resolutions in columns 3 and 4). Per example, the rays on which the clean edited and unedited frames differ:

  | example | column(s) | seed | 5-ray (the filter) | dw-noiseless (128 rays, r 0.5) | dw-128ray (128 rays, r 1.0) | edited object | tile changes |
  |---|---|---|---|---|---|---|---|
  | 1 | 1, Standard continuous | 0 | 2: rays 0, 4 (delta 0.4, 0.4) | 28 | 52 | 0 | yes |
  | 2 | 2, Standard continuous | 2 | 4: rays 1, 2, 3, 4 (delta 0.4 each) | 44 | 92 | 0 | yes |
  | 3 | 3 and 4, categorical | 1 | 2: rays 0, 3 (delta 0.4, 0.4) | 26 | 26 | 0 | yes |

  Locator ray centres (cyan origin / pink destination) in each column's own renderer: Example 1 on dw-noiseless 107.5 / 28.5;
  Example 2 on dw-noiseless 109.5 / 61.5; Example 3 on dw-128ray 21.5 / 88.0 and on dw-5ray ray 0 / ray 3. As drawn, the
  5-ray column's Ground truth and Unedited Pred differ on exactly those two rays (ray 0: 0.39 to 0.00, ray 3: 0.07 to 0.40).
- **Othello eligibility.** Among the 1000 bench cases of a variant (all at move 20): at least 3 squares change legality
  (|legal_pre XOR legal_post| ≥ 3) and the window {flipped tile} ∪ changed squares + one-square margin fits 5 x 5. Eligible:
  65 / 396 / 379 on Standard / Adjacent Flip / Adjacent NoFlip.
- **Typical rule, rank 2.** Eligible cases ranked by the sum over PI, GS, IM of |per-case Edit Index (symdiff construction)
  − the variant's population mean at the guarded arm|; rank 1 is the closest (842 / 986 / 627, round 1's picks), the figure
  uses **rank 2** (Sevan's ask): **342 / 261 / 39**. An editorial rule; the caption must say so.
- **Arms.** Every editor is drawn at the arm the tables report: `pim.metrics.selection.best_arm`, the best Edit Index inside
  the fidelity guard (ratio ≤ 1), the unguarded best only where an editor has no arm inside the guard (Adjacent NoFlip GS,
  ratio 6.68). Never changed here; copied from the caches into the sidecars.

## What is drawn (caption facts)

- **(a)** Context = the last 8 observed frames fed to the model, time downward, the `gray` map on the dark panel as in
  every waterfall, fixed 0..1. Unedited Pred = the model's next frame with no edit; Ground truth = the clean render of the
  edited world at the edit frame (the reference the Edit Index scores against). PI / GS / IM = the next frame after each write
  at the guarded arm, each with a strip beneath showing prediction minus truth on the canonical signed-error map (red =
  under-prediction, green = over, black = correct), fixed ±1, the prediction clipped to 0..1 before differencing (the scorer
  does not clip). Cyan line = ray centre of the edited disc before the edit, pink = after, in each column's own renderer.
  Single next-step frames, not rollouts; the write targets the pre-dynamics state; edit at frame 20. **Columns 1-2** edit the
  continuous full state of the Standard model (dw-noiseless, Cartesian block: PI pt 5 a 12, GS pt 0 a 0.35, IM pt 6) on
  scenarios seed 0 and seed 2. **Column 3** edits the factorised appearance labels of the 128-ray model (dw-128ray,
  appearance-fac block: PI pt 3 a 0.5, GS pt 0 a 0.35, IM pt 6) on scenario seed 1; **column 4** the same labels of the 5-ray
  model (dw-5ray, appearance-fac block: PI pt 1 a 20, GS pt 0 a 0.35, IM pt 5) on that SAME scenario, seen through 5 rays. Categorical IM = the categorical
  inverse map (one-hot labels + Cartesian velocity; `pim.probes.inverse.encode_categorical_state`). Discs have radius 0.5 in
  columns 1-2 and 1.0 in columns 3-4 (each instance's own geometry).
- **(b)** Unedited Pred = the pre-edit board with the model's next-move distribution; every other column = the post-edit
  board (the flipped tile has changed colour). Ground truth = uniform over the post-edit legal moves. Yellow tint = predicted
  probability, fully tinted at 0.02 and above, (p / 0.02)^0.6 below (`draw_board` defaults). **Marks:** on the Unedited Pred
  board the flipped tile and every square whose legality the flip switches (legal_pre XOR legal_post, exactly the squares the
  symmetric-difference Edit Index scores) are outlined cyan; on every other board the same squares pink; nothing else is
  outlined (key: cyan dot pre-edit, pink dot post-edit). Windows are 5 x 5 around those squares, one zoom per figure (the
  `_thumbnail` pieces show the window on the full board). The gap between Ground truth and PI is wider than the other gaps,
  as between (a)'s Ground truth and edit rows. Boards are absolute colours replayed under the instance's own rules. Arms:
  Standard PI pt 4 a 3, GS pt 4 a 0.2, IM pt 5; Adjacent Flip PI pt 2 a 5, GS pt 0 a 0.2, IM pt 5; Adjacent NoFlip PI pt 1
  a 10, GS pt 2 a 1.5 (unguarded), IM pt 1.

## The population numbers the panels are read against (guarded `best_arm`, from `runs/<run>/scores.json`)

Edit Index / Fidelity Ratio at the drawn arm (residual point pt, step size a). `*` = no arm inside the guard, the unguarded
best is reported and drawn. Unedited = the unedited model's index on the same block. Only the starred-in-bold blocks are drawn.

| model (instance, run) | block | unedited | PI | GS | IM |
|---|---|---|---|---|---|
| Standard (dw-noiseless, `noise_ablation/L-dw-noiseless-20m`) | **continuous (`cartesian`), drawn** | −0.93 | −0.10 / 0.96 (pt5, a12) | −0.16 / 0.96 (pt0, a0.35) | +0.59 / 0.34 (pt6) |
| | categorical (`appearance-fac`) | −0.93 | −0.34 / 0.99 (pt4, a0.25) | +0.33 / 0.71 (pt0, a0.35) | no arm |
| 128-ray (dw-128ray, `ray_ablation/L-dw-128ray-20m`) | continuous (`cartesian`) | −0.94 | −0.02 / 0.98 (pt1, a35) | −0.10 / 0.97 (pt0, a0.35) | +0.57 / 0.32 (pt6) |
| | **categorical (`appearance-fac`), drawn** | −0.94 | −0.31 / 0.98 (pt3, a0.5) | +0.31 / 0.79 (pt0, a0.35) | +0.64 / 0.29 (pt6) |
| 5-ray (dw-5ray, `ray_ablation/L-dw-5ray-20m`) | continuous (`cartesian`) | −0.89 | +0.14 / 0.85 (pt4, a175) | −0.10 / 0.86 (pt0, a0.7) | +0.81 / 0.23 (pt0) |
| | **categorical (`appearance-fac`), drawn** | −0.91 | +0.51 / 0.70 (pt1, a20) | +0.60 / 0.47 (pt0, a0.35) | +0.91 / 0.25 (pt5) |

Othello (symmetric-difference Edit Index):

| variant | run | unedited | PI | GS | IM |
|---|---|---|---|---|---|
| Standard | initial_othello_comparison/L-oth-20m | −0.93 | +0.82 / 0.30 (pt4, a3) | +0.83 / 0.28 (pt4, a0.2) | +0.81 / 0.38 (pt5) |
| Adjacent Flip (side-by-side only) | adjacent_flip_ablation/L-oth-adjacent-flip-20m | −0.96 | +0.35 / 0.83 (pt2, a5) | −0.06 / 0.79 (pt0, a0.2) | +0.66 / 0.51 (pt5) |
| Adjacent NoFlip | adjacency_ablation/L-oth-adjacent-20m | −0.96 | −0.23 / 0.81 (pt1, a10) | −0.16 / 6.68* (pt2, a1.5) | −0.03 / 0.74 (pt1) |

Per-case Edit Index (symdiff) of the drawn cases, for the caption's "read against" sentence:

| variant | case | flipped tile (row, col) | marked squares | Unedited / PI / GS / IM |
|---|---|---|---|---|
| Standard | 342 | 50 (6, 2) | 49, 50, 57, 58 | −1.00 / +0.83 / +0.83 / +0.81 |
| Adjacent Flip | 261 | 41 (5, 1) | 32, 33, 40, 41, 48, 49, 50 | −1.00 / +0.36 / −0.07 / +0.87 |
| Adjacent NoFlip | 39 | 37 (4, 5) | 37, 38, 45, 46 | −1.00 / −0.00 / −0.54 / −0.08 |

No per-case statistic exists for the drawn Rayworld examples (the cache holds frames, not indices); the appendix's
`qualitative_edits/more_seeds/` shows how much the picture moves with the scenario. That set (seeds 5, 7, 9, 10, 12) has been
**disjoint from this figure's scenarios** since round 5, so no appendix column repeats a column drawn here. The primary appendix
figure is still seed 0, so its Standard column is the same world as Example 1 here; Sevan's call whether to move it.

## Caveats

- **The draft's Table 2 is not the guarded table.** `paper/paper_draft.tex` still quotes pre-2026-09-19 cells (and, for the
  categorical IM, the pre-2026-09-20 continuous-map arm: it says 5-ray categorical IM +0.84 / 0.26, the cache and the guarded
  scores say +0.91 / 0.25 at pt 5); the figure draws the guarded arms and the tables above are the guarded numbers. The
  caption must use these.
- **Three models in one panel.** Columns 1-2 are the Standard model of the continuous table (dw-noiseless), columns 3-4 the
  128-ray and 5-ray members of the ray family; the caption names all three. The categorical inverse map is fitted on the ray
  family only, so the categorical IM cell of the Standard model is blank in the table and is not drawn here (round 3 drew the
  128-ray model as "Standard" for that reason; round 4 puts the Standard model back and shows the categorical block on the
  ray family instead).
- The figure is 5.72 in wide (0.22 in for the key) and is scaled by LaTeX to the 5.5 in column: 8 pt text prints at 7.7 pt,
  the 7.5 pt key at 7.2 pt. The side-by-side's 7 pt example titles print at 6.7 pt, the one place under the 7 pt floor, and
  its group titles share the block qualifier across the two categorical columns (see Files).
- The Othello writes are the 2026-09-19 guarded cache (unchanged by the categorical deployment, which touched discworld only).
- Nothing in `runs/`, `datasets/`, `logs/`, `pim/`, or the paper `.tex` was touched; the GPU built the Rayworld caches (one
  model at a time, freed after) and fitted nothing.
