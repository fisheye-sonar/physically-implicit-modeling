# qualitative_main — the main-text editability figure: (a) Rayworld, (b) Othello

Two deliverables (round 3, 2026-09-21): **`composite_final`** and **`composite_final_sidebyside`**, Sevan's spec
(`BRIEF_ROUND2.md` and the round-3 changes: "Unedited Pred", "Example 1 / 2", the pre-edit / post-edit key as dots at the
top right above the colour bar, the folder pruned to these two figures). Both draw the 128-ray model of the ray family
(`ray_ablation/L-dw-128ray-20m`, instance dw-128ray, disc radius 1.0); the column titles say "Standard" and the caption
names the model. Everything is drawn from caches by the appendix scripts' own helpers (`paper/figs/qualitative_edits/make_figure.py`:
`_panel`, `_blank`, `error`; `paper/figs/qualitative_edits_othello/make_figure.py`: `draw_board`, `marked_squares`, `mark`,
`mark_key`), style by `paper/figs/paper_style.py` (Arial as TrueType, white page, zero outer padding). No metric is computed
here except the canonical per-case Othello Edit Index (`pim.metrics.set_editability.edit_index_legal`) behind the typical-case
rule and the sidecars; no write is recomputed. The Rayworld cache (`.scratch/qualitative_edits_catim_128ray_seed{0..5}_ctx8.pkl`)
was built on the GPU by `common.py --build-128ray` through the appendix script's `build()` (probes and inverse maps from the run's
`probes/` cache, never fitted; the categorical block's IM through the categorical inverse map deployed 2026-09-20). The Othello
cache is the appendix's `.scratch/othello_edits_guarded_cache.pkl` (2026-09-19 guarded arms).

## Regenerate

    .pim/bin/python paper/figs/qualitative_main/common.py --build-128ray 0 1 2 3 4 5   # the Rayworld cache (GPU, one model)
    .pim/bin/python paper/figs/qualitative_main/composite_final.py                     # both figures + pieces + sidecars
    # options kept in the scripts, not delivered: composite_final.py --version A1 (Standard = dw-noiseless, blank categorical
    # IM cells); rayworld_panel.py --options R1 R2 R3 R4 A1 A2; othello_panel.py [--rule typical --rank k] [--gs]; composite.py

## Files

| file | what |
|---|---|
| `composite_final.{pdf,png}` | **the main-text figure.** (a) four columns: Standard (continuous) x examples 1, 2 and Standard (categorical) x the same two examples; rows Context, Unedited Pred, Ground truth, PI, GS, IM, a signed-error strip under each edit row; the pre-edit / post-edit key (dots) top right above the colour bar. (b) Othello Standard / Adjacent NoFlip (rows) x Unedited Pred, Ground truth, PI, GS, IM (columns), 5 x 5 zoom, typical cases of rank 2, cyan / pink marks. **5.72 x 4.62 in** ((a) 2.23 in, (b) 2.23 in, gap 0.16 in; boards 0.98 in; strip unit 0.142 in). Include at `\linewidth`: LaTeX scales 5.72 to 5.5 in (x 0.96), 8 pt text prints at 7.7 pt, the 7.5 pt key at 7.2 pt |
| `composite_final_sidebyside.{pdf,png}` | (a) left (3.15 in), (b) right (2.57 in): variants Standard / Adjacent Flip / Adjacent NoFlip as columns, Unedited Pred / Ground truth / PI / IM as rows (no GS), boards 0.64 in, rims 1.0 pt; key above the colour bar. 5.72 x 3.09 in; (a)'s strips are 0.48 x 0.20 in with 7 pt example titles (readable, dense) |
| `pieces/composite_final/` | every element as its own PDF (+ PNG preview): the 38 strips of (a) (`col<k>_Standard-128ray_seed<s>_{context,unedited,ground_truth,<block>_<editor>_{prediction,error}}`, block `cont` / `cat`), `key_error_scale` (the ±1 bar), `key_locators` (cyan / pink lines); the ten boards of (b) at 1.4 in (`<Variant>_case<i>_<condition>`), a faded full-board `_thumbnail` per variant with the 5 x 5 window, `key_marks` (the dots), `key_tint` |
| `pieces/composite_final_sidebyside/` | the twelve boards of the side-by-side (b) (rims 1.0 pt), thumbnails, keys; its strips are `pieces/composite_final/`'s |
| `composite_final.json`, `composite_final_sidebyside.json` | sidecars: geometry (inches); the Rayworld columns (variant, instance, run, seed, cache, block, edit object, locator positions, arms drawn for both blocks, the guarded Table 2 cells); the Othello cases (run, case id, rank, flipped tile, marked squares, legal sets, window, per-case Edit Index of every condition, population means, arms drawn, Table 2 cells) |
| `common.py`, `rayworld_panel.py`, `othello_panel.py`, `composite_final.py`, `composite.py` | the scripts (caches and selection; panel (a); panel (b); the two composites; round 1's composites, kept only as a script) |

## Selection rules

- **Rayworld examples.** One two-disc teleport case from the edit-set generator per seed, under the radius-1.0 geometry
  (the appendix figure's base, so example k is the appendix's seed k world), rendered under the drawn model's own renderer,
  edit at frame 20; the write targets the pre-dynamics state; single next-step frames, not rollouts. The two examples are the
  first two cached seeds (0..5) whose edited disc is visible before the edit (finite origin locator) AND whose teleport changes
  a factorised appearance tile, on the drawn model: **seeds 0 and 1** (seeds 3 and 4 have the disc out of view before the edit;
  2 and 5 also qualify). Edited object 0 in both; origin / destination ray centres 108.5 / 25.0 and 21.5 / 88.0.
- **Othello eligibility.** Among the 1000 bench cases of a variant (all at move 20): at least 3 squares change legality
  (|legal_pre XOR legal_post| ≥ 3) and the window {flipped tile} ∪ changed squares + one-square margin fits 5 x 5. Eligible:
  65 / 396 / 379 on Standard / Adjacent Flip / Adjacent NoFlip.
- **Typical rule, rank 2.** Eligible cases ranked by the sum over PI, GS, IM of |per-case Edit Index (symdiff construction)
  − the variant's population mean at the guarded arm|; rank 1 is the closest (842 / 986 / 627, round 1's picks), the figure
  uses **rank 2** (Sevan's ask): **342 / 261 / 39**. An editorial rule; the caption must say so. Rank 3 (642 / 744 / 466) was
  not needed: the rank-2 pictures are clear.
- **Arms.** Every editor is drawn at the arm the tables report: `pim.metrics.selection.best_arm`, the best Edit Index inside
  the fidelity guard (ratio ≤ 1), the unguarded best only where an editor has no arm inside the guard (Adjacent NoFlip GS,
  ratio 6.68). Never changed here; copied from the caches into the sidecars.

## What is drawn (caption facts)

- **(a)** Context = the last 8 observed frames fed to the model, time downward, the `gray` map on the dark panel as in
  every waterfall, fixed 0..1. Unedited Pred = the model's next frame with no edit; Ground truth = the clean render of the
  edited world at the edit frame (the reference the Edit Index scores against). PI / GS / IM = the next frame after each write
  at the guarded arm, each with a strip beneath showing prediction minus truth on the canonical signed-error map (red =
  under-prediction, green = over, black = correct), fixed ±1, the prediction clipped to 0..1 before differencing (the scorer
  does not clip). Cyan line = ray centre of the edited disc before the edit, pink = after. The two left columns edit the
  continuous full state (Cartesian block: PI pt 1 a 35, GS pt 0 a 0.35, IM pt 6); the two right columns the factorised
  appearance labels (categorical block: PI pt 3 a 0.5, GS pt 0 a 0.35, IM pt 6) of the SAME two examples, so Context /
  Unedited Pred / Ground truth repeat across the pair. Categorical IM = the categorical inverse map (one-hot labels +
  Cartesian velocity; `pim.probes.inverse.encode_categorical_state`). All frames are the 128-ray model's; discs have radius 1.0.
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
best is reported and drawn. Unedited = the unedited model's index on the same block.

Rayworld, `ray_ablation/L-dw-128ray-20m`:

| block | unedited | PI | GS | IM |
|---|---|---|---|---|
| continuous (`cartesian`) | −0.94 | −0.02 / 0.98 (pt1, a35) | −0.10 / 0.97 (pt0, a0.35) | +0.57 / 0.32 (pt6) |
| categorical (`appearance-fac`) | −0.94 | −0.31 / 0.98 (pt3, a0.5) | +0.31 / 0.79 (pt0, a0.35) | +0.64 / 0.29 (pt6) |

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
`qualitative_edits/more_seeds/` shows how much the picture moves with the scenario.

## Caveats

- **The draft's Table 2 is not the guarded table.** `paper/paper_draft.tex` still quotes pre-2026-09-19 cells; the figure
  draws the guarded arms and the tables above are the guarded numbers. The caption must use these.
- **The model is the 128-ray member of the ray family**, not the noise-free Standard model of the continuous table; the
  caption states it. It is the only model whose categorical block carries every editor's arm (the categorical inverse map is
  fitted on the ray family only), which is why it was chosen over dw-noiseless (`--version A1` regenerates that alternative
  with two blank categorical IM cells). Its categorical IM arm is +0.64 / 0.29; its landing on the exact label cell is poor
  (`experiments/categorical_inverse/README.md`), which is why the categorical IM frames show a slightly soft disc edge.
- The figure is 5.72 in wide (0.22 in for the key) and is scaled by LaTeX to the 5.5 in column: 8 pt text prints at 7.7 pt,
  the 7.5 pt key at 7.2 pt. The side-by-side's 7 pt example titles print at 6.7 pt, the one place under the 7 pt floor.
- The Othello writes are the 2026-09-19 guarded cache (unchanged by the categorical deployment, which touched discworld only).
- Nothing in `runs/`, `datasets/`, `logs/`, `pim/`, or the paper `.tex` was touched; the GPU built the one Rayworld cache
  family (one model at a time, freed after) and fitted nothing.
