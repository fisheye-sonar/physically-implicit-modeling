# qualitative_main — the main-text editability figure: (a) Rayworld, (b) Othello

Round 2 (2026-09-21 afternoon): `composite_final_A1` / `composite_final_A2` and `composite_final_sidebyside_A1` / `_A2`,
Sevan's spec in `BRIEF_ROUND2.md`, built on round 1's `composite_R3_T3_typical_2row`. Everything is drawn from the
appendix caches (`.scratch/qualitative_edits_catim_seed{0..5}_ctx8.pkl`, `.scratch/qualitative_edits_catim_128ray_seed{0..5}_ctx8.pkl`,
`.scratch/othello_edits_guarded_cache.pkl`) by the appendix scripts' own helpers (`paper/figs/qualitative_edits/make_figure.py`:
`_panel`, `_blank`, `error`; `paper/figs/qualitative_edits_othello/make_figure.py`: `draw_board`, `marked_squares`, `mark`,
`mark_key`), style by `paper/figs/paper_style.py` (Arial as TrueType, white page, zero outer padding). No metric is computed
here except the canonical per-case Othello Edit Index (`pim.metrics.set_editability.edit_index_legal`) that the typical-case
rule and the sidecars use; no write is recomputed; the Rayworld caches were built by the appendix script on the GPU (one model
at a time, inverse maps from each run's `probes/` cache, never fitted). The two Rayworld cache families are new: the categorical
blocks' IM row now goes through the categorical inverse map (deployed 2026-09-20), and a block with no IM arm gives a blank cell.

## Regenerate

    .pim/bin/python paper/figs/qualitative_edits/make_figure.py --seed k                # the appendix caches, k = 0..5 (GPU)
    .pim/bin/python paper/figs/qualitative_main/common.py --build-128ray 0 1 2 3 4 5   # the A2 caches (GPU, 128-ray model)
    .pim/bin/python paper/figs/qualitative_main/composite_final.py [--versions A1 A2] [--rank 2]   # the four composites + pieces + sidecars
    .pim/bin/python paper/figs/qualitative_main/rayworld_panel.py --options A1 A2      # panel (a) alone (rayworld_A1 / A2 + pieces)
    .pim/bin/python paper/figs/qualitative_main/othello_panel.py --rule typical --rank 2 --gs --options T3   # panel (b) alone
    # round 1 (kept, regenerated under the current scripts): rayworld_panel.py --options R1 R2 R3 R4; othello_panel.py [--rule typical] [--gs]; composite.py

## Files (round 2)

| file | what |
|---|---|
| `composite_final_A2.{pdf,png}` | **recommended.** (a) four columns: Standard (continuous) x scenarios 1, 2 and Standard (categorical) x the same two; rows Context, Unedited, Ground truth, PI, GS, IM with the signed-error strip under each edit row; (b) Othello Standard / Adjacent NoFlip (rows) x Unedited, Ground truth, PI, GS, IM (columns), 5 x 5 zoom, typical cases of rank 2. Standard = the 128-ray model of the ray family (`ray_ablation/L-dw-128ray-20m`, dw-128ray, radius 1.0): every cell is a scored arm. 5.5 x 4.78 in; (a) and (b) 2.31 in each |
| `composite_final_A1.{pdf,png}` | the same with Standard = dw-noiseless (`noise_ablation/L-dw-noiseless-20m`); its categorical block carries no IM arm, so the two categorical IM cells (prediction + error) are blank framed panels, as in the table. Same size |
| `composite_final_sidebyside_A1/_A2.{pdf,png}` | (a) left (2.93 in), (b) right (2.57 in): variants Standard / Adjacent Flip / Adjacent NoFlip as columns, Unedited / Ground truth / PI / IM as rows (no GS), boards 0.64 in, rims 1.0 pt. 5.5 x 3.26 in |
| `rayworld_A1/_A2.{pdf,png,json}` | panel (a) alone at 5.5 in (unit 0.155 in) |
| `othello_T3_2row_typical_r2_gs.*`, `othello_T3_{4col,2col}_typical_r2_gs.*`, `othello_cases_typical_r2.json` | panel (b) alone (`--rule typical --rank 2 --gs`): the two-variant row cut, the four-variant and half-width cuts, and the rank-2 cases of all four variants with per-case indices |
| `pieces/final_rayworld_A1/`, `pieces/final_rayworld_A2/` | every strip of (a) as its own PDF (+ PNG preview): `col<k>_Standard[-128ray]_seed<s>_{context,unedited,ground_truth,<block>_<editor>_{prediction,error}}` (block `cont` / `cat`), plus `key_error_scale` (the ±1 bar) and `key_locators`. A blank cell exports nothing (A1: 34 files, A2: 38) |
| `pieces/final_othello_2row_rank2/` | the ten boards of the stacked (b) at 1.4 in (`<Variant>_case<i>_<condition>`), a faded full-board `_thumbnail` per variant with the 5 x 5 window, `key_marks` (cyan / pink swatches), `key_tint` |
| `pieces/final_othello_sidebyside_rank2/` | the twelve boards of the side-by-side (b) (three variants x four conditions, rims 1.0 pt), thumbnails, keys |
| `composite_final_<v>.json`, `composite_final_sidebyside_<v>.json` | sidecars: geometry (inches), the Rayworld columns (variant, instance, run, seed, cache, block, edit object, locator positions, arms drawn for both blocks, the guarded Table 2 cells, the blank cells), the Othello cases (run, case id, rank, flipped tile, marked squares, legal sets, window, per-case Edit Index of every condition, population means, arms drawn, Table 2 cells) |
| round-1 files (`rayworld_R1..R4`, `othello_T1..T3_*`, `composite_R3_T3_*`, `pieces/rayworld_R*`, `pieces/othello_T*`, `othello_cases_{random,typical}.json`) | round 1's options, regenerated 2026-09-21 under the current scripts (Arial, zero padding, the new Othello marking, the `_catim` caches; content otherwise as documented in git history). Superseded by the final composites |

## Recommendation: A2

**A2.** On the 128-ray model every one of the twelve edit cells is a scored arm, so the categorical pair shows what the
categorical inverse map was built to show (IM renders the moved disc from the factorised appearance labels, PI and GS
return streaky frames), and the reader sees the same three editors succeed or fail on both targets of one model. A1 shows
the same continuous story on dw-noiseless but has to leave the two categorical IM cells blank, which costs a caption sentence
("no arm: the categorical map is fitted only on the ray family") and draws the eye to what is missing. A2 also lines up with
Table 2, whose categorical section starts with this model, so the numbers the caption quotes come from one run. Cost: the
caption must say the model is the 128-ray member of the ray family (disc radius 1.0), not the noise-free Standard model of
the continuous table; the discs are therefore wider in A2's strips than in A1's (same scenario, each model's own renderer).

**Othello cases: rank 2, as asked.** Standard 342 (per-case PI +0.83, GS +0.83, IM +0.81 against population +0.82 / +0.83 /
+0.81), Adjacent Flip 261 (+0.36 / −0.07 / +0.87 against +0.35 / −0.06 / +0.66), Adjacent NoFlip 39 (−0.00 / −0.54 / −0.08
against −0.23 / −0.16 / −0.03). The pictures are clear (Standard: all three land on the three newly legal squares; Adjacent
NoFlip: none lands, GS and IM put mass on occupied squares outside the marks), so rank 3 was not used; rank 3 would be
642 / 744 / 466 (466 has eight marked squares and IM −0.48, further from the population than rank 2).

## Selection rules

- **Rayworld scenarios (A1, A2).** One two-disc teleport case from the edit-set generator per seed, under the radius-1.0
  geometry (the appendix's base, so the same world as the appendix figure's seed k), rendered under the drawn model's own
  renderer, edit at frame 20; the write targets the pre-dynamics state; single next-step frames, not rollouts. The two
  scenarios are the first two cached seeds (0..5) whose edited disc is visible before the edit (finite origin locator) AND
  whose teleport changes a factorised appearance tile, both judged on the drawn model: seeds **0 and 1** for both A1 and A2
  (seeds 3 and 4 have the disc out of view before the edit on both models; 2 and 5 also qualify). Edited object 0 in all.
  Origin / destination ray centres: A1 107.5 / 28.5 and 31.5 / 81.5; A2 108.5 / 25.0 and 21.5 / 88.0.
- **Othello eligibility.** Among the 1000 bench cases of a variant (all at move 20): at least 3 squares change legality
  (|legal_pre XOR legal_post| ≥ 3) and the window {flipped tile} ∪ changed squares + one-square margin fits 5 x 5. Eligible:
  65 / 396 / 379 / 67 on Standard / Adjacent Flip / Adjacent NoFlip / Standard NoFlip (recomputed 2026-09-21; the round-1
  README quoted 63 / 391 / 367 / 67 with the same rule, which I could not reproduce for the first three; the rank-1 picks
  are unchanged).
- **Typical rule, rank k.** Eligible cases ranked by the sum over PI, GS, IM of |per-case Edit Index (symdiff construction)
  − the variant's population mean at the guarded arm|; rank 1 is the closest (842 / 986 / 627, round 1's picks), the final
  figure uses **rank 2** (342 / 261 / 39). An editorial rule; the caption must say so.
- **Arms.** Every editor is drawn at the arm the tables report: `pim.metrics.selection.best_arm`, the best Edit Index inside
  the fidelity guard (ratio ≤ 1), the unguarded best only where an editor has no arm inside the guard (Adjacent NoFlip GS,
  ratio 6.68). Where a block has no arm for an editor at all (A1's categorical IM) the cell is blank. Never changed here;
  copied from the caches into the sidecars.

## What is drawn (caption facts)

- **(a)** Context = the last 8 observed frames fed to the model, time downward, the `gray` map on the dark panel as in
  every waterfall, fixed 0..1. Unedited = the model's next frame with no edit; Ground truth = the clean render of the edited
  world at the edit frame (the reference the Edit Index scores against). PI / GS / IM = the next frame after each write at
  the guarded arm, each with a strip beneath showing prediction minus truth on the canonical signed-error map (red =
  under-prediction, green = over, black = correct), fixed ±1, the prediction clipped to 0..1 before differencing (the scorer
  does not clip). Cyan line = ray centre of the edited disc before the edit, pink = after. The two left columns edit the
  continuous full state (Cartesian block); the two right columns the factorised appearance labels (categorical block) of the
  SAME two scenarios, so Context / Unedited / Ground truth repeat across the pair. Categorical IM = the categorical inverse
  map (one-hot labels + Cartesian velocity, 2026-09-20); on A1 it has no arm and the cell is blank.
- **(b)** Unedited = the pre-edit board with the model's next-move distribution; every other column = the post-edit board
  (the flipped tile has changed colour). Ground truth = uniform over the post-edit legal moves. Yellow tint = predicted
  probability, fully tinted at 0.02 and above, (p / 0.02)^0.6 below (`draw_board` defaults). **Marks:** on the Unedited board
  the flipped tile and every square whose legality the flip switches (legal_pre XOR legal_post, exactly the squares the
  symmetric-difference Edit Index scores) are outlined cyan; on every other board the same squares pink; nothing else is
  outlined (key: cyan pre-edit, pink post-edit). Windows are 5 x 5 around those squares (one uniform zoom per figure; the
  `_thumbnail` pieces show the window on the full board). The gap between the Ground truth and PI columns (rows in the
  side-by-side) is wider than the others, as between (a)'s Ground truth and edit rows. Boards are absolute colours replayed
  under the instance's own rules.

## The population numbers the panels are read against (guarded `best_arm`, from `runs/<run>/scores.json`)

Edit Index / Fidelity Ratio at the drawn arm (residual point pt, step size a). `*` = no arm inside the guard, the unguarded
best is reported and drawn. Unedited = the unedited model's index on the same block.

Rayworld, the two blocks of the drawn model:

| version | run | block | unedited | PI | GS | IM |
|---|---|---|---|---|---|---|
| A2 | ray_ablation/L-dw-128ray-20m | continuous (`cartesian`) | −0.94 | −0.02 / 0.98 (pt1, a35) | −0.10 / 0.97 (pt0, a0.35) | +0.57 / 0.32 (pt6) |
| A2 | ray_ablation/L-dw-128ray-20m | categorical (`appearance-fac`) | −0.94 | −0.31 / 0.98 (pt3, a0.5) | +0.31 / 0.79 (pt0, a0.35) | +0.64 / 0.29 (pt6) |
| A1 | noise_ablation/L-dw-noiseless-20m | continuous (`cartesian`) | −0.93 | −0.10 / 0.96 (pt5, a12) | −0.16 / 0.96 (pt0, a0.35) | +0.59 / 0.34 (pt6) |
| A1 | noise_ablation/L-dw-noiseless-20m | categorical (`appearance-fac`) | −0.93 | −0.34 / 0.99 (pt4, a0.25) | +0.33 / 0.71 (pt0, a0.35) | blank (no arm) |

Othello (symmetric-difference Edit Index):

| variant | run | unedited | PI | GS | IM |
|---|---|---|---|---|---|
| Standard | initial_othello_comparison/L-oth-20m | −0.93 | +0.82 / 0.30 (pt4, a3) | +0.83 / 0.28 (pt4, a0.2) | +0.81 / 0.38 (pt5) |
| Adjacent Flip (side-by-side only) | adjacent_flip_ablation/L-oth-adjacent-flip-20m | −0.96 | +0.35 / 0.83 (pt2, a5) | −0.06 / 0.79 (pt0, a0.2) | +0.66 / 0.51 (pt5) |
| Adjacent NoFlip | adjacency_ablation/L-oth-adjacent-20m | −0.96 | −0.23 / 0.81 (pt1, a10) | −0.16 / 6.68* (pt2, a1.5) | −0.03 / 0.74 (pt1) |

Per-case Edit Index (symdiff) of the drawn Othello cases (rank 2), for the caption's "read against" sentence:

| variant | case | flipped tile (row, col) | marked squares | Unedited / PI / GS / IM |
|---|---|---|---|---|
| Standard | 342 | 50 (6, 2) | 49, 50, 57, 58 | −1.00 / +0.83 / +0.83 / +0.81 |
| Adjacent Flip | 261 | 41 (5, 1) | 32, 33, 40, 41, 48, 49, 50 | −1.00 / +0.36 / −0.07 / +0.87 |
| Adjacent NoFlip | 39 | 37 (4, 5) | 37, 38, 45, 46 | −1.00 / −0.00 / −0.54 / −0.08 |

No per-case statistic exists for the drawn Rayworld scenarios (the caches hold frames, not indices); the appendix's
`more_seeds/` shows how much the picture moves with the scenario (seeds 1..5, all variants, same arms).

## Geometry (what the spec asked for, what 5.5 in allows)

- Stacked: 5.5 x 4.78 in; (a) 2.31 in (strip unit 0.155 in, strips 1.13 in wide), (b) 2.31 in, gap 0.16 in; panel letters
  11.5 pt bold; one 0.52 in label gutter shared by both panels ("Ground truth" and "Adjacent NoFlip" set in two lines).
- Boards 0.94 in in the stacked (b): +6 % over round 1's five-column `_gs` cut (0.886 in). The +10 % of the brief (0.975 in)
  is not reachable at 5.5 in with five columns plus a label gutter that fits "Adjacent" at 8 pt (five boards of 0.975 in and
  four gaps leave 0.40 in for labels; "Adjacent" needs 0.49 in). The four-column round-1 base had 1.13 in boards; adding GS
  is what costs the size. Rotated row labels would reach +11 % (0.98 in) at the price of misaligned panel edges; not done.
- Side-by-side: 5.5 x 3.26 in; (b) 2.57 in wide with 0.64 in boards (+10 % over round 1's 0.58 in), rims 1.0 pt (1.6 pt in
  the stacked); (a) 2.93 in wide, strips 0.44 x 0.22 in, group titles in two lines, scenario titles 7 pt.

## Caveats

- **The draft's Table 2 is not the guarded table.** `paper/paper_draft.tex` still quotes pre-2026-09-19 cells; the figures
  draw the guarded arms and the tables above are the guarded numbers. The caption must use these.
- **Categorical IM exists only on the ray family** (dw-128ray / 16 / 8 / 5-ray, `appearance-fac`). A1's blank cells are
  the table's blanks, not a failed write. The 128-ray categorical arm is +0.64 / 0.29; its landing on the exact label cell is
  poor (`experiments/categorical_inverse/README.md`), which is why the A2 categorical IM frames show a slightly soft disc edge.
- The Rayworld scenario is one world (radius 1.0) seen through the drawn model's renderer: A2 shows radius-1.0 discs (its own
  bench geometry), A1 radius-0.5 discs on the same positions.
- "Standard" in A2's column titles is the 128-ray model; the caption names it. Titles are the same in A1 and A2 by design.
- The Othello writes are the 2026-09-19 guarded cache (unchanged by the categorical deployment, which touched discworld only).
- Nothing in `runs/`, `datasets/`, `logs/`, `pim/`, or the paper `.tex` was touched; the GPU was used only to build the two
  Rayworld cache families (one model at a time, freed after), never to fit anything.
