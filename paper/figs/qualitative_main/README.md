# qualitative_main — the main-text editability figure: (a) Rayworld, (b) Othello

Built 2026-09-21 from the appendix caches only (`.scratch/qualitative_edits_guarded_seed{0..5}_ctx8.pkl`,
`.scratch/othello_edits_guarded_cache.pkl`); no model was loaded, no write recomputed, no metric computed
except the canonical per-case Othello Edit Index (`pim.metrics.set_editability.edit_index_legal`) that the
optional "typical case" rule and the sidecars use. Strips are drawn by `paper/figs/qualitative_edits/make_figure.py`'s
`_panel` / `error`, boards by `paper/figs/qualitative_edits_othello/make_figure.py`'s `draw_board` (both imported by
`common.py`), style by `paper/figs/paper_style.py`. Every PDF is vector with Times New Roman embedded as TrueType
(rasters only for the observation strips, nearest interpolation); a 300-dpi PNG sits beside each PDF.

## Regenerate

    .pim/bin/python paper/figs/qualitative_main/rayworld_panel.py            # R1..R4 + pieces + rayworld_<opt>.json
    .pim/bin/python paper/figs/qualitative_main/othello_panel.py             # T1..T3 x {4col, 2col, 2row}, random rule, + pieces
    .pim/bin/python paper/figs/qualitative_main/othello_panel.py --rule typical
    .pim/bin/python paper/figs/qualitative_main/othello_panel.py --rule typical --gs --options T3   # with the GS row
    .pim/bin/python paper/figs/qualitative_main/composite.py [--rule typical]                      # the three composites

## Files

| file | what |
|---|---|
| `rayworld_R1.{pdf,png,json}` | five variants (Standard, Blink, 16-ray, 8-ray, 5-ray) x PI, GS (categorical), IM; seed 0; 5.5 x 1.72 in |
| `rayworld_R2.*` | Standard, 8-ray, 5-ray x PI, GS, GS (categorical), IM; seed 0; 5.5 x 2.39 in |
| `rayworld_R3.*` | **the hero**: Standard only, three scenarios (seeds 0, 1, 2) x PI, GS, IM, all continuous; 5.5 x 2.09 in |
| `rayworld_R4.*` | my extra cut: R3's three Standard scenarios plus a 5-ray column on scenario 1, with the GS (categorical) row; 5.5 x 2.10 in |
| `othello_T1_<cut>[_typical][_gs].*` | full boards, yellow tint by predicted mass everywhere, the squares whose legality the flip changes outlined |
| `othello_T2_<cut>[_typical][_gs].*` | full boards, tint ONLY the changed squares, edited tile pink |
| `othello_T3_<cut>[_typical][_gs].*` | **zoom**: every board of a variant cropped to the same 5 x 5 window (edited tile + changed squares + one-square margin), T1's tint and outlines |
| cuts | `4col` = four variants across, conditions down (5.5 in wide); `2col` = Standard vs Adjacent NoFlip across (half width, 2.65 in); `2row` = the same two variants DOWN, conditions across (5.5 x 2.52 in) |
| `_typical` | cases chosen by the typical rule (below) instead of random seed 0; `_gs` = with the GS row / column |
| `composite_R3_T3[_typical]_4col.*` | (a) R3 above (b) T3 with all four variants; 5.5 x 7.11 in |
| `composite_R3_T3[_typical]_2row.*` | **recommended**: (a) R3 above (b) T3 as Standard / Adjacent NoFlip rows by Unedited / Ground truth / PI / IM columns; 5.5 x 4.74 in |
| `composite_R3_T3[_typical]_sidebyside.*` | (a) left (3.47 in), (b) the two-column zoom right (2.03 in, boards 0.58 in); 5.5 x 2.9 in |
| `pieces/rayworld_<opt>/` | every strip of the option as its own PDF + PNG (`col<k>_<variant>_seed<s>_<row>`), plus `key_error_scale` (the ±1 bar) and `key_locators` |
| `pieces/othello_<opt>_<cut>[_typical][_gs]/` | every board (`<Variant>_case<i>_<condition>`), for T3 a faded full-board `_thumbnail` with the crop rectangle, and `key_boards` (tint / outline / edited tile). `2row` reuses `2col`'s boards |
| `rayworld_<opt>.json` | per column: variant, instance, run, seed, edit object, locator positions, arms drawn (point, alpha) for both blocks, and the guarded Table 2 cells of that run |
| `othello_cases_{random,typical}.json` | per variant: run, instance, case id, edited tile, changed squares, legal sets, window, per-case Edit Index of every condition, arms drawn, guarded Table 2 cells |

## Recommendation

**(a) R3, (b) T3, composed as `composite_R3_T3_typical_2row`.** R3 carries Sevan's message on the model that
matters (128 rays): PI and GS return streaky frames that move no disc on any of three worlds, IM renders the
teleported disc on all three. R1 and R2 spend the page on the coarse variants, which are the foil, not the hero;
R4 keeps the foil as a fourth column if the text wants the "coarse and categorical makes it easy" point in the same
figure (I would not: the eye goes to the odd column). T3 is the only Othello option whose squares are readable at
print size; T1 shows the same content at a third of the square size, T2 is the cleanest read of the metric (only the
scored squares carry colour) but hides what the distribution does elsewhere, so a destroyed output would look like a
miss instead of a wreck. The 2row (b) cut costs 2.4 in less page than the four-variant cut and keeps the two variants
the text contrasts (all editors land on Standard, none on Adjacent NoFlip); the four-variant composite is there if
the page allows it. Side-by-side is the most compact (2.9 in) but its boards are 0.58 in.

## Selection rules

- **Rayworld.** The scenario is the appendix's: one two-disc teleport case from the edit-set generator (seed =
  generator seed) under the radius-1.0 geometry, rendered under every variant's own renderer, edit at frame 20;
  the write targets the pre-dynamics state; single next-step frames, not rollouts. R1 / R2 use seed 0 like the
  appendix figure. R3 / R4 use the first three cached seeds (0..5) whose edited disc is visible before the edit on
  the Standard variant (finite origin locator): seeds 0, 1, 2 (seeds 3 and 4 have the disc occluded or out of view
  before the edit, so only the destination line could be drawn; seed 5 also qualifies). Edited object 0 in all
  three; origin / destination ray centres 107.5 / 28.5, 31.5 / 81.5, 109.5 / 61.5.
- **Othello, eligibility (both rules).** Among the 1000 bench cases of a variant (all at move 20), those where at
  least 3 squares change legality (|legal_pre XOR legal_post| ≥ 3) and the window {edited tile} ∪ changed squares
  + one-square margin fits 5 x 5 (63 / 391 / 367 / 67 eligible cases on Standard / Adjacent Flip / Adjacent NoFlip /
  Standard NoFlip). The size floor keeps the scored region visible; the window cap keeps one uniform zoom.
- **`random` (files without a suffix):** one eligible case per variant from `numpy.random.default_rng(0)`, drawn
  in the order Standard, Adjacent Flip, Adjacent NoFlip, Standard NoFlip: cases 899, 668, 503, 219.
- **`typical` (`_typical` files):** the eligible case whose per-case Edit Indices (symdiff construction) summed over
  PI, GS, IM are closest to the variant's population means (the guarded arms' indices): cases 842, 986, 627, 262.
  This is an editorial rule and the caption must say so. I recommend it for the main text because the seed-0 draw
  puts an Adjacent NoFlip case in the figure where IM lands (+0.88) against a population index of −0.03, which
  would read as the typical outcome; the typical picks sit within 0.03 of every population mean on Standard and
  within 0.2 elsewhere (table below).
- **Arms.** Every editor is drawn at the arm the tables report: `pim.metrics.selection.best_arm`, the best Edit
  Index inside the fidelity guard (ratio ≤ 1), the unguarded best only where an editor has no arm inside the
  guard. Never changed here; copied from the caches into the sidecars.

## What is drawn (caption facts)

- **(a)** Context = the last 8 observed frames fed to the model, time downward, the same `gray` map on the dark
  panel as every waterfall, fixed 0..1. Unedited = the model's next frame with no edit; Ground truth = the clean
  render of the edited world at the edit frame (the reference the Edit Index scores against); PI / GS / IM = the
  next frame after each write, each with a strip beneath showing prediction minus truth on the canonical
  signed-error map (red = under-prediction, green = over, black = correct), fixed ±1 scale, the prediction clipped
  to 0..1 before differencing (the scorer does not clip). Cyan line = ray centre of the edited disc before the edit
  (its unedited rays), pink = after (its target rays). All edit rows in R3 are the continuous (full-state,
  Cartesian) target; "GS (categorical)" rows are the factorised appearance target.
- **(b)** Unedited row = pre-edit board with the model's next-move distribution; every other row = the post-edit
  board (the edited tile, pink outline, has flipped colour). Ground truth = uniform over the post-edit legal moves.
  Yellow tint = predicted probability, fully tinted at 0.02 and above, (p / 0.02)^0.6 below (the appendix's
  `draw_board` defaults; so the tint is essentially "does this square carry more than 2 % of the mass"). Thin dark
  outline = the squares whose legality the flip changes, i.e. exactly the squares the Edit Index scores
  (symmetric-difference construction). T3 windows are 5 x 5 squares around those; `_thumbnail` pieces show the
  window on the full board. Boards are absolute colours replayed under the instance's own rules.

## The population numbers the panels are read against (guarded `best_arm`, from `runs/<run>/scores.json`)

Edit Index / Fidelity Ratio at the drawn arm (residual point pt, step size a). `*` = no arm inside the guard, the
unguarded best is reported and drawn. Unedited = the unedited model's index on the same block.

Rayworld, continuous target (block `cartesian`; the PI / GS / IM rows of R1..R4):

| variant | run | unedited | PI | GS | IM |
|---|---|---|---|---|---|
| Standard | noise_ablation/L-dw-noiseless-20m | −0.93 | −0.10 / 0.96 (pt5, a12) | −0.16 / 0.96 (pt0, a0.35) | +0.59 / 0.34 (pt6) |
| Blink | blink_ablation/L-dw-blink-20m | −0.92 | +0.02 / 0.99 (pt2, a35) | −0.09 / 0.98 (pt0, a0.7) | +0.51 / 0.45 (pt7) |
| 16-ray | ray_ablation/L-dw-16ray-20m | −0.91 | +0.16 / 0.96 (pt2, a100) | −0.10 / 0.90 (pt0, a0.7) | +0.66 / 0.28 (pt6) |
| 8-ray | ray_ablation/L-dw-8ray-20m | −0.90 | +0.21 / 1.00 (pt3, a175) | −0.07 / 0.88 (pt0, a0.7) | +0.71 / 0.27 (pt6) |
| 5-ray | ray_ablation/L-dw-5ray-20m | −0.89 | +0.14 / 0.85 (pt4, a175) | −0.10 / 0.86 (pt0, a0.7) | +0.81 / 0.23 (pt0) |

Rayworld, categorical target (block `appearance-fac`; only the "GS (categorical)" rows are drawn):

| variant | unedited | PI (not drawn) | GS (drawn) | IM (not drawn) |
|---|---|---|---|---|
| Standard | −0.93 | −0.34 / 0.99 (pt4, a0.25) | +0.33 / 0.71 (pt0, a0.35) | no arm in the block today (see caveats) |
| Blink | −0.92 | −0.36 / 0.98 (pt3, a0.25) | +0.25 / 0.92 (pt0, a0.7) | no arm in the block today |
| 16-ray | −0.91 | −0.06 / 0.96 (pt1, a20) | +0.28 / 0.85 (pt0, a1.5) | +0.82 / 0.28 (pt6) |
| 8-ray | −0.90 | +0.38 / 0.95 (pt1, a60) | +0.46 / 0.54 (pt0, a0.35) | +0.87 / 0.27 (pt6) |
| 5-ray | −0.91 | +0.51 / 0.70 (pt1, a20) | +0.60 / 0.47 (pt0, a0.35) | +0.91 / 0.25 (pt5) |

Othello (symmetric-difference Edit Index, `edit_index_symdiff`):

| variant | run | unedited | PI | GS | IM |
|---|---|---|---|---|---|
| Standard | initial_othello_comparison/L-oth-20m | −0.93 | +0.82 / 0.30 (pt4, a3) | +0.83 / 0.28 (pt4, a0.2) | +0.81 / 0.38 (pt5) |
| Adjacent Flip | adjacent_flip_ablation/L-oth-adjacent-flip-20m | −0.96 | +0.35 / 0.83 (pt2, a5) | −0.06 / 0.79 (pt0, a0.2) | +0.66 / 0.51 (pt5) |
| Adjacent NoFlip | adjacency_ablation/L-oth-adjacent-20m | −0.96 | −0.23 / 0.81 (pt1, a10) | −0.16 / 6.68* (pt2, a1.5) | −0.03 / 0.74 (pt1) |
| Standard NoFlip | flip_ablation/L-oth-noflip-20m | −0.98 | −0.95 / 1.00 (pt6, a0.25) | −0.52 / 4.78* (pt0, a1.5) | −0.55 / 2.58* (pt7) |

Per-case Edit Index (symdiff) of the drawn Othello cases, for the caption's "read against" sentence:

| variant | typical case | Unedited / PI / GS / IM | random (seed 0) case | Unedited / PI / GS / IM |
|---|---|---|---|---|
| Standard | 842 | −0.99 / +0.81 / +0.83 / +0.80 | 899 | −0.78 / +1.00 / +1.00 / +1.00 |
| Adjacent Flip | 986 | −1.00 / +0.23 / −0.04 / +0.64 | 668 | −1.00 / +0.22 / −0.75 / +0.07 |
| Adjacent NoFlip | 627 | −1.00 / −0.43 / +0.05 / +0.07 | 503 | −1.00 / −1.00 / −0.69 / +0.88 |
| Standard NoFlip | 262 | −1.00 / −1.00 / −1.00 / −0.59 | 219 | −1.00 / −1.00 / +0.02 / −1.00 |

No population statistic exists for the drawn Rayworld scenarios beyond the table above (the caches hold frames,
not per-case indices); the appendix's `more_seeds/` shows how much the picture moves with the scenario.

## Caveats

- **The draft's Table 2 is not the guarded table.** `paper/paper_draft.tex` still quotes several pre-2026-09-19
  cells (e.g. Rayworld standard PI +0.20 / 1.71, Othello adjacent-flip PI +0.40 / 2.09, adjacent-noflip IM +0.00 /
  1.48); the figures draw the guarded arms (Rayworld standard PI −0.10 / 0.96; adjacent-flip PI +0.35 / 0.83;
  adjacent-noflip IM −0.03 / 0.74) and the tables above are the guarded numbers. The caption must use these.
- **Categorical IM is never drawn.** The categorical-block IM arms were cleared on 2026-09-20 for the categorical
  inverse map; Standard and Blink have no IM arm in `appearance-fac` today, and the cached categorical-block IM
  frames are the OLD full-state map (its points no longer match the new arms on 8-ray and 5-ray). The brief's
  options only ask for continuous IM, which the caches hold at the current arms (unchanged by the deployment).
- The Rayworld scenario is one world (radius 1.0, the coarse family's geometry) seen through every variant's
  renderer, so the Standard column shows radius-1.0 discs, not the radius-0.5 discs of its own bench.
- Nothing in `runs/`, `datasets/`, `logs/`, `.scratch/`, `pim/`, or the paper `.tex` was touched; no GPU was used.
