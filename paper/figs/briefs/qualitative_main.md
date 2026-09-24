# BRIEF — the main-text qualitative editability figure (Rayworld + Othello, editorial)

Read `paper/figs/FIGURE_BRIEF_COMMON.md` first. Folder: `paper/figs/qualitative_main/`. Mostly CPU (caches); GPU
only if you need a new Rayworld scenario.

## Why this figure exists
The full panels (`paper/figs/qualitative_edits/qualitative_edits_seed0_paired.pdf`, 5 variants × 6 editors, and
`paper/figs/qualitative_edits_othello/othello_edits_seed0_cols.pdf`, 4 variants × 5 conditions) stay in the appendix
(`more_seeds/` gives the random extra examples). The MAIN TEXT gets one editorial figure with two panels,
(a) Rayworld and (b) Othello, that gets the point across with less on the page.

Sevan on the Rayworld message: "the model is not exposing steerable representations of our interpretable 2D interface,
unless we make it very coarse and effectively turn the observations themselves into the state, neither of which is
desirable. But if we learn the inverse map, even a poor one, then we can operate directly in our 2D interface state and
the generative model obeys" — and the IM generations look good. So the hero is the standard 128-ray model: PI and GS
fail there, IM lands cleanly. 5-ray is the "coarse makes it easy" foil, not the hero.
Sevan on Othello: the current boards are hard to read; try zooming on the region of the board that matters.

## Data (no recomputation of writes)
- Rayworld predictions per seed: `<repo>/.scratch/qualitative_edits_guarded_seed{0..5}_ctx8.pkl` — the dict
  `build()` returns in `paper/figs/qualitative_edits/make_figure.py` (`cols[name]` has `context`, `gt`,
  `cont`/`cat` dicts with `unedited`, `PI`, `GS`, `IM`, `ghost_x`, `target_x`, `arms`). Read that script's `draw()`
  to see how the rows are assembled; reuse its `_panel`, `error`, `STRIP`/`CTX_ROW` conventions by importing the module.
  Arms are the tables' guarded arms (`pim.metrics.selection.best_arm`) — keep them; never pick a different arm.
- Othello: `<repo>/.scratch/othello_edits_guarded_cache.pkl` — `{run: compute(run)}` with, for all 1000 bench cases,
  `board_pre`, `board_post`, `pos` (edited tile), `legal_pre`, `legal_post`, `probs` for `Unedited`, `Ground truth`,
  `PI`, `GS`, `IM`. Runs: standard `initial_othello_comparison/L-oth-20m` and the three variants (see `VARIANTS` in
  the Othello script). Drawing: import `draw_board`.
- If you generate a new Rayworld seed, run the existing script (`--seed k`) so its cache holds the predictions; do
  not write a new prediction path.

## Deliver (pieces + composite; several options)
**Rayworld panel options** (all with the context waterfall, Unedited, Ground truth rows; paired signed-error strips
under each edit row as in `_paired`; locators on):
- R1 five columns (Standard, Blink, 16-ray, 8-ray, 5-ray) × three edit rows: PI (continuous), GS (categorical), IM
  (continuous). Half the height of the appendix figure.
- R2 three columns (Standard, 8-ray, 5-ray) × four edit rows: PI cont, GS cont, GS cat, IM cont.
- R3 the hero: Standard only, three different scenarios (seeds) as columns, rows Unedited / GT / PI / GS / IM
  (continuous). Shows IM's generations across worlds.
- Propose one more if you see a better cut, and say why.
**Othello panel options** (rows Unedited, Ground truth, PI, IM; GS optional):
- T1 all four variants, full boards, the symmetric-difference squares (legal_pre XOR legal_post — the squares the
  Edit Index scores) outlined thinly, yellow tint by mass as now.
- T2 full boards, tint ONLY the symmetric-difference squares by mass (everything else plain), edited tile pink.
- T3 zoom: crop each board to the bounding box of {edited tile} ∪ symdiff squares plus a one-square margin (same
  crop for every row of a column), boards drawn large; a faint full-board thumbnail with the crop rectangle is
  optional as a separate piece.
- Do Standard vs Adjacent NoFlip as the two-column cut of each option as well.
**Case selection**: for Othello choose cases by a rule (e.g. at least 3 squares in the symmetric difference, then
random with seed 0); for Rayworld use the cached seeds and say which. Record ids / seeds / arms in a sidecar JSON.
**Composite**: a full-width figure with (a) the recommended Rayworld option above (b) the recommended Othello
option, panel letters only, editor labels shared. Also each panel alone.
**README** with the caption facts: arms drawn (point, alpha), guarded, selection rule, what the tints and strips mean,
population statistics the panel should be read against (the Table 2 numbers for those cells, copied from
`runs/<run>/scores.json` via `best_arm`, printed in the README not in the figure).
