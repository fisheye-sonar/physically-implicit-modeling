# qualitative_edits_othello — real boards, the model's next-move distribution under each edit

`make_figure.py --seed <k> --layout rows|cols --games <n>` → `othello_edits_seed<k>_<layout>.{pdf,png,json}`.

**Variants** (`VARIANTS`): Standard, Adjacent Flip, Adjacent NoFlip, Standard NoFlip — one run each.
**Boards** are cases of each run's canonical edit bench (`load_benchmark`; real games, every case at move
20 in the shipped set), `--games` (default 1) of them per variant chosen by `--seed`; the variants do not share a
scenario since their rules differ. Boards are replayed under the instance's own rules
(`tokens_and_labels`, absolute colours). **Conditions**: Unedited (pre-edit board, the model's
distribution), Ground truth (post-edit board, uniform over its legal moves — the reference the Edit
Index is scored against), PI / GS / IM (post-edit board, the distribution after the write at the run's
scored best arm under the symmetric-difference construction, called exactly as the scorer calls them:
`linear_arm` pinv, `grad_steer_arm`, `inverse_arms`). Squares are tinted yellow by probability mass: fully at `--tint-scale` (0.02 — any square carrying
more than 2 % of the mass) and (p / scale)^`--gamma` below; the edited tile is outlined pink (`--no-locator` drops it).

**Layouts.** `rows` (default): variants down, the five conditions across, repeated per game — 5·n boards
wide by 4 tall, the paper-width shape. `cols`: the sketch's arrangement, variants across and conditions
down (n games per variant side by side). Every write is computed once for all 1000 cases per run and
cached in `.scratch/othello_edits_cache.pkl`, so a new seed, layout or game count is a redraw
(`--recompute` refreshes it). The sidecar JSON records the cases, move numbers and arms used.

## 2026-09-21: Arial and the marking

- Fonts and page come from `paper/figs/paper_style.py` (`ps.apply()`: Arial, TrueType embedding, white page,
  no outer padding on save) instead of the script's own Times New Roman block; board colours are `ps.BOARD_*`.
- **Marking (Sevan's semantics, shared with the main-text figure through `marked_squares` / `mark_color` /
  `mark` / `mark_key`).** On the pre-edit board (Unedited) the flipped tile AND every square whose legality the
  flip switches (`legal_pre` XOR `legal_post`, exactly the squares the symmetric-difference Edit Index scores) are
  outlined in cyan (`ps.ORIGIN_C`); on every post-edit board (Ground truth, PI, GS, IM) the same squares in pink
  (`ps.DEST_C`). Nothing else is outlined; the yellow tint is unchanged. A two-swatch key ("pre-edit" / "post-edit")
  sits once per figure in the free corner above the row labels. The old lone pink outline of the edited tile is gone
  (`draw_board(..., locator=False)`); `--no-locator` now drops the marks.
- The writes are the 2026-09-19 guarded cache, unchanged (`.scratch/othello_edits_guarded_cache.pkl`); the
  2026-09-21 outputs (seed 0 and `more_seeds/`) are redraws.

## Which arm is drawn (2026-09-19)

Each editor is drawn at the arm the TABLES report — `pim.metrics.selection.best_arm`: the best Edit Index among the
arms inside the fidelity guard (ratio ≤ 1), the unguarded best only where an editor has none. Until 2026-09-19 the
figure used the scorer's unguarded `best`, which for PI on the fine-ray instances and on the adjacency Othello
variants is a different (more destructive) write than the one whose numbers the paper quotes. The cached writes in
`.scratch/` carry `_guarded` in their names, so the old caches are not reused.

## more_seeds/

`more_seeds/seed<k>/` holds the same figure for other seeds (k = 1…5: `--seed <k> --out-dir more_seeds/seed<k>`,
file names `othello_edits_seed<k>_{cols,rows}`), to see how much the picture depends on the drawn scenario. Same arms, same models — only the
scenario (discworld) or the sampled bench cases (Othello) change.
