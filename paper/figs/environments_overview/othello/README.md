# Othello and its three rule variants (fig:othello_and_variants)

Pieces and composites for the environment figure of the paper's Othello section. Round 2 (2026-09-21, after
Sevan's review): option O2 (before / after) is the main figure; panels (a) standard, (b) adjacent-flip and
(c) adjacent-noflip show ONE shared board, a real standard position that is provably reachable under the other
two rule sets, so the three panels differ only in the legal squares and in what the move flips; (d)
standard-noflip keeps its own board. All text is Arial and every PDF is cropped to its content with no padding
(`paper_style` as of 11:50 PT).

Everything canonical is imported: rules from `pim.environments.othello.corpus.rules_of`, board replay and
legality from the vendored `OthelloBoardState` (`umpire`, `get_valid_moves`, `tentative_move`), games from
`pim.environments.othello.bench.load_benchmark`, the board drawing from
`paper/figs/qualitative_edits_othello/make_figure.py::draw_board`, colours and fonts from
`paper/figs/paper_style.py`. No metric is computed anywhere in this folder.

Regenerate (CPU, ~20 s and ~10 s):

    .pim/bin/python paper/figs/environments_overview/othello/make_figure.py        # --seed 0 --move 14 --min-flips 1
    .pim/bin/python paper/figs/environments_overview/othello/make_predictive.py    # --seed 0   (appendix)

## Files

Top level (`.pdf` vector, Arial embedded as TrueType, page box = content; a 300 dpi `.png` preview beside each):

| file | what it is |
|---|---|
| `composite_O2.pdf` | **Main figure** (5.5 x 3.5 in): (a) to (c) the shared board before (top) and after (bottom) the shared move c4, (d) standard-noflip on its own game; arrows between rows; key strip beneath. |
| `composite_O2_alt.pdf` | Same boards, per-rule-set chosen moves: (a) the standard move that flips the most discs (d1, three flips), (b) the adjacency-legal move that flips the most (c4), (c) a square legal only by adjacency (c6). |
| `composite_O1.pdf` | One board per variant (5.5 x 2.0 in): the shared board for (a) to (c) with the discs c4 would flip ringed, (d) own game. |
| `composite_O3.pdf` | Three consecutive positions per variant from each instance's own game (tall, 5.5 x 5.1 in; appendix material). |
| `legend_key.pdf` | The key alone (3.4 x 0.3 in): legal move, chosen move, flipped disc. |
| `predictive_composite.pdf` | Appendix: per variant the board tinted by its true legal set ("legal moves") over the same board tinted by the trained model's unedited next-move distribution ("model"). |
| `boards.json` | Sidecar: the shared board (case, move, board array, standard moves, the placement orders that prove reachability, both legal sets, chosen and alternative moves, the search record), the failed mid-game fallback, and every variant's own-game facts. |
| `predictive.json` | Sidecar for the appendix figure: cache column, run, instance, case id, legal-set size, tint parameters. |

`pieces/` (boards 2.0 in on the page, one PDF + PNG per element; `<v>` in `standard`, `adjacent_flip`,
`adjacent_noflip`, `standard_noflip`):

| piece | board | what it shows |
|---|---|---|
| `<v>_legal` | shared (a to c), own (d) | The position with the mover's legal squares dotted. |
| `<v>_before`, `<v>_after`, `<v>_O2_pair` | shared (a to c), own (d) | O2: before = legal dots + the chosen move as a ghost disc with a cyan ring; after = the placed disc ringed cyan, the flipped discs ringed pink; the pair with an arrow. |
| `<v>_before_alt`, `<v>_after_alt`, `<v>_O2_pair_alt` | shared (a to c) | O2 with the per-rule-set chosen moves. |
| `<v>_O1` | shared (a to c), own (d) | O1: legal dots, ghost disc, the discs the move would flip ringed (still in their pre-move colour). |
| `<v>_O3_1`, `_O3_2`, `_O3_3`, `<v>_O3_seq` | own (all four) | The positions after moves 14, 15, 16 of the own game: the disc just placed ringed cyan, the discs it flipped ringed pink, the next mover's legal squares dotted. |
| `<v>_terminal` | generator | The final position of one whole game from the instance's generator (`synthetic_games(1, seed=0, **rules_of(inst))[0]`, game 0 of the training pool by the index law). Standard-noflip's is the checkerboard every game of that rule set ends in. |
| `predictive_<v>_legal`, `_model`, `_pair` | bench (cache) | The appendix figure's boards, singly and as a pair. |

## Markers (defined once in `make_figure.py`, colours from `paper_style`)

- Legal move: small yellow dot (`ps.BOARD_TINT`, #ffe600) on each square the mover may play.
- Chosen move: cyan ring (`ps.ORIGIN_C`, #00bcd4); before the move around a half-transparent disc of the mover's
  colour (alpha 0.5), after the move around the placed disc.
- Flipped disc: pink ring (`ps.DEST_C`, #ff4fa3), the pink the qualitative figures use for a tile whose colour
  changed. In O1 the ringed discs are in their pre-move colour (they would flip); in after boards and O3 frames
  they have flipped.
- Boards: `draw_board`'s geometry and colours. Rows a to h run top to bottom, columns 1 to 8 left to right, as in
  the vendored code ("c4" = row c, column 4).

## The shared board (panels a, b, c)

Position after move 6 of standard bench case 182 (`datasets/othello/oth-uniform/edits/v1/cases_1000.pkl`, a
held-out 20-move game), 10 discs, black to move. Rows a to h top to bottom, columns 1 to 8 left to right:

    a  . . . . . . . .
    b  . . . . . . . .
    c  . . W . . . . .
    d  . W W W B . . .
    e  . . B B W . . .
    f  . B B . . . . .
    g  . . . . . . . .
    h  . . . . . . . .

- Standard game: `d3 e3 f2 c3 f3 d2` (black first; several discs flipped along the way).
- Adjacent-noflip and adjacent-flip: the placement order `f3 d3 f2 d2 e3 c3` (black first, strict alternation)
  rebuilds the same board with no flip at any step; under adjacent-flip none of these placements encloses a
  disc, so the same order is valid there too. Both replays are re-run and asserted equal to the standard board
  in the script; the orders are in `boards.json` under `shared_board.placement_orders`.
- Legal squares for black: standard (enclosure) `b3 c1 c2 c4 c5 d1 e6 f5` (8); adjacency (b and c, identical)
  `c4 c5 c6 d6 e1 e2 e6 f1 f4 f5 g1 g2 g3 g4` (14). Legal under all three: `c4 c5 e6 f5`.
- Shared chosen move: **c4** (black), which encloses d4 against e4 and flips it under (a) and (b); under (c) the
  disc is placed and nothing changes colour. Every move legal under all three flips exactly one disc here.
- Alternative moves (`_alt`): (a) `d1`, flipping `d2 d3 d4` along row d against d5; (b) `c4`; (c) `c6`, adjacent
  to a black disc but enclosing nothing, so illegal under the enclosure rule.

### How it was found (`shared_board` / `reach` in `make_figure.py`)

Candidates are the positions after m moves of the 1000 standard bench games, m = 14, 13, ..., 4. For each, a
depth-first search over placement orders from the opening uses the vendor's own legality (`get_valid_moves`,
`umpire`), only lands on the target's occupied squares, and keeps a child only while every disc on the board
has the target's colour. Under adjacent-noflip this is exact (discs never change, passes follow the vendor's
forfeit logic). Under adjacent-flip it accepts only flip-free orders, a sufficient condition; an adjacency-legal
standard prefix (same sequence, same flips, same board) is accepted as well. Visited states are skipped, so
every candidate was searched exhaustively (at most ~1600 states; cap 20000 never reached). The scan stops at
the first m with a survivor; among survivors the one whose best shared move flips the most discs wins, then the
largest legal-set difference, then the lowest case id.

Result: **no position after 7 to 14 moves survives; after 6 moves three do (cases 138, 182, 745)**, each with a
flip-free adjacency order, black to move, 4 to 5 shared legal moves, all shared moves flipping one disc; case
182 wins on the legal-set difference (14 squares) and the lowest id. The fallback layout (a mid-game board after
8 to 14 moves shared by (a) and (b) only) does not exist for this bench within the same sufficient condition (0
of 1000 games at every m from 8 to 14), so it was not built; `boards.json` records the negative result. An
order with flips that happens to end on the target board is not excluded by this search, so "not reachable"
for adjacent-flip means "not reachable by a flip-free order or the standard sequence".

## Own-game boards (panel d, the O3 frames, terminal boards)

Seeded rule (`--seed 0`, `--move 14`, `--min-flips 1`): the instance's 1000 bench cases in a seeded permutation,
the first whose moves 15 and 16 are regular (no pass) and, where the rule set flips, whose move 15 flips at
least one disc. Boards after move 14 (18 discs), black to move, the chosen move = the game's real move 15.

| variant | instance | bench case | move 15 | flips | legal at move 14 | O3 frame flips (moves 14, 15, 16) |
|---|---|---|---|---|---|---|
| standard | `oth-uniform` | 459 | g4 | f4, f5 | 12 | 1, 2, 4 |
| adjacent-flip | `oth-adjacent-flip` | 358 | b5 | c4 | 19 | 0, 1, 2 |
| adjacent-noflip | `oth-adjacent` | 72 | c4 | none | 22 | 0, 0, 0 |
| standard-noflip (panel d) | `oth-noflip` | 943 | g2 | none | 7 | 0, 0, 0 |

Only standard-noflip's own game appears in the O2 / O1 composites; the other three own games appear in the O3
composite and pieces only.

## Appendix figure (`make_predictive.py`)

One random case per variant (`rng.integers` over the 1000 cached cases, seed 0) from
`.scratch/othello_edits_guarded_cache.pkl` (the qualitative Othello figure's cache, guarded best arms,
2026-09-19). Boards are the bench's 20-move positions; the "legal moves" row is
`pim.metrics.set_editability.uniform_over_legal(legal_pre)`, the "model" row the cached `probs["Unedited"]`.
Tint as `draw_board` draws it (full at 0.02 probability mass and above, `(p / 0.02) ** 0.6` below).

| panel | run | bench case | legal squares |
|---|---|---|---|
| (a) | `initial_othello_comparison/L-oth-20m` | 850 | 18 |
| (b) | `adjacent_flip_ablation/L-oth-adjacent-flip-20m` | 636 | 24 |
| (c) | `adjacency_ablation/L-oth-adjacent-20m` | 511 | 19 |
| (d) | `flip_ablation/L-oth-noflip-20m` | 269 | 8 |

## Caption facts (main figure, `composite_O2`)

- (a) to (c): one real position (a standard bench game after six moves, black to move) that can also arise
  under the adjacency rule sets, so only the rules differ. Yellow dots: the squares black may play; the enclosure
  rule allows 8, adjacency 14. Cyan ring: the move c4, a ghost disc before, the placed disc after. Pink ring: the
  disc it flips (d4, enclosed against e4) under standard and adjacent-flip; under adjacent-noflip the disc is
  placed and nothing changes colour.
- (d): a standard-noflip game after 14 moves; the enclosure rule allows 7 squares, the move g2 flips nothing, and
  the discs already form the checkerboard (orthogonal neighbours always differ) that every game of that rule set
  ends in (`pieces/standard_noflip_terminal.pdf`).
- `_alt`: (a) d1 flips the three white discs d2, d3, d4 in a row; (c) c6 is legal only because it touches a black
  disc.
- No number appears inside any figure; the only text is panel letters, the key's three labels, and the appendix
  figure's two row labels.

## Notes

- PDFs are cropped to their content with no padding (composites exactly 5.5 in wide), so spacing is set in
  LaTeX; panel letters sit at the top edge of the box.
- The shared board is early-game (10 discs) because nothing later survives the reachability test; the alt
  composite is the way to show a longer flip line on the same board.
- The appendix figure's rows look identical at the canonical tint saturation (0.02); a model that puts 1/|L| on
  every legal square looks exactly like the legal-set board, which is the intended message.
