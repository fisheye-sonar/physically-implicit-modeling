# Othello and its three rule variants (fig:othello_and_variants)

Two candidate main figures for the paper's Othello section (round 4, 2026-09-22). Each panel is two boards: the
top row, **Legal Moves**, fills every square the mover may play solid in the board tint; the bottom row, **Board
Update**, shows the position after the chosen move, that disc ringed cyan and every disc it flips ringed pink.
Panels (a) standard, (b) adjacent-flip and (c) adjacent-noflip show ONE shared board, a real standard position
that is provably reachable under the other two rule sets, so the panels differ only in the legal squares and in
what a move flips; (d) standard-noflip is on its own game. Row names are rotated down the left side, the key is
a column on the right, panel letters are bold. Text is Arial, every PDF is cropped to its content with no
padding; the composites are 5.81 x 2.67 in after the crop (the key column sets the width; the boards are a fixed 1.09 in each).

Everything canonical is imported: rules from `pim.environments.othello.corpus.rules_of`, board replay and
legality from the vendored `OthelloBoardState` (`umpire`, `get_valid_moves`, `tentative_move`), games from
`pim.environments.othello.bench.load_benchmark`, the board drawing from
`paper/figs/qualitative_edits_othello/make_figure.py::draw_board`, colours and fonts from
`paper/figs/paper_style.py`. No metric is computed here.

Regenerate (CPU, ~15 s):

    .pim/bin/python paper/figs/environments_overview/othello/make_figure.py     # --seed 0 --move 14 --min-flips 1

## Files

| file | what it is |
|---|---|
| `composite_O2.pdf` / `.png` | 5.81 x 2.67 in. (a) d1 flips d2, d3, d4; (b) c4 flips d4; **(c) c4**, the same board and move as (b), flips nothing; (d) own game, g2 flips nothing. Key column on the right. |
| `composite_O2_altmove.pdf` / `.png` | Same, except **(c) f5**: a move that would flip e5 under the flip rules (it is legal under all three rule sets) and flips nothing here. |
| `legend_key.pdf` / `.png` | The key alone (0.87 x 0.75 in), the same vertical column the composites carry: legal move, chosen move, flipped token. |
| `boards.json` | Sidecar: the shared board (case, move, board array, standard moves, the placement orders that prove reachability, both legal sets, the search record), every panel's move and flips for both composites, the own-game rule and its picks. |
| `make_figure.py` | The one script (search, drawing, composites). |
| `pieces/<variant>_before.pdf`, `_after.pdf`, `_O2_pair.pdf` | Each panel's two boards at 2.0 in and the pair with an arrow (`standard`, `adjacent_flip`, `adjacent_noflip`, `standard_noflip`); `adjacent_noflip_*_altmove.pdf` are panel (c) of the altmove composite. PNG previews beside them are gitignored. |

(`make_predictive.py`, `predictive.json`, `predictive_composite.*` and `pieces/predictive_*` belong to the
appendix figure being moved to `paper/figs/predictive_quality/`; not part of this figure. Earlier options O1 and
O3, the terminal boards and the round-2 pieces were pruned on 2026-09-21; the round-2 tree in git has them.)

## Markers (defined once in `make_figure.py`, colours from `paper_style`)

- Legal move, top row only: the whole square filled in the board tint (`ps.BOARD_TINT`, #ffe600) at full
  strength, through `draw_board`'s own tint path, so a legal square looks exactly like a fully tinted square in
  the prediction and qualitative figures. Discs are drawn on top.
- Chosen move, bottom row only: cyan ring (`ps.ORIGIN_C`, #00bcd4) around the placed disc. The top row carries
  no chosen-move marker: the ring is redundant with the row below, and a half-transparent ghost disc over a
  filled square reads as a third disc colour. To restore the ghost, pass
  `ghost=(Q["placed"], Q["board"][Q["placed"]])` in `pair`'s before spec (`board` still supports it).
- Flipped disc, bottom row only: pink ring (`ps.DEST_C`, #ff4fa3), the pink the qualitative figures use for a
  tile whose colour changed.
- Boards: `draw_board`'s geometry and colours. Rows a to h run top to bottom, columns 1 to 8 left to right, as in
  the vendored code ("c4" = row c, column 4).

## The shared board (panels a, b, c)

Position after move 6 of standard bench case 182 (`datasets/othello/oth-uniform/edits/v1/cases_1000.pkl`, a
held-out 20-move game), 10 discs, black to move:

    a  . . . . . . . .
    b  . . . . . . . .
    c  . . W . . . . .
    d  . W W W B . . .
    e  . . B B W . . .
    f  . B B . . . . .
    g  . . . . . . . .
    h  . . . . . . . .

- Standard game: `d3 e3 f2 c3 f3 d2` (black first; discs flipped along the way).
- Adjacent-noflip and adjacent-flip: the placement order `f3 d3 f2 d2 e3 c3` (black first, strict alternation)
  rebuilds the same board with no flip at any step; under adjacent-flip none of these placements encloses a
  disc, so the same order holds there. Both replays are re-run and asserted equal to the standard board by the
  script; the orders are in `boards.json` under `shared_board.placement_orders`.
- Legal squares for black: enclosure rule `b3 c1 c2 c4 c5 d1 e6 f5` (8, panel a); adjacency rule
  `c4 c5 c6 d6 e1 e2 e6 f1 f4 f5 g1 g2 g3 g4` (14, panels b and c, identical). Legal under all three:
  `c4 c5 e6 f5`; each of them encloses exactly one disc (c4 and c5 enclose d4, e6 and f5 enclose e5).

Panel moves (rules in `boards.json` under `panel_moves`):
- (a) the enclosure-legal move flipping the most discs: **d1**, which encloses d2, d3, d4 along row d against d5.
- (b) the adjacency-legal move flipping the most discs (adjacency-only squares enclose nothing, so this is the best
  move legal under all three): **c4**, enclosing d4 against e4.
- (c) `composite_O2`: **c4**, the same move as (b), so the before boards of (b) and (c) are identical and the
  after boards differ only in d4. `composite_O2_altmove`: **f5**, the move legal under all three rule sets that
  lies farthest from c4 (three squares); it would flip e5 under the flip rules and flips nothing here.
- (d) the own game's real next move: **g2** (standard-noflip case 943, board after move 14, black to move, 7
  legal squares `a4 b3 e2 e6 f5 g2 h3`), flipping nothing.

### How the shared board was found (`shared_board` / `reach` in `make_figure.py`)

Candidates are the positions after m moves of the 1000 standard bench games, m = 14, 13, ..., 4. For each, a
depth-first search over placement orders from the opening uses the vendor's own legality (`get_valid_moves`,
`umpire`), only lands on the target's occupied squares, and keeps a child only while every disc on the board
has the target's colour. Under adjacent-noflip this is exact (discs never change; passes follow the vendor's
forfeit logic). Under adjacent-flip it accepts only flip-free orders, a sufficient condition; an adjacency-legal
standard prefix (same sequence, same flips, same board) is accepted as well. Visited states are skipped, so
every candidate was searched exhaustively (at most ~1600 states; cap 20000 never reached). The scan stops at
the first m with a survivor; among survivors the one whose best shared move flips the most discs wins, then the
largest legal-set difference, then the lowest case id.

Result: **no position after 7 to 14 moves survives; after 6 moves three do (cases 138, 182, 745)**, each with a
flip-free adjacency order, black to move, 4 to 5 shared legal moves that each flip one disc; case 182 wins on
the legal-set difference (14 squares) and the lowest id. A mid-game board (8 to 14 moves) shared by (a) and (b)
only does not exist either under the same condition (0 of 1000 games at every m; round 2, reproducible with
`shared_board(hist, range(14, 7, -1), ["oth-adjacent-flip"])`). An adjacency order with flips that happens to end
on the target board is not excluded by this search, so "not reachable" for adjacent-flip means "not by a
flip-free order or by the standard sequence".

## Panel (d): the own game

Seeded rule (`--seed 0`, `--move 14`, `--min-flips 1`): the instance's 1000 bench cases in a seeded permutation,
the first whose moves 15 and 16 are regular (no pass) and, where the rule set flips, whose move 15 flips at
least one disc. The rule is applied to every instance in the order standard, adjacent-flip, adjacent-noflip,
standard-noflip (picks 459, 358, 72, 943, so the seeded stream matches rounds 1 and 2) and only standard-noflip
is drawn: case 943, board after move 14 (18 discs), black to move, the game's real move 15 = g2.

## Caption facts

- (a) to (c): one real position (a standard game after six moves, black to move) that can also arise under the
  adjacency rule sets, so only the rules differ. Yellow squares (top row): the squares black may play; the
  enclosure rule allows 8, adjacency 14. Cyan ring (bottom row): the placed disc. Pink rings: the discs
  it flips. (a) d1 flips the three white discs in row d; (b) c4 flips d4; (c) the same placement (or f5 in the
  altmove version) flips nothing under adjacent-noflip.
- (d): a standard-noflip game after 14 moves; the enclosure rule allows 7 squares, g2 flips nothing, and the discs
  already form the checkerboard (orthogonal neighbours always differ) that every game of that rule set ends in.
- No number appears inside either figure; the only text is the bold panel letters and the key's three labels.

## Notes

- PDFs are cropped to their content (composites exactly 5.5 in wide, no padding), so spacing is set in LaTeX;
  the panel letters sit at the top edge of the box.
- The shared board is early-game (10 discs) because nothing later survives the reachability test.
- (d) is at move 14 (18 discs) while (a) to (c) are at move 6; `--move 6` would re-pick (d) at the same disc
  count (a different case).
