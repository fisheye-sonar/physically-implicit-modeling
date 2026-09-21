# Othello and its three rule variants (fig:othello_and_variants)

Pieces and composites for the environment figure of the paper's Othello section: for each of the four rule
sets, a real mid-game board from the instance's own edit bench, replayed under the instance's rules, with the
mover's legal squares, one move, and the discs that move flips. Everything canonical is imported: rules from
`pim.environments.othello.corpus.rules_of`, board replay from the vendored `OthelloBoardState`
(`umpire`, `get_valid_moves`, `tentative_move`), games from `pim.environments.othello.bench.load_benchmark`,
the board drawing from `paper/figs/qualitative_edits_othello/make_figure.py::draw_board`, colours and fonts
from `paper/figs/paper_style.py`. No metric is computed anywhere in this folder.

Built 2026-09-21. Regenerate (CPU, about 10 s each):

    .pim/bin/python paper/figs/environments_overview/othello/make_figure.py        # --seed 0 --move 14 --min-flips 1
    .pim/bin/python paper/figs/environments_overview/othello/make_predictive.py    # --seed 0   (stretch, appendix)

## Files

Top level (each `.pdf` is vector, text embedded as TrueType; a 300 dpi `.png` preview sits beside it):

| file | what it is |
|---|---|
| `composite_O1.pdf` | Recommended main-text composite, 5.5 in wide: (a) standard, (b) adjacent-flip, (c) adjacent-noflip, (d) standard-noflip, one board each (option O1), legend key strip beneath. |
| `composite_O2.pdf` | Second composite, option O2: before (top) and after (bottom) boards per variant, arrow between, key strip beneath. |
| `composite_O3.pdf` | Option O3 as a composite: three consecutive positions per variant, top to bottom. Tall (5.3 in); appendix material. |
| `legend_key.pdf` | The key on its own (3.4 x 0.3 in): legal move, chosen move, flipped disc. |
| `predictive_composite.pdf` | Stretch (appendix): per variant, the board tinted by its true legal set (row "legal moves") over the same board tinted by the trained model's unedited next-move distribution (row "model"). |
| `boards.json` | Sidecar for every rule-figure piece: instance, rules, bench case id, move number, mover, legal set, chosen move, flipped squares, the O3 frames, the terminal board's source. |
| `predictive.json` | Sidecar for the stretch figure: cache column, run, instance, case id, legal-set size, tint parameters. |

`pieces/` (boards 2.0 in on the page, one PDF + PNG per element, `<variant>` in `standard`, `adjacent_flip`,
`adjacent_noflip`, `standard_noflip`):

| piece | what it shows |
|---|---|
| `<variant>_legal` | The board after move 14 with the mover's legal squares dotted (deliverable 1). |
| `<variant>_O1` | Option O1: legal dots, the chosen move as a half-transparent disc of the mover's colour with a cyan ring, the discs it would flip ringed pink (still in their pre-move colour). |
| `<variant>_before`, `<variant>_after` | Option O2's two boards: before = legal dots + the chosen move as a ghost disc; after = the board after the move, the placed disc ringed cyan, the flipped discs ringed pink. |
| `<variant>_O2_pair` | The before / after pair with an arrow. |
| `<variant>_O3_1`, `_O3_2`, `_O3_3` | Option O3's frames: the positions after moves 14, 15, 16, each with the disc just placed ringed cyan, the discs it flipped ringed pink, and the next mover's legal squares dotted. |
| `<variant>_O3_seq` | The three frames in a row with arrows. |
| `<variant>_terminal` | Extra: the final position of one whole game from the instance's generator (`synthetic_games(1, seed=0, **rules_of(inst))[0]`, i.e. the generator's game 0 under seed 0, which is game 0 of the training pool by the index law). Standard-noflip's is the full checkerboard that every game of that rule set ends in; adjacent-noflip's shows colour free of the square. |
| `predictive_<variant>_legal`, `_model`, `_pair` | The stretch figure's boards, singly and as a pair. |

## Markers (defined once in `make_figure.py`, colours from `paper_style`)

- Legal move: small yellow dot (`ps.BOARD_TINT`, #ffe600) on each square the mover may play.
- Chosen move: cyan ring (`ps.ORIGIN_C`, #00bcd4). Before the move it surrounds a half-transparent disc of the
  mover's colour (alpha 0.5); after the move it surrounds the placed disc.
- Flipped disc: pink ring (`ps.DEST_C`, #ff4fa3), the same pink the qualitative figures use for a tile whose
  colour changed. In O1 the ringed discs are shown in their pre-move colour (they would flip); in the after
  boards and O3 frames they have already flipped.
- Boards: `draw_board`'s geometry and colours (board green, black / white discs, thin grid). Rows a to h run top
  to bottom and columns 1 to 8 left to right, as in the vendored code, so square "g4" is row g, column 4.

## Data, cases, selection rule

Boards are real games from each instance's canonical edit bench (`datasets/othello/<inst>/edits/v1/cases_1000.pkl`,
1000 held-out 20-move prefixes cut from the instance's own `edits` games), replayed under the instance's rules
via `OthelloBoardState(**rules_of(inst))`. All four boards are the position after move 14 (18 discs); the chosen
move is the game's real move 15; the O3 frames are the positions after moves 14, 15, 16. The mover at move 15 is
black in all four (odd moves are black's when nobody has passed, and no case with a pass in the window is used).

Selection rule (`--seed 0`): take the instance's 1000 cases in a seeded random permutation and accept the first
whose moves 15 and 16 are regular moves (no pass) and, where the rule set flips, whose move 15 flips at least
`--min-flips` = 1 disc. The flip requirement is editorial: most adjacent-flip moves flip nothing (the paper
quotes 0.27 flipped discs per move), and a rules figure needs a move that flips. For the two noflip rule sets
no move ever flips, so the rule reduces to the first seeded case without a pass.

| panel | variant | instance | rules | bench case | move 15 | flips | legal squares at move 14 |
|---|---|---|---|---|---|---|---|
| (a) | standard | `oth-uniform` | enclosure placement, flips on | 459 | g4 | f4, f5 | 12 |
| (b) | adjacent-flip | `oth-adjacent-flip` | adjacency placement, flips on | 358 | b5 | c4 | 19 |
| (c) | adjacent-noflip | `oth-adjacent` | adjacency placement, flips off | 72 | c4 | none | 22 |
| (d) | standard-noflip | `oth-noflip` | enclosure placement, flips off | 943 | g2 | none | 7 |

O3 frames (moves 14, 15, 16) flip 1, 2, 4 discs in (a), 0, 1, 2 in (b), and none in (c) and (d); full lists
in `boards.json`.

Stretch figure (`predictive.json`): one random case per variant (`rng.integers` over the 1000 cached cases, seed
0) read from `.scratch/othello_edits_guarded_cache.pkl`, the cache the qualitative Othello figure computed at the
guarded best arms on 2026-09-19. Boards are the bench's 20-move positions; the "legal moves" row is
`pim.metrics.set_editability.uniform_over_legal(legal_pre)`, the "model" row is the cached `probs["Unedited"]`
of the run in the table below. Tint as `draw_board` draws it (full yellow at 0.02 probability mass and above,
`(p / 0.02) ** 0.6` below), the same rendering as the qualitative figure.

| panel | run | bench case | legal squares |
|---|---|---|---|
| (a) | `initial_othello_comparison/L-oth-20m` | 850 | 18 |
| (b) | `adjacent_flip_ablation/L-oth-adjacent-flip-20m` | 636 | 24 |
| (c) | `adjacency_ablation/L-oth-adjacent-20m` | 511 | 19 |
| (d) | `flip_ablation/L-oth-noflip-20m` | 269 | 8 |

## Options and recommendation

- **O1 (recommended for the main text).** One board per variant, 5.5 x 2.0 in with the key. Every rule fact is
  on one board: how many squares are playable (few under enclosure, many under adjacency), where the move goes,
  and which discs it flips (none in (c) and (d)). Cheapest in space; the reader must accept that pink-ringed
  discs are about to change colour.
- **O2.** Before / after makes the flip explicit (the ringed discs are white above and black below) and makes
  "nothing flips" explicit for (c) and (d) (the after board is the before board plus one disc). Twice the height
  (5.5 x 3.7 in with the key). Use it if the caption cannot carry the "would flip" convention.
- **O3.** Three consecutive positions show the dynamics (in (a) the third move flips a column of four) but it
  is the tallest and busiest; appendix material, or a single variant's `_O3_seq` as an inset.

## Caption facts

- Four real bench games, one per rule set, each replayed under its own rules to the position after move 14;
  black to move; the move shown is the one actually played next in that game.
- Yellow dots: the squares black may play. Cyan ring: the move (a ghost disc before it is played, the placed disc
  after). Pink rings: the discs that move flips.
- (a) g4 encloses f4 vertically (against e4) and f5 diagonally (against e6), flipping both. (b) b5 is legal only
  because it touches a black disc; it happens to enclose c4, which flips. (c) the same adjacency rule with flips
  off: many legal squares, nothing changes colour. (d) enclosure placement with flips off: few legal squares,
  nothing changes colour, and the discs already form the checkerboard (orthogonal neighbours always differ)
  that every standard-noflip game ends in (`pieces/standard_noflip_terminal.pdf`).
- Predictive figure: the model rows tint essentially the same squares as the legal-set rows on all four variants;
  any residual tint on a non-legal square is mass the model misplaces.

## Notes

- No number appears inside any figure; the only text is panel letters, the key's three labels, and the
  predictive composite's two row labels.
- `paper_style.save` writes the PDF with `bbox_inches="tight"` (0.1 in pad), so the composites' page boxes are
  5.7 in wide. Include at `width=\linewidth` (a 0.965 scale, 9 pt text prints at 8.7 pt) or trim 7.2 pt per
  side for exactly 5.5 in.
- The tint in the predictive figure saturates at 0.02, the qualitative figure's canonical setting, so a model
  that puts 1/|L| on every legal square looks identical to the legal-set board; that is the intended message.
  A different saturation would need a `tint_scale` argument passed to `draw_board`.
- The `_terminal` boards are from the generator's game 0 (a training-pool game), used only to show a finished
  position; standard-noflip's terminal board is the same for every game.
