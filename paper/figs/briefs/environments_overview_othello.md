# BRIEF — Othello and its three variants (environment figure pieces)

Read `paper/figs/FIGURE_BRIEF_COMMON.md` first. CPU only. Folder: `paper/figs/environments_overview/othello/`.

## What the figure must convey (paper §Experimental Setup → Othello; caption already drafted in
`paper/paper_draft.tex`, figure `fig:othello_and_variants`)
(a) **Standard Othello**: players alternate placing discs; a legal move must enclose opponent discs along a line
(including diagonals) and the enclosed discs flip colour. The model predicts the distribution over next legal moves.
(b) **adjacent-flip**: placement only needs to be adjacent (incl. diagonally) to a disc of your own colour; flips kept
on (rare and incidental). (c) **adjacent-noflip**: adjacency placement, flips off — placed discs never change colour.
(d) **standard-noflip**: enclosure placement, flips off — every game plays out the same checkerboard pattern in a
different order, so colour is fixed by the square.

## Canonical objects (import, never re-implement)
- Rules per instance: `pim.environments.othello.corpus.rules_of(inst)` for `inst` in `oth-uniform` (standard),
  `oth-adjacent-flip`, `oth-adjacent`, `oth-noflip` (dirs under `datasets/othello/`). Board replay:
  `pim.environments.othello.vendor.othello.OthelloBoardState(**rules_of(inst))` — `umpire(move)`,
  `get_valid_moves()`, `state` (8×8, −1 white / 0 blank / +1 black; check the sign convention in the file),
  `tentative_move(move)` (what a move would flip; read the vendor code to see what it returns). Real games under each
  rule set: `pim.environments.othello.bench.load_benchmark(inst).tokens` (1000 real 20-move prefixes; token→square via
  `{v: k for k, v in pim.environments.othello.data.canonical_vocab().items()}` exactly as
  `paper/figs/qualitative_edits_othello/make_figure.py::compute` does), or replay via `data.tokens_and_labels`.
- Drawing: `draw_board` in `paper/figs/qualitative_edits_othello/make_figure.py` (import it; the module imports torch
  but runs nothing at import). Use its geometry and colours; you may add a marker style for "legal move" and
  "would flip" (e.g. a small dot for legal squares, a ring / arrow for flipped discs) — keep the markers in
  `paper_style` colours and define them once.

## Deliver
1. **Pieces** (each its own PDF + PNG, board at least 2.0 in on the page, vector): for each of the four rule
   sets, a real mid-game board (same move number for all four, e.g. after move 12–16 of a bench game; the noflip
   boards should show their fixed checkerboard) with the current player's legal moves marked. For the two
   flip rule sets also a "before → after" pair for one chosen legal move showing the discs it flips.
2. **Options** for how the rule is conveyed (produce all three, Sevan chooses):
   - O1 single board per variant: legal squares marked; the chosen move marked; discs it would flip ringed.
   - O2 before / after pair per variant (two boards, an arrow between them; flipped discs ringed after).
   - O3 a 3-frame mini sequence per variant (three consecutive positions of one game).
3. **A legend key** as a separate small PDF (legal move marker, chosen move, flipped disc).
4. **Composite attempt**: one full-width figure (5.5 in), the four variants in a row (a)–(d) with panel letters
   only, using the option you recommend; and a second composite using O2.
5. `README.md` + sidecar JSON (instance, bench case id, move number, chosen move) per board.

## Stretch (only if the above is done and verified)
Predictive-quality illustration for the appendix (`paper/figs/environments_overview/othello/predictive_*`):
a board with the true legal set beside the same board tinted by the trained model's unedited move distribution,
one per variant, from the cached `Unedited` probabilities in `<repo>/.scratch/othello_edits_guarded_cache.pkl`
(structure: see `compute()` in the qualitative Othello script — `board_pre`, `legal_pre`, `probs["Unedited"]`).
No model needed. Random case per variant, seed 0, ids in the JSON.
