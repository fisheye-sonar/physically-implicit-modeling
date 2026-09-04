# The decoded Othello board is the full board, minus part of the discs nothing can read

**Date** 2026-09-03 (v2, same day: relevance criterion corrected — see "The criterion") ·
**Run** L-oth-20m, cached canonical LIN and MLP-128 mine/theirs probes, every residual
point · **Code** `experiments/othello_causal_state/` · **Data**
`experiments/othello_causal_state/scores/` (v1 single-flip artefacts under
`scores/v1_single_flip/`, `outputs/v1_single_flip/`) · **Figures**
`outputs/relevance_by_move_pt{7,8}.png`, `outputs/qual_*.png`

## Question

Does the probe read discs the game no longer depends on as accurately and as confidently
as discs it does depend on? A next-token model that re-reads the history at every position
has no need to represent a disc that no legality computation touches.

## The criterion

Legality and flips at an empty square depend only on the discs along the gap-free runs
from that square in the 8 directions. Two colour-blind notions follow from the rules
(vendored `OthelloBoardState`, nothing assumed):

* **dead** — the disc lies on no ray (row, column, diagonals, any distance) from any
  empty square. Empties only shrink and rays are fixed, so a dead disc can never
  influence any future legal set or flip. Sufficient, not necessary: conservative.
* **unread now** — the disc lies on no *gap-free* run from any empty square, so no
  legality or flip computation at this position can traverse it, whatever the colours.

**These coincide.** If a disc is on a ray from some empty square, the *nearest* empty
square to it along that line has a gap-free run to it (everything in between is occupied
by construction). So a disc is unread now iff it is dead: there is no colour-blind
"irrelevant now, relevant later" class. Verified on all 10 M occupied squares (0 unread
discs that are not dead).

The v1 write-up used a third flag, **flip-irrelevant** — the legal set is unchanged when
the disc alone is flipped — as "irrelevant now". Sevan's objection (2026-09-03) is correct
and the flag is retired as a relevance criterion: it is blind to redundancy (two discs
that each bracket the same move both flip-out as "irrelevant", though the model must
track at least one, and which one is not identifiable) and it conflates "not read" with
"read but colour-inert" (an opponent disc inside a bracketed run). Every non-dead disc is
traversed by some legality computation at every position, so flip-irrelevant discs are
all *read*; 89 % of them are non-dead ("read but inert"), the rest are dead. The flag is
kept only to split the read discs into **alive** (a flip changes the legal set) and **read
inert** (it does not), which is descriptive, not causal.

Categories (occupied squares only; empties are trivially decodable from the history):
alive frontier (touches an empty) · alive interior · read inert · dead.

## Data

5,000 TEST-split games (held out from training and from the probe games), 294,743
positions, 10.0 M occupied squares; 6.2 % dead, 50 % read-inert. Dead discs first appear
at move 33 and dominate only from move ~50 (28 of 62 discs at move 58). Nothing fitted.

## Result

**Dead discs are decoded worse; every disc a legality computation can reach is decoded
equally well, whether or not its colour currently matters.** Paired within the same
position (game phase held fixed), argmax accuracy at the canonical best points (7 LIN /
8 MLP), 30,471 positions holding both classes:

| comparison (same board) | LIN Δ accuracy | MLP-128 Δ accuracy | Δ entropy (nats) |
|---|---|---|---|
| dead − alive interior | −0.017 [−0.019, −0.015] | −0.017 [−0.019, −0.015] | +0.039 / +0.034 |
| dead − read inert | −0.013 [−0.015, −0.013] | −0.013 [−0.015, −0.012] | +0.021 / +0.021 |
| read inert − alive interior | −0.0015 [−0.002, −0.001] | −0.0014 [−0.002, −0.001] | +0.016 / +0.007 |
| alive interior − alive frontier | −0.007 | −0.006 | +0.016 / +0.013 |

Over moves 41–56 (point 7): accuracy alive-frontier 0.988 · alive-interior 0.978 ·
read-inert 0.977 · **dead 0.963**; mean entropy 0.040 · 0.063 · 0.079 · **0.098** (LIN;
MLP 0.028 · 0.045 · 0.057 · 0.073). Small in absolute terms — three quarters of the
residual error on dead squares — and robust: the bootstrap CIs exclude zero at every
point from 1 to 8, and the gap survives the controls: within every stratum of "moves
since the disc last changed colour" and of "number of flips", dead discs are 2–4 points
below alive interior discs (never flipped: 0.956 vs 0.984; flipped ≥ 3 times: 0.922 vs
0.963; LIN, point 7, moves 41–58). Not disc age, not flip history.

**The deficit is a layer story.** Paired dead − alive-interior accuracy by residual point:
+0.03 (point 0, the embedding) → −0.065 (1) → −0.066 (2) → −0.046 (3) → −0.032 (4) →
−0.022 (5) → −0.019 (6) → −0.017 (7, 8). The board is assembled through the stack and the
dead discs are assembled last and least; the gap narrows but never closes.

## Reading

The residual stream carries essentially the whole board. The one systematic gap follows
the geometry of legality exactly: a disc drops (partially) out of the model's board when
no empty square has it on a line — the one configuration in which no legality or flip
computation would ever traverse it. Discs that are traversed are decoded equally well
whether their colour currently changes the legal set or not, which is what a mechanism
that maintains board state through line scans anchored on empty squares would produce
(a hypothesis consistent with, not tested by, this experiment). The strong
"task-relevant board" reading — only the discs the next move needs — is ruled out.

## Qualitative

`outputs/qual_natural_{1,2,3}.png`: the held-out positions with the most dead squares at
move ≤ 54 (13–14 each). The effect is statistical, not a blind spot: both probes read
every dead square correctly in two of the three boards, with somewhat higher entropy on
the dead cluster. `outputs/qual_engineered.png`: a legal game played by "always take the
legal square nearest the bottom-left corner" — 55 black discs to 3 white at move 54, a
board the model never sees under uniform play; the LIN probe inverts most colours and the
MLP's entropy is 0.5–0.9 everywhere. Off-distribution failure, not relevance; kept as a
caution against engineered positions for this question.

## Saved

`scores/relevance_test5000_v2.npz` (all flags incl. `traversable`, and the controls),
`scores/probe_by_relevance_L-oth-20m.json`, `scores/per_square_pt{7,8}_{linear,mlp}.npz`
(per-square correctness, entropy, p(true)), `scores/qualitative_examples.json`. No
canonical code changed.
