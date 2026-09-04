# Othello editability by move number (PI, ND; canonical linear probes)

**Date** 2026-09-02 · **Run** L-oth-20m · **Code** `experiments/othello_edit_by_step/` ·
**Data** `experiments/othello_edit_by_step/scores/` (per-case arrays; see "Saved for later")
· **Figures** `experiments/othello_edit_by_step/outputs/edit_by_step_{PI,ND}.png`

## Question

How does the Edit Index vary with where we are in the game? Li's shipped benchmark only
covers prefixes of length 5–30, so the full game needed new cases.

## Method

* **Cases.** 256 per move number, built with the shipped benchmark's recipe as measured
  on its 1001 cases (all: a real game prefix + one occupied, non-centre square flipped to
  the opposite colour, the flip changing the legal-move set): held-out TEST-split games,
  a uniformly random qualifying square, flips that leave the legal set unchanged or empty
  rejected. **Move 59 cannot exist**: with one empty square left the next move is forced
  whatever the colours, so no flip changes the legal set (583,843 rejections, 0 cases) —
  the Edit Index has no support there. 14,848 cases, moves 1–58.
* **Scoring.** The master_eval Othello loop verbatim — the run's cached canonical LIN
  mine/theirs probes (cache hit `probes_37e891b4…`), ND `add_sub` (9 α × 9 points), PI
  `pinv` (7 α × 9 points), the same scorecard — kept per case and grouped by prefix
  length. Two curves per editor: the best arm *at* each move (a per-move argmax over
  63–81 arms with 256 cases; ~+0.05 optimistic) and the arm that is best over all cases
  pooled, read at each move. Fidelity ratio per case from `move_fidelity_ratio_per_case`.
  4 min on the local GPU for everything.
* **Validation.** The shipped 1001 through the identical pipeline (`li`), compared at
  moves 5–30.

## Result

| move | unedited floor | PI best (fid) | ND best (fid) | winning point |
|---|---|---|---|---|
| 1 | −0.92 | +0.16 (1.40) | +0.24 (1.22) | 0 |
| 2 | −0.88 | +0.30 (1.00) | +0.14 (1.64) | 3 |
| 3 | −0.89 | +0.72 (0.17) | +0.50 (0.28) | 4 |
| 5 | −0.83 | +0.81 (0.11) | +0.71 (0.18) | 4 |
| 10 | −0.76 | +0.67 (0.19) | +0.61 (0.24) | 4 |
| 20 | −0.66 | +0.55 (0.32) | +0.53 (0.31) | 4 |
| 30 | −0.60 | +0.47 (0.39) | +0.42 (0.41) | 4 |
| 40 | −0.59 | +0.36 (0.56) | +0.41 (0.44) | 4 |
| 50 | −0.61 | +0.38 (0.53) | +0.40 (0.43) | 4 |
| 58 | −0.80 | +0.49 (0.55) | +0.48 (0.39) | 4–5 |

* **Editability is a function of game phase.** Both editors land from move 3 on and peak
  at moves 4–8 (PI +0.78 to +0.81, fidelity ≈ 0.1), then decline steadily to ≈ +0.35 by
  moves 40–55 (fidelity ≈ 0.5), with a small rise at moves 57–58. Pooled over the whole
  game the best arm is PI pt4·α3 (+0.44) and ND pt4·α0.5 (+0.44); the shipped benchmark's
  +0.61/+0.62 is that same arm read over moves 5–30 only.
* **The first two moves are not editable**: fidelity ≥ 1 with a small positive index —
  the write disrupts rather than steers. At moves 1–2 the board has 5–6 discs, the flipped
  square is one of at most two non-centre discs, and the legal set is tiny.
* **The winning point is residual point 4 almost everywhere** (5 occasionally; 0 and 3 at
  moves 1–2), matching Table 2's pooled choice — the layer at which the board is edited
  does not move with game phase, even though the layer at which it is best *decoded* does
  (decode-by-step: point 0–1 early, 7–8 late).
* **Decodability and editability move in opposite directions across the game** where both
  are measured: the board is read out perfectly through move 30 and still at skill 0.95
  at move 54, while the index falls from +0.8 to +0.35 over the same span. Late-game
  edits fail for a reason other than the probe losing the board.

## Validation against the shipped benchmark

At moves 5–30 the synthesised cases score lower: PI Δ = −0.034 (sd 0.041 over 26 moves),
ND Δ = −0.081 (sd 0.048); the unedited floors agree. Stratifying by the size of the
legal-set change the flip causes, the two sets agree closely (PI pt4·α3, moves 5–30):

| |Δ legal set| | synth EI (n) | shipped EI (n) |
|---|---|---|
| 1 square | +0.49 (2670) | +0.47 (260) |
| 2 squares | +0.62 (2423) | +0.63 (429) |
| 3–4 squares | +0.68 (1534) | +0.69 (311) |

The gap is case mix: the shipped cases have more consequential flips (74 % change ≥ 2
legal squares vs 60 % here) and fewer edge squares (3.5 % vs 8.2 %), i.e. Li's square
choice was not uniform over occupied squares. The uniform recipe is kept, documented; the
by-move shape is the same under either mix.

## Saved for later (the MLP replication needs nothing recomputed)

`scores/synth_cases.{npz,json}` and `li_cases.*`: every case's move number, square,
current/target label, legal sets, unsteered probabilities and unedited per-case index.
`scores/<set>_arms_<editor>.{npz,json}`: every arm's per-case Edit Index, fidelity and
Li error with the arm list and the probe-grid file name. The cases themselves:
`cases/synth_seed0_n256.{pkl,json}`. `EDITORS="GS" drivers/edit_by_step.sh` runs the
GS arms on the cached MLP-128 probes against the same cases and baseline, adding one arm
file per set; `plot_by_step.py` then draws the GS figure beside these.

## Canonical changes

`bench.benchmark_from_cases` (the body of `load_benchmark`, callable on any case list;
`case_targets` now served from the Benchmark), `othello_moves.move_fidelity_ratio_per_case`
(the guard before its mean). Behaviour-preserving, pinned by `tests/test_othello_bench.py`.
