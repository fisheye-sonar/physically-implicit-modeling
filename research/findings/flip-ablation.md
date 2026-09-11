# Flip ablation: Othello without the flipping rule (L-oth-noflip-20m)

**Date** 2026-09-07 · **Instance** `oth-noflip` (`datasets/othello/oth-noflip/`: `oth-uniform`
with ONE rule changed — a placed disc never recolours the discs it encloses; legality by
enclosure, passes, game end, uniform-over-legal moves and the 20M-game index law are
unchanged) · **Run** `runs/flip_ablation/L-oth-noflip-20m` (Transformer-L tokens, 8 × 512,
25.3M params, 780k steps, the matched recipe — identical to `L-oth-20m` except the instance)
· **Driver** `scripts/drivers/oth_noflip.sh` · **Logs** `logs/flip_ablation/L-oth-noflip-20m/`
· **Chain** corpus 3 h 0 min (1.87k games/s), cases + labels 1.5 min, train 19 h 21 min,
score + tables 24 min · **Experiment** `experiments/flip_ablation/` (pilot, control bench).

## Question

Othello's editability (PI +0.61 / ND +0.62 / GS +0.65) survives a change of objective
(`othello-mse-head.md`); discworld's non-editability survives a change of interface
(`interface-ablation.md`). Is it the FLIP DYNAMICS — the board being a non-trivial function
of the move history — that gives Othello a state representation an editor can write to? Remove
only the flipping rule and keep everything else.

**What the rule change does to the world** (pilot, 5k games per rule set,
`experiments/flip_ablation/scores/pilot_noflip.json`): with no flips both colours keep discs
everywhere, so a capturing move always exists and NOBODY EVER PASSES (0 passes in 300k
positions; 0.44/game with flips). Every game runs 60 moves. Hence a disc's colour is exactly
the parity of the offset at which its square was played — mine/theirs equals that parity rule
in 100% of positions (64% with flips). The board is a syntactic function of the token list.
Branching is narrower: mean legal set 5.8 (8.5), Bayes CE floor 1.672 (1.999).

**Bench.** Li's shipped 1001 cases are flip-Othello positions and cannot be used; the instance's
1001 cases are synthesised from its test split with Li's recipe and prefix-length mix
(`scripts/make_othello_edits.py`). Control: `L-oth-20m` re-read on a same-recipe synthesised
FLIP bench gives PI +0.60 / ND +0.54 / GS +0.61 (shipped: +0.61 / +0.62 / +0.65), unedited
−0.70 vs −0.71 — bench construction is not a confound (`experiments/flip_ablation/scores/summary_control.md`).

## Numbers (canonical scoring, EVAL_VERSION 2026-09-01.4; baselines b4)

| | no-flip · L-oth-noflip-20m | flip · L-oth-20m |
|---|---|---|
| val CE (best) · Bayes floor · excess | 1.6801 @ 745k · 1.679 · **+0.0015** | 2.0280 @ 775k · 2.011 · +0.018 |
| legal mass · top-1 legal (10k test games) | 0.9998 · 1.0000 | 0.9925 · 0.9980 |
| Probe Skill mine LIN / MLP-128 (best point) | **1.000 / 1.000** (points 1–8) | 0.975 / 0.976 (pt 7–8) |
| random-init floor LIN / MLP | 1.000 / 1.000 | 0.577 / 0.583 |
| observation floor, right-aligned one-hot history, LIN / MLP (170k games) | 1.000 / 1.000 | 0.792 / 0.809 |
| observation floor, left-aligned (canonical layout), LIN / MLP | 1.000 / 0.999 | 0.530 / 0.737 |
| unedited Edit Index | −0.823 | −0.713 |
| **PI** best arm · EI / fidelity | pt8·α5 · **−0.001 / 3.25** | pt4·α3 · +0.608 / 0.24 |
| **ND** best arm · EI / fidelity | pt2·α0.2 · **+0.086 / 2.57** | pt4·α0.35 · +0.622 / 0.23 |
| **GS** best arm · EI / fidelity | pt2·α0.7 · **+0.020 / 4.00** | pt0·α0.05 · +0.647 / 0.21 |
| legal mass after the best edit PI / ND / GS | 0.68 / 0.13 / 0.33 | 0.99 / 0.99 / 1.00 |
| li_error vs post at the best edit PI / ND / GS (null 2.7–3.4) | 5.7 / 14.1 / 11.5 | 0.11 / 0.10 / 0.05 |

Per-point Probe Skill (LIN): no-flip 0.718 then 1.000 at every point 1–8; flip 0.342 0.808
0.881 0.928 0.953 0.967 0.974 0.975 0.975.

## The checkerboard theorem (found 2026-09-07, after scoring — it changes the reading)

**Every no-flip game is the same game up to move order.** The four starting discs form a
2 × 2 checkerboard, and the enclosure rule preserves it: along a row or column, consecutive
squares alternate the parity of r + c, so a run of ≥ 2 opponent discs would need two
opposite-parity squares of one colour — impossible if colour has so far followed parity — and
along a diagonal every square has the SAME parity, so no diagonal line can ever hold an
opponent disc followed by an own disc. By induction, every legal placement lands the mover's
colour on its own parity class. Verified on 60,000 positions of 1,000 test games: colour ==
square parity in every position, 0 diagonal sandwiches, 0 runs longer than one disc, and all
3,000 test games end in the identical full checkerboard (flip Othello: 3,000 distinct finals).

Consequently **legality never depends on colour**: the true legal set equals "the empty
square is in the mover's parity class and two consecutive squares in some row/column
direction are occupied" in 60,000 / 60,000 positions. The world's only state is the SET of
occupied squares (plus whose turn it is); colours are a fixed pattern painted on it.

## What it says

1. **Editability disappears — but the ablation removed more than the flips.** No arm of any
   editor moves the output toward the edited board; the best indices (−0.001, +0.086, +0.020)
   come only from writes that destroy it (legal mass 0.13–0.68, li_error 6–14 vs a null of
   3.4, fidelity 2.6–4.0), and every smaller write sits at the unedited floor. Read with the
   theorem, this is expected by construction: the edit flips a tile's COLOUR, and in this
   world the next move does not depend on colour. A model that learned the occupancy rule has
   no colour-dependence to steer; the benchmark's post-edit legal set (computed by the colour-
   aware rule engine on a board that violates the parity invariant) is a target the world's
   dynamics never produce. Non-editability here is a fact about the WORLD, not a clean
   statement about the model's representation.
2. **Decodability at 1.000 is occupancy times a fixed pattern.** mine/theirs of an occupied
   square is its parity class relative to the mover; the random-init model (1.000) and a
   linear read of the right-aligned one-hot history (1.000) read it just as well. The model
   is Bayes-optimal (CE excess 0.0015, legal mass 0.9998) and learns the rule almost
   immediately (96% of the random-init excess gone by step 1k, `GOTCHAS 2026-09-07`).
3. **What the run does establish.** (a) A world where the probed state variable is
   perfectly decodable, at every point, from random init onward, and causally irrelevant to
   the output — the cleanest instance yet of "decodable ≠ used", and a control that every
   editability claim should be read against. (b) The scoring pipeline behaves correctly
   under it: PI/ND/GS return the unedited floor until they destroy the output, and the guard
   says so. (c) Removing flips from Othello with the standard opening does NOT yield a
   "no-flip Othello with a colour state"; it yields a colour-free occupancy game.
4. **What it does not establish** is the thing it was built for: whether the flip DYNAMICS
   (colour as a non-trivial function of history) are what gives Othello an editable state.
   That needs a no-flip world in which colour still matters for legality. Two candidates,
   both one new instance each (corpus ~3 h, train ~20 h): (i) keep the enclosure rule but
   break the invariant at the start — a randomised, non-checkerboard opening (e.g. the
   first k moves played under flip rules, or random initial discs), after which colours
   carry information the mover cannot read off the square, sandwiches run long and diagonal,
   and legality depends on colour; (ii) a different legality rule (place adjacent to any own
   disc), where colour matters but nothing flips. Either needs the pilot to CHECK
   colour-dependence (legal set ≠ occupancy rule) before generating 20M games — the check
   this run should have had.
5. The floors table now carries both history layouts (b4): on flip Othello the right-aligned
   linear observation floor is 0.79 vs the model's 0.975, so the flip model does exceed a
   linear read of its input by a wide margin; the no-flip and tokenised-discworld models do not.


## Presence is editable on the same model (2026-09-07, `experiments/flip_ablation/`)

The user's test: edit PRESENCE (occupied vs empty), the variable legality actually depends
on, with a dedicated 2-class linear probe fitted on the trained no-flip model (error 0.001%),
400 cases (remove a disc / add a parity-consistent disc), PI at points 2, 4, 6:

| model | best PI presence arm · EI / fid | remove / add | colour edits, same model |
|---|---|---|---|
| no-flip, trained | pt2·α3 · **+0.387 / 0.44** (li_post 0.48 vs 1.96 unedited) | +0.15 / **+0.64** | −0.001 / 3.25 |
| flip, trained | pt4·α3 · +0.447 / 0.37 | +0.60 / +0.28 | +0.608 / 0.24 |
| no-flip, random init | +0.001 / 1.00 | | |

The probe read-out lands at every α ≥ 1 on all three models; only the trained outputs follow.
So within ONE model two variables that are both linearly decodable at 1.000 — from random
init onward and from a shallow read of the input — separate cleanly by causal role: colour
(never consulted by the dynamics) cannot be edited, presence (what legality depends on) can,
at the flip model's level. This overturns the "computed rather than read" reading of §3
above: decodability from the input does not preclude editability. The operative condition is
whether the output computation consumes the probed variable. Discworld — position causally
necessary, decodable, not editable — is the case this leaves open, and the natural
hypothesis is that its next-frame computation consumes ray-level features directly, so the
linear position read-out is a spectator projection there in a way presence is not here.

**2026-09-11 addendum (`adjacent-flip-ablation.md`, presence section).** Re-run rule-aware on the two
adjacency instances: presence is NOT editable on oth-adjacent (best guarded −0.12; +0.015 only at
fidelity 5) or oth-adjacent-flip (+0.05 at fidelity 0.74; removal +0.19–0.25, add ≈ 0), while the
standard-Othello row reproduces (+0.447 / 0.37). Adjacency legality consumes presence directly, so
"the output consumes the probed variable" is not sufficient for editability; the reading above is
superseded — see `adjacent-flip-ablation.md`.

## Caveats

- The main caveat is the theorem above: the ablation confounds "no flips" with "colour is
  causally irrelevant". The pilot reported mine/theirs == parity-of-move-offset (true) without
  noticing that it also equals parity-of-SQUARE — i.e. that every game is the same board.
- One seed; one instance. The synthesised bench is the only bench available (control above).
- The α grids are the flip model's; nothing lands at any grid point, and the fidelity
  trajectory says larger α would only destroy more.
- GS's grid is edge-pinned at α 0.7 here (best) — in the destroyed regime, so extending it is
  pointless; PI's best at α 5 likewise.
