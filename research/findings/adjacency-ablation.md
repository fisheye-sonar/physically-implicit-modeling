# Adjacency ablation — colour is USED and still not editable (2026-09-09)

**Question.** oth-noflip was not editable, but the checkerboard theorem made that vacuous:
colour never entered its dynamics. oth-adjacent keeps "no recolouring" and changes the
placement rule to "touch one of your own discs (8-neighbourhood)", so legality depends on
colour without any enclosure geometry. Prediction (both of us): editability returns.

**Answer: it does not.** `runs/adjacency_ablation/L-oth-adjacent-20m` (Transformer-L, 20M
games, 780k steps, matched recipe; 19.2 h) is Bayes-optimal (CE 2.435 vs floor 2.433, legal
mass 1.000, top-1 legal 1.000) — it knows the legal set, which provably depends on colour —
and every editor is inert or destructive. The read-out LANDS (the probe reads the target
class at the tile in 100% of cases from point 1 on) and the model's legal-move distribution
does not move.

## Results (`scores.json`; extended α in `experiments/adjacency_ablation/scores/extended_alpha.json`)

| run | val CE (Bayes) | skill LIN / MLP | unedited | PI | ND | GS |
|---|---|---|---|---|---|---|
| L-oth-20m (standard) | 2.029 (2.011) | 0.975 / 0.976 | −0.71 | +0.61 / fid 0.24 | +0.62 / 0.23 | +0.65 / 0.21 |
| L-oth-noflip-20m | 1.680 (1.679) | 1.000 / 1.000 | −0.82 | −0.00 / 3.25 | +0.09 / 2.57 | +0.02 / 4.00 |
| **L-oth-adjacent-20m** | 2.435 (2.433) | 0.988 / 0.990 | −0.68 | −0.05 / 2.31 | +0.12 / 1.05 | +0.00 / 5.73 |

Extended sweep (ND α to 12, PI α to 35, every point): ND's best is +0.13 at fidelity 1.9
(pt 0, α 12) and +0.12 at fidelity 1.05 (pt 1, α 2); PI never exceeds +0.01 — at low α the
index sits at the unedited value with fidelity 1.0, at high α fidelity climbs to 3–6 with the
index still ≈ 0. Read-out landing is 0.99–1.00 at every point ≥ 2 for both editors: the
probe's own reading of the edited residual shows the flipped colour, and nothing downstream
changes. Pilot (20k games): 68% of random recolourings change the legal set, so the bench is
not the problem (1001 synthesised cases, each changing legality).

**Decodability is an input lookup.** Trained LIN 0.988 = right-aligned observation floor
0.988 (random-init 0.58). In any no-flip game a disc's colour is the parity of the move that
placed it (up to the 1.1 passes/game), so the probes read a feature the input already
carries; training adds nothing to it. In standard Othello the observation floor is well below
the trained model (colour must be computed through the flips).

## Reading

1. **Causal use is not sufficient for editability.** Legality here depends on colour and the
   model predicts legality perfectly, yet writing colour through the probes does nothing. The
   flip ablation's negative therefore did NOT hinge on the checkerboard theorem.
2. **The probe finds a shadow, not the load-bearing copy.** The linearly decodable colour is
   the placement-parity lookup; the legality computation can be done directly from the token
   sequence ("empty squares next to discs placed by moves of my parity") without ever
   consulting that copy. The write lands on the copy the probe reads and misses the copy the
   output reads. Standard Othello is the case where the decodable board IS the computed
   intermediate the output depends on — the flips force the model to construct it.
3. **Across the programme** the one property that has tracked editability is not causal use,
   not decodability, not the target type, not carried state, but whether the decodable
   representation is one the model had to COMPUTE from the dynamics and then read back
   (standard Othello: yes; every discworld instance, both no-flip variants: the state is a
   lookup on the input or, in blink, carried but still not the copy the renderer reads).
   The no-flip presence edits (+0.39) are the one datum that does not fit cleanly and must be
   kept in view.

Caveats: 1001 synthesised cases; GS only at the canonical grid (its best is at fidelity 5.7,
pure destruction, so a wider grid cannot rescue it); "landed" is the linear probe's own
reading, the MLP was not checked.

Assets: `experiments/adjacency_ablation/` (pilot gate, extended sweep); `scripts/drivers/oth_adjacent.sh`;
`logs/adjacency_ablation/L-oth-adjacent-20m/`; `datasets/othello/oth-adjacent/instance.json`.
