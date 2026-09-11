# Adjacent-flip ablation — colour USED and REWRITTEN: editability returns, at a third of Othello's level (2026-09-11)

## Current understanding

**Status: `observed`** (one seed, one configuration, 1001 synthesised cases; canonical grid only).

oth-adjacent-flip keeps oth-adjacent's placement rule (a move must touch one of the mover's
own discs) and turns recolouring back on (the placed disc flips the discs it encloses, as in
Othello). Colour is therefore both **used** by legality and **rewritten** by the dynamics, so
it is no longer a parity lookup on the input — the cell oth-adjacent could not reach.

**Result: editability comes back, partially.** `runs/adjacent_flip_ablation/L-oth-adjacent-flip-20m`
(Transformer-L tokens, 20M games, 780k steps, matched recipe; 25.7 h on the WSL remote's
RTX 4090) is Bayes-optimal (CE 2.301 vs floor 2.295, legal mass 0.998, top-1 legal 1.000).
The probes now read something training COMPUTED: trained skill 0.947 / 0.970 (LIN / MLP)
against a right-aligned observation floor of 0.888 / 0.914 (oth-adjacent: trained 0.988 =
floor 0.988). And the edits move the output in the right direction with the world intact:
**ND +0.24 at fidelity 0.62, PI +0.17 at fidelity 0.85** (unedited −0.70) — clearly above
oth-adjacent (+0.12 / 1.05 and −0.05 / 2.31) and oth-noflip (≈ 0, guards 2.6–4), clearly
below standard Othello (+0.61 to +0.65, guards 0.21–0.24). GS stays inert and destructive
(+0.03 / 2.62), as on both no-flip variants.

## Results (`scores.json`, EVAL_VERSION 2026-09-01.4; floors in `runs/_baselines/<instance>/baselines.json`)

| run | CE (Bayes) | skill LIN / MLP | observation floor LIN / MLP | random-init LIN / MLP | unedited | PI EI / fid | ND EI / fid | GS EI / fid |
|---|---|---|---|---|---|---|---|---|
| L-oth-20m | 2.029 (2.011) | 0.975 / 0.976 | 0.792 / 0.809 | 0.58 / 0.58 | -0.71 | +0.61 / 0.24 | +0.62 / 0.23 | +0.65 / 0.21 |
| **L-oth-noflip-20m** | 1.680 (1.679) | 1.000 / 1.000 | 1.000 / 1.000 | 1.00 / 1.00 | -0.82 | -0.00 / 3.25 | +0.09 / 2.57 | +0.02 / 4.00 |
| L-oth-adjacent-20m | 2.435 (2.433) | 0.988 / 0.990 | 0.988 / 0.983 | 0.58 / 0.59 | -0.68 | -0.05 / 2.31 | +0.12 / 1.05 | +0.00 / 5.73 |
| **L-oth-adjacent-flip-20m** | 2.301 (2.295) | 0.947 / 0.970 | 0.888 / 0.914 | 0.67 / 0.70 | -0.70 | +0.17 / 0.85 | +0.24 / 0.62 | +0.03 / 2.62 |

Signed regression target `mine_signed` on the new run: skill 0.704 / 0.838 (observation floor 0.631 / 0.720, random-init 0.13 / 0.20); editors PI +0.17 / fid 0.59, ND +0.26 / fid 0.63, GS +0.01 / fid 3.44.

Training: best val CE 2.301237 at step 770k of 780k; the run sat within 0.01 of the Bayes
floor from step 100k on (2.3085 at 100k, 2.3012 at 770k). Pilot (20k games, `experiments/
adjacent_flip_ablation/scores/pilot_adjacent_flip.json`): 70.3% of random recolourings change
the legal set (mean differing set 2.7 squares), colour == square parity 0.497, 14,986/20,000
distinct terminal boards, 16.1 flips per game (0.27 per move), 3.9 passes per game, every game
60 moves, one side wiped out in 0.01% of games. The 1001 cases are Li's recipe on the held-out
test split (every prefix-length quota met, `datasets/othello/oth-adjacent-flip/edits/v1/`).

## Reading

1. **The "computed, not looked up" hypothesis survives, in direction.** Across the programme
   the property tracking editability has been whether the decodable state is one the model
   had to compute from the dynamics (`adjacency-ablation.md` §3). Turning recolouring on is
   exactly the manipulation that makes colour computed (trained skill rises 0.06 above the
   observation floor, where oth-adjacent sat on it) — and editability rises from ≈ 0 to
   +0.17 / +0.24 with the guards below 1 for the first time on any Othello variant other
   than the standard game.
2. **But the magnitude is a third of Othello's, so "computed" is not the whole story.** Two
   candidates, not yet separated: (a) *leverage* — under adjacency a single recoloured tile
   changes legality only in its 8-neighbourhood (mean differing set 2.7 squares of ~15
   legal moves), whereas under the enclosure rule one tile's colour enters the line scans of
   many candidate moves, so a unit colour edit has less output to move here by construction;
   (b) *how much of the colour computation the probe's copy carries* — with 0.27 flips per
   move most discs keep their placement-parity colour, so a parity lookup still explains most
   of the label and the probe may still read mostly that lookup, with only the flipped
   minority actually computed. (b) predicts editability concentrated on cases whose target
   tile was actually flipped in the game; (a) predicts it scales with the differing-set size.
   Both are readable from the existing arms + cases without new training.
3. GS's failure on every non-standard Othello instance (best arm at fidelity 2.6–5.7) is now
   three-for-three; the gradient editor needs the enclosure-game output surface, or a wider
   grid — untested.

**What would move the status:** a second seed (replicated); the two case-level splits in
(2); an extended-α sweep and read-out-landing measurement as in
`experiments/adjacency_ablation/scripts/extended_alpha.py` (landing was NOT measured here).

Caveats: single seed; 1001 synthesised cases; canonical α grid only; landing not measured;
trained on the WSL remote after its 2026-09-09 microcode fix (hardware-error count flat at
40 throughout; pilot numbers byte-identical to the lab box's).

Assets: `experiments/adjacent_flip_ablation/` (pilot gate, driver, README); run + probes
`runs/adjacent_flip_ablation/L-oth-adjacent-flip-20m/`; floors `runs/_baselines/oth-adjacent-flip/`;
instance `datasets/othello/oth-adjacent-flip/` (`instance.json`); logs
`logs/adjacent_flip_ablation/L-oth-adjacent-flip-20m/` (on wsl-sevan; not yet copied).

## Log

- **2026-09-11** — `observed`. First and only run scored (chain 2026-09-10 00:15 → 2026-09-11
  04:22 PT on wsl-sevan, unit `oth_adjacent_flip`). Numbers above.
