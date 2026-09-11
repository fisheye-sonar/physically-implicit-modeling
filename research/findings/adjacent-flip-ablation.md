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

⚠ **Read the Edit Index against its ceiling (added later on 2026-09-11).** On the clean
exact-counterfactual cases the model run on the true counterfactual world scores only +0.14
here (standard Othello +0.69): adjacency legality changes ~2 of 16 legal moves per
recolouring, so the union index has little dynamic range on this instance. On those same
cases ND scores +0.34 and PI +0.17 with 98–99% of their mass on the post-edit legal set —
at or above what the real counterfactual achieves. The "third of Othello" gap is therefore
mostly metric range, not a third of the effect; see the alignment section below.

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

## Alignment with the true edit direction, Haufe correction, and the honest ceiling (2026-09-11)

Same method as `edit-direction-alignment.md` (`experiments/edit_direction_alignment/scripts/othello_alignment.py --run …`):
for each bench case a real history of the same length and mover whose board is EXACTLY the
edited board (search over move substitutions / swaps), kept only if the model treats it as an
ordinary game (legal mass ≥ 0.99); Δ = h_cf − h at the last position; the probe subspace is
the 3 rows of the flipped tile; generic = the displacement to an unrelated clean case. Extras
(`experiments/adjacent_flip_ablation/scripts/alignment_extras.py`, `honesty_check.py`,
`haufe_edit_adjacent_flip.py`): the single ND write direction's cos² with Δ, raw vs Haufe vs a
random direction; the split by whether the flipped tile was actually recoloured in the game;
the true counterfactual's own Edit Index (the ceiling) against the canonical best arms on the
SAME cases; Haufe-corrected PI/ND edits on the full bench.

**Alignment at each model's best point** (fraction of Δ in the probe subspace; ND-direction cos²):

| run | n clean | pt | rows (raw) | generic | ×gen | Haufe | genH | ×genH | ND-dir cos² raw | Haufe | random | ‖Δ‖/‖h‖ | canonical PI / ND |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L-oth-20m (standard) | 18 | 5 | 0.188 | 0.007 | 25.7× | 0.238 | 0.009 | 27.7× | 0.181 | 0.256 | 0.0015 | 0.38 | +0.61 / +0.62 |
| L-oth-adjacent-20m | 52 | 1 | 0.077 | 0.009 | 8.4× | 0.236 | 0.022 | 10.9× | 0.032 | 0.134 | 0.0016 | 0.39 | −0.05 / +0.12 |
| **L-oth-adjacent-flip-20m** | 42 | 1 | 0.051 | 0.012 | 4.4× | 0.094 | 0.036 | 2.6× | 0.013 | 0.096 | 0.0016 | 0.43 | +0.17 / +0.24 |

The new model's true direction is the LEAST probe-aligned of the three (4× generic raw, 2.6×
after Haufe; oth-adjacent 8× / 11×, standard 26× / 28×), yet it edits better than oth-adjacent
on the canonical bench — alignment does not order these two, raw or corrected. ⚠ Selection:
exact counterfactual boards are reachable almost only when the flipped tile is a
placement-parity tile — 2/42 clean cases here had a tile that was ever recoloured
(standard Othello: 11/18), so this measurement probes the LOOKUP copy of colour in the
flip model, not the computed one. In standard Othello the split is readable and points the
other way from the intuition: recoloured-tile cases align LESS than parity-tile cases (rows
0.174 vs 0.211 at pt 5).

**Haufe-corrected editing** (full bench): PI-haufe +0.291 / fid 0.53 (pt3 α2),
ND-haufe +0.328 / fid 0.47 (pt3 α1) vs canonical +0.168 / 0.85 and +0.237 / 0.62 — a
+0.1 gain and a closer landing, as on oth-adjacent (+0.12 → +0.375); the two corrected models
now sit within 0.05 of each other, both far below standard Othello's +0.63.

**The honest ceiling — the result that changes the reading.** On the SAME clean cases:

| run | n | unedited | true counterfactual (ceiling) | PI | ND | ND / ceiling | mass on legal_post: cf / PI / ND |
|---|---|---|---|---|---|---|---|
| L-oth-20m (standard) | 18 | -0.602 | **+0.688** | +0.294 (pt4 α3; bench +0.608) | +0.535 (pt4 α0.35; bench +0.622) | 0.78× | 0.999 / 0.971 / 0.993 |
| L-oth-adjacent-20m | 52 | -0.646 | **+0.252** | -0.086 (pt8 α5; bench -0.053) | +0.212 (pt1 α2; bench +0.118) | 0.84× | 1.000 / 0.998 / 0.984 |
| **L-oth-adjacent-flip-20m** | 42 | -0.670 | **+0.140** | +0.169 (pt2 α5; bench +0.168) | +0.336 (pt1 α2; bench +0.237) | 2.39× | 1.000 / 0.984 / 0.991 |

The ceiling is +0.14 here against +0.69 on standard Othello, and it is structural, not a
contaminated counterfactual: the counterfactual puts 100% of its mass on the post-edit legal
set, and so do the editors (98–99%). Under the adjacency rule a one-tile recolouring changes
2.2 of 16 legal moves (standard Othello: 2.0 of 11), so the two uniform reference worlds the
union Edit Index measures between differ by ~1/16 on two squares, and the honest model's
own deviation from uniform is of that size (`GOTCHAS.md` 2026-09-09). ND at +0.34 above a
+0.14 ceiling means it moves the differing squares MORE sharply than a real counterfactual
world does — an over-steer the union index rewards — while keeping legality; PI lands at the
ceiling. Read against ceilings, the picture inverts: the new model's edit is complete
(ND 2.4× ceiling, PI 1.2×), oth-adjacent's ND reaches 0.84× of ITS ceiling (+0.21 vs +0.25;
its PI fails), and standard Othello's ND 0.78×. **"A third of Othello's editability" is
mostly the metric's dynamic range across instances, not a third of the effect.** The union
Edit Index cannot be compared across instances whose legal-set geometry differs; a
ceiling-normalised index (or scoring only the symmetric-difference squares against the
counterfactual) is the right cross-instance quantity — not built here (canonical metric
untouched).

Status of this section: `observed` (single seed; 42 / 52 / 18 clean cases; ceilings on the
clean subsets, editors' full-bench numbers on all 1001).

## Log

- **2026-09-11 (later)** — alignment / Haufe / ceiling section added (`observed`): least probe-aligned Othello model yet edits above its own true-counterfactual ceiling (+0.34 vs +0.14 on the same cases); the ceiling is +0.14 here vs +0.69 on standard Othello, so the cross-instance Edit-Index gap is largely dynamic range.
- **2026-09-11** — `observed`. First and only run scored (chain 2026-09-10 00:15 → 2026-09-11
  04:22 PT on wsl-sevan, unit `oth_adjacent_flip`). Numbers above.
