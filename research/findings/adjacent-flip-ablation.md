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

**Ceiling check (2026-09-11, corrected the same day):** on ordinary exact-counterfactual cases the
model run on the true counterfactual world scores +0.66 here, +0.68 on oth-adjacent and +0.70 on
standard Othello — Bayes-optimal on all three — and the canonical editors sit below it (ND
recovers 55% / 30% / 75% of the ceiling on the same cases). An earlier version of this paragraph
claimed a structurally low ceiling; that came from swap-built counterfactual histories slipping
through a legal-mass filter that does not bite on adjacency instances, and is withdrawn (see the
alignment section and `GOTCHAS.md` 2026-09-11).

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

**The honest ceiling — corrected the same day (supersedes the paragraph this replaces).** The
first pass filtered counterfactual histories on the model's legal mass ≥ 0.99, the criterion
that had worked on standard Othello. On the adjacency instances that filter is TOOTHLESS: those
models put legal mass 1.000 on essentially every history, including the off-distribution ones
built by move SWAPS — and every exact counterfactual board in all three instances is swap-built
(0 substitutions reach the flipped board). The first-pass "ceilings" of +0.14 / +0.25 were
therefore the model reacting to abnormal histories, not a property of the index: a perfect
predictor scores exactly +1.000 on these cases, and on the counterfactual histories the model's
rmse to uniform-over-its-own-legal-set was 3–4× its held-out value (0.0076 / 0.0054 vs 0.0021 /
0.0019). Filtering instead on ORDINARINESS (that rmse within the held-out 95th percentile for
the same prefix length; `experiments/adjacent_flip_ablation/scripts/honesty_check_v2.py`):

| run | exact boards | kept: legal-mass filter → ordinariness filter | unedited | true counterfactual (ceiling) | PI | ND | ND / ceiling | alignment at best pt: rows / Haufe (full-probe patterns) |
|---|---|---|---|---|---|---|---|---|
| Standard Othello | 42 (all swaps) | 18 → 16 | -0.587 | **+0.697** (was +0.688) | +0.247 | +0.520 | 0.75 | pt 5: 0.185 / 0.262 |
| oth-adjacent | 136 (all swaps) | 52 → 26 | -0.669 | **+0.679** (was +0.252) | -0.079 | +0.206 | 0.30 | pt 1: 0.088 / 0.250 |
| **oth-adjacent-flip** | 90 (all swaps) | 42 → 18 | -0.664 | **+0.655** (was +0.140) | +0.073 | +0.359 | 0.55 | pt 1: 0.048 / 0.184 |

With ordinary counterfactuals the ceilings agree across instances (+0.66 to +0.70 — the model
is Bayes-optimal on all three, as its CE says), and the editors sit BELOW them, as they must. On
the same cases ND recovers 55% of the ceiling here, 30% on oth-adjacent and 75% on standard
Othello, so the canonical ordering (standard > flip > adjacent) holds and the "third of
Othello" reading on the full bench stands as written — the earlier claim that it was "mostly
metric range" is withdrawn. The alignment ordering is unchanged on the ordinary subsets. ⚠ Two
definitions of "Haufe rows" are in use: `othello_alignment.py` builds patterns from the 3
selected rows alone (the summary-table column), the extras and the Haufe editor from the full
192-row probe (the column above); they differ (e.g. 0.094 vs 0.161 at pt 1 here) and are
labelled wherever quoted.

Status of this section: `observed` (single seed; 16 / 26 / 18 ordinary cases; lookup-tile
biased — see the selection note above).

## Presence edits — occupied ↔ empty, the standard way (2026-09-11)

Sevan's question: the colour edits above steer badly on the adjacency instances; can the same
models be steered on PRESENCE, the variable adjacency legality most directly depends on? Same
construction as `flip-ablation.md` §"Presence is editable" (2026-09-07): a dedicated 2-class
linear presence probe per residual point (error ≤ 1.2% everywhere, ≤ 0.07% at pts 2–4), PI
through it (swap the tile's empty/occupied logits, re-solve in z-space), 400 artificial boards
from the instance's test split (204 REMOVE an occupied non-centre disc, 196 ADD a parity-colour
disc on an empty square; Li's length mix), scored the standard way against uniform-over-legal
of the pre- and post-edit boards under the instance's OWN rules, with the fidelity guard.
Rule-aware re-implementation: `experiments/adjacent_flip_ablation/scripts/presence_edit.py`
(the 2026-09-07 script only knew the flip flag, which gives the wrong legal sets on adjacency
instances); the standard-Othello row reproduces the 2026-09-07 number exactly.

| model | unedited | best presence PI (pts 1–6, α ≤ 5) | remove / add | best under the guard (fid ≤ 1.1) | extended α (pts 2–3, α ≤ 20) | colour edits, same model |
|---|---|---|---|---|---|---|
| L-oth-20m (standard) | −0.695 | **+0.447 / fid 0.37** (pt4 α3) | +0.60 / +0.28 | same | — | PI +0.61, ND +0.62 |
| L-oth-adjacent-20m | −0.648 | −0.042 / fid 4.02 (pt6 α5) | +0.02 / −0.11 | −0.123 / 1.00 (pt4 α3) | +0.015 / fid 5.0 (pt2 α20) | PI −0.05, ND +0.12 |
| **L-oth-adjacent-flip-20m** | −0.662 | +0.060 / fid 1.40 (pt3 α5) | +0.25 / −0.13 | **+0.049 / 0.74** (pt2 α5) | +0.077 / fid 1.27 (pt2 α8) | PI +0.17, ND +0.24 |

The presence read-out LANDS on every model at every α ≥ 1 (landed 1.00), as before. Standard
Othello's presence is editable at the colour edits' level (+0.45, guard 0.37; removing a disc
+0.60, adding +0.28). On both adjacency models presence is **not** editable: the best arms are
at the unedited floor with the guard intact, or past it only by destroying the output (guard
4–5). Removal edits on the flip model move the output a little (+0.19 to +0.25 at guards 0.7–1.4),
adds not at all. So on the adjacency instances presence — which the placement rule consumes
directly — behaves like colour did on oth-adjacent, while on standard Othello both variables
are editable. The 2026-09-07 reading ("the operative condition is whether the output
computation consumes the probed variable") does not survive this: adjacency legality consumes
presence and colour of the neighbourhood, both are decodable at ~100%, and neither is
editable. What separates standard Othello from the adjacency worlds is not which variables the
output consumes but how — the enclosure rule's line scans versus a local 8-neighbourhood test —
and that difference is now the leading candidate for the editability gap.

Status: `observed` (one seed per model; 400 cases; the α grid extended to 20 at the two best
points only; PI only, as in the 2026-09-07 test). Scores
`experiments/adjacent_flip_ablation/scores/presence_edit_*.json` (+ `_ext_alpha`); log
`logs/adjacent_flip_ablation/presence_edit*.log`.

## Flipped tiles: decoded worse, edited worse, computed later (2026-09-11, `scripts/flipped_tiles.py`)

Sevan's three questions on the flip model, all cheap (1.2 min): (a) does legality degrade with
recolourings? (b) are FLIPPED tiles (current colour ≠ placement colour) more editable, searching
all points? (c) how accurate is the cached mine/theirs probe on flipped tiles?

**(a)** Mildly. Test-split prefixes binned by recolourings so far: legal mass 0.9997 (0–3 flips)
→ 0.9990 (4–8) → 0.9953 (9–15) → 0.9903 (16+); rmse-to-uniform 0.0021 → 0.0036. Confounded with
prefix length (the heavy-flip bins are the long prefixes), but the model does fumble a little in
flip-heavy positions while staying near-optimal.

**(c)** The probe reads the parity lookup. 19% of occupied tile-rows are flipped. Linear
mine/theirs probe error, flipped vs parity tiles: pt 1 28.2% vs 7.1%; pt 2 19.5 / 4.8; pt 3
12.7 / 4.3; pt 4 9.3 / 4.1; pt 5–8 ≈ 9.5 / 4.7. Flipped-tile colour becomes decodable only
progressively over layers 1–4 (a computed variable), and never as well as parity colour; the
aggregate skill 0.947 is mostly the lookup.

**(b)** Flipped tiles are LESS editable, not more. 300 synthesised cases each (Li's length mix,
target tile flipped-in-game vs parity), canonical PI (α ≤ 8) and ND (α ≤ 2) at every point:

| cases | unedited | best PI (guard ≤ 1.1) | best ND (guard ≤ 1.1) | guarded ND by point 1–4 |
|---|---|---|---|---|
| parity tiles | −0.687 | +0.143 / 0.88 (pt2 α5) | **+0.215 / 0.64** (pt1 α2) | +0.22 +0.10 −0.01 −0.33 |
| flipped tiles | −0.645 | −0.035 / 0.87 (pt4 α3) | +0.085 / 0.95 (pt3 α0.7) | +0.05 +0.06 +0.08 −0.36 |

The flipped-tile optimum sits later (pt 3–4, where its decodability has formed) and lower. So
the "registers may sit elsewhere" idea is half right — flipped colour is computed later and
elsewhere in depth — but the canonical probe, fitted on rows that are 81% parity tiles, reads
that register at 9–13% error, and edits through it are misdirected. Natural follow-up: a
per-tile probe fitted on FLIPPED rows only (or on the residual from the parity lookup), then
edit flipped-tile cases through it; if that lands, the register exists and the probe was the
limitation. Status `observed`. Scores `scores/flipped_tiles_L-oth-adjacent-flip-20m.json`.

## Log

- **2026-09-11 (night)** — flipped tiles: decoded worse (9–28% vs 4–7% error), edited worse (ND +0.09 vs +0.22 guarded), computed later (pt 3–4); legality degrades mildly with flips. Follow-up: a flipped-row probe.
- **2026-09-11 (evening)** — presence edits (`observed`): standard Othello +0.447 / 0.37 (reproduced); oth-adjacent and oth-adjacent-flip NOT presence-editable (best guarded −0.12 and +0.05; destructive past that); removal > add on the flip model. Overturns the 2026-09-07 "consumed variable is editable" reading.
- **2026-09-11 (later still) — retracts the ceiling claim of the entry below.** The +0.14 / +0.25 "ceilings" were swap-built counterfactual histories passing a legal-mass filter that is toothless on adjacency instances (mass 1.000 on everything); filtered on ordinariness the ceilings are +0.66 / +0.68 / +0.70 and the editors sit below them (ND 55% / 30% / 75%). Caught by Sevan ("the model is near the Bayes floor; it would not score so poorly"); verified by scoring the ideal uniform-over-legal_post (+1.000) and the model's rmse-to-uniform on the counterfactual histories (3–4× held-out). The "mostly dynamic range" reading is withdrawn; the canonical ordering stands.
- **2026-09-11 (later)** — alignment / Haufe / ceiling section added (`observed`): least probe-aligned Othello model; first-pass ceiling claim (+0.14, "dynamic range") — RETRACTED above.
- **2026-09-11** — `observed`. First and only run scored (chain 2026-09-10 00:15 → 2026-09-11
  04:22 PT on wsl-sevan, unit `oth_adjacent_flip`). Numbers above.
