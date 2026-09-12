# 2026-09-11 — Pilot: the model-referenced Edit Index on paired counterfactual histories

**Ask (Sevan).** Both Edit Index references become the MODEL'S OWN predictions — `p_A` on the
real history, `p_B` on a paired counterfactual history B that differs from A by one simulator
edit — so unedited scores −1 by construction and reproducing `p_B` scores +1. A few dozen
cases per condition, ONE arm per editor at the run's canonical best (point, α, dims), no
sweep. Filter out edits the model's prediction does not register. Report the Othello
tile-change statistics per instance.

**Code** `experiments/edit_index_v2_pilot/scripts/pilot.py` · **scores**
`experiments/edit_index_v2_pilot/scores/` · **figure**
`experiments/edit_index_v2_pilot/outputs/pilot_ei.png` · **log** `logs/edit_index_v2_pilot/`.
Formula = `pim.metrics.edit_index.edit_index_per_case` (the one formula) with new
ingredients; editors = the canonical `pim.editors` primitives, the Othello arms re-wired for
multi-tile targets. Status: `observed`, n = 48 per condition, one arm each, not canonical.

## Counterfactual construction

- **Discworld:** the edited object's whole trajectory shifted by the teleport vector over
  frames 0..EF−1 (`arms.counterfactual_history`), other object untouched, noise matched;
  kept if in-frustum and collision-free at every frame. Valid: 110/192 (noiseless), 66/192
  (8-ray, radius 1.0), 81/192 on the appearance-fac case list (50 with ≥ 2 differing rays).
- **Othello:** one of the last k = 4 moves substituted by another legal move, the original
  remaining moves replayed; kept if the replay is legal, the mover is unchanged, the board and
  the legal set differ. From 900 attempts on each instance's bench histories: 553 / 621 / 692 /
  664 pairs (uniform / adjacent-flip / adjacent / noflip). Rejections: illegal replay 87–227,
  mover changed 116 on uniform (passes) and ≤ 11 elsewhere, legal set unchanged 4 / 181 /
  101 / 31.
- **Filters.** `RMSE(p_A, p_B)` on the support ≥ max(floor, 0.25 × the simulator's own
  separation): dropped 0 (dw frames, noflip), 4–6 of 66–81 (token runs), ~3–4 % on the
  adjacency instances. Othello model-normal (legal mass ≥ 0.98 on BOTH histories): dropped
  40/553 on uniform (min legal mass on B = 0.00 — some substituted histories are broken
  for the model), 5/621 adjacent-flip, 3/692 adjacent, 0/664 noflip. The separation ratio
  model/simulator sits at 1.00 (IQR 0.98–1.03) on frames and Othello, 0.86–0.89 on the token
  model — these models see the counterfactual as sharply as the simulator does.

## Othello tile changes per counterfactual (all valid pairs)

| instance | pairs | tiles changed: mean · median · IQR · range | occupancy | colour (recolourings) | histogram |
|---|---|---|---|---|---|
| oth-uniform | 553 | **5.15** · 5 · [4, 6] · [2, 14] | exactly 2 | 3.15 (0–12) | 2:32 3:48 4:139 5:127 6:95 7:60 8:27 9:17 10+:8 |
| oth-adjacent-flip | 621 | 2.38 · 2 · [2, 2] · [2, 7] | exactly 2 | 0.38 (0–5) | 2:476 3:95 4:27 5:10 6:11 7:2 |
| oth-adjacent | 692 | 2.00 · 2 · [2, 2] · [2, 3] | exactly 2 | 0.00 (0–1) | 2:689 3:3 |
| oth-noflip | 664 | 2.00 · 2 · [2, 2] · [2, 2] | exactly 2 | 0 | 2:664 |

Every substitution vacates one square and occupies another (2 occupancy changes); the rest
is recolouring, which only the flip rules produce — 3.15 discs on average in standard
Othello, 0.38 under adjacency+flip (a locally placed disc encloses little). The count does
not depend on how far back the substitution sits (j = 1..4 all within ±0.2). So a paired
edit in standard Othello is a ~5-tile board change; on the three no-flip / adjacency
instances it is a 2-tile occupancy change.

## Results (n = 48 per row; one canonical arm per editor)

| condition | ceiling v1 | unedited v1 | PI v2 / v1 (g2) | ND v2 / v1 (g2) | GS v2 / v1 (g2) |
|---|---|---|---|---|---|
| L-oth-20m | +0.80 | −0.79 | +0.12 / +0.12 (1.56) | −0.25 / −0.23 (0.84) | **+0.68 / +0.63 (0.23)** |
| L-oth-adjacent-flip-20m | +0.71 | −0.70 | +0.05 / +0.05 (3.9) | +0.08 / +0.05 (0.79) | +0.02 / +0.02 (2.6) |
| L-oth-adjacent-20m | +0.73 | −0.73 | +0.02 / +0.02 (7.5) | +0.06 / +0.06 (1.24) | −0.00 / −0.00 (5.0) |
| L-oth-noflip-20m | +0.82 | −0.82 | +0.02 / +0.02 (2.6) | **+0.58 / +0.50 (0.27)** | +0.05 / +0.05 (2.3) |
| L-dw-noiseless-20m | +0.95 | −0.92 | +0.25 / +0.25 (1.96) | −0.04 / −0.05 (4.1) | −0.02 / −0.02 (1.08) |
| L-dw-8ray-20m | +0.85 | −0.89 | +0.29 / +0.31 (0.92) | −0.17 / −0.19 (1.22) | −0.07 / −0.05 (0.87) |
| L-dw-8ray-tok-20m | +0.76 | −0.80 | +0.05 / +0.01 (0.70) | — | −0.12 / −0.13 (0.73) |
| L-dw-8ray-20m @ appearance-fac | +0.85 | −0.89 | +0.44 / +0.47 (0.75) | +0.48 / +0.51 (0.80) | +0.42 / +0.42 (0.50) |
| L-dw-8ray-tok-20m @ appearance-fac | +0.73 | −0.80 | +0.36 / +0.28 (0.43) | +0.46 / +0.36 (0.41) | **+0.59 / +0.52 (0.31)** |

v2 = both references the model's predictions (unedited −1, `p_B` +1 by construction); v1 =
the canonical simulator references on the SAME cases; g2 = the v2 guard, RMSE(edited, p_B) /
RMSE(p_A, p_B), > 1 degraded; the v1 guard agrees with g2 to ±0.05 everywhere. Ceiling v1 =
`p_B` under the canonical index.

## Reading

1. **The metric change is small; the edit-type change is large.** On the same paired cases
   v2 and v1 agree to within ±0.03 on every frame and Othello row — these models predict
   both worlds about as well as the simulator renders them (separation ratio ≈ 1.00), so
   swapping simulator references for model references moves almost nothing. The token model
   is the exception (+0.04 to +0.10 higher under v2, ceilings 0.73–0.76): its honest mass
   split away from the point-mass frame reference is exactly what v2 removes. The
   fully-model-referenced index therefore does NOT change the discworld story, and it
   changes the Othello story only through what edits it makes askable.
2. **Realisable edits reorder the Othello editors.** On the ~5-tile paired edits of standard
   Othello GS reaches +0.68 (as on single flips), while PI drops from +0.61 to +0.12 and ND
   from +0.62 to −0.25 with a degraded guard. The arms are the canonical single-flip arms
   (point, α) applied to multi-tile targets without re-tuning, so this is a caution, not a
   verdict — but the editor that survives is the one that descends rather than writes a fixed
   step.
3. **⭐ oth-noflip is EDITABLE on realisable edits.** ND at its canonical arm reaches +0.58
   (v2) / +0.50 (v1) against a ceiling of +0.82, guard 0.27, on two-tile occupancy changes.
   The canonical bench scored this instance inert (ND +0.09) — but that bench asks for a
   one-tile recolouring, which is an IMPOSSIBLE state in a no-flip world (colour is locked to
   parity). This is the legal-vs-illegal covariate Sevan asked for, and its first reading is
   that "inert" on the checkerboard world was an artefact of asking for a state the world
   cannot be in. The adjacency instances stay inert on the same realisable two-tile edits
   (ND +0.06 / +0.08, PI and GS destructive), so the 2×2 story sharpens: the checkerboard
   world (colour irrelevant) edits by occupancy; the adjacency worlds (colour used) do not,
   even when asked for a state they can reach.
4. **Discworld unchanged.** Regression editors at the floor or destructive on both frame
   runs; appearance-fac stays editable on both interfaces (GS +0.42 frame, +0.59 token).
5. **The filters barely bite here** (0–7 %), because the substitution counterfactuals are
   in-distribution histories — unlike the swap-built ones of 2026-09-09 (legal mass 0.845).
   Substitution + replay is the right generator.

## Caveats

n = 48, one arm per editor, arms not re-tuned for multi-tile edits (Othello) or for the
subset (discworld); k ≤ 4 substitutions only; standard-Othello edits are 2–14 tiles, so the
magnitude axis is not yet controlled — the by-magnitude figure is the next thing to build.
The v2 support is still the simulator's (differing rays / union of legal sets), not the
model's. Not in `pim.metrics`; no REGISTRY row; nothing canonical changed.

## Open

- Sevan's design decision (v2 fully model-referenced vs p_A + S(B)): on this evidence the
  two agree wherever both are defined; the decision is about illegal edits, which v2 cannot
  score. The noflip result says the legal/illegal split is itself a finding.
- Re-tune PI/ND α for multi-tile Othello edits before reading point 2 as a fact.
- Magnitude-stratified rerun (tiles changed 2 / 3–4 / 5+) on oth-uniform.
