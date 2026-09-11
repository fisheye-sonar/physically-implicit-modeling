# Edit-direction alignment — probe subspace vs the true edit direction (2026-09-09)

**Question.** The RNN-era archive found the true edit direction actively misaligned with the
probe's row space, and Haufe / PCA corrections improved alignment without making the edits
work. Re-done at scale on the three Transformer-L runs with oracle counterfactual states
(`experiments/edit_direction_alignment/`, existing linear probes, no canonical change).

**Method.** Δ = h_cf − h at the last position, where h_cf is the residual the model produces
on a history that really is the edited world. Discworld: the edited object's trajectory is
shifted and re-rendered (exact; the model predicts the edited frame at rmse 0.028 vs 0.27).
Othello: a real history of the same length and mover whose board exactly equals the edited
board, ⛔ **filtered on the model treating it as an ordinary game** (legal mass ≥ 0.99) — see
the gotcha; without that filter the analysis is contaminated and gives the wrong answer.

## Result 1 — alignment tracks editability, sharply

Fraction of Δ inside the probe subspace at the best residual point, against the generic
baseline (an unrelated displacement of the same kind):

| model | rows | generic | ratio | Haufe | editable |
|---|---|---|---|---|---|
| L-oth-20m (standard) | **0.188** (pt 5) | 0.007 | **27×** | 0.238 | yes (+0.61) |
| L-oth-adjacent-20m | 0.077 (pt 1), 0.010 by pt 5 | 0.004–0.009 | 8× → 2× | 0.236 → 0.107 | no (≈0) |
| L-dw-noiseless-20m | 0.013 | 0.020 | **≤1×** | 0.030 | no (≈0) |

The editable model's counterfactual displacement lands 27× more inside a THREE-row subspace
(one tile × three classes) than chance; discworld's lands at or below chance in the four
position rows. Haufe patterns raise signal and baseline together and do not reorder the
models — the archive's conclusion, replicated: a better-aligned direction is not sufficient
(adjacent's Haufe fraction at pt 1, 0.236, equals standard Othello's, and it is inert).

## Result 2 — discworld's residual IS load-bearing, and the code is nonlinear

Patching discworld's last-position residual with the full oracle Δ produces the edit (Edit
Index +0.81…+0.94 at points 3–8, fidelity 0.10–0.14). The complement of the probe row space
carries nearly all of it (+0.71…+0.86); the row-space part alone carries none (−0.93,
fidelity 1.0). Δ is high-rank (≈70 directions for 90% of variance over 110 cases), not linear
in the teleport (oracle-linear R² ≈ 0), and as large as the residual itself (‖Δ‖/‖h‖ ≈ 1;
centred cos(h, h_cf) ≈ 0.43 vs ≈ 0 for unrelated cases). This kills the "later layers
re-derive position from the frame" account: the information that moves the output is present
at the point we edit, in a direction no linear read-out of position spans.

## Result 3 — the editors are honest, and near the achievable ceiling

On oth-uniform cases with a valid, model-normal counterfactual (n = 16):

| | Edit Index | mass on legal_post | rmse to uniform-post |
|---|---|---|---|
| unedited | −0.567 | | |
| true counterfactual (ceiling) | **+0.650** | 0.9990 | 0.0084 |
| ND (pt 4, α 0.35) | +0.592 | 0.9968 | 0.0094 |
| PI (pt 4, α 3) | +0.437 | 0.9709 | 0.0151 |

ND recovers 91% of the achievable edit and no editor exceeds the true counterfactual. The
ceiling sits at +0.65 rather than +1 because with |legal| ≈ 10 and a symmetric difference of
1–2 squares, the model's honest deviation from uniform is a real fraction of the separation
between the two reference worlds. The edits do not game the metric: Li error against the
post-edit world falls 2.67 → 0.00 (ND) / 0.21 (PI) while error against the pre-edit world
rises 0.00 → 2.29 / 2.23, at fidelity 0.20 / 0.30.

## Result 4 — editing ALONG the Haufe directions: helps, does not rescue

`scripts/haufe_edit.py`. The pattern matrix P satisfies W Pᵀ = I, so Pᵀ is a right inverse of
the read-out map exactly as W⁺ is: PI-haufe (Δz = α Pᵀ δy) and ND-haufe (direction = P[target
row] − P[current row]) land the same read-out target and differ from the canonical editors
only in the null-space component of the write.

| model | canonical best | Haufe-corrected best | fidelity |
|---|---|---|---|
| L-oth-20m (control) | PI +0.608 / ND +0.622 | PI +0.635 (pt 5, α 2), ND +0.631 (pt 4, α 0.7) | 0.23 |
| L-oth-adjacent-20m | PI −0.053 / ND +0.118 (ext. sweep ≤ +0.13) | **PI +0.270, ND +0.375** (pt 2) | **0.60** |
| L-dw-noiseless-20m | PI +0.233 @ fid 1.95 | PI +0.259 @ 1.39; guarded **+0.186 @ 1.09** | |

Haufe triples adjacent's best edit and makes it NON-DESTRUCTIVE for the first time (fid 0.60),
and turns discworld's destructive +0.23 into an honest +0.19 — but neither approaches the
+0.63 of the editable model, and on that model the correction changes nothing. The decisive
case is the TOKEN discworld run (`haufe_edit_tokens.py`), which has the largest ABSOLUTE
alignment gain of any model (cos²θ 0.002 → 0.070, ×32) and whose edit moves only +0.004 →
+0.057 against an unedited floor of −0.755: at α 60–175 the write drives p(unedited frame)
0.53 → 0.012 while p(edited frame) reaches just 0.078 — the mass leaves one world without
arriving at the other. Haufe buys editability only where alignment was already ABOVE the
generic baseline; discworld stays at 0.5–1.2× before and after. Decisive
against alignment as the explanation: at its best editing point adjacent has HIGHER Haufe
alignment than standard Othello has at its best editing point (0.200 at pt 2 vs 0.146 at pt 4)
and reaches +0.375 against +0.631. Matched alignment, half the editability.

## Result 5 — the grid (classification) probes are the one above-chance discworld read-out

`scripts/grid_probe_alignment.py`, same model / Δ / cases, 104 cell-changing cases; the
subspace is the 6 rows ({empty, obj0, obj1} × the two cells that change) against the
regression probes' 4 position rows:

| probe target | rows | generic | ratio | Haufe | genH | ratio |
|---|---|---|---|---|---|---|
| grid 3-way (pt 5) | 0.057 | 0.025 | **2.3×** | 0.018 | 0.019 | 1.0× |
| regression (pt 5) | 0.014 | 0.018 | 0.8× | 0.031 | 0.040 | 0.8× |

Making the target categorical lifts a discworld read-out above chance for the first time —
and Haufe then washes it back to chance, which is exactly the ordering its editability
follows (ND +0.373 raw → +0.048 Haufe; `findings/grid-target-control.md`). 2.3× against
Othello's 26× remains the gap.

## Result 6 — the factorised categorical probe is the best-aligned discworld read-out yet, raw; Haufe washes it to chance again

`scripts/fac_probe_alignment.py` (2026-09-10), `L-dw-8ray-20m`, the same oracle counterfactual
(the edited object's trajectory shifted and re-rendered; Δ = h_cf − h at the last context
position). On the 8-ray instance (radius 1.0) only **66 of 192** shifted trajectories stay
inside the frustum and clear of the other disc; the shift changes the object's cell in all 66
(its run centre in 66, its length in 32). Three read-outs on the same model and cases:

| read-out (pt 5; pts 1–8 within ±0.005) | rows | Δ in rows | generic | random rank | ratio vs generic / random | Haufe | genH | randH |
|---|---|---|---|---|---|---|---|---|
| **appearance-fac** — (tile, old) + (tile, new) of the moved factors | 2–4 | **0.070** | 0.025 | 0.006 | **2.8× / 12×** | 0.034 | 0.023 | 0.039 |
| appearance (joint cell) — {empty, obj0, obj1} × the two cells | 6 | 0.048 | 0.020 | — | 2.4× | 0.074 | 0.045 | — |
| regression (full, frustum) — the 4 position rows | 4 | 0.003 | 0.003 | — | 1.0× | 0.078 | 0.090 | — |

And the single ND write direction (target − current contrast) against Δ, cos²:

| direction (pt 5) | raw | Haufe | random direction |
|---|---|---|---|
| **appearance-fac** (summed contrast over the moved factors) | **0.056** | 0.013 | 0.002 |
| appearance (joint cell) | 0.035 | 0.007 | 0.002 |

Point 0 (the input embedding) is the exception everywhere: fac rows 0.28 / generic 0.18, joint
cell 0.57 / 0.30 — there the residual IS the frame, and the probes read it directly.

**Reading.** The factorised probe's raw row space holds 2.8× the generic and 12× the
random-rank share of the true displacement, and its one ND direction captures 5.6% of Δ's
energy — 28× a random direction and 1.6× the joint cell's contrast. That is the ordering
the editors follow (fac ND +0.50 > joint ND +0.43 ≫ regression, `probe-target-type.md`), so
Result 5's pattern holds with the better-aligned probe: **raw alignment tracks editability;
Haufe does not.** The Haufe patterns of the fac rows sit AT the random-pattern floor (0.034 vs
0.039 — a random pattern subspace captures 4% of Δ because patterns concentrate in the
residual's high-variance directions, and so does Δ by chance), and the Haufe ND direction loses
three quarters of its overlap. Across the whole project no Haufe-corrected discworld subspace
has ever risen above its own baseline; the raw categorical rows have, twice. The regression rows
stay at the generic floor (0.003), as in Result 1. Absolute numbers remain small: 5–7% of Δ in
the writable subspace against Othello's 19% (Result 1) — the categorical read-outs close part
of the gap, not all of it.

## Result 7 — oth-adjacent-flip: the least-aligned Othello model edits to (and past) its own ceiling (2026-09-11)

`othello_alignment.py --run runs/adjacent_flip_ablation/L-oth-adjacent-flip-20m` (+ extras in
`experiments/adjacent_flip_ablation/scripts/`): 42 clean cases (90 exact boards of 900; cf legal
mass 0.9997). Best point 1: rows **0.051** vs generic 0.012 (4.4×), Haufe 0.094 vs
0.036 (2.6×); ND-direction cos² raw 0.013 / Haufe 0.096 / random 0.0016; ‖Δ‖/‖h‖ 0.43.
Lower than oth-adjacent (0.077 / 8×, Haufe 0.236) on every measure, yet canonically more
editable (ND +0.24 vs +0.12). Haufe-corrected edits: PI +0.291 / fid 0.53, ND +0.328 / fid 0.47
(oth-adjacent: +0.270 / +0.375) — the correction pulls the two inert-ish models together and
leaves them at half of standard Othello. Two things the extras add:

1. **Selection bias in the Othello counterfactuals.** Exact one-tile-flipped boards are
   reachable almost only for placement-parity tiles: 2/42 clean cases on the flip
   instance had a tile that was ever recoloured (standard Othello 11/18). The alignment number for
   the flip model is therefore about its LOOKUP copy of colour. In standard Othello, where the
   split is readable, recoloured-tile cases are LESS aligned than parity-tile cases (rows
   0.174 vs 0.211 at pt 5).
2. **The ceiling inverts the cross-instance reading.** Same clean cases, canonical best arms:
   flip ceiling **+0.140**, PI +0.169, ND **+0.336** (2.4× the ceiling, 99% mass on legal_post —
   an over-steer of the 2-square symmetric difference, not a contaminated counterfactual);
   oth-adjacent ceiling +0.252, ND +0.212 (0.84×), PI −0.086; standard Othello ceiling +0.688,
   ND +0.535 (0.78×), PI +0.294. Adjacency legality changes ~2 of 16 legal moves per
   recolouring (Othello 2 of 11), so the union index has a fifth of the range there. Cross-
   instance Edit-Index comparisons need a ceiling normalisation; `adjacent-flip-ablation.md`.

## Scope note

⛔ This analysis modifies the INSTRUMENT (write directions), which is outside the project's
core design — vary the environment, hold the analysis pipeline fixed (`RESEARCH.md`, "The
independent variable is the ENVIRONMENT, not the editor"). It is kept because its numbers can
be read as a per-environment MEASUREMENT: how far each world places the true edit direction
from what a fixed linear probe reads. Quoted that way it sits beside decodability and
editability; quoted as "a better editor" it changes the subject. The canonical table keeps the
canonical editors.

## Reading

Alignment separates the editable model from the two inert ones, but it is a reframing, not an
explanation. Correcting the directions (Haufe) buys a real but partial improvement on the
inert models — adjacent +0.12 → +0.375 at fidelity 0.60 — while leaving the editable model
unchanged, and matched alignment still yields half the editability. So alignment is
necessary-ish and clearly not sufficient. The open question is what
property of an environment produces a representation whose counterfactual displacement is
low-dimensional and read-out-aligned, versus discworld's high-rank case-specific code.

Caveats: n = 18 / 52 clean Othello cases (a one-disc flip is usually an unreachable board —
disc counts are parity-locked without flips); discworld 110 of 192 cases; linear probes only,
so the archive's point that the MLP story differs still stands.

Assets: `experiments/edit_direction_alignment/{scripts,scores/summary.md}`; the ceiling and
honesty checks are `scripts/sanity_ceiling.py`, `scripts/ceiling_check.py`.
