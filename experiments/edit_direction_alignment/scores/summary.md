# Edit-direction alignment — probe subspaces vs the oracle counterfactual displacement (2026-09-09)

## What is measured

Δ = h_cf − h at the last position: the displacement from the model's residual on the real
history to its residual on a history that IS the edited world. Everything is in the z-space of
that layer's canonical linear probe (the residual standardised by the probe's own mean/scale —
the space PI and ND write in), dimension d = 512.

For a subspace S with orthonormal basis Q (obtained by QR on the stacked rows):

        frac = ‖Q Qᵀ Δ‖² / ‖Δ‖²  =  cos²θ,   θ = the principal angle between Δ and S

i.e. ordinary cosine alignment, squared, generalised from vector-to-vector to
vector-to-subspace (θ is the smallest angle between Δ and any vector in S).

| column | the subspace S | dim r |
|---|---|---|
| **rows** | the probe's WEIGHT ROWS for the read-outs the edit targets — the backward, min-norm decoding directions PI writes along and ND builds its direction from. **No Haufe correction.** | dw 4 (position), othello 3 (the one flipped tile × 3 classes) |
| **Haufe** | the ACTIVATION PATTERNS of those same read-outs, A = Σ Wᵀ (W Σ Wᵀ)⁻¹ with Σ the residual covariance — the forward/encoding directions a min-norm backward probe systematically misses (Haufe et al. 2014). **This is the corrected version.** | same r |
| **generic (gen)** | the identical measurement with Δ replaced by the displacement to an UNRELATED case's residual at the same layer: the empirical chance level. | same r |

Analytic chance for an isotropic direction is r/d (0.0059 for r=3, 0.0078 for r=4), equivalent
to an angle of ~85°. The empirical `generic` is the honest baseline because between-case
displacements are NOT isotropic — discworld's concentrate in high-variance directions that
partly overlap the probe rows, which is why its generic (0.020) sits above r/d (0.008).

### Headline, in angles

| model | r | cos²θ (rows) | angle | generic | ratio | Haufe cos²θ |
|---|---|---|---|---|---|---|
| L-oth-20m (pt 5) | 3 | 0.188 | 64.3° | 0.007 | 25.7× | 0.238 |
| L-oth-adjacent-20m (pt 1) | 3 | 0.077 | 73.8° | 0.009 | 8.4× | 0.236 |
| L-dw-noiseless-20m (pt 5) | 4 | 0.013 | 83.4° | 0.020 | 0.7× | 0.030 |

A random direction sits at 85.6° from a 3-dim subspace of 512. Standard Othello's true edit
direction leans to 64°; discworld's is orthogonal within noise. ⚠ These are each model's BEST
point and the best point differs; matched at point 5, adjacent is 0.010 vs standard's 0.188.
⚠ After the Haufe correction the two Othello models are indistinguishable (0.238 vs 0.236) and
one is editable while the other is inert — corrected alignment does not predict editability.

**Counterfactual construction.** Discworld: the edited object's whole trajectory is shifted by the teleport and re-rendered —
EXACT (110 of 192 cases stay in-frustum and collision-free; the model predicts the edited frame at rmse 0.028 vs 0.27
unedited). Othello: a real history of the same length and mover whose board EXACTLY equals the edited board, found by search
over move substitutions and swaps, and then FILTERED on the model treating it as an ordinary game (legal mass ≥ 0.99).
That filter is essential: swapped histories are legal but the model handles them badly (legal mass 0.845 vs 0.994 for
substitutions and 0.998 for ordinary prefixes), and they contaminated the first run of this analysis. Clean cases:
oth-uniform 18 of 900 (42 exact boards, 18 model-normal), oth-adjacent 52 of 900 (136 exact, 52 model-normal) —
adjacency yields more exact boards because its own placement rule is local, but a one-disc flip is usually unreachable in
both worlds (disc counts are parity-locked without flips). Exactly ONE tile differs per case, so the probe subspace is 3 rows.

## Fraction of Δ in the probe subspace, by residual point

| point | dw rows / gen | dw Haufe / gen | oth-uniform rows / gen | Haufe / gen | oth-adjacent rows / gen | Haufe / gen |
|---|---|---|---|---|---|---|
| 0 | 0.000 / 0.000 | 0.071 / 0.080 | **0.000** / 0.002 | 0.001 / 0.013 | 0.022 / 0.014 | 0.032 / 0.028 |
| 1 | 0.008 / 0.012 | 0.028 / 0.040 | **0.033** / 0.002 | 0.032 / 0.006 | 0.077 / 0.009 | 0.236 / 0.022 |
| 2 | 0.009 / 0.011 | 0.018 / 0.025 | **0.064** / 0.003 | 0.054 / 0.007 | 0.052 / 0.008 | 0.200 / 0.022 |
| 3 | 0.008 / 0.012 | 0.021 / 0.029 | **0.096** / 0.004 | 0.080 / 0.005 | 0.027 / 0.007 | 0.161 / 0.025 |
| 4 | 0.010 / 0.016 | 0.025 / 0.040 | **0.167** / 0.006 | 0.146 / 0.007 | 0.015 / 0.006 | 0.115 / 0.031 |
| 5 | 0.013 / 0.020 | 0.030 / 0.045 | **0.188** / 0.007 | 0.238 / 0.009 | 0.010 / 0.004 | 0.107 / 0.030 |
| 6 | 0.012 / 0.018 | 0.032 / 0.047 | **0.112** / 0.005 | 0.195 / 0.009 | 0.008 / 0.004 | 0.100 / 0.030 |
| 7 | 0.012 / 0.019 | 0.034 / 0.051 | **0.067** / 0.003 | 0.186 / 0.014 | 0.008 / 0.004 | 0.097 / 0.029 |
| 8 | 0.012 / 0.018 | 0.034 / 0.049 | **0.061** / 0.003 | 0.145 / 0.012 | 0.008 / 0.004 | 0.097 / 0.028 |

Ratio to the generic baseline at the best point: **oth-uniform ~27×** (0.188 vs 0.007, pt 5), oth-adjacent ~8× at pt 1 and
~2× from pt 3 on, discworld **at or below 1×** (0.012 vs 0.018). The ordering tracks editability (+0.61 / ≈0 / ≈0), and only
the editable model is far from its baseline. Haufe raises both signal and baseline; it does not change the ordering.

## Discworld: structure of Δ and the causal patch (canonical Edit Index, unedited -0.93)

| point | ‖Δ‖/‖h‖ | rank for 90% | oracle-linear R² (Δ ~ teleport) | patch full Δ | rows-only part | complement |
|---|---|---|---|---|---|---|
| 0 | 0.96 | 33 | -0.04 | +0.01 / fid 0.59 | -0.93 / 1.00 | +0.01 / 0.59 |
| 1 | 0.98 | 62 | -0.05 | +0.49 / fid 0.31 | -0.93 / 1.00 | +0.41 / 0.37 |
| 2 | 1.01 | 70 | -0.05 | +0.64 / fid 0.22 | -0.93 / 1.00 | +0.53 / 0.30 |
| 3 | 1.03 | 70 | -0.04 | +0.81 / fid 0.14 | -0.92 / 0.99 | +0.71 / 0.21 |
| 4 | 1.05 | 71 | -0.04 | +0.90 / fid 0.12 | -0.91 / 0.99 | +0.82 / 0.18 |
| 5 | 1.05 | 71 | -0.04 | +0.92 / fid 0.11 | -0.88 / 0.97 | +0.84 / 0.18 |
| 6 | 1.05 | 70 | -0.04 | +0.93 / fid 0.11 | -0.87 / 0.96 | +0.86 / 0.18 |
| 7 | 1.04 | 70 | -0.04 | +0.94 / fid 0.10 | -0.85 / 0.96 | +0.85 / 0.19 |
| 8 | 1.04 | 69 | -0.04 | +0.94 / fid 0.10 | -0.80 / 0.96 | +0.81 / 0.26 |

Patching the last-position residual with the FULL oracle Δ produces the edit (+0.81…+0.94 at points 3–8); the complement of
the row space carries nearly all of it and the row-space part carries none. So the residual is load-bearing and position's
load-bearing code is high-rank (≈70 directions for 90% of variance), not linear in the teleport, and as large as the residual
itself (centred cos(h, h_cf) ≈ 0.43 vs ≈ 0 for unrelated cases).

## The ceiling, and what the editors actually do (oth-uniform, `sanity_ceiling.py`)

On cases with a valid counterfactual the model handles normally (n = 16):

| | Edit Index | mass on legal_post | rmse to uniform-post |
|---|---|---|---|
| unedited | −0.567 | | |
| **true counterfactual (the ceiling)** | **+0.650** | 0.9990 | 0.0084 |
| ND (pt 4, α 0.35) | +0.592 | 0.9968 | 0.0094 |
| PI (pt 4, α 3) | +0.437 | 0.9709 | 0.0151 |

ND recovers **91% of the achievable edit**; no editor exceeds the true counterfactual. The ceiling is +0.65 rather than +1
because with |legal| ≈ 10 and a symmetric difference of 1–2 squares the model's honest deviation from uniform (rmse 0.008) is
a real fraction of the gap between the two reference worlds. The edits are not gaming the metric: Li error against the
post-edit world falls from 2.67 (unedited) to 0.00 (ND) / 0.21 (PI) while error against the pre-edit world rises from 0.00 to
2.29 / 2.23, with fidelity 0.20 / 0.30 — mass moves off newly-illegal squares onto newly-legal ones, and the edited output is
as clean as a healthy prediction on a real game.


## Editing along the Haufe directions (`scripts/haufe_edit.py`, `scores/haufe_edit.json`)

W Pᵀ = I, so Pᵀ is a right inverse of the read-out map exactly as W⁺ is; PI-haufe and ND-haufe
hit the same read-out target and differ only in the null-space component of the write.

| model | canonical best | Haufe best | fidelity | Haufe alignment at that point |
|---|---|---|---|---|
| L-oth-20m | PI +0.608 / ND +0.622 | PI +0.635 (pt5 α2), ND +0.631 (pt4 α0.7) | 0.23 | 0.238 / 0.146 |
| L-oth-adjacent-20m | PI −0.053 / ND +0.118 | PI +0.270, ND +0.375 (pt2 α1) | 0.60 | 0.200 |
| L-dw-noiseless-20m | PI +0.233 @ fid 1.95 | PI +0.259 @ 1.39, guarded +0.186 @ 1.09 | | 0.018–0.030 |

Helps (adjacent 3x, and non-destructive for the first time; discworld's destructive write
becomes an honest one) but does not rescue: neither reaches the editable model's +0.63, and on
that model the correction is a wash. ⚠ Adjacent has HIGHER Haufe alignment at its best editing
point (0.200) than standard Othello at its best (0.146) and edits half as well.

## All trained variants — alignment at each model's best residual point

| instance | run | editable | n | pt | rows | gen | ×gen | angle | Haufe | genH | ×genH | rank90 | ‖Δ‖/‖h‖ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dw-pn04 | L-dw-20m | no | 102 | 4 | **0.014** | 0.020 | 0.7× | 83.1° | 0.046 | 0.058 | 0.8× | 63 | 1.07 |
| dw-noiseless | L-dw-noiseless-20m | no | 110 | 5 | **0.013** | 0.020 | 0.7× | 83.4° | 0.030 | 0.045 | 0.7× | 71 | 1.05 |
| dw-8ray | L-dw-8ray-20m | no | 90 | 4 | **0.002** | 0.004 | 0.6× | 87.5° | 0.048 | 0.080 | 0.6× | 48 | 0.89 |
| dw-8ray TOKENS | L-dw-8ray-tok-20m | no | 90 | 8 | **0.002** | 0.004 | 0.5× | 87.3° | 0.070 | 0.124 | 0.6× | 38 | 0.92 |
| dw-blink | L-dw-blink-20m | no | 100 | 6 | **0.026** | 0.022 | 1.2× | 80.6° | 0.047 | 0.052 | 0.9× | 59 | 0.95 |
| oth-uniform | L-oth-20m | YES +0.61 | 18 | 5 | **0.188** | 0.007 | 25.7× | 64.3° | 0.238 | 0.009 | 27.7× | — | 0.38 |
| oth-uniform (MSE) | L-oth-20m-mse | YES +0.68 | 13 | 4 | **0.172** | 0.006 | 30.2× | 65.5° | 0.195 | 0.006 | 35.4× | — | 0.34 |
| oth-adjacent | L-oth-adjacent-20m | no | 52 | 1 | **0.077** | 0.009 | 8.4× | 73.8° | 0.236 | 0.022 | 10.9× | — | 0.39 |
| oth-noflip | L-oth-noflip-20m | no | **0** | — | — | — | — | — | — | — | — | — | — |

`rows` = cos²θ between the true edit direction and the probe read-out subspace (pre-Haufe);
`gen` = the same for an unrelated displacement (empirical chance); `angle` = the equivalent
angle (chance ≈ 85°); `rank90` = directions needed for 90% of Δ's variance across cases.
oth-noflip has NO row because zero of 900 cases admit an exact counterfactual board — colour
there equals the parity of (row + column), so a flipped board is unreachable by any legal game.
Discworld's TOKEN model (frames as tokens, Othello architecture) sits with the other discworld
rows at 0.6×, so the interface does not move alignment; the CE/MSE Othello pair (25.7× / 30.2×)
says the objective does not either.

## Haufe-aligned editing, all three tests (`haufe_edit.py`, `haufe_edit_tokens.py`)

| model | pre-Haufe align (×gen) | post-Haufe (×genH) | canonical best edit | Haufe best edit | fidelity |
|---|---|---|---|---|---|
| oth-uniform | 0.188 (25.7×) | 0.238 (27.7×) | PI +0.608 / ND +0.622 | PI +0.635 / ND +0.631 | 0.23 |
| oth-adjacent | 0.077 (8.4×) | 0.236 (10.9×) | PI −0.053 / ND +0.118 | **PI +0.270 / ND +0.375** | 0.60 |
| dw-noiseless | 0.013 (0.7×) | 0.030 (0.7×) | PI +0.233 @ fid 1.95 | PI +0.259 @ 1.39, guarded +0.186 @ 1.09 | |
| dw-8ray TOKENS | 0.002 (0.5×) | 0.070 (0.6×) | PI +0.004 (frame-set EI) | **PI +0.057** @ fid 0.80 | 0.80 |

The token model is the sharpest test: the largest ABSOLUTE Haufe gain in the table (×32) buys
+0.004 → +0.057 against an unedited floor of −0.755. Its probability columns show why — at
α 60–175 the write drives p(unedited frame) from 0.53 down to 0.012 but lifts p(edited frame)
only to 0.078: the mass leaves one world without arriving at the other, which the Edit Index
correctly reads as ≈ 0.

**Pattern:** Haufe buys editability only where alignment was already ABOVE chance
(oth-adjacent 8.4× → 10.9×, edit +0.12 → +0.375). Where alignment stays below chance (every
discworld variant, 0.5–1.2×), a large absolute gain in overlap buys nothing. Absolute
alignment is not the operative quantity; alignment relative to what any displacement achieves is.
