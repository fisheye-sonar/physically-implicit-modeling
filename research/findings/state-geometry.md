# Finding: State Geometry

*Sub-question 1 — where does the learned state live?*
Model/data context unless noted: GRU `3_dset3_gru_persistentids_inview_400epochs`,
dataset `4_fixed_refl_inview`, 2 objects, fixed reflectivities.

> **Scope (preliminary, 2026-07-09).** Characterizes the visited-state manifold of *this specific
> checkpoint* — a GRU trained purely for next-observation prediction on `dataset 4`. Not a general
> claim about GRU/world-model geometry; a different objective, dataset, or scale could shift the
> numbers. Read "the GRU" as "this pure-next-step-prediction GRU."

## Current understanding

> **Updated 2026-08-21.** Three additions (2026-08-17) plus one qualification.
> (1) The **linear position code spans ~116 of 256 dimensions**, not 4 — smeared, with a gradual decay that makes any single dimensionality
> number threshold-dependent (2026-08-05). (2) **Canonicality gets worse with capacity** — the
> MLP fiber residual rises monotonically in all four model families, so readability and
> canonicality move in opposite directions (2026-08-13). (3) The `Δh` of a *working* edit is
> large, mutually distinct across edits, and probe-orthogonal — the geometric form of the
> reachability ceiling (2026-08-03). Also recorded: the object-superposition evidence was a
> **decoder artifact** on affine decoders, and the result survives re-measurement on nonlinear
> ones (2026-08-05) — but **only in a weak form**: a randomly initialised network of the same
> config matches the composed cosine, so what training buys is that the linear model's latent
> tracks the *observation's own* additivity ceiling, and nothing more (2026-08-21).

### Previous synthesis (mutable summary)

The GRU's visited hidden states (256-dim) occupy a **low-dimensional, strongly
curved manifold**. The **honest intrinsic dimension is ~5–7** (TwoNN 5.2, MLE 6.9),
which brackets the physical 8 DOF; the fatter global-PCA hull (10/38/73 dims at
70/90/95% variance) counts the *curved embedding*, not the true degrees of freedom.
The manifold is genuinely curved: local tangent planes reorient by **~56° at
nearest-neighbor spacing** and never align with the global PCA subspace (principal
angle 48°→26° across k, never ~0). Because of this curvature a global-PCA
off-manifold residual is *blind* to edits that stay in the kept subspace but leave
the curved surface; the honest detector is the local residual against each state's
own neighborhood — which for real states **floors at ~0.75–0.84 and never collapses
to 0** (an earlier "local resid ≈0" was a projection tautology; see log).

## Log

### 2026-08-21 — Latent object-composition is mostly architectural; the strong form does not survive a random-init baseline · `replicated` · **qualifies 2026-08-05**

**Evidence:** `scratch/2026-08-21-composition-random-baseline.md` ·
`notebooks/experiments/editability/latent_linearity/random_baseline.ipynb` (+ `composition_lib.py`)
· dataset 4, N=256 real teleport edits from `edits.h5`, four displacement scales, two random seeds.

**Qualifies the 2026-08-05 entry below**, which cleared the *decoder* confound but never ran the
*training* control. Prompted by Sevan asking whether the composition result holds for an
**untrained** model. It largely does. Additivity is a first-order Taylor property of any smooth map, so random weights —
which give a smooth *deterministic* function, not a random one — have a large architectural floor.

**By composed cosine, training is worth nothing.** At the real teleport scale: trained linear
**+0.904** vs randomly initialised at the identical config **+0.890**; trained nonlinear **+0.835**
vs random **+0.853** — the untrained net is *higher*. Training moves the metric +0.014 on one
family and −0.018 on the other. The measurement is anchored: trained linear reproduces §7's
**+0.873** and its `‖composed‖/‖direct‖` **1.13**.

**Read against the renderer's own non-additivity, training does separate — on the linear family.**
The two objects share rays, so the observation itself is not additive; its relative residual is
**0.406 / 0.373 / 0.285 / 0.207** across scales 1.0 → 0.125. *Excess* over that ceiling:

| scale | 1.0 | 0.5 | 0.25 | 0.125 |
|---|---|---|---|---|
| trained linear | **+0.046** | **−0.004** | **−0.019** | **+0.004** |
| random linear | +0.104 | +0.064 | +0.037 | +0.028 |
| trained nonlinear | +0.241 | +0.138 | +0.077 | +0.073 |
| random nonlinear | +0.208 | +0.150 | +0.121 | +0.109 |

**Why it matters:** the claim survives only in a restated, much weaker form — *the trained linear
model's latent is as additive as the observation it is trained to predict, and no less*. That is a
**ceiling-tracking** claim, not a superposition claim, and the nonlinear family does not support
even that at the real edit scale. The strong reading — "the latent superposes object edits, and
that is learned" — is dead. Three flaws in the first pass had to be fixed to get here (uniform
displacement direction, a shuffled floor that permuted only one delta, and measuring displacement
from `positions[ef]` where the teleport is already in the data); all three made the effect look
larger.

---


### 2026-08-13 — Canonicality gets *worse* with capacity · `replicated`

**Evidence:** `scratch/2026-08-13-action-hidden-size.md` ·
`notebooks/experiments/editability/action_hidden_size/` · four model families ×
`H ∈ {8,32,128,256,512}`.

MLP fiber residual **rises** with capacity in all four families — passive 0.288 → 0.637 ·
exogenous-actions-given 0.410 → 0.695 · exogenous-observer 0.317 → 0.710 · endogenous
0.270 → 0.500.

**Why it matters:** capacity moves readability and canonicality in **opposite** directions. A
bigger latent is more linearly readable and *less* a function of the world's minimal sufficient
statistic. Any hope that scale alone produces a canonical state is measurably wrong here.

---

### 2026-08-05 — Compositionality is real on a nonlinear decoder; the earlier evidence was a decoder artifact · `replicated`

**Evidence:** `scratch/2026-08-05-nonlinear-gru-decoder.md` ·
`notebooks/experiments/editability/nonlinear_gru/nonlinear_gru_findings.ipynb`
(+ `NONLINEAR_GRU_RUNS.md`) · dataset 4, N=256 edits (§3–§4), 67 in-frustum samples (§5).

Prompted by Sevan's question about `delta_h_analysis` §7: *isn't the object-superposition
finding just an artifact of a linear decoder?* It is —
`decode(h0+d1+d2) = decode(h0+d1) + decode(h0+d2) − decode(h0)` holds **identically** for any
affine decoder.

**The artifact is confirmed as a statement about the *evidence*, not the *result*.** On both
affine-decoder models the composed decode equals the affine prediction to **6.6e-08 / 9.0e-08** —
machine precision. Their composed Edit Index (+0.46) is algebraically determined and could not
have falsified anything whatever it read. It also sits at 90% of the **+0.51** ceiling that the
*model-free* render identity (`GT_A + GT_B − GT_BASE` vs `GT_AB`) scores by itself — an identity
that is leaky rather than exact (RMSE 0.177 against an RMS signal of 0.306, because the two
objects share rays in **42%** of samples), which is why the ceiling is +0.51 and not +1.0.

**On the nonlinear models compositionality is real, object-specific, and holds.** Their decoders
depart from affine by 8.5e-02…8.8e-02 — about half their own total error — so the composed index
is a genuine measurement, and it survives against the proper null models (unedited; random Δ at
matched norm; composed with the **wrong** object).

**Why it matters:** a textbook case of a result that was true while its evidence was vacuous.
The correct response was to re-measure on a decoder where the identity does not hold for free,
not to withdraw the claim.


---

### 2026-08-05 — The linear position code spans 116 dimensions · `established`

**Evidence:** `scratch/2026-08-05-iterative-probing-position-dimensionality.md` ·
`notebooks/experiments/editability/iterative_probing/` · GRU `runs/controls/H256` · 78,000
aligned states from 2,000 sequences, split **by sequence**.

Iteratively fitting a linear position probe and projecting its rank-4 row space out of `h`:
**29 probes → 116 dimensions**, with rank and orthogonality asserted at every step. Half the
readability is gone by 24 dimensions, with a long thin tail out to 112. The `4 × #probes`
arithmetic holds because `lstsq` returns the minimum-norm solution, so each new probe's rows land
inside the row space of the already-deflated design matrix.

**Why it matters for geometry:** the position information is not a compact 4-dimensional
subspace. It is smeared across nearly half the latent, with a gradual decay that makes any single
"dimensionality of position" number threshold-dependent — report the curve, not a scalar. The
editing consequence is in `editability.md`.

---

### 2026-08-03 — Δh geometry: successful edits are large, mutually distinct, and probe-invisible · `established`

**Evidence:** `scratch/2026-08-03-delta-h-analysis.md` ·
`notebooks/experiments/editability/delta_h_analysis.ipynb` · GRU `runs/controls/H256` + RSSM
`runs/rssm/4_dset4_refined_best`, **N=256** held-out edits, cosines and fractions computed
**per instance, then averaged**.

The displacement `Δh` that a *working* (oracle) edit applies is roughly as large as the state
itself, essentially orthogonal to everything the position probe can see or move, and different
for every edit — including for edits that make the *same* positional change. Two independent
oracles nevertheless agree strongly on which direction it is.

**Why it matters for geometry:** the set of state changes corresponding to one semantic change is
not a low-dimensional, position-indexed manifold direction. It is state-dependent in a way no
global linear rule captures — which is the geometric statement of the reachability ceiling.

**Caveat:** RSSM alignment is ambiguous at this precision (k=−1 0.1059 vs k=0 0.1067) because its
prior decode is blurry; the GRU numbers are the clean ones.

---

### 2026-06-24 — Local off-manifold residual ≈0 was a projection tautology. `established`
The geodesic's "local resid 0.0002" measured a point's distance to the subspace it had just been projected onto; the honest local residual of real states never collapses (~0.75–0.84 across all k). Real states do not lie on any single linear local patch. Intrinsic dimension (TwoNN 5.2, MLE 6.9) brackets the physical 8 DOF; the 38–73-dim PCA hull reflects a strongly curved embedding (tangents rotate ~56° at NN spacing), not true DOF.

### 2026-06-23 — PCA spectrum of visited states · `established`
Teacher-forced the full test set → 10000×39×256 hidden states. PCA on visited
states: **38/256 components for 90% variance**, sharp elbow at ~5–10 components
(~70%). Off-manifold residual of real states: mean 1.75, p95 2.16.
*Implication:* the state manifold is real and low-dimensional; capacity is
largely unused, consistent with a compact world-state code.

### 2026-06-23 — Global PCA residual is curvature-blind · `established`
A min-norm (pseudoinverse) edit's global-PCA off-manifold residual ≈ that of real
states, because the edit moves *within* the kept high-variance subspace. Projecting
onto a **local tangent-PCA** patch can move a state *further* from the global flat
subspace than not projecting at all — direct evidence the manifold is curved and
the two subspaces are misaligned. The honest off-manifold detector is the *local*
residual against each state's own neighborhood, not the global one.
