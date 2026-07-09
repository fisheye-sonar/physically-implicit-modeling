# Finding: State Geometry

*Sub-question 1 — where does the learned state live?*
Model/data context unless noted: GRU `3_dset3_gru_persistentids_inview_400epochs`,
dataset `4_fixed_refl_inview`, 2 objects, fixed reflectivities.

> **Scope (preliminary, 2026-07-09).** Characterizes the visited-state manifold of *this specific
> checkpoint* — a GRU trained purely for next-observation prediction on `dataset 4`. Not a general
> claim about GRU/world-model geometry; a different objective, dataset, or scale could shift the
> numbers. Read "the GRU" as "this pure-next-step-prediction GRU."

## Current understanding (mutable summary)

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
