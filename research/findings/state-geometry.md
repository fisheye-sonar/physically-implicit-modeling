# Finding: State Geometry

*Sub-question 1 — where does the learned state live?*
Model/data context unless noted: GRU `3_dset3_gru_persistentids_inview_400epochs`,
dataset `4_fixed_refl_inview`, 2 objects, fixed reflectivities.

## Current understanding (mutable summary)

The GRU's visited hidden states (256-dim) occupy a **low-dimensional, curved
manifold**. ~38 dims carry 90% of variance with a sharp elbow at ~5–10 dims
(~70%). The model uses roughly 15% of its representational capacity for the core
dynamics. The manifold's **curvature matters**: a flat global PCA subspace and a
local tangent subspace point in different directions, so a global-PCA off-manifold
residual is *blind* to edits that stay in the kept subspace but leave the curved
surface.

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
