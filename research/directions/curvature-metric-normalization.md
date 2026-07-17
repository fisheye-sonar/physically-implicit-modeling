# Direction: Fix the tangent-rotation "curvature" metric (scale/density normalization)

**Tag:** `[in-frame]` · **Sub-question:** 1 (geometry) · **Status:** proposed (Sevan: address when back —
NOT now; needs more thought) · **Complexity:** low-medium (metric redesign + sweep across notebooks)

## The problem (Sevan flagged 2026-07-16; confirmed by code read)
Our "curvature" number is `tangent_rotation`: sample anchors; at each, fit local PCA on its `k` nearest
neighbours → a top-`n` tangent hyperplane; do the same at the anchor's single nearest neighbour; take the
**mean principal angle** between the two tangents; average over anchors. 0° = flat.

**It is NOT normalized by the step distance to that neighbour.** So the absolute degrees depend on:
- **sample density** — more states → closer nearest neighbours → smaller per-step rotation → smaller angle
  (this is why `00_master_editability` reports ~56° while the newer multistep/action notebooks report ~20°:
  a density/parameterization artifact, *not* a real geometry difference);
- **latent scale** — different architectures (GRU vs RSSM) have different `‖h‖`, so "degrees per nearest-
  neighbour step" is not comparable across models either.

So "average rotation of 20°" is uninterpretable in the abstract and not comparable across notebooks/models.

## Proposed fix (to discuss — Sevan wants to think about it)
Make it scale-free. Candidate options, in rough order of preference:
1. **Excess rotation over a matched-density flat surrogate.** Sample the same number of points from a flat
   (linear/Gaussian) manifold at the *same* local density and report `angle_observed − angle_flat` (finite-
   sample noise gives a nonzero flat baseline). Scale-free and density-controlled.
2. **Rotation per unit arc length.** `principal_angle / nearest-neighbour distance`, i.e. an estimate of
   local extrinsic curvature (radians per unit latent distance). Removes density but still carries latent
   scale → normalize the distance by the manifold's local diameter / mean pairwise distance to also kill scale.
3. **Rotation-vs-neighbour-rank slope.** Plot principal angle as a function of neighbour rank/distance; the
   slope near 0 is a density-free curvature estimate.

## Scope of the fix
- Replace the metric in the geometry sections of the newer notebooks
  (`multistep_objective_structure.ipynb`, `action_conditioned_structure.ipynb`) and, when Sevan re-opens it,
  `00_master_editability.ipynb` (the 56° there is the same un-normalized number). **Do NOT edit the master
  notebook without Sevan's explicit go** (standing constraint).
- Until fixed: any reported tangent-rotation degrees are only comparable **within one notebook at fixed
  sample density**; state the bank size + `k` used and never compare absolute degrees across notebooks/models.

## Note
This does NOT change the current findings' conclusions — intrinsic dim (TwoNN/MLE) and the linear-hull
numbers are the load-bearing geometry quantities; curvature was a secondary descriptor. The fix is about
making the curvature descriptor honest and cross-comparable.
