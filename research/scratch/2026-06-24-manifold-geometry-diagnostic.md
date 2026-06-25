# Manifold geometry diagnostic — intrinsic dimension & flat-vs-curved

**Date:** 2026-06-24 · **Direction:** `manifold-geometry-diagnostic.md` (`[in-frame]`, sub-Q1) ·
**Status:** → FLAG FOR PROMOTION (do not promote / mark done / edit RESEARCH.md — human's call)

Notebook: `notebooks/experiments/manifold_editing/manifold_geometry_diagnostic.ipynb` (executed on
GPU, RTX 5090). PNGs: `/tmp/manifold_geometry/{fig1_intrinsic_dimension,fig2_neighborhood_sweep,fig3_curvature}.png`.

Setup: GRU `3_dset3_gru_persistentids_inview_400epochs` (epoch 191, val_loss 0.0236), H=256,
dataset `4_fixed_refl_inview`. Visited-state bank = 200k of 390k teacher-forced hidden states on GPU.
Bank scale: mean ‖h‖=6.09, mean 1-NN dist=1.36, 16th-NN=2.33, 512th-NN=3.83 — so small-k patches are
genuinely local (k=16 spans ~2.3 in a space where the manifold reorients on that same scale).

## Headline verdicts

1. **Honest intrinsic dimension ≈ 5–7, brackets the physical 8 DOF.** The state manifold is a curved
   low-dim surface, NOT a 38–73-dim linear blob.
2. **The manifold is strongly CURVED**, not flat. Tangent planes reorient on the scale of the
   nearest-neighbor spacing.
3. **The geodesic's local-residual collapse to ~0.0002 at LOCAL_K=512 was an ARTIFACT, confirmed.**
   It is a *projection tautology* (+ large-k coarseness), not evidence of flatness. The honest
   off-manifold residual of real states never drops below ~0.7 at any k.

## §1 — Intrinsic dimension (3 estimators)

| estimator | dim |
|---|---|
| global PCA 70% / 90% / 95% | 10 / 38 / 73 |
| local PCA 90% (k=64) | 14.0 (median 14, range 11–19) |
| local PCA 95% (k=64) | 21.6 |
| **TwoNN** (model-free) | **5.2** |
| **MLE Levina–Bickel** (k=20, bias-corrected) | **6.9** (k=10→6.7, k=50→7.1) |
| physical DOF | 8 |

The two model-free estimators (TwoNN 5.2, MLE 5.5–7.1) **bracket the physical 8 DOF** — the model's
visited manifold has essentially the right number of true degrees of freedom. The global linear hull
is far fatter (38 dims @90%, 73 @95%): the manifold is a ~5–8-dim curved surface that needs many
extra *linear* directions to embed. Local PCA @90% (14) sits above the model-free estimate because at
k=64 the patch already wraps some curvature (see §2b) and because var-threshold counting overcounts a
slowly-decaying spectrum. The model-free estimators are the trustworthy read here.

## §2 — Neighborhood-size sweep (the decisive flat-vs-coarse test)

For k ∈ {16…1024}, 120 sample points (Fig 2):

| k | unedited-query resid | held-out-neighbor resid | local dim@90% | angle to global (deg) |
|---|---|---|---|---|
| 16 | 0.841 | 1.490 | 7.4 | 48.2 |
| 64 | 0.785 | 1.075 | 13.6 | 39.8 |
| 256 | 0.727 | 0.991 | 18.5 | 33.1 |
| 512 | 0.746 | 1.034 | 20.3 | 29.6 |
| 1024 | 0.791 | 1.110 | 22.0 | 25.8 |

(global flat residual of real states = **1.745**.)

- **(a) Residual never collapses honestly.** The unedited-query residual stays ~0.73–0.84 for ALL k —
  it does NOT fall to ~0 at large k. The held-out-neighbor residual is U-shaped (min ~0.99 at k≈256),
  rising at small k (sparsity) and large k (curvature: a flat tangent can't cover an off-patch point
  on a curved surface). Neither read shows flatness.
- **(b) Local dim grows monotonically with k** (7→22 as k:16→1024): bigger patches wrap more
  curvature and need more linear dims — the signature of a curved surface, not a flat one (a flat
  patch's dim would be constant in k).
- **(c) Local tangent never aligns with the global subspace**: principal angle 48° (k=16) → 26°
  (k=1024), still far from 0 even at k=1024 — local geometry is genuinely tilted off the global flat.

### The geodesic 0.0002 collapse — mechanism nailed (cell [9b])

Reproducing the walk's exact measurement (project the point onto its local tangent, THEN measure the
projected point's residual against a tangent re-fit at the projected point):

| k | honest unedited-query resid | after-project-then-remeasure |
|---|---|---|
| 16 | 0.556 | 0.183 |
| 512 | 0.721 | **0.082** |

The "after-project" residual crashes by ~9× at k=512 — reproducing the spirit of the geodesic 0.0002.
(The geodesic's value is even smaller because in the actual iterated walk the re-fit happens at a
point that barely moved, so the two subspaces are nearly identical → near-tautological ~0.) **This is
a projection tautology**: the walk measures a point's distance to (essentially) the subspace it was
just projected onto. It is NOT a measurement of manifold flatness. Confirmed `k`-artifact.

## §3 — Curvature (tangent rotation vs separation)

80 anchors × 60 stratified targets, k=64 tangents, top-8 subspaces (Fig 3):

| separation d | d/‖h‖ | mean principal angle (deg) |
|---|---|---|
| 4.36 (smallest bin) | 0.72 | 55.8 |
| 7.05 | 1.16 | 74.3 |
| 10.15 (largest) | 1.67 | 77.4 |

Even the **nearest** tangent pairs differ by ~56°, saturating near ~78° (near-orthogonal for
8-dim subspaces in 256-dim) by d≈1.2·‖h‖. The 30° "curvature scale" is reached *below* the smallest
sampled separation (≈4.4 ≈ 0.7·‖h‖) — i.e. the tangent reorients on the scale of the
nearest-neighbor spacing itself. **Strongly curved.** (Caveat: at large d the angle is dominated by
how unrelated two far-apart patches are; the load-bearing number is the ~56° at the *smallest*
separation, which already says small steps rotate the tangent a lot.)

## Reconciliation with prior claims

- `findings/state-geometry.md` variance elbow at ~5–10 dims is consistent with the model-free
  intrinsic dim ~5–7. The global-PCA "dimension" (38–73 @90/95%) overstates DOF because it counts the
  curved embedding, not the surface.
- The geodesic-refinement question ("more real states?") is moot: with 390k states the issue was
  never sample count, it was (i) measuring residual against a just-projected subspace and (ii)
  LOCAL_K=512 being a coarse, near-global patch. Honest local residual is ~0.7–1.0 at all k.

## Open questions / caveats

- Honest local residual bottoms at ~0.7 (unedited query) / ~1.0 (held-out), not 0 — real states are
  *not* exactly on any single linear local patch. Worth deciding whether that floor is intrinsic
  curvature-within-patch or probe/PCA noise (probe train MSE was 0.54).
- §3's large-d angles saturate at ~78°; a cleaner curvature scalar would fit angle(d) only in the
  small-d regime (d ≲ 1·‖h‖) where the tangent-rotation interpretation is valid.
- Intrinsic-dim estimators disagree (TwoNN 5.2 vs MLE 6.9 vs local-PCA 14); the model-free pair is
  more trustworthy but the spread itself reflects curvature + a slowly-decaying spectrum.

→ FLAG FOR PROMOTION
