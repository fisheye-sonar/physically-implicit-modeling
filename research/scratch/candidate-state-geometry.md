# CANDIDATE FINDING — State geometry: low intrinsic dim, strongly curved, tautology-corrected

**Consolidates:** `archive/2026-06-24-manifold-geometry-diagnostic.md` (primary),
geometry content from `archive/2026-06-23-pca-component-position.md`.
**Model/data:** GRU `3_dset3_gru_persistentids_inview_400epochs` (val_loss 0.0236),
`4_fixed_refl_inview`, H=256; visited-state bank 200k of 390k teacher-forced states.
Notebook: `notebooks/experiments/manifold_editing/manifold_geometry_diagnostic.ipynb`.
**Sub-question:** 1 (geometry). **Status:** candidate → `findings/state-geometry.md` (summary already
corrected to this as of 2026-07-08).

## The claim (one line)
The visited-state manifold is a **~5–7-dim, strongly curved** surface embedded in a much fatter linear
hull; the earlier "local off-manifold residual ≈0" was a **projection tautology**, not flatness.

## Key numbers
- **Intrinsic dimension:** TwoNN **5.2**, MLE **6.9** (k-robust 6.7–7.1) — brackets the physical **8
  DOF**. Global PCA hull far fatter: **10 / 38 / 73** dims at 70/90/95% variance (counts the curved
  embedding, not DOF). Local PCA@90% grows 7→22 as k:16→1024 (curvature signature).
- **Curvature:** tangent planes reorient **~56° at nearest-neighbor spacing** (smallest sampled
  separation), saturating ~78° by d≈1.2·‖h‖. The 30° "curvature scale" is reached *below* NN spacing.
- **Local tangent never aligns with global:** principal angle 48° (k=16) → 26° (k=1024), never ~0.
- **Honest local residual never collapses:** ~0.75–0.84 (unedited query) / ~1.0 (held-out neighbor)
  for ALL k; global flat residual of real states 1.75. The geodesic's 0.0002 was reproduced as a
  project-then-remeasure-against-the-just-projected-subspace artifact (crashes ~9× at k=512).

## Why it matters
Fixes what the manifold *is* and retires a misleading number. The curvature is the geometric reason
linear/min-norm probe edits leave the manifold (links directly to editability's curved-embedding
finding). Establishes the honest off-manifold detector (local, leave-out neighborhood).

## Caveats
- Honest local residual floors at ~0.75 (not 0) — undecided whether that floor is intrinsic
  within-patch curvature or probe/PCA noise (probe train MSE 0.54).
- Large-d principal angles saturate (~78°) and partly reflect "two far patches are unrelated"; the
  load-bearing number is the ~56° at *smallest* separation. A cleaner curvature scalar would fit
  angle(d) only in the small-d regime.
- Intrinsic-dim estimators disagree (TwoNN 5.2 / MLE 6.9 / local-PCA 14); the model-free pair is the
  trustworthy read — the spread itself reflects curvature.

## Promotion recommendation
**PROMOTE into `findings/state-geometry.md`** (intrinsic dim + curvature + tautology retraction). The
summary is already corrected; this candidate is the backing detail. High confidence (full bank).
