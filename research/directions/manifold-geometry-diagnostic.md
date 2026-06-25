# Direction: Manifold Geometry Diagnostic — intrinsic dimension & flat-vs-curved

**Tag:** `[in-frame]` · **Sub-question:** 1 (geometry) · **Status:** in progress (2026-06-24) ·
**Complexity:** low-medium (one notebook, cheap)

> Run as one notebook: `notebooks/experiments/manifold_editing/manifold_geometry_diagnostic.ipynb`.
> Number every cell (`# [N]`) and figure (`Fig K — …`). Plots AND printed tables. PNGs to
> `/tmp/manifold_geometry/`.

## Motivation

Two loose ends from the geodesic work and the geometry finding:
1. The geodesic's local off-manifold residual **collapsed to ~0.0002** with `LOCAL_K=512` — far
   *below* the real-state local residual (0.87). That is suspicious: it likely means the
   neighborhood is **too large** (local PCA ≈ global PCA, so projection is near-tautological), not
   that the manifold is genuinely flat. We cannot currently tell "flat manifold" from "coarse local
   approximation."
2. The sim's true task DOF is **~8** (`(pos,vel)` × 2 objects). `findings/state-geometry.md` put the
   variance elbow at ~5–10 dims. We should pin the **intrinsic dimension** honestly and check it
   against 8, and characterize **curvature** — both feed sub-Q1 and retire the geodesic-refinement
   question (the answer to "more real states?" is no — we already have 390k; the issue is `k`).

This is a geometry diagnostic, NOT an editing experiment. It also de-risks earlier claims that
leaned on untrustworthy global-PCA residual magnitudes.

## Bootstrap (cold-start)

Same as `geodesic_walk_k150.ipynb`/`pca_component_position.ipynb` (paths 3-deep:
`../../..` for pim, `../../../runs`, `../../../datasets`): `load_checkpoint`,
`load_dataset(n_obj_keep=2)`, `make_test_loader`, `eval.teacher_force` → `states_tf`,
`fit_state_subspace`. Build the visited-state bank (use a **large** subsample — up to the full
390k if memory allows — so small-`k` neighborhoods are genuinely local/dense).

## Section 1 — Intrinsic dimension (3 estimators, cross-checked)

- Global PCA scree recap (cumulative variance; dims for 70/90/95%).
- **Local** PCA spectrum: at many sampled points, eigenvalue spectrum of the `k`-NN patch — how
  many components for 90% *within* a patch (the local intrinsic dim), as a function of `k`.
- A **model-free** estimator (TwoNN or MLE intrinsic-dimension) on the state bank.
- Report all three; compare to the physical DOF **8**. (Expect intrinsic dim < global-PCA dim if
  the manifold is a curved low-dim surface embedded in a higher-dim linear span.)

## Section 2 — Neighborhood-size sweep (flat vs coarse)

For `k ∈ {16, 32, 64, 128, 256, 512, 1024}` (and an absolute-radius variant if convenient), over
many sample points, measure and plot vs `k`:
- (a) **Local off-manifold residual of real states** — does it *rise toward the global ~1.75* as
  `k` grows, and *fall* as `k` shrinks? (If residual is ~0 only at large `k`, the earlier collapse
  was the coarse-approximation artifact.)
- (b) **Local intrinsic dim** (components for 90% within the patch) vs `k`.
- (c) **Principal angles** between the local tangent subspace and the **global** PCA subspace —
  large angles ⇒ the local surface is misaligned with the global flat ⇒ curvature.

## Section 3 — Curvature estimate

How fast does the tangent rotate as you move along the manifold? For pairs of nearby visited
states at separation `d`, compute principal angles between their local tangent subspaces and plot
**angle vs `d`**. Flat ⇒ angles ≈ 0 for all `d`; curved ⇒ angles grow with `d`. Give a single
summary curvature scale (e.g. the `d` at which the mean principal angle hits ~30°).

## Decision / deliverables

- State the **honest intrinsic dimension** and whether the manifold is **~flat or curved**, with
  the numbers. Reconcile with the geodesic's residual-collapse (artifact of large `k`? confirm).
- Dated note `research/scratch/2026-06-24-manifold-geometry-diagnostic.md`, flagged
  `→ FLAG FOR PROMOTION`. Do NOT promote, mark done, or edit `RESEARCH.md`.
- PNGs to `/tmp/manifold_geometry/`. Structured report: intrinsic dim (3 estimators), flat-vs-curved
  verdict + curvature scale, and whether the geodesic residual-collapse was a `k`-artifact.

## Context

- Checkpoint/data as the other manifold_editing notebooks. Primitives: `fit_state_subspace`,
  `fit_local_subspace`, `offmanifold_residual`, `project_to_subspace` (`pim/editors`). Principal
  angles: `scipy.linalg.subspace_angles` or SVD of basis products.
