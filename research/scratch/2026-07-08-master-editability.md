# Master Editability notebook — consolidated visual synthesis (GRU + RSSM) (2026-07-08)

→ FLAG FOR PROMOTION *(pointer only)*: this is the **consolidated visual synthesis** of the
editability / canonical-state pillar. It introduces **no new science**; the promotion decisions
live in the `candidate-*.md` files (`candidate-editability.md`, `candidate-rssm-replication.md`,
`candidate-state-geometry.md`, `candidate-predictive-quality.md`). Do not promote from this note.

**Notebook:** `notebooks/experiments/manifold_editing/00_master_editability.ipynb` (executed on GPU,
numbered cells `# [N]`, numbered figures `Fig K`). **PNGs:** `/tmp/master_editability/`.

**Models/data:** GRU `runs/gru/3_dset3_gru_persistentids_inview_400epochs` (H=256, val_loss 0.0236),
refined RSSM `runs/rssm/4_dset4_refined_best` (det256+stoch64=320, `sample=False`),
`datasets/4_fixed_refl_inview`, n_obj_keep=2, teacher-forced test set. Velocities from HDF5
`velocities` (temporal std ≈1.3e-8 → constant-velocity sim confirmed).

## What this notebook is
ONE presentation-grade walkthrough structured like a paper: §0 premise → §1 geometry → §2
recoverability → §3 canonicality/fiber → §4 editing head-to-head (centrepiece) → §5 synthesis.
GRU + refined RSSM side by side in each section where applicable. Cheap artifacts (states, probes,
edits, waterfalls) recomputed cleanly; expensive ones (intrinsic dim on the 200k bank, RSSM
curvature) **cited** from the source notebooks. All probes are in-sample fit — comparisons are
load-bearing, absolute R² optimistic (stated once in the notebook).

## Corrected numbers used (2026-07-08 pass) — the reason this consolidation exists
- **Velocity is nonlinear-INSTANTANEOUS, not temporal.** Single-frame MLP ≈ 2-frame MLP (Δ ≤ 0.015
  all-t, ≤ 0.007 late-t on both models); `dh` differencing is *worse*. The old "temporal 0.47→0.76"
  reading was a linear-vs-MLP confound → **RETIRED**. (§2)
- **RSSM det-only fiber residual ≈ GRU.** det-only ≈0.368 ≈ GRU ≈0.337; the full-320 ≈0.602 was
  inflated by the stochastic `s` (≈0.891). Do NOT call the RSSM "less canonical." (§3)
- **Position lives in the deterministic `h`,** not the stochastic `s`. (§2/§3)

## Sections (headline per section)
- **§0 Premise:** constant-velocity sim ⇒ minimal sufficient statistic `(pos,vel)` = 8-dim.
- **§1 Geometry:** intrinsic dim ≈5–7 (TwoNN 5.2 / MLE 6.9, cited) brackets 8 DOF; fat linear hull
  (recomputed scree GRU ~38/256, RSSM ~35/320 @90%); strongly curved (tangent ~56°/65° at NN
  spacing, cited). Fig 1.
- **§2 Recoverability:** position ~linear (lin ~0.84 / MLP ~0.96); velocity 2×2 (nonlinear-
  instantaneous); RSSM position split det≈full≫s. Fig 2.
- **§3 Canonicality:** GRU h ~34% non-canonical (fiber resid ≈0.337); RSSM det ≈0.368 ≈ GRU; s holds
  none (≈0.891). KL buys no canonicity. Fig 3.
- **§4 Editing head-to-head (CENTREPIECE):** five editors — GT / Unsteered / Manifold-global /
  PCA-geodesic (iterative local-tangent walk) / MLP-gradient (obs-driven Adam). Unified dark-theme
  waterfall overlay (green=target, red=ghost) + metrics table (→target, obs-change vs GT-swap %,
  ghost, per-step persistence, honest leave-out off-manifold residual). **Reversion example:** the
  MLP-gradient edit reaches target at step 0 and reverts by ~step 4 (Fig 6). RSSM echo confirms
  readable≠controllable is architecture-independent (§4b). Figs 4–6.
- **§5 Synthesis:** predictively-sufficient but non-canonical; curved `(pos,vel)→h`; velocity
  nonlinear-instantaneous; readable≠controllable; architecture-independent (RSSM replicates, KL
  delivers no canonicity/controllability). Frames — as *hypothesis* — editability ⟺ canonical,
  factored, predictively-sufficient state. Fig 7.

## Caveats preserved (honest)
- "obs-change % of a full swap" small-k geodesic uses a weak pseudoinverse denominator; the notebook
  additionally prints a **teacher-forced GT true-post-edit** 100% reference for a proper baseline.
- Naming kept distinct: "local-tangent projection" (one-shot) ≠ "PCA geodesic" (iterative walk).
- Recomputed RSSM scree @90% here ≈35 vs cited 34 (within rounding of a different bank subsample) —
  not a disagreement, just sampling. Any numbers that differed from source notebooks are flagged in
  the notebook's §5 auto-summary; the load-bearing corrected values (velocity 2×2, det-only fiber)
  reproduced the `diagnostic_corrections` results.

## Open
None new — consolidation only. Feeds the pending promotion of `findings/editability.md`,
`findings/state-geometry.md`, and a new architecture-independence finding (see `candidate-*.md`).
