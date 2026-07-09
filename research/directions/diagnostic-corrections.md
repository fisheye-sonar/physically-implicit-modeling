# Direction: Diagnostic Corrections — velocity 2×2, det-only fiber, honest small-k geodesic

**Tag:** `[in-frame]` · **Sub-question:** 1 (geometry) + 2 (identifiability) + 3 (editability) ·
**Status:** in progress (2026-07-08) · **Complexity:** low-medium (one notebook, three self-contained
corrections that share one bootstrap)

> Run as ONE notebook: `notebooks/experiments/editability/diagnostic_corrections.ipynb`
> (use the **NotebookEdit** tool). Number every cell (`# [N]`) and figure (`Fig K — …`, panels
> `(a)/(b)/(c)`). Produce BOTH rich plots AND printed metric tables. Export PNGs to
> `/tmp/diagnostic_corrections/`.

## Why this exists

Three specific numbers in the existing GRU + RSSM notes are either **confounded** or **untrustworthy**,
and each gates a promotion decision. This is a corrections pass, NOT a new experiment. Fix all three
honestly; if a fix overturns a prior claim, say so plainly — do not soften.

1. **Velocity "is temporal" may be a confound.** The GRU keystone compared single-frame **linear**
   (R² 0.47) against 2-frame **MLP** (0.76) — changing *two* things at once (linear→MLP AND
   single→temporal). The RSSM note already shows single-frame **MLP** 0.69 ≈ 2-frame **MLP** 0.695 —
   i.e. on the RSSM the temporal window adds almost nothing once you allow nonlinearity, suggesting
   velocity is **instantaneously readable, just nonlinearly**. We never ran single-frame-MLP on the
   GRU. Resolve with the full 2×2.
2. **RSSM fiber-collapse residual (0.605) is not apples-to-apples with the GRU (0.347)** — it was
   measured on the full 320-d state including the 64-d KL-regularized stochastic `s`, which
   legitimately is NOT a function of `(pos,vel)`. Refit against the deterministic `h` only.
3. **The geodesic's on-manifold claim used LOCAL_K=512** (≈ global PCA → projection tautology, local
   residual collapsed to ~0). Redo with honest small `k` and report the honest residual.

## Bootstrap (cold-start — run from a fresh kernel)

Notebook is **3 levels deep** (`notebooks/experiments/editability/`):
`sys.path.insert(0,"../../..")` (repo root, `import pim`) and `"../.."` (for `helpers`).
Working references to mirror: `canonical_state_editing.ipynb`, `geodesic_walk_k150.ipynb` (GRU) and
`../rssm_structure/rssm_state_geometry.ipynb` (RSSM).

- **GRU:** `CHECKPOINT="../../../runs/gru/3_dset3_gru_persistentids_inview_400epochs/best_model.pt"`.
  Flat state = `h` (256).
- **RSSM:** `CHECKPOINT="../../../runs/rssm/4_dset4_refined_best/best_model.pt"`, `model.sample=False`
  (prior-mean, deterministic). Flat state = `cat([h_det(256), s_stoch(64)])` = 320; keep the
  det/stoch split indices so Section 2 can slice `h_det`.
- **Data:** `DATA_DIR="../../../datasets/4_fixed_refl_inview"`, `load_dataset(DATA_DIR, n_obj_keep=2)`.
- Teacher-force the test set → `states_tf` (GRU 10000×39×256; RSSM 10000×39×320, posterior-mean).
- **Velocities:** `_load_h5_dataset` does NOT load them. Read directly:
  `import h5py; v = h5py.File(test.h5_path)["velocities"][:, :, :2, :]`, aligned like positions
  (`test.positions[:, :-1, :2, :]`). Verify temporal std ≈ 0 (constant-velocity sim).

## Section 1 — Velocity 2×2 {linear, MLP} × {single-frame, 2-frame}, BOTH models  [sub-Q2]

Target = 4-dim velocity `(vx,vy)×2 objects`. For **each model** fit four probes and report R²
(overall + per-component) vs a predict-the-mean baseline:

| feature | linear | MLP |
|---|---|---|
| single-frame `h_t` | (a) | (b) |
| 2-frame `[h_{t-1}, h_t]` | (c) | (d) |

Also report the **`dh = h_t − h_{t-1}`** MLP as a secondary temporal feature. Run on **all-t** AND a
**late-t subset (t ≥ 15)** — early frames underdetermine velocity, so late-t is the fair test (this
also folds in the previously-queued "velocity probe check").

**Decision rule (state it explicitly in the notebook):**
- If **single-frame MLP ≈ 2-frame MLP** (Δ R² ≲ 0.03): velocity is **instantaneously readable,
  nonlinearly** → the "velocity is a temporal feature, not a snapshot coordinate" claim is
  **RETIRED/reframed**. This is the headline outcome to check.
- If **2-frame MLP ≫ single-frame MLP**: the temporal claim survives, and the earlier 0.47-vs-0.76
  gap was partly linear→MLP but temporal info genuinely adds. Report the size of each contribution.

## Section 2 — Det-only fiber refit, RSSM  [sub-Q2]

Refit `g(pos,vel) → h_det` on the **deterministic 256-d block only** (linear + MLP). Report residual
fraction `‖h_det − g‖/‖h_det‖` and R²(h_det). Compare three numbers head-to-head in one table:
GRU `h` (0.347), RSSM full-320 (0.605), **RSSM det-only (new)**. Also fit `g(pos,vel)→s` (stochastic
64) separately to show how much of the 0.605 was the `s` block. **Decision:** is the RSSM's
deterministic block as canonical as the GRU's `h`, or still less canonical (more curved)? Predicted
~0.4–0.5 but likely ≥ GRU — confirm or refute with the number.

## Section 3 — Honest small-k geodesic, GRU (RSSM if cheap)  [sub-Q1/3]

Redo the geodesic walk toward the position-probe target with **constant step** (not fractional — the
K=150 note showed fractional steps decay geometrically and fake a "barrier") at **LOCAL_K ∈ {16, 32,
64}** (contrast with the old 512). At each `k` report:
- Does the readout reach the target (final readout RMSE)?
- **Honest local off-manifold residual** along the walk — computed with a **leave-out neighborhood**
  (fit the local tangent on the k neighbors *excluding* the query point, then measure the point's
  residual) so it is NOT the projection tautology. Expect it to stay ~0.75–0.84, NOT collapse to ~0.
- Does the observation move (obs change vs unsteered, % of a full-state-swap)?

Produce the **1D-line waterfall viz** (green = target loc, red = ghost/original) for ~3 samples, in the
clean style of `geodesic_walk_k150.ipynb` — these panels feed the later unified-comparison notebook.

## Deliverables (HARD REQUIREMENTS)

- Executed notebook (GPU, numbered cells/figures), plots + printed tables for all three sections.
- PNGs to `/tmp/diagnostic_corrections/`.
- Dated note `research/scratch/2026-07-08-diagnostic-corrections.md`, flagged `→ FLAG FOR PROMOTION`:
  the velocity 2×2 tables + verdict (is "temporal" retired?), the det-only fiber number vs GRU/full,
  the honest small-k geodesic residual + reachability, caveats. **Do NOT** promote to `findings/`,
  mark this direction done, or edit `RESEARCH.md`.
- Tight structured report back: headline per section, key numbers, PNG paths.

## Context

- Primitives: `pim/extractors` (`LinearExtractor`, `StateDefinition` + an MLP probe head),
  `pim/editors` (`fit_state_subspace`, `fit_local_subspace`, `offmanifold_residual`,
  `manifold_steer_local`, `inject_state`). Obs-space metrics + waterfalls: copy from
  `geodesic_walk_k150.ipynb`. Model implements `HiddenStateModel`
  (`flat_state`/`state_from_flat`/`decode`/`observe_sequence`/`predict_step`).
