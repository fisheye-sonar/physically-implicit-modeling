# CANDIDATE FINDING — Editability: the GRU state is predictively sufficient but non-canonical

**Consolidates:** `archive/2026-06-24-canonical-state-editing.md` (keystone),
`archive/2026-06-24-geodesic-walk-k150.md` (subsumed), `archive/2026-06-23-geodesic-walk.md`
(superseded), `archive/2026-06-23-pca-component-position.md` (parked/metric-dependent).
**Model/data:** GRU `3_dset3_gru_persistentids_inview_400epochs` (val_loss 0.0236),
`4_fixed_refl_inview`, 2 obj, constant-velocity, H=256. Notebooks under
`notebooks/experiments/manifold_editing/`.
**Sub-question:** 3 (editability). **Status:** candidate for promotion → `findings/editability.md`
(whose summary is already corrected to this framing as of 2026-07-08).

## The claim (one line)
Targeted latent edits fail not because the target is unreachable on the manifold, but because the GRU
hidden state is **predictively sufficient yet non-canonical**: ~35% of `h` is not a function of the
minimal `(pos,vel)` statistic, the `(pos,vel)→h` embedding is strongly curved, and **the readable code
≠ the controllable code** ("readable ≠ controllable").

## Key numbers
- **Fiber not collapsed (keystone):** best nonlinear `g(pos,vel)→h` leaves **residual 0.347** (R²(h)
  0.859); linear leaves 0.877. Linear→MLP drop **0.53** ⇒ strongly curved embedding. Adding velocity
  to position lifts R²(h) by only +0.06 ⇒ velocity is a minor axis of `h`.
- **Recoverability:** position linear R² **0.84** (MLP 0.96). **Velocity is instantaneously readable
  from a single `h_t` but only NONLINEARLY** (2×2 correction, `diagnostic_corrections`): single-frame
  MLP R² **0.94 late-t** (linear 0.59); 2-frame MLP adds ~nothing (Δ ≤ 0.007 both models); `dh`
  differencing is *worse*. The keystone's "velocity is temporal (0.47 linear → 0.76 2-frame MLP)" was
  a linear-vs-MLP confound — **RETIRED**. Velocity is a nonlinear/entangled coordinate of the state,
  not a temporal feature.
- **Completing the target to `(pos,vel)` does NOT fix editing:** obs change **1.4%** of a swap, ghost
  0.99 — identical to position-only ⇒ kills the velocity-incompleteness hypothesis.
- **Readable ≠ controllable, localized:** obs-gradient objective renders the target at step 0 but lands
  **15.7 off-manifold** / 16.7 from the canonical state and reverts by ~step 4; the probe-objective
  moves the readout exactly but the obs not at all. Same target, opposite outcomes.
- **Target IS reachable on-manifold (geodesic, subsumes the old "curvature barrier"):** the K=30
  plateau was a fractional-step **schedule artifact**; constant-step geodesic reaches readout RMSE
  **0.35** — yet the obs still does not move. The old "strictly on-manifold, local resid 0.0002"
  sub-claim is **retracted** (projection tautology — see candidate-state-geometry).

## Why it matters
This is the sharpest interventional test of "model vs compression." The GRU has the *dimensionality*
of the world state (~6–8) but not its *canonicality* — so no low-dim, on-manifold, single-`h` edit
produces a state that (a) renders the target and (b) persists. Frames the organizing hypothesis
(editability ⟺ canonical, factored, predictively-sufficient state).

## Caveats
- N=64 edits; obs-space numbers are means (per-sample heterogeneity real — some edits do move+persist).
- |v|≈0.05 (tiny) depresses velocity R² in absolute terms; the *relative* story is robust.
- "Canonical state" reference is teacher-forced (soft oracle; GRU's own rollout at a hard teleport is
  imperfect). Fiber-collapse uses the full 200k–390k bank ⇒ robust.
- **Velocity (RESOLVED 2026-07-08):** instantaneously nonlinear, not temporal — "temporal" retired.
- **Open (35% residual):** does it decompose into obs-noise memory vs genuine extra history vs
  legitimate dynamics scaffolding? (feeds dynamics-identifiability).
- **Open (constructive):** can a learned/temporal editor induce clean edits? → `directions/learn-to-edit.md`.

## Promotion recommendation
**PROMOTE as the core of `findings/editability.md`** — the velocity 2×2 has landed (nonlinear-
instantaneous, "temporal" retired), so the summary is now stated correctly. Ready for your read.
