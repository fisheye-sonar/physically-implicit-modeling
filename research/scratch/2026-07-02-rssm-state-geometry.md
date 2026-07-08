# RSSM State Geometry & Editability — GRU replication on the refined RSSM (2026-07-02)

**Direction:** RSSM structure replication of the GRU state-geometry / editability analysis.
**Notebook:** `notebooks/experiments/rssm_structure/rssm_state_geometry.ipynb` (executed clean top-to-bottom on GPU, RTX 5090; 0 error cells).
**PNGs:** `/tmp/rssm_state_geometry/fig1..fig8` (fig7 = 3 waterfall panels: `fig7_waterfall_0/1/2`).
**Model/data:** `runs/rssm/4_dset4_refined_best/best_model.pt` (RSSMModel, epoch 500, val_loss 3.0237), dataset `datasets/4_fixed_refl_inview` (2 obj, fixed refl, constant velocity, dt=1). Flat state = `cat([h_det(256), s_stoch(64)])` → **H=320**. `model.sample=False` (deterministic prior-mean rollouts). `states_tf=(10000,39,320)` posterior-mean teacher-forced. Velocities read from HDF5 `velocities`, temporal std 0.0 (verified constant-vel).

→ **FLAG FOR PROMOTION** (do not promote to findings/, do not mark direction done, do not edit RESEARCH.md — human's call).

---

## HEADLINE VERDICT

**The GRU story REPLICATES on the RSSM — and in the crucial respects the RSSM is *worse*, not better.** The structured latent (256 det + 64 stoch, KL-regularized prior) does **not** produce a more canonical or more editable representation. Same failure mode: low-dim curved manifold, position linearly readable, velocity only temporally readable, `(pos,vel)→state` strongly non-canonical, and readable ≠ controllable. The KL/prior regularization did **not** buy canonicity.

1. **Geometry replicates (RSSM slightly more compact + more curved).** Full-320 state keeps **34/320 dims @90%** (GRU 38/256), elbow ~12 @70% (GRU ~8). Tangent→global angle **65.2°** (GRU ~56°) → *more* curved, not less.

2. **Position lives in the DETERMINISTIC `h`, not the stochastic `s`.** Linear pos R²: full **0.857**, **det-only 0.841**, stoch-only **0.594**. det alone ≈ full; the stochastic `s` carries only a weak/partial position readout. **The theory-predicted "s captures the compact world state" does NOT hold — the GRU-cell pushes position into `h`.** MLP full lifts pos to 0.933. This matches GRU (lin 0.84 / MLP 0.96) closely.

3. **Velocity is temporal, same as GRU.** Single-frame lin vel R² **0.43** (GRU 0.47); best recovery = MLP on 2-frame window **0.69** (GRU 0.76). dh linear is bad (0.10). Same "velocity is implicit in the change of state, not an instantaneous coordinate" story.

4. **Fiber is MORE collapsed-away (LESS canonical) than the GRU.** Best `g_MLP(pos,vel)→state` leaves residual fraction **0.605** (GRU 0.347) and explains only **R²(state)=0.59** of the state. i.e. **~60% of the RSSM flat state is NOT a function of the minimal (pos,vel) sufficient statistic** vs ~35% for the GRU. **CAVEAT (important, see below): this is inflated by the 64-dim stochastic `s`, which is a KL-regularized latent carrying prior/uncertainty structure that is legitimately not a function of (pos,vel). This number is on the full 320-d state and is NOT apples-to-apples with the GRU's 256-d `h`.** Treat the *magnitude* as shaky; the *direction* (non-canonical, curved: lin→MLP residual drop 0.90→0.60) replicates.

5. **Editing replicates the GRU failure — and "readable ≠ controllable" is SHARPER on the RSSM.**
   - Manifold (global-PCA) edit: obs change **36.5% of a full-state swap** (GRU 37%) — nearly identical. But it *increases* obs error vs the true post-edit GT (Fig 4d green rises) → it moves the obs by *scrambling / wrong direction*, not by relocating the object to the target. Waterfalls (Fig 7) confirm: streak shifts but does not cleanly land on the GT target; ghost/curved-embedding artifact identical to GRU.
   - **Pseudoinverse edit: readout RMSE 0.000 (probe perfectly on target) yet obs change ≈ 0.0001 = 0.0% of swap** (GRU was 9.5%). The edit is read perfectly by the probe and rendered *invisibly* by the decoder, and reverts to the unsteered trajectory by rollout step 1 (Fig 4b: orange snaps to target at step 0, back to unsteered at step 1). **This is the strongest "readable ≠ controllable" instance in either model.**

6. **Generative sensitivity: probe direction is the LEAST generative per unit norm.** At matched ‖Δh‖, PCA#1/#2 ≫ random > det-only-probe ≫ full-probe≈0 (Fig 6b). The position-probe direction is an extremely low-variance, low-sensitivity axis: **σ(probe obj0-x)=0.017** vs PCA#1 σ=2.79 (~165×), and vs GRU probe σ=0.26. Even at matched absolute magnitude the probe barely moves the obs. This is the mechanistic reason the pseudoinv edit renders invisibly. **NOTE this DIVERGES from the GRU finding** where matched-magnitude probe *beat* PCA/random — on the RSSM the probe direction is decoder-inert even at matched norm.

---

## GRU vs RSSM — numbers table

| Metric | GRU | RSSM (refined) | verdict |
|---|---|---|---|
| H (total) | 256 | 320 (256 det + 64 stoch) | — |
| k @90% var | 38 | **34** | similar / more compact |
| k @70% var | ~8 | **12** | similar |
| real-state global resid | 1.75 | **2.86** | (scale differs: 320-d) |
| tangent→global angle | ~56° | **65.2°** | more curved |
| pos R² linear (full) | 0.84 | **0.857** | replicates |
| pos R² MLP (full) | 0.96 | **0.933** | replicates |
| pos R² linear **det-only** | — | **0.841** | pos lives in `h` |
| pos R² linear **stoch-only** | — | **0.594** | `s` weak for pos |
| vel R² single-h linear | 0.47 | **0.431** | replicates |
| vel R² [h-1,h] MLP | 0.76 | **0.695** | replicates |
| fiber resid g_MLP(pos) | — | **0.642** | non-canonical |
| fiber resid g_MLP(pos,vel) | 0.347 | **0.605** | worse (caveat: incl. `s`) |
| pseudoinv obs change (%swap) | 9.5% | **0.0%** | readable≠controllable sharper |
| manifold obs change (%swap) | 37% | **36.5%** | replicates |
| σ(probe obj0-x) | 0.26 | **0.017** | probe dir near-inert |

---

## Per-section findings

**Geometry (Fig 1).** RSSM full-320 state: 34/320 @90%, 12 @70%, real-state global off-manifold resid 2.86 (p95 3.63), tangent angle 65.2°. Component split: det-256 keeps 42 @90% (resid 2.26), stoch-64 keeps 6 @90% (resid 1.09) — the stochastic block is *very* low-rank (6 dims carry 90%), consistent with it being a compressed KL-regularized code. Curvature is real and slightly stronger than the GRU.

**Recoverability (Fig 2).** Position: full lin 0.857 / MLP 0.933; **det-only lin 0.841 ≈ full → position concentrates in the deterministic h**; stoch-only lin 0.594. Velocity single-frame lin 0.43 / MLP 0.69; temporal dh lin 0.10 / MLP 0.54; 2-frame window lin 0.47 / MLP 0.695 (best). Same temporal-velocity signature as GRU.

**Fiber collapse (Fig 3).** g(pos) lin resid 0.903 / MLP 0.642; g(pos,vel) lin 0.900 / MLP 0.605. Adding velocity to position barely helps (MLP 0.642→0.605), same as GRU (velocity is a minor axis of the state). Strong lin→MLP drop = curved embedding. **Magnitude shaky (inflated by stochastic `s`); direction robust.**

**Editing (Figs 4–5, 7).** Diagnostic table (readout RMSE / global resid / local resid): unsteered 1.758/2.840/1.362; pseudoinv **0.000**/2.840/1.362; manifold 1.747/**0.000**/1.851; local 1.744/2.470/**0.239**; real 2.862/1.608. Obs change vs unsteered (swap baseline 0.311): pseudoinv 0.0001 (0.0%), manifold 0.113 (36.5%), local 0.043 (13.8%). Fig 4d: manifold *increases* GT error → wrong-direction motion. Fig 4b: pseudoinv reverts at step 1. Fig 5: manifold/local drift (persistent divergence, not toward target); pseudoinv reverts to unsteered/pre-edit. Waterfalls: same ghost/curved failure as GRU.

**Sensitivity (Fig 6).** σ per dir: probe-x 0.017, probe-y 0.015, PCA#1 2.79, PCA#2 2.72, random 0.47, det-only-probe 0.41. Matched-‖Δh‖: PCA ≫ random > det-only-probe ≫ full-probe(≈0). Probe direction decoder-inert.

## h-vs-s decomposition (the RSSM-specific question)

**Position information lives in the deterministic `h` (256d), not the stochastic `s` (64d).** det-only linear pos R² 0.841 ≈ full 0.857; stoch-only only 0.594. The stochastic latent is low-rank (6/64 @90%) and does not hold the clean world-state readout the RSSM "prior" story would predict. Practically the RSSM behaves like the GRU with a small extra stochastic block bolted on: dynamics + readable state in `h`, `s` a compressed uncertainty code that partially correlates with position but is not where control lives.

---

## Caveats (flagged shaky where shaky)

- **[SHAKY] Fiber-resid magnitude (0.605) is not apples-to-apples with the GRU (0.347).** It's on the full 320-d state including the 64-d KL-regularized `s`, which legitimately carries prior/uncertainty structure that is *not* a function of (pos,vel). A fair comparison would refit `g` against **det-only h (256d)** and/or subtract the `s`-block's intrinsic non-canonicality. NOT yet done. The qualitative conclusion (non-canonical, curved) is robust; the exact number should not be quoted as "RSSM is 2× worse" without the det-only control.
- **Off-manifold residual scale (2.86 vs 1.75)** is not directly comparable — different H (320 vs 256) and different state normalization. Use the *structural* comparisons (k@90%, %swap, R²), not raw resid magnitudes.
- **pseudoinv obs-change = 0.0% is real but partly a probe-direction artifact:** the fitted position-probe direction happens to be an ultra-low-σ (0.017), decoder-inert axis of the RSSM state. A differently-conditioned probe might read the same position but point somewhere more generative. This sharpens "readable≠controllable" but the *specific* 0.0% depends on the probe's null-space alignment.
- N=500 warm-up / edit samples; N=64 sensitivity sweep; top-3 waterfalls. Aggregates; per-sample heterogeneity (as in GRU) not stratified here.
- `val_loss=3.02` for the RSSM is a different loss (incl. KL / different obs scaling) than the GRU's 0.024 — do not compare val losses across architectures.

## Open questions

- **Refit fiber-collapse `g` against det-only h (256d)** to get the apples-to-apples canonicity number vs GRU's 0.347. Predict it drops toward ~0.4–0.5 but likely still ≥ GRU (more curved).
- **Why is the RSSM probe direction decoder-inert (σ 0.017) when the GRU's was matched-magnitude generative?** Is the RSSM decoder reading position primarily off a *different* (PCA-heavy) axis than the linear probe recovers? A decoder-Jacobian / probe-onto-decoder-SVD alignment (as done for GRU) would localize this.
- **Does editing the stochastic `s` (or the prior parameters) do anything the h-edit can't?** All edits here were on the full flat state via a position probe dominated by `h`. An `s`-targeted or prior-targeted intervention is untested and is the natural RSSM-specific editor.
- Does the temporal (2-frame) velocity code enable a temporal editor that the single-frame probe edit can't — same open Q as GRU.

## One-line takeaway

The refined RSSM's latent is **predictively-used but non-canonical, curved, and history-entangled just like the GRU** — position lives in the deterministic `h` (not the stochastic `s`), velocity is only temporally readable, and editing fails identically (manifold edit moves obs 36.5% of a swap but in the wrong/scrambled direction; a perfect-readout pseudoinverse edit renders invisibly and reverts in one step). **The KL-regularized structured prior did not deliver a more canonical or more controllable representation; "readable ≠ controllable" is if anything sharper on the RSSM.**
