# 2026-06-23 — Geodesic walk: editability via on-manifold readout-targeting

**Direction:** `directions/geodesic-walk.md` (`[in-frame]`, sub-Q3).
**Notebook:** `notebooks/experiments/editability/geodesic_walk.ipynb` (runs top-to-bottom on GPU).
**PNGs:** `/tmp/geodesic/{1_convergence,2_obs_space_metrics,3_scans,4_waterfalls}.png`
**Context:** GRU `3_dset3_..._400epochs`, dataset `4_fixed_refl_inview`, N=64 edits,
K=30 geodesic iters, step_frac=0.34, linear position probe.
*(Record written by the orchestrator — the worker agent completed the notebook+figures
but failed to write this note or report; numbers extracted from notebook outputs + figure reads.)*

## Headline

Two ways to edit, **both fail to produce a coherent counterfactual observation**:
- **Stay on-manifold (geodesic):** respects the manifold almost perfectly but only
  *crawls* toward the target readout and never reaches it (curvature barrier confirmed).
- **Go off-manifold to hit the readout (pseudoinv / global manifold):** reaches the
  target in *decoded* space but the model does **not generate** the corresponding
  observation — the object doesn't move to target and the old object (ghost) remains.

→ FLAG FOR PROMOTION (artifact-or-signal check owed; see caveats): "No probe-targeted
latent edit of this GRU produces a clean counterfactual observation. On-manifold edits
can't reach the target readout (geodesic: RMSE 1.71→1.13 over 30 steps, still > 1, step
size decaying); off-manifold edits reach the readout but the dynamics don't render it
(decode≠generate). Across ALL methods the edited object fails to appear at the target
location and the original persists (ghost ratio 0.93–0.98)."

## Readout / on-manifold (fig 1)

| edit | readout RMSE | global resid | local resid |
|---|---|---|---|
| unsteered | 1.84 | 1.75 | 0.91 |
| pseudoinv | 0.00 | 1.83 | 1.09 |
| manifold (global) | 0.03 | 0.00 | 1.68 |
| one-shot local | 0.68 | 1.94 | 1.08 |
| **GEODESIC** | **1.25** | 1.52 | **0.059** |
| real states | — | 1.75 | 0.87 |

Geodesic local resid ≈ 0.0002 *throughout* the walk (final 0.059, even **below** real
states 0.87) — it is genuinely, strictly on-manifold. Readout RMSE crawls 1.71→1.13 in
30 iters, **still decreasing** but far from 0; step size decays 0.9→0.06 (slowing →
consistent with approaching a barrier, not linear progress). This is the curvature
barrier from `findings/editability.md`, now confirmed by a curvature-aware walk: staying
on the manifold, the target readout is effectively unreachable.

## Observation space (figs 2 & 4 — the decisive evidence)

`->target render` (lower = closer to a perfect edit's obs; unsteered=0.278):
pseudoinv 0.274, manifold 0.266, one-shot local 0.276, **geodesic 0.271**. The best
(manifold) closes only ~4% of the gap. Ghost ratio (≈1 = old object still fully present):
all methods 0.93–0.98 — **the ghost is never removed.**

**Waterfalls (fig 4) are the clean visual proof** (model-generated, renderer-free):
green = where the edited object SHOULD be, red = ghost (pre-edit) location. Across
samples 26/6/45, **geodesic looks essentially identical to unsteered** — the bright
streak stays at the ghost (red) line; nothing appears at green. Off-manifold edits
(pseudoinv/manifold) occasionally smear faint intensity but never relocate the object;
ghost always remains. **No method moves the object to the target in observation space.**

Both-spaces check: manifold reaches target in DECODED space (obj-dist 0.029) but NOT in
obs space — the cleanest decode≠generate instance yet. Geodesic reaches neither (decoded
obj-dist 1.40, obs unchanged): slow in both.

## Caveats (for the artifact-or-signal call)

1. **K=30 may be too few** — convergence "still decreasing," so we can't fully separate
   "slow but would converge" from "plateaus at a floor." The decaying step size leans
   toward a barrier, but a **K≥100–200 run** is the clean confirmation before promotion.
2. `->target render` has the known renderer mismatch floor (~0.24–0.28 from the PCA run),
   so the scalar obs metric is noisy — **but the waterfalls are direct model output** and
   show no object motion, so the qualitative conclusion is robust.
3. step_frac=0.34 is one schedule; geodesic is not a tuned optimizer. A different schedule
   might crawl faster but is unlikely to cross a real curvature barrier.

## Open questions for Sevan

- Promote the two-horns result now, or gate on the K≥100 confirmation run first?
- Does "can't edit cleanly" hold for the **MLP/gradient** probe step too (only linear tested)?
- Is the ghost-persistence itself the deeper phenomenon — the GRU encodes object *identity/
  history* in a way a position edit can't overwrite? (Connects to sub-Q2 identifiability.)
