# Direction: Geodesic Walk (gradient steering along the curved manifold)

**Tag:** `[in-frame]` · **Sub-question:** 3 (editability) · **Status:** proposed ·
**Complexity:** medium (likely its own notebook) · **Status:** in progress (2026-06-23).

---

## ⟳ CONFIRMATION RUN (2026-06-24) — execute THIS section

The first pass (K=30, in `notebooks/experiments/editability/geodesic_walk.ipynb`) found the on-manifold
geodesic *crawls* toward the target readout (RMSE 1.71→1.13) and never reaches it, with a
decaying step size. That is consistent with a **curvature barrier (true plateau)** but the K=30
data cannot rule out **slow-but-would-converge**. Your job is the clean confirmation. Two things
the first pass left open, both of which you must resolve:

1. **Run to large K.** Set `K_ITERS = 150` (and keep N=64, same checkpoint/data/probe,
   `STEP_FRAC = 0.34`). Log the full per-iteration readout-RMSE and local-residual curves.
   **Decision rule:** fit the RMSE tail (last 50 iters). If the slope over the last 50 iters is
   effectively flat (|Δ RMSE| < 0.02 across them) AND final RMSE ≫ 0 (say > 0.5), call it a
   **plateau / barrier**. If RMSE is still descending materially (and extrapolates toward ~0),
   call it **slow convergence**. State which, with the numbers.

2. **Constant-step control — the methodological fix (REQUIRED).** The fractional nudge
   `h + step_frac·(target − h)` shrinks geometrically as `h` approaches the target *even on a
   perfectly flat manifold*, so "step size decayed" is NOT by itself evidence of a barrier. Add a
   second geodesic variant that takes a **constant-magnitude** step each iteration: move a fixed
   ‖Δh‖ (set it to the *first* iteration's step norm from the fractional run) in the direction of
   `(inject_state(h,target) − h)` normalized, then do the same local-tangent reprojection. Run it
   to K=150. **This is the clean test:** if RMSE *still* plateaus far from 0 under a constant step
   while local-residual stays ≈ real (genuinely on-manifold), the barrier is real curvature, not a
   schedule artifact. If constant-step reaches the target, the K=30 "barrier" was just the decaying
   schedule. Report both curves on the same axes.

Also (cheap, do it): at the **final** iterate of each geodesic variant, regenerate the obs-space
waterfalls (object reach target? ghost gone?) so the obs-space conclusion is anchored at large K,
not just K=30.

Work in a **new** notebook `notebooks/experiments/editability/geodesic_walk_k150.ipynb`. You may **read**
(not modify) `notebooks/experiments/editability/geodesic_walk.ipynb` as the structural template — copy its
cold-start bootstrap and helpers, bump `K_ITERS`, add the constant-step variant and the tail-fit
decision. Everything else (deliverables, both-spaces requirement) follows the sections below.

---

## Motivation

`findings/editability.md` localized the editing failure to **target unreachability
under manifold constraint**: a single jump-then-project edit can't reach the
target because the manifold curves away from the local neighborhood, and a global
PCA projection lands off the curved surface. The natural fix is to stop *jumping*
and instead *walk*: take many small steps, refitting the local tangent subspace at
each step, climbing along the curved manifold toward the target readout. This
directly tests whether the curvature barrier is *traversable* — a clean binary:
does readout RMSE converge to ~0 as you iterate, or plateau?

This is also where local tangent PCA earns its keep (it was a dead end as a
one-shot editor, but it's exactly the right primitive for per-step projection).

## Bootstrap (cold-start — run from a fresh kernel)

This brief runs in its **own new notebook**; do not assume a live kernel. Reconstruct
from the checkpoint/data paths in Context, mirroring
`notebooks/experiments/editability/pca_component_position.ipynb`, which already does exactly this:
load model + linear probe; teacher-force the test set → `states_tf`; `fit_state_subspace`
→ `subspace`; `warm_up_to_edit` → `warm` / `h_at_edit`; reuse its `rollout_from_flat` /
`decode_pos` helpers. (These are duplicated across notebooks pending a factor-out into
`pim/` — see PROGRESS open decisions.)

## Method sketch

Iterate, starting from `h_at_edit`:
1. Take a small step toward the target readout — either a gradient step on the
   probe loss (`gradient_steering.gradient_steer`, supports the MLP probe too) or
   a small linear pseudoinverse nudge.
2. Refit a *fresh* local tangent subspace around the *new* point
   (`fit_local_subspace`) and project onto it.
3. Repeat for K steps, logging readout RMSE, local off-manifold residual, and
   step size each iteration.

Compare against the existing one-shot baselines (pseudoinverse, global manifold,
one-shot local) from the editability notebook.

## Questions to answer

- Does iterated small-step + local-reproject **converge** to the target readout
  (RMSE→0) while staying on-manifold (local residual ≈ real)? If yes, the barrier
  is curvature, traversable — strong evidence the state is editable in principle.
- If it plateaus, *where* (what RMSE), and is the plateau a different attractor?
- Does a converged geodesic edit produce coherent observations (object at target,
  no ghost) — i.e. does solving unreachability also fix the phantom-object artifact?
- Linear-probe step vs MLP-gradient step: does the nonlinear probe change reachability?

## Measurement & visualization (required)

Readout RMSE (decoded space) is the convergence metric, but it is **not** sufficient
evidence of success. *(Lesson from the 2026-06-23 PCA run: decoded-position space and
observation space can disagree — the probe can report motion the model doesn't generate.)*
So also measure success in **observation space**: after a converged edit, does the
*generated* observation actually move toward what the target object position would
produce? Report both, and visualize per today's convention:
- Plot readout-RMSE / off-manifold-residual vs iteration (does it converge or plateau?).
- Plot the **generated 1D observation scans + waterfalls** for unsteered vs geodesic-
  edited vs the one-shot baselines — show whether the object reaches the target *in
  observation space* and whether the phantom/ghost artifact is gone.
- Produce **both** the plots (for Sevan) and printed metric tables (for the agent).

## Notes / risks

- Cost: per-step local-subspace refit × K steps × N samples. Start small
  (N≤64, K≤30) before scaling.
- Don't over-engineer the optimizer; the question is reachability, not SOTA editing.
- Keep it a *latent* intervention (don't smuggle in observations) to stay within
  the editability framing.

## Context

- Primitives exist: `pim/editors/manifold_steering.py` (`fit_local_subspace`,
  `manifold_steer`), `pim/editors/gradient_steering.py` (`gradient_steer`),
  `pim/editors/probe_steering.py` (`inject_state`, `probe_decomposition`).
- Same checkpoint/data as the editability notebook.
