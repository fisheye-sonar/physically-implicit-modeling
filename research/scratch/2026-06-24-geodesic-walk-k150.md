# Geodesic walk — K=150 confirmation run + constant-step control (2026-06-24)

→ FLAG FOR PROMOTION

**Direction:** `research/directions/geodesic-walk.md` (⟳ CONFIRMATION RUN section).
**Notebook:** `notebooks/experiments/manifold_editing/geodesic_walk_k150.ipynb` (executed on GPU, RTX 5090).
**PNGs:** `/tmp/geodesic_k150/{1_convergence,2_obs_space_metrics,3_scans,4_waterfalls}.png`
**Scale:** N=64, K=150, STEP_FRAC=0.34, same checkpoint/data/probe as the K=30 pass.
**NOT promoted to findings; direction not marked done; RESEARCH.md untouched.**

## Headline

The K=30 "curvature barrier" in **readout space was largely a schedule artifact, not a true
plateau.** Run to K=150, *both* geodesic variants keep descending; the constant-step control
descends much faster and further. BUT — and this is the load-bearing result — **even as
readout RMSE drops dramatically, observation space barely moves (≤3% of the gap closed, ghost
remains).** So the editing failure does NOT live where the K=30 readout-plateau framing put
it. The probe can be walked most of the way to the target readout while the model's *generated
observation* stays essentially put. This is the same decoded-vs-observation disagreement the
2026-06-23 PCA run flagged, now reproduced under a long, careful on-manifold walk.

## Key numbers

Readout RMSE (decoded space), cold start = 1.712 for both:

| variant     | iter 25 | iter 50 | iter 100 | iter 150 (final) | tail slope (last 50) | Δ over tail | verdict (rule) |
|-------------|---------|---------|----------|------------------|----------------------|-------------|----------------|
| fractional  | 1.187   | 0.971   | 0.771    | **0.667**        | −0.00213 /iter       | −0.105      | SLOW CONVERGENCE |
| constant    | 0.612   | 0.475   | 0.393    | **0.354**        | −0.00069 /iter       | −0.034      | AMBIGUOUS (still descending) |

- Decision rule (last 50 iters): plateau requires |Δ RMSE| < 0.02 AND final > 0.5. Fractional
  fails the flatness test (Δ = −0.105, clearly still descending) → **not a plateau**. Constant
  is flatter (Δ = −0.034) but at a *lower* RMSE (0.354) and still negative-sloped; extrapolated
  iters-to-zero ≈ 512. Neither is the flat-far-from-zero signature of a real barrier.
- CONST_STEP = 0.914 (= first fractional iteration's mean step norm, per the brief).
- One-shot baselines (reference bars): pseudoinv readout=0.000, manifold=0.026, one-shot-local=0.679.
  Fractional geodesic (0.667) only just matches one-shot-local after 150 iters; constant (0.354)
  beats it.

Local off-manifold residual (the on-manifold check):
- Both variants: collapses to **~0.0002** after iter 1 and stays there for all 150 iters.
- real-state reference = 0.868. So the walk is hugging the local tangent plane **far more tightly
  than real states do** (ratio ~0.0002, i.e. ≈0). final-state local resid (refit fresh):
  fractional 0.094 (ratio 0.11), constant 0.211 (ratio 0.24).

Observation space at the FINAL iterate (step 0; unsteered→target render gap = 0.278):

| variant         | →target render | obs change | ghost ratio | % gap closed | decoded dist-to-target (mean) |
|-----------------|----------------|-----------|-------------|--------------|-------------------------------|
| unsteered       | 0.2777         | 0.000     | 1.000       | 0.0%         | 2.005 |
| manifold (1shot)| 0.2656         | 0.096     | 0.928       | 4.3%         | 0.029 |
| geodesic frac   | 0.2715         | 0.080     | 0.910       | 2.2%         | 0.845 |
| geodesic const  | 0.2698         | 0.146     | **0.735**   | 2.8%         | 0.447 |

- ghost ratio ~1 = old object still fully present at pre-edit location; ~0 = removed. Best is
  constant-step at 0.735 → ghost only partially reduced, never resolved.
- Note the decoded-vs-obs split: constant-step gets decoded dist to 0.447 (a big readout move
  from 2.005) yet closes only 2.8% of the obs gap. The probe reports motion the model does not
  generate.

## Answers to the brief's two questions

1. **Plateau vs slow convergence (K=150, fractional, tail rule):** NOT a plateau. RMSE is still
   descending at iter 150 (Δ −0.105 over last 50, slope −0.0021/iter, extrap-to-zero ≈ 313 iters).
   Call it **slow convergence in readout space** — the K=30 flattening was the fractional step
   shrinking, not a barrier.

2. **Constant-step control — barrier real or schedule artifact?** **Schedule artifact (in readout
   space).** With a fixed step magnitude the walk descends ~2× faster and reaches RMSE 0.354 (vs
   0.667 fractional) while staying genuinely on the local tangent (resid ≈ 0). So the apparent
   K=30 barrier was the decaying fractional schedule, not real curvature blocking readout descent.

## The twist that matters more than the brief's framing

Resolving readout reachability did **NOT** fix the edit. Observation space is unmoved (≤3% gap
closed, ghost 0.74–0.91) for *both* variants even at K=150 and even when decoded position is
dragged most of the way to target. So:

- The editing failure is **not** "the manifold curves away from the target readout" (we can walk
  there, slowly). It is that **moving the linear-probe readout — even along a tight on-manifold
  path — does not make the GRU generate the corresponding observation.** The probe direction and
  the model's generative-control direction are not the same direction.
- This is consistent with and strengthens the 2026-06-23 lesson: decoded-position space and
  observation space disagree; readout RMSE is a necessary-but-very-insufficient success metric.

## Caveats / open questions

- **Local residual ≈ 0.0002 is suspiciously small** — far below real states (0.868). The fresh
  per-step projection makes the iterate sit essentially exactly on its own local tangent plane,
  so this residual mostly measures projection tightness, not "looks like a real trajectory state."
  A better on-manifold check would be distance to the kNN bank / likelihood under the global
  manifold, not residual to the very plane we just projected onto. Treat "on-manifold" claims with
  this caveat.
- Constant-step verdict is "AMBIGUOUS" by the literal rule (final 0.354 < 0.5 fails the plateau
  gate but it's still descending) — it would likely reach low readout RMSE with more iters. The
  point stands: no flat-far-from-zero barrier.
- Obs-space effect is small for everyone including the one-shot manifold edit that *does* reach
  readout RMSE 0.026 — so the obs failure is a property of probe-vs-model misalignment, largely
  independent of how well the readout is solved. Worth a dedicated direction: is there ANY latent
  intervention (probe or otherwise) that moves the generated observation to target? If not, the
  position information the linear probe reads may not be in the form the recurrence uses to render.
- Did not run an MLP-gradient-step variant (brief's optional Q4); the linear-nudge reachability
  question is answered and the obs-space ceiling looks probe-agnostic.

## Files

- Notebook: `notebooks/experiments/manifold_editing/geodesic_walk_k150.ipynb`
- Figures: `/tmp/geodesic_k150/1_convergence.png` (RMSE/resid/step both variants),
  `/tmp/geodesic_k150/2_obs_space_metrics.png` (reach-target / obs-change / ghost vs rollout step),
  `/tmp/geodesic_k150/3_scans.png` (1D scans vs target render, final iterate),
  `/tmp/geodesic_k150/4_waterfalls.png` (per-variant waterfalls, target vs ghost centroids).
