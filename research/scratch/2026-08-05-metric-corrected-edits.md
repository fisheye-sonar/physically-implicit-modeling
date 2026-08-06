# Metric-corrected edits — the whitening hypothesis is real, and it is not enough

*Scratch, 2026-08-05. Notebook: `notebooks/experiments/editability/metric_corrected_edits/metric_corrected_edits.ipynb`
(12 cells, 0 errors, 5 figures). GRU `runs/controls/H256`, dataset `4_fixed_refl_inview`: bank = 78,000 aligned
`test` states, N=256 held-out edits.*

## Sevan's hypothesis

A least-squares probe is `W = Σ_ph Σ_hh⁻¹`. If `h ≈ h₀ + Jp` with `J = ∂h/∂p` — the direction `h` actually moves
when position changes — then `Σ_ph = Σ_pp Jᵀ` and `Wᵀ = Σ_hh⁻¹ J Σ_pp`. **The probe's row space is `J` whitened
by the inverse state covariance, not `J`.** With anisotropic `Σ_hh` the two can be near-orthogonal even for a
perfectly accurate probe — which would explain every probe-derived editor failure in this thread without
implicating the representation. Un-whitening (multiply by `Σ_hh`) should recover `J`.

## Verdict: cosine jump, no working edit

Per the interpretation guide fixed *before* the run. The mechanism is real and measurable; correcting for it is
not sufficient.

**Gate — passed decisively.** `Σ_hh` condition number **1.79e4** (λ_max 4.72, λ_min 2.6e-4); states occupy
40/76/173 dims at 90/95/99% variance. It also **independently corroborates the derivation**: `iterative_probing`
found the linear position code sits in *below-average-variance* directions, which is precisely what `Σ⁻¹` does
to a least-squares probe. Two independent routes to the same structural claim.

**Test 1 — the direction improves substantially.**

| Δh_true from | α=0 (existing editor) | α=1 (full un-whitening) |
|---|---|---|
| counterfactual overwrite | cos **+0.079** (85°), mass 0.098 = **0.78× chance** | cos **+0.236** (76°), mass 0.380 = **3.04× chance** |
| freeze-time forcing | cos **+0.058** (87°), mass 0.072 = **0.58× chance** | cos **+0.232** (77°), mass 0.366 = **2.93× chance** |

Monotone in α. The reachable subspace goes from **below chance to 3× chance** — the first probe-derived subspace
in this thread that is meaningfully enriched. Absolute scale still matters though: 76° is nearly orthogonal.
(The constraint-satisfying family `Δ_α = Σ^α Wᵀ(WΣ^αWᵀ+εI)⁻¹δ` and the literal `Σ^α W⁺δ` agree to ±0.003, so the
reformulation needed to keep the readout constraint exact costs nothing.)

**Test 2 — real, non-degrading, still not an edit.** Only the metric differs from the existing failing editor:
unsteered −0.67 → Euclidean α=0 **−0.65** → Mahalanobis α=1 **−0.51**, fidelity **0.98**, Target RMSE
0.488→0.432, Ghost 0.589→0.534. **+0.14 index points, training-free, at zero fidelity cost.** For scale, the
*heavy fine-tuning* arm of the trained-editability thread bought +0.13 and cost 13% of next-step prediction.
**This is the best training-free structural editor the thread has produced** — and it is still deep on the
unedited side.

**Test 3 — magnitude is not the missing ingredient either.** For α=1 there is a genuine optimum, and it sits
almost exactly where ‖Δ‖ matches the oracle's (3.39× a dynamics step vs the oracle's 3.75×): Target and Ghost
RMSE both **minimise at ×3** (0.380 / 0.483). But fidelity crosses 1 at ×3, so the best *legitimate* arm is
**×2 → −0.33** = **25% of the oracle's gain**. Beyond that the index rises only by degrading: ×8 hits +0.01 with
fidelity **1.57** and every zone worse than unsteered — Fig 5's ×8 column is bright striped garbage. Scaling the
*Euclidean* direction is much weaker (α=0 ×8 reaches only −0.45, Target 0.479→0.447), so the metric correction
contributes beyond magnitude. **Neither "wrong direction" nor "wrong magnitude" alone; fixing both gets a
quarter of the way.**

**Test 4 — the local metric is worse.** Local `Σ_hh` from 1024-NN gives cos **+0.143** at α=1 vs the global
**+0.236**, peaking at α=0.5 (0.177) then declining. Its Edit Index (−0.38) *looks* better than global (−0.51)
but that is entirely magnitude: local produces ‖Δ‖ = 2.27× a step vs global 1.13×, and **at matched displacement
the global metric wins** (global ×2 → −0.33 at 2.26×). The anisotropy that matters is a **global** property of
the state distribution, not local curvature. (This is exactly the confusion the scale sweep exists to prevent.)

## The pattern across three unrelated constructions

| method | fraction of `Δh_true` captured | fraction of the oracle's Edit Index gain |
|---|---|---|
| tangent-constrained projection (22-dim local PCA) | 57% | ~33% |
| projection onto the 116-dim linear position code | 57% | 33% |
| metric-corrected direction at its best legitimate scale | ~24–38% | 25% |

Three unrelated ways of capturing part of a working edit, and the recovered *effect* is consistently well below
the recovered *fraction of the vector*. The all-or-nothing reading now has a graded form: **partial capture buys
sub-proportional effect.**

## What it changes

The whitening account is **mechanistically real** and should be adopted as the explanation for why
probe-derived directions have looked orthogonal to successful edits — that specific puzzle is now solved, and it
was a metric artifact rather than a fact about the representation. Every past row-space/orthogonality number in
the thread should be read with that caveat: they were measured in the Euclidean metric, which is the wrong one.

But it does not rescue editing, and it narrows things usefully. The remaining gap is **not**: the probe's 4-dim
slice (`iterative_probing`), the linear position code (same), manifold-tangency (tangent experiment), the metric
(here), or magnitude (here). The consistent finding across all four is sub-proportional return on partial
capture, which points at whatever the *remaining* 60–75% of `Δh_true` is doing.

## Follow-ons

- **Characterise `Δh_true`'s complement directly** rather than testing a fifth candidate subspace — four misses
  now share a signature.
- **Resolve the ×2–×3 optimum**: the grid is coarse and fidelity crosses 1 inside the bracket. A finer sweep
  with the fidelity constraint enforced would give the honest best-case for this editor family.
- **Re-express old orthogonality numbers in the Mahalanobis metric**, since the Euclidean ones are now known to
  be the wrong measurement.
- Neighbourhood-size sweep for the local metric (only `k = 1024` was tried).
- One model, one seed, position probe only.
