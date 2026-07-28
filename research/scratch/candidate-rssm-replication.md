# CANDIDATE FINDING — Architecture-independence: the RSSM replicates the GRU failure (KL structure buys no canonicity)

**✅ PROMOTED → `findings/architecture-independence.md` (2026-07-09)**, with preliminary/scoped hedging
(two checkpoints, not a law). Kept as backing detail.

**Consolidates:** `archive/2026-07-02-rssm-state-geometry.md`.
**Model/data:** refined RSSM `runs/rssm/4_dset4_refined_best/best_model.pt` (epoch 500,
det256+stoch64 ⇒ flat H=320, `model.sample=False`), `4_fixed_refl_inview`.
Notebook: `notebooks/experiments/editability/rssm_structure/rssm_state_geometry.ipynb`.
**Sub-question:** 1+2+3 (cross-cutting). **Status:** candidate for a **new** finding (kept separate
from the GRU editability/geometry findings — this is the *generalization* claim, per Sevan's
"don't squish adjacent things into one master finding").

## The claim (one line)
The GRU's non-canonical / readable≠controllable failure is **architecture-independent**: a refined,
KL-regularized RSSM with an explicit stochastic latent replicates every part of it — the structured
prior delivers **no** gain in canonicity or controllability. Position and the non-canonical code live
in the **deterministic recurrent core** (as canonical/non-canonical as the GRU's `h`, resid 0.368 vs
0.337), the stochastic `s` is a low-rank uncertainty code that holds neither; and in one respect —
a perfect-readout edit that renders invisibly — the failure is even sharper.

## Key numbers (vs GRU)
- **Geometry replicates, slightly more curved:** 34/320 dims @90% (GRU 38/256); tangent→global angle
  **65.2°** (GRU 56°).
- **Position lives in the deterministic `h`, not the stochastic `s`:** det-only linear pos R² **0.841**
  ≈ full 0.857; stoch-only only 0.594; `s` is low-rank (6/64 @90%). **Refutes** "the stochastic latent
  captures the compact world state" — the GRU cell pushes position into `h`.
- **Velocity temporal signature replicates:** single-frame linear 0.43, 2-frame MLP 0.695 *(⚠ single-
  frame MLP 0.69 ≈ 2-frame MLP 0.695 — the temporal window adds ~nothing once nonlinear; this is the
  seed of the velocity-reframe in `directions/diagnostic-corrections.md`)*.
- **Fiber: det core ≈ as (non-)canonical as GRU (CORRECTION LANDED 2026-07-08):** RSSM **det-only**
  `g(pos,vel)` residual **0.368** ≈ GRU **0.337**; the stochastic `s` block is 0.891 (legitimately not
  a function of `(pos,vel)`) and inflated the full-320 number to 0.602. The earlier "RSSM ~2× less
  canonical (0.605 vs 0.347)" was a measurement artifact of including `s` — on the deterministic core
  the two architectures are on par. **Do NOT claim the RSSM is less canonical.**
- **Editing fails identically; readable≠controllable is SHARPER:** pseudoinverse edit hits probe target
  exactly (readout RMSE **0.000**) yet moves obs **0.0%** of a swap and reverts in **one step**;
  global-manifold edit moves obs 36.5% of a swap (≈GRU 37%) but in the wrong/scrambled direction.
- **Divergence from GRU (mechanism):** the RSSM position-probe direction is **decoder-inert**
  (σ 0.017 vs PCA 2.79, ~165×); on the GRU the matched-magnitude probe direction *was* generative.

## Why it matters
This is arguably the most paper-worthy result on the table: it lifts the editability story from
"a fact about one GRU" to "a property of implicit predictive world models" — and shows a popular
structured-latent remedy (RSSM/KL prior) does not fix it. Direct motivation for the RESEARCH.md thesis
that explicit physical scaffolding, not stochastic latents, is what canonicality requires.

## Caveats / RSSM-eval refinements owed
- **fiber-resid RESOLVED (2026-07-08):** det-only 0.368 ≈ GRU 0.337 — on par; the full-320 0.602 was
  the stochastic `s` inflating it. Quote the det-only number, not the full-320.
- **Off-manifold residual scale (2.86 vs 1.75)** not comparable across H=320 vs 256 — use structural
  comparisons (k@90%, %swap, R²), not raw resid magnitudes.
- **pseudoinv 0.0% is partly a probe-direction artifact** (the fitted probe axis is ultra-low-σ /
  decoder-inert); a differently-conditioned probe might read the same position on a more generative
  axis. Sharpens readable≠controllable but the specific 0.0% depends on null-space alignment.
- Do NOT compare RSSM val_loss (3.02, incl. KL) to GRU (0.024) — different objectives.
- **General note (RSSM eval needs refinement):** several RSSM diagnostics were computed on the full
  320-d flat state, which mixes the deterministic world-state carrier `h` with the stochastic
  uncertainty code `s`. Future RSSM evals should report **h-only, s-only, and full** consistently
  (as the recoverability section already did) rather than defaulting to full-state.

## Promotion recommendation
**PROMOTE as a new finding (e.g. `findings/architecture-independence.md` or a section in
`editability.md`)** — the cross-architecture claims are solid and the fiber magnitude is now resolved
(det-only 0.368 ≈ GRU 0.337; drop the "less canonical" framing → "no better, same code in the det
core"). Keep the generative-quality gap in the separate predictive-quality candidate. Ready for your read.
