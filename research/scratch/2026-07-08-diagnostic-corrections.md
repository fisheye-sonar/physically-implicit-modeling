# Diagnostic corrections — velocity 2×2, det-only fiber, honest small-k geodesic (2026-07-08)

→ FLAG FOR PROMOTION

Corrections pass on three suspect numbers in the GRU + RSSM notes. Notebook:
`notebooks/experiments/manifold_editing/diagnostic_corrections.ipynb` (executed on GPU).
Models: GRU `runs/gru/3_dset3_gru_persistentids_inview_400epochs` (H=256, val_loss 0.0236),
refined RSSM `runs/rssm/4_dset4_refined_best` (det=256, stoch=64, total 320; `sample=False`,
posterior-mean states). Data `datasets/4_fixed_refl_inview`, n_obj_keep=2, teacher-forced test
set (10000×39). Velocities read directly from HDF5 `velocities` (temporal std ≈ 1.3e-8 → confirmed
constant-velocity sim). PNGs: `/tmp/diagnostic_corrections/{1_velocity_2x2,2_fiber_detonly,3_geodesic_smallk,4_waterfalls}.png`.

---

## Section 1 — Velocity 2×2 {linear,MLP} × {single-frame,2-frame}, both models

Target = 4-d velocity (vx,vy)×2. R² overall (all comps), vs predict-the-mean baseline. MLP =
256-wide 2-hidden-layer ReLU. "late-t" = frames t≥15 (fair test; early frames underdetermine v).

**GRU**

| feature | linear R² (all / late) | MLP R² (all / late) |
|---|---|---|
| single-frame h_t | 0.493 / 0.588 | **0.886 / 0.944** |
| 2-frame [h_{t-1},h_t] | 0.508 / 0.606 | **0.901 / 0.951** |
| dh = h_t − h_{t-1} | 0.111 / 0.244 | 0.631 / 0.723 |

**RSSM (det+stoch, 320)**

| feature | linear R² (all / late) | MLP R² (all / late) |
|---|---|---|
| single-frame h_t | 0.438 / 0.573 | **0.844 / 0.931** |
| 2-frame [h_{t-1},h_t] | 0.465 / 0.610 | **0.842 / 0.931** |
| dh = h_t − h_{t-1} | 0.099 / 0.360 | 0.698 / 0.807 |

Δ(2-frame MLP − single-frame MLP): GRU all +0.015, GRU late **+0.007**; RSSM all −0.002,
RSSM late **−0.000**. All ≤ 0.03 threshold.

**VERDICT — "velocity is temporal" is RETIRED / reframed.** On BOTH models, single-frame MLP ≈
2-frame MLP (Δ ≤ 0.015 all-t, ≤ 0.007 late-t). Velocity is **instantaneously readable from a single
frame's hidden state — just nonlinearly** (single-frame R²: linear ~0.5–0.59 → MLP ~0.88–0.94). The
temporal window adds essentially nothing once nonlinearity is allowed. The original GRU keystone
(single-frame LINEAR 0.47 vs 2-frame MLP 0.76) changed two things at once; the entire gap is the
**linear→MLP** axis, not single→temporal. The `dh` feature is strictly worse than single-frame h_t
(differencing destroys the instantaneously-encoded signal), further contradicting a temporal story.
Late-t per-component R² (MLP) all ≥ 0.88 for both models — velocity is cleanly, snapshot-readable.

## Section 2 — Det-only fiber-collapse refit (RSSM)

g(pos,vel)→block via linear + MLP (512-wide 2-hidden). Fiber metric = MLP residual fraction
‖block−g‖/‖block‖ (lower = block is more nearly a function of (pos,vel) → more canonical).

| block | linear resid / R² | **MLP resid** / R² |
|---|---|---|
| GRU h (256) | 0.877 / 0.100 | **0.337** / 0.867 |
| RSSM full (320) | 0.900 / 0.091 | **0.602** / 0.593 |
| RSSM h_det (256) | 0.854 / 0.138 | **0.368** / 0.840 |
| RSSM s_stoch (64) | 0.988 / 0.008 | **0.891** / 0.193 |

**VERDICT.** The apples-to-apples comparison is GRU h 0.337 vs RSSM det-only **0.368** — nearly
identical. The RSSM's deterministic block is **almost as canonical as the GRU's h** (slightly less;
det > GRU by 0.031, just outside the 0.03 "equal" band). The old 0.605 full-320 number was inflated
almost entirely by the **stochastic s block**, which is NOT a function of (pos,vel) (its own MLP
residual 0.891, R² 0.19 — as expected for a KL-regularized latent). Splitting it out: full 0.602 →
det 0.368 (Δ = 0.234 is the s-block contribution). So the earlier "RSSM is much less canonical than
the GRU (0.605 vs 0.347)" claim is a measurement artifact of including s; on the deterministic block
the two architectures are essentially on par. Predicted range in the brief was ~0.4–0.5; the actual
det-only number (0.368) is slightly better (more canonical) than predicted.

## Section 3 — Honest small-k geodesic (GRU)

Constant-step walk toward the position-probe target, N=400 edit samples, K=120 iters, step =
STEP_FRAC·(mean pseudoinv jump norm) = 0.34·0.534 = **0.182** (matches `geodesic_walk_k150`; an
earlier draft used a 10× smaller step and the walk barely moved — corrected). Honest local residual =
leave-out-neighborhood: fit tangent PCA (var 0.90) on the k nearest bank points **excluding the
query's nearest neighbor**, then measure ‖h − proj‖ (raw) and fraction ‖h−proj‖/‖h−local_mean‖.
Cold-start per-sample readout RMSE = 1.586.

| LOCAL_K | final readout RMSE (gap closed) | honest resid frac (walked) | honest resid frac (REAL states) | obs move (% of full-swap) |
|---|---|---|---|---|
| 16 | 1.116 (30%) | 0.297 | 0.788 | 312% |
| 32 | 0.979 (38%) | 0.147 | 0.696 | 353% |
| 64 | 0.860 (**46%**) | 0.108 | 0.577 | 376% |

Tautology contrast (self-INCLUDED) real-state frac: 0.51 / 0.47 / 0.43 — so the leave-out honesty
matters (it roughly doubles the real-state residual at k=16), confirming the metric is not the
projection tautology that collapsed to ~0 at LOCAL_K=512.

**VERDICT — nuanced; the old on-manifold claim does NOT cleanly reproduce at honest small k.**
1. **Reachability:** the walk moves the readout substantially (1.586 → 0.860 at k=64) and the
   observation moves a lot (300–376% of a full state-swap's obs change — it overshoots the swap
   scale), so the edit is *not* inert. But it does **NOT reach the target**: only ~30–46% of the
   readout gap is closed after 120 constant steps; larger k walks further. This is "partial /
   descending," consistent with the k150 note's "slow, does not converge" rather than a hard barrier.
2. **Honest residual is LOW, not 0.75–0.84 — and this overturns the brief's stated expectation.**
   The walked states' honest leave-out residual is 0.11–0.30, *below* the real-state honest reference
   (0.58–0.79) at the same k. The brief expected the walk to sit at ~0.75–0.84 (real-manifold
   thickness) and worried the tautology would fake ~0; the actual result is in between and biased
   LOW. Interpretation: because every iterate is projected onto a fresh local tangent of its k
   neighbors, the walk stays *hugging* the local tangent patch — arguably MORE "locally on-plane"
   than a generic real state (which, minus its duplicate neighbor, sticks out 0.58–0.79). So the
   walked states are geometrically plausible (low off-tangent residual) yet the readout is only
   half-corrected. The bottleneck is **reachability along the manifold, not an off-manifold barrier**.
3. Net: the corrected honest metric does not support a clean "stays at 0.75–0.84 = real-manifold
   barrier" story. It supports: the local-projection walk keeps states on-manifold (low honest
   residual) and moves the observation strongly, but cannot fully drive the position readout to
   target within the step budget — a **curvature/reachability limit**, not off-manifold ejection.
   The old LOCAL_K=512 "residual ~0" was indeed a tautology; the honest small-k number is small but
   nonzero and, importantly, *smaller than real states*, which is the opposite of the predicted 0.75–0.84.

## Caveats

- **Section 1:** single/2-frame MLP probes use identical capacity/epochs; the ~0 gap is robust
  across all-t and late-t and both models, so not a tuning fluke. Probes are fit + evaluated on the
  same masked entries (no held-out split) — R² magnitudes are optimistic in absolute terms, but the
  *comparison* (single vs 2-frame) is what carries the verdict and is unaffected.
- **Section 2:** same in-sample-fit caveat; the comparison across blocks (GRU vs det vs s) is the
  load-bearing quantity. Residual fraction uses ‖·‖ over centered-at-origin block norm (matches the
  existing note's definition), so numbers are directly comparable to the prior 0.347/0.605.
- **Section 2 nuance:** det-only 0.368 vs GRU 0.337 is a 0.031 gap — the notebook's auto-verdict
  labels it "det > GRU (less canonical)" because it's a hair outside the 0.03 "equal" band, but the
  honest read is **essentially on par** (both ~0.35, both R²≈0.85 on the block). Do not over-read the
  0.031 as a meaningful architectural difference.
- **Section 3 caveats:**
  - The honest-residual result **contradicts the brief's predicted 0.75–0.84**: walked states sit at
    0.11–0.30 (below real-state reference 0.58–0.79). Flagged plainly in the verdict above — this is
    the one place a correction *overturns* an expectation rather than confirming it.
  - The "obs moves >100% of full-swap" is because the full-swap proxy is the one-shot pseudoinverse
    injection (off-manifold, and its own obs change is small, 0.0184); the geodesic moves the obs
    MORE than that proxy. So "% of swap" here should be read as "obs moves a lot," not a clean 0–100%
    fraction — the denominator is a weak reference. A stronger 100%-reference (teacher-forced true
    post-edit state) was not built for cost; worth adding if this feeds the unified comparison.
  - Reachability is partial (30–46% gap closed at K=120); a longer walk or larger step may close more.
    The result is "slow/partial, on-manifold," not "hard barrier."
  - N=400 edit samples, honest residual probed on 150; stable but not the full 10k.
