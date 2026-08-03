# Multi-step (latent-overshooting) training objective — RSSM replication

**Date:** 2026-07-16 · **Direction:** `multistep-objective-rssm` · **Status:** → **FLAG FOR PROMOTION**
(completes the OWED RSSM leg of the promoted GRU multistep finding) · **Author:** orchestrator (ran autonomously
after Sevan green-lit + left)

## TL;DR (headline)
The GRU's clean **negative replicates on the RSSM** — a multi-step **latent-overshooting** training objective
(`W∈{2,5}`) does **NOT** make the RSSM's latent more editable; no non-oracle editor reaches the true-state swap
for any `W`, exactly as at `W=1`. **And on the RSSM the objective is worse than neutral — it actively HARMS the
model:** it degrades single-step *and* rollout predictive quality, **collapses the decoder into blur**
(rollout TV/GT 1.23 → 0.43 — objects fade), **collapses the linear hull** (90%-var dim 36 → 10), and makes the
physical state **less linearly readable** and **less canonical**. So for the architecture *built for* multi-step
latent rollout, adding the overshoot objective buys no editability and costs predictive quality + structure. This
is a stronger negative than the GRU's (where multi-step was roughly neutral / mildly rollout-*helpful* with no blur).

## Setup / provenance
- **New objective (`scripts/train_rssm_multistep.py`):** standard ELBO (recon + 1-step KL) PLUS PlaNet-style
  **latent overshooting** for `W≥2` — from each posterior state, imagine `W` steps through the PRIOR and add
  obs-reconstruction of the future frames + `KL(sg(posterior[t+d]) ‖ imagined-prior_d)` (free-nats clamped);
  overshoot starts subsampled `n_start=8`. `W=1` = pure ELBO (= the standard RSSM objective).
- **Three RSSMs, matched budget:** det 256 / stoch 64, dataset 4 (noisy 0.2), **150 epochs each** (reduced from
  the refined RSSM's 500 to fit the ≤3h cap — the **cross-`W` comparison is the load-bearing quantity**). Best-recon
  checkpoints: `runs/rssm_multistep/w{1,2,5}_dset4` (best val_recon 0.0247 / 0.0323 / 0.0365).
- **Eval:** `sample=False` (prior-mean, deterministic). Teacher-forced **test** split (§0–§3), edits split (§4, N=64).
  Probes in-sample. Notebook `notebooks/experiments/editability/multistep/multistep_objective_rssm.ipynb` (0 error cells).
  Figures `/tmp/multistep_objective_rssm/`.

## Results by section (w=1 / w=2 / w=5)

**§0 Sharpness & predictive quality — the big RSSM-specific failure (OPPOSITE the GRU).**
- Teacher-forced next-step RMSE vs clean: **0.113 / 0.142 / 0.166** — multi-step **hurts** single-step prediction
  (GRU barely moved: 0.104→0.109).
- Open-loop horizon RMSE (mean): **0.204 / 0.227 / 0.247** — multi-step **hurts** rollout too. (GRU: multi-step
  *improved* rollout 0.208→0.188 — the RSSM is the reverse.)
- Rollout TV / GT TV: **1.234 / 0.431 / 0.434** — the overshoot models **BLUR badly** (TV collapses to <0.45×GT;
  `W=1` is over-sharp at 1.23). **This is the blur / mode-collapse watch-item firing** — the multi-step objective
  through the KL-regularised stochastic prior washes the rollout out (objects fade). GRU showed **no** blur.

**§1 Geometry — linear-hull collapse; intrinsic dim ~stable at ~8.**
- PCA hull @90%: **36 / 15 / 10** — a dramatic linear-hull compaction under overshoot.
- Intrinsic dim TwoNN **8.1 / 7.8 / 9.2**, MLE **8.5 / 8.7 / 8.9** — stays near the physical 8 for all `W`.
- Tangent rotation **21° / 20° / 18°** (un-normalized — not comparable across notebooks; see caveat). So the state
  becomes much lower-rank *linearly* but not lower intrinsic-dim — consistent with a blurrier, more compressed code.

**§2 Recoverability — degrades (more than the GRU).**
- Position R² linear **0.82 / 0.66 / 0.64** (MLP 0.95 / 0.93 / 0.93 — still nonlinearly readable); velocity R²
  single-frame linear (late) **0.54 / 0.37 / 0.35**. The physical state becomes **less linearly readable** under
  overshoot (GRU pos-linear dropped more mildly, 0.84→0.76). Velocity remains instantaneous-nonlinear (2-frame ≈
  single-frame), as everywhere in this project.

**§3 Canonicality — degrades.** MLP fiber residual **0.421 / 0.590 / 0.518** — overshoot makes `h` **less** a
function of the 8-dim `(pos,vel)` (GRU also rose, 0.357→0.457). `W=2` is the least canonical.

**§3b det vs stoch.** The **deterministic `h` carries essentially all the (pos,vel) code** (det pos-linear ≈ full:
0.80/0.63/0.61 vs full 0.82/0.66/0.64; stoch weak: 0.61/0.42/0.43) and det intrinsic dim ≈ **4** (stoch ~9–11,
full ~8) — det is the compact world-state core, stoch adds spread. Canonicality shift under overshoot: at `W=1` the
**det core is the canonical part** (det fiber 0.35 < stoch 0.81), but overshoot **de-canonicalises the det core**
(det fiber 0.35 → 0.58 → 0.67) while making the stochastic latent somewhat *more* canonical (stoch fiber 0.81 →
0.40 → 0.45) — the objective redistributes structure between det and stoch but nets out **less canonical overall**.

**§4 Editing head-to-head (headline) — the negative REPLICATES.** No non-oracle editor relocates the object on any
`W`: every structural editor's GT next-step RMSE sits at the Unsteered level (~0.27–0.28) vs the true-state swap
(0.24–0.26), ghost ~0.90–1.0. Readout injection is decoder-inert (readout→0, obs-change 0.4–5% of swap). The
`% of swap` looks large for `W=2` (MLP-grad 398%, Global-PCA 117%) **only because the swap denominator collapses**
— the true-state-swap obs-change **shrinks with overshoot** (0.075 / 0.034 / smaller), i.e. the editing *ceiling*
itself drops (more belief inertia + blur). The oracle decoder-gradient still hits low step-0 obs error only by going
**far off-manifold** (global-PCA hull resid 19.6 / 8.2 / — vs real-state ~2.4/0.26/0.30; readout RMSE 59/162/192).

## Reading (interpretation — calibrated)
For the RSSM — the architecture explicitly designed for multi-step latent rollout — a multi-step **training**
objective (i) **does not confer editability** (the §4 pathology is unchanged: decoder-inert probe direction, belief
inertia, off-manifold oracle collapse), replicating the GRU negative; and (ii) **is actively harmful to the model**:
it blurs the decoder (mode-collapse through the stochastic prior), worsens single-step and open-loop prediction,
compresses the linear hull, and reduces linear readability + canonicality. This **strengthens** the GRU finding's
conclusion that the editing difficulty is a **structural** property of these implicit world states, not an artifact
of a next-step loss that a rollout objective would repair — and it adds that, for a KL-regularised latent-variable
model, the rollout objective's main effect is **predictive-quality degradation via blur**, not any editability gain.

## Caveats / shakiness (for the artifact-or-signal call)
- **150-epoch matched-but-undertrained budget** (vs 500 for the refined RSSM); `W=1` recon 0.0247 is close to the
  refined 0.0236, so `W=1` is well-trained, but the overshoot models are compared at the same budget — cross-`W` is
  the trusted axis, absolute values are optimistic.
- **Overshoot best-recon checkpoints are EARLY** (`W=2` ep 64, `W=5` ep 25): single-step recon was best early and got
  *worse* as the overshoot term took over, so `best_model` is the *least*-overshoot-degraded epoch. Using `latest.pt`
  (ep 150) would show **more** degradation → the harm reported here is if anything **understated**.
- In-sample probes; N=64 edits; single overshoot recipe (`n_start=8`, obs+latent overshoot, free-nats 3). A different
  overshoot weighting / KL-balance / longer schedule might blur less, but the monotone degradation across `W` argues
  against a hidden reversal.
- Tangent-rotation curvature is **not distance-normalized** (deferred fix, `directions/curvature-metric-normalization.md`)
  — only compared across `W` within this notebook; do not compare its degrees to the GRU/master.

## Pointers
- Training: `scripts/train_rssm_multistep.py`; checkpoints `runs/rssm_multistep/w{1,2,5}_dset4` (gitignored).
- Notebook: `notebooks/experiments/editability/multistep/multistep_objective_rssm.ipynb` (0 error cells, 12 figures).
- Figures `/tmp/multistep_objective_rssm/`: fig0_sharpness (the blur), fig1_geometry, fig2_recoverability, fig3_fiber,
  **fig3b_det_stoch**, fig4_editor_metrics, fig5{a,b,c}_waterfalls, fig6{a,b,c}_scans.
- Companion (GRU): `findings/editability.md` 2026-07-16 entry + `scratch/2026-07-16-multistep-objective-structure.md`.
