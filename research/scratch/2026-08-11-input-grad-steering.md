# 2026-08-11 — Input Grad Steering (GRU + transformer): probe gradients on the INPUT are adversarial, not semantic

> **⚠ Probe-standard fix (same day, after Sevan's flag):** all MLP R² values in this note and the notebooks'
> first executions under-read — the standard MLP probe was undertrained (~30 Adam steps; `STD_EPOCHS` 30 → 300
> in `pim/extractors/standard.py`, validated GRU h MLP 0.17→0.89 vs linear 0.81). Linear R² (and every steering
> result — all steering probes are linear) is unaffected. DiT-side notebooks re-executed with the fix.

**Thread:** `notebooks/experiments/editability/input_grad_steering/` (README + 2 notebooks, both executed clean).
**Models:** transformer · window 16 (`runs/transformers/W16`), GRU · H=256 (`runs/controls/H256`).
Dataset 4, N=64 edits, K=15, canonical §4 metrics. Status: scratch, not promoted.

**Question (Sevan's):** backprop a frozen linear position probe through the network to the input observation,
drive the readout to the post-edit positions — does the observation change *semantically* (bump moves), and does
the rollout follow? Plus: is δ aligned with the "true edit direction" Δ_true = clean edited render − actual obs?

**Result — clean negative with an informative oracle gap:**
- The probe readout is always driven to (near) target: residual 3.4→0.06–0.19 (transformer), 3.45→0.25–0.42
  (GRU, λ≤0.1). The optimization never fails; the *direction* is what fails.
- cos(δ*, Δ_true): transformer +0.21…+0.27 (74–78°), GRU +0.09…+0.13 (82–85°); raw first gradient ≈ 0 cosine
  (only the transformer encoder port is nonzero at +0.13). Shuffled-pair chance ≈ 0. Visually (Fig 2 both
  notebooks): broadband adversarial fuzz, ghost bump untouched; GRU λ=0 saturates to binary spikes with
  ‖δ‖ = 4.17 > ‖Δ_true‖ = 3.45.
- Generation response: Edit Index transformer −0.69 → −0.50 (best), GRU −0.68 → −0.44 (best); fidelity ≈ 1.0
  everywhere (ignored, not destroyed). Residual point ℓ and n (1 vs all frames) are second-order.
- **Oracle on the same write surface** (Render write @1 = newest frame ← clean edited render): transformer
  **+0.27**, GRU **−0.01** (matches First Obs TF −0.08 — belief inertia). So the surface works; the gradient's
  content doesn't. The transformer's buffer is a stronger write channel than one GRU recurrent update.

**Interpretation (scratch):** extends readable≠controllable from h-space to input space — even on the one
surface that is fully on-manifold-parameterizable, the probe gradient points off-manifold. Sharperns the DiT
motivation: probe-guided *diffusion* (classifier guidance + denoising steps that project back to the manifold)
supplies exactly the missing operator. New editor + Δ_true/cosine definitions in the thread README; fold into
`METRICS_AND_EDITORS.md` if the editor recurs.

**DiT leg (`input_grad_steering_dit.ipynb`, model `9_dset4_dit_w4_d256`) — the guidance hypothesis FAILS in
its naive form.** (1) The history-write Input Grad arm replicates the negative (cos +0.11…+0.15, index
−0.65→−0.52). (2) **Probe-guided sampling** (classifier guidance in the 8-step Euler loop, fresh noise):
g=10 → index −0.14 with collateral already 0.45; g≥100 → −0.05 with collateral 0.55 and rail-to-rail saturated
guided frames — **degradation, not landing**. Decisive control: **early-τ-only guidance (kicks only while
τ≥0.5, low-noise half denoises freely) is IDENTICAL to full-schedule** — so the failure is the gradient
*direction* (adversarial at every noise level), not the kick timing; 4 deterministic Euler steps have far too
little contraction to repair a strong kick. (3) The genuinely diffusion-specific observation: the *rollout*
re-coheres corrupted frames into clean object-like bands — the projector is real — but to the nearest coherent
world (split/duplicated objects, ghost intact), not the target world. (4) Oracle render write @1: +0.12
(between GRU −0.01 @1-frame-state and transformer W16 +0.27 @20-frame-state — consistent with W4's 4-frame
buffer). Next candidates that attack the real failure: SDE/renoise–denoise sampling, render-space (decoder)
guidance instead of the state probe, many-small-kicks-across-frames.

**EXTENSION (same day, Sevan-directed) — (ℓ × τ) probe grid + pause–optimize–resume latent steering.**
- New notebook `notebooks/experiments/editability/DiT/dit_world_state.ipynb` (+ `resid_sink` hook in
  `pim/world_models/dit/model.py`, tested): held-out linear position R² by residual point × diffusion time,
  computed on actual fresh-noise Euler iterates. **Depth-monotone (0.26 → 0.70–0.76) and FLAT across τ** —
  the DiT's position belief is context-driven; the iterate contributes only a small τ=0 bump. (Standard MLP
  probe under-fits these states at N=500 — linear panel is load-bearing.)
- `input_grad_steering_dit.ipynb` extended: **whole-window arm (n=4): −0.52, identical to n=1** (joint
  optimization over all carried frames still adversarial). **Latent Grad Steering @(L3, τ_pause)** — pause the
  ODE at τ ∈ {0.75, 0.5, 0.25}, Adam the iterate to an exact readout of a probe fit AND verified at that same
  (L, τ) (R² 0.70), resume: Edit Index **−0.18/−0.20/−0.26** vs unguided-sample control −0.51, collateral
  0.28–0.35, fidelity ~1.05 — **the best probe-only editor on any architecture in the thread**, but the
  waterfall shows the mechanism is **duplication, not relocation**: a new persistent band appears near the
  target while the ghost survives (the ghost lives in the clean context tokens, which iterate steering never
  touches — matching the flat-in-τ grid). Ladder on one probe: raw input grad −0.52 (ignored) → per-step
  guidance −0.05 (destroyed) → pause–optimize–resume −0.18 (partial, duplicated). Escalations if continued:
  SDEdit-style re-noise of the *history*, repeated latent edits over several frames, render-space objective.

**EXTENSION 2 (same day) — full-Euler robustness + MLP-probe variants (`input_grad_steering_dit.ipynb`
cells [10]–[12]).** (1) All verdicts are rollout-mode-invariant under full fresh-noise Euler rollouts (Latent
Grad −0.18/−0.20/−0.26 identical; oracle +0.13; sampling alone softens unedited belief −0.65→−0.51). (2) With
the FIXED standard MLP probes (R² 0.85–0.90 at the steering points): **Input Grad · MLP −0.31 vs linear −0.52**
— best history-write on any architecture in the thread, cos +0.22, waterfall shows ghost dimming + a new target
band from a clean-frame edit — while **latent steering is probe-capacity-invariant** (MLP −0.29/−0.22/−0.21 ≈
linear). Reading: the latent plateau ≈ −0.2 is set by belief dynamics (clean context tokens), not readout
quality; the clean input surface DOES respond to a better-shaped gradient.

**EXTENSION 3 (same day, Sevan-directed) — whole-window arms (`input_grad_steering_dit.ipynb` cell [13]).**
(1) Input Grad · MLP over the whole 4-frame window: **−0.31, identical to n=1** — probe gradients don't exploit
surface width. (2) **Counterfactual window write (n=4, oracle) — Edit Index +0.71, the edit essentially fully
lands**: all 4 window frames ← clean renders of the velocity-consistent counterfactual (edited object keeps its
own velocity on a path constant-offset to hit the target at ef). Target RMSE 0.092, ghost 0.096, collateral at
the unsteered baseline (0.110), GT-traj RMSE 0.206 **better than unsteered** (0.303). Waterfall: object ON
target with GT-matched motion, no ghost. **Implication: the Render write @1 ceiling (+0.12) was never belief
inertia — it was conflicting velocity evidence** (one time-offset frame vs three unedited); consistent evidence
across the window removes it. This is the DiT's full-editability ceiling for probe-only editors to chase
(current best probe-only: −0.31), and matches/beats the repo's best editors (GRU Decoder Grad k=15 +0.83→+0.77,
Counterfactual Overwriting +0.70→+0.45) with better fidelity.

**Side results (DiT thread, same day):**
- Sample-mode autoregressive collapse (vertical stripes) diagnosed: the FIXED ODE start-noise reused every
  rollout step; fresh noise per step restores coherent stochastic rollouts (RMSE vs clean 0.257 → 0.186,
  beating mean mode's 0.192). Model API fix pending (currently done by mutating `_eps_bank` externally).
- (a)-vs-(b) tokenization at matched frame span is a WASH on single-step quality: concat 0.02480/0.02445 vs
  single-frame (vanilla diffusion forcing) 0.02504/0.02424 (2/4 ctx frames; GRU bar 0.02362). Registry:
  `notebooks/experiments/editability/DiT/DIT_RUNS.md`.
