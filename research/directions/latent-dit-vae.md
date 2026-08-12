# Direction: Pre-trained VAE + latent DiT — does a semantic bottleneck make probe gradients controllable?

**Tag:** `[architecture]` · **Sub-question:** 3 (editability) + architecture coverage · **Status:** **EXECUTED 2026-08-11** (results in `research/scratch/2026-08-11-latent-dit.md`; registry
`notebooks/experiments/editability/latent_DiT/LATENT_DIT_RUNS.md`) ·
**Complexity:** medium (~1–2 focused days; individual pieces are hours each)

## Why (the hypothesis this tests)

The 2026-08-11 input-grad-steering results (`notebooks/experiments/editability/input_grad_steering/`, scratch
note `scratch/2026-08-11-input-grad-steering.md`) showed that frozen-probe gradients on the **128-d observation
surface** are adversarial-dominated on GRU, MSE transformer, and pixel-space DiT alike: the readout is always
fully driven, the perturbation is fuzz (cos(δ, Δ_true) ≈ 0.1–0.25), and even the DiT's denoiser only converts
the failure from "inert" to "partial duplication" (best probe-only Edit Index −0.18 vs oracle-content +0.12 on
the same surface). One candidate explanation is **dimensionality of adversarial freedom**: 128-d observation
space has a large subspace of readout-flipping fuzz directions orthogonal to content.

A **z = 8–16 VAE latent** has almost no room for that — nearly every direction is semantic by construction, and
the frozen decoder is a second, unconditional manifold projector (anything done in z decodes to a valid-looking
observation). Transplanting the same editors to z-space is therefore the cleanest test of whether
readable≠controllable is about **representation geometry** (fixable by compression) or **belief dynamics** (the
ghost persists even in z — which the flat-in-τ probe grid of `DiT/dit_world_state.ipynb` hints at, since the
DiT's position belief is context-driven). Either outcome sharpens the claim. Secondary motivation: this is the
actual modern video-model recipe (VAE + latent DiT), completing architecture coverage.

## Agreed specs (Sevan, 2026-08-11)

- **VAE:** continuous per-frame **vector latent, z ∈ 8–16**, low-KL (LDM-style: near-deterministic AE with a
  small KL regularizer; no VQ, no perceptual/GAN loss at this resolution). Encoder/decoder small MLP or 1D conv
  over the 128-ray scan. Trained on the train split (~4M frames), minutes on the 5090.
  - Noise handling: a tight z will largely **denoise** — accepted ("the models wash out the noise anyway").
    Consequence to keep in mind: the reproduce-the-noise-realisation story of pixel sample-mode disappears, and
    RMSE floors shift. **Report the VAE's own reconstruction RMSE (vs clean AND vs noisy) as the reference
    scale everywhere.**
- **Latent DiT:** the existing DiT trunk unchanged at **d_model 256** (geometry comparability), `input_dim` = z.
  Both variants (concat / single-frame) apply. Precompute encoded latents for the whole dataset (in-memory,
  tiny). Add an LDM-style **latent scale factor** so the flow endpoints are balanced (replaces the [0,1]→[−1,1]
  rescale, which no longer applies).
- **Protocol wrapper:** composite `LatentDiT` (frozen VAE + DiT core) implements `HiddenStateModel` —
  `step()` encodes incoming obs, `decode()` decodes predictions back to obs space — so the entire eval suite,
  §4 metrics, and waterfalls run unchanged. Loader dispatch + checkpoint format like `single_frame`.
- **Registry:** new runs get rows in `notebooks/experiments/editability/DiT/DIT_RUNS.md` in the same change;
  notebooks live in `DiT/` except editor experiments, which live in `input_grad_steering/`.

## Fold in at the same time (owed API fix)

`predict_mode="sample"` reuses a fixed start-noise vector; iterated rollouts collapse (diagnosed 2026-08-11,
fixed ad-hoc by mutating `_eps_bank` per step, with the redrawn vector shared across the batch). Add a proper
**fresh-noise sampling mode with per-sample noise** to both DiT variants (+ tests) instead of the external
mutation hack.

## Experiment plan (in order)

1. **Quality gate:** VAE recon RMSE; latent-DiT mean-mode val MSE (decoded, vs noisy targets) against the
   pixel DiT (0.02445 @ concat W4 d256) and GRU bar (0.02362) — expect the tight latent to *beat* the noisy-target
   floor comparison in a way that needs the clean-target metric to interpret; report both.
2. **Sampled-rollout check** (the `dit_world_state.ipynb` Figs 2–3 protocol) — latent sampling stability.
3. **Probe grid in z-space** ((residual point × τ), same notebook pattern) — is position readable, and is it
   still context-driven?
4. **The point: editors in z-space** (`input_grad_steering/` notebook): Input Grad on the latent history window;
   pause–optimize–resume Latent Grad at the grid's best point; Render write oracle (encode the clean edited
   render). Same §4 scorecard + waterfall (decoded to obs space). The headline comparison: cos(δ_z decoded,
   Δ_true) and Edit Index vs the pixel-space ladder (−0.52 / −0.05 / −0.18).

## Open (decide at build time)

- z = 8 vs 16 (start 16, drop to 8 if recon is comfortable); KL weight (start ~1e-4·recon scale).
- Whether the VAE trains on noisy obs only (default) or with clean-render targets as an ablation arm
  (an explicitly *denoising* AE — changes the semantics of "faithful generation"; flag, don't default).
- Whether a latent GRU control is wanted for symmetry (same VAE, GRU dynamics) — cheap, decides whether any
  z-space editability gain is diffusion-specific or bottleneck-specific. Probably yes; confirm with Sevan.


---

## Outcome (2026-08-11) — hypothesis refuted

Built, trained and analysed in one session. The quality gate passed (VAE recon 0.0986 vs clean; decoded
next-step 0.01174 vs clean = the pixel DiT's 0.01186; stable `sample_fresh` rollouts; probe R² 0.81 > pixel's
0.70). **The editability result is a clean negative for the compression hypothesis**: cos(δ, Δ_true) is
+0.118…+0.168 for latent-space writes versus +0.11…+0.22 in pixel space, and every Edit Index matches the pixel
DiT arm-for-arm, with both oracles (+0.12 single-frame, +0.71 velocity-consistent window) identical. The probe-
gradient failure is therefore **belief dynamics, not representation geometry**. Remaining optional arms from
this brief that were NOT run: the latent-GRU control, and the editors on the z=8 / window-2 latent runs.
