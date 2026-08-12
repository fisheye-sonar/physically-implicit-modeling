# 2026-08-11 — VAE + latent DiT: the compression hypothesis is REFUTED

**Thread:** `notebooks/experiments/editability/latent_DiT/` (registry + world-state notebook) and
`notebooks/experiments/editability/input_grad_steering/input_grad_steering_latent_dit.ipynb` (editors).
Direction brief: `research/directions/latent-dit-vae.md` (now executed). Status: scratch, not promoted.

**What was built** (Sevan-directed, "treat the latent DiT as a wholly separate architecture"):
`pim/world_models/vae.py` + `scripts/train_vae.py` (per-frame MLP VAE, continuous vector latent, LDM-style tiny
KL, measured `latent_scale`); `pim/world_models/latent_dit/` + `scripts/train_latent_dit.py` (frozen VAE + the
concat DiT core with a new `data_transform="identity"` flag, implementing `HiddenStateModel` in observation
space so the whole eval suite runs unchanged); tests `tests/test_latent_dit.py` (17) and new DiT-core tests
(40 total). **Also folded in the owed API fix**: `predict_mode="sample_fresh"` with per-sample fresh noise and
a seedable `model.noise_gen`, replacing the `_eps_bank` mutation hack in every notebook.

**The gate passes** (`latent_dit_world_state.ipynb`): VAE recon RMSE 0.1294 vs noisy / **0.0986 vs clean**
(dataset noise floor 0.1541 — it denoises); 16-d code retains position as well as the raw 128-d observation
(MLP R² 0.540 vs 0.515); decoded next-step MSE **0.02517 vs noisy** with a **VAE floor of 0.01672**, and
**0.01174 vs clean vs the pixel DiT W4's 0.01186 on identical data** — equal-or-better structure prediction;
`sample_fresh` rollouts stable (0.232–0.241 vs mean 0.217). Probe grid: linear R² **0.807–0.810** at late
residual points (pixel DiT 0.70), **flat across τ** — replicating "the position belief lives in the context
pathway, not the iterate".

**THE RESULT — an 8× semantic bottleneck does not make probe gradients semantic.**
- cos(δ*, Δ_true) for **latent-space** writes: **+0.118 … +0.168** (80–83°). Same model, **observation-space**
  writes: +0.146 … +0.212. Pixel DiT: +0.11 … +0.22. All shuffled-pair chance ≈ −0.03. **No improvement.**
- Edit Index, probe-only editors: obs-grad −0.30 … −0.44, **latent-window grad −0.38 … −0.48**, iterate grad
  −0.13 … −0.23 — matching the pixel DiT arm-for-arm (−0.31/−0.52, −0.18/−0.21). Probe capacity (linear vs
  MLP) and window width move things ≲0.1, as in pixel space.
- **Both oracles reproduce their pixel-space values exactly**: Render write @1 **+0.12**; velocity-consistent
  **Counterfactual window write +0.71** (collateral at the unsteered baseline, GT-traj RMSE 0.200 < unsteered
  0.303). The architecture is fully editable through consistent multi-frame evidence.
- **Most readable state in the thread** (linear 0.810 / MLP 0.905 on activations, vs pixel 0.70/0.85) and
  **not one bit more controllable** — the sharpest readable≠controllable demonstration we have.

**Interpretation (scratch).** The "dimensionality of adversarial freedom" explanation is refuted for this
world: the failure survives compression to a space where nearly every direction should be semantic, on a model
that is *more* readable and whose decoder is an extra unconditional projector to valid observations. What
remains is **belief dynamics** — the ghost is carried by the clean context frames, and only evidence consistent
*across the window* removes it. Practical corollary: stop searching for a better space to take the gradient in;
search for objectives that synthesise multi-frame velocity-consistent evidence (SDEdit-style re-noise/re-denoise
of the whole window, a render-space objective, or repeated small edits across generated frames).

**Owed / not done:** `single_frame.py` did not get the `resid_sink` hook (only the concat core needs it today);
no latent GRU control (the direction brief's optional arm — would separate bottleneck effects from diffusion
effects); z=8 and window-2 latent runs exist and passed the quality gate but were not put through the editors.
