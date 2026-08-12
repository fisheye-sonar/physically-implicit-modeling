# DiT thread — canonical RUN REGISTRY

**The single source of truth for what every run code under `runs/dit/` means.**
Per `CLAUDE.md`: no notebook may use a run code without copying its row into its own definitions table, and
**figures use the descriptive label, never the raw code**. Adding a run means adding its row here in the same commit.

Branch `orthogonal_edit_analysis`. Origin: Sevan, 2026-08-11 — DiT architecture coverage for the editability
thread (what do "latent world state" and "intervention" mean for a diffusion world model?).
Checkpoints live in gitignored `runs/dit/<code>/best_model.pt` — **always `best_model.pt`**, selected on
mean-mode val MSE (the diffusion loss is not a reliable selection signal; cf. the RSSM best-by-ELBO bug).

## The two variants (both in `pim/world_models/dit/`)

| variant | class | token | conditions on | prediction |
|---|---|---|---|---|
| **concat** (paired-frame tokens) | `DiTModel` (`model.py`) | `concat(obs[t], x_τ(obs[t+1]))` — clean current frame + rectified-flow interpolant of the *next* frame | always-**clean** history (the clean channel guarantees it at every τ pattern) | last token's noised channel is denoised |
| **single-frame** (vanilla diffusion forcing, Chen et al. 2024) | `SingleFrameDiTModel` (`single_frame.py`) | `x_τ(obs[t])` — the (noised) frame itself | history at whatever noise levels training drew (`p_clean = 0.3` per token → a fully clean context is the rare corner: `0.3^(W−1)`) | a pure-noise token appended after the observed frames is denoised |

Both: rectified flow (`v = ε − x₀`), per-token τ mixture (30% τ=0 excluded from loss / 10% τ=1 trains the
mean readout / 60% uniform), AdaLN-Zero τ-conditioning, RoPE, band-causal attention, and two deterministic
prediction modes — **mean** (single forward at τ=1 averaged over an 8-vector fixed noise bank → conditional
mean; the GRU-comparable mode, used for selection) and **sample** (8-step Euler ODE from fixed noise → a
distribution-typical generation; its MSE vs noisy targets is intrinsically ~1.7× worse because it reproduces
a noise realisation — faithfulness, not error).

## ⚠ Window semantics — `window` is NOT frames-of-context, and differs between variants

`window` = attention band width in **tokens**. A concat token carries a frame *pair*, so concat-W conditions
on **W** past frames; a single-frame token spends the last band slot on the prediction, so single-frame-W
conditions on **W − 1** past frames. **Frame-span-matched pairs: concat W2 ↔ single-frame W3, concat W4 ↔
single-frame W5.** Additionally, training's full-sequence banded forward has receptive field up to
`n_layers × (window−1) + 1` frames while inference hard-truncates to the carried window — all quality numbers
below are computed through the *inference* path, so they price that mismatch in.

## Shared training recipe (2026-08-11 batch)

`scripts/train_dit.py`, 150 epochs, batch 256, AdamW lr 3e-4, weight decay 0, 500-step linear warmup then
cosine to 0.1×, grad-clip 1.0, seed 0, 10% of train held out for validation, in-memory loader.
Dataset `datasets/4_fixed_refl_inview` — same data as every other architecture in the thread (2 objects,
40 frames, obs_res 128, obs noise 0.2, position noise 0.04, edit frame 20).

## Architecture (2026-08-11 batch — matched to the MSE transformer / GRU comparison)

`d_model` **256** (= GRU hidden size = MSE transformer width; required for state-geometry comparability —
chance levels scale as `√(d/H)`), 4 layers, 4 heads (64-dim), mlp_ratio 4.0. ~5.0M params — the AdaLN
projections add ~1.6M over the MSE transformer's 3.23M at equal width; flag in write-ups, don't match.
Residual point 0 is **not** the shared encoder port (`token_proj` ingests the noised channel too) — the DiT
residual stream mixes world-state with the denoising iterate at every depth; there is no hyperparameter fix.

## Runs — 2026-08-11 batch (the analysis targets)

| code | descriptive label (use this in every figure) | variant | window | past frames seen | best val MSE (mean mode) | role |
|---|---|---|---|---|---|---|
| `8_dset4_dit_w2_d256` | **DiT concat · d256 · window 2 (2 ctx frames)** | concat | 2 | 2 | **0.02480** (ep 145) | (a) minimal-context, span-matches transformer W2 |
| `9_dset4_dit_w4_d256` | **DiT concat · d256 · window 4 (4 ctx frames)** | concat | 4 | 4 | **0.02445** (ep 145) | (a) intermediate, span-matches transformer W4; used by `../input_grad_steering/input_grad_steering_dit.ipynb` |
| `10_dset4_dit_sf_w3_d256` | **DiT single-frame · d256 · window 3 (2 ctx frames)** | single-frame | 3 | 2 | **0.02504** (ep 150) | (b) vanilla diffusion forcing, span-matched to `8_…w2` |
| `11_dset4_dit_sf_w5_d256` | **DiT single-frame · d256 · window 5 (4 ctx frames)** | single-frame | 5 | 4 | **0.02424** (ep 130) | (b) vanilla diffusion forcing, span-matched to `9_…w4` |

The (a)-vs-(b) comparison at matched frame span isolates **the tokenization/training-conditional package**
(clean-channel concat vs standard diffusion forcing) as the only variable.

**Result (2026-08-11, single-step prediction quality):** the package deal is a wash — concat 0.02480/0.02445 vs
single-frame 0.02504/0.02424 at matched span (2/4 ctx frames). Vanilla diffusion forcing trains fine despite the
rare-clean-context training distribution; the concat's clean-history guarantee buys no measurable next-step
accuracy here. Any difference between the designs will have to show up in rollout robustness or
editability/guidance behaviour, not the quality gate.

## Legacy runs — 2026-07-09 batch (concat variant only; pre-date the d256 width-match convention)

All 150 epochs unless noted; d128 unless noted; window 16 unless noted; 1.32M params (d128) / 4.23M (d192).

| code | config delta | best val MSE (mean mode) | note |
|---|---|---|---|
| `0_dset4_dit_baseline` | trained before `p_one` existed | 0.02659 | superseded by `4_…pone` |
| `1_dset4_dit_w4` | window 4 | 0.02602 | |
| `2_dset4_dit_w32` | window 32 | 0.02576 | |
| `3_dset4_dit_small` | d64 · 3 layers (278k params) | 0.04282 | capacity floor |
| `4_dset4_dit_pone` | — (the d128 reference) | 0.02581 | |
| `5_dset4_dit_pone_400ep` | 400 epochs | 0.02526 | |
| `6_dset4_dit_big` | d192 · 6 layers · 6 heads | 0.02411 | |
| `7_dset4_dit_big_400ep` | d192 · 6L · 400 epochs | **0.02390** | best DiT to date; used for the sampling-modes waterfall |

## Cross-thread reference runs (defined elsewhere, copied here for convenience)

| code | descriptive label | source registry | why it is here |
|---|---|---|---|
| `H256` | **GRU · H=256 (reference)** | `../controls/CONTROL_RUNS.md` | width-matched GRU; best val MSE **0.02362** — the quality bar |
| `W2`/`W4`/`W16` | **transformer · window 2/4/16** | `../transformers/TRANSFORMER_RUNS.md` | the MSE (non-diffusion) transformer the DiT is architecture-matched against; best val 0.02396/0.02372/0.02359 |
