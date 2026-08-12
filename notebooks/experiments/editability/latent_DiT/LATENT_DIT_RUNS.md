# Latent DiT thread — canonical RUN REGISTRY

**The single source of truth for `runs/vae/` and `runs/latent_dit/` run codes.**
Per `CLAUDE.md`: no notebook may use a run code without copying its row into its own definitions table, and
**figures use the descriptive label, never the raw code**. Adding a run means adding its row here in the same commit.

Branch `orthogonal_edit_analysis`. Origin: Sevan, 2026-08-11 — "treat the latent DiT as a wholly separate
architecture". Direction brief: `research/directions/latent-dit-vae.md`. Checkpoints are gitignored.

## The architecture (two frozen-then-composed pieces)

`obs (B,T,128) --VAE encode--> z (B,T,Z) --DiT (window W over latents)--> ẑ_{t+1} --VAE decode--> ôbs_{t+1}`

- **VAE** (`pim/world_models/vae.py`): per-frame MLP encoder/decoder, continuous vector latent, LDM-style tiny
  KL (`kl_weight` 1e-6 → effectively a deterministic autoencoder). Carries **no temporal information** — all
  dynamics live in the DiT. Trained on the *noisy* observations the world models consume.
- **Latent DiT** (`pim/world_models/latent_dit/`): the concat-token DiT core (`pim/world_models/dit/model.py`)
  with `data_transform="identity"`, running on latents normalised by the VAE's measured `latent_scale`
  (the LDM "scale factor"). The VAE is frozen (eval mode, no grads) and stored inside the checkpoint, so a
  latent-DiT checkpoint is a self-contained observation-space world model.
- **Protocol**: implements `HiddenStateModel` in observation space (`step` encodes, `decode` decodes), so the
  whole eval suite / §4 metrics / waterfalls run unchanged. `predict_step` keeps the feedback loop **in latent
  space** (no VAE round-trip per step — that would inject reconstruction error the model never saw).

## State views (renamed vs the pixel DiT)

| view | what it is | dims | writable? |
|---|---|---|---|
| **`latent_window`** (default) | the W·Z carried latent buffer | W·Z | **yes** (`state_from_flat`) |
| `activations` | final-block token features at the current position | d_model | no |
| `kv_cache` | post-RoPE K/V of every layer | n_layers·2·(W−1)·d_model | no |

## Prediction modes (shared with the pixel DiT after the 2026-08-11 API fix)

`mean` (conditional-mean readout, deterministic — **selection metric**) · `sample` (Euler ODE from the fixed
noise-bank vector, deterministic) · **`sample_fresh`** (Euler ODE from per-sample fresh noise; seed
`model.noise_gen`). **Autoregressive rollouts must use `sample_fresh`** — reusing the fixed vector collapses
the rollout (diagnosed 2026-08-11 on the pixel DiT).

## VAE runs (`runs/vae/<code>/best_model.pt`)

Trained by `scripts/train_vae.py` on `datasets/4_fixed_refl_inview/train.h5` (3.42M frames), 80 epochs, batch
4096, AdamW lr 1e-3 cosine, 5% frame-level val split, seed 0.

| code | descriptive label | z | params | recon RMSE vs **noisy** | recon RMSE vs **clean** | latent scale |
|---|---|---|---|---|---|---|
| `vae_z16` | **VAE · z=16** | 16 | 210,080 | 0.1293 | **0.0984** | 1.626 |
| `vae_z8` | **VAE · z=8** | 8 | 205,960 | 0.1379 | **0.0923** | 2.488 |

**Reference scales (test split, N=2000):** the data's own noise floor — RMSE(noisy obs, clean render) — is
**0.1539**. Both VAEs reconstruct *closer to the clean render than the noisy input is*, i.e. the bottleneck
partially denoises; that is expected and accepted (Sevan, 2026-08-11: "the models wash out the noise anyway").
**Quote both numbers**; the clean-target one says whether the code kept the world state.

**Latent retains the world state:** single-frame position readout from `vae_z16` latents is linear R² 0.229 /
MLP R² 0.540 versus **0.261 / 0.515 from the raw 128-d observation** — a 16-d code loses essentially nothing
about object positions. (Single-frame position readout is intrinsically limited; the comparison to raw obs is
the load-bearing part, not the absolute value.)

## Latent DiT runs (`runs/latent_dit/<code>/best_model.pt` — always `best_model.pt`)

Shared recipe: `scripts/train_latent_dit.py`, 150 epochs, batch 256, AdamW lr 3e-4, wd 0, 500-step warmup then
cosine to 0.1×, grad-clip 1.0, seed 0, τ-mixture `p_clean` 0.3 / `p_one` 0.1, 10% val split. Core: **d_model
256**, 4 layers, 4 heads, mlp_ratio 4.0 (d256 matches the GRU hidden size and the pixel DiT, so state-geometry
numbers stay comparable). ~5.0M trainable core params + frozen VAE.

| code | descriptive label (use this in every figure) | VAE | window | ctx frames | best decoded MSE (mean, vs noisy) | vs clean | sampled (fresh) | role |
|---|---|---|---|---|---|---|---|---|
| `0_latent_dit_z16_w4` | **Latent DiT · z=16 · window 4** | `vae_z16` | 4 | 4 | **0.02517** (ep 110) | **0.01174** | 0.0296 | primary; span-matches pixel DiT concat W4 |
| `1_latent_dit_z16_w2` | **Latent DiT · z=16 · window 2** | `vae_z16` | 2 | 2 | 0.02614 (ep 90) | 0.01284 | 0.0303 | minimal context; span-matches pixel W2 |
| `2_latent_dit_z8_w4` | **Latent DiT · z=8 · window 4** | `vae_z8` | 4 | 4 | 0.02552 (ep 140) | 0.01215 | 0.0283 | tighter bottleneck control |

**Result (2026-08-11).** Against the **VAE floor** (0.01672 vs noisy / 0.00971 vs clean), the primary run's
dynamics add only +0.0085 of excess error; on the **clean-target** metric it reads **0.01174 vs the pixel DiT
W4's 0.01186 measured on identical data** — i.e. equal-or-better structure prediction, and the noisy-target
gap (0.02517 vs 0.02443) is the autoencoder, not the world model. 15-step sampled rollouts are stable
(RMSE 0.232–0.241 vs mean mode 0.217). Full gate: `latent_dit_world_state.ipynb`.

## Cross-thread reference runs (defined elsewhere; copied for convenience)

| code | descriptive label | source registry | best val MSE (vs noisy) |
|---|---|---|---|
| `9_dset4_dit_w4_d256` | **DiT concat · d256 · window 4** (pixel space) | `../DiT/DIT_RUNS.md` | 0.02445 |
| `8_dset4_dit_w2_d256` | **DiT concat · d256 · window 2** (pixel space) | `../DiT/DIT_RUNS.md` | 0.02480 |
| `H256` | **GRU · H=256 (reference)** | `../controls/CONTROL_RUNS.md` | 0.02362 |
| `W16` | **transformer · window 16** | `../transformers/TRANSFORMER_RUNS.md` | 0.02359 |

> ⚠ **Reading the comparison fairly.** The latent model's decoded next-step MSE is **floored by the VAE's own
> reconstruction error** (0.1293 RMSE ⇒ ≈0.0167 MSE against noisy targets): it cannot beat a pixel model on the
> noisy-target metric by more than the autoencoder allows, no matter how good its dynamics are. Report
> `val_mse_clean` (decoded prediction vs the clean render) alongside — it is the fairer cross-architecture
> structure metric — and always name the VAE floor when quoting either.
