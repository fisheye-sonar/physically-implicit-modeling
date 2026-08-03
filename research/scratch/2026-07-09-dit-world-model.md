# 2026-07-09 — DiT world model: implementation + first training (engineering)

**Branch:** `diff_transformer`. **Status:** DRAFT — full-eval numbers land below as they
complete. Engineering note (substrate), not a science claim; nothing here is promoted.

## What was built

A causal diffusion-transformer world model (`pim/world_models/dit/`), trained exactly like
the GRU/RSSM (next-step prediction, one observation at a time, dataset 4), implementing the
`HiddenStateModel` protocol **unchanged**. ~1.3M params (d=128, 4 layers, 4 heads, W=16).

Formulation — *diffusion forcing over a causal token stream*:
- One token per frame t: `concat(obs_t, x_τ(obs_{t+1}))` where `x_τ = (1-τ)·obs_{t+1} + τ·ε`
  (rectified flow); per-token τ with AdaLN-Zero conditioning; RoPE; **banded causal
  attention** (window W). Model predicts flow velocity `v = ε − obs_{t+1}` at every position
  in one training forward (per-position independent τ; with p=0.3 a position is exactly
  clean τ=0 — the pattern deployment produces for past positions).
- Training: `diffusion_loss(obs)` = flow-matching MSE. Loss masked at exact-clean positions.

## The state question (Sevan's KV-cache framing)

Since attention is windowed and recomputed each step, the minimal carried-forward object is
**the window of the last W raw observations** — the KV cache is a deterministic *function*
of it (a cache, not extra state). Implemented as `model.state_view` (runtime toggle):
1. `obs_window` (default) — flat = last W obs (W·R = 2048 dims). Exact, invertible
   (`state_from_flat` works → all editors/controllability eval run on this view).
2. `activations` — final-block token features at the current position, computed at τ=1 from
   fixed noise (d=128 dims). The GRU-`h` analogue: the learned representation the model
   builds from context alone. Read-only.
3. `kv_cache` — post-RoPE K/V, all layers, W−1 completed tokens (n_layers·2·(W−1)·d ≈ 15k
   dims). What an incremental implementation would carry. Read-only; subset probing only.

Implication worth flagging: unlike GRU/RSSM, the DiT's *canonical* state is literally raw
observations — "recoverability from the state" in the default view measures the
observations, not a learned compression. The learned representation lives in views 2/3.
This is a real architectural asymmetry, not a framework artifact.

## Two deterministic prediction modes (`model.predict_mode`)

Key diagnostic from the first training run: **flow-matching loss ↓ monotonically while
sampled next-step MSE ↑ after ~epoch 10** (0.045 → 0.059 plateau). Not a bug: the model
transitions from conditional-mean-like predictions to *distribution-typical* samples that
include a hallucinated realisation of the observation noise. (Noise energy after [0,1]
clipping is E[(obs−clean)²] ≈ 0.0237, not σ²=0.04; a perfect sampler scores ≈ 2× that vs
noisy targets.)

Resolution — both modes deterministic, seeded noise buffers, runtime toggle:
- `"mean"` (default): at τ=1 the optimal velocity field is v*(x,1) = x − E[x₀|ctx] for
  *every* x, so `ε − v̂(ε, 1, ctx)` reads out the **conditional mean**; averaging over a
  fixed bank of 8 ε's washes out per-ε transport idiosyncrasy (1→4 ε large gain, 4→16
  marginal). One batched forward. GRU-comparable; analogue of RSSM prior-mean eval mode.
- `"sample"`: K=8-step Euler ODE from fixed ε → distribution-typical prediction. For
  generative-quality questions; its MSE is intrinsically worse (faithfulness, not error).

Model selection = mean-mode val MSE (diffusion loss is NOT a selection signal — same
lesson as the RSSM best-by-ELBO bug, new mechanism).

## Numbers so far (val split of dset4 train.h5, 2048 samples)

| model / mode | MSE vs noisy | MSE vs clean |
|---|---|---|
| noise-energy floor | 0.0237 | 0 |
| DiT ep150 mean ×8-bank | 0.0266 | **0.0139** |
| DiT ep150 sample (K=8 ODE) | 0.0597 | 0.0465 |
| GRU ref (dset3-brighter, 400 ep) | — | ~0.0152 (near-horizon clean, prior compare) |

Baseline run: `runs/dit/0_dset4_dit_baseline` (150 epochs, **7.5 min** on the 5090; best
ckpt ep 125, val mean-MSE 0.0266). Curves healthy: mean-MSE monotone ↓, sample-MSE ↑ to
~0.06 as distribution-faithfulness improves — both understood, both logged
(`val_mse_mean` / `val_mse_sample` in metrics.jsonl).

## Substrate side-findings

- **The gzip-HDF5 dataloader was the training bottleneck all along** (189 ms/batch vs
  6 ms/batch model fwd+bwd → 30×). Added `in_memory=True` option to `build_dataloaders`
  (dataset ≈ 1.8 GB in RAM); train_dit defaults to it; `--in-memory` flag added to
  train_gru.py. GRU/RSSM historical training times were dataloader-bound.
- Pre-existing ruff failures in tests/test_{extractors,renderer,sim}.py (E741/F401/F841) —
  not from this work; left untouched, flagged for Sevan.

## Fair comparison + iteration (same val subset, mean mode, clean targets)

| run | noisy | clean | note |
|---|---|---|---|
| GRU dset4 400ep (`runs/gru/7_dset4_gru_400epochs`) | 0.0236 | **0.0109** | the target |
| DiT baseline 150ep | 0.0266 | 0.0144 | |
| DiT + p_one=0.1 150ep (`4_dset4_dit_pone`) | 0.0258 | 0.0138 | τ=1 readout training helps |
| DiT p_one ×1-ε readout | 0.0286 | 0.0165 | bank averaging still needed (τ=1 training did not make the single-ε readout ε-independent) |

`p_one`: new mixture component in `diffusion_loss` — with prob 0.1 a position trains at
exactly τ=1, which regresses v̂(ε,1,ctx) → ε − E[x₀|ctx], i.e. directly trains the
conditional-mean readout. Adopted as default.

## Full eval suite on DiT baseline (outputs/eval/0_dset4_dit_baseline/20260709_174455)

Whole `run_eval.py` pipeline ran end-to-end through the unchanged protocol (28 figures).
- prediction next-step (noisy, test) 0.0265; rollout smoothness MLP 0.913 vs GT 0.892.
- **recovery on obs_window state: linear 2.34 / MLP 0.87 position-MSE** — poor by design:
  the canonical DiT state IS raw observations; depth is not a linear function of pixels.
  The learned-representation recoverability question moves to the `activations` view
  (probing session, not done here).
- **controllability: steered 0.0880 ≈ unsteered 0.0893** — linear-probe steering of a raw
  obs-window is semantically meaningless, so the editor is a no-op (injection error 2e-13,
  the roundtrip itself is exact). Architectural asymmetry to study later: for the DiT the
  *natural* state edit is trivial and exact — render the counterfactual observation and
  write it into the window buffer. "Editability" changes meaning entirely for this class
  of model (no entangled hidden code to fight; the state is the interface).

## Final results — GRU PARITY REACHED (same val subset, mean-mode ×8-ε, next-step)

| run | params | noisy MSE | clean MSE |
|---|---|---|---|
| **GRU-256 dset4 400ep** (`runs/gru/7_dset4_gru_400epochs`) | 0.46M | 0.0236 | **0.0109** |
| **DiT flagship: d192 L6 W16, 400ep** (`runs/dit/7_dset4_dit_big_400ep`) | 4.23M | 0.0239 | **0.0112** |
| DiT d192 L6 150ep (`6_dset4_dit_big`) | 4.23M | 0.0241 | 0.0116 |
| DiT d128 L4 400ep (`5_dset4_dit_pone_400ep`) | 1.32M | 0.0253 | ~0.0134 |
| DiT d128 L4 150ep = **default config** (`4_dset4_dit_pone`) | 1.32M | 0.0258 | 0.0138 |
| DiT d128 **W32** (`2_dset4_dit_w32`) | 1.32M | 0.0258 | 0.0136 |
| DiT d128 **W4** (`1_dset4_dit_w4`) | 1.32M | 0.0260 | 0.0140 |
| DiT **d64 L3** (`3_dset4_dit_small`) | 0.28M | 0.0428 | 0.0311 |

Lever analysis (each isolated):
- **Capacity is the binding constraint**: d64 collapses (best ckpt at ep 20, then worse —
  can't even hold the mean task), d128 lands 27% off GRU, d192/L6 reaches parity (2.8%).
- **Window is NOT**: W=4 ≈ W=16 ≈ W=32 at d128 (0.0260/0.0258/0.0258). The task is
  constant-velocity; ~4 frames suffice, and extra noise-averaging context goes unused.
  The GRU's edge was never history length.
- **Budget is NOT (at d128)**: 400 vs 150 epochs buys only −0.0005.
- **p_one (τ=1 readout training) is real but modest**: −0.0008 at d128.

Defaults decision: train_dit.py keeps **d128/L4/W16 (1.32M, 3s/epoch)** as the default —
"as small as possible" per Sevan's brief, 0.0138 clean, right for iteration. The parity
recipe is `--d-model 192 --n-layers 6 --n-heads 6 --n-epochs 400` (≈35 min).

## Round-2 artifacts

- Flagship full eval: `outputs/eval/7_dset4_dit_big_400ep/<ts>/` (28 figures + metrics).
- GRU-dset4 full eval: `outputs/eval/7_dset4_gru_400epochs/20260709_185117/`.
  GRU recovery from h: linear 0.547 / MLP 0.169 vs DiT obs_window 2.34 / 0.87 — the
  learned-compression vs raw-window asymmetry, as expected. GRU steering is ALSO a no-op
  (0.0865 vs 0.0870 unsteered) — DiT's dead steering is not anomalous.
- 3-way comparison (GRU / RSSM / DiT mean+sampled): `outputs/eval/compare_gru_rssm_dit/`
  (horizon_curve.png, waterfalls.png, sharpness.txt) via `compare_rollouts.py --dit`.
- Activations-view smoke probe (2k samples): linear recovery 1.23 (H=128) vs 2.27 raw
  obs-window (H=2048) — the learned representation is markedly more linearly decodable
  than raw obs. (MLP numbers unreliable at 2k samples; proper probing = its own session.)

## 3-way rollout comparison (outputs/eval/compare_gru_rssm_dit/, n=2000, ctx=10)

Near-horizon open-loop clean MSE (H1–5): **GRU 0.0150 · DiT-mean 0.0162 · RSSM-mean
0.0173** · RSSM-samp 0.0247 · DiT-samp 0.0369 · persistence 0.0482. The DiT beats the
refined RSSM and lands 8% behind the GRU on the canonical rollout metric.

TV sharpness (GT clean = 2.19): **DiT-mean 2.37 — closest to GT of all models** (GRU 2.79,
RSSM 2.86–2.90). In the waterfalls DiT-mean keeps both objects sharp and distinct where
the GRU smears/fades the second object.

Two honest caveats, both visible in the figures:
1. **Long-horizon drift**: DiT-mean tracks GRU to ~step 13 then climbs faster, ending near
   RSSM-sampled at step 30. Consistent with the bounded-W window: past W self-generated
   frames the model has literally zero memory of real context, while GRU/RSSM carry an
   unbounded recurrent state. Directly relevant to the RESEARCH.md *persistence* north
   star — a crisp architectural contrast (bounded-window attention vs recurrence) to
   study, not a bug.
2. **Sample-mode noise resonance**: DiT-sampled open-loop rollouts develop fixed vertical
   noise streaks (TV 14.8 vs GT 2.19) — the fixed-ε₀ noise texture is re-generated each
   step and fed back, and the pattern locks in. Single-step samples are fine; the
   compounding is an open-loop phenomenon. Worth a small study before any generative-
   quality claims about sample mode.

## Open questions (for probing sessions, not this one)

- Probe the `activations` view vs `obs_window` view: does the DiT's learned representation
  linearly encode position better than raw obs? (The GRU-comparable recoverability
  question.) Velocity from a single window is trivially available (W frames!) — the
  interesting contrast with the GRU's "velocity is temporal" story.
- kv_cache view editing (would require carrying edited KV forward instead of recomputing —
  deliberate future work; state_from_flat rejects it today).
- W=4 vs W=16 vs W=32: how much history does the DiT actually need? (W≥2 suffices for
  constant-velocity in principle.)
