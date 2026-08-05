# Transformer thread — canonical RUN REGISTRY

**The single source of truth for what every run code in `notebooks/experiments/editability/transformers/` means.**
Per `CLAUDE.md`: no notebook may use a run code without copying its row into its own definitions table, and
**figures use the descriptive label, never the raw code**. Adding a run means adding its row here in the same commit.

Branch `michael_controls`. Origin: Sevan, 2026-08-04 — paper-1 architecture coverage.
Direction: `research/directions/transformer-world-state.md`.
Checkpoints live in gitignored `runs/transformers/<code>/best_model.pt`.

## Shared training recipe (every run below — `window` is the ONLY variable)

`scripts/train_transformer.py`, pre-norm causal transformer, **300 epochs**, batch 256, AdamW **lr 1e-3**,
weight decay 1e-4, **5% linear warmup then cosine decay to zero**, **grad-clip 1.0**, seed 0, 10% of the train
split held out for validation, in-memory GPU-resident loader.

Deliberately mirrors `scripts/train_gru.py` — same dataset, same MSE next-step objective, same loader, same
checkpoint format — so **"recurrence vs attention" is the only architectural variable**. The two deviations
(warmup, grad clipping) are the ones a pre-norm transformer at this depth actually needs; without them it
trains far less stably than the GRU.

**Why lr 1e-3 and 300 epochs.** A 25-epoch LR sweep on `W16` gave val loss 0.02426 (3e-4) / 0.02369 (1e-3) /
0.02349 (3e-3). 3e-3 was marginally best at 25 epochs but noticeably less stable early; 1e-3 with a longer
cosine schedule reaches the same place with margin. The GRU's 400-epoch reference val loss is **0.02362**,
so the schedule is set to clear that bar rather than to a round number.

## Shared world settings (all runs)

Dataset `datasets/4_fixed_refl_inview` — the same data every other architecture in this thread is trained on.
2 objects, 40 frames, `obs_res=128`, open boundary, fixed reflectivities, always-in-frustum, radius 0.5,
speed 0.05–0.12. **Observation noise 0.2, position noise 0.04.** Splits 90k train / 10k val / 10k test /
10k edits, base seed 0, **edit frame 20**, edits are in-frustum teleports of one object.

## Architecture (fixed across the sweep)

| field | value | why this value |
|---|---|---|
| `d_model` | **256** | **matches the GRU's hidden size** so state-geometry numbers are directly comparable — row-space chance level is `√(d/H)`, so the width must match or every geometry comparison is confounded |
| `n_layers` | 4 | enough depth for the receptive field to exceed the window (the point of §5); shallow enough to train in minutes |
| `n_heads` | 4 | 64-dim heads |
| `mlp_ratio` | 4.0 | standard |
| embedding | **linear** `Linear(obs_res, d_model)` + ReLU | observations are continuous 128-d intensity scans, not tokens — no vocabulary, no softmax head. The ReLU makes residual point 0 **exactly** the GRU's encoder port `relu(Linear(obs))`, so the encoder-space-editing result transfers unchanged |
| decoder | `Linear(d_model, obs_res)` on the final pre-LayerNorm stream, **untied** from the encoder | mirrors the GRU (which also has untied encode/decode); tying would impose a constraint the GRU never had and confound the comparison |
| positional | RoPE | relative, so a shifted buffer at inference behaves like the same offsets seen in training |
| mask | band-causal, width `window` | each position attends to itself and the `window − 1` before it |

## `window` vs `state_span` — these are NOT the same number

`window` is the **per-layer** attention span. Stacking layers widens the receptive field, so the **carried
state** — the frames you must supply to reproduce the model's own next prediction — spans

    state_span = n_layers × (window − 1) + 1

Pinned by `tests/test_transformer.py::test_buffer_rollout_matches_full_sequence`: the one-pass banded forward
and the step-by-step buffer rollout agree to float tolerance only at `state_span`, and diverge from exactly
`t = window` onward otherwise. **Always quote `state_span` when talking about how much history an edit
overwrites.**

## Runs

| code | descriptive label (use this in every figure) | window | carried `state_span` | params | role |
|---|---|---|---|---|---|
| `W2` | **transformer · window 2** | 2 | 5 frames | 3,225,472 | minimum window that can see velocity; history *must* be compressed into the residual stream |
| `W4` | **transformer · window 4** | 4 | 13 frames | 3,225,472 | intermediate |
| `W16` | **transformer · window 16** | 16 | 61 frames | 3,225,472 | effectively full context — span exceeds the 20 frames available before the edit frame, so its *effective* carried state is the whole history |

Parameter count is identical across the sweep: `window` only changes the attention **mask**, not any weight.

**`window` is an explanatory variable, not a robustness check.** It dials continuously between "no compressed
state, just a lookup over raw history" (large W) and "history must be compressed into the residual stream"
(small W). It is therefore the *mechanism* behind any GRU-vs-transformer difference, not a sensitivity
analysis of it.

## Cross-thread reference run (defined elsewhere, copied here for convenience)

| code | descriptive label | source registry | why it is here |
|---|---|---|---|
| `H256` | **GRU · H=256 (reference)** | `../controls/CONTROL_RUNS.md` | the width-matched GRU. 460,672 params, 400 epochs, best val **0.02362**. Every transformer figure carries it as the reference series |

> **On the parameter gap.** The transformers carry ~7× the GRU's parameters at equal state width (3.23M vs
> 0.46M) — depth × MLP blocks, whereas the GRU is one recurrent cell. This is *not* controlled, and no claim
> in this thread should rest on a raw capacity comparison. It is acceptable here because `d_model` is matched
> to `hidden_size`, which is what the geometry and probe metrics are normalized by, and because the
> predictive-quality gate (§1 of the notebook) verifies the two models are at **comparable prediction
> quality** before any editability difference is read. A parameter-matched transformer would need either a
> narrower `d_model` (breaking the geometry comparison) or fewer layers (breaking the `state_span` sweep) —
> both worse trades. Flag it in any write-up rather than matching it.

## Residual points (the layer axis used in every by-depth figure)

`probe_layer` indexes **residual points**, of which there are `n_layers + 1 = 5`:

| point | what it is | label used in figures |
|---|---|---|
| **0** | encoder output `relu(Linear(obs))` at the current frame | `0 · encoder port` |
| 1 | input to block 1 | `1 · early` |
| 2 | input to block 2 | `2 · middle` |
| 3 | input to block 3 | `3 · late` |
| **4** | final pre-LayerNorm stream the decoder reads | `4 · last (decoder input)` |

An edit at point ℓ changes the block inputs for layers **> ℓ only** — so editing the *last* point alters this
position's own prediction and propagates to nothing, while editing point 0 propagates furthest.

## Wall-clock (measured 2026-08-04, local 5090)

All three train sequentially from one launch (`logs/transformers/train.log`):

| run | epochs | s/epoch | wall-clock | best val |
|---|---|---|---|---|
| `W16` | 300 | 2.5 | **12.6 min** | 0.02359 |
| `W4` | 300 | 2.6 | **12.9 min** | 0.02372 |
| `W2` | 300 | 2.5 | **12.7 min** | 0.02396 |
| **total** | 900 | — | **38.2 min** | — |

`window` does not affect cost: the band mask is applied to the full T×T attention, so all three runs are the
same shape.

> **300 epochs is far more than these models need — they overfit.** Validation loss bottoms out around
> **epoch 40** and then rises steadily (`W16`: 0.02359 at best → 0.02590 at epoch 300). `best_model.pt` is
> therefore a genuinely different checkpoint from `latest.pt`, unlike the GRU whose curve is flat after
> convergence — **always load `best_model.pt`**. Budget **~60 epochs (≈2.5 min)** for future runs in this
> sweep; the long schedule here bought nothing but a cleaner cosine tail. Fig 1a of the notebook marks the
> checkpoint actually used on each curve.
