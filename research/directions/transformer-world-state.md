# Direction: Transformer world models — what *is* the state, and can you edit it?

**Tag:** `[reframe]` · **Sub-questions:** 1 (what is represented), 2 (canonicality), 3 (editability) ·
**Status:** in progress (2026-08-04) · **Complexity:** medium-high (new architecture + training sweep) ·
**Models:** causal transformer, window ∈ {2, 4, 16}, against the GRU H=256 reference. Branch `michael_controls`.

Notebook: `notebooks/experiments/editability/transformers/transformer_world_state.ipynb`.
Registry: `notebooks/experiments/editability/transformers/TRANSFORMER_RUNS.md`.
Code: `pim/world_models/transformer/model.py`, `scripts/train_transformer.py`, `tests/test_transformer.py`.
Origin: Sevan, 2026-08-04 — "implement the transformer end to end", paper-1 architecture coverage.

## The gap this closes — and why it is a `[reframe]`

Every §4 result so far rests on a premise so uniform across GRU / RSSM / DiT that it has never been stated:
**the model has exactly one state**, a vector `h` that is simultaneously (a) what the model carries from step
to step and (b) what a probe reads the world out of. "Edit the world state" is well-posed only because those
are the same object — writing to the thing you read *is* the intervention.

A causal transformer breaks that premise. It has **two** state objects and they come apart:

| | what it is | carried across steps? | history-dependent? |
|---|---|---|---|
| **carried state** — the observation buffer | the recent frames you must supply to reproduce the model's own next prediction | **yes** | no — each slot is one frame |
| **readable state** — residual stream at (layer ℓ, current position) | what attention has mixed at this position | **no** — recomputed from the buffer every step | **yes** |

So the negative result "probe-directed writes to `h` do not move the object" has to be *re-asked*, because
there are now two different writes with different expected fates:

- A write to the **readable** state is transient **by construction** — the next step recomputes the residual
  stream from the buffer, so the edit is erased whether or not the dynamics "accept" it. If it decays, that is
  **not** the GRU's reversion failure; it is architecture. Reporting it as the same phenomenon would be wrong.
- A write to the **carried** state is the only channel that can persist, and it is *not* a latent write at all
  — it is a rewrite of observed history.

That is the reframe: **on this architecture, "editing the world state" and "editing the observation history"
are the same operation**, and the readable state is not a control surface at all. If that holds, the
editability question is architecture-dependent in a way none of the §4 results anticipated.

## The load-bearing structural fact (already established, do not re-derive)

Stacking layers widens the receptive field. At layer L, position `t` attends to keys that were themselves
computed from a window, so the carried state spans

    state_span = n_layers × (window − 1) + 1

frames — **not** `window`. `tests/test_transformer.py::test_buffer_rollout_matches_full_sequence` pins this:
a one-pass banded forward and a step-by-step buffer rollout agree to float tolerance only when the buffer
holds `state_span` frames, and diverge from exactly `t = window` onward otherwise. Sizing the buffer by
`window` would mis-state how much history an edit must overwrite by a factor of `n_layers` — which is the
entire quantity §5 measures. Any reimplementation must keep this test.

## What to run

Standard §4 spine (same metrics, same estimators, same figures as every other architecture — do not invent
new ones; import `scripts/editability_metrics.py`):

1. **Predictive quality gate.** Match the GRU's val loss and multi-step rollout RMSE before reading anything
   into an editability difference. A worse predictor that is harder to edit tells you nothing.
2. **Readability by depth.** Position / velocity R² (linear and MLP, held out) at every residual point
   0 … n_layers, where point 0 is the encoder port `relu(Linear(obs))` — the transformer's exact analogue of
   the GRU's `x`, so the encoder-space-editing result transfers.
3. **Geometry.** Global-PCA hull residual, TwoNN / MLE intrinsic dimension, fiber residual (canonicality)
   at the same residual points. `d_model = 256` is chosen to match the GRU's hidden size so the row-space
   chance level `√(d/H)` is directly comparable.
4. **Activation edit by residual point.** Decoder-gradient edit applied at each residual point; report the
   full canonical scorecard. **Expected to move the edit-frame prediction and then vanish.** The measurement
   that matters is not *whether* it decays but *how fast relative to the GRU's* — and the honest reading is
   that a one-step effect here is the *ceiling*, not a failure.
5. **History overwrite sweep** — the headline. Replace the newest `n` frames of the carried buffer with
   renders of the counterfactual world and sweep `n` from 0 to the span. Report the Edit Index as a function
   of `n`, in **both** absolute frames and as a fraction of `state_span`, for all three windows. Two live
   predictions are registered below; the sweep adjudicates them.
6. **Canonical waterfalls** (per the `CLAUDE.md` spec, via a single `waterfall_grid` helper).

**Sweep hygiene:** the sweep must top out at the history that actually exists. At edit frame `ef = 20` a model
with `state_span = 61` has still only seen 20 frames, so its *effective* carried state is 20 and the buffer's
`length` mask must keep the padding invalid. Build the overwritten state through `state_from_obs` so padding
and `length` stay correct — never hand-assemble a full-span buffer, or the sweep silently gives the model
context it never had.

## Registered predictions (recorded 2026-08-04, before the sweep ran)

- **Sevan:** edits stick at **≈50% of the window overwritten, or less** — 100% is not needed.
- **Claude:** the threshold is **≈2–4 absolute frames regardless of window**, because what the model needs is
  a locally consistent position+velocity at the current position, not a filled buffer. If so the *fraction*
  falls as window grows while the *absolute* count stays flat.

These make opposite predictions about which panel of the §5 figure is flat, so plot both. Do not quietly
resolve them in prose — report the crossing point per window and say which is supported.

**RESOLVED 2026-08-04 — neither.** They are the two endpoints of one scaling law `n_sat ∝ span^β` (β = 0 is
Claude's, β = 1 is Sevan's). Measured **β = 0.47**: saturation grows like the **square root** of available
history — 3/5 frames (60%) at window 2, 4/13 (31%) at window 4, 6/20 (30%) at window 16. Fit on 3 points, so
order-of-magnitude only. See `scratch/2026-08-04-transformer-world-state.md`. Future window sweeps should fit β
rather than re-run the flat-vs-flat comparison.

## Bootstrap (cold start)

```python
sys.path.insert(0, "<repo>"); sys.path.insert(0, "<repo>/scripts")
from pim.world_models import load_checkpoint, load_dataset
from editability_metrics import build_edit_zones, edit_scorecard, fidelity_ratio
MODELS = {r: load_checkpoint(f"runs/transformers/{r}/best_model.pt")[0] for r in ("W2","W4","W16")}
MODELS["H256"] = load_checkpoint("runs/controls/H256/best_model.pt")[0]   # GRU reference
```

Data: `datasets/4_fixed_refl_inview` (obs noise 0.2, position noise 0.04), `edits` split, edit frame 20,
2 objects, `obs_res=128`. Same data as every other architecture — this is a controlled comparison.

`model.state_view` toggles what `flat_state` returns (`"obs_window"` / `"activations"` / `"kv_cache"`);
`model.probe_layer` selects the residual point. `state_from_flat` deliberately **raises** for the
`activations` view — a residual-stream vector does not determine a future, and silently pretending it does
is the bug this guard exists to prevent.

## What would make this a finding

Not "the transformer is less editable." The candidate finding is sharper and is about the *premise*:

> **On an architecture whose readable state is not carried, editability is not a property of the latent at
> all — it is a property of the observation history.** The single-`h` framing that makes §4 well-posed for
> GRU/RSSM is an architectural coincidence, not a general fact about world models.

The supporting measurement is the pair (activation edit decays within one step at every depth, *and* history
overwrite works with a threshold that is `n`-frames-absolute rather than window-relative). Either half alone
is weak.

## Follow-ons (do not do in the first pass)

- KV-cache view as a third state object — an edit there is carried *and* history-dependent, the closest
  transformer analogue of a GRU `h` write. `state_view="kv_cache"` already exposes it.
- Window 1 (no attention over history at all) as the degenerate control.
- Whether a transformer trained on the multi-step objective moves any of this, matching the RSSM arm.
