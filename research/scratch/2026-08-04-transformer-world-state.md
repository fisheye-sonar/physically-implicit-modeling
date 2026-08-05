# Transformers: the readable state is not the carried state, and editing is a property of the history

**Date:** 2026-08-04 · **Branch:** `michael_controls` · **Direction:** `transformer-world-state` (`[reframe]`,
sub-Q 1/2/3) · **Status:** → **FLAG FOR PROMOTION** (it retires a premise that every §4 result silently
assumed) · **Author:** orchestrator, at Sevan's request ("implement the transformer end to end").

## The question, and why it is a reframe

Every §4 result so far — GRU, RSSM, DiT — rests on a premise so uniform it was never stated: **the model has
exactly one state**, a vector `h` that is both what it carries between steps and what a probe reads. "Edit the
world state" is only well-posed because those are the same object.

A causal transformer breaks that. It has **two** state objects and they come apart:

| | what it is | carried? | history-dependent? |
|---|---|---|---|
| **carried state** — the observation buffer | the frames you must supply to reproduce the model's own next prediction | **yes** | no — each slot is one frame |
| **readable state** — residual stream at (layer ℓ, current position) | what attention has mixed at this position | **no**, recomputed every step | **yes** |

So "does a latent write stick?" splits into two questions with different expected answers, and a decayed
activation write here is **architecture, not the GRU's reversion failure**.

## Setup / provenance

Notebook `notebooks/experiments/editability/transformers/transformer_world_state.ipynb` (25 cells, 0 errors,
10 figures), PNGs `/tmp/transformer_wm/`. Registry `transformers/TRANSFORMER_RUNS.md`. New code:
`pim/world_models/transformer/model.py`, `scripts/train_transformer.py`, `tests/test_transformer.py` (6 tests).

Runs `W2` / `W4` / `W16` (window 2 / 4 / 16), all `d_model=256` (**matched to the GRU's hidden size** so
geometry chance levels are comparable), 4 layers, 3.23M params each — `window` changes only the attention
mask. 300 epochs, AdamW lr 1e-3, 5% warmup + cosine, grad-clip 1.0, on `datasets/4_fixed_refl_inview`
(the same data as every other architecture). Reference: GRU `runs/controls/H256`. `ef=20`, `K=15`, **N=192**
held-out edits. Metrics = the canonical §4 set (`scripts/editability_metrics.py`).

**Training cost: 12.6 + 12.9 + 12.7 = 38.2 min total** on the local 5090 (~2.5 s/epoch, 300 epochs each).

**Alignment verified** on the test split for all four models: RMSE is minimised at `k=0`, so rollout step 0 ↔
frame `ef` holds (`W2` 0.1053/0.1050/0.1217 for k=−1/0/+1, and similar for the rest).

## The structural fact that had to be got right first

Stacking layers widens the receptive field, so the carried state spans

    state_span = n_layers × (window − 1) + 1

frames — **not** `window`. Verified numerically: a one-pass banded forward and a step-by-step buffer rollout
agree to float tolerance only at `state_span`, and diverge from exactly `t = window` onward otherwise
(`tests/test_transformer.py::test_buffer_rollout_matches_full_sequence`). Sizing the buffer by `window` would
have understated the history an edit must overwrite by a factor of `n_layers` — the entire quantity §5
measures. Windows 2/4/16 give spans 5/13/61.

## Results

**0. The quality gate passes, so everything below is interpretable.** Next-step RMSE vs clean: `W16` **0.1039**,
`W4` 0.1049, `W2` 0.1061, against the GRU's **0.1041** and a noise floor of 0.1539. `W16` also beats the GRU on
val loss (0.02359 vs 0.02362). This is a like-for-like comparison, not a weak model being hard to edit.

**1. Readability peaks in the middle of the stack, not at the end.** Position R² (linear) by residual point:
0.60 (encoder port) → 0.79 → **0.81** (middle) → 0.77 → 0.76 (last), against the GRU's 0.83. Identical shape
for all three windows. Velocity R² separates by window at the middle point — 0.13 (`W2`) / 0.20 (`W4`) /
**0.32** (`W16`) — which is the expected mechanism: a longer window carries more velocity evidence. The fiber
residual (MLP) traces the same arc, 0.38 → 0.65–0.70 (middle) → 0.17 (last), i.e. the middle of the stack
holds the most beyond `(position, velocity)` and the last residual point holds the least.

**2. Editing the readable state reproduces the GRU's negative — and adds nothing.** Readout injection is
**inert at every depth and every window** (Edit Index −0.65…−0.68, exactly each model's own unsteered value;
fidelity ratio 1.00). The decoder-gradient oracle lands at every depth (+0.79…+0.88, best at residual point 3)
but the GRU still beats it (+0.93). So "readable ≠ grabbable" is **not** an artifact of recurrence — it
survives to attention, at every layer.

**3. The transient/persistent split is real and measurable.** Edit Index at step 0 → step 14:

| intervention | acts on | step 0 | step 14 |
|---|---|---|---|
| activation edit (last residual point) | readable state | **+0.86** | **+0.04** |
| history overwrite, n = saturation | carried state | +0.63…+0.67 | **+0.27…+0.28** |
| unsteered | — | −0.68 | −0.43 |

The activation write is the **strongest step-0 edit in the whole notebook** and is gone within ~2 steps. This
is the cleanest demonstration to date that a high step-0 Edit Index is not evidence of a durable edit — and
here the decay is provably architectural, since the next step recomputes the stream from the buffer.

**4. THE HEADLINE — a registered prediction, and both of us were wrong.** Before running, two predictions were
recorded: **Sevan** — edits stick at ≲50% of the window, a fixed *fraction*; **Claude** — ~2–4 frames
regardless of window, a fixed *count*. Measured saturation points (smallest n reaching 90% of that model's own
maximum Edit Index):

| model | effective span | saturation n | % of effective span | % of architectural span |
|---|---|---|---|---|
| window 2 | 5 | 3 | 60% | 60% |
| window 4 | 13 | 4 | 31% | 31% |
| window 16 | 20 | 6 | 30% | 10% |

Those are the two endpoints of one scaling law, `n_sat ∝ span^β`: β = 0 is Claude's prediction, β = 1 is
Sevan's. **Measured β = 0.47** — the requirement grows roughly like the **square root** of the available
history. The absolute count does grow with span (against Claude), but far more slowly than proportionally
(against Sevan). *Caveat: a 3-point fit; treat β as an order-of-magnitude statement.*

The **crossover** point (Edit Index > 0) is n = 1 for every model — a single overwritten frame flips which
world the output is closer to. It is a useless discriminator and should not be quoted as "how much history an
edit needs".

**Effective vs architectural span matters.** At `ef = 20` a model has only ever seen 20 frames, so `W16`
(architectural span 61) has an *effective* carried state of 20. Percentages use the effective span; the
architectural column is reported alongside so the difference is visible rather than hidden.

## What this means

The candidate finding is about the premise, not about which architecture wins:

> **On an architecture whose readable state is not carried, editability is not a property of the latent at
> all — it is a property of the observation history.** The single-`h` framing that makes §4 well-posed for
> GRU/RSSM is an architectural coincidence, not a general fact about world models.

It also strengthens the thread's through-line rather than complicating it. Sevan's framing — *no successful
edit is free of dynamics* — now has its sharpest instance: on a transformer the **only** channel that
persists **is** the observation history, and the one intervention that works is literally rewriting what the
model saw. The `√span` scaling says the model does not need its whole history rewritten, but it needs more
than a constant patch: consistency has to extend back far enough that attention cannot find contradicting
evidence.

## Caveats

- **Parameter count is not controlled** (3.23M transformer vs 0.46M GRU). `d_model` is matched to
  `hidden_size`, which is what the geometry/probe metrics normalise by, and the quality gate confirms
  comparable prediction quality — but no claim should rest on raw capacity. Flag it in any write-up.
- **The transformers overfit.** Val loss bottoms at ~epoch 40 and rises to ~0.0259 by epoch 300; best-checkpoint
  selection is doing real work (unlike the GRU, whose curve is flat). 300 epochs is far more than needed —
  ~60 would do. Fig 1a marks the checkpoint actually used.
- β is fit on 3 points, and the two shortest spans nearly tie in absolute frames.
- Only the decoder-gradient and readout-injection editors were run in activation space; the structural editors
  (PCA geodesic, global-PCA projection) were not, since the transient-by-construction result makes them
  uninformative there.

## Follow-ons

- **KV-cache view as a third state object** — carried *and* history-dependent, the closest transformer analogue
  of a GRU `h` write. `state_view="kv_cache"` already exposes it; this is the sharpest remaining test.
- Window 1 (no attention over history) as the degenerate control.
- More window values to tighten β — the current fit is the weakest number in the note.
- Whether a multi-step training objective moves any of this, matching the RSSM arm.

## Harness changes made in the same session

- `scripts/editability_metrics.py`: `SCORECARD_COLUMNS` listed `gt_traj_rmse` **twice**, so every scorecard
  table in the repo rendered a duplicate column. Fixed.
- `pim/world_models/loader.py`: the transformer branch never called `.to(device)`. Fixed.
- `METRICS_AND_EDITORS.md`: the "Oracle observation" rename had overwritten its own former name
  ("true-state swap"), making the provenance note circular. Fixed. Added the two transformer-specific editors
  and the **saturation point** metric.

## Follow-up (2026-08-04, Sevan's review questions)

**Q: does pseudoinverse readout injection actually work on the transformer? — No.** Fig 4 combined both
editors and the injection line lay exactly on top of the unsteered line, which made the null look like a
result. Split into Fig 4a/4b and instrumented (Table 3). The write **is** applied and lands exactly:
probe error 3.2 → ~1e-6 sim units, `‖Δh‖/‖h‖` up to 0.15. What reaches the render is ~nothing:
`‖Δrender‖/‖render‖` 0.007–0.036, and Edit Index moves −0.684 → −0.681 (last point) / −0.669 (middle),
against a −1…+1 scale, with fidelity ratio 1.00. The GRU behaves identically (−0.670 → −0.655). **This is a
null result with a working editor, not a broken editor** — and Fig 4a panel (c) is now the evidence.

**New: retention keeps improving long after the step-0 index saturates (Fig 7, Table 4).** §5's saturation
point is measured at the edit frame only. Sweeping `n` and reading the Edit Index across the whole rollout
shows a second, slower quantity: for window 16 the step-0 index saturates at n≈6 (+0.63) but *retention*
(step-14 ÷ step-0) keeps climbing — 0.37 at n=2, 0.44 at n=6, 0.49 at n=8, **0.62 at n=16–20**. So "how much
history must be rewritten" has two answers: **~30% of the span to land the edit, and essentially the whole
span to make it hold.** The step-0 saturation point understates the requirement; report both.

**Q: how is the counterfactual world built, and is it oracle? — Partly, and the notebook now says so.**
The edited object's counterfactual history is a **straight line arriving at the target at frame `ef`,
travelling backwards at the true post-edit velocity** `v = VEL_E[i, ef, o]`; the other object is re-rendered
on its **true** trajectory; all frames are rendered with the training observation noise. So the object is
**not** held stationary — which is exactly why the rollouts continue to move, as Sevan noticed. Oracle
content: (a) the target position — that is the edit specification, legitimate; (b) the object's velocity —
**oracle in provenance but not in information**, because the teleport preserves velocity exactly (verified:
`max|v[t+1]−v[t]| = 0` on the edits split) so this equals the pre-edit velocity already present in the
model's own observation history; (c) the other object's true positions — identical to what the model already
saw. A stricter version should estimate the velocity from the pre-edit observations instead of reading it
from the sim; that is a **follow-on, and until it is run the sweep should be described as velocity-oracle.**

**Q: is the late "drift" in sample 6 evidence of leakage? — No, on three independent grounds.**
1. **Seeds are disjoint.** train 0–89999, val 90000–99999, test 100000–109999, edits 110000–119999. Edits
   sample 6 is scene seed 110006. Nearest train scene by initial `(pos, vel)` over 20000 train scenes is
   L2 = 0.55 against a median inter-scene distance of 5.31 — nearest-neighbour, not a duplicate.
2. **The drift is not noise; it is ballistic motion.** `direction_noise_std = speed_noise_std = 0`, so
   velocity is *exactly* constant. The only stochasticity is `position_noise_std = 0.04` applied to position
   as a random walk. Over 15 steps the ballistic displacement is **1.09 units** and the accumulated noise
   **0.18 units** — **84% of the post-edit motion is fixed by (position, velocity)**.
3. **We hand the model the velocity.** The counterfactual history *is* a constant-velocity track, so the
   model reads the velocity off the frames it is given and extrapolates. Tracking the ground truth is the
   expected behaviour, not a lucky guess. Fig 8 (six unselected samples) and Table 5 make this checkable.

**Also added:** Fig 8 (six *unselected* samples — Fig 6 showed the three largest teleports, a selected view).

**Waterfall line-up corrected.** Both waterfalls previously showed only the **decoder-gradient oracle** in
their single activation-edit column, and that column was labelled `"activation edit (last residual point)"` —
naming the *site* but not the *editor*. That label is what let the oracle's success be read as the
pseudoinverse injection's. Both figures now carry an explicit **pseudoinverse injection** column (at the
middle residual point, where Table 3 shows the injection perturbs the render most — 0.036 vs 0.015 — so it
gets its most favourable showing), every column title names its editor, and every column title carries its
own Edit Index. **Observation-space confirmation of the null:** the pseudoinverse columns are visually
indistinguishable from unsteered in all nine samples — the object stays on the red ghost locator and never
reaches the green target. **Rule going forward: an editor column in a waterfall must be named by its editor,
never by its edit site.**

**NEW — §6 explains *why* the injection is inert, and turns the null into a geometric measurement.**
Readout injection can only move `h` inside the probe's row space `row(A)`, by construction. The decoder has
its own opinion about which direction would change the render: the descent direction
`g = −∇_h ‖decode(h) − gt_obs[ef]‖²` at the unedited activation. Per sample, then averaged, N = 192:

| residual point | cos(g, Δh_pinv) | angle | shuffled control | row-space fraction of `g` | ÷ chance (0.125) |
|---|---|---|---|---|---|
| 0 · encoder port | +0.007 ± 0.080 | 89.6° | −0.001 ± 0.078 | 0.142 ± 0.069 | 1.13× |
| 2 · middle | +0.050 ± 0.150 | 87.1° | −0.014 ± 0.169 | 0.289 ± 0.142 | **2.31×** |
| 4 · last (decoder input) | +0.014 ± 0.042 | 89.2° | +0.001 ± 0.040 | 0.071 ± 0.035 | **0.57×** |
| GRU (single state) | +0.034 ± 0.100 | 88.1° | −0.017 ± 0.098 | 0.189 ± 0.074 | 1.51× |

(window 16 shown; windows 2 and 4 are the same to ±0.01. The converged 200-step oracle displacement gives
the same picture: +0.005…+0.069.)

**Every cosine sits inside the shuffled-pair band — the two directions are orthogonal to measurement
precision (87–90°).** And the row-space fraction has a striking depth profile: it is **2.3× chance at the
middle residual point but 0.57× chance — *below* chance — at the last residual point, the one the decoder
actually reads.** So at the very layer where a write would most directly change the render, the probe's row
space is a subspace the decoder is *less* sensitive to than a random direction would be.

This is the transformer's version of the GRU reachability ceiling from `2026-08-03-delta-h-analysis` (where
the row-space fraction of Δh_true was 0.096 against a 0.125 chance level), and it converts "readable ≠
grabbable" from an observation into a **hard ceiling**: `‖P_row g‖/‖g‖` is the best any injection-style
editor could ever achieve, independent of how the target is chosen or how the probe is tuned. It also
explains the one non-zero signal in Table 3 — the middle point is where injection perturbed the render most
(0.036 vs 0.015), and it is exactly where the row-space enrichment peaks.

**Amended caveat.** The scaling exponent β = 0.47 is measured on the *landing* criterion. On the *retention*
criterion the requirement is closer to the full span for every window, so β is criterion-dependent and the
"√span" statement applies to landing only.
