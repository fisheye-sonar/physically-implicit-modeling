# 2026-08-21 — Our transformer, our recipe, their world

Thread `notebooks/experiments/editability/ours_on_othello/`; brief
`directions/our-architecture-on-othello.md`; registry `OURS_ON_OTHELLO_RUNS.md`. Overnight run,
started 16:30. **This note is written while the ladder is still training — the intervention
results below cover arm `M` only.** Everything is `observed` until the full-ladder notebook lands.

Origin — Sevan: *"take our transformer which we trained on discworld, adapt it to match the input
and output scheme of the OthelloGPT setup, train it similar to how we trained on discworld, and see
if it is still editable… applying our model on their setting instead of the reverse as a first test,
since it's cheaper."*

## The setup, and what is *not* new about it

Three substitutions from `runs/transformers/W16`, nothing else: `Linear(128,256)+ReLU` →
`nn.Embedding(61,256)`; `Linear(256,128)` → `Linear(256,61)`; MSE → cross-entropy. `d_model` 256,
4 layers, 4 heads, RoPE, banded attention, AdamW 1e-3, batch 256, 5% warmup + cosine, grad-clip 1.0
— all `W16`'s. Fresh init (Sevan's call).

**There is no shim.** `othello_transfer/othello_shim.py` exists because minGPT had to be taught the
seven names our editing code calls; our model has all seven natively, including `_run`'s
`fn(layer, x) -> x` hook. The entire bridge is `evaluate.attach()`: set `N_POINTS` to 5, and point
the probe cache at a per-model directory.

**The band width is not a limitation.** `state_span = n_layers·(window−1)+1` = 61 at `window` 16, so
a 60-move game is inside the receptive field. `window` 40 (span 157) runs alongside at zero extra
compute — measured 25,018 vs 25,001 games/s, because the band mask is applied to the full T×T
attention.

## Scale is a controlled axis, not a free variable

Sevan's objection, which reshaped the whole run: *"I don't really like how we are comparing very
different data sizes… I don't really know if success came from scale."* Training at Li's 20M games
is **222× the unique sequences** and **333× the unique tokens** `W16` ever saw.

So every rung runs the **same 95,100 optimiser steps** at batch 256 with the identical schedule, and
only the pool changes. Corpora are nested prefixes of one index-seeded stream, so `M ⊂ L1 ⊂ L2 ⊂ D`.
`F` (8 passes over 20M) is the one arm with more compute and is never quoted as a scale datapoint.

## Result 1 — the scaling curve saturates, and 90k is a memorisation regime

Held out on games from a disjoint index range, at `best_model.pt`. Bayes floor **2.0092** nats
(their generator draws uniformly from the legal set, so `E[log|legal|]` is the true ceiling —
`|legal|` averages 8.61, median 9).

| rung | unique games | excess CE w16 | excess CE w40 | legal mass w16 | top-1 legal | best val at |
|---|---|---|---|---|---|---|
| `M` | 90,000 | +0.580 | +0.503 | 0.734 | 0.878 | **6%** |
| `L1` | 1,000,000 | +0.210 | +0.195 | 0.889 | 0.973 | 100% |
| `L2` | 5,000,000 | +0.171 | +0.159 | 0.895 | 0.986 | 100% |
| `D` | 20,000,000 | +0.167 | +0.156 | 0.896 (w40 **0.905**) | 0.988 | 100% |

Three readings:

1. **The overfitting at `M` is purely a data artifact.** Best val is 6% into the schedule at 90k and
   100% from 1M onward. It never turns over again.
2. **Almost everything arrives by 1M, then flattens.** Excess CE falls 0.580 → 0.210 over the first
   11×, then 0.210 → 0.167 over the next 20×.
3. ⚠ **The pre-registered ≥0.95 legal-mass gate is never met.** Best is 0.905. Data alone does not
   close the gap to Li's 0.9998; the residual is capacity/depth (3.19M vs 25.28M params, 4 vs 8
   layers). That is a result about **our architecture's ceiling on this world**, and it is
   independent of editability.

## Result 2 — memorisation and rule-learning are different targets

`M_w16` at end of training (step 95,100), which is *not* the checkpoint anything else uses:

| split | legal mass | excess CE |
|---|---|---|
| TRAIN (seen) | 0.863 | **−0.824** |
| val | 0.745 | +4.242 |
| TEST (disjoint) | 0.743 | +4.205 |

Training CE goes **below the Bayes floor** — only achievable by recalling which move each specific
game happened to draw. Yet training legal mass reaches only 0.863. Recalling one move per position
is cheap; distributing mass correctly over all ~8.6 legal moves is not, and memorisation does not
buy it. Note also that held-out **legal mass rises slightly** (0.734 → 0.743) while held-out CE
degrades 2.4× — the rule knowledge survives; the calibration collapses.

At `best_model.pt` the train/test gap is only 0.8pp (0.7425 vs 0.7341), so `M`'s weakness is *not*
a generalisation failure. It simply has not learned the rules at that volume.

## Result 3 — arm `M` interventions: uninformative, by the pre-registered rule

Null Edit Index for `M_w16` is **−0.078** (Li's model: −0.829, *because that model predicts the
unedited world well*; a weak predictor starts near 0). ⚠ **Report the null and the absolute
post-edit value together** — see Result 4 for why the gain alone is misleading.

| editor | Edit Index | symdiff |
|---|---|---|
| Li gradient steering, `L_s` 0…4 | −0.044 … −0.051 | −0.040 … −0.007 |
| Nanda, add target direction | −0.045 | −0.006 |
| **Nanda, target − current** | **+0.009** | **+0.294** |
| our pseudoinverse | −0.042 | +0.002 |

Best gain over its own null: **+0.087** (`M_w16`), **+0.090** (`M_w40`), **+0.000** (random init) —
against **+1.526** for the same code on Li's model and **+0.490** on discworld `W16`. Single-point
writes do not help either (Table 4).

⚠ **Not reported as an editability result.** `M` puts a quarter of its mass on illegal moves; a
model without a board state has nothing to edit. This is the "gates not passed → uninformative"
branch written into the brief before the run.

The one real signal: Nanda's `target − current` moves the **symdiff** index −0.172 → **+0.294**, a
directional move on exactly the squares whose legality changed, at the cost of legal mass
(0.664 → 0.601). I under-reported this in my first summary and Sevan caught it.

## Probes — the board *is* decodable, well above the architectural floor

Best held-out error, mine/theirs, held out by sequence (majority floor 53.12%):

| | linear | MLP 512 |
|---|---|---|
| `M_w16` | 17.74% | 16.12% |
| `M_w40` | 15.58% | **13.51%** |
| *random init (seeded control)* | *27.40%* | *25.87%* |

Training buys 10–12pp over an identical untrained network, and decodability rises monotonically with
depth in trained models while staying flat in the random one. Li's published nonlinear probe is
1.7%. The `MLP ≥ linear` tripwire fired **0** violations across 180 cells.

## Traps hit, all now guarded

1. ⛔ **The probe cache was keyed on settings and data but not on the model.** `M_w16` and the
   random-init control returned *identical* errors (37.08%–57.94%) — the control was being served
   the trained model's probes, which would have destroyed the only baseline that makes an absolute
   probe error interpretable. `evaluate.attach` now keys the cache on a **hash of the weights**.
2. **The random-init control was unseeded**, so its weights — and its cache fingerprint — changed
   every execution. A control that moves between runs is not a control. Seeded and verified.
3. `evaluate.probe_data` called `corpus.build` without `only=`, which silently started a 3-hour
   20M-game generation from what looks like a cheap lookup.
4. Generation would have OOM'd: 20M games as `list[list[int]]` is ~33 GB. Streamed in chunks
   straight into an int8 array (1.2 GB).
5. **Two throughput estimates were wrong by ~8×**, both because I read a multi-core rate as
   per-core — Othello generation (1,487/s was already 32-core) and discworld generation ("645/s/core"
   → really ~180/s/core). Measure it; don't scale it.
6. A queue script without `PYTHONPATH` would have died at midnight; and `pkill -f <script>` matched
   its own shell three separate times, twice silently preventing a fix from landing.

## Result 4 — ⚠ data does NOT make our architecture editable; the "gain" metric said otherwise

`othello_transfer/`-style linear mine/theirs probes on every rung, all three direction editors,
all-points and single-point (`scratch/ladder_edit_preview.json`).

| run | unique games | null Edit Index | **best Edit Index (absolute)** | "gain over own null" |
|---|---|---|---|---|
| `M_w16` | 90,000 | −0.078 | **+0.059** | +0.137 |
| `L1_w16` | 1,000,000 | −0.340 | **+0.095** | +0.435 |
| `L2_w16` | 5,000,000 | −0.416 | **+0.095** | +0.511 |
| `D_w16` | 20,000,000 | −0.425 | **+0.098** | +0.523 |
| *their model, same code* | — | *−0.829* | ***+0.697*** | *+1.526* |

⛔ **The gain column rises 3.8× across the ladder and means almost nothing.** The absolute
post-edit Edit Index is flat — **+0.059 → +0.098** — and the entire apparent improvement is the
*null* falling from −0.078 to −0.425, i.e. the model becoming a better predictor of the **unedited**
world. Gain-over-own-null conflates "became a better predictor" with "became more editable", and
only the first is happening here.

**Sevan caught this**, on being shown only the gain column: *"I assume L1 unedited starts lower
which could explain the bigger delta. +0.435 isn't very large."* Correct on both counts. I had led
with gain for several messages; the absolute post-edit value is the quantity, and it should be
reported beside the null every time.

*(The gain framing was introduced for a good reason — on 2026-08-21 the null on Li's model is −0.829
because that model predicts the unedited world well, so comparing raw levels across models is also
wrong. The fix is to report **both**, never one.)*

**Consequence for `directions/discworld-at-scale.md`:** its trigger was "gain rises materially with
data". On the gain column it fires; on the absolute column it does not. The honest reading is
**it does not fire** — 222× the data moves absolute editability by +0.04, and our architecture
plateaus at +0.098 against their +0.697 in the same world with the same code.

## Result 5 — ⛔ every "best edit" makes the model WORSE on both Li metrics, at every rung

The Edit Index alone hid this. Full scorecards (`runs/ours_on_othello/ladder_edit_full.json`),
best arm selected by **minimum Li error**:

| run | unedited Li post | unedited Li pre | unedited EI | best-edit Li post | best-edit Li pre | best-edit EI | legal |
|---|---|---|---|---|---|---|---|
| `M_w16` | 5.656 | 4.823 | −0.078 | **6.807** | 7.686 | +0.009 | 0.600 |
| `L1_w16` | 3.259 | 1.237 | −0.340 | **5.403** | 6.825 | +0.076 | 0.690 |
| `L2_w16` | 3.043 | 0.883 | −0.416 | **5.646** | 7.187 | +0.081 | 0.670 |
| `D_w16` | 3.025 | 0.855 | −0.425 | **5.367** | 6.673 | +0.072 | 0.684 |
| *their model, same code* | *2.723* | *0.002* | *−0.829* | ***0.052*** | *2.9* | *+0.697* | *0.993* |

**On their model the edit divides Li error by 52× (2.723 → 0.052). On ours it multiplies it by
1.8× (3.025 → 5.367).** The arm that minimises Li error is still *worse than doing nothing*, at
every rung and both windows. Li error against the **pre**-flip world rises too (0.855 → 6.673), so
the output is not moving toward the target board — it is moving away from **both** boards. Legal
mass falls (0.90 → 0.68).

⚠ **The positive Edit Index (+0.072) is an artifact.** The index rewards moving away from the
unedited world, and degradation does exactly that. This is precisely the failure the guard column
(`Li error vs pre-flip`) was added to expose, and it is the second time in two days it has earned
its place — see 2026-08-21 on Nanda's method on discworld (target RMSE worsened while the index
"improved").

**Prediction quality, by contrast, improves a great deal with data**: unedited Li post
5.656 → 3.025 (their model 2.723) and unedited Li pre 4.823 → 0.855. So the ladder produces
genuinely better world models that are *no more editable* — and whose probe-derived writes are
strictly harmful.

**This settles the `discworld-at-scale` trigger: it does not fire.** Editability does not emerge
with data in a world where their architecture achieves it with the same code. The remaining
difference is architectural, which is what the run-A pilot tests.

## Result 6 — ⭐ the environment is what flips editability, and it replicates at 3.5x the training

`othello_arch/` (`model.py`, `model_othello.py`, `train.py`, `editability.py`, `envctrl_eval.py`,
`discworld_eval.py`). Sevan, 2026-08-22: *"we don't yet have any example where … the only thing
changed being the environment, has editability on Othello but not discworld."*

Both rows use **their** architecture (25,312,768 params — Li's exact count — 8 blocks, `d_model`
512, full causal, learned absolute positions, dropout 0.1), ~900k training sequences, batch 256,
lr 1e-3, 5% warmup + cosine, grad-clip 1.0. On discworld the brief's two substitutions apply
(`Embedding`→`Linear(128,512)`, logit head→`Linear(512,128)`, CE→MSE); on Othello the architecture
is used exactly as published. **Every editor is run on every probe target** (`mine`/`state` for
Othello, `pos`/`full` for discworld) with a matched 8–9 value alpha sweep.

| epochs | environment | val | unedited EI | **best EI** | editor · target | Li ↓ | legal mass |
|---|---|---|---|---|---|---|---|
| 4 | **Othello** | 2.15199 | −0.482 | **+0.241** ✓ | Nanda t−c · `mine` | 2.915 → **0.969** | 0.824 → 0.923 |
| 14 | **Othello** | 2.08413 | −0.591 | **+0.231** ✓ | Nanda t−c · `mine` | 2.763 → **0.432** | 0.849 → **0.973** |
| 4 | **discworld** | 0.02080 | −0.699 | −0.113 ⚠ | Nanda · `pos` | — | fidelity 1.042 |
| 14 | **discworld** | 0.02035 | −0.689 | −0.182 ⚠ | grad steering · `full` | — | fidelity 1.005 |

✓ guards pass · ⚠ degradation (fidelity ≥ 1.005, collateral RMSE 2–5x the unedited 0.133).

**1. The split is stable across a 3.5x change in training.** Othello +0.241 → +0.231; discworld
−0.113 → −0.182. Both arms improved as models (Othello excess CE over the Bayes floor +0.143 →
+0.075; discworld val 0.02080 → 0.02035) and **both were still descending at the final epoch**, so
neither is saturated and the comparison is not tilted by one arm training better.

**2. The Othello edit sharpens even though the index does not.** Li error 0.969 → **0.432** and
legal mass 0.923 → **0.973** between 4 and 14 epochs, while the Edit Index moves −0.010. The index
is saturating while the underlying edit improves — **reading the index alone would have said
"no change"**, which is a reason to always report Li error and legal mass beside it.

**3. ⚠ Exactly ONE editor works, on ONE basis — narrower than I first reported.** With the target
confound removed:

| editor | `mine` | `state` |
|---|---|---|
| Nanda, target − current | **+0.231** | −0.007 (Li 15.8, legal 0.091) |
| Nanda, addition | +0.094 | −0.007 (Li 16.0, legal 0.086) |
| PI injection, 1 point | −0.013 | −0.103 |
| MLP grad steering | −0.010 | +0.001 |

At 4 epochs PI injection read +0.138 and I reported it as a second working editor; at 14 epochs it
is −0.013, so **it does not survive** and that claim is withdrawn. Gradient steering fails on both
targets, so its earlier −0.010 was **not** a target artifact — I had suspected the confound
explained it and was wrong.

This narrowing *strengthens* the finding. What transfers is specifically **Nanda's linear
mine/theirs direction** — the representation his paper argues Othello-GPT encodes linearly. The
claim is not "editing works in Othello" but "one editor works, on the one representation that world
is known to encode linearly, and the same editor on the absolute-colour basis is catastrophic."

**4. Why it matters.** This closes the confound ladder: probe implementation (2026-08-20), probe
training data (2026-08-21), our editor (2026-08-21), data volume (2026-08-22, 222x moves absolute
editability +0.059 → +0.098), and now **architecture, data volume and training length jointly**.
What remains is the world.

⚠ **Two things that are still not tidy.** (a) The two environments' Edit Index share a name and a
range but **not a construction** — ray-RMSE over the observation vs a move distribution over 64
squares against uniform-over-legal; read each row against **its own** unedited column. (b) The
Othello arms got ~11% more training sequences and steps than the discworld arms (900k/14,064 vs
810k/12,660). Too small to explain a sign flip; not an exact match.

## Open

- The full-ladder analysis (probe grid + all editors on every rung) is queued. **The trigger for
  `directions/discworld-at-scale.md` depends on it**: does Edit Index gain rise with data, or
  plateau the way CE does?
- `F` (8 passes over 20M) is the "can this architecture do Othello at all" arm and is still training.
- The ≥0.95 gate is unmet at every rung, so an editability negative from this thread will need the
  qualifier *"on a model that never exceeded 0.905 legal-move mass"*.
