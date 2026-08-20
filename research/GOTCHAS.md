# GOTCHAS.md — project traps

Landmines specific to this project: stale conventions, non-comparable historical numbers,
data-generation quirks, and "if you see X, check Y first" diagnostics.

**Not the findings record** (that is what is *true* about the research) and **not the harness**
(that is how to work anywhere). This is what will silently waste a day.

Newest first. Every entry dated.

---

### 2026-08-19 — Two state-plumbing traps: the RSSM's prior/posterior chain, and the DiT family's hedged decode

**Symptom (1):** a mechanism that teacher-forces *extra* frames onto an existing state (freeze-time, first-obs)
behaves oddly on the RSSM but not on the GRU.

**Cause:** `model.step(obs_t, state)` expects the **posterior** state at `t−1`, while the state that is aligned
for `decode` is the **prior** at `t` (one `imagine_step` later). Round-tripping the prior back through
`state_from_flat` and calling `step` therefore advances the deterministic core **twice**. `delta_h_analysis`'s
`continue_from` does exactly this; the GRU is unaffected (its `imagine` is the identity), the RSSM is not.

**Fix / check:** keep the **posterior** chain and apply `imagine` only at read-out — see
`latent_linearity/edit_directions.py` (`observe` / `predictive`). Never round-trip a prior state through
`state_from_flat` to resume teacher forcing.

**Blast radius, measured 2026-08-19 (N=256, dataset 4):** smaller than it looks. The two paths give states
**6.5%** apart in norm and a freeze-time Edit Index of **+0.097 (corrected) vs +0.091 (legacy)** — so the
`delta_h_analysis` RSSM numbers stand, and **the double-advance is NOT the explanation for the RSSM's weak
freeze-time arm** (+0.09 vs the GRU's +0.54). That remains open.

---

**Symptom (2):** an alignment check says the DiT's `decode(state)` matches the frame it just *consumed* better
than the next one (k=−1 0.0985 vs k=0 0.1088) — "the DiT is off by one".

**Cause:** it is not. The concat DiT's last token is `(obs[t], noise at τ=1)` and it predicts `obs[t+1]` by
construction. Its **conditional-mean readout under-moves**: frame-to-frame change is 0.1024, so a prediction that
hedges toward the current frame lands closer to it than to the next one while still being a correctly-aligned
next-frame predictor. Both DiT variants do this; the GRU, RSSM and transformer do not.

**Fix / check:** settle alignment against the model's **independently published next-step RMSE vs the clean
render**, not against the argmin of a k-profile. Measured 2026-08-19: latent DiT `k=0` 0.1080 vs published
0.1083; pixel DiT 0.1088 vs 0.1089. `latent_linearity/edit_directions.alignment_profile` returns the profile and
the frame-change scale together for exactly this reading.

---

### 2026-08-19 — A probe fit on MIXED-SCALE targets barely trains the small ones

**Symptom:** a nonlinear probe scores **below** the linear probe on some target dimension. That
ordering is impossible for a strictly more expressive model that has actually trained, so it is
always a training failure, never a fact about the representation. (Second instance of this class —
the first was the 2026-08-11 undertraining fix recorded in `pim/extractors/standard.py`.)

**Cause:** an unweighted MSE taken in **raw target units** weights each output dimension by its
variance. In sim units the position dimensions have variance **3.0–3.6** and the velocity
dimensions **0.0033** — a **~1000×** gradient-share imbalance — so a single probe fit on
`(pos, vel)` together barely trains velocity. Standardising the targets *inside* the probe does not
help if `forward` un-standardises before the loss: the residual becomes
`y_std · (net − u)`, so the weighting reappears as `y_std²`.

**Measured** (2026-08-19, `W16` residual point 3, 8-output probe, held-out by sequence):

| | raw-units loss | balanced loss | velocity-only control | linear |
|---|---|---|---|---|
| obj0 vx | 0.211 | **0.518** | 0.495 | 0.252 |
| obj1 vx | 0.450 | **0.730** | 0.733 | 0.435 |
| mean velocity | **0.158** | **0.276** | 0.272 | 0.200 |
| mean position | 0.938 | 0.927 | — | — |

The balanced fit matches a dedicated velocity-only probe, so the imbalance was the entire gap.

**Fix / check:** take the loss in **standardised** target space (each dimension weighted equally),
or fit separate probes per target group. `othello_gpt/othello_probe.fit_probe` now does the former.
`pim.extractors.fit_readability_probes` still takes a raw-units MSE but **has never been bitten,
because the repo has always fit position and velocity as separate probes** (`eval_controls.py`);
it now emits a warning if handed targets spanning >100× in variance.

**Blast radius:** only probes fit on **combined** position+velocity targets — in practice just the
`full` target in `othello_gpt/`. Every position-only probe is unaffected (its four dims span 1.2×),
so all edit results that ran off the position probe stand. Pre-2026-08-19 velocity numbers from the
`othello_gpt/` full-state probe under-read and are not comparable.

**Diagnostic to keep:** *always check MLP ≥ linear per dimension.* It is the cheapest possible
tripwire for a mis-trained probe and it caught both instances.

---

### 2026-08-18 — The Edit Index cannot see direction, and largest-teleport samples are a biased draw

**Symptom:** a qualitative panel plainly shows the edit working — intensity appearing at the target,
decaying at the ghost, the untouched object left alone — while the Edit Index reads like failure.

**Cause (1): the index is a ratio of distances and is blind to direction.** It measures how *close*
the output got, so "wrong direction" and "right direction, 5% of the way" both land near the
unsteered value. Measured on `W16`/N=256, the single-frame Othello write reads Edit Index −0.538 (vs
unsteered −0.684) while the change it actually made has **cos +0.443 (64°) to the required change
against a shuffled chance level of +0.053**, and an **achieved fraction of 0.072**. Directionally
real, ~7% complete. Both readings are correct; they answer different questions.

**Fix:** `editability_metrics.direction_report(pred0, unsteered0, zones)` — report `direction_cos`
(with angle and shuffled chance) and `achieved_fraction` beside any Edit Index backing a claim about
a *direction*. Added to the canonical set 2026-08-18.

**Cause (2): sample selection.** These editors' effect grows with teleport size while the unsteered
baseline is flat, so picking the k largest teleports for a waterfall is a biased draw — the four
largest sat at the **98th percentile** (+0.07 vs a −0.54 mean). By quartile: −0.617 / −0.611 /
−0.532 / −0.395.

**Fix:** `pipeline.representative_samples(teleport, k)` spreads samples across quantile bands. State
the selection rule in the caption.

**Blast radius:** any figure in this repo whose samples were chosen as "the largest teleports" —
including the first versions of both `othello_gpt/` notebooks — reads more favourably than the mean.
The metric values themselves are unaffected.

---

### 2026-08-18 — Edit Index silently drops unscoreable episodes, and scores only ~22% of the frame

**Symptom (1):** a frame that looks nearly identical to the ground truth yet scores a strongly
negative Edit Index — "the metric must be broken".

**Cause:** it is not. The index is computed **only over the rays where the two ground-truth worlds
differ** — median **28 of 128 rays, ~22% of the frame**. The other ~78% (the untouched object plus
background) is *shared* by both worlds and carries no information about which one the output
matches, so it is excluded by design. Measured on `W16`, N=256: the unsteered output is 0.276 RMSE
from the edited world over the **full frame** but **0.564** over the differing rays, against
**0.093** to the unedited world there — about **6× closer to the world without the teleport**.
Both readings are correct and they answer different questions.

**Fix / check:** when a frame "looks right" but the index disagrees, plot the two GT worlds and the
output together with the differing mask shaded (see `othello_gpt/othello_gpt_probing.ipynb` Fig 9)
before suspecting the metric. Report full-frame RMSE beside the index if the distinction matters.

**Symptom (2):** the episode count behind an Edit Index is not what you think.

**Cause:** `_index_from` returns `np.nanmean` over per-sample values and skips episodes whose
`differing` mask is **empty** — i.e. the teleport is invisible in observation space (occluded, or
the object lands where it already looked the same). On `W16`/N=256 this is **3 episodes at step 0**
and varies by step (0–3), so the denominator quietly changes across a by-step curve. It also emits
`RuntimeWarning: Mean of empty slice`.

**Fix / check:** report the scored-episode count alongside the index when it matters; treat that
warning as information, not noise. Not a correctness bug — an unscoreable episode *should* be
dropped — but it is silent.

**Blast radius:** every Edit Index in the repo. The values are correct; the interpretation
("the frame is wrong") and the effective N are what get misread.

---

### 2026-08-17 — Two standardization debts, measured

**Symptom:** numbers or panels from different notebooks that should be comparable and are not.

**Cause — waterfalls:** **18** separate implementations of the comparison waterfall existed
(`scripts/eval_editability_endogenous.py`, `history_editing/history_tools.py`, 16 notebooks),
each re-deciding colormap, context frames, and alignment.
**Fix:** `pim.figures.waterfall_grid` is now the single implementation. New work must use it;
existing copies are unmigrated and their panels are unverified against the spec.

**Cause — probes:** only `scripts/eval_action_sweep.py` calls the standard
`pim.extractors.fit_readability_probes`. `eval_controls.py`, `eval_editability_endogenous.py`,
`eval_action_editors.py`, `eval_endogenous.py`, `run_eval.py`, `sweep_rssm.py`,
`train_action_editors.py`, and `train_editable_gru.py` each fit their own `lstsq`/MLP path.
**Fix:** route readability numbers through the standard. Where a script legitimately needs a
different estimator (e.g. the frozen 1×128 `MLPExtractor` inside the MLP Grad Steering editor —
a *different object*), label it distinctly and never quote it as a readability R².

**Blast radius:** any cross-notebook comparison of readability R², and any qualitative
comparison of waterfalls across threads.

---

### 2026-08-14 — Probe held-out split: whole trajectories, not frames

**Symptom:** a velocity R² that looks anomalously *low* against an older notebook — or an
implausibly high one (≈0.90+) in older work.

**Cause:** two different held-out conventions. Holding out **frames** (pool ~19,000
frame-examples, shuffle, keep 30%) leaves a test frame's immediate neighbours from the *same*
trajectory in the training set — and in this world **velocity is constant for a whole
trajectory**, so those neighbours carry the *identical label*. The probe recognises the
trajectory instead of decoding the velocity. Holding out **trajectories** (20% of sequences
entirely) is the standard, and is what `fit_readability_probes` does.

Measured 2026-08-14 on `controls/H256`, same 2×256 probe, same late-t window:

| target | hold out trajectories | hold out frames | inflation |
|---|---|---|---|
| velocity | **0.565** | 0.905 | **+0.34** |
| position | 0.924 | 0.971 | +0.05 |

**Fix / check:** use `pim.extractors.fit_readability_probes`. Never hand-roll a probe.
`scripts/eval_controls.py` still uses the old frame-holdout convention — treat its numbers
accordingly.

**Blast radius:** **every velocity number in this repo from before 2026-08-06** is on the
frame-holdout convention and is **not comparable** — including `runs/controls/eval`'s 0.877
and `findings/editability.md`'s 0.94. When two velocity numbers disagree, check the split
convention before the science; the newer number is probably correct.

---

### 2026-08-14 — Velocity readability is reported late-t (t ≥ 15)

**Symptom:** a velocity R² that reads much lower than every other notebook in the repo.

**Cause:** all-t reporting. The belief has not converged before t ≈ 15, so an all-t velocity
R² under-reads it badly (H256: all-t MLP **0.784** vs late-t MLP **0.877**).

**Fix / check:** report velocity late-t. Split position the same way for consistency, though
it is far less sensitive. State the window in the definitions table — "late-t = frames t ≥ 15"
is not self-evident from an axis label.

---

### 2026-08-14 — The edit set must be intervention-free

**Symptom:** trajectory metrics (GT-traj RMSE, fidelity ratio, Edit Index by step) that score
the model on events it was never told about; unexplained jumps in the waterfall context.

**Cause:** `datasets/7_cont_teleport` fires a teleport on **~30% of transitions**. Extra
events land after the edit frame (contaminating the scored horizon) and before it
(contaminating the visible context, making dataset-4 and dataset-7 episodes structurally
different — which is exactly what a cross-family control exists to make possible). Filtering
the *edited* object's later actions is not enough; the other object's contaminate the same
window.

**Fix / check:** generate the edit set with the world's own interventions off
(`--p-action 0.0`), **synthesise** the single edit, and construct both reference futures by
rolling the frame-`ef` state forward under passive dynamics — never by reading later frames
from the dataset. `scripts/eval_action_sweep.py` **asserts** this rather than trusting it.

**Blast radius:** cross-model-family comparisons only. A within-episode unsteered-vs-edited
comparison stays fair even with contaminated context, because those arms share episodes.

---

### 2026-08-06 — Two different MLP probes were quoted interchangeably

**Symptom:** MLP R² values from different notebooks that do not line up.

**Cause:** `MLPExtractor` on its defaults (**1×128**, scored **in-sample**) in
`00_master_editability` and `controls/`, versus a hand-rolled **2×256** probe scored
**held-out** in `iterative_probing` and `nonlinear_gru`. Two axes conflated at once: probe
capacity, and whether the score is in-sample.

**Fix / check:** the standard is `fit_readability_probes` — linear `lstsq` plus a 2×256 ReLU
MLP, 30 epochs, Adam `lr=1e-3`, both fit on the same 80% of **sequences** and scored on the
same held-out 20%, R² against the **train** mean. An in-sample R² is not a readability claim.

**Note:** the **MLP Grad Steering** editor's frozen 1×128 `MLPExtractor` is a *different
object* with a different purpose. Do not change it, and never quote one as the other.

---

### 2026-08-04 — Observation-space error is scored against `clean_obs`, always

**Symptom:** methods that differ 2× in real error appearing ~14% apart; everything compressed
toward the noise level.

**Cause:** scoring against the noisy `obs` instead of the clean render. Errors add in
quadrature — `err_noisy ≈ √(err_clean² + noise²)` — and on the canonical dataset the noise
term is **0.1539**. A true 0.05 reads as 0.162; a true 0.10 reads as 0.185.

**Fix / check:** score every observation-space error against `clean_obs`, including one-off
panels that do not run the §4 scorecard — that is exactly where it keeps breaking
(`editability_structure` and `rssm_structure/rssm_state_geometry` each built a
`gt_obs = edits.obs[...]` panel titled "error vs post-edit GT"). It also governs the **GT
column of every waterfall**. If a noisy-referenced quantity is genuinely wanted, it carries
`vs noisy` in its name, axis label, and legend. `pim/eval/controllability.py` is the reference:
every field is suffixed `_vs_clean` / `_vs_noisy`.

**Note:** `noise_floor_rmse` is **not a floor** against clean targets — a perfect predictor
scores 0, and sub-floor values are the normal result of a recurrent model denoising many
frames. Use it as a reference scale ("no better than echoing the input"), never as a bound.

**Blast radius:** cannot be undone from a reported number after the fact. Re-compute.

---

### 2026-08-03 — `edit_index_by_step` gets stripped on the way to JSON

**Symptom:** the required per-step Edit Index plot is impossible to make without a full
re-evaluation.

**Cause:** a `{k: v for k, v in scorecard.items() if not isinstance(v, list)}` filter applied
when serialising `edit_scorecard` output. It silently discards the curve.

**Fix / check:** serialise the curves. This exact filter has been written **four** times —
`eval_action_sweep.py` ×2, `eval_action_editors.py`, and once in a notebook cell copying
published results — costing two full re-evaluations.

---

### 2026-07-30 — Retired editability metrics

**Symptom:** an old number on a scale that no longer exists.

**Cause:** `reach % of swap`, `collateral % of swap`, `selectivity`, and `ghost ratio` were
retired 2026-07-30. They scored *change* rather than *correctness*, and normalised by a
model-dependent soft reference.

**Fix / check:** the canonical set is **Edit Index** plus **Target / Ghost / Collateral /
Edit-frame / GT-traj RMSE** and the **fidelity ratio**, implemented in
`scripts/editability_metrics.py`. Do not reintroduce the retired ones.

**Blast radius:** treat pre-2026-07-30 numbers on that scale as **not comparable**.

---

### 2026-07-08 — Tangent rotation ("curvature") is not scale-normalized

**Symptom:** curvature values that appear to differ across architectures or notebooks.

**Cause:** the metric is a mean principal angle between local-PCA tangents and is **not**
distance- or scale-normalized. Absolute degrees are a density and latent-scale artifact.

**Fix / check:** compare only within one notebook at fixed density. A normalization fix is
owed: `research/directions/curvature-metric-normalization.md`.

---

### Standing — pandas is not installed in the `.pim` venv

**Symptom:** a notebook cell fails on `import pandas`.

**Fix / check:** render dense value sets with `display(Markdown(...))` tables — which is the
required form anyway (visible row/column structure, not aligned-monospace `print`). Do not
reach for a DataFrame.

---

### Standing — figure-heavy notebooks exceed the Read token cap

**Symptom:** reading a notebook blows the context window.

**Cause:** embedded PNG outputs.

**Fix / check:** extract printed tables with a small script that iterates cells and prints
`stream` / `text/plain` outputs while skipping `image/png`. Keep notebook outputs lean.
