# GOTCHAS.md — project traps

Landmines specific to this project: stale conventions, non-comparable historical numbers,
data-generation quirks, and "if you see X, check Y first" diagnostics.

**Not the findings record** (that is what is *true* about the research) and **not the harness**
(that is how to work anywhere). This is what will silently waste a day.

Newest first. Every entry dated.

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
