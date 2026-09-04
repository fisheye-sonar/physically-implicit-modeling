# GOTCHAS.md — project traps

Landmines specific to this project: stale conventions, non-comparable historical numbers,
data-generation quirks, and "if you see X, check Y first" diagnostics.

**Not the findings record** (that is what is *true* about the research) and **not the harness**
(that is how to work anywhere). This is what will silently waste a day.

Newest first. Every entry dated.

---

### 2026-09-01 — ND (Nanda direction addition) is ill-posed for a continuous target

Nanda's method is **one fixed direction, one swept scalar**. That is coherent on Othello,
where every edit is the same categorical change (flip one tile to one of three classes),
so a single α can be correct for all 1001 cases and the class selects the direction's
sign for free.

It is **not** coherent on discworld, where each of the 192 edits is a teleport of a
different distance and direction. Three defects compound: the direction is sign-blind
(α > 0, so it can only push each read-out up, while ~half the teleports need a decrease);
the x and y rows are summed into one direction, so a single scalar moves both in a fixed
ratio; and it never consults `t − p(h)`, so it cannot tell a nudge from a teleport.
Sweeping α therefore picks the best single COMPROMISE across heterogeneous edits, which
can be bad for every individual case.

⛔ **Do not compare discworld's ND number to Othello's** — they are not the same method
applied to two worlds. Discworld ND is computed into `scores.json` for the record and is
omitted from the tables.

If a discworld analogue is ever wanted, note where each candidate lands:
* per-case magnitude `Σⱼ (tⱼ − pⱼ(h))·Wⱼ/x_std` is `Aᵀ(t − p)` — the **transpose** step,
  genuinely distinct from PI's `A⁺(t − p)` (transpose ignores A's conditioning), but
  already most of the way toward PI;
* holding the other object fixed is a **projection onto null(A_hold)**, `d − A_hold⁺(A_hold d)`,
  NOT a subtraction of its rows (subtracting would drive the other object backwards, which
  is not what "hold" means). `pim/editors/nullspace.py` already has that machinery — and
  PI satisfies this constraint already, via its full-state target.

The more ND is repaired, the more it becomes PI; the value of three editors is that they
are three mechanisms.

### 2026-09-01 — The model NEVER OBSERVES DEPTH. Do not reason from `hit_depth`.

`render_frame` computes and returns `hit_depth`, but **it is not part of the observation
and never has been.** The observation is `obs_intensity` alone — for each ray, the
*reflectivity* of the first object hit (`obs_intensity[hit] = reflectivities[first_hit]`),
a value that does not depend on distance at all. `obs_depth` is not even stored: the 20M
corpus drops it as "dead weight" (`bigcorpus.strip_shard`), and training consumes
`obs_intensity` exclusively.

So depth reaches the model through exactly two channels, both indirect:

1. **Apparent width** — a nearer object subtends more rays. This is the dominant one.
2. **Occlusion ordering** — which object wins a contested ray.

⛔ **The trap** (hit twice, most recently in the frustum-basis discussion): reading
"`render_frame` returns `hit_depth = y`, so depth is `y` not Euclidean range" and
concluding that `y` is therefore the natural depth coordinate for a probe target. That
sentence is about the renderer's *internals*. Since the model cannot see depth in any
form, the right question is not "what does the renderer compute" but "**what function of
depth is linear in what the model can actually see**" — which points at apparent width
(`width`, and its close relative `inv_y`: for an on-axis object the apparent half-width
is exactly `r/(scale·y)`), not at `y` or `rho`.

The lateral coordinate has no such ambiguity: rays are uniform in `tan θ`, so
`u = x/(scale·y)` is *literally* the ray index, and it is observed directly.

### 2026-08-21 — `is_visible` is a no-op, and object index is confounded with brightness

Two traps found while building occlusion controls. Neither invalidates a past result; both make it
easy to write a wrong one.

**1. `is_visible` means "overlaps the frustum", not "is unoccluded".** And every current dataset
sets `always_in_frustum=True`, which makes the field **identically True** — measured 100.00% on
`4_fixed_refl_inview`. So the `is_visible` masks in `othello_gpt/pipeline.probe_table` and
everything downstream have **never filtered a single frame**. Harmless historically (masking
nothing equals no mask) but the field is not an occlusion signal and must not be used as one.

**The actual occlusion signal is `obs_id`** — which object each ray hit, `-1` for a miss. An object
is fully occluded at frame `t` when it contributes **zero rays**: `(obs_id[:, t] == j).sum(-1) == 0`.
On `4_fixed_refl_inview` that happens for **3.1% of frames per object** (6.3% of frames have at
least one occluded object), in runs averaging 5.5 frames.

⚠ **And occlusion in this world is not absence of information.** A fully hidden disc must lie
*behind* the visible one and inside its angular shadow, so a single-frame MLP still reads its
position at RMSE 0.87–0.98 against a 1.83 mean-predictor floor. Any "the observation cannot supply
this, so the model must be carrying it" argument built on occlusion here is **invalid**. A clean
test needs objects that leave the frustum, which `always_in_frustum=True` prevents by construction.

**2. Object index is confounded with brightness.** `fixed_reflectivities=True` spaces
reflectivities uniformly **in the same order every sample**, so "object 0" is always the dim one and
"object 1" always the bright one. Object 1 is measurably easier to decode everywhere (visible
position MLP RMSE 0.287 vs 0.422). 2026-08-05 found the same effect from the other side ("a linear
map keys on brightness to tell the two apart"). **Per-object numbers are not interchangeable, and a
result reported for one object is not a result about "an object".** Average over objects, or report
both.

---

### 2026-08-20 — Seeding multiprocessing workers by pid makes a "seeded" corpus non-reproducible

**Symptom:** a script sets one global `SEED`, generates its training corpus in a
`multiprocessing.Pool`, and produces a **different corpus every run**. Measured here: the same
`SEED = 0` and the same `N_GAMES = 20_000` gave **1,179,692 / 1,179,508 / 1,179,665** rows across
three consecutive runs.

**Cause:** the natural-looking pool idiom seeds each *worker process* once, from something
process-local:

```python
Pool(n, initializer=lambda s: random.seed(s * 100_003 + os.getpid()), initargs=(seed,))
```

The corpus is then a function of process ids and of how the pool happened to chunk the work, not of
`seed`. Nothing errors and the numbers look fine, because a large random corpus is statistically
interchangeable with another one — which is exactly why it survives review.

**Second-order damage:** any cache keyed on a property of the data (row count, a content hash)
**can never hit**, so the expensive step silently recomputes every run. That is how this was found —
a 37-minute probe grid refitting itself when it should have loaded from disk in seconds.

**Fix:** seed per **work item**, not per worker, so each element is a pure function of
`(seed, index)`:

```python
def _one(args):
    i, seed = args
    random.seed(seed * 1_000_003 + i)
    return generate(i)

pool.imap(_one, [(i, seed) for i in range(n)], chunksize=64)
```

**Check:** generate twice with the same seed and assert equality, and once with a different seed and
assert inequality. Two lines, and it is the only thing that actually catches this.
`othello_transfer/othello_data.synthetic_games` now does the above and is verified both ways.

⚠ **Blast radius:** the `othello_transfer` runs of 2026-08-20 before this fix each used a *different*
20k-game corpus. The reported numbers were stable across them (null baseline 2.723 identical, probe
error 0.56–0.58%), so no conclusion changes — but pre-fix runs are not bit-reproducible.

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

### Standing — do not render tables with pandas, even though it is now installed

**Symptom:** a notebook renders a dense value set as a DataFrame instead of a markdown table.

**Status change 2026-08-20:** pandas *is* now in the `.pim` venv, pulled in as a transitive
dependency of `seaborn` when `othello_transfer/` vendored Li et al.'s repo (`data/othello.py`
imports `seaborn`, `psutil` and `pgn` at module level). Verified additive only — numpy 2.4.3
and torch 2.11.0+cu130 unchanged, 178 tests still pass. The rule below is unchanged and is
now a *convention* rather than an environment fact.

**Fix / check:** render dense value sets with `display(Markdown(...))` tables — visible
row/column structure, not aligned-monospace `print` and not a DataFrame repr.

---

### Standing — `pgrep -f <name>` / `pkill -f <name>` match the shell running them

**Symptom:** two flavours, both seen on 2026-08-21/22.
1. A wait loop guarded by `pgrep -f "train\.py"` **never goes false**, because the polling
   shell's own command line contains the string. An overnight chain sat idle for **6 hours**
   after its job finished at 02:31.
2. `pkill -f "<script>"` or `for p in $(pgrep -f "<script>")` **kills the shell executing it**,
   mid-command. Happened three times in one night; twice it silently prevented a fix from being
   written, so the old, broken version stayed on disk and looked applied.

**Cause:** `-f` is a substring match over **every** command line on the box — including the
current shell, any `tail`/`grep`/editor mentioning the name, and the agent's own tool calls.

**Fix / check:**
```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader   # GPU work — authoritative
ps -p "$PID" >/dev/null                                     # a PID captured at launch
ps -eo pid,args | awk -v me=$$ '$1!=me && /bash .*driver\.sh/ {print $1}'   # exclude self
```
Always give a wait loop a timeout with a loud message, so a guard bug becomes a late start
rather than a hang. After any `kill`, **verify the intended change actually landed** — do not
assume the rest of the compound command ran.

### Standing — figure-heavy notebooks exceed the Read token cap

**Symptom:** reading a notebook blows the context window.

**Cause:** embedded PNG outputs.

**Fix / check:** extract printed tables with a small script that iterates cells and prints
`stream` / `text/plain` outputs while skipping `image/png`. Keep notebook outputs lean.

**The editing consequence (2026-08-21).** `NotebookEdit` requires the notebook to have been read
first, so once a notebook is over the cap it becomes **uneditable by the normal tool** — the case
that bit `othello_transfer/controls.ipynb` (26k tokens against a 25k cap) and
`probe_transfer.ipynb`. `CLAUDE.md` §8 forbids manipulating notebook JSON through Bash for good
reason, and this is the one sanctioned exception: use a helper that touches **only** `source`,
identifies the cell by its `# [N]` tag rather than by index, and **asserts the cell inventory is
unchanged** afterwards. Never edit outputs, `execution_count`, or metadata by hand — re-execute
with `nbconvert --inplace` instead. Prefer restructuring the notebook so it stays under the cap.

## 2026-09-01 — `fit_probes` is a 20 GB call, and OOM takes the whole desktop with it

`dwb.fit_probes(..., n_seq=30_000)` calls `collect_residuals`, which materialises
**every residual point for every frame at once**: `30_000 × 39 × 512 × 9 × 4 B ≈ 21.6 GB`.
On a 59 GB box that is survivable alone and fatal in company. It has now killed VSCode
twice: once run in the foreground, once by launching a pilot in the background and then
running a second probe-touching script **in the same message**, so two collections
overlapped.

**Rules, in order of how often I break them:**

1. **Never start a second probe/model script while a pilot is running.** Check first:
   `ps -eo pid,etime,args | awk '/pilots\//&&!/awk/'`. One probe job at a time, always.
2. **Never call `fit_probes` from an inline `python - << EOF` heredoc.** Those run in the
   foreground, uncapped, and take the editor down with them. Pilots go in
   `experiments/<name>/scripts/*.py`, launched detached with a memory cap.
3. **To inspect a probe, load the cache file — do not refit.** `probes_<hash>.pt` is a few
   MB; `torch.load(..., map_location="cpu")` answers almost every question ("are these two
   probes the same?", "what is W?") for ~0 memory. `INDEX.md` in the cache dir maps hash →
   provenance. Refitting to look at a probe is never the right move.
4. **Cap every launch**: `systemd-run --scope -p MemoryMax=24G` (or `ulimit -v`) so a
   runaway job dies alone instead of taking the session with it.
5. `n_seq = 8_000` (≈5.8 GB) is the pilot default; 30 000 is for the scorer only, run alone.

A killed job leaves a truncated `logs/<name>/*.log` with only its header line — that is the
OOM signature, distinct from a traceback. Check `free -g` and the log tail together.

## 2026-09-01 — pre-housecleaning Othello checkpoints say `arch: "theirs"`

The original Othello trainer stamped every intermediate `ckpt/step_*.pt` of `L-oth-20m` with
`arch: "theirs"` (its name for the vendored minGPT); only `best_model.pt` was re-stamped
`transformer_l_tokens` during the housecleaning. `pim.models.registry._infer_arch` trusts an
explicit `arch` key, so loading one of those files raises `KeyError: unknown arch 'theirs'`.
The discworld intermediates carry no `arch` key at all and infer correctly from their shape.
`experiments/training_curve/scripts/make_training_curve.py` normalises the key when it lays a checkpoint out as a run
dir; do the same for any other consumer of the old `ckpt/` files. (Cost the overnight chain one
restart — the failure came after eight discworld points had scored.)

## 2026-09-02 — `collect_residuals` held the residual stack TWICE (fixed)

`np.concatenate` over a list of per-batch arrays allocates the full result while the list
is still alive, so the peak was **2×** the stack: 43 GB for Transformer-L at 30k sequences
(the "44.9 GB under a 45G cap" near-miss on 2026-09-01) and 49 GB for the 1024-wide
Recurrent-L, which was OOM-killed 40 s into scoring. Now preallocated and filled in place;
peak = one copy (21.6 / 24.6 GB). The lesson generalises: every "materialise the whole
probe corpus" path must be checked for a hidden second copy before it is trusted under a
memory cap, because the cap turns a transient into a kill.

## dw-8ray (2026-09-04): edge-pinned α grid, no-support edit cases, edits sampler attempts

* The canonical PI α grid (0.1 … 175) was tuned on 128-ray instances. On `dw-8ray` the best
  PI arm is the LAST grid value with the index still rising; the extended check
  (`experiments/dw8ray_alpha_check/`) shows the index plateaus at +0.3 by α ≈ 250–1000, so the
  canonical +0.297 is a lower bound by ~0.02. Read an edge-pinned α as "check the plateau",
  not as a wrong-units bug (the y-affine signature) — the readout errors are sane here.
* At 8 rays ~15 % of the 192 edit cases have NO differing ray between the edited and unedited
  worlds; those cases carry no Edit-Index support (nanmean drops them). Coarse observation =
  wider error bars on the mean, same definition.
* Radius-1.0 discs make a collision-free, in-frustum teleport rarer: `generate_dataset.py`'s
  default `--max-edit-attempts 50` fails ~1 case in 100. dw-8ray is generated with 2000
  (cases that succeed within 50 draws are unchanged); `bigcorpus` carries the flag for the
  shards' throwaway edits splits too.
