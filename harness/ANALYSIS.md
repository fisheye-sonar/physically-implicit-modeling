# ANALYSIS.md — measuring things without fooling yourself

Read this **before** computing a metric, fitting anything, or making an empirical claim.

The value of diagnostic work is entirely in the judgment of whether an effect is real signal
or an artifact. **"Bug reframed as insight" is the failure mode to guard against above all
others.** Everything below serves that.

---

## 1. Standardization — the highest-value rule in this file

Consistency across implementations is central. The failure is **not** that people invent new
metrics; that is fine and often the point of an experiment. The failure is **silent
re-implementation of things that are supposed to be the same** — which produces several
mutually incompatible versions of one named quantity, all reported as though comparable.

This has been expensive. Documented outcomes from a single project: one serialization bug
written three separate times across two scripts and a notebook, costing two full
re-evaluations; five incompatible definitions of a single named metric; two different
capacities of a standard estimator quoted interchangeably for months.

**Prose does not prevent this. Code does.** The rules:

- **Every recurring computation has exactly one implementation, and it is imported.** Metrics,
  evaluation splits, standard estimators, the canonical comparison panel — one module, one
  function, imported everywhere. If you are about to re-derive one, stop and find it.
- **If it does not exist yet, write it *as* the one implementation** — in the shared module,
  not inline in the notebook that needed it first.
- **Genuinely new metric? Fine — add its registry row in the same commit.** Name, formula,
  units, better-direction. A registry that lags the code becomes a lie, and then nobody trusts
  it enough to check it.
- **Never quote two different implementations of a name as if they were one quantity.** If an
  estimator's capacity, fitting procedure, or scoring set changed, the numbers are not
  comparable and must be labelled distinctly.
- **The check before writing:** *is this supposed to be the same as something that already
  exists?* If yes, import. If no, register. There is no third option that ends well.

**Do not strip per-step or per-element curves when serializing results.** A
`{k: v for ... if not isinstance(v, list)}` filter on the way to JSON silently discards the
curve a required plot depends on, and makes it unrecoverable without a full re-run. This
exact filter has been written repeatedly. Serialize the curves.

---

## 2. Held-out sets: hold out whole units, never sub-units

**Identify the unit of independence and split on it.** When examples are derived from a
larger grouping — frames from a sequence, patches from an image, samples from a subject or
session — the sub-units within one group are near-duplicates of each other. Pooling all
sub-units and shuffling puts a test example's own neighbours in the training set.

The inflation is not subtle. Measured on one project, same estimator, same data, only the
split convention changed:

| target | hold out whole groups | hold out sub-units | inflation |
|---|---|---|---|
| a per-group constant label | **0.565** | 0.905 | **+0.34** |
| a per-sub-unit varying label | 0.924 | 0.971 | +0.05 |

**Why the first row is catastrophic:** that label is *constant across the whole group*. A
sub-unit split therefore leaves, for every test example, other examples from the same group
carrying the **identical label** — the model can recognise the group instead of decoding the
quantity. Labels that vary within a group leak much less.

Consequences: any quantity that is constant or slowly-varying within a group is where this
bites hardest. **Numbers computed under the two conventions are not comparable** — when a
score looks anomalously low against older work, check the split convention first, and assume
the newer, stricter number is the correct one. Record the convention beside every reported
score.

---

## 3. Report the whole curve, not one point

Landing an effect and holding it are different results, and mechanisms routinely invert
between them: an arm trained on a multi-step objective can start *worse* at the first step
and overtake a few steps later. **A single-step report can therefore state the opposite of
the truth.** Report the metric across the full horizon.

Two rules that travel with it:

- **Read every curve against its own arm's baseline.** A baseline that itself drifts over the
  horizon will make raw persistence look better or worse than it is.
- **Do not report a step-N ÷ step-0 "retention" ratio when the step-0 value is near zero.** It
  explodes or flips sign and means nothing.

## 4. "Gain over baseline" and "does it hold" are different questions

The gap to an arm's own baseline is the right statistic for *is this distinguishable from
doing nothing*, especially when comparing systems whose baselines differ. It is the **wrong**
statistic for persistence: a baseline can move on its own over a horizon and can also shift
as a system's underlying quality improves — so the gap can grow while the raw value is flat.

Report the **raw curve** for persistence, the **gap** for distinguishability, both when in
doubt. A gain axis also hides the case where a method only moved the output toward garbage.

## 5. Evaluation windows must be uncontaminated

**An evaluated episode contains exactly one manipulation — the one under test — anywhere the
system or the reader can see.** Two distinct failures, both real:

1. **In the scored window.** If the environment or data-generating process performs its own
   events, extra events land inside the window and every trajectory-level metric scores the
   system on things it was never told about. Filtering events on the *manipulated* entity is
   not enough; events elsewhere contaminate the same window.
2. **In the visible context.** Even with a clean scored window, events *before* the
   manipulation make the episodes structurally different from a clean evaluation set — so two
   systems evaluated on different generators are no longer comparable, which is exactly what a
   control exists to make possible.

**The fix:** generate the evaluation set with the generator's own interventions switched off,
synthesise the single manipulation under test, and construct reference outcomes by rolling
the pre-manipulation state forward under passive dynamics rather than reading later data.
**Assert** the evaluation set is clean in code rather than trusting it.

One caveat worth stating rather than assuming: contamination in the *context* does not make a
within-episode paired comparison unfair — those arms share episodes. What it breaks is
comparability *across* systems.

## 6. Score against the clean reference

Where a noisy observed signal and a clean underlying signal both exist, **every error is
scored against the clean one.** Errors add in quadrature: with a noise term `n`, a true error
`e` reads as `√(e² + n²)`. Methods differing 2× in real error can differ by ~14% on the noisy
scale. It compresses every method toward the noise level and **cannot be undone afterwards**.

If a noisy-referenced quantity is genuinely wanted, it carries "vs noisy" in its **name, axis
label, and legend**, and never shares an axis with a clean-referenced one unless both are
labelled. A noise floor bounds error only against noisy targets — against a clean reference a
perfect predictor scores zero, and sub-floor values are a normal result of denoising, not a
leak. Use the floor as a *reference scale*, never as a bound.

---

## 7. Metric discipline

- **Consistent metrics and units across everything compared.** Same metric set, same units.
  Prefer **RMSE over MSE**; never plot one and tabulate the other; never compare method A on
  metric-set X against method B on metric-set Y.
- **No invented thresholds.** Do not introduce a cutoff ("within 2% of best", "saturates at")
  unless it is already canonical in the project or you can say why *that* number is
  meaningful. An arbitrary threshold dressed as analysis tells the reader nothing the raw
  curve did not, and manufactures false precision.
- **No derived duplicates.** Before adding a panel or column, check whether it is recoverable
  from what is already shown. If it is, show it *instead of*, not *alongside*. A redundant
  metric adds no information, grows the zoo, and reads as a **contradiction** when the reader
  cannot see the identity linking the two. This applies to figure panels as much as table
  columns.
- **Every reported magnitude needs a reference scale and units.** A bare distance or residual
  is uninterpretable — show the matched reference beside it and state the normalization (raw
  magnitude vs fraction). **Name the estimator in the metric name**, never a bare generic noun
  or a pet adjective.
- **Every comparison needs a control.** If a claim is that an intervention did something,
  there is a no-intervention arm computed identically. If a claim is that structure matters,
  there is a matched arm without the structure.

## 8. High-dimensional intuitions — the everyday ones are wrong

Three that have caused real errors:

1. **Cosine is not correlation-like.** cos 0.9 is a **26° angle**, and two equal-length vectors
   26° apart differ by `2·sin(θ/2)` ≈ 0.45 of their length. Report the **angle** alongside any
   cosine, or state explicitly what would count as aligned.
2. **The mean cosine between random vectors is 0**, not `1/√H` — that is the *per-pair standard
   deviation*. Never quote it as a floor the mean should sit at. For a mean, use an empirical
   shuffled-pair control.
3. **A random vector already has `√(d/H)` of its norm in any d-dimensional subspace** — so a
   "small" projection fraction can be at or below chance. Always report the chance level, and
   when the ambient dimension varies across a comparison, plot the **enrichment**
   (value ÷ chance), never the raw fraction. The raw version manufactures a trend that is
   entirely the moving chance level.

## 9. Say precisely what failed

These are different dynamics and different diagnoses. Look at the actual output before
choosing the word.

- **Reverts** — returns toward the baseline or pre-intervention behaviour.
- **Collapses** — output degenerates, goes off-distribution.
- **Drifts** — diverges without returning.

Extend this vocabulary per project with the failure modes that domain actually has, and use
the words strictly.

## 10. Calibration

- Do **not** soften the "is this signal or an artifact?" question to make a result land. If a
  result is shaky, say so plainly, in the write-up, in the same sentence as the result.
- State claims as quantities, not verdict adjectives. Do not binarize graded quantities in
  body prose.
- Report scope honestly: how many samples, which checkpoints, which conditions. A result from
  one checkpoint is a result about that checkpoint until replicated.
- When a number contradicts earlier work, check the conventions before the science — split
  convention, reference signal, estimator capacity, units.

---

## Local instantiations (this project — not portable)

- Shared metric implementations → `pim/metrics/` (`editability.py`, `othello_moves.py`,
  `decodability.py`) — arrays in, numbers out; never re-derive at a call site
- Standard held-out read-out estimators → `pim.probes` (`fit_linear`, `fit_mlp` — MLP-128,
  held out by sequence, `check_probe_sanity` tripwire)
- The registry of canonical objects (environments, architectures, probes, editors,
  metrics, runs) → `../research/REGISTRY.md`
- Canonical scores → `runs/<topic>/<run>/scores.json`, written ONLY by
  `../notebooks/master_eval.ipynb` (no metric math in the notebook)
- Known traps, stale conventions, and non-comparable historical numbers →
  `../research/GOTCHAS.md`
