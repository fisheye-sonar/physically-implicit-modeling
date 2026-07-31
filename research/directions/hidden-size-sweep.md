# Direction: Hidden-size sweep — how do the four affordances scale with latent capacity?

**Tag:** `[in-frame]` · **Sub-questions:** 1, 2, 3 (all of them) · **Status:** active (2026-07-30) ·
**Complexity:** low (existing training script, existing metric suite; 5 runs) · **Model:** GRU only.

Origin: Michael's controls, 2026-07-30.

## The gap this closes

Every result in this repo — the identifiability numbers, the fiber residual, the whole §4 editability negative —
was measured at **one hidden size, `H=256`**, chosen by default and never justified. That is a load-bearing
unexamined constant. Two specific worries:

- **The editability negative could be a capacity artifact in either direction.** A latent with far more
  dimensions than the world has degrees of freedom (the world state is 8 numbers: 2 objects × (x, y, vx, vy))
  has enormous room for the "extra" non-canonical content that the fiber residual keeps measuring — and a
  probe-directed write into a 256-dim space is a write into a mostly-unconstrained space. Conversely, a latent
  squeezed near 8 dimensions is *forced* to be canonical, and might be exactly where a grabbable handle appears.
- **Fiber residual is reported as a fraction of ‖h‖ but never against a capacity axis.** "34% of ‖h‖ is not
  explained by (pos, vel)" means something different at `H=8` than at `H=512`.

There are stale `H∈{16, 64, 512}` runs in `runs/gru/` but they are on **dataset 3** at **50 epochs** — not
comparable to anything current. This direction replaces them with a matched sweep.

## Design

**One variable: `hidden_size ∈ {8, 32, 128, 256, 512}`.** Everything else identical and identical to the
existing dataset-4 baseline recipe: `datasets/4_fixed_refl_inview` (obs noise 0.2, position noise 0.04),
400 epochs, batch 256, AdamW lr 1e-3, weight decay 1e-4, seed 0, 1 layer, no dropout.

Runs: `runs/controls/H{8,32,128,256,512}`. `H256` is the shared baseline that Directions 1 and 3 also use.

**Reference points for reading the sweep.** The world's true state is **8 numbers**; the observation is **128
rays**. So `H=8` is exactly at the world's dimensionality (and far below the observation's), `H=128` matches the
observation, `H=512` is 4× over-complete in observation terms and 64× in world terms.

## Hypotheses (state before running)

1. **Predictive quality saturates early** — most of the gain is in by `H≈128`; `H=8` is materially worse.
2. **Recoverability is non-monotonic in the interesting way.** Position R² should be high everywhere, but the
   **linear** readout should be *best at small `H`* (a squeezed latent cannot afford a curved embedding) even
   where absolute prediction is worse. Velocity, the historically hard coordinate, is where capacity should
   matter most.
3. **Fiber residual falls with smaller `H`** — near-forced canonicality at `H=8`.
4. **Editability does not improve at any `H`.** The §4 negative is structural, so ghost ratio should sit ≈1.0
   for structural editors across the whole sweep while the decoder-gradient oracle and oracle observation succeed
   at every `H`. *If a small-`H` model does become grabbable, that is a major positive and points straight at
   the RESEARCH.md endgame — capacity was the thing standing between us and an object handle.*

## Readouts (identical set + identical units for every model — the comparison rule)

> **Metric note (added 2026-07-30, after this brief was written).** The §4 readouts below were pre-registered using
> the old ratio metrics (`reach % of swap`, `ghost ratio`, `selectivity`). Those were **retired** partway through
> this thread — they scored *change* rather than *correctness* and normalised by a model-dependent soft reference.
> The results were recomputed on the **canonical set** (`../../notebooks/experiments/editability/METRICS_AND_EDITORS.md` §4,
> implemented in `scripts/editability_metrics.py`): **Edit Index** in [−1,+1] plus **Target / Ghost / Collateral /
> Edit-frame / GT-traj RMSE** and the **fidelity ratio**. The hypotheses are left as originally written — they are
> the pre-registration — and map over as: "ghost ratio stays ≈ 1.0 for structural editors" ⇒ "the Edit Index stays
> near the unsteered (−1) end for structural editors".

1. **Predictive quality** — training/validation loss curves; free-run RMSE vs rollout step with the standard
   dashed baselines (copy-previous-frame, observation noise floor, random frame) from `pim/eval/baselines.py`.
2. **Recoverability** — position and velocity R², **linear and MLP**.
3. **Canonicality** — fiber residual, **linear and MLP** (fraction of ‖h‖).
4. **Editability** — per-step RMSE against the time-evolving post-edit GT (`clean_obs[ef+s]`, compared at the
   same step), **ghost ratio**, **reach (% of swap)**, collateral, selectivity, for the standard editor suite
   plus the oracle-observation and decoder-gradient brackets.
5. **Waterfalls** — canonical spec, one column per hidden size at a fixed editor, plus one editor-by-editor
   panel at the baseline `H`.

## Presentation constraint (this is the actual difficulty)

Five models × four metric families is a lot of numbers and the notebook must stay readable. Rules:
- **Capacity is an axis, not a series, wherever it can be.** Scalar metrics (R², fiber residual, ghost, reach)
  go on **`H` as the x-axis** — that turns "5 series to disentangle" into one line per metric and makes the
  trend the visual content. This is the main layout decision.
- Per-step curves (rollout RMSE, edit-vs-GT) *do* need one series per model — those get a shared legend and one
  panel per editor, never all editors in one panel.
- One consolidated numeric table at the end (`display(Markdown(...))`, not `print`), not a table per figure.
- Descriptive labels everywhere: "H=256 (baseline)", never a bare code.

## Deliverables

`notebooks/experiments/editability/controls/hidden_size_sweep.ipynb`, PNGs to `/tmp/hidden_size_sweep/`, dated
`research/scratch/2026-07-..-hidden-size-sweep.md`. Registry: `notebooks/experiments/editability/controls/CONTROL_RUNS.md`.
Do NOT edit `findings/` or `RESEARCH.md`. Short, crisp notebook.
