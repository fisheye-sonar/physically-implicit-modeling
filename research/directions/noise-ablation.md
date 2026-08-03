# Direction: Noise ablation — which of our two noise sources is doing the work?

**Tag:** `[in-frame]` · **Sub-questions:** 1, 2, 3 (all of them) · **Status:** active (2026-07-30) ·
**Complexity:** low (3 new datasets + 3 runs; existing metric suite) · **Model:** GRU only.

Origin: Michael's controls, 2026-07-30.

## The gap this closes

Every dataset in this repo carries **two** independent noise sources, and no result has ever separated them:

| source | config field | value | what it corrupts |
|---|---|---|---|
| **observation noise** | `obs_noise_std` | **0.2** | the 1D intensity scan — *sensing*. The world is exact; the model's view of it is not. |
| **position noise** | `position_noise_std` | **0.04** | the discs' positions each step (Brownian diffusion on top of constant-velocity drift) — *the world itself*. Makes the true dynamics stochastic. |

These are conceptually opposite. Observation noise leaves a **deterministic world** that is merely hard to see;
position noise makes the world **genuinely unpredictable** no matter how well it is seen. They plausibly have
opposite effects on the latent: sensing noise should push toward *averaging/filtering* (a smoother, more
canonical state), while dynamics noise caps how much long-horizon structure is worth representing at all.

We have been reading every finding off the both-on cell of a 2×2 we never filled in.

## Design

**Full 2×2, one variable at a time**, everything else matched to `datasets/4_fixed_refl_inview` exactly
(2 objects, 40 frames, 128 rays, open boundary, fixed reflectivities, always-in-frustum, 90k/10k/10k/10k splits,
edit frame 20, seed 0):

| cell | obs noise | position noise | dataset | run |
|---|---|---|---|---|
| **neither** | 0.0 | 0.00 | `datasets/9_obsnoise0_posnoise0` | `runs/controls/N_obs0_pos0` |
| **world only** | 0.0 | 0.04 | `datasets/10_obsnoise0_posnoise004` | `runs/controls/N_obs0_pos004` |
| **sensing only** | 0.2 | 0.00 | `datasets/11_obsnoise02_posnoise0` | `runs/controls/N_obs02_pos0` |
| **both (the status quo)** | 0.2 | 0.04 | `datasets/4_fixed_refl_inview` | `runs/controls/H256` (shared with the hidden-size sweep) |

Training identical across all four: `H=256`, 400 epochs, batch 256, AdamW lr 1e-3, weight decay 1e-4, seed 0.

## Hypotheses (state before running)

1. **Absolute RMSE is not comparable across cells and must never be compared raw.** Each model's error must be
   read against **its own** noise floor and copy-previous-frame baseline — the noise-free cells have a floor of
   ~0, so a "better" RMSE there is mostly bookkeeping. This is the trap the endogenous thread already fell into
   once (the 0.05-vs-0.2 deviation); the notebook states it up front and plots baselines per model.
2. **Observation noise is the canonicalising force.** Removing it (cells 1, 2) should *raise* the fiber residual
   — with a clean view there is no filtering pressure, so the state can afford idiosyncratic extra content.
3. **Position noise limits recoverability of velocity, not position.** Removing it (cells 1, 3) should raise
   velocity R² most, since velocity becomes exactly constant and therefore perfectly inferable from history.
4. **Editability is unchanged by both.** Ghost ratio ≈1.0 for structural editors in all four cells. *If the
   noise-free world becomes grabbable, the §4 negative was partly a stochasticity artifact — a real and
   surprising result, and the cheapest possible test of it.*

## Readouts (identical set + identical units for every model)

> **Metric note (added 2026-07-30, after this brief was written).** The §4 readouts below were pre-registered using
> the old ratio metrics (`reach % of swap`, `ghost ratio`, `selectivity`). Those were **retired** partway through
> this thread — they scored *change* rather than *correctness* and normalised by a model-dependent soft reference.
> The results were recomputed on the **canonical set** (`../../notebooks/experiments/editability/METRICS_AND_EDITORS.md` §4,
> implemented in `scripts/editability_metrics.py`): **Edit Index** in [−1,+1] plus **Target / Ghost / Collateral /
> Edit-frame / GT-traj RMSE** and the **fidelity ratio**. The hypotheses are left as originally written — they are
> the pre-registration — and map over as: "ghost ratio stays ≈ 1.0 for structural editors" ⇒ "the Edit Index stays
> near the unsteered (−1) end for structural editors".

1. **Predictive quality** — training/validation loss curves; free-run RMSE vs rollout step, each model against
   **its own** dashed baselines (copy-previous-frame, observation noise floor, random frame) from
   `pim/eval/baselines.py`. The per-model floor is the whole point — see hypothesis 1.
2. **Recoverability** — position and velocity R², **linear and MLP**.
3. **Canonicality** — fiber residual, **linear and MLP** (fraction of ‖h‖).
4. **Editability** — per-step RMSE against the time-evolving post-edit GT (`clean_obs[ef+s]`, compared at the
   same step), **ghost ratio**, **reach (% of swap)**, collateral, selectivity, standard editor suite plus the
   oracle-observation and decoder-gradient brackets.
5. **Waterfalls** — canonical spec, one column per noise cell at a fixed editor.

## Presentation constraint

Four cells of a 2×2, so the layout should *look* like a 2×2 wherever possible: use a consistent colour per cell
and a consistent ordering (neither → world only → sensing only → both) in every figure and table, so the reader
can track a cell across panels without re-reading legends. Scalar metrics go in grouped bars (metric on the
x-axis, one bar per cell); per-step curves get one panel per editor with a shared legend. One consolidated table
at the end. Descriptive labels — "sensing noise only (obs 0.2, pos 0.00)", never `N_obs02_pos0`.

## Deliverables

`notebooks/experiments/editability/controls/noise_ablation.ipynb`, PNGs to `/tmp/noise_ablation/`, dated
`research/scratch/2026-07-..-noise-ablation.md`. Registry: `notebooks/experiments/editability/controls/CONTROL_RUNS.md`.
Do NOT edit `findings/` or `RESEARCH.md`. Short, crisp notebook.
