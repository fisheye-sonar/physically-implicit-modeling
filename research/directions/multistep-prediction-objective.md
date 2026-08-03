# Direction: Does a multi-step (rollout) training objective change the latent world structure?

**Tag:** `[reframe]` · **Sub-question:** 3 (editability) + 1/2 (geometry/identifiability) ·
**Status:** proposed · **Complexity:** medium (modify training objective + retrain; GRU primary, RSSM
capped at ≤2–3h total and cut if slower)

> A **standalone experiment in its own new notebook** —
> `notebooks/experiments/editability/multistep/multistep_objective_structure.ipynb`. **Do NOT touch the master
> notebook.** **Replicates most of the master metric spread** (§1 geometry, §2 recoverability, §3
> fiber-collapse, §4 latent editing) for the new architectures vs the single-step baseline.

## Why this exists
Our GRU/RSSM are trained **purely for next-step accuracy**. Any model *can* be rolled out multi-step;
the question here is about the **training objective**: does backpropagating a **multi-step rollout loss**
(the model receives obs up to frame `n`, then free-runs to `n+w` and is penalized on all `w` predicted
frames, BPTT through time) produce a latent world state with different geometry / recoverability /
canonicality / **editability**? Intuition: a next-step-only loss never forces the state to be
*coherent under its own iterated dynamics*, which may be exactly the property editing needs.

## Objective
Two settings, both vs the single-step baseline (`w=1`):
- **Minimal:** `w=2` (two steps ahead).
- **Big:** `w≈5` (or larger if stable) — a genuinely multi-step rollout objective.

**Loss (free-running / "overshooting"):** teacher-force to build context, then from each context state
free-run `w` steps feeding the model's **own predictions** back in (like `predict_step` — decode →
step), and take MSE against the true obs at each of the `w` offsets; average over start indices and
offsets. BPTT flows through the whole `w`-step imagination. (Slide the start index across the sequence so
every frame contributes.) Implement as a new training script / flag; do NOT alter the baseline script's
default behaviour.

## Models to train (GRU; each ~9 min single-step — multi-step is w× the rollout but still cheap)
- Baseline `w=1`: reuse `runs/gru/7_dset4_gru_400epochs` (dataset 4, 256 hidden, 400 ep).
- `w=2` GRU, `w≈5` GRU — same architecture / data / hidden size / epochs, only the objective changes.
- **RSSM: nice-to-have only.** Multi-step for the RSSM (latent overshooting) is more involved; **cap the
  RSSM at ≤2–3h total and cut it entirely if it would exceed that.** GRU is the primary deliverable.

## Watch-items
- **Blur / mode-collapse:** multi-step objectives can push the decoder toward blurry mean-hedging.
  Report **predictive sharpness** (not just editability) — e.g. obs high-frequency energy / a rollout
  sharpness metric — so a "more editable" latent that is actually just blurrier is caught.
- Report next-step predictive quality too (does multi-step help or hurt the 1-step metric?).

## Metrics — replicate most of the master spread, for w∈{1,2,5}
- **§1 Geometry:** intrinsic dim (TwoNN + MLE) + curvature.
- **§2 Recoverability:** linear + MLP probes for `(pos, vel)`, early-t vs late-t.
- **§3 Fiber-collapse:** `g(pos,vel)→h` residual; linear→MLP drop.
- **§4 Latent editing head-to-head (headline):** the master editor line-up + GT/Unsteered/true-state-swap
  refs, obs-space selectivity/ghost/persistence. Does the multi-step objective make the latent more
  editable?
- Plus the sharpness / next-step quality panel above.

## Figures
Light academic theme for §1–§3 (all `w` color-coded, GT/reference columns); dark-theme observation-space
waterfalls for §4 (GT(sim) column, green target / red ghost). Build every figure to hold the 3 `w`
settings side by side. Definitions table up front. Both plots AND tables.

## Deliverables
- Executed `notebooks/experiments/editability/multistep/multistep_objective_structure.ipynb` — run **synchronously
  in-turn**, 0 error cells. PNGs → `/tmp/multistep_objective/`. New checkpoints in gitignored `runs/`.
- Dated note `research/scratch/2026-07-16-multistep-objective-structure.md`: does the multi-step
  objective move §1–§4 (esp. editability) vs single-step? Blur trade-off? RSSM done or cut (say which
  and why). Mark `→ FLAG FOR PROMOTION` if signal. Do NOT touch the master notebook, `findings/`, or
  `RESEARCH.md`.

## Bootstrap
GRU `runs/gru/7_dset4_gru_400epochs` (matched baseline) + master §1–§4 helpers; data
`datasets/4_fixed_refl_inview` (train for retraining; test/edits for eval). Paths 3-deep. RSSM baseline
`runs/rssm/4_dset4_refined_best` only if the RSSM leg is attempted within budget.
