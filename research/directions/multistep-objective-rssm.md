# Direction: Multi-step training objective — RSSM replication

**Tag:** `[in-frame]` · **Sub-question:** 1/2/3 · **Status:** proposed (write-up only; DO NOT execute
until Sevan gives the go-ahead) · **Complexity:** medium-high (RSSM retraining ×≤3; compute-capped)

> Companion to `multistep-prediction-objective.md` (the GRU version, DONE: clean negative). **New
> standalone notebook** `notebooks/experiments/multistep/multistep_objective_rssm.ipynb`. **Do NOT touch
> the master notebook** or the GRU multistep notebook.

## Why this exists (why the RSSM specifically matters here)
The GRU multi-step-objective result was a clean **negative** (rollout accuracy up; editability /
identifiability / canonicality flat-to-slightly-worse). But the RSSM is the more interesting test: it is
**architecturally designed for multi-step latent rollout** (prior "imagination" + KL-regularised latent,
à la PlaNet/Dreamer latent overshooting), not next-step reconstruction. So a multi-step *training* objective
is closer to its native regime, and the negative might NOT replicate. Sevan's expectation: it probably
replicates, but this is genuinely worth checking because it's the architecture the objective was built for.

## Objective (mirror the GRU version, adapted to the RSSM)
Three training regimes, compared: single-step (`w=1`) baseline vs multi-step `w=2` and `w≈5`.
- **RSSM multi-step / latent overshooting:** teacher-force (posterior) to a context state, then **imagine
  `w` steps forward through the PRIOR** (no observations), decoding each imagined latent, and penalise the
  reconstruction of the true obs at each of the `w` steps (plus the RSSM's usual KL terms). This is the
  RSSM-native analogue of the GRU free-run overshoot — do NOT just copy the GRU's decoded-obs feedback;
  use the prior transition. Slide the context start index across the sequence as the GRU version does.
- `w=1` reduces to the standard single-step ELBO objective — verify this analytically + numerically, and
  use the existing refined RSSM training recipe (`scripts/train_rssm.py`, `runs/rssm/4_dset4_refined_best`
  config) as the `w=1` reference so the only lever is the overshoot horizon.

## Compute cap (HARD)
- **≤2–3h total**, GRU-run discipline. RSSM training is the expensive leg. If a single RSSM train would
  blow the budget: cut `w=5` first (keep `w=1` vs `w=2`), then reduce epochs. **Report what was run vs cut
  and why** in the note — a partial result (w=1 vs w=2 only) is acceptable and useful.
- **Follow the decoupled-execution rule (WORKER.md):** train each RSSM via a **standalone foreground
  script call** (not inside the notebook); the analysis notebook only *loads* checkpoints. Never end a turn
  with a train still running.

## Metrics — same spread as the GRU multistep notebook, per `w`
- **§S sharpness / next-step quality** (blur watch-item): next-step RMSE vs **clean** obs, open-loop horizon
  RMSE, rollout total-variation sharpness ÷ GT.
- **§1 geometry, §2 recoverability (pos/vel lin+MLP, early/late-t), §3 fiber-collapse, §4 latent editing
  head-to-head** (master editor line-up + GT/Unsteered/true-state-swap refs, obs-space
  selectivity/ghost/persistence).
- **RSSM-specific:** report the **deterministic `h`** and **stochastic `s`** states **separately** (det is
  the primary world-state carrier — the GRU work showed det-only carries essentially all the position code).
  Compute every §1–§4 metric on det-only, and note s-only where cheap.
- **Consistency with the GRU notebook:** same metric definitions, same units (RMSE not MSE), same
  clean-vs-noisy evaluation targets. Note the two known metric caveats to avoid inheriting them:
  (i) evaluate editability against the **time-evolving clean GT** (`clean_obs[ef+s]`), NOT a frozen
  edit-frame target render; (ii) if a curvature/tangent-rotation number is reported, it is **not
  distance-normalised** so it is only comparable *within this notebook at fixed sample density* — state
  the bank size/K used and do not compare its absolute degrees to other notebooks.

## Deliverables
- Executed `notebooks/experiments/multistep/multistep_objective_rssm.ipynb` (0 error cells). PNGs →
  `/tmp/multistep_objective_rssm/`. New checkpoints in gitignored `runs/`.
- Dated note `research/scratch/2026-07-XX-multistep-objective-rssm.md`: does the multi-step objective move
  §S/§1–§4 for the RSSM (esp. editability)? Does the GRU negative replicate on the architecture built for
  multi-step? det vs stoch. What was run vs cut under the cap. Flag for promotion if signal. Do NOT touch
  the master notebook, `findings/`, or `RESEARCH.md`.

## Bootstrap
RSSM `runs/rssm/4_dset4_refined_best` + `scripts/train_rssm.py` (add the overshoot horizon as a new flag or
a new uniquely-named script — do NOT overwrite the existing single-step script). Data
`datasets/4_fixed_refl_inview` (T=40, R=128, ef=20, 2 obj; noisy obs_noise_std=0.2). Mirror the GRU
notebook `notebooks/experiments/multistep/multistep_objective_structure.ipynb` for the metric pipeline and
the master §4 editors. Paths 3-deep.
