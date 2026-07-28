# Direction: Does the action space individuate objects in the latent world? (generalization test)

**Tag:** `[reframe]` · **Sub-question:** 3 (editability/objecthood) + 2 (identifiability) · **Status:** active
(2026-07-17) · **Complexity:** high (new continuous-action sim + model + 3 datasets + 4 GRUs + rich analysis)

> Standalone experiment, **new** notebook `notebooks/experiments/editability/actions/action_space_object_individuation.ipynb`.
> **Do NOT touch** the master notebook or the Exp-2 notebook. GRU only (RSSM is a later follow-up).

## Why this exists (read carefully — this is NOT "bigger actions")
Exp 2 found that training on small (0.7-unit) discrete-nudge actions modestly improved the *passive* latent's
identifiability/canonicality (localized to action-knowledge) but did **not** make it editable. The reframe (from
discussion with Sevan): the real target is **object individuation** — is there, in the *passive* latent, a
**separable, localizable, manipulable handle that IS "object k"**, such that you can grab it and move it, like a
handle in the simulator? Editability was only ever a *probe* for this. The non-trivial question is not "can the
model do the edit it was trained on" (a trained button) but **does training on an interaction affordance reorganize
the passive latent into an object handle that generalizes to interventions it was never trained on** (a real object).

Framing note: think of the GRU+latent **as the world** ("the latent world"), and ask whether that world contains
objects as first-class entities — measured by what you can do to it that you did not put in (structural-realist /
pragmatic stance: "realism" = the structure supports untrained interventions + persistence).

## The independent variable: the ACTION SPACE (its type reshapes latent structure)
All actions are applied to **one object per event**, **sparse** (~15% of transitions carry an action; the rest are
a genuine **no-op**), on top of dataset-4 dynamics (noisy `obs_noise_std=0.2`), and are **frustum/collision-guarded**
(rejected → no-op that frame). Three affordance types (one model each), spanning the axes we care about:

- **`dxdy`** — relative displacement `(dx,dy)` per object, **large** (uniform in ±M, M ≈ half the object's coordinate
  range, i.e. much bigger than the old 0.7). Forces object-*tracking* (you must know where obj k is to move it by
  `(dx,dy)`). Recorded action value = the displacement.
- **`teleport`** — absolute placement: object set to a target `(x',y')` sampled in-frustum (reuse the edits-split
  sampler `pim/simulator/edits_dataset._sample_in_frustum`). Saturates the target/content space; maximal, varied
  magnitude; forces **ghost removal** (clear the old location) which is the #1 editor failure. Recorded action value
  = the absolute target `(x',y')`.
- **`axis_x`** — relative displacement restricted to the **x-axis only** (`(dx,0)`, large). This is the **content-
  generalization probe**: train x-only, then test whether the passive latent can be edited along **y** (never trained).

## Confound control (REQUIRED)
Keep the Exp-2 three-model logic so a positive result localizes to *action-knowledge*, not perturbation-diversity.
At minimum: a **perturbed-passive control for `teleport`** — trained on the **exact same** teleport-perturbed
trajectories but with the **action channel withheld** (sees only the perturbed obs). Baseline = the existing passive
`runs/gru/7_dset4_gru_400epochs` (no perturbations, no action channel). Comparisons: baseline → perturbed-passive =
perturbation-diversity; perturbed-passive → action-conditioned = **action-knowledge** (the headline gap).

## Substrate to build (NEW modules — do NOT edit existing ones or break imports; do NOT touch pyproject)
- **Continuous-action sim** (new module, e.g. `pim/simulator/actions_continuous.py`; you may import helpers from the
  existing `actions.py`/`edits_dataset.py`). One generator with a `mode ∈ {"dxdy","teleport","axis_x"}` that applies
  the affordance as a **persistent per-object position change** from the event frame, re-renders, and writes an
  `actions` field of shape `(N, T, n_obj, 3)` = per object `[active, a1, a2]` (active∈{0,1}; a1,a2 = the recorded
  action value per the type above, **normalized** to ≈[-1,1] by the frustum extent so model inputs are O(1)). no-op =
  active=0 (a1,a2=0). Match dataset 4: obs_res 128, T 40, 2 objects, fixed reflectivities, always-in-frustum base,
  `obs_noise_std=0.2` (NOISY — verify the generated obs are noisy, not clean).
- **Continuous-action GRU** (new module `pim/world_models/action_gru_continuous.py`, mirror `action_gru.py`): encoder
  ingests `obs (128) ⊕ proj(action)` where `proj` is a small `Linear(n_obj*3 → 16)+ReLU` on the flattened action
  vector. **Conform to the `HiddenStateModel` protocol with action defaulting to no-op (zeros)** so the ENTIRE
  master/Exp-2 eval + editor suite runs UNCHANGED on the passive (no-op) model — no `isinstance` branches. `forward`,
  `step`, `observe_sequence`, `predict_step`, `decode`, `flat_state`, `state_from_flat`, `get_hidden_states`,
  `hidden_size` all take an optional trailing `actions`/`action` arg defaulting to no-op.

## Models (GRU, 256 hidden, ~400 ep, matched to baseline; ~7-9 min each — DECOUPLE training per WORKER.md)
Train each via a **standalone foreground script call** (checkpoint written), NOT inside the notebook. New checkpoints
in gitignored `runs/gru/…`. **CORE:** `M_dxdy`, `M_teleport`, `M_axis`, `M_teleport_ctrl` (perturbed-passive on the
teleport data). Baseline = `7_dset4_gru_400epochs` (reuse). **STRETCH (only if time):** perturbed-passive controls for
`dxdy`/`axis`; a "hold/persist" manipulation-type reach test; RSSM.

## Eval — ALL on the PASSIVE latent (action channel OFF = no-op). Reuse Exp-2 `action_conditioned_structure.ipynb`
## + master §4 machinery (read them via Grep/Read; do NOT edit them). Headline readouts:

1. **Object-handle selectivity (the PRIMARY new readout).** For each model, on the passive state at the edit frame,
   apply the master editors targeting **object k's** position, roll out, and score the handle (obs-space, per master §4):
   - **reach** — obs-change toward object-k's target (as % of the true-state swap);
   - **collateral** — obs-change on the OTHER object's rays (a clean handle leaves it ~0);
   - **ghost** — residual intensity at object-k's vacated rays (clean = ghost→0);
   - **persistence** — does it hold over `K=15`.
   A clean, selective, ghost-free, persistent handle = "the object is individuated." Report a per-model **handle
   scorecard** (reach / collateral / ghost) and compare across affordance types + the confound triad.

2. **Interface generalization (the core non-triviality).** The editors are a **different write-mechanism** than the
   action channel the model was trained on (they bypass the transition dynamics and write the state directly). So the
   §4 editing head-to-head **on the passive latent** asks: did training the affordance move it *into the state*
   (grabbable by any mechanism) or leave it *in the input→dynamics pathway* (a button)? This is the master §4 suite
   run per model — reuse it verbatim.

3. **Content generalization (`M_axis`).** Construct **y-only** and **x-only** edit targets in-notebook (move one
   object purely along each axis from its edit-frame position). For `M_axis` (trained on x-only), compare passive-latent
   edit success on **y** (never trained) vs **x** (trained), and against `M_dxdy` (trained on both). y-editability ≈
   x-editability ⇒ the handle generalizes across content ⇒ real object; y ≪ x ⇒ memorized affordance.

4. **Structure context (lighter):** master §1 geometry (intrinsic dim, hull), §2 recoverability (pos/vel linear+MLP),
   §3 fiber-collapse — per model, to see whether/how each affordance reorganizes the passive latent.

5. **Exposition (so Sevan can see the affordances).** For each type: render a short demo trajectory with its actions
   and show the observation-space effect + a change-the-action sanity check (flip the action → rollout diverges), and
   one 2D world GIF. Mirror the Exp-2 E1–E3 exposition. Confirm the new actions are **perceptually large** (report
   mean |Δobs| per action type — should be much bigger than Exp-2's ~0.03–0.13).

## Metric definitions (put in the notebook's definitions table)
Selectivity, reach (%-of-swap), collateral, ghost-ray ratio, persistence — same formulas/units as master §4 (RMSE
throughout, obs intensity [0,1]). Content-generalization = ratio of y-edit reach to x-edit reach for `M_axis`.
Action-knowledge gap = (perturbed-passive → action-conditioned) delta on the handle scorecard.

## Deliverables
- New sim + model modules; datasets in `datasets/` (gitignored ok); checkpoints in `runs/gru/`.
- Executed `notebooks/experiments/editability/actions/action_space_object_individuation.ipynb` (0 error cells; run the analysis
  synchronously — training decoupled into foreground script calls). Produce BOTH rich plots (obs-space waterfalls +
  the handle scorecard bars, all models side by side, GT/reference columns) AND printed metric tables + a definitions
  table. PNGs → `/tmp/action_space_object/`.
- Dated note `research/scratch/2026-07-17-action-space-object-individuation.md`: does a stronger/different action
  space produce a **selective, generalizable object handle** in the passive latent? Is teleport necessary, or does
  dxdy/axis suffice? Does content generalize (`M_axis` y vs x)? Localize to action-knowledge (confound triad). Be
  calibrated; a **clean negative** (no affordance individuates the object) is a strong, publishable result (motivates
  explicit scaffolding — RESEARCH.md's endgame) — say so plainly. Flag for promotion. Do NOT edit `findings/`,
  `RESEARCH.md`, the master notebook, or the Exp-2 notebook.

## Bootstrap
Baseline `runs/gru/7_dset4_gru_400epochs`; data `datasets/4_fixed_refl_inview` (its `sim` config = the dataset-4
`SimConfig`; frustum x_near 1.5 / x_far 6.0 / y_near 3.0 / y_far 12.0, radius 0.5). Mirror
`notebooks/experiments/editability/actions/action_conditioned_structure.ipynb` (Exp-2) for the passive-eval pipeline, the 3-model
confound logic, the editor line-up, and the E1–E3 exposition; mirror `00_master_editability.ipynb` §4 for the editing
metrics. Reuse `pim/simulator/edits_dataset._sample_in_frustum` for teleport targets. Paths 3-deep (`../../..`).
Two known metric caveats to inherit correctly: evaluate against the **time-evolving clean GT** (not a frozen target
render); do not compare the un-normalized tangent-rotation curvature across notebooks.
