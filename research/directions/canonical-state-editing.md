# Direction: Canonical-State Editing — (pos,vel) fiber collapse + obs-driven probe

**Tag:** `[reframe]` · **Sub-question:** 2 (identifiability) + 3 (editability) ·
**Status:** in progress (2026-06-24) · **Complexity:** medium-high (one notebook, several sections)

> **Run THIS as one notebook:** `notebooks/experiments/editability/canonical_state_editing.ipynb`.
> Number every code cell (`# [N]`) and every figure (`Fig K — …`, panels `(a)/(b)/(c)`), per the
> CLAUDE.md convention. Produce BOTH rich plots (Sevan judges from these) AND printed metric
> tables (the agent self-verifies from these). Export key PNGs to `/tmp/canonical_state/`.

## The premise (verified against the simulator)

The sim dynamics are **constant-velocity**: `pos_{t+1} = pos_t + vel·dt`, with
`direction_noise_std=0`, `speed_noise_std=0` (velocity never changes), small
`position_noise_std=0.04`, `obs_noise_std=0.2`, `always_in_frustum=True`, fixed reflectivities,
fixed radius. Therefore the **minimal sufficient statistic of the world is `(positions,
velocities)`** — 8-dim for 2 objects (`{x,y,vx,vy}` × 2). From `(pos,vel)` you can render the
optimal (MMSE/expected) rollout; nothing else is needed (identity = fixed reflectivity; no
appearance dynamics). Note the corroboration: `findings/state-geometry.md` put the variance
**elbow at ~5–10 dims**, i.e. right at the physical DOF (~8) — the GRU has compressed to roughly
the right dimensionality, so the editing failure is about *non-canonical coordinates*, not junk.

**Hypothesis under test.** Editing fails because the GRU hidden state `h` is **non-canonical**:
it is predictively sufficient but (a) may carry *history beyond `(pos,vel)`* (uncollapsed decode
fiber), and/or (b) is a *nonlinear embedding* of `(pos,vel)` so linear edits leave the manifold.
The "readable" code (what a probe extracts) is not the "controllable" code (what the recurrence
uses to render). **Sharpening fact:** the edits dataset *preserves the original velocity* of the
teleported object, and our min-norm position edit ≈ preserves velocity too — so the ghost is
**probably NOT a velocity-incompleteness artifact.** Prediction: if editing the *complete*
`(pos,vel)` target still ghosts, the culprit is non-canonicality / nonlinear embedding, not
incompleteness. Either outcome is a finding — do not soften whichever way it lands.

## Bootstrap (cold-start — run from a fresh kernel)

Mirror `notebooks/experiments/editability/geodesic_walk_k150.ipynb` and
`pca_component_position.ipynb` (both are working references). The notebook lives **3 levels deep**
(`notebooks/experiments/editability/`), so relative paths are:
`sys.path.insert(0,"../../..")` (repo root, for `import pim`) and `"../.."` (for `helpers`);
`CHECKPOINT_PATH="../../../runs/gru/3_dset3_gru_persistentids_inview_400epochs/best_model.pt"`,
`DATA_DIR="../../../datasets/4_fixed_refl_inview"`. Bootstrap steps:
1. `load_checkpoint`, `load_dataset(DATA_DIR, n_obj_keep=2)` → `test`, `edits`; `make_test_loader`.
2. `eval.teacher_force(model, test_loader)` → `preds_tf`, `states_tf` (10000×39×256).
3. Linear position probe via `LinearExtractor` (as in the reference notebooks).
4. `fit_state_subspace(states_tf, var_threshold=0.90)` → `subspace` (+ on-device copy); the local
   tangent bank (`bank_dev`) and `_local_resid` helper as in `geodesic_walk_k150.ipynb`.
5. `eval.warm_up_to_edit(model, edits.obs[:N], edits.edit_frame, n_viz=N)` → `warm`, `h_at_edit`.
6. `rollout_from_flat`, `decode_pos` helpers (copy from the reference notebooks).
7. **Velocities.** `_load_h5_dataset` does NOT load velocities into `test`/`edits`. Read them
   directly: `import h5py; v = h5py.File(test.h5_path)["velocities"][:, :, :2, :]` (and likewise
   the edits split's h5 for `edits` velocities), aligned the same way positions are
   (`test.positions[:, :-1, :2, :]`). Fallback if loading is awkward: finite-difference
   `vel_t = (pos_{t+1}-pos_t)/dt` with `dt=1.0` (noisier due to position noise — prefer stored).

## Section A — `(pos,vel)` probe + recoverability  [sub-Q2]

Train a probe `h → (pos, vel)` (8-dim target = stacked positions and velocities for both objects).
- Start **linear** (`LinearExtractor` with an 8-dim `StateDefinition`). Report per-component RMSE
  and R² (pos-x, pos-y, vel-x, vel-y for each object), vs a sensible baseline (predict-the-mean).
- **If velocity is not linearly recoverable** (R² low / RMSE near the predict-mean baseline),
  switch to an **MLP probe** and report the same. **If the MLP also can't decode velocity, do NOT
  give up** — investigate: is velocity recoverable from a *temporal* feature (e.g. `h_t − h_{t-1}`,
  or a 2-frame window), or from the predicted next observation? A finding either way: "velocity is
  linearly readable", "only nonlinearly readable", or "only readable from temporal differences".

## Section B — Fiber-collapse metric (the keystone)  [sub-Q2]

Quantify *how much of `h` is NOT a function of `(pos,vel)`* — i.e. whether the decode fiber is
collapsed (canonical) or carries extra history.
- Regress `h ≈ g(pos,vel)` over all real teacher-forced states. Fit **linear** then **MLP** `g`.
  Report the **residual fraction** `‖h − g(pos,vel)‖ / ‖h‖` (and explained-variance R² on `h`).
  Decision: residual ≈ 0 ⇒ `h` is determined by `(pos,vel)` (canonical, fiber collapsed); large
  residual ⇒ `h` carries information beyond the sufficient statistic (uncollapsed / history).
- Complementary: regress `h ≈ g(pos)` (positions only) and compare residuals — the **incremental
  variance explained by adding velocity** tells you whether velocity is a major axis of `h`.
- The **linear-vs-MLP residual gap** measures how *nonlinear* the `(pos,vel)→h` embedding is
  (large gap ⇒ curved embedding ⇒ linear edits will leave the manifold — connects to geometry).

## Section C — Joint `(pos,vel)` editing  [sub-Q3]

Edit the **complete** sufficient statistic and see if the observation finally moves cleanly.
- Target = post-edit `(pos, original_vel)` at `edit_frame` (positions teleported per `edits`;
  velocities = the edits' preserved original velocities). Use `inject_state` against the 8-dim
  `(pos,vel)` probe (min-norm edit onto the joint readout). Also run the global-manifold variant.
- Compare head-to-head with the **position-only** edit (the existing baseline).
- Measure in **observation space** exactly as `geodesic_walk_k150.ipynb` does: roll out, compute
  `->target render` distance, `obs change` vs unsteered, and **ghost ratio**; plus generated **1D
  scans + waterfalls** (green = target loc, red = ghost loc). Headline question: **does completing
  the target to `(pos,vel)` fix the ghost / move the object in obs space, or not?**

## Section D — Observation-driven editing as a structure probe  [sub-Q3]

Drive `h` so the model's **generated** observation matches a target, then study *where it lands*.
This is an oracle probe of structure (it uses the GT obs as a target to *find* a latent), NOT a
deployable editor — say so. Two targets:
- **single GT edit frame** (under-determines velocity — the obs fiber is velocity-free), and
- **a short GT sequence** (a few post-edit frames, which pins velocity).
Optimizer: gradient descent on `h` (Adam) minimizing `‖decode(state_from_flat(h)) − target_obs‖²`
for the single frame, and a differentiable k-step rollout (`predict_step`) matched to the sequence.
At the endpoint `h*` report:
- (a) **`(pos,vel)` probe readout** — does it match the GT target `(pos,vel)`? (Does obs-reaching
  also reach the readout?)
- (b) **off-manifold residual** of `h*` (global + local) — is the obs-reaching state on the visited
  manifold, or off it?
- (c) **distance to the real-trajectory state** that genuinely produces that observation (the
  canonical state) — does the model reach the obs via the *same* state reality uses, or an alien one?
- (d) **roll out from `h*`** — does the edit **stick** (obs stays at target, ghost gone) or revert?
  Single-frame vs sequence target: does pinning velocity make it stick?
- **Decisive contrast:** the probe-objective (Sections C) endpoint vs the obs-objective endpoint —
  if obs-objective moves the observation and probe-objective doesn't, that *is* the "readable ≠
  controllable" result, localized.

## Deliverables (HARD REQUIREMENTS)

- Executed notebook (GPU, numbered cells/figures), plots + printed tables for every section.
- PNGs to `/tmp/canonical_state/`.
- A dated `research/scratch/2026-06-24-canonical-state-editing.md` note: per-section results, the
  verdict on the hypothesis (canonical? linear-recoverable velocity? does completing the target fix
  editing? does obs-driven land on/off manifold and stick?), caveats, open questions — flagged
  `→ FLAG FOR PROMOTION`. **Do NOT** promote to `findings/`, mark this direction done, or edit
  `RESEARCH.md`.
- A tight structured report back to the orchestrator: headline, key numbers per section, PNG paths.

## Context

- Checkpoint `runs/gru/3_dset3_gru_persistentids_inview_400epochs/best_model.pt`,
  data `datasets/4_fixed_refl_inview` (2 objects, fixed reflectivities, constant velocity).
- Primitives: `pim/extractors` (`LinearExtractor`, `StateDefinition`), `pim/editors`
  (`inject_state`, `probe_decomposition`, `fit_state_subspace`, `manifold_steer`,
  `fit_local_subspace`, `offmanifold_residual`), `pim/eval.controllability._rollout`.
  Obs-space metrics + waterfalls: copy from `geodesic_walk_k150.ipynb` Sections 5–5b.
- Model implements the `HiddenStateModel` protocol (`flat_state`/`state_from_flat`/`decode`/
  `observe_sequence`/`predict_step`) — use it for the differentiable obs-driven optimization.
