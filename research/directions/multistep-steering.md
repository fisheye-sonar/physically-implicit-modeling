# Direction: Multi-step / freeze-time editing — does spreading the edit over frames help?

**Tag:** `[in-frame]` · **Sub-question:** 3 (editability) · **Status:** proposed ·
**Complexity:** low-medium (one notebook, GRU primary + smaller RSSM section; NO retraining)

> A **standalone experiment in its own new notebook** —
> `notebooks/experiments/editability/multistep_steering.ipynb`. **Do NOT touch the master
> notebook `00_master_editability.ipynb`.** Uses the existing trained checkpoints — no training.
> Scope is deliberately narrow: the deliverable is **editability success/failure**, NOT the full
> master metric spread.

## Why this exists
The master result is *readable ≠ controllable*: a single big latent push to the readout target
either reverts (off-manifold, dynamics reject it) or moves the obs by scrambling (ghosts). The
hypothesis here is that **the one-shot latent jump is what breaks it** — the model never gets to
*see* what it is editing and re-stabilize. Two ways to give it that chance:

- **(1a) Interleaved latent steering (closed loop).** Instead of one big edit, push `h` a *small*
  step toward the target, then **decode → feed that obs back through `step` → push again**, repeating
  for many small steps. The world model gets to observe and settle the object mid-edit. Because the
  dynamics will also try to drag the *other* (unedited) object, **re-assert the unedited object's
  readout target every step** (a small corrective injection) and see whether the closed loop can hold
  it fixed while relocating the edited one.
- **(1b) Freeze-time teacher forcing.** Teacher-force the model up to the edit frame `ef`, then
  **freeze the world** and interpolate the edit over `N` frames: the unedited object held fixed at its
  `ef` position, the edited object stepped incrementally from its `ef` position toward the target,
  **teacher-forcing the rendered obs the whole time**; then **unfreeze** and roll out. Sweep `N`
  (e.g. {1, 2, 3, 5, 8, 12, 15}); `N=1` is the existing single-frame teleport baseline.

## Construction

**Shared setup (mirror master §4 / `canonical_state_editing.ipynb`):** load GRU
`runs/gru/3_dset3_gru_persistentids_inview_400epochs/best_model.pt` and (for the smaller replication)
RSSM `runs/rssm/4_dset4_refined_best/best_model.pt`; data `datasets/4_fixed_refl_inview` **edits**
split (`edit_frame=ef=20`, T=40, R=128, 2 objects). Use the same `warm_up_to_edit`, `Scene` /
`SimConfig` / `render_scene`, `_rollout`, `tf_hidden_at`, and probe (`LinearExtractor` +
`inject_state`) helpers. Paths 3-deep (`../../..`).

**1a — interleaved steering.** Start from the teacher-forced pre-edit state `h0` at `ef`. Editor loop
for `S` steps (e.g. 20–40):
- push `h ← h + η·Δ` where `Δ` moves the **edited** object's readout toward `target` (reuse an existing
  editor primitive as the per-step move: `pim.editors.probe_steering` readout injection with a small
  step, and/or `pim.editors.manifold_steering` geodesic step — try both);
- **also** push the **unedited** object's readout back toward its held value (small corrective);
- `obs_hat = model.decode(h)`; `_, h = model.step(obs_hat, h)` (observe-and-settle);
- record per-step obs, decoded positions of BOTH objects.
Compare against the **one-shot** version of the same editor (all the push in a single step) and against
Unsteered and GT(sim). Sweep step size `η` / number of steps.

**1b — freeze-time TF.** For each edit sample, build the frozen-interpolation obs sequence in-notebook
via `render_scene` (this is NOT in `edits.h5`): frames `ef .. ef+N` render a Scene with the unedited
object at `positions[ef, other]` and the edited object at `lerp(positions[ef, o] → target)`. Teacher-
force these `N` frames from `h0`, then roll out `K=15` from the resulting state and compare to
`edits.clean_obs[ef:ef+K]` (the sim's true post-edit obs).

## Metrics (definitions table up front; obs-space, mirror master §4 — but only the editability set)
Per method / per `N` / per `η`:
- **obs RMSE vs GT target** over the post-edit rollout (per-step curve + summary).
- **ghost ratio** (energy at the *original* location vs the target location).
- **selectivity / collateral:** decoded-position error of the **unedited** object vs its held value
  (did we disturb the object we were supposed to leave alone?), AND its **velocity** artifact (1b is
  expected to corrupt the unedited object's velocity because "held fixed" reads as zero-velocity — and
  the edited object's interpolation-velocity ≠ its post-edit preserved velocity; measure both).
- **persistence:** does the edit hold over `K` steps or revert/drift.

## Figures (CLAUDE.md legibility: definitions table, demarcated tables, plain language, GT column)
- **Observation-space waterfalls** (dark theme, `world_model_eval` style) with a **GT(sim)** column,
  green target / red ghost markers, a few pre-edit context frames — show the *edited* and *unedited*
  object behaviour. This is where the effect actually lives (per `feedback-visual-analysis`).
- (1a) per-step obs-RMSE-to-target + unedited-object-collateral curves, one-shot vs interleaved.
- (1b) `N`-sweep: obs-RMSE-to-target and ghost vs `N`; the velocity-artifact panel.
- Light academic theme for the metric/summary panels.

## Deliverables
- Executed `notebooks/experiments/editability/multistep_steering.ipynb` — run **synchronously
  in-turn**, 0 error cells. PNGs → `/tmp/multistep_steering/`.
- Dated note `research/scratch/2026-07-16-multistep-steering.md`: does spreading the edit over
  steps/frames beat the one-shot edit? Under what `N`/`η`? What breaks (esp. unedited-object velocity)?
  Mark `→ FLAG FOR PROMOTION` if a clear positive/negative signal. Do NOT touch the master notebook,
  `findings/`, or `RESEARCH.md`.

## Notes
GRU is the primary target; add a **smaller** RSSM section (deterministic `h` is the primary
world-state carrier — examine det and stochastic separately if you touch the RSSM). Do NOT retrain.
