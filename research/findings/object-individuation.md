# Finding: Object individuation via (exogenous) action-conditioning

*Sub-question 3 (editability/objecthood) + 2 (identifiability).* Does training a passive-prediction world
model on actions that causally move objects reorganize its latent into **objects as first-class, grabbable
entities** — a separable, localizable, manipulable handle for "object k" — rather than a smear of
history-entangled state?

> **Scope (preliminary, 2026-07-17).** These claims concern **exogenous** actions only — externally imposed,
> random object-moving perturbations fed as an input channel, on the GRU (with RSSM corroboration on the weaker
> positive from the earlier pass). They are **not** claims about **endogenous** action (actions an agent
> *chooses* from its own state/policy, sensorimotor contingencies it controls), which is untested and is the
> natural next question. Read "actions" below as "exogenous actions." Also: this GRU checkpoint family, dataset 4,
> 2 objects, N=48 edits, in-sample probes.

## Current understanding

> **Updated 2026-08-19.** The action interface's *latent* effect is now measured: it writes the same displacement as the counterfactual oracle (cos +0.872, 29°) while remaining as probe-invisible as every other successful edit, and models trained with cued teleports are **less** willing to believe an uncued one (+0.22) than models trained on the same teleports uncued (+0.53). Previously:
>
> **Updated 2026-08-17.** The negative now holds under **endogenous** action too — an actor that
> generates its own actions and acts on the world it must predict gains identifiability and
> steerability localized to goal-directed agency, but still no grabbable object handle
> (2026-07-28). That closes the explicit gap the 2026-07-17 entry left open. Separately, when a
> model's action space *contains* the intervention, the **action interface** is a strong handle
> (+0.62) over exactly the capacity range where latent editing decays to zero (2026-08-13) —
> confirming the affordance lives in the input→dynamics pathway, not the state.

### Previous synthesis (mutable summary)

**Training on exogenous object-moving actions does NOT individuate objects into a grabbable state handle.** Even
large, ghost-clearing affordances (relative `dxdy`, absolute `teleport`, axis-restricted) leave the *passive*
(no-op) latent un-editable by any tractable structural editor: targeting object k, the **ghost never clears**
(the object does not leave its old location) and edits are **non-selective** (the other object is disturbed
nearly as much). This holds for every affordance type and the full confound triad; the affordance-free
**baseline is as good or better**. The interpretation is a clean structural one: **objecthood lives in the
input→dynamics pathway (a "button" the model integrates), not in the state (a "handle" a foreign write-mechanism
could grab).** The action channel demonstrably *works* (the model executes the trained moves; the perturbed-passive
control proves the channel is used), but that capability does **not migrate into the state** as a manipulable
entity — so it does not generalize across write-mechanisms (latent surgery) the way a real object would.

**The one real, weaker positive — localized to action-knowledge:** exogenous actions do reorganize the passive
latent toward a **more canonical, more linearly-recoverable function of the physical `(pos,vel)` statistic**. This
is *representation legibility*, not *manipulability* — **readable ≠ grabbable**, the same "readable ≠ controllable"
theme as `editability.md`, now sharpened: even action-training only buys readability.

**Why it matters:** this is a "you-can't-lose" negative that motivates the constructive endgame — passive
prediction *plus* exogenous interaction affordances are **jointly insufficient** to put objects into the state;
you cannot induce object structure from the observation/action *interface* alone. It points at **explicit object
scaffolding** in the architecture (RESEARCH.md) — and it deliberately leaves open whether **endogenous** action
would differ.

## Log

### 2026-08-19 — The action interface writes the *same latent displacement* as the oracle, and it still is not a state handle · `observed` ★-candidate

**Evidence:** `scratch/2026-08-19-latent-edit-directions.md` ·
`notebooks/experiments/editability/latent_linearity/` · `datasets/15_teleport_eval_single/eval.h5`, N=256 ·
runs `XG_A_H256`, `XG_C_H256`, `H256`. No models trained.

The 2026-08-13 entry established that the **action interface** is a strong handle (+0.62) where latent editing
is not. This measures *what it does to the state*: the displacement the trained action channel writes is
**+0.872 (29°)** from the displacement the counterfactual-overwrite oracle writes — the tightest pair in that
study, 5.9× the shuffled-pair chance level. The learned input pathway and a full history rewrite arrive at
nearly the same place in latent space.

**This does not turn the action channel into a state handle, and the numbers say why.** The action-induced Δh is
**0.91× chance** in the linear position probe's row space — no more probe-visible than the oracles' — and its
cross-episode cosine is **+0.010**, so there is no generic "teleport object k" direction to write either. The
affordance still lives in the input→dynamics pathway; what is new is that the pathway's *effect on the state* is
the same object the oracles produce, rather than some other route to a similar picture.

**A second result about the input pathway.** Whether a single **uncued** post-edit observation persists is a
fact about the training distribution: step-0 Edit Index **−0.002** (never saw a teleport) → **+0.216**
(`XG_A`, teleports always cued by an action) → **+0.532** (`XG_C`, identical data and recipe with the action
input removed). Being *told* about interventions during training makes the model **less** willing to believe an
unexplained one — the action channel buys steerability and costs credulity.

**Caveats:** GRU only (no teleport-trained RSSM/transformer/DiT exists); one seed; the alignment result is
correlational until the action channel's write is corrupted at fixed read-out accuracy.

---


### 2026-08-13 — The action interface is the handle; the latent is not · `replicated`

**Evidence:** `scratch/2026-08-13-action-hidden-size.md` ·
`notebooks/experiments/editability/action_hidden_size/` (+ `ACTION_SWEEP_RUNS.md`) ·
`scripts/eval_action_sweep.py` · `datasets/7_cont_teleport`, held-out `13_cont_teleport_eval`.

The exogenous-teleport family is conditioned on continuous *teleport-to-absolute-coordinate*
actions, so **its action space contains the intervention under test** — a built-in ground-truth
handle no passive model can offer. Across `H ∈ {8,32,128,256,512}` the action interface rises
**+0.216 → +0.455 → +0.582 → +0.618 → +0.608** over exactly the capacity range where latent
editing decays to nothing (see `editability.md`, same date).

**Why it matters:** it separates two claims that are easy to conflate — "no handle for this
intervention exists" (false) and "no handle exists **in the state**" (true, and it gets *more*
true with capacity). The affordance lives in the input→dynamics pathway, not in the state.

The endogenous family cannot teleport at any capacity — its actions are forces — so it has no
action-interface arm by construction. This is a structural asymmetry, not a missing measurement.

---

### 2026-07-28 — Endogenous action: identifiability and steerability yes, grabbable object handle no · `replicated`

**Evidence:** `scratch/2026-07-28-endogenous-action-actor-observer.md` ·
`directions/endogenous-action-interactive-world.md` · new substrate
`pim/simulator/interactive.py` (`InteractiveWorld`), emulator `scripts/play.py` (human-playable;
Sevan playtested and signed off), model `pim/world_models/actor_gru.py`,
`scripts/train_endogenous.py`, `scripts/eval_endogenous.py` · runs
`runs/endogenous/{L1,L2,L3,L3b}` (GRU 256h, 2 objects, obs_res 128, batch 64 × rollout 48).

The `2026-07-17` negative was measured with **exogenous** actions supplied as an input channel,
and explicitly left **endogenous** action open. This closes it: an **actor** (GRU predictor plus
policy and value heads reading the same hidden state, acting on the world it must then predict,
with the REINFORCE gradient flowing into the **shared trunk** at level 3) versus an **observer**
— identical architecture, fed the actor's actions, prediction-only, never acts.

**Result, nuanced:** identifiability and steerability improve and are **localized to
goal-directed agency** — but a clean **grabbable object handle** does not appear.
**`readable ≠ grabbable` holds under endogenous action.**

**Why it matters:** the strong-enactivist version of the hypothesis — that *acting* rather than
merely observing is what builds a manipulable world state — gets its clean test here, and the
handle still does not materialise in the state.

---

### 2026-07-17 — Clean negative: no exogenous action space individuates a grabbable object handle · `established`
Independent variable = the action space (all applied one-object-at-a-time, sparse ~15%, genuine no-op, on dataset-4
noisy dynamics, frustum/collision-guarded), one continuous-action GRU each: **`dxdy`** (large relative
displacement), **`teleport`** (absolute in-frustum placement — saturates the target space, forces ghost-removal),
**`axis_x`** (x-only restricted — a content-generalization probe). Confound triad: baseline
(`7_dset4_gru_400epochs`, no perturbations/no channel) → perturbed-passive-teleport control (same perturbed data,
channel withheld) → action-conditioned (channel fed). **All eval on the PASSIVE latent (action off)** with the
master §4 editors — a *different* write-mechanism than the trained action channel, so this is an **interface-
generalization** test (does the affordance live in the state or the input pathway?).
- **Actions were genuinely large** (mean per-event |Δobs| **0.19–0.22**, 2–7× the 0.7-unit Exp-2 nudge's 0.03–0.13),
  so the negative is not a dose problem.
- **Object-handle scorecard (best structural editor = PCA geodesic, per model):** ghost ratio **0.90–0.93 for all
  five models** (1.0 = object stays put; a real teleport / true-state-swap = 0.44–0.67; decoder-gradient *oracle* =
  0.09) — **the ghost never clears for anyone**. reach = 21–37 % of a true swap; **selectivity ≈ 0.56–0.58**
  everywhere (moving object k disturbs the other nearly as much). The **baseline had the highest reach (36.7 %)** —
  the affordances do not help the handle at all. Confirmed in obs-space: the structural editors keep the bright band
  on object k's **ghost** line and leave the **target** line empty, while the true-state-swap jumps to the target.
- **Teleport is not necessary and `dxdy`/`axis` do not suffice** — none produce a handle; the teleport model, trained
  explicitly to clear ghosts via absolute placement, does not transfer ghost-clearing to the editor mechanism
  (ghost 0.917, same as the rest).
- **Content generalization moot:** `M_axis` (x-only) is indistinguishable from baseline; the y>x reach asymmetry is a
  lateral-vs-depth *geometry* artifact present in baseline too — there is no x-specific handle to generalize or fail.
- **Weaker positive (action-knowledge):** MLP fiber residual (lower = more canonical) baseline 0.395 → perturbed-passive
  0.488 (perturbation alone *worsens* it) → `M_teleport` 0.316 → `M_axis` 0.282; linear velocity R² rises. The gain
  localizes to the perturbed-passive→action-conditioned (action-knowledge) step. Representation, not editability.
Notebook `notebooks/experiments/editability/actions/action_space_object_individuation.ipynb`; substrate
`pim/simulator/actions_continuous.py`, `pim/world_models/action_gru_continuous.py`; note
`scratch/2026-07-17-action-space-object-individuation.md`. Caveats: GRU only, N=48, in-sample probes, **exogenous
actions only**, single-frame belief inertia softens even the true-state-swap ceiling (ghost 0.44–0.67, not 0).

### 2026-07-16 — (subsumed) Exp-2: small exogenous nudges improve legibility, not editability · `established`
The earlier, weaker version (0.7-unit discrete-token nudges; `scratch/2026-07-16-action-conditioned-structure.md`).
Three GRUs on byte-identical trajectories (baseline / perturbed-passive / action-conditioned). **Action-training
improved the passive latent's identifiability + canonicality, localized to action-knowledge** (position linear R²
0.838→0.890, velocity linear R² 0.582→0.659, MLP fiber residual 0.379→0.324; the 3→2 gap carried it) — **but
editability did not follow** (the master editors failed on the action-trained model as on the baseline). Read then as
"necessary-direction but not sufficient-magnitude." The 2026-07-17 experiment cranked the magnitude ~3–7× and still
found no handle, so the "too-small-dose" reading is retired: the ceiling is structural, not a magnitude problem.
Validity checks confirmed the actions were causally used (change-the-action sanity) and noise-matched (no confound).
