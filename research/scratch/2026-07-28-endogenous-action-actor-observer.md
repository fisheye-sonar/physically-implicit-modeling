# Endogenous actions: does ACTING (not just observing) reshape the latent? — GRU actor vs observer

**Date:** 2026-07-28 · **Direction:** `endogenous-action-interactive-world` (`[reframe]`, sub-Q 2+3, strong-enactivist) ·
**Status:** → **FLAG FOR PROMOTION** (nuanced: identifiability + steerability YES, localized to *goal-directed agency*;
but a clean grabbable **object handle** NO — "readable ≠ grabbable" holds under endogenous action) · **Author:**
orchestrator, autonomous overnight run + §4 grabbability follow-up (same night).

## The question
`object-individuation` showed **exogenous** actions (given as an input channel) do NOT put a grabbable object handle in
the passive latent ("readable ≠ grabbable"), and explicitly left **endogenous** action open. This tests it: does a model
that **generates its own actions and acts on the world it must then predict** (closed sensorimotor loop) develop a
different latent than an identical model that merely **observes** the same actions?

- **ACTOR** — GRU predictor + a policy head (per-object categorical over {−1,0,+1}, matches the keyboard) + a value head,
  all read off the same hidden state. It *acts* on the world; at L3 the REINFORCE policy gradient flows into the
  **shared GRU trunk** (the mechanism under test). Encoder is **obs-only** — the action only conditions the *decoder*
  for the immediate prediction (no action-input channel; efference carried by the recurrence).
- **OBSERVER** — the *same architecture*, *fed the actor's actions*, trained on prediction only; never acts. (≈ the
  exogenous action-conditioned model from `object-individuation`, now fed on-policy goal-directed actions.)
- **Three levels:** L1 shift dynamics (guarded, no death; prediction-only), L2 force/momentum (prediction-only), **L3
  force + REINFORCE** on a survival goal (avoid object–object and wall collisions; death → SMiRL-style noise+rebirth).

## Setup / provenance
- **Substrate (new, reuses `pim/simulator`):** `pim/simulator/interactive.py` (`InteractiveWorld` — stateful step-able
  world, `shift`/`force` dynamics, wall-death + collision-death + rebirth, momentum `init_speed=0.28`); emulator
  `scripts/play.py` (human-playable; Sevan playtested + signed off). Model `pim/world_models/actor_gru.py`
  (`EndogenousActorGRU`). Training `scripts/train_endogenous.py` (batched on-policy rollout in the world; predictor loss
  for actor + observer; REINFORCE + value baseline for the actor at L3, into the shared trunk). Eval
  `scripts/eval_endogenous.py`.
- **Runs (GRU 256h, 2 obj, obs_res 128, batch 64 × rollout 48):** `runs/endogenous/{L1,L2,L3,L3b}` — L1/L2 2500 iters,
  L3/L3b 6000 iters; L3b = L3 seed 1 (robustness). ~55 min total on GPU.
- **Eval:** actor drives the world (deterministic policy) → clean trace (mask death/dying/rebirth frames); teacher-force
  BOTH models on the same obs → passive latents; **held-out (70/30)** linear + MLP probes. Notebook
  `notebooks/experiments/editability/actions/endogenous_actor_observer.ipynb` (0 error cells, 4 figs).

## Headline
**Endogenous, GOAL-DIRECTED action makes the physical world-state markedly more *linearly recoverable* from the passive
latent — and this is localized to the goal (RL), not to self-generating actions or to force dynamics.** The actor at L3
reads position/velocity off its latent far better than the observer that saw the identical obs+action stream; at L1/L2
(no goal) actor and observer are indistinguishable. The gain is **identifiability/legibility**, not raw prediction or
canonicality: the L3 actor is a slightly *worse* next-step predictor and *less* canonical (its latent carries extra
control state), while making the (pos, vel) part much more linearly accessible.

## Results (actor / observer, held-out; Δ = actor − observer)

**L3 goal is learned** (Fig 1): survival (frames per life) 12 → ~1536–3072 (deaths ~250 → 0–2), reward −0.03 → +0.10
(the per-step max), plateau by ~it 1700; **replicates on both seeds**.

**§2 recoverability — the positive (Fig 2, 4a):**
| metric | L1 shift | L2 force | **L3 force+RL** | **L3b (seed 1)** |
|---|---|---|---|---|
| pos R² linear | 0.794 / 0.794 (Δ **+0.00**) | 0.504 / 0.516 (Δ −0.01) | **0.762 / 0.589 (Δ +0.173)** | **0.766 / 0.632 (Δ +0.135)** |
| vel R² linear | 0.185 / 0.184 (Δ +0.00) | 0.229 / 0.245 (Δ −0.02) | **0.562 / 0.394 (Δ +0.168)** | **0.553 / 0.388 (Δ +0.165)** |

→ At L3 the actor's latent is **+0.17 more linearly recoverable** for both position and velocity, both seeds. **Velocity**
— historically the hard-to-read, nonlinear/temporal coordinate — shows the largest relative gain (0.39 → 0.56),
consistent with collision-avoidance *requiring* the agent to track where objects are heading. L1/L2 are clean nulls.

**§3 canonicality — actor is LESS canonical (Fig 3, 4a):** fiber residual MLP L3 actor **0.402** vs observer **0.341**
(Δ +0.060; L3b +0.071). Higher = more of `h` is *not* a function of (pos, vel). Coherent with §2: the actor makes
(pos, vel) more *accessible* while adding non-(pos, vel) **control-relevant** state (policy/value structure).

**Prediction — actor is slightly WORSE (Fig 4b):** next-step RMSE L3 actor 0.131 vs observer 0.109 (L1/L2 ≈ tied). The
actor trades a little predictive accuracy for control-relevant structure — **readable ≠ better-predictor**.

**§4 grabbability — is the (much more readable) L3 latent an EDITABLE object handle? Mostly NO** (`eval_editability_endogenous.py`;
passive latent, foreign latent-surgery editors targeting object-0's teleport target; N=64; ghost 1 = object never left,
0 = fully gone). Object-handle scorecard (L3 / L3b), structural editors vs the true-swap (soft ref) + decoder-gradient
(off-manifold oracle):
| editor | reach %swap | collat %swap | **ghost ↓** | select |
|---|---|---|---|---|
| Readout injection — actor | 4 / 3 | 5 / 3 | **1.00 / 1.00** | 0.45 / 0.50 |
| Readout injection — observer | 2 / 2 | 2 / 1 | 0.99 / 1.00 | 0.45 / 0.55 |
| MLP-probe gradient — **actor** | **75 / 83** | 102 / 103 | **0.91 / 1.12** | 0.42 / 0.45 |
| MLP-probe gradient — observer | 35 / 45 | 41 / 46 | 1.06 / 1.16 | 0.46 / 0.49 |
| Decoder gradient (oracle) — actor | 89 / 90 | 50 / 48 | **0.02 / 0.01** | 0.64 / 0.65 |
| True-swap (soft ref) — actor | 100 | 57 / 52 | **−0.05 / −0.04** | 0.64 / 0.66 |

→ **The actor's latent is more *steerable* but not *grabbable*.** The genuine structural editor (MLP-probe gradient)
reaches **2× further on the actor than the observer** (75–83% vs 35–45%) — the readability gain *does* translate into
obs-space reach. **But the two hallmarks of a real object handle fail for both:** the **ghost never clears**
(0.91–1.16, vs oracle 0.01–0.02 and true-swap ≈0 — the object does **not leave** its old location) and edits are
**non-selective** (collateral ~100%, selectivity ~0.45 — the other object is dragged along). Waterfalls
(`runs/endogenous/edit_figs/`) confirm: only true-swap + the oracle move object-0 to the green target *and* clear the red
ghost; MLP-probe gradient **paints a copy at the target while keeping the ghost** (high reach = smearing, not moving);
readout injection is inert. So the reach gain is intensity-toward-target, not object-motion.

## Reading (interpretation — calibrated)
- **Agency with a purpose is the active ingredient.** It is not self-generating actions (L1/L2 actors do that, with no
  effect) nor physical momentum (L2) — it is **acting toward a goal**, i.e. the policy gradient from the survival task
  reshaping the *shared* trunk, that makes the world-state legible. This sharpens the naive "actions help" into
  "**goal-directed** action helps."
- **The observer is a strong control.** Fed the *same* goal-directed obs+action stream, but not generating/owning it and
  not shaped by the goal, its latent stays much less legible. So it is not the information (obs+actions) but the
  **agency** (being the one shaped by acting toward the goal) that does it — the enactivist prediction.
- **Identifiability improves strongly, grabbability does not** — the sharpened headline. Endogenous goal-directed action
  gives a large **readability** gain and roughly doubles structural-editor **reach**, but the passive latent is still
  **not a clean, selective, ghost-clearing object handle** (§4): the object won't *leave* its old location and the edit
  drags the other object, exactly as in `object-individuation`. So agency moves us along the *legibility/steerability*
  axis without delivering the *manipulable-object* affordance — "readable ≠ grabbable" survives even endogenous action,
  now with a graded twist (agency buys readability + reach, not the handle). This is the "you-can't-lose" negative that
  keeps pointing at **explicit object scaffolding** (RESEARCH.md endgame): the interface (obs + self-generated goal-directed
  action) is jointly insufficient to put a grabbable object into the implicit state.

## Caveats / threats to validity
- **§4 grabbability DONE (this night); action-interface controllability (edit *through* the trained action channel) still
  owed** — the latent-surgery §4 negative is in; whether the object is controllable via the action channel it was trained
  on (vs a foreign write) is the remaining editability angle.
- **Editor line-up is 2 structural editors + oracle + swap** (readout injection, MLP-probe gradient). A stronger
  on-manifold editor (PCA geodesic, the object-individuation canonical) was not run here; but MLP-probe gradient already
  moves the obs a lot (reach 75–83%) and still fails ghost/selectivity, so the negative is not an editor-weakness artifact.
- **Goal vs generic auxiliary task.** The actor gets an extra training signal (the goal) the observer lacks; the L1/L2
  nulls show it is not action-generation per se, but whether *any* auxiliary objective (vs specifically *acting*) would
  also help is untested (a non-action auxiliary-task control is the clean disambiguator).
- **GRU only; god's-hand (not embodied); 2 objects; single trace per level for probing** (held-out split, N≈2–5k
  frames). Two seeds at L3 agree, but L1/L2 are single-seed. In-world on-policy eval (mild train/eval policy shift).
- Fiber-residual "less canonical" is a fraction-of-‖h‖ measure; read as "carries more non-(pos,vel) content," not a
  degradation.

---

# REVISION (2026-07-29) — stronger predictors: editability verdict CONFIRMED, identifiability headline DOWNGRADED

Sevan reviewed the first pass and pushed back on two things: a **waterfall bug**, and *"the predictions are so messy
it's hard to tell whether it's really failing, or just a bad predictor."* Both were right. This section supersedes the
identifiability magnitudes above and strengthens the §4 verdict.

## What changed methodologically
- **Waterfall bug FIXED.** v1 injected the TRUE target-obs row into *every* column and dropped each editor's own
  step-0 decode, so every column looked teacher-forced past the edit **and the exact frame the scorecard scores was
  hidden**. v2: each column shows **its own free-run from step 0**; GT is its own column. (Only True-swap ever sees the
  target obs, by construction.)
- **Quality gate added** (open-loop free-run RMSE under the TRUE actions + **sharpness = rollout TV ÷ GT TV**). The
  original model was genuinely weak: free-run RMSE **0.26**, sharpness **0.607**.
- **Stronger models trained:** hidden **512**, 2-layer MLP encoder + residual MLP decoder, **5-step free-run
  (multistep) objective**, **25k** iters, 2 seeds (`L3s0`,`L3s1`) + a strong no-goal control (`L2s0`). Capacity flags
  default to the old architecture, so old checkpoints still load and weak-vs-strong is apples-to-apples.
- **Editor line-up widened** to Readout injection / Global-PCA projection / PCA geodesic / MLP-probe gradient, plus the
  True-swap and decoder-gradient-oracle brackets. **Rollout modes** `self` (model's own policy acts on its imagined
  world — in-distribution) and `noop`; results near-identical, so off-policy rollout was NOT the driver.

## §4 grabbability — CONFIRMED, and now it is not a predictor artifact
| editor (actor, mode `self`) | weak L3 | strong L3s0 | strong L3s1 |
|---|---|---|---|
| Readout injection | ghost 0.998 / reach 4.1% | **1.000** / 0.3% | **1.001** / 0.5% |
| Global-PCA projection | 0.990 / 13.6% | **1.001** / 4.1% | **1.005** / 3.7% |
| PCA geodesic | 0.989 / 28.1% | **1.004** / 3.7% | **1.010** / 3.8% |
| MLP-probe gradient | 0.925 / 78.0% | 0.662 / 50.0% | 1.169 / 53.5% |
| **Decoder gradient (ORACLE)** | **0.010** / 91.1% | **0.012** / 93.4% | **0.004** / 89.2% |
| **True-state swap (reference)** | **−0.027** / 100% | **−0.002** / 100% | **0.028** / 100% |

**The decisive control:** on the *same model, same decoder, same rollout regime*, the oracle and the true-swap succeed
completely (ghost ≈ 0 — the object genuinely leaves and reappears at the target) while every structural editor is inert
(ghost ≈ **1.000**). **If blur/weak prediction caused the failure, the oracle would fail too.** So a state rendering the
target exists and the model can roll it out; probe-directed writes cannot reach it. The failure localizes to the
**edit map's reachability**, not the predictor — this is the control the first pass lacked. Replicated across 2 strong
seeds × 2 rollout modes. Notably the structural editors became **more** inert as the predictor improved (PCA geodesic
reach 28% → 4%): a better predictor is *less* structurally editable, not more.

## Identifiability — DOWNGRADED (this supersedes the magnitudes in the original Results section)
| Δ (actor − observer) | weak L3 (256h, 6k it) | **strong L3 (512h+MLP, 25k it)** | strong L2 — no goal (control) |
|---|---|---|---|
| position R² linear | **+0.155** (0.173 / 0.135) | **+0.017** (0.030 / 0.005) | +0.018 |
| velocity R² linear | **+0.165** | **+0.052** (0.044 / 0.060) | −0.015 |
| fiber residual MLP (↓ = canonical) | +0.066 (actor *worse*) | **−0.074** (actor *better*) | −0.026 |

- **The position advantage disappears** — at strength it is **+0.017, identical to the no-goal control (+0.018)**, i.e.
  no longer goal-specific at all. The observer simply catches up (0.589 → 0.863) given capacity and training.
- **Velocity survives** (+0.052 vs control −0.015) — ~3× smaller than the weak-model estimate but goal-specific.
- **Canonicality FLIPS SIGN and becomes a cleaner positive:** the strong actor is consistently *more* canonical
  (−0.070 / −0.084 across seeds; control −0.026).
- **Revised reading:** goal-directed agency mainly **accelerates** the emergence of linearly-readable structure rather
  than producing a large, durable representational advantage; what durably survives at convergence is a **modest gain
  in velocity readability and canonicality**. The 2026-07-28 headline ("+0.17 on both position and velocity") was an
  artifact of comparing two under-trained models and should not be cited.

## Honest limitations found in this pass
- **The stronger models did NOT fix the blur.** Sharpness only 0.607 → 0.633 TV ratio; free-run RMSE slightly *worse*
  (0.26 → 0.28). Capacity + a multistep objective were insufficient; rollout blur remains a real property of this
  architecture. (The §4 verdict no longer depends on it, thanks to the oracle control.)
- **The action-channel control is NOT a clean "button" result.** A PD controller in the **real simulator** closes
  **93–95%** of the distance to the target (the channel has real authority), but the model's *imagination* of those
  same actions barely moves the object (reach 2–6%, ghost ≈ 0.98). Those actions are **off-policy**, so this conflates
  "the affordance doesn't transfer to the state" with "off-policy action generalization is poor." **Corrected framing:**
  the model's imagined world supports **no** tested intervention route — latent surgery *or* action channel — except
  direct decoder optimization (oracle) and fresh observational evidence (true-swap). It behaves as an **on-policy
  predictor, not an intervention-supporting simulator.** The earlier "button, not a handle" phrasing overclaimed and is
  retracted pending an on-policy action-intervention test.
- Unchanged caveats: GRU only, god's-hand, 2 objects, N=64 edits, 2 seeds at L3.

**Artifacts:** notebook `notebooks/experiments/editability/actions/endogenous_grabbability.ipynb` (0 err, 7 figs);
`runs/endogenous/{L3s0,L3s1,L2s0}`, `editability_metrics_v2.json`, `eval_metrics.json`, `edit_figs_v2/`.

## Pointers
Substrate: `pim/simulator/interactive.py`, `pim/world_models/actor_gru.py`, `scripts/{train,eval,eval_editability}_endogenous.py`,
`scripts/play.py`. Runs `runs/endogenous/{L1,L2,L3,L3b}` (gitignored) + `eval_metrics.json` + `editability_metrics.json` +
`edit_figs/`. Notebook `notebooks/experiments/editability/actions/endogenous_actor_observer.ipynb`. Direction
`research/directions/endogenous-action-interactive-world.md`. Builds on + sharpens `findings/object-individuation.md`
(exogenous → endogenous, same "readable ≠ grabbable" structural conclusion) + `findings/editability.md`. Next:
action-interface controllability (edit through the trained action channel); a non-action auxiliary-task control (goal vs
generic aux); embodied variant; RSSM; then the constructive move (explicit object scaffolding).
