# Project summary — state of play as of 2026-08-02

*Written for a strategy conversation (and PI-meeting prep). Self-contained: assumes no access to the repo.
Sources: `RESEARCH.md`, `research/findings/*`, `research/PROGRESS.md`, the scratch notes cited inline.
Status labels follow the repo's gate: **established** = promoted to `findings/`; **flagged** = written up in
`scratch/`, awaiting promotion; **open** = unresolved.*

---

## 1. The question

**What is the nature of the "world state" that a world model forms when trained primarily to predict
observations?** Can a world state be found at all, where does it live, what structure does it have, and under what
architectures/objectives do the affordances we care about appear?

Four affordances, which double as the evaluation suite:
1. **Predictive quality** — are the generated observations good?
2. **Recoverability** — can the true environment state be read out of the latent?
3. **Coherent rollout** — does it stay sane over long horizons, internally and in observations?
4. **Causal editability** — can we intervene on the latent and get a *coherent, intended* change?

**Working unification:** affordances 2–4 may all be forms of *identifiability* (of state, of dynamics, of causal
factors), and all may be downstream of a single property — a **canonical, factored, predictively-sufficient**
state (a clean function of the world's minimal sufficient statistic). **Editability is the sharpest test** of that,
which is why the project has concentrated there.

**North star:** *persistence* — world models with stable structure over long horizons. **Long-term aspiration:**
show that implicit architectures have failure modes that motivate **explicit physical/object scaffolding**, then
propose architectures that bake it in.

## 2. The setup

- **World:** 2D perspective frustum, 2 moving discs, observation = a **1D 128-ray intensity scan** (the only
  observation). Ground-truth state fully known, so probes and edits have a reference. Minimal sufficient statistic
  = `(positions, velocities)` = **8 dimensions** for 2 objects.
- **Models:** GRU (256 hidden) and RSSM (det 256 + stoch 64), both trained **purely to predict observations** — no
  state supervision. A DiT exists but is not part of the main thread.
- **Method:** probes (linear + MLP) read the latent; *editors* write to it; a model-agnostic eval suite scores the
  result in observation space against ground truth.

## 3. What is established

### 3.1 State geometry (`findings/state-geometry.md`)
The visited hidden states occupy a **low-dimensional, strongly curved manifold**. Honest intrinsic dimension
**~5–7** (TwoNN 5.2, MLE 6.9), which *brackets the physical 8 DOF*. The fatter global-PCA hull (38–73 dims) counts
the **curved embedding**, not degrees of freedom. Local tangent planes rotate **~56°** at nearest-neighbour spacing
and never align with the global PCA subspace. Practical consequence: global-PCA off-manifold residuals are
*curvature-blind*, so they cannot detect edits that stay inside the kept subspace but leave the curved surface.

### 3.2 Editability — the central negative (`findings/editability.md`)
**"Readable ≠ controllable."** Position is linearly readable (R² ~0.84) and velocity nonlinearly readable, yet no
tractable structural editor produces a clean, selective object move. The barrier has been progressively localised,
and two earlier explanations were **refuted along the way**:
- *not* "decode ≠ generate" (that was a magnitude-scaling artifact);
- *not* "target unreachable under a manifold constraint" (a constant-step geodesic *does* reach the readout target);
- *not* velocity-incompleteness (completing the target to full `(pos,vel)` changes the observation by ~1.4%).

What remains: the latent is **predictively sufficient but non-canonical** — ~35% of `h` is not a function of
`(pos,vel)`, and the `(pos,vel)→h` embedding is strongly curved, so linear/min-norm edits leave the manifold. An
unconstrained observation-gradient objective *can* render the target, but only by landing **off-manifold**, where
the dynamics reject it.

### 3.3 It is not GRU-specific (`findings/architecture-independence.md`)
A refined, KL-regularised RSSM reproduces **every part** of the failure. Two notable sub-results: the world state
lives in the **deterministic recurrent core, not the stochastic latent** (linear position R²: det-only 0.84 ≈ full
0.85 ≫ stoch-only 0.59 — refuting the expectation that the stochastic latent carries the compact world state), and
the KL structure buys **no** extra canonicity (fiber residual: GRU 0.337 ≈ RSSM det-only 0.368). Caveat the repo
states explicitly: *two points is a line, not a law.*

### 3.4 Actions don't put objects in the state (`findings/object-individuation.md`)
Training on **exogenous** object-moving actions (large relative moves, absolute teleports, axis-restricted) does
**not** create a grabbable object handle. Targeting object *k* on the passive latent, the **ghost never clears** and
edits are **non-selective**. The affordance-free baseline is as good or better. Reading: **objecthood lives in the
input→dynamics pathway (a "button"), not in the state (a "handle")** — it does not transfer across write
mechanisms. The one positive is *legibility*: action-training makes the latent a more canonical, more linearly
readable function of `(pos,vel)`. So: **readable ≠ grabbable**.

## 4. Recent control experiments (flagged, 2026-07-30)

Three controls that each attacked a plausible "your negative is an artifact" objection. **All three left the
editability negative standing**, which considerably strengthens it:

- **Noise ablation** — the repo's worlds carry two independent noise sources (observation noise 0.2, and dynamics
  noise) that had never been separated. The §4 negative holds in a **fully deterministic, perfectly-sensed** world.
  Surprise, opposite to the pre-registered guess: **observation noise acts as a regulariser** that makes position
  more *linearly* readable and the state more canonical.
- **Hidden-size sweep** — the negative is **capacity-independent**. Prediction saturates by H=128; linear
  readability rises *monotonically* with capacity (refuting the guess that a squeezed latent would be more linearly
  readable); at **every** size the probe-directed editors leave the ghost while the decoder-gradient oracle clears
  it on the same model and decoder.
- **Encoder-space editing** — *where you write matters*, which is new: the same pseudoinverse edit that is inert on
  `h` moves the state measurably at the encoder port `x`. But every probe-directed encoder write still lands on the
  **unedited** side of the Edit Index, while a freeze-time render oracle through the *identical* port crosses to the
  edited side. So a target encoder vector that moves the object **demonstrably exists**, and the probe-directed
  write cannot find it.

## 5. Can editability be trained in? (flagged, 2026-07-30)

Two mechanisms, five arms, evaluated on held-out edits:
- **Fine-tune the world model** so a *fixed, frozen* readout-injection editor works: Edit Index improves
  **+0.13** (heavy budget) — and the earlier "light fine-tune fails" negative turns out to be **partly a budget
  artifact** (+0.04 at 300 steps → +0.13 at 3000).
- **Amortized editor** `E_θ(h,target)→Δh` against a *frozen* model: **+0.54**.

But: **even the best arm only reaches "equidistant"** (absolute Edit Index −0.14, versus the oracle's +0.94); there
is **no mechanism generalisation** (a freshly-fit probe of the same class moves +0.01…+0.04) and **no content
generalisation** (training on object-0 edits only costs ~0.17 index points on object 1). Fine-tuning costs 13% of
next-step prediction; without a retention term the world model is destroyed *and editing gets no better*.

**The cleanest new fact is an asymmetry:** learning a bespoke editor for a *frozen* latent works far better than
making the latent obey a *fixed* editor. That is consistent with the standing reading — the obstacle is the
**reachability of the edit map**, not the representation itself.

## 6. Endogenous action (2026-07-28→29) — partly done, largely inconclusive

The follow-up the object-individuation finding explicitly left open: does it matter if the model **generates** the
actions rather than merely observing them? Setup: an **actor** (policy head on its own latent, acts on the world it
must predict) versus an **observer** (identical architecture, fed the actor's actions, never acts) — a control that
keeps *agency* separable from *architecture*. Three levels: position-shift, force/momentum, and force + a survival
goal trained by REINFORCE. Required building an interactive, steppable world and a keyboard emulator.

**What it produced:**
- **Goal-directed agency improves latent legibility** — but the effect **shrank sharply when both roles were
  trained to strength**: the position advantage collapsed to the level of the *no-goal control*, i.e. it stopped
  being goal-specific at all. What survived: a smaller velocity-readability gain and a canonicality gain.
- **Editability still failed**, with a sharper control than before: on the *same model, same decoder, same rollout*,
  the decoder-gradient oracle and the true-state swap succeed while every structural editor is inert. That rules out
  "the model is just bad" as the explanation and points squarely at edit-map reachability.
- **A separate, unresolved problem surfaced: closed-loop imagination decouples from reality.** Running the model on
  its own predictions, it dies at roughly the rate of having no policy at all, and its imagined observation stream
  reaches random-frame-level error within ~10–20 steps. This survived every fix attempted (three genuine
  implementation bugs found and fixed, 16× batch, 3× data) and an RSSM attempt.
- **The RSSM leg is incomplete.** World model trains fine (KL active, no posterior collapse), but the
  imagination-trained actor never learned the task, so the agency comparison could not be run.

**Status: set aside, unresolved.** An earlier speculation of mine — that the 1D observation channel might be too
impoverished for long-horizon coherence — **has been retracted**; it generalised from under-engineered attempts to a
claim about what is *achievable*. Teacher-forced prediction is good, so the observation carries the information;
Dreamer-class models do this on harder domains. The honest label is **"not achieved by our implementation yet."**

## 7. Methodological assets built along the way

- **The Edit Index** — a calibrated, bounded (−1…+1) editability metric: is the output closer to the world where
  the edit happened, or the one where it didn't? Deliberately hard to game: a scrambled or collapsed output scores
  ≈0 rather than spuriously well, and this repo's dominant failure ("paint a copy at the target while keeping the
  ghost") correctly reads ≈0. Must be read against each model's own unsteered row.
- **A canonical §4 metric/editor registry** implemented once in code, so notebooks stop re-deriving formulas.
- **An interactive world + human-playable emulator**, and a **vectorised GPU simulator** (bit-exact parity-tested
  against the scalar one) that makes large-batch online training feasible.

## 8. Open questions — the material for a strategy discussion

1. **The constructive move.** The negative results were always meant to motivate **explicit object scaffolding**.
   Four independent lines now say the interface (observations, actions, capacity, noise, training) is jointly
   insufficient. Is it time to *build* the explicit-structure architecture and show what it buys?
2. **Why is a reachable target unreachable?** The oracles prove a latent that renders the edit **exists** and is
   reachable through the encoder port. Every probe-directed method fails to find it. That gap — not the
   representation — is arguably the real finding, and it is not yet characterised.
3. **Does the "canonical state" unification survive?** It predicted that canonicality gains would buy editability.
   Action-training and observation-noise both *increased* canonicality with **no** editability gain. That is
   evidence against the unifying hypothesis as stated, and worth confronting directly.
4. **Coherent rollout (affordance 3) is the weakest-covered axis** and the endogenous thread suggests it is badly
   broken — but our own implementation quality is confounded with it.
5. **Endogenous action remains genuinely untested** — the question the object-individuation finding raised is still
   open, pending a working imagination-trained agent.
6. **Scope.** Nearly everything is 2 objects, one toy world, one observation modality, N≈48–64 edits, often one
   seed. What would need to hold at larger scale before any of it is publishable as a general claim?

## 9. Honest caveats

- Findings are scoped to **specific trained checkpoints**, not architectures in general; the repo is disciplined
  about this and the scope banners should survive into any talk.
- The strongest, most-replicated result is a **negative** (editability fails, robustly, across capacity, noise,
  actions, architectures, and training). Negatives are informative here but need framing.
- Several threads are **flagged, not promoted** — the controls, trained editability, and endogenous action all await
  a promotion decision.
- Some deliverables are outstanding: an RSSM leg for several threads, a consolidated endogenous notebook, and a
  known un-normalised curvature metric that should not be compared across notebooks.
