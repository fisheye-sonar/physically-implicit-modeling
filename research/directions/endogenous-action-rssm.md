# Direction: Endogenous action in an RSSM — does a latent-consistent world model become editable?

**Tag:** `[reframe]` · **Sub-question:** 3 (editability/objecthood) + 2 (identifiability) · **Status:** proposed
(2026-07-29) · **Complexity:** high (action-conditioned RSSM transition + policy/value heads + imagination-based actor
training + online interactive loop). Builds on `pim/world_models/rssm/`.

Companion to `endogenous-action-interactive-world.md` (the GRU thread). Registry: `ENDOGENOUS_RUNS.md`.

## Why now — the GRU thread hit a wall we can localise

The endogenous-action GRU actor **learns the survival task** but its world model is only usable while it can see the
world. Under closed-loop control (its own predictions fed back after a 15-frame warm-up) it dies at roughly the rate of
having no policy at all, and its imagined observation stream reaches **random-frame-level error within ~10–20 steps** —
it *decouples* from reality rather than degrading. Critically, we **exhausted the plumbing explanations**:

| run | fixes applied | teacher-forced | **closed-loop** |
|---|---|---|---|
| `L3s0` | none | 2.8 | **85.0** |
| `L3s0_ait` | action in the transition | 2.8 | **72.2** |
| `L3_bestgru_b1024` | + carried state, dead-state reset, value bootstrap, 16× batch, **3× data (236M frames)** | 6.1 | **87.2** |
*(deaths per 1000 frames; the no-goal control dies at 79.1, i.e. "no policy at all")*

Three real implementation bugs were found and fixed along the way (the action had a **literally zero-effect** pathway to
the state; the recurrent state was zeroed every 48 frames; carried state of dead-and-reborn worlds was stale). **None of
them was the cause.** Nor was capacity, training length, or data volume.

**What is missing is a training signal, not plumbing.** Nothing in the GRU objective ties the *imagined* latent
trajectory to the *observation-informed* one. The obs-space multistep loss was a weak proxy: it constrains decoded
pixels a few steps out, not the latent's own dynamics. **KL(posterior ‖ prior) is exactly that missing constraint**, and
training the actor *in imagination* makes the imagined rollout the thing the policy is optimised against — so a
fantasy-prone model is penalised directly instead of incidentally.

Secondary motive (Sevan, standing): every finding in this repo is checked on both GRU and RSSM. Also, our previous RSSM
was bent toward pure prediction; this is the chance to run **standard RSSM procedure** instead.

## The question

Does a world model whose latent dynamics are held consistent with observation (KL prior↔posterior) and whose policy is
trained inside its own imagination (a) produce coherent closed-loop rollouts, and (b) become **editable** — a
separable, grabbable object handle in the latent — where the GRU did not? And is any gain attributable to **agency**
(actor vs observer) rather than to the architecture change?

## Hypotheses (state before running)

1. **Closed-loop coherence improves substantially.** Prediction: closed-loop death rate drops well below the no-goal
   control's 79.1, and imagined-vs-real RMSE stays below the copy-previous-frame baseline (0.160) for ≥20 steps.
   *This is the one I expect to hold.*
2. **Editability remains poor.** The §4 latent-surgery negative is structural (`findings/object-individuation.md`,
   `findings/editability.md`), and a better-behaved latent need not be a *grabbable* one. Prediction: ghost ratio stays
   ≈1.0 for structural editors while the decoder-gradient oracle and true-state swap still succeed. *If this is wrong —
   if RSSM's latent IS editable — that is the most important positive result of the whole thread.*
3. **The actor-vs-observer gap survives** at least on velocity readability and canonicality. Suggestive lead from
   `L3_bestgru_b1024` (position Δ +0.045, velocity Δ **+0.114**, actor more canonical by 0.028) — but confounded there
   by batch/data changes, so RSSM is where it gets tested properly.

## What to build

### 1. Action-conditioned RSSM (`pim/world_models/rssm_actor.py`, NEW — do not modify `rssm/model.py`)
The existing `RSSMModel` already provides `RSSMState(h, s)`, `_prior(h)`, `_posterior(h, e)`, `observe_step`,
`imagine_step`, `decode`, `flat_state`/`state_from_flat`, a `sample` toggle, and KL with free nats. **Two gaps:**

- **The transition ignores actions.** Currently `h_t = GRUCell(s_{t-1}, h_{t-1})`. Change to
  `h_t = GRUCell([s_{t-1}, proj(a_{t-1})], h_{t-1})`. This is non-negotiable: it is the bug that cost us a day on the
  GRU, where the action could not affect the state at all.
- **No policy/value heads.** Add `policy(flat_state) → logits` (same factored discrete space as the GRU: per object ×
  per axis over {−1,0,+1}, so the emulator's key overlay keeps working) and `value(flat_state) → scalar`.

Conform to `HiddenStateModel` with the action defaulting to no-op so the whole existing eval/editor suite runs unchanged.

### 2. Standard RSSM/Dreamer objective (explicitly *not* the pure-predictive hack we used before)
```
world-model loss = recon(obs)  +  β · KL_balanced(posterior ‖ prior)   with free bits
KL balancing (DreamerV2): β_post·KL(sg[post] ‖ prior) + β_prior·KL(post ‖ sg[prior]),  ~0.8 / 0.2
free bits: clamp the per-dimension KL below `free_nats` (repo's tuned RSSM used 3.0)
actor  loss = −λ-return over an IMAGINED rollout (horizon H≈15) from posterior states, + entropy bonus
critic loss = regression onto the same λ-returns
```
**Latents:** keep the existing **diagonal-Gaussian** stochastic latent rather than moving to DreamerV2 categoricals —
one fewer changed variable, and it preserves comparability with the repo's existing RSSM findings. Revisit only if the
KL misbehaves.

**Actor trained in imagination** is the substantive departure from the GRU thread: the policy never sees real rollouts,
so a self-consistent fantasy is directly penalised through the critic rather than tolerated.

### 3. Online interactive loop (`scripts/train_rssm_endogenous.py`)
Reuse `BatchedInteractiveWorld` (vectorised, GPU-resident, parity-tested — 11 tests incl. bit-exact float64). Same world
as the GRU thread so the comparison is meaningful: `force` dynamics, lethal (collision + wall), `init_speed=0.28`,
death → 4 noise frames → rebirth.

**Carry the recurrent state across chunk boundaries, and clear it for worlds that died** — both lessons from the GRU
thread; the second one caused a total policy-entropy collapse when omitted.

**⚠ Observation noise: use `obs_noise_std=0.2`** (the repo standard for every dataset). The GRU thread accidentally ran
at 0.05 — see the deviation box in `ENDOGENOUS_RUNS.md`. Starting RSSM at the standard removes that debt, at the cost
that absolute RMSE is not comparable to the GRU runs (the *actor-vs-observer* and closed-loop comparisons still are,
since both roles share the setting).

### 4. Actor / observer twin (the central control — Sevan's constraint)
Identical architecture, trained simultaneously on the same data. The **actor** acts and is trained on world-model +
actor + critic losses. The **observer** never acts, is fed the actor's actions, and is trained on the **world-model loss
only**. This keeps *agency* separable from *architecture*: any RSSM-vs-GRU difference is architectural, while
actor-vs-observer *within* RSSM isolates acting. Without this the whole comparison is confounded.

## Runs (register each row in `ENDOGENOUS_RUNS.md` in the same commit)
| code | purpose |
|---|---|
| `R3s0` | RSSM level 3 (force + survival goal), seed 0 — the main run |
| `R3s1` | seed replication |
| `R2s0` | RSSM level 2 (force, **no goal**) — the no-goal control at matched capacity; the analogue of `L2s0` |

Large batch (1024) now that the vectorised simulator makes it ~126 ms/iteration; size the frame budget rather than the
iteration count. Decouple training into foreground script calls per `WORKER.md`.

## Readouts (reuse existing machinery; do not reinvent metrics)
1. **Closed-loop coherence (the primary new readout).** `AutoregressiveModelDriver` in `scripts/play.py` +
   deaths-per-1000-frames, teacher-forced vs closed-loop, and imagined-vs-real RMSE per step against the
   copy-previous-frame (0.160) and random-frame (0.393) baselines. **Directly comparable to the GRU table above.**
2. **§4 editability / grabbability.** `scripts/eval_editability_endogenous.py` — reach / collateral / **ghost** /
   selectivity / persistence, with the true-state swap and decoder-gradient oracle as brackets. Ghost is the decisive
   axis (1.0 = the object never left).
3. **Identifiability + canonicality.** `scripts/eval_endogenous.py` — linear/MLP (pos, vel) R² and fiber residual,
   actor vs observer. **Report det-only `h`, stoch-only `s`, and full `[h,s]` separately** (the repo's standing RSSM
   convention — the 2026-07-08 correction showed conflating them killed a claim).
4. **Animations.** `endogenous_agent_animations.ipynb` machinery — teacher-forced and closed-loop, actor's dream beside
   the observer's, in the same `play.py` view.

## What would count as an answer
- **Closed-loop fixed + still not editable** → the strongest version of "readable ≠ grabbable": even a latent-consistent,
  imagination-trained world model does not contain grabbable objects. Motivates explicit object scaffolding
  (RESEARCH.md endgame). *Most likely outcome.*
- **Closed-loop fixed AND editable** → the major positive of the thread; would mean editability was gated on latent
  dynamical consistency all along, and would send us back to the GRU with a KL-like term.
- **Closed-loop still broken** → the pathology is not about the objective either; suspicion moves to the observation
  channel (1D scans may be too impoverished for long-horizon self-consistency) or the task.

## Risks / caveats to watch
- **KL collapse / posterior collapse** — the classic RSSM failure. Watch the KL term and the prior/posterior gap; free
  bits exist to prevent it. If the prior collapses to the posterior, imagination will look great and mean nothing.
- **Imagination training instability.** λ-returns over imagined rollouts can diverge if the critic is poor early. Warm
  up the world model before enabling the actor loss.
- **Three bugs in three days in the hand-rolled GRU RL loop** is the reason to follow standard practice closely here
  and to add the same kind of diagnostics up front (does the action change the state? do the states in the update match
  the ones acted from? does entropy stay alive?).
- Deaths make the world non-stationary in a way that interacts with imagination horizon H; if H spans a death, the
  imagined rollout crosses an unpredictable reset. Consider masking or truncating imagination at predicted deaths.

## Deliverables
New `pim/world_models/rssm_actor.py`, `scripts/train_rssm_endogenous.py`, checkpoints in gitignored
`runs/endogenous_rssm/`, an executed notebook under `notebooks/experiments/editability/actions/`, and a dated
`research/scratch/2026-07-..-endogenous-action-rssm.md`. Do NOT edit `findings/`, `RESEARCH.md`, or the master notebook.
Calibrated claims; a clean negative is a strong result.
