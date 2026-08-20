# Finding: Editability (causal manipulability of hidden state)

*Sub-question 3 — are targeted latent edits coherent, intended behavioral changes?*
Model/data context unless noted: GRU `3_dset3_gru_persistentids_inview_400epochs`,
dataset `4_fixed_refl_inview`, 2 objects, N=500 edit samples.
Notebooks: `notebooks/experiments/editability/` (`canonical_state_editing`,
`geodesic_walk_k150`, `manifold_geometry_diagnostic`).

> **Scope (preliminary, 2026-07-09).** These claims concern the *specific trained checkpoint* under
> study — a GRU trained **purely to predict the next observation** (no state supervision), on
> `dataset 4`, at this stage of the investigation. They are **not** claims about GRUs / recurrent
> world models in general. A different training objective (e.g. an editability- or
> disentanglement-aware loss), dataset, or scale could change them. Read "the GRU" below as "this
> pure-next-step-prediction GRU."

## Current understanding

_Updated 2026-08-19._

_(2026-08-17: backfill of 2026-07-18 → 2026-08-14, which the old promotion gate had blocked since
2026-07-17.)_

**The negative is real, located, and now has a positive counterexample.**

1. **What fails, and it is not what we first thought.** Every failing editor is a
   **probe-derived write** — a correlational direction, inverted. The barrier is not
   reachability (a constant-step geodesic reaches the target readout; writing to the *entire*
   116-dimensional linear position code still does not edit), not capacity
   (the failure holds at `H` = 8…512 and *worsens* with capacity), not the observation's
   lossiness (it survives full 2D observability), not the space the write happens in
   (`h`, encoder input `x`, and a VAE latent all fail), not the metric used to take the
   step (whitening helps and is not enough), and **not this repo's editor implementations** —
   the strongest published probe-derived write, Othello-GPT's multi-layer activation gradient,
   was ported exactly and fails here too (2026-08-18). What was missing is a **map from the
   intended change to the required state change** — which a pseudoinverse estimates badly.
2. **Why the probe direction is wrong is inherited from the world, not learned.** The
   required displacement is near-orthogonal to what a probe-derived write can apply, and this
   holds **in observation space itself**, before any model is trained. `readable ≠ grabbable`
   is a property of the renderer.
3. **What works needs dynamics — and a source of VALID observations.** The reliable mechanisms
   (counterfactual state overwrite, freeze-time teacher forcing, the decoder-gradient oracle, and
   a history rewrite that re-renders each frame) all make the model **consume evidence over
   time**, and all obtain that evidence from something that can *generate a real observation*.
   Where the evidence is synthesised by a probe-derived write instead, it fails — including when
   applied to **every frame of the history at once** and including when applied to the **raw
   observations** (2026-08-18). So the barrier is **frame validity**, not the coherence of the
   evidence across time and not the precision of the write, both of which were ruled out by that
   experiment. No edit that works without an external observation model has been found.
4. **A learned editor crosses zero.** `E_θ(h, start, target) → Δh` with the world model
   **frozen** reaches Edit Index **+0.20** against unsteered −0.67, at zero prediction cost —
   the first probe-free latent editor to go positive. See `trained-editors.md`.
5. **The un-edited complement is a decaying sensory trace**, not a record of past positions —
   the first content-level account of what the rest of `h` holds.

6. **The edit direction is a well-defined object in every architecture, and a trained pathway reaches it.**
   The two oracle mechanisms agree on the displacement at cos +0.59 … +0.91 across GRU, RSSM, transformer and
   both DiT variants, and on the action-conditioned GRU the **trained action channel** lands within 29° of the
   counterfactual oracle — the tightest pair measured. The agreement still buys no shared edit axis (cross-episode
   cosine ≈ 0) and no probe visibility (at or below chance, except the latent DiT's 64-d code at 1.17×), so it
   sharpens rather than overturns point 1: the map from intended change to required state change **exists**; it
   is not a probe direction (2026-08-19).
7. **`First Obs. TF` is not one number.** Whether a single uncued post-edit frame persists runs
   −0.00 → +0.22 → +0.53 across GRUs that differ only in whether they saw teleports and whether those teleports
   were cued by an action (2026-08-19).

**Status of the older synthesis** (predictively sufficient but non-canonical; `readable ≠
controllable`; the fiber not collapsed; curvature of the `(pos,vel)→h` embedding) — all still
holds and is preserved below.

<details>
<summary>Previous synthesis (2026-07-09) — still accurate, superseded in framing only</summary>

The probe (decode) direction **IS causally connected to the observations** (the
earlier "decode≠generate" reading was a magnitude-scaling artifact). But editing
still fails, and the barrier is **not** "target unreachability under a manifold
constraint" (superseded — the target readout *is* substantially reachable
on-manifold via a constant-step geodesic, RMSE→0.35). The real barrier is that
**this GRU's `h` is predictively sufficient but non-canonical**: (i) ~35% of `h` is not a
(nonlinear) function of the world's minimal `(pos,vel)` sufficient statistic (the
decode fiber is not collapsed); (ii) the `(pos,vel)→h` embedding is strongly curved
(linear→MLP fiber-residual drop ~0.53), so linear/min-norm edits leave the manifold;
(iii) neither is a clean *linear* coordinate — position is linearly readable
(R² 0.84) but velocity only *nonlinearly* (single-frame MLP R²≈0.94 late-t vs linear
≈0.59); velocity is **not** a temporal feature (see 2026-07-08 log). Consequently
**completing the edit target to the full `(pos,vel)` does not fix the ghost** (obs moves ~1.4%, identical to position-only) — killing the
velocity-incompleteness hypothesis. An unconstrained obs-gradient objective *can*
render the target, but only by jumping to an **off-manifold, non-canonical latent**
(residual 15.7 vs ~1.75 for real states) that the dynamics reject within a few
steps. The probe-objective moves the readout but not the obs; the obs-objective
moves the obs but off-manifold — **readable ≠ controllable**, localized. Forcing an
on-manifold global-PCA edit still yields only ghost-ridden ~37%-of-swap change (it
moves the obs partly by *scrambling*, not clean relocation).

</details>

## Log

### 2026-08-19 — The edit direction is well defined in every architecture, and a *trained* pathway lands on it · `replicated` ★-candidate

**Evidence:** `scratch/2026-08-19-latent-edit-directions.md` ·
`notebooks/experiments/editability/latent_linearity/` (notebook + `edit_directions.py` + `figures.py` +
`LATENT_LINEARITY_RUNS.md`) · dataset 4 `edits` split N=256 (Part 1) and
`datasets/15_teleport_eval_single/eval.h5` N=256 (Part 2) · runs `H256`, `4_dset4_refined_best`, `W16`,
`0_latent_dit_z16_w4`, `9_dset4_dit_w4_d256`, `XG_A_H256`, `XG_C_H256`. **No models trained.**

Sevan's spec: extend the ground-truth edit-direction analysis of 2026-08-03 to all four architectures, then ask
whether the two *learned* edit pathways write the same displacement as the training-free oracles.

**1. The two oracles agree on the displacement in every architecture.**
`cos(Counterfactual Overwriting, Freeze-time Interp. TF @8)` per episode then averaged, edit-only Δh
(`edited − matched control`, same noise draw), N=256: pixel DiT residual stream **+0.910 (25°)** · GRU `h`
**+0.808 (36°)** · transformer residual stream **+0.806 (36°)** · latent DiT latent window **+0.667 (48°)** ·
RSSM det+stoch **+0.593 (54°)**, against shuffled-pair controls of +0.00 ± 0.22 (4.0–5.5× enrichment on the
projection fraction). Both mechanisms land the edit on every model (Edit Index +0.52 … +0.66, except the RSSM's
freeze-time at +0.097) with fidelity 0.57–0.83, and the waterfalls confirm it in observation space.

`replicated` because the GRU and RSSM values reproduce 2026-08-03's +0.799 / +0.569 on an **independently
constructed** Δh — edit-only rather than raw, and with the RSSM's posterior/prior chain corrected — and because
three further architectures give the same answer.

**2. The trained action channel writes the oracle's displacement.** On `XG_A_H256`, the one checkpoint where all
four mechanisms exist, all four land (counterfactual +0.643 · freeze-time +0.563 · **action interface +0.645** ·
first-obs +0.216 vs unsteered −0.641), and their Δh mutually align at +0.72 … +0.87. The **tightest pair measured
anywhere** is counterfactual overwrite vs the trained action channel: **+0.872 (29°), 5.9× chance.** A pathway
learned from data and an oracle that rewrites the model's history arrive at nearly the same latent displacement.
This is the first evidence here that "train something that emits Δh" targets a well-defined object rather than a
hoped-for one. It is **correlational**: the falsifying test is to corrupt the action channel's write while
holding its read-out accuracy fixed.

**3. Whether one uncued post-edit frame persists is a fact about the training distribution, not the mechanism.**
`First Obs. TF` step-0 Edit Index on three GRUs identical but for what they saw in training: **−0.002** never saw
a teleport · **+0.216** teleports always cued by an action (`XG_A`) · **+0.532** teleports seen uncued (`XG_C`).
Sevan predicted the action-conditioned model would not commit to an *uncued* teleport, and it does not, while the
identical recipe with the action input removed does. `First Obs. TF` must therefore always be quoted with the
model's training distribution attached.

**4. What did not appear.** No shared "an object moved" axis: the cross-episode cosine between different edits'
Δh is **+0.00 … +0.04** in every model and mechanism against a chance level of 0 (replicating 2026-08-03 §5's
+0.011 on four more state objects). And the direction stays at or below chance visibility to a linear position
probe — row-space fraction ÷ chance: GRU 0.73× · transformer 0.49× · pixel DiT 0.14× · RSSM 0.03×; the
action-induced Δh is 0.91×. **One exception: the latent DiT's 64-d carried code at 1.17×** (1.46× for first-obs)
— the only state object above chance in the study, and it is also by far the least linearly readable
(position R² 0.220 vs 0.74–0.86). Magnitudes are 2.4–6.4 × one ordinary dynamics step everywhere.

**Why it matters for the thread.** The negative half of this thread's synthesis — probe-derived writes fail
because there is no map from the intended change to the required state change — now has its positive half
measured on four architectures: **the map exists and is well defined per episode**; it is simply not a probe
direction and not a single shared axis. And point 2 says a learned pathway can reach it.

**Caveats.** One checkpoint per architecture, one seed, one world. Part 2 is GRU-only because no teleport-trained
RSSM / transformer / DiT exists in this repo (audit in `LATENT_LINEARITY_RUNS.md`), so mechanisms 3 and 4 are
untested for architecture-independence. The RSSM is the outlier in every measure and its freeze-time arm barely
edits, so part of its low cosine is the displacement of an edit that did not land.

---


### 2026-08-18 — The Othello write applied to the WHOLE history, renderer-free: still fails. Frame validity, not consistency, is the barrier · `replicated` ★-candidate

**Supersedes 2026-08-18** (the history-rewrite entry below): that entry's numbers stand, its
*interpretation* does not. **Evidence:** `scratch/2026-08-18-history-rewrite-renderer-free.md` ·
`notebooks/experiments/editability/othello_gpt/history_rewrite.ipynb` (+ `history_edit.py`) ·
`runs/transformers/W16`, `datasets/4_fixed_refl_inview`, `ef=20`, `K=15`, **N=256**.

The earlier entry reported a history rewrite reaching **+0.626** and read the barrier as *coherence
of the evidence*. That arm rebuilt each frame with the **simulator's renderer**. Re-run without it —
the paper's own MLP write applied at **every history frame**, and the same write applied to the
**observations themselves** — the effect disappears:

| arm | renderer-free | read-out after | EI step 0 | step 14 | fidelity |
|---|---|---|---|---|---|
| Unsteered | — | — | −0.684 | −0.439 | 1.000 |
| Latent write · single frame | ✅ | 0.018 | −0.538 | −0.428 | 0.994 |
| **Activation history edit @0** (every frame) | ✅ | **0.008** | **−0.544** | −0.428 | 0.995 |
| **Observation history edit** (MLP write on the frames) | ✅ | 0.079 | **−0.459** | −0.302 | 1.001 |
| *Rebuilt history* | ❌ **render** | — | *+0.626* | *+0.351* | *0.674* |

1. **Widening the write from one frame to twenty buys 0.006 index points** (−0.538 → −0.544), with
   *better* read-out convergence (3.64 → **0.008**). Consistency across frames is **not** the missing
   ingredient.
2. Applied-layer ordering unchanged: −0.544 (point 0) → −0.637 (point 4).
3. Writing the observations directly helps but not qualitatively: **−0.459**, the only renderer-free
   arm whose effect survives the rollout at all (−0.302 vs unsteered −0.439).
4. **The whole difference is frame validity.** The observation edit and the render reference use the
   same `δ`, the same per-frame targets, and the same channel, and differ by **1.085 index points**.
5. **Off-manifold, not timid:** the MLP edit changes the frames *less* than the render does (relative
   change 0.539 vs 0.881) while scoring far worse. Real observations here are strongly saturated —
   **39.3%** of pixels at the intensity rails; after the MLP write, **2.6%**. Visibly broadband
   striping with the plateau structure destroyed.

**Why it matters:** it removes an over-reading that had entered this file the same day, and it
sharpens the standing negative. Probe-derived writes now fail on **every** surface tried — `h`,
encoder port, VAE latent, residual stream at one frame, residual stream at *all* frames, and raw
observations — for one reason: **they cannot synthesise a valid observation.** The probe pins a
4-dimensional read-out and leaves 124 dimensions free, and essentially every member of that solution
set is not a picture of this world.

Third independent route to the same conclusion, after `observation-space geometry` (2026-08-05,
measured with **no model**) and `input_grad_steering` (2026-08-11, single-frame input gradients).

**Owed:** replace the simulator's renderer with a **learned** `positions → observation` map fit on the
training split. If that recovers most of the +0.626, the requirement is "any valid observation model",
not "the true one" — which is something a model could own.

**Caveats:** one model, one dataset, N=256. Step sizes chosen by read-out convergence; both writes
reach an interesting-looking index only past that optimum and by degrading (activation α=0.4 →
−0.084 at fidelity 1.091).

---

### 2026-08-18 — Rewriting the whole observed history crosses zero and holds, with no ground truth · `observed` ★-candidate

**Evidence:** `scratch/2026-08-18-history-rewrite.md` ·
`notebooks/experiments/editability/othello_gpt/history_rewrite.ipynb` (+ `history_edit.py`) ·
`runs/transformers/W16`, `datasets/4_fixed_refl_inview`, `ef=20`, `K=15`, **N=256** — the same
episodes as the Othello-port entry below.

To teleport an object by `δ` at the edit frame, apply **the same `δ` to every prior frame**,
rebuilding each observation from the model's **own decoded positions**, and teacher-force on that
history. Velocity is constant in this world, so translating a whole track by a constant `δ` is
itself a valid trajectory — the rewritten history is a *consistent world*, not one inconsistent
frame the dynamics must absorb. **No ground truth is used by the method**: positions come from the
probe reading the model's own residual stream, and rendering needs only radius and reflectivities,
which on a `fixed_reflectivities` dataset are world constants shared by every episode.

| arm | EI step 0 | EI step 14 | fidelity |
|---|---|---|---|
| Unsteered | −0.684 | −0.439 | 1.000 |
| Reconstruction control (`δ=0`) | −0.569 | −0.375 | 1.039 |
| Latent write (Othello method) | −0.538 | −0.428 | 0.994 |
| Oracle observation *(leads by one)* | +0.126 | −0.030 | 0.858 |
| **History rewrite (`δ`)** | **+0.626** | **+0.351** | **0.674** |
| Oracle history rewrite (`δ`, GT positions) | +0.640 | +0.364 | 0.603 |

1. Gain over its **own** reconstruction control: **+1.195** at step 0, **+0.727** at step 14. The
   latent write gains +0.146 → +0.010 on the same episodes.
2. **Not a degradation artefact** — fidelity **0.674**, i.e. the rollout ends **33% closer** to the
   true post-edit world than doing nothing.
3. **Decode error is nearly irrelevant.** The GT-position oracle reaches +0.640 against the decoded
   +0.626 — a gap of **0.014** — despite a decoded-position RMSE of 0.49 sim units. The method needs
   a **consistent** read-out, not an accurate one.
4. **Depth sweep:** step-0 index +0.265 (rewrite depth 1) → +0.594 (depth 5), flat from depth 8;
   step-14 index keeps climbing +0.080 → +0.302 (8) → **+0.355 (16)**, flattening near the model's
   16-frame per-layer attention window. Placing the object and *holding* it need different amounts
   of history. The window correspondence is **suggestive and untested** — it needs W2/W4.
5. **A single rewritten observation frame beats every latent write**: depth 1 = +0.265 vs −0.538.

**Why it matters:** it sharpens the thread's standing conclusion rather than overturning it. The
negative was never "this world state cannot be changed" but **"the latent is not the surface on
which it can be changed."** Everything that has ever worked here — counterfactual overwrite,
freeze-time teacher forcing, and now this — writes through the **observation channel**. What is new
is that this one needs **no oracle**.

**Reading (not established):** result 3 points at **coherence of the evidence**, not precision of
the write, as the barrier — an *inconsistent* write is rejected however accurately it hits the probe
target, while a *consistent* one is honoured even when substantially inaccurate. Falsifiable:
corrupt the rewritten history's internal consistency while holding accuracy fixed; the effect should
die.

**Caveats:** ⚠ **uses the renderer** — this is *not* a pure latent intervention and must never be
quoted beside the latent editors as if it were one; the observation function is treated as known.
Translating a track pushes it out of frustum on 10.0% of frames / 46.1% of episodes. One model, one
dataset, N=256. Constant velocity is what makes a constant `δ` a valid rewrite.

---

### 2026-08-18 — Othello-GPT's probe and intervention, ported exactly: the probing replicates, the editing does not · `replicated` ★-candidate

**Evidence:** `scratch/2026-08-18-othello-gpt-method-port.md` ·
`notebooks/experiments/editability/othello_gpt/` (notebook + `othello_probe.py` + `pipeline.py`) ·
transformer `runs/transformers/{W2,W4,W16}` (no new world models), `datasets/4_fixed_refl_inview`,
`ef=20`, `K=15`, **N=256** edits; probes on 1500 test sequences held out **by sequence**.
Source: Li et al., *Emergent World Representations*, ICLR 2023 (arXiv:2210.13382).

Every editor in this thread is a **probe-derived write**, and every one fails. Othello-GPT is the
strongest published claim that exactly this kind of write **succeeds**. Their method was ported
unchanged — probe families (linear vs **one-hidden-layer** MLP), the activation update
`x' ← x − α ∂L(p_θ(x), B')/∂x`, the **sequential multi-layer schedule** (write at the last timestep at
residual point `L_s` **and every point after it**, alternating write and compute), the hold-the-rest
term with weight `β` (App. G), and their null-intervention baseline.

**1. The probing half replicates cleanly.** Best position R² **linear 0.798 → MLP 0.934** (+0.136),
MLP rising monotonically with depth. Their §3 headline holds here: a nonlinear world representation
is present and a linear probe under-reads it.

**2. The intervention half does not.** The optimisation succeeds completely — read-out driven
**3.35 → 0.008–0.038** sim units at every applied layer — while the generation barely responds:
Edit Index **−0.684 (unsteered) → −0.534**, a gain of **+0.150** on a ±1 scale. The waterfall is
unambiguous: the object stays on the ghost locator and never reaches the target.

⚠ **Two corrections, 2026-08-19, neither changing the qualitative conclusion.** (a) That is the
**read-out-convergence** operating point (α = 0.05), chosen on principle — never select a step size
by the outcome metric — but it is **not** the arm's best honest setting: the step-size sweep reaches
**−0.194 at fidelity 1.014** (α = 0.3), inside the 1.05 guard, a gain of **+0.49**; +0.015 comes only
at fidelity 1.263, i.e. by degrading. The originally reported gain understated the method ~3×.
(b) The waterfall backing this was drawn from the **four largest teleports**, which sit at the
**98th percentile** of the Edit Index distribution (+0.07 vs a −0.54 mean); panels are now randomly
sampled. Both flagged by Sevan.

**3. Ignored, not destroyed; reverts within one frame.** Fidelity ratio **0.993–0.999** everywhere
(no arm near the 1.05 guard). Arms collapse onto the unsteered curve **by step 1**; the gap decays
**+0.146 → +0.010** by step 14.

**4. Earlier applied layers propagate further** — −0.538 (points 0/1) → −0.565 (2) → −0.606 (3) →
−0.622 (point 4), matching the structural prediction that an edit at residual point ℓ changes block
inputs for layers > ℓ only.

**5. A probe reading the ENTIRE world state changes nothing** — −0.553 (positions + velocities,
8 dims, identical edit objective) vs −0.534 (positions only); if anything marginally worse.
**Corrected 2026-08-19 — and the correction STRENGTHENS this point.** It was first reported with the
caveat that it was a *weak* test of completeness because velocity read at only −0.04…0.45. That was a
**bug in the probe's loss**, not a property of the model: the MSE was taken in raw target units, so
each output dimension's gradient share scaled with its variance and position outweighed velocity
~1000×. With the loss taken in standardised target space, velocity reads **−0.21 … 0.73**
(x-components 0.518 / 0.730; mean 0.158 → 0.276, matching a dedicated velocity-only probe) while the
edit arm moved only −0.539 → −0.553. The null therefore holds **with a genuinely informative velocity
read-out**. Position R² and every edit result are unaffected — position dims span only 1.2× in
variance. See `../GOTCHAS.md` (2026-08-19).

**6. The single-frame ceiling on this model is itself low.** The oracle observation — the model simply
*shown* the true post-edit frame — reaches only **+0.126**, decaying to −0.030. The probe write
achieves ~**18%** of that (+0.146 of +0.810).

**7. The optimiser decides which probe-satisfying write you land on.** At a matched selection rule
(lowest read-out error), Adam's write is **1.7–4.9× larger in norm** and moves the generation; plain
gradient descent lands the read-out with a smaller write and moves it essentially not at all (point 0:
read-out 0.192, Edit Index **−0.680** = unsteered). The set of activations satisfying the probe is
large and the probe constraint does not pin down a member the dynamics honour — the same shape as the
2026-08-05 tangent-constrained result, reached from a different direction.

**8. Flat across attention windows** — gain over each model's own unsteered index **+0.153** (W2) /
**+0.137** (W4) / **+0.146** (W16).

**Why it matters:** the probe-derived-write failure is **not an artefact of this repo's editor
implementations**. The strongest published version of that method, with its own schedule, loss,
multi-layer write and baseline, fails here too.

**Reading (not established):** this does not contradict Li et al., it locates the difference, and the
notebook does not separate the two candidates — (a) **the world**: their board state is discrete and
exactly determined by the move sequence and the flipped tile is consumed directly by the legal-move
computation, while ours is continuous and reaches the output only through a renderer (consistent with
2026-08-05, which put `readable ≠ grabbable` in the world rather than the model); (b) **the read-out**:
their probe predicts a quantity the next-token computation consumes, ours one merely correlated with
what the decoder consumes.

**Caveats:** one dataset, one architecture family, N=256. The probe is the **paper's** one-hidden-layer
MLP, **not** `fit_readability_probes` (2×256) — the R² values are not comparable to readability numbers
elsewhere in the repo. Held out by sequence, unlike the paper's by-frame split. Gates passed:
`state_from_obs` vs one-pass forward 8.3e-07; identity write == free-run exactly 0; one intervention
per episode (max jump 2.97 at `ef`, 0.246 anywhere else).

---

### 2026-08-13 — Grabbability *decays* with capacity while the action interface *rises* · `replicated`

**Evidence:** `scratch/2026-08-13-action-hidden-size.md` ·
`notebooks/experiments/editability/action_hidden_size/` (+ `ACTION_SWEEP_RUNS.md`) ·
`scripts/eval_action_sweep.py` · 15 runs across 3 model families × `H ∈ {8,32,128,256,512}` ·
`datasets/7_cont_teleport`, held-out `13_cont_teleport_eval`.

The legitimate gain of readout injection over its **own** unsteered row shrinks toward zero as
capacity grows, in every family:

| family | H=8 | H=32 | H=128 | H=256 | H=512 |
|---|---|---|---|---|---|
| passive GRU | +0.479 ⚠ | +0.181 ⚠ | +0.016 | +0.008 | +0.004 |
| exogenous · actions withheld | +0.382 ⚠ | +0.234 ⚠ | +0.025 | +0.026 | +0.016 |
| exogenous · actions given | +0.419 ⚠ | +0.278 | +0.026 | +0.020 | +0.007 |
| endogenous L3 | +0.194 | +0.057 | +0.001 | +0.001 | −0.001 |

⚠ = fidelity ratio > 1.05. **Every apparently large gain at H=8–32 is degradation**, not an
edit (fidelity 2.3–3.1; the H=8 waterfall shows saturated white bands, not a relocated
object). By H ≥ 128 the editors are **inert** instead — fidelity 1.00, Edit Index sitting
exactly on the unsteered line. The two failure modes trade places as capacity grows and
neither is an edit.

**What makes it decisive:** the exogenous-teleport family's action space *contains* the
intervention, giving a built-in ground-truth handle no passive model can offer. Over exactly
the capacity range where latent editing decays to nothing, that **action interface rises**:
**+0.216 → +0.455 → +0.582 → +0.618 → +0.608**. The handle exists; it is simply not in the
latent.

**Why it matters:** kills the "too much capacity" explanation in the direction opposite to the
pre-registered guess, and separates "no handle exists" from "no handle exists *in the state*".

**Caveats:** endogenous readability falls at H=512 (0.636 → 0.546); the honest reading is
under-training (fixed 6000 iterations, and the RL arm has the most to learn per parameter) —
flagged, not explained.

---

### 2026-08-13 — The un-edited complement of `h` is a decaying sensory trace, not past positions · `observed`

**Evidence:** `scratch/2026-08-13-history-editing.md` ·
`notebooks/experiments/editability/history_editing/{gru,transformer}_history_editing.ipynb` ·
`directions/history-editing.md` · GRU `runs/controls/H256`, dataset 4, N=256 edits, N=1500
probe sequences. Interpretation pre-registered in the brief before running.

The hypothesis: the part of `h` outside the probe's row space — which we never edit — is
information about *previous frames*. Tested by holding content, displacement `δ`, and history
length `n` fixed and varying **only the channel** (latent write vs teacher-forced rendered
observations).

1. **The past is readable, and only nonlinearly.** Linear probes read `pos(t−k)` at R² 0.828
   (k=0) → 0.784 (k=20) — but so does a **no-stored-history null** (read `(pos,v)` off `h`,
   extrapolate back), to within +0.0008 at every lag. The MLP probe separates: direct **0.883**
   vs learned null **0.737** at k=20. Calibration: the direct probe sits below the true-`(pos,v)`
   ceiling (0.991), so `h`'s knowledge of the past does not exceed what perfect knowledge of the
   present implies.
2. **The complement is observation content, not past positions.** Linear fiber residual 0.856
   of ‖h‖ (MLP 0.467). Regressed on predictors residualised against the present `(pos,v)`:

   | predictor | held-out R² | shuffled control |
   |---|---|---|
   | past positions, 1 / 2 / 5 / 10 frames | −0.0001 … −0.0007 | ≈ same |
   | **obs(t)** alone | **0.659** | −0.007 |
   | obs(t−1) / obs(t−2) / obs(t−5) | 0.609 / 0.550 / 0.364 | ≈ 0 |
   | past observations, 10 frames | 0.636 | −0.060 |

**Why it matters:** the first **content**-level account of the complement, rather than a
geometric description of it. It also explains why writing a position into the latent cannot
work: the complement is not holding a position record to contradict.

**Caveats:** one model family per notebook; `observed` until replicated at another capacity.

---

### 2026-08-11 — The negative survives full observability (omniscient 2D) · `replicated`

**Evidence:** `scratch/2026-08-11-omniscient-2d.md` ·
`notebooks/experiments/editability/omniscient_2d/` (+ `OMNISCIENT_2D_RUNS.md`) · N=256 held-out
edits, K=15.

Every prior result was measured through a 1D perspective scan that is lossy twice — it
**projects** and it **occludes**. Removing both changes nothing.

- **Pipeline validated first:** the 1D control `1D_H256_30k_s0` reproduces published
  `controls/H256` numbers to within 0.03 index points on *every* editor (injection −0.63 vs
  −0.66; counterfactual +0.68 vs +0.70; decoder-grad k=1 +0.96 vs +0.97; freeze-time +0.52 vs
  +0.52). The 30k restriction costs nothing, so no 1D↔2D difference is a sample-size effect.
- **The negative survives:** best training-free editor gain over its own unsteered row
  **+0.11 / +0.14** on the two omniscient arms vs **+0.13** in 1D. Pseudoinverse injection
  inert in both (+0.02 vs +0.03).

**Why it matters:** removes observation lossiness — the most intuitive remaining explanation —
as the cause.

**Note:** this thread also produced the sanctioned 2D qualitative panel
(`WATERFALL_SPEC_2D.md`, `frame_grid` + `frame_trails`), since a literal 1D waterfall cannot be
drawn for a 2D raster.

---

### 2026-08-11 — Probe gradients on the *input* are adversarial, not semantic · `observed`

**Evidence:** `scratch/2026-08-11-input-grad-steering.md` ·
`notebooks/experiments/editability/input_grad_steering/` · transformer `runs/transformers/W16`,
GRU `runs/controls/H256`, dataset 4, N=64 edits, K=15.

Steering the **input** (rather than `h`) until a probe reads the target: the optimization never
fails, but the direction does. Readout residual driven 3.4 → 0.06–0.19 (transformer) and
3.45 → 0.25–0.42 (GRU), while cos(δ*, Δ_true) is only +0.21…+0.27 (74–78°, transformer) and
+0.09…+0.13 (82–85°, GRU); shuffled-pair chance ≈ 0. Visually: broadband adversarial fuzz with
the ghost bump untouched. Edit Index −0.69 → −0.50 (transformer) and −0.68 → −0.44 (GRU), at
fidelity ≈ 1.0 — **ignored, not destroyed**.

**The oracle on the same write surface** (newest frame ← clean edited render) gives transformer
**+0.27**, GRU **−0.01**. So the surface works; the gradient's *content* does not. A transformer's
observation buffer is a stronger write channel than one GRU recurrent update.

**Why it matters:** extends `readable ≠ controllable` from `h`-space into input space — even on
a surface that is fully on-manifold-parameterizable.

**Caveat:** the same-day probe-standard fix (`STD_EPOCHS` 30 → 300) means MLP R² values in the
original note under-read; **linear R² and every steering result are unaffected** (all steering
probes are linear).

---

### 2026-08-05 — `readable ≠ grabbable` is inherited from the renderer, not learned · `replicated` ★-candidate

**Evidence:** `scratch/2026-08-05-observation-space-geometry.md` ·
`directions/orthogonal-edits.md` · N=2000, measured **in observation space with no model
involved**.

Every previous `readable ≠ grabbable` result was measured inside a trained model. This measures
the same geometry in the raw observation space:

- The linear probe is weak **as a property of the map, not the fit**: linear R² **0.259** vs MLP
  **0.754** on the same inputs. Per coordinate, a linear map keys on brightness (reflectivities
  0.4 vs 0.8) to tell the two plateaus apart, so the dimmer object is nearly unreadable.
- The required change is **orthogonal to what injection can apply**, with the shuffled control
  and chance-level row-space fraction reported alongside.

**Why it matters:** this relocates the thread's central negative **from the models to the
world**. A linear probe reads an object's *plateau*; moving the object changes something the
probe direction barely touches. The models are not failing to learn a grabbable code — the
observation function does not present one.

**Trap recorded:** the repo's `_fit_mlp` is tuned for `h`-vectors and does **not** converge on
position targets without standardising them (returns R² ≈ −0.5, worse than the mean). That is a
failed fit, not a result.

---

### 2026-08-05 — The linear position code is 116 dimensions, and writing to all of it still does not edit · `established`

**Evidence:** `scratch/2026-08-05-iterative-probing-position-dimensionality.md` ·
`notebooks/experiments/editability/iterative_probing/` · GRU `runs/controls/H256` · 78,000
aligned states from 2,000 sequences (split **by sequence**) for Part 1; N=256 edits for Part 2.

Fitting a linear position probe, projecting its 4-dim row space out of `h`, and repeating until
chance: **29 probes, every one exactly rank 4 → 116 dimensions.** Rank and orthogonality
(max |inner product| < 1e-6) are **asserted at every step**, not assumed — `lstsq` returns the
minimum-norm solution, so each new probe's rows land inside the row space of the already-deflated
design matrix.

The decay is gradual, so the dimensionality is threshold-dependent:

| dims removed | 0 | 24 | 44 | 68 | 88 | 112 |
|---|---|---|---|---|---|---|
| position R² (held-out) | 0.822 | 0.479 | 0.236 | 0.091 | 0.049 | 0.020 |

Half the readability is gone by **24 dims** (6 probes) — the core — with a long thin tail to 112.

**And writing to the whole thing still does not edit** (Part 2). This is the cleanest refutation
of "the edit fails because we only write a 4-dimensional slice".

**Why it matters:** eliminates the last "not enough of the code" explanation, and gives the
position code a measured size for the first time.

---

### 2026-08-05 — Metric correction (whitening) is real and is not enough · `replicated`

**Evidence:** `scratch/2026-08-05-metric-corrected-edits.md` ·
`notebooks/experiments/editability/metric_corrected_edits/` · GRU `runs/controls/H256`, bank =
78,000 aligned `test` states, N=256 held-out edits.

Sevan's hypothesis: a least-squares probe is `W = Σ_ph Σ_hh⁻¹`, so **the probe's row space is
the true displacement direction `J` whitened by `Σ_hh`** — and the un-whitened write `Σ_hh W`
should point closer to what the state actually does.

- **Gate passed decisively:** `Σ_hh` condition number **1.79e4** (λ_max 4.72, λ_min 2.6e-4).
- **Direction improves substantially** — cosine to the true displacement jumps to **+0.236**.
- **Still not an edit:** Edit Index **−0.51** vs pseudoinverse ≈ −0.66. This is **the best
  training-free structural editor the thread has produced**, at zero fidelity cost, and it is
  still deep on the wrong side of zero.
- **Magnitude is not the missing ingredient:** the α sweep has a genuine optimum at ×2
  (**−0.33**, 25% of the oracle's gain); beyond it the index rises only by degrading (×8 reaches
  +0.01 by wrecking the frame).
- **The local metric is worse:** local `Σ_hh` from 1024-NN gives cos +0.143 at α=1 vs global
  +0.236. Its Edit Index (−0.38) *looks* better than global (−0.51) — read this against fidelity
  before believing it.

**Why it matters:** confirms the mechanism (the probe direction is a whitened version of the
right one) while showing the correction is partial — which is exactly the gap the learned editor
later closes.

---

### 2026-08-05 — Tangent-constrained injection: a new direction, still no edit · `observed`

**Evidence:** `scratch/2026-08-05-tangent-constrained-injection.md` ·
`directions/orthogonal-edits.md`.

`Wᵀ Δ = δ` is 4 equations in 256 unknowns, so its solution set is a 252-dimensional affine
subspace; plain injection takes the **minimum-norm** member (a fixed 4-d subspace chosen by the
probe with no reference to where the state is). Choosing a different member — one constrained to
the local tangent — gives a genuinely different direction and still does not produce an edit.

**Process note, recorded because it cost real time:** this was analysed through four scalar
figures before a waterfall was added, and only the waterfall showed that the apparently
"successful" arms were generating vertical-stripe garbage. This incident is the origin of the
standing rule that any claim about generations ships with the qualitative panel.

---

### 2026-08-03 — A successful edit is large, edit-specific, and invisible to the probe · `established` ★-candidate

**Evidence:** `scratch/2026-08-03-delta-h-analysis.md` ·
`notebooks/experiments/editability/delta_h_analysis.ipynb` · GRU `runs/controls/H256` and RSSM
`runs/rssm/4_dset4_refined_best`, dataset 4, `ef=20`, K=15, **N=256** held-out edits (previous
version of this measurement was N=64, GRU only, one construction).

Using the two mechanisms that reliably work — counterfactual state overwrite and freeze-time
teacher forcing — as ground truth for what a successful edit *is* as a latent displacement
`Δh = h_post − h_pre`:

A successful edit is **roughly as large as the state itself**, points in a direction
**essentially orthogonal to everything the position probe can see or move**, is **different for
every edit** — even for edits making the *same* positional change — and **does not generalise
when learned**. The two independent oracles nevertheless **agree strongly on which direction it
is**, so the target is well-defined even though no probe-derived rule finds it.

**Why it matters:** turns `readable ≠ controllable` from an observation into a **measured
reachability ceiling**, and supplies the quantitative target that the later trained editor aims
at.

**Caveats:** the GRU/RSSM alignment convention is verified on ordinary sequences — GRU passes
cleanly, but the **RSSM is ambiguous at this precision** (k=−1 0.1059 vs k=0 0.1067, a 0.8% gap)
because its prior decode is blurry. Treat the RSSM Δh numbers with that caveat.

---

### 2026-07-30 — Editing at the encoder port: the interface matters, and the write still repaints · `observed`

**Evidence:** `scratch/2026-07-30-encoder-space-editing.md` ·
`notebooks/experiments/editability/controls/encoder_editing.ipynb` · GRU `runs/controls/H256`,
`ef=20`, 64 edits. Origin: Michael's controls.

Every editor to date writes to `h`, the *accumulated* state. The encoder output
`x_t = relu(W_enc·obs_t + b_enc)` is the model's **input port** — the representation the
recurrence was trained to accept every step — so a write there is in-distribution by
construction. Editors in `x`: readout injection, global-PCA projection, PCA geodesic, MLP-probe
gradient; bracketed by a freeze-time render oracle and the `h`-space injection baseline.

**Result:** the interface genuinely matters — and the write is still **repainting rather than
relocating**. The freeze-time win does not survive replacing the renderer with a latent write at
the port.

**Why it matters:** rules out "we were writing to the wrong surface" as the explanation, and
sharpens what freeze-time teacher forcing is actually providing (externally rendered,
velocity-consistent evidence over time — not a better write location).

---

### 2026-07-30 — The editability negative is capacity-independent · `established`

**Evidence:** `scratch/2026-07-30-hidden-size-sweep.md` ·
`notebooks/experiments/editability/controls/hidden_size_sweep.ipynb` (+ `CONTROL_RUNS.md`) ·
five GRUs, `hidden_size ∈ {8,32,128,256,512}`, one variable, `datasets/4_fixed_refl_inview`,
2000 test sequences for probes, 64 edits, `ef=20`, K=15.

Reference points: the world's true state is **8 numbers**; the observation is **128 rays**.

**Capacity moves predictive quality and readability a great deal, and grabbability not at all.**
Prediction saturates by `H=128`. Linear readability rises *monotonically* with capacity —
**refuting the pre-registered guess** that a squeezed latent would be more linearly readable. At
**every** hidden size the probe-directed editors leave the ghost in place (or remove it only by
wrecking the rollout) while the decoder-gradient oracle clears it on the same model and decoder.

**Why it matters:** kills the "256 dimensions for an 8-dimensional world" explanation. Extended
to action-conditioned families on 2026-08-13 (above), where the decay with capacity is sharper.



### 2026-07-16 — A multi-step (rollout) TRAINING objective does not induce editability (GRU) · `established`
Tested whether the editing failure is an artifact of the pure next-step training loss. Trained GRUs with a
**free-running multi-step rollout objective** (teacher-force context, then free-run `w` steps feeding the
model's own decoded predictions, BPTT through the whole imagination, MSE on all `w` frames), `w∈{1,2,5}`,
same architecture/data/hidden/epochs — only the objective changes. Data `datasets/4_fixed_refl_inview`
(**noisy**, `obs_noise_std=0.2`), `w=1` = the standard single-step baseline (`runs/gru/7_dset4_gru_400epochs`);
`w=2,5` = `runs/gru_multistep/w{2,5}_dset4_gru_400epochs`.
**Result — clean NEGATIVE.** The multi-step objective does what it's designed to on the *rollout* (open-loop
horizon RMSE 0.208→0.197→0.188; rollout total-variation sharpness moves *toward* GT, 1.28→1.07, **not** below
— so **no blurry mean-hedging / mode-collapse**) but **buys no editability and no identifiability/canonicality**:
the whole §4 pathology — a **decoder-inert** position-probe direction, **belief sluggishness** (even the
true-state swap moves the obs only ~0.12), and **off-manifold oracle collapse** — replicates essentially
unchanged across `w`. No non-oracle editor approaches the true-state swap for any `w` (best-editor GT
next-step RMSE ≈ Unsteered, ~0.27, vs swap ~0.20). The PCA-geodesic even drives the *readout* progressively
lower with higher `w` (1.20→0.99) while obs/ghost/next-step barely move → the state↔observation **decoupling
is structural, not a budget/geometry artifact**. If anything the objective mildly *degrades* canonicality
(MLP fiber residual 0.357→0.382→0.457; position-linear R² 0.84→0.82→0.76; linear hull + curvature inflate).
**Reading:** the editing difficulty here is a **structural** property of the learned code (decoder-inert probe
direction + single-frame belief inertia), **not** an artifact of a next-step-only loss that a rollout
objective would fix — refuting the "coherence-under-iterated-dynamics ⇒ editable state" intuition for this GRU.
*Scope:* this GRU family, dataset 4 (noisy), `w∈{1,2,5}` only; probes in-sample (cross-`w` deltas are the
load-bearing quantities). **RSSM replication — DONE (2026-07-16, scratch, pending Sevan's artifact-or-signal
review before folding in):** the negative **replicates** on the RSSM (latent-overshooting objective, `W∈{2,5}`;
no editor reaches the true-state swap) AND the objective is *additionally harmful* there — it blurs the decoder
(rollout TV/GT 1.23→0.43, objects fade), worsens single-step + open-loop prediction, collapses the linear hull
(36→10 dims), and reduces linear readability + canonicality. So for the architecture built for multi-step, the
objective buys no editability and costs predictive quality. See `scratch/2026-07-16-multistep-objective-rssm.md`.
Notebook `notebooks/experiments/editability/multistep/multistep_objective_structure.ipynb`; training helper
`scripts/train_gru_multistep.py`; note `scratch/2026-07-16-multistep-objective-structure.md`. *(Two metric
caveats noted in that thread — not affecting this result: the curvature/tangent-rotation number is not
distance-normalised, and any static-target-render metric inflates as the object moves; §4 here uses the
time-evolving clean GT.)*

### 2026-07-08 — Summary rewritten; velocity is instantaneously NONLINEAR, not temporal · `established`
The current-understanding summary now leads with **non-canonicality / readable≠controllable** (from
the 2026-06-24 keystone), replacing the superseded "target unreachability under manifold constraint."
**Velocity correction (resolved, `editability/diagnostic_corrections.ipynb`):** the keystone's
"velocity is a temporal feature" was a confound — it compared single-frame **linear** (0.47) against
2-frame **MLP** (0.76), changing two axes at once. The 2×2 {linear,MLP}×{single,2-frame} on both GRU
and RSSM shows single-frame MLP ≈ 2-frame MLP (Δ ≤ 0.007 late-t both models; GRU single-frame MLP R²
**0.94** late-t), and `dh` differencing is *worse* than single-frame. **Velocity is instantaneously
readable from one `h_t`, just nonlinearly** — the entire 0.47→0.76 gap is the linear→MLP axis, not
single→temporal. "Velocity is a temporal feature" is **RETIRED**. *Strategic:* this reframes the
planned dynamics-identifiability thrust — velocity lives in the **state** (nonlinear/entangled
coordinate), not deferred to the transition.

### 2026-06-24 — Supersedes "target unreachability under manifold constraint." The target IS substantially reachable on-manifold (geodesic constant-step RMSE→0.35). `established`
The real barrier is that the state is non-canonical: position is readable but velocity is not an instantaneous coordinate (lives in the dynamics), the (pos,vel)→h embedding is strongly curved, and ~35% of h is not a function of (pos,vel). Completing the edit target to (pos,vel) does not fix editing; the only h that renders the target is off-manifold and the dynamics reject it. "Readable ≠ controllable."

### 2026-06-23 — Matched-magnitude sweep overturns "decode≠generate" · `established`
σ along directions: probe obj0-x=0.26, probe obj0-y=0.22, PCA#1=2.23, PCA#2=2.22 —
PCA dirs have ~10× larger data-σ. The earlier "probe ≈ random ≪ PCA" result was a
**confound of σ-scaling**. At matched absolute ‖Δh‖=4, the probe direction produces
*more* RMS observation change than PCA or random. Decoder-Jacobian corroborates:
probe projection onto the top-8 decoder-sensitive singular subspace = 0.086 vs
random 0.034 (probe 2.5× better aligned). **The probe direction is generative.**
Caveat: the relationship is nonlinear (divergence mainly at large magnitude), and
realistic probe edits are 10× smaller in σ units — trust the Jacobian numbers over
the high-magnitude tail of the sweep.

### 2026-06-23 — Edit diagnostic table + reversion vs drift · `established`
| edit | readout RMSE | global resid | local resid |
|---|---|---|---|
| real states | — | 1.72 | 0.91 |
| pseudoinv | 0.00 | 1.80 | 1.06 |
| manifold (global PCA) | 0.02 | 0.00 | 1.59 |
| local tangent PCA | 0.62 | 1.86 | 0.93 |

Observation change vs unsteered (swap baseline = 0.317 = full state change):
pseudoinv 0.030 (9.5%), manifold 0.119 (37%), local 0.109 (34%).
Reversion vs drift: **pseudoinverse reverts** to the unsteered trajectory by step
~14 (dynamics project the off-manifold edit away); **manifold/local persistently
diverge** from unsteered but do *not* track the intended GT target — they go
elsewhere on the manifold. Local tangent: local_resid≈0.93 (genuinely on-manifold)
but readout RMSE=0.62 after 50 POCS iters → **the target is not reachable in the
local neighborhood**. → core diagnosis: target unreachability, not non-generative
direction.

### 2026-06-23 — Per-sample heterogeneity + ghost objects · `tentative`
Macro averages obscure strong per-sample structure. Many individual edits *do*
move the decoded position toward the target and persist; others revert or drift —
the mix dilutes aggregates. Qualitatively (waterfalls), the global manifold edit
sometimes places the object correctly (sample 156: bar goes to the correct far-left
GT position) but spawns a **phantom object at the original location** — incomplete
identity displacement. Object-identity *swaps* also observed. *Why tentative:* read
from individual plots, not yet quantified by a stratified per-sample metric.
*Next:* stratify by edit success and characterize what distinguishes persistent
edits from reverting ones.
