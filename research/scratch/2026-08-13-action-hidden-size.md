# 2026-08-13 — Hidden-size ablation on ACTION models → **FLAG FOR PROMOTION**

> `scratch/` is ungated — nothing here is "true" yet. Promotion to `findings/` is Sevan's call.

Sevan: *"run an additional ablation of hidden state sizes but all on the action models and make sure
that our findings replicate or are proven to change."* Notebook
`notebooks/experiments/editability/action_hidden_size/action_hidden_size_sweep.ipynb` (7 code cells,
10 figures, 0 errors); registry `ACTION_SWEEP_RUNS.md`; driver `scripts/train_action_hidden_sweep.sh`;
harness `scripts/eval_action_sweep.py`. Branch `rogerio_controls`. **15 new runs**, ~90 min GPU.

## Design

Two action families were chosen to differ on the axis that matters — **whether the model's own action
space contains the intervention**:

* **exogenous teleport** (`XG_A_H*`) — GRU conditioned on continuous *teleport-to-absolute-coordinate*
  actions (`datasets/7_cont_teleport`). Its action space **contains the edit under test**, so it ships
  with a **built-in ground-truth handle**: "issue the action" is a positive control no passive model
  can offer. Plus `XG_C_H*`, identical data and recipe with **actions withheld** (action-knowledge
  control).
* **endogenous interactive** (`EN_H*`) — `EndogenousActorGRU` level 3: forces, death on object/wall
  collision, **REINFORCE survival reward into the same GRU trunk that predicts**. Its actions are
  forces, so it **cannot** teleport at any capacity → no action-interface arm, by construction.

`H ∈ {8, 32, 128, 256, 512}`, hidden size the **only** variable within a family. The **passive GRU**
(`runs/controls/H*`) is recomputed with the identical estimator so all four curves are comparable.

## Results

**F1 — prediction saturates by H≈128. REPLICATES everywhere.** Next-step RMSE vs clean:
passive 0.1499→0.1040 · XG_A 0.1681→0.1071 (0.1079 @512) · XG_C 0.2016→0.1772 · EN 0.2080→0.1553.
Action knowledge is worth a large constant (0.1071 vs 0.1772 @256 — teleports are unpredictable
without the action) but does not move *where* the curve flattens.

**F2 — readability rises. REPLICATES for both exogenous families; PARTLY CHANGES for endogenous.**
Linear position R²: passive 0.155→0.834 · XG_A 0.195→0.786 · XG_C 0.225→0.711, all monotone. **EN
rises 0.274→0.636 @256 then FALLS to 0.546 @512**, velocity 0.496→0.196 over the same step. The honest
reading is **under-training, not the objective**: every EN run got the same 6000 iterations and the RL
arm has the most to learn per parameter. Flagged, not explained.

**F3 — canonicality moves the opposite way. REPLICATES everywhere.** MLP fiber residual rises with
capacity in all four: passive 0.288→0.637 · XG_A 0.410→0.695 · XG_C 0.317→0.710 · EN 0.270→0.500.

**F4 — grabbability. REPLICATES, and is STRONGER than "flat".** The legitimate gain of readout
injection over its own unsteered row **shrinks toward zero as capacity grows**:

| family | H=8 | H=32 | H=128 | H=256 | H=512 |
|---|---|---|---|---|---|
| Passive GRU | +0.479 ⚠ | +0.181 ⚠ | +0.016 | +0.008 | +0.004 |
| Exogenous · actions withheld | +0.382 ⚠ | +0.234 ⚠ | +0.025 | +0.026 | +0.016 |
| Exogenous · actions given | +0.419 ⚠ | +0.278 | +0.026 | +0.020 | **+0.007** |
| Endogenous L3 | +0.194 | +0.057 | +0.001 | +0.001 | **−0.001** |

⚠ = fidelity ratio > 1.05. **Every apparently large gain at H=8–32 is degradation** (fidelity 2.3–3.1;
the H=8 waterfall shows saturated white bands, not a relocated object). By H≥128 the editors are
**inert** instead (fidelity 1.00, Edit Index sitting exactly on the unsteered line). The two failure
modes trade places as capacity grows and neither is an edit.

**The built-in handle is what makes it decisive.** Over exactly the range where latent editing decays
to nothing, the **action interface rises**: **+0.216 → +0.455 → +0.582 → +0.618 → +0.608** at fidelity
0.71–0.83. The decoder-gradient oracle rises the same way in every family (XG_A +0.521 → +0.984).

## F4 in time — added 2026-08-14 (Sevan asked why the by-step curve was missing)

The §4 scorecard scores the Edit Index at **step 0**. `METRICS_AND_EDITORS.md` requires the by-step
curve wherever the step-0 index is reported, and it was missing here because
`eval_action_sweep.py` **stripped every list when serialising its JSON** (and the notebook stripped
them a second time when copying the passive family's published editability). Both fixed; the eval
re-run for all 15 runs. New notebook §4 / Fig 4.

**The negative is not confined to step 0 — the injection curve never separates from unsteered at
all.** Gap between readout injection and that model's OWN unsteered curve at step 14:

| family | H=256 | H=512 |
|---|---|---|
| Passive GRU | −0.002 | −0.001 |
| Exogenous · actions withheld | +0.007 | +0.006 |
| Exogenous · actions given | +0.052 | +0.007 |
| Endogenous L3 | +0.002 | −0.001 |

(post-GT-fix values) — indistinguishable from the curve it is supposed to be steering away from. The apparent rise of the injection curve over the rollout is
entirely the **unsteered baseline climbing** (a free-running model drifts away from *both* reference
worlds) — the edited and unedited curves lie on top of each other for all 15 steps.

**CORRECTION — capacity does NOT buy persistence.** An earlier version of this note read the *gap to
unsteered* at step 14 and concluded the oracle holds edits better at larger `H`. That was an artefact
of the statistic. The **raw** step-14 index of the decoder-gradient oracle is flat and near zero at
every capacity, in every family:

| family | H=8 | H=32 | H=128 | H=256 | H=512 |
|---|---|---|---|---|---|
| Passive GRU | +0.228 | +0.089 | +0.106 | +0.023 | +0.165 |
| Exogenous · actions withheld | +0.045 | +0.088 | +0.089 | +0.124 | +0.141 |
| Exogenous · actions given | +0.141 | +0.135 | +0.084 | +0.078 | +0.070 |
| Endogenous L3 | +0.018 | −0.018 | +0.049 | −0.063 | −0.021 |

It starts at +0.45…+0.99 and **reverts to ≈0 within 15 steps everywhere** — reproducing the published
decoder-gradient behaviour (+0.94 → +0.08) and visible in the waterfall, where the column jumps to the
target then drifts back and streaks. The *gap* grows with `H` only because the **unsteered** index
falls as the predictor improves. **Sevan caught this from the figure** ("in my other experiments the
standard GRU with decoder gradient reverted strongly") before the numbers were re-checked.
*Methodological rule now in `CLAUDE.md`:* gap-to-unsteered answers "distinguishable from doing
nothing"; only the **raw** curve answers "does the edit hold".


## 2026-08-14 (review round 2) — Sevan's figure + methodology review; every value recomputed again

**Data fix (the big one).** The edit episodes still carried the world's own random teleports *before*
the edit frame. Even with a clean scored horizon that is wrong: it makes the dataset-7 episodes
structurally unlike the dataset-4 control's, so the cross-model comparison the control exists for was
invalid, and the stray jumps showed up in the waterfalls. **Both edit sets are now generated with
`--p-action 0.0`** (`datasets/15_teleport_eval_single`, `datasets/16_teleport_edittrain_single`) and
the single teleport is synthesised; `eval_action_sweep.xg_data` **asserts** the set is
intervention-free. All 18 trained-editor arms were retrained on the new pool and every evaluation
re-run. *(One correction to the framing: contamination in the context does **not** make the
unsteered-vs-edited comparison unfair — those arms share the same episodes, so it is paired. What it
broke was comparability **across model families**.)*

⚠ **Consequence worth carrying:** F1 is now measured on a world where nothing unpredictable happens,
which is exactly what the action-conditioned model is told about and the observer is not. The
observer's next-step RMSE at H=256 is **0.1234** here vs **0.1769** on the training distribution, so
the action-knowledge advantage reads **0.017** instead of **0.070**. The saturation *shape* is
unchanged. Measuring F1 on the training distribution while keeping edits single-intervention is the
clean complement and is not yet done.

**Figure fixes** (all from Sevan's review): Fig 4's legend no longer runs off the figure and drops the
ALL-CAPS; Fig 1 labels its dotted lines (each family's noise floor) and the invented "within 2% of
best" threshold is gone; Fig 2 reports **late-t** readouts; Fig 3's passive column was silently
dropping the decoder-gradient oracle (name alias) and now shows it; Fig 4's unsteered baselines are
coloured per hidden size with a legend entry; waterfalls now cover **all four families including the
passive GRU, 3 rows each**.

**Velocity R², explained.** Not a bug: the older high numbers come from holding out individual
*frames* rather than whole *trajectories*. Same 2×256 probe, same late-t window, only the split
changed: **velocity 0.565 (trajectory holdout) vs 0.905 (frame holdout)**, against **0.924 vs 0.971**
for position. Velocity is constant within a trajectory, so a frame holdout leaves other frames of the
same trajectory — carrying the identical label — in the training set. Every pre-2026-08-06 velocity
number in the repo is on the frame-holdout convention.

## Reading (mine, not established)

**Capacity improves prediction, readability, and action-channel controllability simultaneously, while
making the latent readout channel *less* effective.** That is a dissociation, not a plateau, and it
argues against "latent editing just needs a bigger/better representation". Note the internal
consistency: the same capacity that makes position more linearly decodable (F2) makes `h` **less** a
function of the physical state (F3), and the probe-derived write correspondingly reaches less of it.

The exogenous family is the strongest control the thread has run: a model **trained to perform this
exact edit on command**, whose action channel demonstrably gets better at it with capacity — and none
of that transfers to writing the target into `h`. Read with the other two threads on this branch
(`history_editing/`: the complement is *observation-shaped* history; `full_rowspace_edit/`: handing the
editor the entire hidden state as row space changes nothing), the capacity axis is now closed too.

## ⛔ TWO MEASUREMENT BUGS FOUND 2026-08-14 — every value in this thread was recomputed

**1. The ground truth contained MORE THAN ONE intervention.** `datasets/7_cont_teleport` fires its own
random teleports on ~30% of transitions. Filtering the *edited* object's later actions (which the
first version did) is not enough — the **other** object's teleports land in the same scored window, so
GT-traj RMSE, the fidelity ratio and the by-step Edit Index were all scoring the model on events it
was never told about and could not predict, and none of it was comparable to notebooks whose edits
split carries a single teleport. **Fix:** both reference worlds are now *constructed*, by rolling the
frame-`ef` state forward under the passive (ballistic) dynamics, so the scored window contains exactly
the one intervention under test. The model free-runs without actions, so the passive continuation is
the fair target, and it matches `build_edit_zones` on the canonical splits. Caught by Sevan.

**2. Velocity R² was reported ALL-t, and the split convention differs from older notebooks.** The
registry requires the late-t split (`t ≥ 15`); reporting all-t under-read velocity badly. Separately,
this thread's numbers look much lower than older ones because `fit_readability_probes` splits **by
sequence** while `eval_controls.py` splits **by row**. Measured on `controls/H256`, same 2×256 probe,
same late-t window: **by sequence 0.565 vs by row 0.905** (+0.34 inflation), against **0.924 vs 0.971**
for position. Velocity is nearly constant within a sequence, so a row split hands the probe the
answer. **Every pre-2026-08-06 velocity number in the repo is on the leaky convention** and is not
comparable to anything here — the by-sequence number is the honest one.

## Harness work done here

* **New:** `scripts/train_action_hidden_sweep.sh`, `scripts/eval_action_sweep.py` (one pass per
  checkpoint → `runs/action_sweep/eval/<code>.json` + `_rollouts.npz`), `ACTION_SWEEP_RUNS.md`,
  held-out split `datasets/13_cont_teleport_eval` (base seed 200000, disjoint from training).
* **BUG FIXED — `scripts/gen_continuous_dataset.py` was broken.** It rebuilt `SimConfig` by iterating
  over *every* dataclass field of dataset 4's stored `sim` dict, which predates the soft-render and
  omniscient-2D fields → `KeyError: 'soft_edge'`. **No continuous-action dataset could be regenerated
  at all.** Now takes only the stored fields and lets the rest default (all default to OFF, which is
  what "matched to dataset 4" means), and prints which ones defaulted.
* **⛔ `scripts/eval_editability_endogenous.py` is STALE and was deliberately not used.** It still
  computes the metric set **retired 2026-07-30** (`reach` / `collat` / `ghost` / `select`) — the ones
  `CLAUDE.md` forbids reintroducing. Everything here goes through `editability_metrics.py`. That
  script should either be migrated or marked superseded.
* **Estimator mismatch caught:** `eval_controls.py`'s readability block splits **by row** (leaking
  neighbouring frames) and uses a **1×128** MLP, so its published F2/F3 numbers cannot be plotted
  beside `fit_readability_probes` results. The passive family is therefore **recomputed** here with
  the standard probes; it reproduces the published trends (next-step 0.1499→0.1040 vs published
  0.1495→0.1042; position R² 0.155→0.834 vs 0.175→0.855), which also validates the new harness.

## Method notes worth keeping

* **The exogenous edit is synthesised, not harvested.** With `p_action = 0.30`, a quiet 15-frame scored
  window occurs with probability ~1e-5 — measured: **9 usable episodes in 4000**. So sequences with no
  action at `ef−1` keep their real context, and the teleport is constructed (target sampled in-frustum,
  encoded with the generator's own `normalize_action`). Since teleport is an *absolute* placement and
  the edited object is required to take no further action in the window, the edited world is the true
  world with that object rigidly offset for all K steps.
* **Endogenous ground truth comes from forking the simulator** at `ef` and stepping both forks under
  the same actions. In a force world `pos(t+1) ≠ pos(t) + v·dt`, so `build_edit_zones`' ballistic
  roll-forward would be wrong; `pre_vel = 0` makes it use the true unedited frame-`ef` world.

## Owed / not done

* One seed per cell. **6000 iterations for every EN run regardless of `H`** — the most likely cause of
  the `EN_H512` readability dip; a longer run should be done before that dip is read as anything about
  the objective.
* The endogenous counterfactual is **open-loop** (the actor's own policy would diverge after a
  teleport).
* `EN_*` use `obs_noise 0.2` (repo standard) so they are **not** bit-comparable to `runs/endogenous/L*`
  (known 0.05 deviation).
* No RSSM/transformer action arm; no second seed; position-only edits.
