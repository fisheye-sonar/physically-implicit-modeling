# Does training on actions induce causal/editable latent structure? — GRU (3-model)

**Date:** 2026-07-16 · **Direction:** `action-conditioned-structure` (`[reframe]`, sub-Q 2+3) ·
**Status:** → **FLAG FOR PROMOTION** (nuanced result: identifiability/canonicality YES, editability NO)
**Author:** orchestrator-reconstructed from the executed notebook's verified outputs — *the worker built the
pipeline + dataset and launched the full run but backgrounded the nbconvert and stopped before writing this
note; I adopted the run (0 error cells) and reconstructed the note from its printed tables + figures.*

## The question (reframed by Sevan)
NOT "can we steer objects through the action channel." The test: does **training on random discrete-token
actions that causally move the world** induce more causal/disentangled/editable latent structure — measured
by **discarding the action channel (hold it at no-op) and re-running the master §1–§4 suite on the passive
model**? Three GRUs trained on **byte-identical trajectories** isolate two effects:
- **1→3 (perturbation-diversity):** baseline vs perturbed-passive — the world just jitters more.
- **3→2 (action-knowledge):** perturbed-passive vs action-conditioned — the model is *told* the cause.
The enactivist / Merleau-Ponty prediction is that the **3→2** gap carries the effect.

## Setup / provenance
- **Data:** `datasets/5_action_augmented` (train 90k / val 10k, T=40, R=128, 2 obj; new `actions` field
  `(N,T)`), built to match dataset 4. Discrete tokens `{no-op, obj0±x/±y, obj1±x/±y}`; no-op dominant,
  nudges sparse; a token applies a small **persistent** position nudge (frustum/collision-guarded → else
  no-op). Substrate (new modules, nothing existing modified): `pim/simulator/actions.py`,
  `pim/world_models/action_gru.py` (conforms to `HiddenStateModel` at no-op → master suite runs unchanged).
- **Models (256 hidden, 400 ep each):**
  - **(1) Baseline** — reused `runs/gru/7_dset4_gru_400epochs` (dataset 4, no nudges, no action channel).
  - **(3) Perturbed-passive** — `runs/gru/9_perturbed_passive_gru_400ep` (same nudged trajectories, token WITHHELD).
  - **(2) Action-conditioned** — `runs/gru/8_action_cond_gru_400ep` (token fed in; evaluated passively at no-op).
- **Eval:** all metrics on the **passive/no-op** models. Edits split (first 64), teacher-forced test split
  (10k) for §1–§3. In-sample probes → the **cross-model deltas are the load-bearing quantities**, not the
  absolute R²/residual.
- **Notebook (executed, 0 error cells, 22 cells):** `notebooks/experiments/editability/actions/action_conditioned_structure.ipynb`.
- **Figures:** `/tmp/action_conditioned/` — fig1_geometry, fig2_recoverability, fig3_fiber, fig4_editor_metrics,
  fig5{a,b,c}_waterfalls_{baseline,perturbed,action}, fig6_summary.

## Headline
**Action-training modestly but consistently improves the passive latent's IDENTIFIABILITY and CANONICALITY,
and the gain is localized to action-KNOWLEDGE (3→2), not perturbation-diversity (1→3) — as the enactivist
hypothesis predicts. BUT this did NOT translate into latent EDITABILITY: the existing latent editors still
fail to move the observation to the target on all three models — "readable ≠ controllable" persists.**

## Results (model 1 baseline / 3 perturbed-passive / 2 action-cond)

**§2 Recoverability — the clearest positive (LINEAR readability):**
| metric | (1) baseline | (3) perturbed | (2) action | 3→2 (action-knowledge) |
|---|---|---|---|---|
| pos R² **linear** | 0.838 | 0.842 | **0.890** | +0.048 |
| vel R² **linear** (late) | 0.582 | 0.580 | **0.659** | +0.079 |
| pos R² MLP | 0.974 | 0.971 | 0.973 | ~0 (saturated) |
| vel R² MLP (late) | 0.965 | 0.943 | 0.959 | +0.016 |

→ Position and velocity become **more *linearly* readable** in the action-conditioned latent (pos 0.84→0.89,
vel 0.58→0.66), while perturbation alone (1→3) barely moves them. MLP readouts are saturated (~0.97) and hide
this — the signal is in the *linear* coordinate. (Fig 6a's saturated-MLP bars understate this; trust the linear row.)

**§3 Canonicality / fiber-collapse — second positive:**
| ‖h−g(pos,vel)‖/‖h‖ | (1) | (3) | (2) | 3→2 |
|---|---|---|---|---|
| linear g resid | 0.880 | 0.863 | **0.804** | −0.059 |
| **MLP g resid** | 0.379 | 0.397 | **0.324** | **−0.073** |

→ The action-conditioned `h` is **more a function of the 8-dim physical `(pos,vel)`** (most canonical on both
linear and MLP). Again localized to 3→2; perturbation alone (1→3) slightly *worsens* it (0.379→0.397).

**§1 Geometry:** intrinsic dim 4.83 / 6.61 / 5.57; curvature 19.7° / 22.6° / 19.7°. Perturbation inflates
dim+curvature; action-knowledge pulls both back toward baseline. Neutral-to-mildly-positive.

**§4 Editing head-to-head — the null (editability did NOT improve):**
- **No non-oracle editor reaches the true-state-swap ceiling on ANY model.** GT next-step RMSE for every
  structural editor sits at the Unsteered level (~0.27–0.28) vs the true-swap (0.14–0.21); ghost stays ~0.93–1.0.
  Best structural obs-change (% of *that model's own* swap): baseline 79% (Global-PCA), perturbed 40%, action
  44% (PCA-geodesic) — none clean; ghost persists (Fig 5c: object stays near the pre-edit ghost line).
- **Belief-inertia is the one §4 axis that moves — via perturbation, not action:** the **true-state swap**
  itself gets stronger with perturbation training — obs-change 0.121→**0.202** (3) / 0.162 (2); ghost
  0.680→**0.347** (3) / 0.463 (2); GT-RMSE 0.207→**0.143** (3) / 0.174 (2). Reading: models that saw
  *unexplained* discontinuities (3) learned to trust single-frame evidence more (less sluggish belief update);
  the action model (2) at no-op attributes discontinuities to the (now-absent) action, so it partly reverts to
  baseline sluggishness. This is a **coherent-rollout / belief-update** effect, not editability.
- Oracle **decoder-gradient** still nails step-0 obs only by going off-manifold (loo resid ~0.99–0.81, readout
  RMSE 6.6–7.4) and collapsing — unchanged across models.

**Secondary — action-channel editing (completeness):** driving the tokens produces large obs-change
(~169% of a swap in magnitude) but only **55% of samples move toward** a specific frustum-spanning teleport
target (mean Δ(RMSE)=+0.012) — the small persistent nudge shifts the scene but cannot controllably reach a
large target in free-run. Expected per the brief; not the payoff.

## Reading (interpretation — calibrated)
Training on actions **does** change the passive representation in the hypothesized direction: it makes the
physical state more *linearly identifiable* and the hidden state more *canonical* (a cleaner function of
`(pos,vel)`), and — importantly — this is attributable to the model **knowing the cause** (the 3→2 gap), not
merely to seeing a more diverse/discontinuous world (the 1→3 gap). That is a genuine, if modest, positive for
the enactivist framing and for sub-Q2 (identifiability).

**But the affordance we actually care about — causal editability — did not follow.** The existing latent
editors fail on the action-trained model just as they do on the baseline. So on this evidence, the canonicality
gain from action-training is **necessary-direction but not sufficient-magnitude** to unlock latent editing:
canonicality moved ~0.05–0.07 while the editing gap is much larger. This *sharpens* rather than confirms the
"editability ⟹ canonical" story — a small dose of the ingredient did not buy the affordance. Separately, the
belief-inertia reduction from *unexplained* perturbations (model 3) is an independent, interpretable
coherent-rollout effect worth its own note.

## Validity checks + exposition (added 2026-07-16, review round)
- **No noise confound (checked).** The action training data (`datasets/5_action_augmented`) was generated
  from `SIM4` at `obs_noise_std=0.2` — the **same noise as dataset 4** (verified: identical noise signature).
  So models 2/3 are noise-matched to the baseline (model 1 = `7_dset4`); the improvements are not a
  clean-vs-noisy artifact. (The `obs_noise_std=0.0` in the notebook is only the exposition/edit render.)
- **The action channel is causally used — not a trivial leakage bypass (item-12 check).** A new
  exposition section (**E1–E3**, inserted before §1) shows: (E1) a demo trajectory with random tokens and
  each token's observation-space effect; (E2) a **change-the-action sanity check** — identical input up to
  `t0`, then feed no-op vs a nudge token → the model's predicted rollout **diverges** (mean|Δobs| > 0, e.g.
  ~0.03 for obj0+x, ~0.13 for obj1+x on the demo) → the model genuinely conditions on the action; (E3) a 2D
  world GIF. Because the nudge perturbs the *world state* and the scene is re-rendered (the ray-shift depends
  on the object's depth via the frustum), the model cannot predict the next frame from the token alone — it
  must bind the action to *which* object is *where*. A **shallow "token→local band-shift" shortcut is only
  partially available** (bounded by the depth-dependence); this is a plausible contributor to why the effect
  is modest, and a reason a **larger/compositional nudge** might strengthen it.
- **Perceptual magnitude:** the 0.7-unit nudge is small relative to a full frustum-spanning edit teleport —
  consistent with the action-channel edit falling short of a big target (secondary result). Figures:
  `/tmp/action_conditioned/{expo_actions_obs,expo_change_action}.png`, `action_demo.gif`.

## Caveats / open questions (for the artifact-or-signal call)
- **In-sample probes; N=64 edits.** Cross-model deltas are the trusted quantities; absolute values optimistic.
  The linear-R² and fiber-residual deltas are consistent and directionally clean, but modest — worth a rerun at
  N≈256 edits + held-out probes before promotion.
- **%-of-swap denominators differ per model** (each model's own true-state swap), so Fig 6b's cross-model
  editor bars are not apples-to-apples on the y-axis; the absolute obs-change / ghost / GT-RMSE rows are the
  fair comparison and they agree that editability did not improve.
- **Nudge magnitude / sparsity were a single choice.** A larger nudge range or denser actions might grow the
  canonicality effect (or start to genuinely disturb the passive dynamics) — a natural sweep.
- Would a **bigger** identifiability/canonicality gain (heavier action training, or actions that decompose the
  full DOF) cross the threshold into editable? The necessary-not-sufficient reading predicts a dose-response.
- **RSSM leg NOT attempted** (GRU-primary; time). Natural follow-up for architecture-independence.

## Pointers
- Substrate (new): `pim/simulator/actions.py`, `pim/world_models/action_gru.py`; dataset `datasets/5_action_augmented`.
- Checkpoints (gitignored): `runs/gru/8_action_cond_gru_400ep`, `runs/gru/9_perturbed_passive_gru_400ep`.
- Notebook: `notebooks/experiments/editability/actions/action_conditioned_structure.ipynb` (executed, 0 error cells).
- Figures: `/tmp/action_conditioned/*.png` (8 PNGs).
