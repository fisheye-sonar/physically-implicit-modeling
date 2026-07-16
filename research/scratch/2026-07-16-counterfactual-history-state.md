# Counterfactual-History State — existence check for a clean-edit hidden state `h*`

**Date:** 2026-07-16 · **Direction:** `counterfactual-history-state.md` · `[in-frame]` sub-Q3 ·
**Notebook:** `notebooks/experiments/editability/counterfactual_history_state.ipynb` (GRU only, N=64, executed
synchronously, 0 error cells) · **PNGs:** `/tmp/counterfactual_history/{fig1_per_step,fig2_waterfalls,fig3_h_geometry}.png`

**Reference sanity check — not for promotion.** (The dominant result — a much-cleaner at-target edit state
EXISTS, and it is unreachable by a low-dim position injection — is the *expected* direction and confirms the
reachability thesis. It carries one honest nuance, logged below as an open question, but nothing here overturns
a prior finding, so I am **not** flagging for promotion. Human can re-weigh the nuance.)

## Setup
- Model `3_dset3_gru_persistentids_inview_400epochs/best_model.pt` (val_loss 0.0236), data `4_fixed_refl_inview`
  edits split, `ef=20`, `K=15`, `N=64`. All histories teacher-forced on **clean** obs (renderer reproduces
  `edits.clean_obs` exactly, max|diff|=0). Rollout↔frame alignment mirrors master §4 (step `s` ↔ frame `ef+s`;
  post-edit states consumed through `ef`).
- `h*` = teacher-force the counterfactual world (edited object back-extrapolated through target at `ef` on its
  preserved velocity; other object true history). Head-to-head vs GT (`clean_obs[ef:ef+K]`), one-frame-evidence
  `h_gt` (true-state swap), unsteered `h0`, readout-injection `h_ro`, plus a frustum shared-context `h*_shared`.

## Verdict: QUALIFIED PASS — a clean-edit state exists, but not a *perfectly* ghost-free one
`h*` decisively beats the one-frame-evidence state on every metric and places the edited object at target, but
keeps a residual ghost above the geometric floor.

| state | mean RMSE→GT ↓ | ghost s0 / s14 ↓ | target-fill s0 / s14 →1 |
|---|---|---|---|
| GT (sim floor) | 0.000 | 0.086 / 0.337 | 1.00 / 1.00 |
| **h\* counterfactual** | **0.170** | **0.229 / 0.560** | **0.949 / 1.211** |
| h\*_shared (frustum-robust) | 0.183 | 0.266 / 0.539 | 0.931 / 1.413 |
| h_gt (one-frame evidence) | 0.240 | 0.614 / 0.909 | 0.569 / 0.981 |
| h0 (unsteered) | 0.293 | 1.000 / 1.000 | 0.289 / 0.655 |
| h_readout (probe pinv) | 0.292 | 0.990 / 0.983 | 0.311 / 0.646 |

Key controls that make this honest:
- **Model free-running floor** (normal un-edited trajectories, n=256): mean obs-RMSE **0.161**. `h*` at 0.170 sits
  only **+0.009 above the model's own prediction floor** — its observation fidelity is essentially as good as the
  GRU can ever roll out. `h_gt` sits +0.079 above floor.
- **GT-floor ghost is NOT zero:** even the true `clean_obs` has ghost ratio 0.086→0.337 (the fixed ghost ray set
  gets re-entered as objects drift). So "ghost≈0" means "≈GT floor". `h*` ghost excess above the GT floor is
  **+0.144 (s0)** — modest but real, and it grows over the rollout (tracking the GT floor's own growth with an offset).
- Paired bootstrap gap (h_gt − h*) mean RMSE→GT = **0.065, 95% CI [0.050, 0.081]** — robustly > 0.

So: `h*` renders the object cleanly AT TARGET (fill 0.95 ≈ GT, vs h_gt's 0.57), at the model's fidelity floor, and
with the ghost cut from 0.61 (h_gt) / 1.0 (h0) down to 0.23 — but **not fully to the ~0.09 geometric floor**.

## h-space geometry — the reachability argument (robust, as predicted)
- ‖h*−h0‖ = **6.24** ≈ mean‖h0‖ = 6.36 → `h*` is almost a *full state-norm* away from the unsteered state.
- ‖h*−h_gt‖ = 4.27, ‖h*−h_readout‖ = 6.21.
- **Probe-aligned fraction of (h*−h0) = 0.101** — only ~10% of the h0→h* edit lies in the 4-D position-probe row
  space (the subspace a pseudoinverse position-injection can move within). The other ~90% is null-space /
  history-laden. (h_gt−h0 is even lower, 0.073.)
- Consequence: the readout injection `h_ro` reads the target position *perfectly* (readout RMSE 0.000) yet rolls
  out **no better than unsteered** (RMSE→GT 0.292 vs 0.293; ghost ~0.99). A low-dim injection cannot reach `h*`.

**This confirms:** the editing failure localizes to the **reachability of the edit map**, not to the target's
existence or to missing information — a much better state (`h*`) demonstrably exists and carries the information;
it just lives ~90% outside the injectable subspace. Ties to the ~35% history-entangled fiber / learn-to-edit
negative result.

## Frustum caveat
14.1% of samples have the back-extrapolated edited object out-of-frustum in ≥1 early counterfactual frame
(mean 0.73 / 20 frames — small). The shared-context variant `h*_shared` (real obs for the first `ef−W=10`
frames, counterfactual only for the last W=10) gives essentially the same numbers (RMSE 0.183, ghost s0 0.266,
fill s0 0.931). **The caveat does not change the verdict.**

## Open question / nuance for the human (why this is *qualified*, not a crisp tautology)
Even the ideal-information counterfactual state leaves a **residual ghost ~+0.14 above the geometric floor** that
the master notebook attributes wholesale to "belief inertia / one-frame evidence." Since `h*` was given a fully
consistent history ending with the object at target, this residual is NOT belief inertia — it is intrinsic to the
GRU's decoder/rollout from a valid post-edit state (the fidelity is at the model floor, but the *ghost-zone* mass
is not fully suppressed). Interpretation: the achievable editing ceiling is high (much better than the one-frame
lower bound) but is **not a pixel-perfect ghost-free render** — part of the "ghost" is model rollout error, not
removable belief inertia. Worth a look if the reachability story is ever quantified against an assumed-perfect
ceiling.

## Caveats on method
- One-frame alignment offset (inherited from master §4): post-edit states decode a prediction of `ef+1` while GT
  step 0 is frame `ef`; object motion per frame is sub-pixel so this negligibly affects position/ghost, but it
  slightly inflates absolute RMSE for `h*`/`h_gt` vs `h0`. Held identical across `h*`/`h_gt`, so the head-to-head
  is clean.
- GRU only (sufficient per brief). Ghost/target ray sets from single-frame sim renders at `ef-1`/`ef`.
