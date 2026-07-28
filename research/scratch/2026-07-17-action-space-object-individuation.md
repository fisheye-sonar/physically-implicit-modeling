# Does the action space individuate a grabbable OBJECT HANDLE in the passive latent? — clean NEGATIVE (with a real, weaker positive on structure)

**Date:** 2026-07-17 · **Direction:** `research/directions/action-space-object-individuation.md` · **Status:** ✅ **PROMOTED 2026-07-27** → `findings/object-individuation.md` (Sevan-approved; scoped to *exogenous* actions; Exp-2 folded in)
**Notebook:** `notebooks/experiments/editability/actions/action_space_object_individuation.ipynb` (executed, 0 error cells) · **PNGs:** `/tmp/action_space_object/`
**Reframe:** not "bigger actions", not editing via the action channel. Question = does training on an interaction *affordance*
reorganize the **passive** (no-op) latent into a **separable, ghost-clearing, generalizable object handle** that a
**master §4 editor** (a *different* write-mechanism than the trained action channel) can grab — a real object vs a memorised button.
All eval is on the passive latent, dataset-4 held-out.

## Verdict

**No action space individuates a selective, generalizable, ghost-clearing object handle in the passive latent.** This is a
**clean negative on the primary readout**, and it holds for *every* affordance type (dxdy / teleport / axis_x) and the confound triad.
Concretely, with the canonical structural editor (PCA geodesic) targeting object k on the passive latent:

- **Ghost never clears.** ghost ratio = 0.90–0.93 for *all five* models (1.0 = object stays put; a real teleport / true-state-swap = 0.31–0.67;
  the decoder-gradient *oracle* = 0.09). The object k does **not leave its old location** — the #1 handle criterion fails for everyone.
- **Reach is modest and non-selective.** reach = 21–37 % of a true swap; collateral (disturbance of the OTHER object, same units) = 17–28 %;
  **selectivity ≈ 0.56–0.58** for all models (an editor moving object k disturbs the other object nearly as much). Confirmed in obs-space:
  in the waterfalls the structural editors keep the bright band on object k's **ghost** line and leave the green **target** line empty
  (`fig5*`), while the true-state-swap correctly jumps to the target.
- **Teleport is NOT necessary and dxdy/axis do NOT suffice** — none produce a handle. M_teleport (trained explicitly to clear ghosts via
  absolute placement) does **not** transfer ghost-clearing to the editor write-mechanism (ghost 0.917, same as the others).

**But a real, weaker positive is present and localizes to action-knowledge:** training on *large* affordances reorganizes the passive
latent toward a **more canonical, more linearly-recoverable** function of (pos, vel) — replicating and *strengthening* Exp-2 (whose 0.7-unit
nudge gave only a modest effect). This is a representation-quality gain, **not** editability. Better readout ≠ a grabbable handle.

## Key numbers (all passive / no-op, dataset-4 held-out)

Models: **Baseline** `runs/gru/7_dset4_gru_400epochs` (clean, no actions) · **Pert-pass** `M_teleport_ctrl` (teleport trajectories, action
withheld) · **M_dxdy** · **M_teleport** · **M_axis** (all action-conditioned). Confound triad: Baseline→Pert-pass = perturbation-diversity;
**Pert-pass→M_teleport = action-knowledge**.

**§4 object-handle scorecard — PCA geodesic, same editor for every model (Fig 4 / Fig 5):**

| model | reach %swap ↑ | collateral %swap ↓ | selectivity ↑ | ghost ratio ↓ | persistence ↑ |
|---|---|---|---|---|---|
| Baseline | 36.7 | 27.9 | 0.57 | 0.928 | 2.18 |
| Pert-pass (teleport) | 21.0 | 16.6 | 0.56 | 0.896 | 1.62 |
| M_dxdy | 28.2 | 21.9 | 0.56 | 0.922 | 1.91 |
| M_teleport | 31.5 | 22.6 | 0.58 | 0.917 | 2.04 |
| M_axis | 30.4 | 24.3 | 0.56 | 0.916 | 1.61 |
| *reference: true-state swap* | *100* | *~30* | *0.72–0.80* | *0.31–0.67* | *~1.2* |
| *reference: decoder-gradient (oracle)* | *125–232* | *35–55* | *0.77–0.81* | *~0.09* | *~1.2* |

Confound-triad deltas on the scorecard: **Baseline→Pert-pass** reach −15.8, collat −11.3, ghost −0.032, persist −0.56 (perturbation-diversity
mostly *hurts* the handle); **Pert-pass→M_teleport** reach +10.5, collat +6.0, **ghost +0.021 (slightly worse)**, persist +0.42 — action-knowledge
recovers reach lost to perturbation but does **not** buy ghost-clearing.

**§2/§3 passive structure (the real positive; Fig 2 / Fig 3):**

| model | pos R² (lin) | vel R² (lin, late) | fiber resid ‖h−g(pos,vel)‖/‖h‖ (MLP) ↓ | intrinsic dim (TwoNN) |
|---|---|---|---|---|
| Baseline | 0.839 | 0.585 | 0.395 | 4.64 |
| Pert-pass | 0.756 | 0.564 | **0.488** (worse) | 7.85 |
| M_dxdy | 0.941 | 0.695 | 0.304 | 6.02 |
| M_teleport | 0.883 | 0.690 | 0.316 | 5.74 |
| M_axis | **0.946** | **0.804** | **0.282** (most canonical) | 5.95 |

Localization: fiber residual Pert-pass 0.488 (worse than baseline) → M_teleport 0.316 (much better) → the canonicalization is **action-knowledge**,
not perturbation-diversity. Linear velocity recoverability jumps from ~0.58 (Baseline/Pert-pass) to 0.69–0.80 for the action-cond models.
(Position/velocity MLP R² are ~0.95–0.98 for all — saturated, non-discriminative.)

**Content generalization — M_axis edited along y it never trained (Fig 6):** geodesic obs-projection reach on object-k rays is x≈0.02–0.03 and
y≈0.13–0.18 for **every** model, ratio ~4–6 — including the affordance-free **Baseline** (x=0.03, y=0.17). **M_axis (x=0.03, y=0.18) is
indistinguishable from Baseline.** So (a) the y>x asymmetry is a **geometry artifact** of lateral vs depth moves shared by all models, not an
axis-specific training signature; (b) the x-only affordance created **no x-specific handle** to generalize *or* to fail — there is simply no handle.
The linear-probe coordinate is writable along both axes (readout injection drives readout RMSE→0) but this is **trivial** and produces **~0 obs
change** — writing the probe coordinate ≠ moving the object.

**Exposition (Fig E1/E2):** the affordances are perceptually **large** — mean per-event |Δobs| = dxdy 0.190, teleport 0.224, axis_x 0.193
(vs Exp-2's 0.7-nudge range 0.03–0.13); mean accepted move 1.7–3.9 world units. Change-the-action divergence (channel is causally used):
M_dxdy 0.056, M_teleport 0.105, M_axis 0.061 (all > 0). So the negative is **not** because the actions were too weak to matter — they were 2–7×
larger than Exp-2 and the channel is demonstrably used (M_teleport_ctrl's val loss 0.0396 ≫ M_teleport's 0.0245 confirms the actions carry real,
un-withholdable information).

## Interpretation

Large interaction affordances **do** carve real structure into the passive latent — a more canonical, more linearly-decodable (pos,vel)
representation, cleanly attributable to action-knowledge — but this is **representation quality, not object-hood**. The individuation test
(can an *untrained* write-mechanism grab object k, move it, clear its ghost, and leave the other object alone) **fails for every affordance**.
Even teleport, which is *by construction* ghost-clearing in the action channel, leaves the passive latent's editability unchanged. The
"objecthood" lives in the input→dynamics pathway (a button), not in the state (a handle). This is the structural-realist negative the direction
anticipated, and it **motivates explicit object scaffolding** (RESEARCH.md endgame): passive prediction + even large, ghost-clearing action
affordances are jointly insufficient to make objects first-class, grabbable entities in the latent.

## Caveats / open questions

- **In-sample probes.** §2/§3 probes are fit in-sample (identical subset per model), matching Exp-2; the *ordering* across models is the claim,
  not absolute R². 
- **N_EDIT = 48, geodesic 60 iters** (bounded for the time budget). The negative is robust across all 5 editors and both waterfall samples, but the
  scorecard absolute numbers would tighten with more edit samples. The **ghost≈0.92-for-all** result is the load-bearing, unambiguous finding.
- **Best-editor choice.** Scorecard fixed to PCA geodesic (same editor for all models) for apples-to-apples; per-model-best would flatter Baseline
  (Global-PCA reach 48 %) but at the same non-selective, non-ghost-clearing profile — does not change the verdict.
- **Content-gen y>x is a geometry artifact** (lateral vs depth), present in the affordance-free Baseline; do not read it as content asymmetry.
- **Not tried (stretch):** perturbed-passive controls for dxdy/axis; a hold/persist manipulation-type reach test; RSSM. A cleaner individuation
  probe might target a *single object's* rays with an editor that is explicitly selectivity-regularized — but the current result is that no
  affordance makes the passive latent natively grabbable.

## Substrate built (new; no existing pim/ files edited, pyproject untouched)

- `pim/simulator/actions_continuous.py` — continuous-action sim, mode ∈ {dxdy, teleport, axis_x}, writes `actions (N,T,n_obj,3)=[active,a1,a2]`
  (normalized by frustum extent), noisy obs (obs_noise_std 0.2, verified), reuses `edits_dataset._sample_in_frustum` for teleport targets.
- `pim/world_models/action_gru_continuous.py` — `ActionGRUContinuousModel`, encoder ingests `obs ⊕ ReLU(Linear(n_obj*3→16)(action))`, conforms to
  `HiddenStateModel` with action defaulting to no-op (zeros) → master/Exp-2 eval runs unchanged in passive mode.
- Scripts: `scripts/gen_continuous_dataset.py`, `scripts/train_action_gru_continuous.py` (decoupled, ~4.5 min/model).
- Datasets (90k train each, base seeds match dataset-4): `datasets/6_cont_dxdy`, `datasets/7_cont_teleport`, `datasets/8_cont_axis_x`.
- Checkpoints: `runs/gru/{M_dxdy (val 0.0247), M_teleport (0.0245), M_axis (0.0249), M_teleport_ctrl (0.0396)}`; baseline reused `7_dset4_gru_400epochs`.

## PNG manifest (`/tmp/action_space_object/`)

figE1_magnitude, figE2_change_action, fig1_geometry, fig2_recoverability, fig3_fiber, **fig4_scorecard** (headline), fig4b_perstep,
fig5a_waterfall_Base, fig5b_waterfall_Pertpass, fig5c_waterfall_M_dxdy, **fig5d_waterfall_M_tele**, fig5e_waterfall_M_axis, fig6_content_gen, fig7_summary.
