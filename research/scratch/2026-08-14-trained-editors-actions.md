# 2026-08-14 — Trained editors on exogenous-action models → **FLAG FOR PROMOTION**

> `scratch/` is ungated — nothing here is "true" yet. Promotion to `findings/` is Sevan's call.

Sevan's spec: two world models trained where **objects teleport during training** — one told the
teleport (actions as input), one observer — plus a **control** with no actions and no teleports; a
thorough ablation over **all** edit types including trained ones; a fine-tuning variant and an
**MLP editor taking `(h, start_pos, target_pos)`**; each trained under a **next-step** loss and a
**k-step rollout** loss; and (added mid-run) a second fine-tuning variant using the **un-whitened
pseudoinverse** from `metric_corrected_edits`.

Notebook `notebooks/experiments/editability/trained_editors_actions/trained_editors_actions.ipynb`
(7 code cells, 7 figures, 0 errors); registry `TRAINED_EDITOR_RUNS.md`; training
`scripts/train_action_editors.py`; eval `scripts/eval_action_editors.py`. Branch `rogerio_controls`.
**18 trained-editor arms** (3 models × 3 editors × 2 losses); no new world models were needed —
`XG_A_H256` / `XG_C_H256` came from the hidden-size sweep, `CTRL_H256` is `runs/controls/H256`.

## THE HEADLINE — the first probe-free latent editor in this thread to cross zero

`E_θ(h, start_pos, target_pos) → Δh`, **world model completely frozen**:

| model | Edit Index | own unsteered | **gain** | fidelity | Target RMSE | prediction cost |
|---|---|---|---|---|---|---|
| Control (no actions, no teleports) | **+0.204** | −0.671 | **+0.875** | 0.84 | 0.250 | **none — model frozen** |
| Exogenous · actions given | **+0.111** | −0.669 | **+0.780** | 0.89 | 0.279 | none |
| Exogenous · observer | **+0.117** | −0.578 | **+0.695** | 0.82 | 0.281 | none |

For scale: the previously published best *learned* mechanism (amortized `E(h, target)`) reached
**−0.14**, and the best training-free structural editor here is metric-corrected injection at
−0.42…−0.52. Target RMSE roughly halves (≈0.48 → 0.25–0.28). **Fig 3 confirms it in observation
space** — the object appears on the green target locator and the ghost dims — with visible streaking
the counterfactual oracle does not have, which is what the 0.84 fidelity measures.

**The only difference from the published editor is the extra input.** Handing the network the
*starting* positions gives it the displacement instead of making it infer the current world from
`h`. That single change moves the mechanism from −0.14 to +0.20.

## Sevan's addition: the un-whitened write is a BETTER fine-tuning target — 6/6 cells

Gains over each arm's own unsteered row:

| loss | model | pseudoinverse write | **un-whitened (Σ¹) write** |
|---|---|---|---|
| k=1 | control | +0.233 | **+0.341** |
| k=1 | actions given | +0.187 | **+0.292** |
| k=1 | observer | +0.182 | **+0.302** |
| k=8 | control | +0.141 | **+0.263** |
| k=8 | actions given | +0.126 | **+0.178** |
| k=8 | observer | +0.109 | **+0.241** |

and it is marginally *cheaper* in prediction (control 0.1218 vs 0.1253). So the metric correction is
not only the best training-free structural editor (reproduced here at −0.516 / −0.516 / −0.423 vs
pseudoinverse −0.656 / −0.649 / −0.552, matching the published −0.51) — it is also a **better thing
to fine-tune a model towards**. Measured Σ_hh condition number 8.85e3 (published 1.79e4 on dataset
4): strongly anisotropic, so the un-whitening gate passes.

## Adapting the EDITOR beats adapting the MODEL

Every fine-tune arm degrades prediction — control 0.1041 → 0.1218–0.1253 (+17–20%), actions-given
0.1071 → 0.1205–0.1356 (+13–27%) — and still ends **negative** on the Edit Index. The MLP arms leave
the world model untouched (prediction unchanged by construction) and end **positive**. All fine-tune
arms carried the retention term (weight 1.0, confirmed with Sevan), so this is not the known
no-retention degeneracy.

## The two losses do NOT simply trade index for fidelity — the k=8 arms OVERTAKE

**Added 2026-08-14 (Sevan asked for the Edit Index across the rollout; it corrects what the step-0
numbers alone implied).** At step 0 it looks like a clean trade — control MLP editor **+0.204** at
`k=1` vs **−0.035** at `k=8`, fidelity 0.84 vs 0.67. The by-step curves show that is incomplete: the
`k=8` arms start lower, **cross above their `k=1` counterparts by about step 4, and stay there for
the rest of the rollout**, in every mechanism and every model.

| control · mechanism | step 0 | step 4 | step 8 | step 14 |
|---|---|---|---|---|
| MLP editor `k=1` | **+0.204** | +0.215 | +0.223 | +0.127 |
| MLP editor `k=8` | −0.035 | **+0.267** | **+0.286** | **+0.230** |
| Fine-tune · un-whitened `k=1` | −0.224 | −0.196 | −0.160 | −0.100 |
| Fine-tune · un-whitened `k=8` | −0.339 | −0.041 | **+0.041** | **+0.042** |

So the rollout loss produces an edit that **takes a few steps to materialise and then holds**, ending
higher on the index *and* lower on GT-traj RMSE. The `k=1` arms land hardest on exactly the frame
they were trained on and then decay. **Only the step-0 number makes `k=1` look better**; past ~4
steps `k=8` is the better editor. Reporting step 0 alone would have inverted this — which is why the
registry requires the by-step curve alongside it.

⚠ **The unsteered curve climbs on its own** (control −0.671 → −0.438) because a free-running model
drifts away from *both* reference worlds. Part of every arm's rise is that drift, so the notebook's
Table 6 reports the **gap to that arm's own unsteered curve**, and a step-14 ÷ step-0 "retention"
ratio is deliberately **not** reported (several arms have a step-0 index near zero, where it explodes
or flips sign).

Also visible only in these curves: **Decoder Grad k=8 is the strongest oracle across the whole
rollout** (+0.80 → +0.38 on the actions-given model), above Counterfactual Overwriting and the
action interface — consistent with the editor gallery's "the only editor that both lands *and*
persists".

## Actions/teleports in world-model training buy nothing for LATENT editing

The control — trained with no actions and no teleports at all — is not worse at any of this; it is
the **best** cell for both the MLP editor (+0.875) and the metric fine-tune (+0.341). The
action-conditioned model's advantage appears only in its **action interface** (+0.618, fidelity
0.71) — the channel that bypasses the latent entirely.


## 2026-08-14 (review round 2) — corrections from Sevan's review

**Data:** the edit sets now carry **exactly one intervention** — regenerated with `--p-action 0.0`
(`datasets/15_teleport_eval_single` / `16_teleport_edittrain_single`), the single teleport synthesised.
All 18 arms retrained and re-evaluated on them.

**Figures:** Fig 1's rows are now identically ordered across all three panels with an explicit *n/a*
slot where a model has no action channel (so it can be scanned horizontally), the fidelity ratio is a
**hatch on failing bars** rather than a number on every bar, and the axis says
*Edit Index — edit frame (step 0)*. Fig 2 plots the **absolute** Edit Index like every other figure,
with each arm's own unsteered value marked on the same axis, instead of a "gain" axis with a different
quantity printed above the bars. Waterfalls are 3 rows, all three models, both fine-tune writes.

**Calibration — what +0.2 actually means.** Sevan's qualitative read: *"the MLP trained editors, even
though they hit an edit index near +0.2, still are not great — the edits barely land, frequently fall
apart, and introduce noise."* The arithmetic agrees: the index is `(1−r)/(1+r)` with
`r = d_edit/d_uned`, so **+0.2 means only 33% closer to the edited world**, versus the counterfactual
oracle's +0.63 (`r = 0.23`). **The honest description is "the edits barely land", not "the editor
works."** A calibration table is now in `METRICS_AND_EDITORS.md`, and the headline claim in this note
should be read with it: the mechanism is the first probe-free one to move the index at all, and it is
still far from a usable handle.

## Reading (mine, not established)

The thread's negative has always concerned **probe-derived** writes: take a direction that
*correlates* with position and invert it. Every §1 editor is that, and every one fails. The MLP
editor is the first mechanism handed the *edit itself* (start → target) and allowed to learn its own
write into a **frozen** state — and the first to produce a positive Edit Index.

This does not overturn the negative; it **locates** it. What was missing was never reachability
(`full_rowspace_edit`: the ceiling can be raised to 1.0 with no effect) nor capacity
(`action_hidden_size`: the negative deepens as capacity grows) but a **map from the intended change
to the state change** — which a linear probe's pseudoinverse estimates badly and which a small
network can learn from examples. That also explains the un-whitened result: it is a *better
approximation of that map*, so a model fine-tuned toward it has less to correct.

Honest limit: the best cell is +0.20 at step 0 (`k=1`) or ~+0.27 mid-rollout (`k=8`) — past
equidistant, but well short of the counterfactual oracle's +0.63 and of Decoder Grad k=8's +0.80.
**The mechanism edits; it does not yet edit cleanly.**

## Owed / not done

- One seed per cell; 3000 steps everywhere; one editor architecture (2×512), width not swept;
  retention weight fixed at 1.0.
- **The MLP editor is given ground-truth start and target positions.** A deployable version would
  read the start from the model's own probe — untested, and that gap is the obvious next measurement.
- Held out by construction (disjoint edit pools) but **within one edit distribution**: no withheld
  object and no displacements outside the training range — exactly where the published fine-tuning
  arms failed ("a button, not a handle"). Until that is run, this is not shown to be a *handle*.
- `Oracle observation` varies sharply by model (+0.013 control, +0.234 actions-given, **+0.557**
  observer) — unexplained.

## New code / harness

- `scripts/train_action_editors.py` — generalises `train_editable_gru.py` to action-conditioned
  models, dataset-7 synthesised teleports, `--edit-k`, the new `StateTargetEditor`, and
  `--write {pinv,metric}`; `scripts/eval_action_editors.py` — the full 3-family ablation with each
  fine-tuned arm carrying its **own** unsteered row and next-step RMSE.
- New splits: `datasets/14_cont_teleport_edittrain` (base seed 300000, editor training), evaluated
  on `datasets/13_cont_teleport_eval` (base seed 200000); both disjoint from world-model seeds.
- **Fixed in `eval_action_sweep.xg_data`:** the "edited object stays separated" filter was hardcoded
  to a 15-step window even when fewer steps were rendered, discarding usable episodes (3462 vs 6000
  needed). Now honours `n_gt_steps`.
- Trap re-encountered and documented in code: **cuDNN cannot backprop through an RNN in `eval()`
  mode**, so any editor that optimises `h` through the dynamics must flip to `train()` (identical
  behaviour here — no dropout) and restore afterwards.
