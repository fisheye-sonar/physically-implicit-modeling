# Direction: Can editability be INDUCED BY TRAINING?

**Tag:** `[in-frame]` · **Sub-question:** 3 (editability) · **Status:** active (2026-07-30) ·
**Complexity:** medium (differentiable edit-and-rollout objective; 5 arms, minutes each) · **Model:** GRU only.

Branch `more_trained_editability`. Notebooks: `notebooks/experiments/editability/trained_editability/`.
Registry: `TRAINED_EDITABILITY_RUNS.md`. Pays a debt recorded in `METRICS_AND_EDITORS.md`.

## The gap this closes

Every §4 result to date uses **inference-time** editors on a **frozen** world model, and every probe-directed one
fails: on the same model and decoder, the decoder-gradient oracle reaches the edited world (Edit Index **+0.94**)
while readout injection, Global-PCA projection and PCA geodesic sit at the unedited end (**≈ −0.6**, i.e. within
0.05 of doing nothing). The standing interpretation is that the failure is the **reachability of the edit map**,
not the representation and not the predictor.

`learn_to_edit` already asked whether *training* changes this, and returned two negatives — a learned amortized
editor (memorisation signature) and a fine-tune for editability. **But both were run at a deliberately light
budget, and "heavier fine-tuning still owed" has been an explicit OWED item since.** A negative obtained at a light
budget is not a negative about trainability; it is a negative about that budget.

## The question

If we *train* for editability — either by adapting the world model to a fixed write mechanism, or by learning the
write mechanism against a fixed world model — does a latent object handle appear? And if something appears, is it a
**handle** (generalises across write mechanisms and across content) or a **button** (works only for the exact
interface and content it was trained on)?

## Design

Two mechanisms, one evaluation. All arms start from `runs/controls/H256`; `scripts/train_editable_gru.py`.

**A — fine-tune the world model to a FIXED editor.** The editor is readout injection through a linear probe fit
once on the base model and then **frozen**. Nothing about the editor is learned, so all adaptation is in the world
model, which must learn to honour writes along `A⁺` as "put the object here".
`edit = MSE(rollout(h_edited, K=15), clean_obs[ef:ef+K])`, `total = edit + λ·retention` where `retention` is the
ordinary next-step prediction loss. **λ is the arm that matters**: it separates "the model became editable" from
"the model was destroyed and now renders whatever it is asked for".

**B — amortized editor, world model frozen.** Learn `E_θ(h, target) → Δh` against the same edit loss.

**Arms:** `FT_light` (300 steps), `FT_heavy` (3000), `FT_heavy_noret` (λ=0), `FT_heavy_obj0` (object-0 edits only),
`AMORT`. Training uses `edits[2000:]`; **every reported number is the held-out `edits[:64]`**, the same samples the
`controls/` notebooks use.

## Hypotheses (state before running)

1. **Heavier fine-tuning does move the edit** — the light-budget negative was about budget, so `FT_heavy` should
   beat `FT_light`. *(Confirmed: +0.13 vs +0.04 index points.)*
2. **It will be a button, not a handle.** Prediction: the trained interface improves, but the *same mechanism with
   a freshly-fit probe*, and the other standard editors, do not. *If mechanism generalisation appears, that is the
   most important positive of the whole editability thread.*
3. **Withholding an object costs the edit.** `FT_heavy_obj0` should do worse on object-1 edits than the
   both-objects control does, i.e. it learns a per-object button rather than "move an object".
4. **Editability is bought with prediction.** Removing the retention term should improve nothing about editing and
   degrade the world model toward the observation noise floor.

## Readouts

The canonical §4 set (`../../notebooks/experiments/editability/METRICS_AND_EDITORS.md` §4, implemented in
`scripts/editability_metrics.py`; evaluated by `scripts/eval_controls.py --root runs/trained_editability`):
**Edit Index** and **Edit Index by step**, Target / Ghost / Collateral / Edit-frame / GT-traj RMSE, fidelity ratio,
plus predictive quality, recoverability and canonicality to price the cost. Waterfalls per the fixed spec.

**Everything is reported as Δ from that arm's own unsteered value**, because a fine-tune that damages prediction
raises the unsteered Edit Index for free (r = +0.987 with next-step RMSE across models) and would otherwise look
like an editing gain.

## What would count as an answer

- **Mechanism generalisation appears** → the strongest positive available to this thread: editability is trainable
  and produces a real handle. Sends us back to every prior negative.
- **Only the trained interface improves** → training wires a *button*, exactly as the exogenous-action work found
  for the input→dynamics pathway. Strengthens the case for explicit object scaffolding (RESEARCH.md endgame).
- **Nothing improves even at a heavy budget** → the light-budget negative was not a budget artifact after all.

## Deliverables

`trained_editability/finetune_for_editability.ipynb` (+ `learn_to_edit.ipynb`, moved into this directory),
`scripts/train_editable_gru.py`, checkpoints in gitignored `runs/trained_editability/`, PNGs to
`/tmp/trained_editability/`, and a dated `research/scratch/2026-07-..-trained-editability.md`.
Do NOT edit `findings/` or `RESEARCH.md`.
