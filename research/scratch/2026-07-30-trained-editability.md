# Trained editability: heavier fine-tuning wires a BUTTON, not a handle

**Date:** 2026-07-30 · **Branch:** `more_trained_editability` · **Direction:** `trained-editability` (`[in-frame]`,
sub-Q 3) · **Status:** → **FLAG FOR PROMOTION** (pays the OWED heavy-fine-tune debt; the light-budget negative was
partly a budget artifact, but the corrected result is still a clean button-not-handle negative) · **Author:**
orchestrator.

## The question
Every §4 result before this used **inference-time** editors on a **frozen** world model, and every probe-directed
one fails (base model: unsteered Edit Index −0.68, structural editors −0.63…−0.67, decoder-gradient oracle +0.94).
`learn_to_edit` asked whether *training* fixes it and returned two negatives — **both at a deliberately light
budget**, with "heavier fine-tune still owed" recorded in `METRICS_AND_EDITORS.md` ever since. This pays that debt.

## Setup / provenance
`scripts/train_editable_gru.py`; all arms start from `runs/controls/H256`; batch 64, Adam lr 1e-4, `K=15`.
Training on `edits[2000:]`, **every number below on the held-out `edits[:64]`** (the same samples the `controls/`
notebooks report on). Registry: `notebooks/experiments/editability/trained_editability/TRAINED_EDITABILITY_RUNS.md`.
Notebook `trained_editability/finetune_for_editability.ipynb` (14 cells, 0 errors, 4 figures);
PNGs `/tmp/trained_editability/`. Metrics = the canonical §4 set (`editability_metrics.py`), evaluated by
`eval_controls.py --root runs/trained_editability`.

Two mechanisms: **fine-tune the world model** to obey a *fixed, frozen* readout-injection probe (nothing about the
editor is learned), and an **amortized editor** `E_θ(h,target)→Δh` against a *frozen* world model.

> **Everything is reported as Δ from that arm's OWN unsteered index**, because a fine-tune that damages prediction
> raises the unsteered index for free (r = +0.987 with next-step RMSE). `FT_heavy_noret` is the live example: its
> unsteered index rises −0.68 → −0.39 purely because its prediction degraded.

## Headline
**Training does move the edit — and it wires a button.** The best arm gets roughly half-way from "did nothing" to
"equidistant between the two worlds", and *none* of it transfers to another write mechanism or to an object the
training withheld.

## Results

**§1 Does training work at all? (Table 1)** Δ Edit Index of the **trained interface** vs that arm's own unsteered:
| arm | Δ trained interface | prediction cost (next-step RMSE) |
|---|---|---|
| base model (no editability training) | **+0.01** | 0.1041 |
| fine-tuned · light (300 steps) | **+0.04** | 0.1176 |
| fine-tuned · heavy (3000 steps) | **+0.13** | 0.1173 |
| fine-tuned · heavy · no retention | +0.10 | **0.1486** |
| fine-tuned · heavy · object-0 only | +0.10 | 0.1091 |
| **amortized editor (world model frozen)** | **+0.54** | **0.1041** (unchanged by construction) |

**Hypothesis 1 confirmed — the light-budget negative was partly a budget artifact:** 300 → 3000 steps takes the
fine-tune from +0.04 to +0.13. But the effect is small, and **the amortized editor beats every fine-tune by 4×**
while costing the world model nothing. Note what that means: it is *easier to learn a bespoke write mechanism for a
frozen latent than to make the latent obey a fixed one.*
Absolute level for the best arm: Edit Index **−0.14**, against its own unsteered **−0.68** and the
decoder-gradient oracle's **+0.94**. So even the best trained edit only reaches "equidistant / neither world" — it
never arrives at the edited world.

**§2 Mechanism generalisation — none (Fig 1).** The *same mechanism* with a freshly-fit probe moves only
**+0.01…+0.04** on every arm, versus **+0.13/+0.54** for the interface actually trained. The other standard editors
are likewise unmoved. **Hypothesis 2 confirmed:** the model obeys the specific interface it was trained for, not
the mechanism class. This is the same shape as the exogenous-action finding — objecthood lives in the trained
pathway, not in the state.

**§3 Content generalisation — fails (Fig 2, Table 2).** `FT_heavy_obj0` (object-0 edits only) has an
obj1−obj0 gap of **−0.08**; the both-objects control `FT_heavy` has **+0.09** on the same split. So withholding an
object costs ≈ **0.17 index points**. **Hypothesis 3 confirmed:** what is learned is a per-object button, not
"move an object".

**§4 Cost (Fig 3b/3c).** Fine-tuning costs prediction even with the retention term (0.1041 → 0.1173, +13%).
Without it, next-step RMSE degrades to **0.1486** — essentially the observation noise floor (0.1539), i.e. the
world model is destroyed — **while editing gets no better** (+0.10 vs +0.13). **Hypothesis 4 confirmed:**
editability bought by discarding prediction is not editability.

**Persistence (Fig 3a).** The trained edits do not decay much over the rollout — but they start close to the
unsteered end, so there is little to decay. The contrast is the base model's decoder-gradient oracle, which starts
at **+0.94** and falls to **−0.12** by step 14: the only mechanism that truly reaches the edited world holds it for
about one frame.

## Reading
The heavy-budget test changes the *strength* of the earlier negative but not its *direction*. Editability is
trainable to a small degree and only as a button: bound to one write mechanism and one object. The cleanest new
fact is the asymmetry — **learning a bespoke editor for a frozen latent (+0.54) works far better than making the
latent obey a fixed editor (+0.13), and costs the world model nothing.** That is consistent with the standing
interpretation that the obstacle is the *reachability of the edit map* rather than the representation: if you are
allowed to redesign the map, you get further than if you try to move the representation to meet a fixed one.

## Caveats / open
- One seed per arm; GRU only; one base model (`H256`).
- The fine-tune trains through a 20-frame teacher-forced warm-up plus a 15-step rollout — a long differentiable
  chain; no gradient-truncation ablation was run.
- The amortized editor was not given the mechanism-generalisation *training* it would need to be fair on that axis
  (it is trained as its own mechanism); its +0.54 is a within-interface number.
- Budget: 3000 steps is ~12 s of GPU. "Heavy" here is heavy relative to `learn_to_edit`, not absolutely — a
  genuinely large budget (or joint training from scratch with an edit objective) is still untested.

## Open questions for Sevan
- Artifact or signal? The button-not-handle conclusion now survives a real fine-tuning budget, which was the main
  loophole left open in `findings/editability.md`.
- Does this fold into the existing `findings/editability.md` entry as "and it is not fixable by light-to-moderate
  training", or is the amortized-vs-finetune asymmetry its own entry?
- Worth training from scratch with the edit objective in the loss, rather than fine-tuning? That is the one
  remaining version of "train for editability" this does not cover.
