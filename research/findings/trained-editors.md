# Finding: Trained edit mechanisms

*Sub-question 3 — can a **learned** editor do what probe-derived writes cannot, and what does
the answer say about why they fail?*

Split out as its own concept 2026-08-17. Previously these results were spread between
`editability.md` and `scratch/`, which made the trained arms look like a footnote to the
training-free negative rather than the line of work that eventually crossed zero.

## Current understanding

_Updated 2026-08-17._

**Adapting the editor beats adapting the model, and the first probe-free latent editor to reach
a positive Edit Index does so with the world model frozen.**

`E_θ(h, start_pos, target_pos) → Δh` reaches **Edit Index +0.20** (control model) against an
unsteered baseline of **−0.67**, at **zero prediction cost** — the world model is untouched. For
scale: the previously published best *learned* mechanism was **−0.14**, and the best
training-free structural editor is **−0.42 … −0.52**.

**The single change that did it was an extra input.** The published amortized editor took
`(h, target)`. Giving it the **starting** positions as well hands it the *displacement* instead
of making it infer the world from `h` — and that moves the mechanism from −0.14 to +0.20.

This does not overturn the negative; it **locates** it. Every failing editor is a probe-derived
write: a correlational direction, inverted. What was missing was never reachability or capacity
but a **map from the intended change to the required state change** — which a pseudoinverse
estimates badly and a small network can learn.

**Honest limit:** +0.20 is a long way below the decoder-gradient oracle (+0.80 at step 0), and
the waterfall shows real relocation but with streaking the oracle lacks.

## Log

### 2026-08-14 — A trained `(h, start, target) → Δh` editor crosses zero with the world model frozen · `replicated` ★-candidate

**Evidence:** `scratch/2026-08-14-trained-editors-actions.md` ·
`notebooks/experiments/editability/trained_editors_actions/` (+ `TRAINED_EDITOR_RUNS.md`) ·
`scripts/train_action_editors.py`, `scripts/eval_action_editors.py` · **18 trained arms** ·
world models `XG_A_H256` / `XG_C_H256` (from the hidden-size sweep) and `CTRL_H256`
(= `controls/H256`); no new world models trained.

Sevan's spec: two world models trained where objects teleport during training — one told the
teleport, one an observer — plus a control with no actions or teleports; a thorough ablation
over all edit types; a fine-tuning variant and an MLP editor taking `(h, start, target)`; each
under a **next-step** and a **k=8 rollout** loss.

**1. The headline.** `E_θ(h, start, target) → Δh`, world model frozen:

| model | trained editor | own unsteered | gain | fidelity |
|---|---|---|---|---|
| control (no actions) | **+0.204** | −0.671 | +0.875 | 0.84 |
| exogenous · actions given | **+0.111** | −0.669 | +0.780 | 0.89 |
| exogenous · observer | **+0.117** | −0.578 | +0.695 | 0.82 |

Target RMSE ≈ 0.48 → 0.25–0.28. Zero prediction cost.

**2. The un-whitened write is a better fine-tuning target — 6/6 cells.** Sevan's addition.
Pseudoinverse → un-whitened (`Σ¹`), gains over each arm's own unsteered:
k=1 control +0.233 → **+0.341**, actions +0.187 → **+0.292**, observer +0.182 → **+0.302**;
k=8 +0.141 → **+0.263**, +0.126 → **+0.178**, +0.109 → **+0.241** — and marginally cheaper in
prediction. It also reproduces as the best **training-free** structural editor
(−0.516/−0.516/−0.423 vs pseudoinverse −0.656/−0.649/−0.552; independently published at −0.51).
`Σ_hh` condition number 8.85e3, so the un-whitening gate passes.

**3. Adapting the editor beats adapting the model.** Every fine-tune arm degrades next-step RMSE
(control 0.1041 → 0.1218–0.1253; actions 0.1071 → 0.1205–0.1356) and **still ends negative**.
The MLP editor arms cost nothing and end positive. All fine-tunes carried retention (weight 1.0,
confirmed with Sevan), so this is not the known no-retention degeneracy.

**4. The two losses do not trade index for fidelity — the k=8 arms overtake.** Control MLP
editor: `k=1` +0.204 → +0.215 (step 4) → +0.127 (step 14); `k=8` **−0.035 → +0.267 → +0.230**.
The k=8 arms cross above their k=1 counterparts by ~step 4 in **every** mechanism and **every**
model, ending higher on the index *and* lower on GT-traj RMSE. A rollout loss produces an edit
that takes a few steps to materialise and then holds.

⚠ **The unsteered curve also climbs on its own** (−0.671 → −0.438) as a free-run drifts away
from *both* reference worlds, so every number above is read against each arm's **own** unsteered
curve. A step-0-only report of this experiment states the opposite of the truth — this is the
result that produced the standing rule to report the index across the whole rollout.

**5. Actions and teleports in world-model training buy nothing for latent editing.** The
**control** is the best cell for both new mechanisms. The action model's advantage appears only
in its **action interface** (+0.618), which bypasses the latent entirely.

**Reading (interpretation, not established):** this locates the negative rather than overturning
it — see Current understanding above.

---

### 2026-07-30 — Heavier fine-tuning wires a button, not a handle · `replicated`

**Evidence:** `scratch/2026-07-30-trained-editability.md` ·
`notebooks/experiments/editability/trained_editability/finetune_for_editability.ipynb`
(+ `TRAINED_EDITABILITY_RUNS.md`) · `scripts/train_editable_gru.py` · all arms start from
`runs/controls/H256`; trained on `edits[2000:]`, **every number on the held-out `edits[:64]`** —
the same samples the `controls/` notebooks report on.

Pays the "heavier fine-tune still owed" debt left by `learn_to_edit`, whose two negatives were
both at a deliberately light budget. Two mechanisms: **fine-tune the world model** to obey a
fixed, frozen readout-injection probe (nothing about the editor is learned), and an **amortized
editor** `E_θ(h, target) → Δh` against a frozen world model.

**Training does move the edit — and it wires a button.** The light-budget negative was partly a
budget artifact; the corrected result is still a clean button-not-handle negative. Base model
for reference: unsteered −0.68, structural editors −0.63…−0.67, decoder-gradient oracle +0.94.

**Methodological point that generalises:** everything is reported as Δ from **that arm's own**
unsteered index, because a fine-tune that damages prediction raises the unsteered index for free
(**r = +0.987** with next-step RMSE). `FT_heavy_noret` is the live example — its unsteered index
rises −0.68 → −0.39 purely because its prediction degraded.

**Superseded in part by 2026-08-14:** the amortized editor here took `(h, target)`; adding the
*start* positions is what later crossed zero. The fine-tune-the-model conclusion stands and was
independently reproduced across six cells.
