# Direction: Learn-to-Edit — can editability be *induced*? (frozen editor + light fine-tune)

**Tag:** `[reframe]` · **Sub-question:** 3 (editability) · **Status:** v1 done; REVISION PASS v2 (2026-07-09) ·
**Complexity:** medium-high (2 variants; Variant A first, clean; Variant B deeper)

> ## REVISION PASS v2 (Sevan feedback, 2026-07-09) — revise `editability/learn_to_edit.ipynb`
> The v1 result stands (editability not cleanly induced); this pass is **legibility + completeness**, not
> new science. Apply CLAUDE.md's new **Notebook legibility** standard throughout, plus these specifics:
> 1. **Definitions table up front** — every metric (`d_gt`, `d_tgt`, `ghost ratio`, `sel_err`, off-manifold
>    `resid`) with its **explicit formula**, units, and ↑/↓. No buried print-sidenote definitions.
> 2. **RMSE everywhere, not MSE** — the training-loss curves (Fig 1a, Fig 5a) plot raw MSE while the tables
>    use RMS; make ALL obs-error quantities RMSE (match the rest of the repo). Same fix on axis labels.
> 3. **Same metric suite + presentation for Variant A and Variant B** — currently A gets the head-to-head +
>    data-scaling and B gets "quality + canonicity"; make them comparable (both get the same core RMSE
>    metric table). **Add a Variant-B fine-tune data-size sweep** mirroring A's (N_TRAIN sweep; keep it
>    tractable — ~4 points is fine given fine-tuning is heavier than editor-training; note the trade-off).
> 4. **Primary waterfalls for A and B at the SAME train size**, and **the FT waterfalls MUST include a GT
>    (post-edit) target column** — v1's FT waterfall has only ORIG-unsteered / ORIG+edit / FT+edit, which is
>    uninterpretable without GT. Every editing comparison figure needs the GT/target reference column.
> 5. **Document the Variant-B setup in markdown** (so the reader isn't guessing): what "ORIG + fixed editor"
>    vs "FT + fixed editor" means; that the (pos,vel) linear probe is **re-fit (detached) on the current
>    model's states**, not fixed-weights; that the fine-tune objective is a **K=15-step (multistep) rollout**
>    match to the GT clean post-edit obs plus a next-step prediction-fidelity anchor; and that eval injects a
>    **held-out** target via that pseudo-inverse and rolls out K steps.
> 6. **Data-source provenance** on every borrowed constant (e.g. the GRU fiber/geometry refs from
>    `diagnostic_corrections` / `candidate-*`). Re-run on GPU; keep outputs lean.

> This is a **constructive** turn on the editability question. We have *diagnosed* why hand-crafted
> edits fail (non-canonical state, curved `(pos,vel)→h` embedding, readable≠controllable). Now ask:
> is that failure **fundamental** to the representation, or just a failure of our *editing method*?
> Train an editor / lightly fine-tune the model on the SAME small data budget the probes used, and
> see whether clean, persistent, selective edits become possible.

## The two variants (do A first — it is the clean test)

### Variant A — Frozen-model learned editor (information-presence test)
Freeze the trained world model. Train a small network `E_θ: (h, target) → h_edit` (predict `Δh` or
`h_edit` directly) on the edits split. `target` = the desired post-edit world-state spec (teleported
position of ONE object, the other unchanged; velocity = the edits' preserved original velocity).
**Loss:** roll the frozen dynamics forward from `h_edit` and match the **GT post-edit observation
sequence** (obs-space MSE over K rollout steps), optionally + a small on-manifold / ghost penalty.
Only `E_θ` trains; model weights frozen.

- **If A succeeds** (clean, persistent, ghost-free, selective edits on *held-out* edits): the
  information needed for control **is present in `h`** and reachable by *some* function of it —
  controllability was a **parametrization** problem, not an information problem. The linear/manifold
  editors just used the wrong map.
- **If A fails** even when allowed to overfit: the non-canonical state is **intrinsically
  uncontrollable via any function of a single `h`** — a strong structural claim, and direct support
  for the RESEARCH.md thesis that implicit architectures resist editability (→ explicit scaffolding).

### Variant B — Light fine-tune for editability (inducibility test)
Lightly fine-tune the model (or a small adapter) on the edits split so that injecting a target via a
**fixed simple editor** (e.g. the linear-probe pseudo-inverse) produces the intended rollout.
- **If B succeeds:** editability is **trainable/inducible** from few examples. Then **re-run the
  fiber-collapse + geometry diagnostics on the fine-tuned model** — did inducing editability make the
  state **more canonical** (fiber residual drops, embedding flattens)? That measurement directly tests
  the organizing hypothesis (editability ⟺ canonical state).
- **If B fails** even fine-tuning on the exact objective: the strongest evidence that this
  architecture structurally resists editability.

## Fairness / controls (HARD — this is where the result lives or dies)
- **Same data budget as the probes.** Train E / fine-tune on the edits split only; report N. This is a
  *few-shot* editability test, not a data-scaling one.
- **Held-out edits.** Split the edits set; train on a subset, evaluate on **unseen** edits. A
  frozen-editor that only works on trained edits = memorization, not controllability. State this test
  explicitly.
- **Head-to-head baselines** (all previously failed): probe pseudo-inverse, global-manifold projection,
  MLP-gradient (obs-driven) editor. Same edits, same obs-space metrics.
- **Metrics (obs-space, the space the effect lives in):** (i) distance to GT post-edit rollout;
  (ii) **ghost ratio** (does the old object vanish); (iii) **persistence** (obs stays at target over
  the full rollout vs reverts — the failure mode of every prior editor); (iv) **selectivity** (the
  non-edited object stays put); (v) **off-manifold residual of `h_edit`** — does the learned editor
  land ON the visited manifold (unlike the obs-gradient editor's 15.7)?
- **Interpretation guard:** distinguish "information present & controllable" from "editor overfit the
  small set" — the held-out + selectivity metrics are what separate them. Do not let a memorizing
  editor read as a controllability result.

## Bootstrap (cold-start)
Mirror `notebooks/experiments/editability/canonical_state_editing.ipynb`. Paths 3-deep
(`../../..` repo, `../../../runs`, `../../../datasets`). GRU
`runs/gru/3_dset3_gru_persistentids_inview_400epochs/best_model.pt`, data `datasets/4_fixed_refl_inview`.
`eval.warm_up_to_edit(model, edits.obs[:N], edits.edit_frame)` → `warm`, `h_at_edit`; the edits HDF5
carries the post-edit GT trajectory + `velocities` (preserved original velocity). Model implements
`HiddenStateModel` — roll out with `predict_step`/`decode` for the differentiable obs-space loss.
GRU first; RSSM (`runs/rssm/4_dset4_refined_best/`, `model.sample=False`) as the generalization pass.

## Deliverables
- Executed notebook(s) (numbered cells/figures), plots + printed tables; head-to-head vs the failed
  baselines; the held-out generalization result. PNGs to `/tmp/learn_to_edit/`.
- Dated `research/scratch/<date>-learn-to-edit.md`, flagged `→ FLAG FOR PROMOTION`: does a frozen
  editor induce clean edits? does fine-tuning? if B works, did the state become more canonical?
  caveats (esp. memorization). Do NOT promote / mark done / edit RESEARCH.md.

## Scope note
This edges toward "training an editor," which risks conflating a fancy editor with a property of the
representation. Variant A (frozen model) keeps it clean — it is an *information-presence* probe, not a
deployable feature. Only escalate to B (fine-tune) after A lands, and interpret B through the
canonicality re-measurement, not the edit quality alone.
