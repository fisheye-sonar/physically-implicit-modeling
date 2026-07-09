# Direction: Learn-to-Edit — can editability be *induced*? (frozen editor + light fine-tune)

**Tag:** `[reframe]` · **Sub-question:** 3 (editability) · **Status:** proposed (awaiting Sevan → active) ·
**Complexity:** medium-high (2 variants; Variant A first, clean; Variant B deeper)

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
