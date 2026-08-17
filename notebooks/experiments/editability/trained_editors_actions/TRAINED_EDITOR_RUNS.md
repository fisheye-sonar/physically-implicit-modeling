# Trained editors on exogenous-action models — canonical RUN REGISTRY

**Single source of truth for every run code used in `trained_editors_actions/`.** Per `CLAUDE.md`,
no notebook may use a code without copying its row into its own definitions table, and figures use
the descriptive label, never the raw code.

Origin: Sevan, 2026-08-14. Branch `rogerio_controls`.
Training: `scripts/train_action_editors.py` (driver `scripts/train_action_editors_all.sh` +
`scripts/train_action_editors_metric.sh`) · Evaluation: `scripts/eval_action_editors.py`
→ `runs/action_editors/eval/<model>.json`. Checkpoints in gitignored `runs/action_editors/`.

## The three world models (all pre-existing — nothing new was trained here)

| code | descriptive label | training data | actions as input? |
|---|---|---|---|
| `XG_A_H256` | **Exogenous teleport · actions given · 256 hidden** | `datasets/7_cont_teleport` — 90k sequences, objects teleport during training (`p_action=0.30`, teleport to absolute coordinates), next-frame prediction | **yes** |
| `XG_C_H256` | **Exogenous teleport · observer · 256 hidden** | identical data and recipe | **no** — the action port is removed, so it must predict teleports it is never told about |
| `CTRL_H256` | **Control · standard GRU · 256 hidden** | `datasets/4_fixed_refl_inview` — **no actions, no teleports in training** | no |

`XG_A_H256` / `XG_C_H256` come from `../action_hidden_size/ACTION_SWEEP_RUNS.md`; `CTRL_H256` is
`runs/controls/H256`, the repo's baseline GRU. All three are 256 hidden, seed 0.

## The trained-editor arms — 18 = 3 models × 3 editors × 2 losses

Every arm: **3000 steps, batch 64, Adam lr 1e-4, seed 0.** Editor training data is disjoint from
everything reported on (dataset 4: `edits[2000:]`, reporting on `edits[:128]`; teleport world:
`datasets/16_teleport_edittrain_single` at base seed 300000, evaluated on
`datasets/15_teleport_eval_single` at base seed 200000 — both disjoint from the world models'
training seeds 0–89999, and **both generated with `--p-action 0.0`** so each episode carries exactly
one intervention: the synthesised teleport under test. The earlier `13_`/`14_` splits, which inherited
the training-time `p_action = 0.30`, are superseded — they put random teleports in the scored window
and in the visible context, breaking comparability with the dataset-4 control.)

| arm suffix | descriptive label | what is trained | what is frozen |
|---|---|---|---|
| `__finetune__k{1,8}` | **Fine-tune · pseudoinverse write** | the **world model** | the editor — a linear-pseudoinverse readout injection through a probe fit once on the BASE model. Nothing about the write is learned; the model must learn to honour it. |
| `__finetune__metric__k{1,8}` | **Fine-tune · un-whitened (metric-corrected) write** | the **world model** | the editor, but the fixed write is `Δ = Σ¹Wᵀ(WΣ¹Wᵀ + εI)⁻¹δ` from `../metric_corrected_edits/` instead of the Euclidean pseudoinverse. **Same readout target, different metric** — so this isolates the metric as a variable in the *trained* setting. |
| `__mlp__k{1,8}` | **MLP editor `E(h, start, target)`** | the **editor network** (2×512 ReLU) | the **world model, entirely** |

### ⭐ The MLP editor is not the published "Trained Editor"

`learn_to_edit` / `trained_editability` trained `E_θ(h, target) → Δh`. This one takes
**`E_θ(h, start_pos, target_pos) → Δh`** — it is also given where the objects *currently* are, so
the displacement it must produce is supplied rather than inferred from `h`. Both position vectors
are the flat `(x₀,y₀,x₁,y₁)` at the edit frame: `start` = the **un-edited** frame-`ef` world (what
the model would render if left alone), `target` = the same world with the edited object teleported.
Do not quote its numbers as the published amortized editor's.

### The two losses

| suffix | loss | what it enforces |
|---|---|---|
| `k1` | `MSE(decode(h_edited), gt_edited[ef])` | the edit **lands** on the next frame |
| `k8` | `MSE(rollout(h_edited, 8), gt_edited[ef:ef+8])` | the edit **survives the dynamics** for 8 free-run steps |

**Retention.** Every `finetune` arm carries the prediction-retention term (ordinary next-step MSE on
non-edit sequences, weight 1.0). Confirmed with Sevan 2026-08-14. It is what separates "the model
became editable" from "the model was destroyed and now echoes the editor" — measured 2026-07-30, a
no-retention fine-tune's *unsteered* index ROSE from degraded prediction alone, making its apparent
gain pure scale movement. The `mlp` arms need no retention term: the world model is frozen, so their
prediction quality is unchanged by construction.

> ### ⚠ A fine-tuned arm is a DIFFERENT world model
> Its Edit Index is meaningless against the base model's unsteered row, so every fine-tune arm
> carries its **own** unsteered row and its **own** next-step RMSE, and the notebook reports the
> **gain over its own unsteered value** plus the prediction cost. The `mlp` arms share the base
> model's unsteered row because the world model is untouched.

## The editor ablation (`scripts/eval_action_editors.py`, N=128 held-out edits, K=15)

**Standard / training-free:** Unsteered · Pseudoinverse Injection · **Metric-corrected Injection
(un-whitened)** · Global PCA Projection (POCS) · Local PCA Geodesic · MLP Grad Steering (frozen
1×128 `MLPExtractor` on its published defaults) · Multistep Steering @8 (the model's **own** decoded
observations fed back — never an external render, which is what separates it from freeze-time).

**Oracle:** Oracle observation (one extra teacher-forced frame — **leads every other column by one**,
labelled not re-aligned) · Counterfactual Overwriting (8 rendered frames of a fabricated history) ·
Freeze-time TF @8 · Decoder Grad k=1 · Decoder Grad k=8 · **Action interface** (`XG_A_H256` only —
command the teleport through the model's own action channel; the other two have no such channel).

All metrics come from `scripts/editability_metrics.py`. `scripts/eval_editability_endogenous.py` is
**not** used — it still computes the metric set retired 2026-07-30.
