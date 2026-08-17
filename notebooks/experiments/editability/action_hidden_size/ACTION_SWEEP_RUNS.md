# Action-model hidden-size ablation — canonical RUN REGISTRY

**The single source of truth for what every run code in `action_hidden_size/` means.**
Per `CLAUDE.md`: no notebook may use a run code without copying its row into its own definitions
table, and **figures use the descriptive label, never the raw code**. Adding a run means adding its
row here in the same commit.

Origin: Sevan, 2026-08-13 — *"run an additional ablation of hidden state sizes but all on the action
models and make sure that our findings replicate or are proven to change."*
Branch `rogerio_controls`. Driver: `scripts/train_action_hidden_sweep.sh`.
Evaluation: `scripts/eval_action_sweep.py` → `runs/action_sweep/eval/<code>.json`.
Checkpoints live in gitignored `runs/action_sweep/<code>/`.

## What this sweep is testing

The plain-GRU hidden-size sweep (`runs/controls/H{8,32,128,256,512}`, `controls/hidden_size_sweep.ipynb`,
2026-07-30) produced four findings. This sweep asks whether they replicate when the world model is
**action-conditioned**:

| # | finding on the passive GRU | replicated here? |
|---|---|---|
| **F1** | prediction saturates by `H≈128` (next-step RMSE 0.1495 → 0.1167 → 0.1054 → 0.1041 → 0.1042) | see notebook §1 |
| **F2** | linear readability rises **monotonically** (position R² 0.175 → 0.855; velocity 0.002 → 0.531) | §2 |
| **F3** | canonicality moves the **opposite** way (MLP fiber residual 0.215 → 0.601) | §2 |
| **F4** | **grabbability does not move at all** — the §4 negative holds at every `H` | §3 |

## Shared design rule

**Hidden size is the ONLY variable within each family.** Architecture depth, objective, optimiser,
iterations/epochs, seed and world settings are identical across a family's five runs.

## Family 1 — exogenous teleport actions (`XG_*`)

`scripts/train_action_gru_continuous.py`, `ActionGRUContinuousModel` (GRU + a per-object continuous
action vector `[active, a1, a2]` projected into the encoder), dataset `datasets/7_cont_teleport`
(90,000 sequences, 40 frames, `obs_res=128`, `p_action=0.30`, `move_scale=4.0`, matched to dataset 4:
2 objects, fixed reflectivities, `obs_noise_std=0.2`, `position_noise_std=0.04`).
**400 epochs, batch 256, AdamW lr 1e-3, weight decay 1e-4, 10% val, seed 0.**

The action is a **teleport to absolute coordinates**, so this family's action space *contains the edit
under test* — the model was trained to render exactly the intervention §3 asks for. That makes
"issue the action" a **built-in ground-truth handle** and the natural positive control.

| code | descriptive label (use this in figures) | hidden | actions | purpose |
|---|---|---|---|---|
| `XG_A_H8` | Exogenous teleport · actions given · 8 hidden | 8 | **given** | world's true state dimensionality |
| `XG_A_H32` | Exogenous teleport · actions given · 32 hidden | 32 | given | |
| `XG_A_H128` | Exogenous teleport · actions given · 128 hidden | 128 | given | = the observation resolution |
| `XG_A_H256` | Exogenous teleport · actions given · 256 hidden | 256 | given | matches the thread's baseline capacity; comparable to the published `M_teleport` |
| `XG_A_H512` | Exogenous teleport · actions given · 512 hidden | 512 | given | |
| `XG_C_H8` … `XG_C_H512` | Exogenous teleport · **actions withheld** · {8,32,128,256,512} hidden | 8–512 | **withheld** | the action-knowledge control: identical data and recipe, the action input removed. Isolates *having actions* from *having capacity* — the `M_teleport` vs `M_teleport_ctrl` contrast of `action_space_object_individuation`, now crossed with `H`. |

**Held-out eval split:** `datasets/15_teleport_eval_single/eval.h5` — 4,000 sequences at **base seed
200000** (disjoint from the training seeds 0–89,999) generated with **`--p-action 0.0`**, i.e. a world
that performs **no teleports of its own**. The single teleport under test is synthesised at `ef`.

> ### ⛔ Why the edit set has to be intervention-free
> Generating it with the training-time `p_action = 0.30` (the original
> `datasets/13_cont_teleport_eval`, superseded 2026-08-14) put random teleports both **inside the
> scored horizon** — where the free-running model is judged on events it was never told about — and
> **in the visible context**, which made these episodes structurally unlike the dataset-4 edits split
> and so destroyed the comparison with the control that the control exists for. `eval_action_sweep.py`
> now **asserts** the edit set contains no interventions of its own.

## Family 2 — endogenous interactive actor (`EN_*`)

`scripts/train_endogenous.py --level 3`, `EndogenousActorGRU`. The model **acts on the world**: its
policy head emits a force, the world integrates it, objects die on object–object and wall collisions,
and the actor is trained with **REINFORCE + value baseline on a survival reward (+0.1/step, −1.0/death)
into the same GRU trunk that is doing next-step prediction**. Level 3 = force dynamics + death + goal.

**6000 iterations, batch 64 parallel worlds × rollout 48 frames, `--batched-sim`, lr 3e-4, γ 0.99,
entropy coef 0.01, single Linear encoder/decoder ("weak" architecture), seed 0.**

| code | descriptive label (use this in figures) | hidden |
|---|---|---|
| `EN_H8` | Endogenous L3 force+goal · 8 hidden | 8 |
| `EN_H32` | Endogenous L3 force+goal · 32 hidden | 32 |
| `EN_H128` | Endogenous L3 force+goal · 128 hidden | 128 |
| `EN_H256` | Endogenous L3 force+goal · 256 hidden | 256 |
| `EN_H512` | Endogenous L3 force+goal · 512 hidden | 512 |

> ### ⚠ Not bit-comparable to `runs/endogenous/L*`
> Those runs carry the **known 0.05 observation-noise deviation** documented in
> `../actions/ENDOGENOUS_RUNS.md`. The runs here use `--obs-noise 0.2`, the **repo standard** used by
> every dataset and by the exogenous family. So `EN_H256` is the same recipe as the registry's `L3`
> *except* for the noise level, and their absolute numbers should not be compared directly.

> ### ⚠ This family has **no action-interface oracle**, and that is a result, not an omission
> The exogenous family's actions are teleports to absolute coordinates, so its action space *contains*
> the edit. The endogenous actor's actions are **forces** — it physically cannot teleport an object, at
> any capacity. The two families therefore differ along an extra axis (does the action space contain the
> intervention?), which is stated wherever they appear together rather than left as a missing bar.

## Metric conventions

Everything is computed by `scripts/eval_action_sweep.py` through `scripts/editability_metrics.py`
(Edit Index, Target / Ghost / Collateral / Edit-frame / GT-traj RMSE, fidelity ratio) and
`pim.extractors.fit_readability_probes` (linear lstsq + 2×256 MLP, both fit on the same 80% of
**sequences**, scored held-out).

> ### ⛔ `scripts/eval_editability_endogenous.py` is NOT used
> That script still computes the metric set **retired on 2026-07-30** (`reach` / `collat` / `ghost` /
> `select`), which scored *change* rather than *correctness* and normalised by a model-dependent soft
> reference. `CLAUDE.md` forbids reintroducing them, and pre-2026-07-30 numbers on that scale are not
> comparable to anything here.

### Ground-truth worlds, per family

* **Exogenous** — the world is generated **teleport-free** (`--p-action 0.0`) and the single edit is
  **synthesised**: a target sampled in-frustum, clear of the other object, encoded with the
  generator's own `normalize_action` so it is an action the model was trained on. Both reference
  futures are then **constructed** by rolling the frame-`ef` state forward under the passive
  (ballistic) dynamics — never read from later dataset frames — so the scored window contains exactly
  the one intervention under test. The evaluator asserts this rather than assuming it.
* **Endogenous** — the two ground-truth worlds are produced by **forking the simulator at `ef`** and
  stepping both forks under the **same action sequence**. In a force world
  `pos(t+1) ≠ pos(t) + v·dt`, so `build_edit_zones`' ballistic roll-forward would be wrong; the fork
  is the honest counterfactual. `pre_vel` is passed as 0 so the frame-`ef` counterfactual is the true
  unedited world rather than a ballistic extrapolation.
