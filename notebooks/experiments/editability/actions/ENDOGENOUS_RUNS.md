# Endogenous-action experiments — canonical RUN REGISTRY

**The single source of truth for what every run name means.** Per `CLAUDE.md`, no notebook may use a run code without
copying its row into its own definitions table, and figures must use the **descriptive label**, not the raw code.

Direction: `research/directions/endogenous-action-interactive-world.md`. All runs train an **actor** and an
**observer** *simultaneously on the same data* (see "Roles" below). Checkpoints live in gitignored
`runs/endogenous/<code>/ckpt_final.pt`.

## Roles (identical architecture, different causal role — the central control)
| role | acts on the world? | sees actions? | trained on |
|---|---|---|---|
| **actor** | **yes** — its policy head emits the action that is applied to the world | its own | next-step prediction (+ REINFORCE policy/value at level 3, into the **shared** GRU trunk) |
| **observer** | no — never influences the world | fed the actor's actions | next-step prediction only, on the **same** (obs, action) trace |

## Levels (the world/objective, per the direction doc)
| level | dynamics | death enabled | goal / policy learning |
|---|---|---|---|
| **L1** | `shift` (action = position delta, frustum/collision-guarded) | no (guard makes death impossible) | none — prediction only (efference-copy ablation) |
| **L2** | `force` (action = force → momentum) | yes (object–object + wall) | none — prediction only |
| **L3** | `force` | yes | **survive**: REINFORCE + value baseline on +0.1/step, −1.0/death |

## Run codes
Suffix key: **no suffix / `b`** = the *original ("weak") configuration*, differing only in seed; **`s`** = the
*strong configuration*; the trailing digit is the **seed**.

| code | descriptive label (use this in figures) | level | architecture | training | seed | purpose |
|---|---|---|---|---|---|---|
| `L1` | L1 shift · prediction-only · 256h · seed 0 | 1 | 256 hidden, 1-layer Linear enc/dec, no multistep | 2500 it | 0 | efference-copy ablation (expected null) |
| `L2` | L2 force · prediction-only · 256h · seed 0 | 2 | same as above | 2500 it | 0 | physical dynamics, still no goal |
| `L3` | **L3 force+goal · weak · 256h · seed 0** | 3 | same as above | 6000 it | 0 | first-pass goal-directed actor |
| `L3b` | L3 force+goal · weak · 256h · **seed 1** | 3 | same as above | 6000 it | 1 | seed replication of `L3` |
| `L2s0` | L2 force · prediction-only · **strong** · seed 0 | 2 | **512 hidden, 2-layer MLP encoder + residual MLP decoder** | 12000 it, **5-step free-run loss** | 0 | **no-goal control at strong capacity** (isolates goal vs capacity) |
| `L3s0` | **L3 force+goal · strong · 512h · seed 0** | 3 | strong (as `L2s0`) | 25000 it, 5-step free-run loss | 0 | main strong actor |
| `L3s1` | L3 force+goal · strong · 512h · **seed 1** | 3 | strong | 25000 it, 5-step free-run loss | 1 | seed replication of `L3s0` |
| `L3s0_ait` | L3 force+goal · strong · seed 0 · **action-in-transition** | 3 | strong **+ previous action fed into the GRU input** | 25000 it, 5-step free-run loss | 0 | ablation: does giving the action a path into the STATE fix closed-loop rollout? (answer: only ~15% better) |
| `L3s0_ait_batched` | L3 force+goal · strong · seed 0 · action-in-transition · **vectorised simulator** | 3 | strong + action-in-transition | 25000 it, `--batched-sim` | 0 | validates `BatchedInteractiveWorld` end-to-end (identical args to `L3s0_ait`) |
| `L3s0_ait_state` | *(never produced a checkpoint — superseded)* | 3 | strong + action-in-transition | `--carry-state --batched-sim` | 0 | started 2026-07-29 16:00 as the carried-state test; **the run directory is empty** (no `ckpt_final.pt`). Superseded by `L3_bestgru_b1024`, which folds in carried state plus the dead-world state reset. Do not cite. |
| `L3_bestgru_b1024` | **L3 force+goal · best GRU · 512h · batch 1024 · seed 0** | 3 | strong + action-in-transition, **+ carried recurrent state with dead worlds' state cleared, + value bootstrap** | 4800 it × **batch 1024** × 48 = **236M frames** (3× the batch-64 runs), `--batched-sim` | 0 | **the best GRU we have** — every known implementation flaw fixed and 16× the batch / 3× the data. The fair point of comparison for RSSM. Result: teacher-forced 6.1, **closed-loop 87.2** deaths/1000 frames (no-goal control 79.1) → none of the plumbing fixes, capacity, or data volume rescues closed-loop rollout. |
| `L3s0_ckpt` | L3 force+goal · strong · seed 0 · **with intermediate checkpoints** | 3 | strong | 25000 it, checkpoint every 2500 it | 0 | training-stage animations (untrained → partway → trained) |

### RSSM runs (`runs/endogenous_rssm/`, direction `endogenous-action-rssm.md`)
Architecture for all: `RSSMActor` (`pim/world_models/rssm_actor.py`) — det 256 / stoch 32 / embed 200, 2-layer MLP
encoder + decoder, **action in the transition**, policy + value + reward + continue heads; DreamerV2-style objective
(recon + KL-balanced 0.8/0.2 with free bits); actor trained on λ-returns over **imagined** rollouts (horizon 15).
Trained on `BatchedInteractiveWorld`, batch 256 × rollout 48, **`obs_noise_std = 0.2` (the repo standard — these runs
do NOT carry the deviation below)**. Observer twin = same architecture, world-model loss only, never acts.

| code | descriptive label (use this in figures) | level | training | seed | purpose / outcome |
|---|---|---|---|---|---|
| `R2s0` | RSSM L2 force · **no goal** · seed 0 | 2 | 10000 it, **no actor loss at all** | 0 | no-goal control at matched capacity; guarantees a trained action-conditioned RSSM world model regardless of the actor bug. World model healthy (recon 0.168, KL 0.153, no posterior collapse). |
| `R3s0_warm` | RSSM L3 force+goal · **world-model warm-up 4000 it** · seed 0 | 3 | 14000 it, `wm_warmup=4000`, `ent_coef=0.05` | 0 | main RSSM actor. Warm-up + entropy bonus fixed the entropy collapse (ends 3.98) but the **actor still does not learn the task**: reward −0.016 vs the no-goal control's −0.033; 72.5 deaths/1000 frames vs 83. |
| `R3s1_warm` | RSSM L3 force+goal · warm-up 4000 it · **seed 1** | 3 | 14000 it, same | 1 | seed replication: reward −0.022, 76.0 deaths/1000 frames. Same conclusion. |
| `R3s0`, `R3s1` | *(never produced a checkpoint — superseded)* | 3 | 1500-it entropy-coefficient sweep, no warm-up | 0, 1 | the first, failed level-3 attempt (`ent_coef` 0.003 → entropy collapse to 0.04; 0.03 → alive but imagined return still fell monotonically). Directories are **empty**; superseded by the `_warm` runs. Do not cite. |

**"weak" vs "strong"** (GRU runs only) is *only* these two configurations:
- **weak** = 256 hidden, single Linear encoder + single Linear decoder, **no** multistep loss, 6000 iterations.
- **strong** = 512 hidden, 2-layer MLP encoder + residual MLP decoder, **5-step free-run (multistep) loss**, 25000
  iterations. Introduced 2026-07-29 to test whether the editability negative was an artifact of a weak predictor.

## Shared world settings (all runs)
2 objects, `obs_res=128`, fixed reflectivities, bouncing walls, `init_speed=0.28` (initial momentum), death → 4 frames
of pure-noise observation → rebirth at a fresh random initial condition. Collection is `batch=64` parallel worlds ×
`rollout=48` frames per iteration.

> ### ⚠ KNOWN DEVIATION — observation noise (found 2026-07-29)
> **Every GRU run in this table used `obs_noise_std = 0.05`** (the RSSM runs above are clean — they use 0.2).
> **The repo standard is `0.2`** — used by *every* dataset
> (`0_initial` … `8_cont_axis_x`), including dataset 4, which backs the exogenous-action / object-individuation work.
> The 0.05 leaked in from a `scripts/play.py` **display default** and was never a deliberate experimental choice.
> **Consequences:** all endogenous runs share the value, so every comparison *within* this thread (actor vs observer,
> weak vs strong, goal vs no-goal) is internally valid. But **absolute observation RMSE, the noise floor (0.066 here vs
> ≈0.2 for the standard) and probe R² are NOT comparable to the earlier notebooks** — less noise makes both prediction
> and probing easier, so these numbers are optimistic relative to prior work.
> **Fixed going forward:** `scripts/train_endogenous.py` now takes `--obs-noise` and **defaults to 0.2**. A matched
> re-run at the standard noise is the check that these conclusions survive; until it is done, do not cross-cite RMSE or
> R² between this thread and the exogenous-action notebooks.

## Metric conventions specific to these runs (read before interpreting the training curves)
| logged name | formula | caveat |
|---|---|---|
| `mean_reward` | **mean reward per step** over the (64 × 48) collection window — *not* a per-episode total | +0.1 survive / −1.0 death, so **+0.1 is the maximum** (every step survived) |
| `survival` | `batch·rollout / max(deaths, 1)` = `3072 / max(deaths, 1)` — mean frames per life | **capped at 3072** and *quantized* (3072, 1536, 1024, …) because it is bounded by the 3072-frame measurement window, **not** by the world (episodes are unbounded; only death ends one). Prefer **deaths per 1000 frames** for an unbounded, linear read. |
| `coll_rate` | `deaths / (batch·rollout)` | unbounded, linear — the preferred survival statistic |
