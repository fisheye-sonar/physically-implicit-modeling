# Trained editability — canonical RUN REGISTRY

**The single source of truth for what every run name in `trained_editability/` means.** Per `CLAUDE.md`: no notebook
may use a run code without copying its row into its own definitions table, and **figures use the descriptive label,
never the raw code**. Adding a run means adding its row here in the same commit.

Branch `more_trained_editability` (opened 2026-07-30). Direction: `research/directions/trained-editability.md`.
Checkpoints in gitignored `runs/trained_editability/<code>/`.

## The question this thread exists to answer

Every §4 result before this used **inference-time** editors on a **frozen** world model, and every probe-directed one
fails. `learn_to_edit` asked whether *training* fixes it and returned two negatives — both under a deliberately
**light** budget, with the heavier fine-tune left **OWED**. These runs pay that debt.

## Shared setup

Every arm starts from **`runs/controls/H256`** (GRU H=256, `datasets/4_fixed_refl_inview`, obs noise 0.2, position
noise 0.04, 400 epochs — see `../controls/CONTROL_RUNS.md`), trained by `scripts/train_editable_gru.py`, batch 64,
Adam lr 1e-4, seed 0, `K=15` optimised rollout steps.

**Losses.** `edit = MSE(rollout(h_edited, K), clean_obs[ef:ef+K])`; `retention` = ordinary teacher-forced next-step
MSE on test sequences; `total = edit + retention_weight · retention`.

**Held-out by construction.** Training uses `edits[2000:]`; **every reported number is on `edits[:64]`** — the same
samples the `controls/` notebooks report on, so results are directly comparable.

## Runs

| code | descriptive label (use this in figures) | what is trained | steps | retention weight | training edits |
|---|---|---|---|---|---|
| `H256` | **base model (no editability training)** | nothing — the frozen baseline (lives in `runs/controls/`) | — | — | — |
| `FT_light` | **fine-tuned · light (300 steps)** | the world model | 300 | 1.0 | both objects |
| `FT_heavy` | **fine-tuned · heavy (3000 steps)** | the world model | 3000 | 1.0 | both objects |
| `FT_heavy_noret` | **fine-tuned · heavy · no retention** | the world model | 3000 | **0.0** | both objects |
| `FT_heavy_obj0` | **fine-tuned · heavy · object-0 edits only** | the world model | 3000 | 1.0 | **object 0 only** — the content-generalisation control |
| `AMORT` | **amortized editor (world model frozen)** | `E_θ(h,target)→Δh` (2×512 MLP) only | 3000 | — | both objects |

## The three write mechanisms the notebook distinguishes — do not conflate them

| name | what it is | the question it answers |
|---|---|---|
| **Trained interface** | readout injection through **the exact frozen probe the arm was fine-tuned for** (saved per-run as `frozen_probe.npz`), or the learned `E_θ` for `AMORT` | did the training work *at all*? |
| **Readout injection (fresh probe)** | the same *mechanism*, but with a linear probe re-fit on the fine-tuned model's own states — this is what `scripts/eval_controls.py` computes | **mechanism generalisation** |
| Global-PCA projection · PCA geodesic · MLP-probe gradient · Decoder gradient | the rest of the standard §4 suite, untouched by training | do other mechanisms benefit? is the target still reachable? |

> **The probe is frozen from the BASE model and never refit.** That is the point: nothing about the editor is
> learned, so all adaptation is in the world model, which must learn to honour writes along `A⁺` as "put the object
> here". A run's `frozen_probe.npz` therefore travels with it — evaluating an arm with a *different* probe is the
> mechanism-generalisation test, not the "did it train" test.

## Metrics

The canonical §4 set from `../METRICS_AND_EDITORS.md`, implemented in `scripts/editability_metrics.py`
(**Edit Index** and **Edit Index by step**, plus Target / Ghost / Collateral / Edit-frame / GT-traj RMSE and the
fidelity ratio), computed by `scripts/eval_controls.py --root runs/trained_editability`. Nothing is re-derived.

> **Always read the Edit Index against that arm's own unsteered row.** A fine-tune that damages prediction moves the
> whole scale: the unsteered index tracks next-step RMSE with r = +0.987 across models, so a worse predictor has a
> *higher* unsteered index for free. Every table here therefore reports **Δ from that arm's own unsteered**, not the
> raw index alone. `FT_heavy_noret` is the live example — its prediction degrades to the noise floor and its
> unsteered index rises from −0.68 to −0.39 without any editing improvement.
