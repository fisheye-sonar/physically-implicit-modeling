# Learn-to-Edit — can editability be *induced*? (frozen learned editor + light fine-tune)

**Date:** 2026-07-09 · **Direction:** `research/directions/learn-to-edit.md` · `[reframe]` · sub-Q3 (editability)
**Notebook:** `notebooks/experiments/editability/learn_to_edit.ipynb` (executed, GRU, GPU) · PNGs → `/tmp/learn_to_edit/`
**Model:** `runs/gru/3_dset3_gru_persistentids_inview_400epochs` (H=256) · **Data:** `datasets/4_fixed_refl_inview` (edits split, edit_frame=20, K=15 post-edit rollout)

→ FLAG FOR PROMOTION

---

## TL;DR verdict

**Editability could NOT be cleanly induced on this GRU** — neither by a frozen amortized editor (Variant A)
nor by a light fine-tune (Variant B), at the few-shot budget the probes used. Both variants show the
**memorization signature**: train obs-loss collapses while **held-out** obs-loss barely beats unsteered.
The *only* method that produces clean, ghost-free, on-target held-out edits is the **obs-gradient oracle**
(per-sample Adam on `h` against GT clean rollout) — but it lands **far off-manifold** (resid 6.8 vs real-state
1.75), i.e. it finds a per-sample latent that is *not* a state the model ever visits and does *not* generalize
into a function. This is the **information-present-but-not-controllable-via-a-single-function** reading, leaning
structural: the info to reconstruct the post-edit obs is recoverable per-sample, but no learned map
`(h,target)→h_edit` (frozen) and no light fine-tune made a *fixed* editor work on unseen edits.

Honest caveats up front: (a) these are *light/short* training runs (E_θ ~2.5k iters; fine-tune 1.5k iters,
lr 2e-4) — a heavier fine-tune or a bigger editor with a proper data budget was not exhausted; (b) held-out
d_gt *does* improve with more editor training data (Variant A data-scaling), so "structural" here means
"not few-shot-reachable and not clean," not "provably impossible." See "what would change the verdict."

---

## Setup / fairness controls (all HARD requirements met)

- **Same small data budget as probes:** default `N_TRAIN=256` edit examples for E_θ; fine-tune budget 1024.
- **Held-out edits:** disjoint `N_HELD=512` unseen edits; every editor and both models evaluated on the
  *same fixed held-out set*.
- **Head-to-head vs the failed baselines** on the same held-out edits, obs-space:
  probe-pinv (pos), probe-pinv (pos,vel), global-manifold (pos,vel) POCS, obs-gradient (oracle), + unsteered.
- **Metrics (obs-space):** d_gt (RMS gen-obs vs GT post-edit clean rollout, mean over K + per-step),
  d_tgt (vs rendered target at step0), ghost ratio (pre-edit-vacated zone), selectivity (RMS error on the
  NON-edited object's rays vs GT), off-manifold residual of `h_edit` (global PCA @90% var).

---

## Variant A — frozen learned editor  (cells [4]–[11]; Figs 1–4)

`E_θ:(h, target8)→Δh`, frozen GRU, loss = obs-space K-step rollout MSE vs GT clean + 0.01·off-manifold penalty.

### Held-out head-to-head (N_HELD=512, K=15) — cell [6]
| variant | d_gt | d_tgt(s0) | ghost | sel_err | resid |
|---|---|---|---|---|---|
| unsteered | 0.2823 | 0.2706 | 1.000 | 0.1360 | 1.735 |
| probe-pinv (pos) | 0.2821 | 0.2679 | 0.991 | 0.1368 | 1.808 |
| probe-pinv (pos,vel) | 0.2818 | 0.2677 | 0.990 | 0.1364 | 1.837 |
| manifold (pos,vel) | 0.2799 | 0.2625 | 0.943 | 0.1524 | **0.000** |
| **obs-gradient (oracle)** | **0.0617** | **0.0519** | **0.089** | **0.0654** | **6.814** |
| learned E_θ | 0.2740 | 0.2723 | 0.649 | **0.2738** | 1.630 |

- The learned editor **barely beats unsteered on d_gt** (0.274 vs 0.282) and **makes selectivity WORSE**
  (0.274 vs 0.136) — it smears the whole scene rather than moving one object. Ghost improves partially
  (0.65) but not to the oracle's 0.09.
- Only the **obs-gradient oracle** solves the edit — but at resid **6.81** (real states ~1.75). It uses GT and
  optimizes each sample independently; it is a structure *probe*, not a deployable/generalizing editor.
- The (pos,vel) linear-probe pseudo-inverse and manifold projection are **indistinguishable from unsteered**
  in obs space — reconfirms the prior "readable≠controllable" failure.

### Memorization diagnostic — cell [6]
| | d_gt | ghost | sel_err | resid |
|---|---|---|---|---|
| TRAIN | 0.0924 | 0.088 | 0.1385 | 0.085 |
| HELD-OUT | 0.2740 | 0.649 | 0.2738 | 1.630 |

train→heldout d_gt gap **+0.18** — the editor **overfits the training edits** (train d_gt 0.09 ≈ oracle-level)
but does not generalize. This is the interpretation-guard result: **memorization, not controllability**.

### Data-scaling (Fig 4) — cell [10], held-out fixed
| N_TRAIN | train d_gt | HO d_gt | HO ghost | HO sel | HO resid |
|---|---|---|---|---|---|
| 64 | 0.087 | 0.285 | 0.785 | 0.237 | 1.78 |
| 128 | 0.089 | 0.280 | 0.687 | 0.253 | 1.71 |
| 256 | 0.092 | 0.276 | 0.643 | 0.258 | 1.61 |
| 512 | 0.098 | 0.270 | 0.593 | 0.267 | 1.44 |
| 1024 | 0.106 | 0.252 | 0.501 | 0.247 | 1.34 |
| 2048 | 0.124 | 0.220 | 0.396 | 0.215 | 1.42 |
| unsteered | — | 0.285 | 1.000 | 0.128 | 1.71 |

Held-out d_gt improves *monotonically but slowly* with data (0.285→0.220) and never approaches train
quality; ghost drops (0.79→0.40); **selectivity stays worse than unsteered at every budget** (never restores
the non-edited object). So: the information needed is *partially present and reachable with enough data*, but
a frozen amortized editor at the probe's few-shot budget **cannot induce clean, selective control**.

---

## Variant B — light fine-tune for editability  (cells [12]–[15]; Fig 5)

Fine-tune the GRU (all params, lr 2e-4, 1.5k iters, budget 1024) so a **fixed** (pos,vel) pseudo-inverse
editor (probe re-fit on current states every 100 iters, detached) induces the rollout; + a prediction-fidelity
anchor on ordinary test rollouts.

### Held-out fixed-editor + fidelity — cell [13]
next-step obs RMSE: orig 0.1536 → fine-tuned 0.1606 (world model kept, slight degradation).

| model | d_gt | d_tgt(s0) | ghost | sel_err | resid |
|---|---|---|---|---|---|
| ORIG + fixed editor | 0.2853 | 0.2692 | 0.992 | 0.1294 | 1.812 |
| FT + fixed editor | 0.2743 | 0.2592 | 0.940 | **0.1548** | **2.944** |
| FT unsteered | 0.2867 | 0.2683 | 0.985 | 0.1509 | 2.834 |

Fine-tune moves the fixed editor **almost nothing** on held-out (d_gt 0.285→0.274, +0.011; ghost 0.99→0.94),
**worsens selectivity** (0.129→0.155), and **pushes h_edit further off-manifold** (1.81→2.94). Train edit-loss
dropped 0.086→0.023 while held-out edit-loss barely moved (0.079→0.075) — **same memorization signature**.

### Canonicity re-measurement — cell [14] (the payoff test)
| metric | ORIGINAL | FINE-TUNED | direction for "more canonical" |
|---|---|---|---|
| fiber resid ‖h−g‖/‖h‖ (MLP) | 0.3822 | 0.3828 | lower — **no change** |
| R²(h) from (pos,vel) (MLP) | 0.829 | 0.808 | higher — **worse** |
| linear pos R² (readability) | 0.841 | 0.789 | higher — **worse** |
| linear vel R² (readability) | 0.484 | 0.351 | higher — **worse** |
| off-manifold resid (own PCA) | 1.744 | 1.727 | lower — flat |
| PCA comps @90% var | 38 | 50 | fewer — **worse (higher-dim)** |

**Inducing (attempted) editability did NOT make the state more canonical — it slightly de-canonicalized it**
(readability down, dimensionality up). Since the fine-tune also failed to actually induce editability, this
does not falsify the `editability⟺canonical` hypothesis; it just fails to support it. What it *does* show: a
light fine-tune on this objective drifts the representation *away* from canonical rather than toward it.

---

## Interpretation (honest)

- **Not a memorization-masquerading-as-control false positive.** The held-out gap, flat held-out curve, and
  worsened selectivity all point the same way: neither variant delivered clean held-out control.
- **Not a "can't even overfit" pure-structural result either.** Variant A overfits trains perfectly, and the
  obs-gradient oracle solves any single sample — so the info to reconstruct the post-edit obs *is* present in
  reach of `h`. The failure is that it's reachable only **off-manifold and per-sample**, not by a fixed or
  amortized *function* of `(h,target)` — and not few-shot.
- **Net:** supports the RESEARCH.md thesis that this implicit GRU **resists editability** — the controllable
  code is not a clean function of the readable code, and a light touch (frozen editor / light fine-tune) does
  not fix it. Strength: medium (light runs; scaling helps slowly), not "impossible."

## What would change the verdict (open questions)
1. **Heavier fine-tune / more params-budget** (Variant B run longer, higher lr, or a dedicated editability head
   trained jointly with the model) — did we just under-train? The fidelity anchor + light budget may be too weak.
2. **Data budget honesty:** the probes' budget is small; a "controllability with N edits" curve (Fig 4)
   suggests clean control might need ≫ few-shot. Worth a heavier E_θ at N=2048 with the on-manifold penalty
   tuned, to see if held-out selectivity can ever recover.
3. **RSSM generalization pass** (`runs/rssm/4_dset4_refined_best/`, `model.sample=False`) — brief lists it as
   the generalization pass; not run here. The stochastic-latent RSSM may be more/less editable.
4. **Off-manifold penalty weight:** λ=0.01 kept E_θ near-manifold (resid 1.63) but that *hurt* d_gt vs the
   off-manifold oracle. There may be a genuine tension: on-manifold ⇒ can't reach the edit; reach the edit ⇒
   off-manifold. Sweeping λ would map that trade-off (this is itself a structural signature).

## Deliverables
- Notebook executed: `notebooks/experiments/editability/learn_to_edit.ipynb` (16 numbered code cells, Figs 1–5).
- PNGs in `/tmp/learn_to_edit/`: fig1_A_train_headtohead, fig2_A_stepcurves, fig3a_A_scans, fig3b_A_waterfalls,
  fig4_A_datascaling, fig5_B_finetune, fig5d_B_waterfalls.

---

## v2 revision (2026-07-09) — legibility + completeness pass (re-run on GPU)

Revised the notebook IN PLACE per the direction's REVISION PASS v2 block. **This was legibility + completeness,
not new science; the v1 verdict is UNCHANGED — editability still could NOT be cleanly induced (memorization
signature in both variants).** What changed:

1. **Definitions table up front** (new markdown cell after setup): every metric — `d_gt`, `d_gt_step`, `d_tgt(s0)`,
   `ghost`, `ghost_step`, `sel_err`, `sel_step`, off-manifold `resid`, and the Variant-B canonicity metrics —
   with explicit formula, units, and ↑/↓. No more buried print-sidenote definitions.
2. **RMSE everywhere.** Training-loss curves Fig 1a and Fig 5a now plot `sqrt(MSE)` with `obs RMSE` axis labels
   (they plotted raw MSE in v1); the unsteered reference line on Fig 1a is now `d_gt` (RMSE), not `d_gt**2`.
   All obs-error quantities are RMSE, matching the tables and the rest of the repo.
3. **Same metric suite for A & B + a new Variant-B budget sweep** (cells [13c]/[13d], Fig 4B) mirroring Variant A's
   N_TRAIN sweep. ~4 points (128/256/512/1024) — fine-tuning is much heavier than editor-training (each point ~16 s
   here; noted as a compute trade-off in a markdown cell). B's fixed-editor table already carried the same
   (d_gt, d_tgt, ghost, sel_err, resid) columns as A, now directly comparable.
4. **GT (post-edit) reference column added to every comparison waterfall.** Fig 3b and Fig 5d now have `GT (post-edit)`
   as the leftmost column (v1's FT waterfall lacked it, uninterpretable). Fig 3a scans gained a GT-clean step-0 trace.
   Primary A and B waterfalls are now at the **same train size**: set `FT_BUDGET = N_TRAIN` (256) so the few-shot
   primary comparison is matched; larger budgets live only in the sweeps.
5. **Variant-B setup documented in markdown** (expanded cell [17]): defines ORIG+fixed vs FT+fixed vs FT-unsteered;
   states the (pos,vel) probe is **re-fit detached on the current model's states** (not fixed-weights); that the FT
   objective is a **K=15-step multistep rollout** match to GT clean post-edit obs + a next-step prediction anchor;
   that eval injects a **held-out** target via the pseudo-inverse and rolls out K. Also corrected a v1 mismatch: the
   markdown claimed "decoder + low-rank adapter" but the code fine-tunes **all params** — markdown now matches code.
6. **Data-source provenance inlined**: GRU fiber resid **0.337** / R²(h) **0.867** cited from
   `2026-07-08-diagnostic-corrections.md` Sec.2 (used as the canonicity reference); obs-gradient **~15.7**
   off-manifold and real-state resid **~1.7** cited from `candidate-editability.md` / this notebook's cell [1].

### New Variant-B fine-tune data-scaling numbers (Fig 4B; FT+fixed editor, fixed held-out N=512, K=15)

| FT_BUDGET | d_gt | d_tgt(s0) | ghost | sel_err | resid |
|---|---|---|---|---|---|
| 128  | 0.2872 | 0.2687 | 0.977 | 0.1405 | 2.709 |
| 256  | 0.2833 | 0.2650 | 0.964 | 0.1469 | 2.673 |
| 512  | 0.2782 | 0.2616 | 0.953 | 0.1502 | 2.862 |
| 1024 | 0.2726 | 0.2588 | 0.932 | 0.1549 | 2.873 |
| ORIG (no FT) | 0.2853 | 0.2692 | 0.992 | 0.1294 | 1.812 |

**This mirrors Variant A's data-scaling and reinforces the v1 verdict.** With more fine-tune data the held-out
`d_gt` improves only slowly (0.2872→0.2726, barely below the no-FT ORIG 0.2853) and never approaches control;
`ghost` drops modestly (0.977→0.932); **`sel_err` gets monotonically WORSE** with budget (0.1405→0.1549, all worse
than ORIG's 0.1294 — fine-tuning smears the non-edited object); and `h_edit` stays firmly **off-manifold**
(resid ~2.7–2.9 vs real-state ~1.75). So the light fine-tune route is *also* not few-shot-inducible and does not
buy clean, selective control even as budget grows — the same "info present but not reachable by a fixed function,
memorization at few-shot" reading as Variant A.

Re-run canonicity (cell [14]) reproduced v1 within noise: ORIGINAL MLP fiber resid 0.382 (≈ the 0.337 reference
order-of-magnitude), and fine-tuning **de-canonicalizes** (MLP fiber resid 0.382→0.407, R²(h) 0.829→0.798,
lin pos R² 0.841→0.804, lin vel R² 0.484→0.367, PCA comps 38→48). Unchanged conclusion: inducing (attempted)
editability did not make the state more canonical.

**Verdict change: NONE.** v1 stands. Executed clean (0 errors, 8 inline figures). Shaky/caveat flags unchanged
from v1 (light/short runs; scaling helps slowly ⇒ "not few-shot-reachable & not clean," not "provably impossible";
RSSM generalization pass still not run). One cosmetic: a `requires_grad→scalar` UserWarning in the FT history
logging was fixed in source (`float(x.detach())`) after the run — values identical, so cached outputs are correct.

### v2 PNGs (`/tmp/learn_to_edit/`)
fig1_A_train_headtohead.png, fig2_A_stepcurves.png, fig3a_A_scans.png, fig3b_A_waterfalls.png,
fig4_A_datascaling.png, **fig4B_B_datascaling.png (new)**, fig5_B_finetune.png, fig5d_B_waterfalls.png.
