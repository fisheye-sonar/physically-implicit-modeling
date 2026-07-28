# Editability — canonical METRICS & EDITORS registry

**One source of truth for the names, formulas, and definitions used across the `editability/` notebooks**
(including its `actions/`, `multistep/`, `rssm_structure/` subdirs). Goal: stop each notebook re-inventing
terms. This is a *provisional* proposal layer (like `00_master_editability.ipynb`), not `pim/` code.

**How to use it.** A notebook picks the **subset** it needs and copies the exact name + formula + units +
better-direction from here into its own definitions table (so the notebook still stands alone). It is fine —
expected — for a notebook to **not** use every entry, and to **go outside this list** when it is specifically
testing a *new* metric or editor (e.g. `multistep_steering` proposes new edit *mechanisms* and does not run the
full suite). When a notebook adds something genuinely new and it recurs, fold it back into this file.
**Consistency rules that always apply:** RMSE (never MSE); obs intensity ∈ [0,1]; positions in sim units;
compare like-for-like (same metric set + units across anything compared); every comparison figure has a
GT/reference column; waterfalls follow the fixed spec in `../../../CLAUDE.md`.

---

## Metrics

### §1 — Geometry of the visited-state manifold
| name | formula | units | better | notes |
|---|---|---|---|---|
| PCA hull @p% | # PCA comps of the visited-`h` bank reaching ≥ p% variance | dims | — | linear-hull upper bound; report @70/90/95 |
| intrinsic dim (TwoNN) | `d = 1/mean(log(r₂/r₁))`, r₁,r₂ = 1st/2nd-NN dist (Facco 2017) | dims | — | model-free; load-bearing geometry number |
| intrinsic dim (MLE) | Levina–Bickel over k=20 NN, ×(k−2)/(k−1) | dims | — | model-free; report alongside TwoNN |
| tangent rotation ("curvature") | mean principal angle between local-PCA tangents (k=64 NN, top-8) of a state and its NN | deg | ↓ flatter | **⚠ NOT distance/scale-normalized** — absolute degrees are a density/latent-scale artifact, **not comparable across notebooks/architectures**. Compare only within one notebook at fixed density. Fix OWED: `research/directions/curvature-metric-normalization.md`. |

### §2 — Recoverability (probe read-out of the physical statistic)
| name | formula | units | better | notes |
|---|---|---|---|---|
| position / velocity R² | `1 − ‖Y − probe(h)‖²/‖Y − Ȳ‖²`, Y ∈ {pos, vel} | — | ↑ | report **linear (lstsq)** and **MLP** separately — the linear axis carries the interesting signal (MLP often saturates ~0.97) |
| single- vs two-frame | probe on `h_t` vs `[h_{t-1}, h_t]` | — | ↑ | velocity is instantaneously *nonlinear*, not temporal (2f ≈ 1f under MLP) |
| early-t / late-t split | early = frames t<15 (filter not converged); late = t≥15 | — | — | always split velocity readout by this |

### §3 — Canonicality / fiber-collapse
| name | formula | units | better | notes |
|---|---|---|---|---|
| fiber residual | `‖h − g(pos,vel)‖ / ‖h‖`, g linear or MLP | frac of ‖h‖ | ↓ (0 = fully canonical) | large linear→MLP drop ⇒ curved embedding. For RSSM report det-only / stoch-only / full. |

### §4 — Editing / object-handle
The `edited_s` = obs at rollout step `s` from the edited state; `unsteered_s` = from the un-edited state;
`GT post-edit_s` = `edits.clean_obs[ef+s]` (the sim's *time-evolving* true post-edit obs). `obj-k rays` /
`other-object rays` / `ghost rays` are per-sample ray masks (object k's target rays / the other object's rays /
object k's vacated pre-edit rays).
| name | formula | units | better | notes |
|---|---|---|---|---|
| readout RMSE | position RMSE of the linear probe read off the edited state vs the teleport target | pos | ↓ | state-space, pre-rollout |
| GT next-step RMSE | `RMSE(edited₁, GT post-edit₁)` | obs | ↓ | ⚠ ±1 decode convention: GRU `decode(h)` = predict-*next*, RSSM = reconstruct-*current* — align columns accordingly |
| **GT-traj RMSE** | `mean_s RMSE(edited_s, GT post-edit_s)` over the K-step rollout | obs | ↓ | **the direct fidelity metric** — did the edit achieve *and hold* the true post-edit world? (prefer over "persistence") |
| reach (% of swap) | `100·RMSE(edited₀, unsteered₀)[obj-k rays] / RMSE(swap₀, unsteered₀)[obj-k rays]` | % | → 100 | 100% = what a real teleport (the swap) does to obj-k's rays |
| collateral (% of swap) | `100·RMSE(edited₀, unsteered₀)[other-object rays] / RMSE(swap₀, unsteered₀)[obj-k rays]` | % | ↓ | same units as reach; clean handle → 0 |
| selectivity | `reach / (reach + collat)` | frac | ↑ | 1 = moved only obj-k |
| ghost ratio | `mean(edited₀[ghost rays]) / mean(unsteered₀[ghost rays])` | ratio | ↓ | <1 = object left its old location; 1 = ghost remains |
| anti-reversion | `mean_{s=10..14} RMSE(edited_s, unsteered_s) / RMSE(edited₀, unsteered₀)` | ratio | ↑ | *stickiness, NOT correctness* (formerly mislabeled "persistence"); an edit can stick while drifting off-distribution — use GT-traj RMSE for correctness |
| obs-change (% of swap) | `100·RMSE(edited₀, unsteered₀) / RMSE(swap₀, unsteered₀)` (all rays) | % | context | how much the editor moved the obs at all |
| leave-out local-PCA resid | `‖q − proj_local(q)‖ / ‖q − local mean‖`, local PCA on k NN excluding q's own NN; ref = real states | frac | ↓ | manifold residency of the edited state |
| global-PCA hull resid | `‖h − proj_global(h)‖` onto the global var-threshold subspace; ref = real states | ‖h‖ | ↓ | off-manifold-ness |
| content-gen ratio | `reach(y-edit)/reach(x-edit)` on the passive latent | ratio | → 1 | for axis-restricted models; ≈1 ⇒ generalises across content ⇒ real object |

**§S — sharpness / predictive quality (blur watch-item, multistep notebooks):** next-step RMSE vs **clean**;
open-loop horizon RMSE vs clean; rollout total-variation ÷ GT-TV (≈1 = as sharp as GT, <1 = blurry mean-hedging).

---

## Editors (write mechanisms on `h`) and references

**References (never editors):**
- **GT (sim)** — the simulator's time-evolving clean post-edit obs. The target, never a model output.
- **Unsteered** — rollout from the un-edited warm-up state `h0`.
- **True-state swap** — teacher-force the model on the *true* post-edit obs through `ef` → `h_swap` → rollout.
  A **soft** reference (belief-inertia-limited: one frame of teleport evidence only partly updates the state,
  so even the swap doesn't fully clear the ghost) — **not** a hard ceiling; a direct latent write could exceed it.

**Inference-time (training-free) editors — the standard §4 suite:**
| editor | mechanism | needs | notes |
|---|---|---|---|
| Readout injection | set the linear position-probe readout via pseudoinverse (null-space preserved) | linear pos probe | decoder-inert on these models |
| MLP-probe gradient | Adam on `h` through a frozen MLP (pos,vel) probe toward the target | MLP (pos,vel) probe | |
| Global-PCA projection | POCS: alternate inject ↔ project onto the global var-threshold PCA subspace | global subspace | keeps `h` on the linear hull |
| PCA geodesic | constant-step walk toward the target, re-projecting onto a fresh local-PCA tangent each step (K≈120) | local bank | the "stay-on-manifold" editor; canonical structural editor for the scorecard |
| Decoder gradient (**oracle**) | Adam on `h` to match the **GT edit-frame obs** through the decoder (single frame) | GT obs @ ef | nails step-0 obs but off-manifold → **collapses** |

**Learned / multi-step / observation-mediated editors (used in specific notebooks, cite don't recompute):**
| editor | mechanism | where | outcome |
|---|---|---|---|
| Obs-gradient full-rollout (**oracle**) | Adam on `h` to match the **whole K-step GT rollout** (backprop through dynamics) | `learn_to_edit` | **persistent** (optimizes persistence directly) but off-manifold; NOT the single-frame decoder-gradient — different objective |
| Learned amortized editor `E_θ(h,target)→Δh` | train a net on TRAIN edits, eval on **held-out** | `learn_to_edit` Variant A | negative (memorization signature); the "did you try a learned editor" answer — needs held-out + data-scaling |
| Light fine-tune for editability | fine-tune the GRU so a fixed pseudo-inverse editor works | `learn_to_edit` Variant B | negative (light budget); heavier FT still owed |
| Interleaved latent steering | push readout a little → decode → feed the model's **own** obs back → repeat (S steps, η/step; `+manifold` = project each step) | `multistep_steering` 1a | fails (drags both objects); self-generated obs, not external |
| Freeze-time teacher forcing | render the edit over N frames (edited obj interpolates to target, other held), teacher-force those **externally-rendered** frames, then unfreeze | `multistep_steering` 1b | **works** (lands + clears ghost, N≈3–8); replicates on RSSM. The success requires *externally-rendered* obs (the true renderer), not the model's own machinery. |

---

## Conventions & known caveats (apply everywhere)
- **Waterfalls:** one `waterfall_grid(...)` helper per notebook, matching the fixed spec (`../../../CLAUDE.md`):
  `cmap="gray"` on dark bg, ~6 noisy context frames above a dashed edit-frame line, green target / red-dash ghost,
  figure-top legend, GT column. **The edit frame `ef` is shown as one shared row = the TRUE post-edit obs / edit
  target** (`edits.clean_obs[ef]`, identical in every column), marked off by a second dotted line, with **each
  column's model rollout below it = its free-run from `ef+1` onward** (GT column: `clean_obs[ef+1:ef+K]`). This is
  **mandatory** and fixes the GRU ±1 offset. **Alignment:** `warm_up_to_edit` teacher-forces `obs[0..ef-1]`, so the
  rollout's **step-0 is `ef`** (`ROLL[:,0] ↔ clean_obs[ef]`, per the §4 scorecard's `gt_traj_obs = clean_obs[ef:]`);
  to seat the shared true-`ef` row above the free-run, **drop each model column's step-0** (`ROLL[...][1:]`) and use
  `clean_obs[ef+1:ef+K]` for GT (both length `K-1`). Canonical implementation:
  `actions/action_space_object_individuation.ipynb` `editor_waterfall_fig`; also in `actions/action_conditioned_structure.ipynb`
  and `learn_to_edit.ipynb` (Fig 3b/5d).
- **±1 decode convention:** GRU `decode(h_t) ≈ obs[t+1]` (predict-next); RSSM `decode` reconstructs the current
  frame. Align rollout columns to the GT accordingly and footnote it.
- **Clean vs noisy targets:** models train on **noisy** obs (`obs_noise_std=0.2`); evaluate next-step against **clean**
  reconstructed obs (removes the irreducible noise floor). Teacher-forced *inputs* should be noise-matched (0.2) to
  stay in-distribution — the freeze-time win is robust to this (checked).
- **Frozen-target metric:** any "vs static edit-frame render" / target-fill using **frozen** target rays inflates as
  the object moves away — use the **time-evolving** GT (`clean_obs[ef+s]`), or track the object's per-step rays.
- **Curvature:** see the §1 tangent-rotation warning (not normalized; OWED fix).

## Where models & data live (cite, don't recompute)
- **Datasets** (`datasets/`): `4_fixed_refl_inview` (test/edits/train, noisy 0.2 — the canonical eval);
  `5_action_augmented` (discrete-token actions, Exp-2); `6_cont_dxdy` / `7_cont_teleport` / `8_cont_axis_x`
  (continuous-action, object-individuation).
- **Checkpoints** (`runs/`, gitignored): GRU `runs/gru/3_dset3_...` (master GRU) & `7_dset4_gru_400epochs`
  (dataset-4 baseline); RSSM `runs/rssm/4_dset4_refined_best`; multistep `runs/gru_multistep/w{2,5}_...`,
  `runs/rssm_multistep/w{1,2,5}_dset4`; action GRUs `runs/gru/{8_action_cond,9_perturbed_passive,M_dxdy,
  M_teleport,M_axis,M_teleport_ctrl}...`.
- Each notebook states its exact checkpoint/dataset/split in cell [1]; borrowed constants cite their source notebook.
