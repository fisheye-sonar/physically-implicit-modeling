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
**every observation-space error is scored against the CLEAN render** (`clean_obs`), never the noisy `obs`;
compare like-for-like (same metric set + units across anything compared); every comparison figure has a
GT/reference column; waterfalls follow the fixed spec in `../../../CLAUDE.md`.

> ### ⛔ Observation-space error is scored against `clean_obs` — always (added 2026-08-04)
> This applies to **every** obs-space error, including one-off panels that do not run the §4 scorecard — that is
> exactly where it keeps breaking (`editability_structure` and `rssm_structure/rssm_state_geometry` each built a
> `gt_obs = edits.obs[...]` panel titled "error vs post-edit GT"). It also governs the **GT column of every
> waterfall**, which is the clean render.
> **Why it is not cosmetic:** errors add in quadrature, `err_noisy ≈ √(err_clean² + noise²)`, and on the canonical
> dataset the noise term is **0.1539**. A true 0.05 reads as 0.162 and a true 0.10 reads as 0.185 — methods
> differing **2× in real error differ by 14%** on the noisy scale. It compresses every method toward the noise
> level, and it cannot be undone from the reported number afterwards.
> **The noise floor is not a floor here.** `noise_floor_rmse` (= RMSE(`obs`, `clean_obs`)) bounds error only
> against *noisy* targets. Against clean, a perfect predictor scores **0** and sub-noise-floor values are the
> normal result of a recurrent model denoising many frames — not a leak. Use the line as a *reference scale*
> ("no better than echoing the input"), never as a bound.
> If a noisy-referenced quantity is genuinely wanted, it must carry `vs noisy` in its **name, axis label and
> legend**, and must never sit on a shared axis with a clean-referenced one without both being labelled
> (`pim/eval/controllability.py` is the reference: every field is suffixed `_vs_clean` / `_vs_noisy`).

> **Before adding a metric to this registry, check it is not derivable from ones already here.** A metric that is
> an algebraic function of two existing ones adds no information, grows the zoo, and *reads as a contradiction*
> when the reader cannot see the identity. Example caught 2026-08-03: `relative residual`
> `‖composed − direct‖/‖direct‖` was reported beside `cosine` and `magnitude ratio r`, but
> `residual² = r² + 1 − 2·r·cos θ` — fully determined by the other two. Report it *instead of*, not *alongside*.
> Same test applies to a figure panel, not just a table column.

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

### §4 — Editing / object-handle  ⭐ **CANONICAL SET, revised 2026-07-30**

> **Implemented once in `scripts/editability_metrics.py`.** Import it; do **not** re-derive these formulas in a
> notebook. Prose definitions here, code there — they must agree.

**Two ground-truth worlds.** At the edit frame `ef` both can be rendered, and every §4 metric is defined against them:
- **`gt_edited`** = `edits.clean_obs[ef]` — the world where the teleport happened.
- **`gt_unedited`** = the counterfactual where it did **not**: the edited object continued from its `ef-1` position
  along its own velocity, the other object at its true `ef` position. (Rendered by `build_edit_zones`.)

**Ray zones** (per sample, from the two renders — so partial occlusion needs no special-casing):
`target rays` = rays the edited object occupies in `gt_edited`; `ghost rays` = rays it occupied pre-edit and now
vacates; `collateral rays` = the **other** object's rays (it must not move); `differing rays` = where the two worlds
differ at all (`|gt_edited − gt_unedited| > 1e-3`) — the support of the Edit Index.

**Layer 1 — absolute error vs ground truth, decomposed by zone.** All at rollout **step 0** (which decodes frame `ef`),
all RMSE against `gt_edited`, all in observation-intensity units, all lower-is-better, no normalisation:
| name | formula | units | better | notes |
|---|---|---|---|---|
| **Target RMSE** | `RMSE(edited₀, gt_edited)` over **target rays** | obs | ↓ | did the object appear where it should? |
| **Ghost RMSE** | `RMSE(edited₀, gt_edited)` over **ghost rays** | obs | ↓ | did it leave where it was? (replaces "ghost ratio") |
| **Collateral RMSE** | `RMSE(edited₀, gt_edited)` over **collateral rays** | obs | ↓ | was the other object left alone? |
| **Edit-frame RMSE** | `RMSE(edited₀, gt_edited)` over **all rays** | obs | ↓ | the whole scan at the edit frame |
| **GT-traj RMSE** | `mean_s RMSE(edited_s, clean_obs[ef+s])` over the K-step rollout | obs | ↓ | did the edit achieve *and hold* the true post-edit world? |
| **fidelity ratio** | `GT-traj RMSE(editor) / GT-traj RMSE(unsteered)` | ratio | ↓ | **> 1 = the edited rollout ended FURTHER from the true post-edit world than doing nothing** — the edit degraded the model rather than steering it. Always report beside any success claim. |

**Layer 2 — the Edit Index, the calibrated headline.** On the differing rays, is the output closer to the world where
the edit happened, or the one where it didn't?
| name | formula | units | better | notes |
|---|---|---|---|---|
| **Edit Index** | `(d_uned − d_edit)/(d_uned + d_edit)`, `d_· = RMSE(edited₀, gt_·)` over **differing rays**; per sample, then averaged | −1…+1 | ↑ | **+1** = it *is* the edited world · **0** = equidistant (ambiguous, or garbage) · **−1** = it *is* the unedited world. |
| **Edit Index by step** | the same, at every rollout step, against the counterfactual world **rolled forward** (the edited object continuing along its own velocity, the other object on its true trajectory) | −1…+1 | ↑ | the bounded trajectory analogue of GT-traj RMSE. **Report it whenever you report the step-0 index** — landing an edit and *holding* it are different things (measured 2026-07-30: the decoder-gradient oracle scores **+0.94** at step 0 and decays to **−0.12** by step 14). |

> **Read the index against that model's own unsteered row.** A *perfect* predictor scores exactly −1 when
> unsteered; a real one falls short by its own blur, because `d_unedited` is its one-step prediction error rather
> than 0. Verified 2026-07-30: across the 8 controls models, unsteered Edit Index tracks next-step RMSE with
> **Pearson r = +0.987** (−0.85 for the best predictor, −0.52 for the worst). So the −1 end of the scale sits at a
> slightly different place per model and the unsteered row must appear in every table. The **+1** end is not
> shifted — scoring `gt_edited` itself returns exactly +1.0 (both boundary cases are asserted).

**Why the index is hard to game** — this is the point of it. An output far from *both* worlds (scrambled, collapsed)
has `d_edit ≈ d_uned` and scores **≈ 0**, not a spuriously good value. "Dim everything toward background" — which
scores perfectly on any ghost-only metric — also cancels, because the differing rays include target rays (where
dimming is wrong) as well as ghost rays (where it is right). And this repo's dominant observed failure, *paint a copy
at the target while keeping the ghost*, correctly reads **≈ 0**.

Supporting/state-space metrics (unchanged):
| name | formula | units | better | notes |
|---|---|---|---|---|
| readout RMSE | position RMSE of the linear probe read off the edited state vs the teleport target | pos | ↓ | state-space, pre-rollout |
| anti-reversion | `mean_{s=10..14} RMSE(edited_s, unsteered_s) / RMSE(edited₀, unsteered₀)` | ratio | ↑ | *stickiness, NOT correctness*; an edit can stick while drifting off-distribution — use GT-traj RMSE for correctness. The one §4 metric still referenced to the unsteered rollout, because it is genuinely about *change*. |
| leave-out local-PCA resid | `‖q − proj_local(q)‖ / ‖q − local mean‖`, local PCA on k NN excluding q's own NN; ref = real states | frac | ↓ | manifold residency of the edited state |
| global-PCA hull resid | `‖h − proj_global(h)‖` onto the global var-threshold subspace; ref = real states | ‖h‖ | ↓ | off-manifold-ness |
| content-gen ratio | `EditIndex(y-edit) − EditIndex(x-edit)` on the passive latent | index pts | → 0 | for axis-restricted models; ≈0 ⇒ generalises across content ⇒ real object |

> ### ⚠ RETIRED 2026-07-30 — do not use, and do not cite their numbers as comparable
> **`reach (% of swap)`, `collateral (% of swap)`, `selectivity`, `ghost ratio`, `obs-change (% of swap)`.**
> All measured **change away from the unsteered rollout**, normalised by the true-state swap. Two fatal problems:
> (1) they scored *change*, not *correctness*, so an editor that merely **scrambled** the observation posted a huge
> "reach" — 400–440% was observed at `H=8`/`H=32`, and the decoder-gradient oracle posted 209–327%, where 100% was
> supposed to be the ceiling; (2) the denominator was the true-state swap, a **soft, model-dependent** reference whose
> own strength varied widely across models (its ghost ratio ranged 0.315–0.868 across the noise-ablation cells), so
> the same physical edit scored differently on different models — fatal for cross-model sweeps.
> The replacement above fixes both: one fixed ground-truth reference, and a bounded index that garbage cannot game.
> **Historical numbers in older notebooks/notes are on the retired scale** and are not comparable to anything computed
> after 2026-07-30; the notebooks re-run on the new set are `00_master_editability` and everything under `controls/`.

**§S — sharpness / predictive quality (blur watch-item, multistep notebooks):** next-step RMSE vs **clean**;
open-loop horizon RMSE vs clean; rollout total-variation ÷ GT-TV (≈1 = as sharp as GT, <1 = blurry mean-hedging).


---

## Editors (write mechanisms on `h`) and references

**References (never editors):**
- **GT (sim)** — the simulator's time-evolving clean post-edit obs. The target, never a model output.
- **Unsteered** — rollout from the un-edited warm-up state `h0`.
- **Oracle observation** *(renamed 2026-07-30; was "true-state swap", a misnomer — nothing about the state is
  swapped)* — teacher-force **one extra frame**, the **REAL (noisy) post-edit observation `edits.obs[ef]`**, then
  roll out. The model simply gets to *see* the teleport happen. A **soft** reference (belief-inertia-limited: one
  frame of evidence only partly updates the state) — **not** a hard ceiling; a direct latent write could exceed it.
  It is fed the *noisy* observation, matching what the model sees at training time; the *clean* render is used only
  as the GT reference the metrics score against. It **leads the other columns by one frame**; label it, never
  re-align the others to it.

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
| Learned amortized editor `E_θ(h,target)→Δh` | train a net on TRAIN edits, eval on **held-out** | `learn_to_edit` Variant A; **re-run at scale in `trained_editability/`** | 2026-07-30: at 3000 steps it is the **best learned mechanism** (+0.54 Edit Index vs its own unsteered) and costs the frozen world model nothing — but reaches only "equidistant", and does not transfer to any other write mechanism |
| Fine-tune for editability | fine-tune the GRU so a fixed pseudo-inverse editor works | `learn_to_edit` Variant B (light); **`trained_editability/finetune_for_editability.ipynb` (heavy — the OWED debt, PAID 2026-07-30)** | light 300 steps +0.04, heavy 3000 steps +0.13 index points, so the light negative was partly a budget artifact — but it is a **button**: no transfer to a freshly-fit probe (+0.02) or to a withheld object (−0.17), and it costs 13% of next-step prediction |
| Interleaved latent steering | push readout a little → decode → feed the model's **own** obs back → repeat (S steps, η/step; `+manifold` = project each step) | `multistep_steering` 1a | fails (drags both objects); self-generated obs, not external |
| Freeze-time teacher forcing | render the edit over N frames (edited obj interpolates to target, other held), teacher-force those **externally-rendered** frames, then unfreeze | `multistep_steering` 1b | **works** (lands + clears ghost, N≈3–8); replicates on RSSM. The success requires *externally-rendered* obs (the true renderer), not the model's own machinery. |

**Architecture-specific editors — transformers only** (added 2026-08-04, `transformers/transformer_world_state.ipynb`).
A causal transformer has **two** state objects, so "write to the state" splits in two. Do not report these as
the same kind of intervention, and do not compare either directly to a GRU `h` write without saying which:

| editor | acts on | mechanism | persists? |
|---|---|---|---|
| **Activation edit (residual point ℓ)** | the **readable** state — residual stream at layer ℓ, current position | any `h`-editor (readout injection, decoder gradient, …) applied to the activation vector at ℓ | **no, by construction** — the next step recomputes the stream from the observation buffer. Decay here is **architecture, not the GRU's reversion failure**; a one-step effect is the ceiling, so never call it "reverts" or "collapses". |
| **History overwrite (n frames)** | the **carried** state — the newest `n` frames of the observation buffer | replace them with renders of the counterfactual world (the object travelling a line that arrives at the target) | **yes** — the only channel that persists. The transformer's form of the GRU's *counterfactual state overwrite*. |

Build the overwritten state through `state_from_obs` so buffer padding and the `length` mask stay correct, and
cap the sweep at the history that actually exists (at `ef = 20`, a model with `state_span = 61` has an
*effective* carried state of 20).

**Metric introduced with the sweep (fold into §4 if it recurs):**

| metric | formula | units | better | reading |
|---|---|---|---|---|
| **saturation point** | smallest `n` whose Edit Index ≥ `0.9 × max_n(Edit Index)` for **that same model** | frames (also reported as % of `state_span`) | ↓ | the point past which more overwritten history buys nothing. Report **both currencies**: whichever is flat across window sizes is the real requirement. Prefer it to the **crossover point** (smallest `n` with Edit Index > 0), which a single frame can clear and which therefore does not discriminate. |

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
