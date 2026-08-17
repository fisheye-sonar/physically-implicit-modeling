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
| position / velocity R² | `1 − ‖Y − probe(h)‖²/‖Y − Ȳ‖²`, Y ∈ {pos, vel}, **scored on held-out sequences against the TRAIN mean** | — | ↑ | report **linear (lstsq)** and **MLP** separately — the linear axis carries the interesting signal. **Fit both with `pim.extractors.fit_readability_probes`** (below); do not hand-roll a probe. |

> ### ⭐ STANDARD READABILITY PROBES (fixed 2026-08-06) — `pim.extractors.fit_readability_probes`
> Two different MLP probes had been in use and their R² values are **not comparable**: `MLPExtractor` on its
> defaults (**1×128**, scored **in-sample**) in `00_master_editability` and `controls/`, versus a hand-rolled
> **2×256** probe scored **held-out** in `iterative_probing` and `nonlinear_gru`. Two axes were conflated —
> probe *capacity*, and whether the score is in-sample. The second is simply an error; the first makes a shallow
> probe under-report how much is nonlinearly decodable.
>
> **The standard, from now on:** linear = `lstsq`; MLP = **2 hidden layers × 256, ReLU**, 30 epochs, Adam
> `lr=1e-3`; **both fit on the same 80% of SEQUENCES and scored on the same held-out 20%**, R² taken against the
> **train** mean. Split by sequence, never by row — consecutive frames are near-duplicates and a row split leaks
> them across the boundary. `fit_readability_probes` also returns `linear_r2_insample` purely as an overfit
> check; it is never the headline.
>
> **Do not confuse this with the steering probe.** **MLP Grad Steering** writes through a *frozen*
> `MLPExtractor` on the original **1×128** defaults, and that must not change — the editor's published results
> are tied to it. `MLPExtractor`'s default is therefore unchanged and asserted bit-identical in
> `tests/test_standard_probes.py`; depth is opt-in via `n_hidden_layers`. Report readability with
> `fit_readability_probes`, steer with `MLPExtractor`, and never quote one as the other.
>
> **Pre-2026-08-06 MLP R² numbers are on a mixture of the two probes** and should be re-fit before being
> compared across notebooks.
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

> ### ⚖ Calibration — what an Edit Index value actually looks like (added 2026-08-14)
> The index is a *ratio of distances*, so it compresses hard near zero and small positive values are
> much weaker edits than they sound. Writing `r = d_edit / d_uned`, the index is `(1 − r)/(1 + r)`:
>
> | Edit Index | `d_edit / d_uned` | what it looks like in a waterfall |
> |---|---|---|
> | **+0.2** | 0.67 | the object **barely** moves toward the target; the ghost is still clearly there, the new blob is faint, and the rollout often picks up noise. **Not a landed edit.** |
> | **+0.5** | 0.33 | the object is recognisably at the target, ghost mostly gone, some smearing |
> | **+0.7** | 0.18 | clean relocation; this is roughly where the counterfactual-overwrite oracle sits |
> | **+0.9** | 0.05 | essentially the edited world |
>
> **So do not describe +0.2 as "the edit works".** Sevan's read of the trained MLP editor at ≈ +0.2
> (2026-08-14): *"the edits barely land, frequently fall apart, and introduce noise"* — and the
> arithmetic agrees. Report the value **and** say what it looks like; the waterfall is the arbiter,
> which is another reason every claim ships with one.
>
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

> ### ⭐ CANONICAL EDITOR NAMES (fixed 2026-08-05 — use these, and only these, in every figure and table)
> Renamed by Sevan for the slide figures; these supersede all earlier names. **PI** = pseudoinverse.
> **TF** = teacher forcing. Reference implementation of all 17: `editor_gallery/editor_gallery.ipynb`.
> Old → new: `Readout injection` → **Pseudoinverse Injection** · `Global-PCA projection` →
> **Global PCA Projection (PI)** · `PCA geodesic` → **Local PCA Geodesic @k (PI)** (it refits a *local*
> tangent — the old name collided with the global one) · `MLP-probe gradient` → **MLP Grad Steering** ·
> `Interleaved latent steering` → **Multistep Steering (PI) @k** · `Decoder gradient` → **Decoder Grad
> Steering k=1** · `Obs-gradient full-rollout` → **Decoder Grad Steering k=15** · `Freeze-time teacher
> forcing` → **Freeze-time Interp. TF @N** · `Counterfactual state overwrite` → **Counterfactual
> Overwriting** · `Oracle observation` → **First Obs. TF** · `Amortized editor` → **Trained Editor**.

**STANDARD EDITORS — training-free writes to `h`:**
| editor | mechanism | needs | outcome on GRU H256 (N=64) |
|---|---|---|---|
| **Pseudoinverse Injection** | `Δ = A⁺(target − (Ah+b))`, minimum-norm write setting the linear position readout to the target | linear pos probe | **−0.66** vs unsteered −0.68 — inert |
| **Global PCA Projection (PI)** | alternating projections (POCS): inject ↔ project onto the global 90%-variance PCA subspace, 50 rounds | global subspace | −0.52, fidelity 0.98 |
| **Local PCA Geodesic @120 (PI)** | constant-step walk toward the injection target, re-projecting onto a **freshly refit local**-PCA tangent (64-NN) each step | local bank | −0.53, fidelity 1.00 |
| **MLP Grad Steering** | Adam on `h` through a frozen **MLP (pos,vel)** probe (d=8), 200 steps | MLP probe | −0.62 |
| **Multistep Steering (PI) @16** | 16 rounds of: nudge the readout by η=0.2 toward the target, then **decode and feed back the model's OWN prediction**. The observation is **model-generated, never rendered** — that is what separates it from freeze-time. | linear probe | −0.22 but **fidelity 1.32**, collateral 0.429 vs 0.127 — it drags both objects; degradation, not editing |
| **Multistep Steering w/ PCA (PI) @16** | as above plus a global-PCA projection each round | + global subspace | −0.44, **fidelity 1.15** |
| **Iterative Nullspace Projection @29 (R² corrected)** | 29 mutually orthogonal probes (fit → delete row space → refit) spanning 116 dims; inject into all at once (block-orthogonal, exactly solvable). Targets shrunk per probe: `target_k = μ + R²_k(target − μ)`, because a probe with R² ≈ 0 reads the population **mean** on a genuine edited state, not the target. | probe cascade | **−0.37, fidelity 0.93** — best standard editor |

> **Uniform (unshrunk) INLP at k=29 is degenerate** — fidelity 1.57, every zone worse than unsteered, visually
> striped garbage. Always use the R²-corrected targets, or truncate to k≈12.

**LEARNED EDITORS — the world model or the editor was trained** (`trained_editability/`; each row is a
**different world model**, so each must be read against **its own** unsteered index):
| editor | what was trained | outcome (own unsteered → edited) |
|---|---|---|
| **Finetuned Model · light · 300 steps** | world model, retention 1.0 | −0.58 → −0.54 |
| **Finetuned Model · heavy · 3000 steps** | world model, retention 1.0 | −0.61 → −0.47 |
| **Finetuned Model · heavy, no retention · 3000 steps** | world model, retention **0** | −0.39 → −0.30 (its unsteered rose purely from degraded prediction) |
| **Finetuned Model · heavy, object-0 edits only · 3000 steps** | world model; content-generalisation control | −0.63 → −0.54 |
| **Trained Editor · 3000 steps** | `E_θ(h,target)→Δh` (2×512 MLP), **world model frozen** | −0.68 → **−0.14**, fidelity 0.68 — best learned |

All four fine-tuned arms write via **Pseudoinverse Injection through their own frozen probe**
(`frozen_probe.npz`, saved per run); evaluating with a *refit* probe instead is the mechanism-generalisation
test, not the "did it train" test.

**ORACLE EDITORS — given ground-truth access:**
| editor | mechanism | outcome (step 0 → step 14) |
|---|---|---|
| **Freeze-time Interp. TF @8** | freeze the world, teacher-force 8 **externally rendered** interpolation frames, unfreeze | **+0.52 → +0.26** |
| **Counterfactual Overwriting** | teacher-force a fabricated history in which the object always travelled toward the target; overwrite the state | **+0.70 → +0.45** |
| **First Obs. TF** | teacher-force **one** frame — the real (**noisy**) `edits.obs[ef]`. The model simply gets to *see* the teleport. **LEADS every other column by one frame**; label it, never re-align the others. | −0.08 → −0.10 |
| **Decoder Grad Steering k=1** | Adam on `h` so the decoder renders the GT edit-frame observation exactly | **+0.97 → +0.08** — nails the frame, then disintegrates into stripes; fidelity 0.98 |
| **Decoder Grad Steering k=15** | Adam on `h` so the **whole 15-step rollout** matches the GT sequence (backprop through the dynamics) | **+0.83 → +0.77**, fidelity **0.20** — the only editor that both lands *and* persists |

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

**HISTORY editors — the same content through two channels** (added 2026-08-13, `history_editing/`).
These exist as a **matched pair**: identical content, identical displacement `δ`, identical number of frames
`n`, differing **only** in the channel. Reporting one without the other loses the entire point — the pair is
what separates "the edit is incomplete" from "the latent is the wrong port".

| editor | acts on | mechanism | outcome |
|---|---|---|---|
| **Latent history translation · n** (GRU) | `h` | min-norm write so a **stacked** lag probe `A_n : h → [pos(t) … pos(t−n)]` reads `pos(t−k) + δ` at every lag — the believed history, rigidly translated (which preserves velocity) | −0.670 → **−0.585** at n=8, **equal to a matched-norm random write**; only continues past that by degrading |
| **Activation history write · all positions × all residual points** (transformer) | residual stream | pseudoinverse injection at every (residual point ℓ, window position j), re-applied at each layer because the stream is recomputed | readout error 3.289 → **0.000** and Edit Index −0.667 → **−0.631** at fidelity 1.00 — the write lands *exactly* and is ignored |
| **Observation history overwrite · n** | the observation the model consumes | teacher-force / substitute the same `n+1` frames, **rendered** from the translated world | **+0.635** (GRU, n=8) / **+0.681** (transformer, n=8), fidelity 0.60–0.64 |

⚠ **Two controls are mandatory for any history-write claim**, because without them an inert result and a
size-driven result are indistinguishable: a **matched-norm random direction**, and the **landing diagnostic**
below. Both caught real ambiguity here.

⚠ **A stacked lag probe does not have rank `4(n+1)`.** Where velocity is constant,
`pos(t−k) = pos(t) − k·dt·v`, so the least-squares block rows satisfy `A_k ≈ A_pos − k·dt·A_vel` and the row
space collapses onto an **8-dimensional `(pos, vel)` core** for every `n` (measured effective ranks
4/7/8/8/8/9 for n = 0/1/2/4/8/16, against numeric ranks 4/8/12/20/36/68). Compute any chance level from the
**effective** rank, never from the output count — and plot **enrichment**, since the chance level moves.

| metric | formula | units | better | reading |
|---|---|---|---|---|
| **probe readout error (landing diagnostic)** | `‖probe(state) − target‖` before vs after the write, averaged over written sites | sim units (position) | ↓ | **required beside any inert result.** It separates "the write failed" from "the write landed and the model ignored it" — a distinction the Edit Index cannot make, and the two have opposite implications. Generalises §4's `readout RMSE` to a multi-site write. |

---

## Conventions & known caveats (apply everywhere)
- **Waterfalls:** one `waterfall_grid(...)` helper per notebook, matching the fixed spec (`../../../CLAUDE.md`):
  `cmap="gray"` on dark bg, ~6 noisy context frames above a dashed edit-frame line, green target / red-dash ghost,
  figure-top legend, GT column. Below the edit line, **every column shows its OWN free-run starting at step 0**;
  the GT column shows `clean_obs[ef:ef+K]`. **Alignment:** `warm_up_to_edit` teacher-forces `obs[0..ef-1]`, so the
  predict-next rollout's **step-0 is `ef`** (`ROLL[:,0] ↔ clean_obs[ef]`, per the §4 scorecard's
  `gt_traj_obs = clean_obs[ef:]`) — plot `ROLL[:, 0:K]` against `clean_obs[ef:ef+K]`, no slicing, no dropped step.
  The one exception is the **Oracle observation** column, which was fed `obs[ef]` and therefore **leads by one
  frame**; label it as such rather than re-aligning the other columns to it. Canonical implementations:
  `controls/encoder_editing.ipynb` `waterfall_grid` and `scripts/eval_editability_endogenous.py` `waterfall()`.
- **2D observations:** a literal waterfall cannot be drawn when a frame is a 2D raster. Use the sanctioned
  pair `frame_grid` + `frame_trails` (`omniscient_2d/frame_grid.py`, spec in
  `omniscient_2d/WATERFALL_SPEC_2D.md`, approved 2026-08-12 and governed by `../../../CLAUDE.md`). Every
  content rule above still binds — including the shared-`ef`-row ban; only the axes (arms become rows),
  the locators (circles, so `aspect="equal"` is mandatory) and the time subsample change. The two panels
  ship **together**. Do not improvise a per-notebook substitute — that is exactly the drift this registry
  exists to stop.

  > ### ⛔ NEVER paint a shared teacher-forced `ef` row across all columns (corrected 2026-07-30)
  > **This file previously mandated the opposite** — one shared row = `clean_obs[ef]` in *every* column, with each
  > model column's step-0 dropped. That is **wrong and is now banned**, and this entry is where the error leaked
  > back into the `controls/` notebooks after being fixed in `eval_editability_endogenous.py` v2. It makes every
  > column look as though it were teacher-forced on the post-edit frame when only the **Oracle observation**
  > reference actually was, and it **hides the exact frame the §4 scorecard scores** (step 0). It also displayed
  > the *clean* render while the model that legitimately sees that frame is fed the **noisy** `edits.obs[ef]`.
  > Seeing the post-edit frame is a **property of one editor**, never a display convention. Treat any pre-2026-07-30
  > waterfall built to the old rule as misaligned. Governing spec: `../../../CLAUDE.md`.
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
