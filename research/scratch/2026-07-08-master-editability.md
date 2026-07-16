# Master Editability notebook — consolidated visual synthesis (GRU + RSSM) (2026-07-08)

→ FLAG FOR PROMOTION *(pointer only)*: this is the **consolidated visual synthesis** of the
editability / canonical-state pillar. It introduces **no new science**; the promotion decisions
live in the `candidate-*.md` files (`candidate-editability.md`, `candidate-rssm-replication.md`,
`candidate-state-geometry.md`, `candidate-predictive-quality.md`). Do not promote from this note.

**Notebook:** `notebooks/experiments/editability/00_master_editability.ipynb` (executed on GPU,
numbered cells `# [N]`, numbered figures `Fig K`). **PNGs:** `/tmp/master_editability/`.

**Models/data:** GRU `runs/gru/3_dset3_gru_persistentids_inview_400epochs` (H=256, val_loss 0.0236),
refined RSSM `runs/rssm/4_dset4_refined_best` (det256+stoch64=320, `sample=False`),
`datasets/4_fixed_refl_inview`, n_obj_keep=2, teacher-forced test set. Velocities from HDF5
`velocities` (temporal std ≈1.3e-8 → constant-velocity sim confirmed).

## What this notebook is
ONE presentation-grade walkthrough structured like a paper: §0 premise → §1 geometry → §2
recoverability → §3 canonicality/fiber → §4 editing head-to-head (centrepiece) → §5 synthesis.
GRU + refined RSSM side by side in each section where applicable. Cheap artifacts (states, probes,
edits, waterfalls) recomputed cleanly; expensive ones (intrinsic dim on the 200k bank, RSSM
curvature) **cited** from the source notebooks. All probes are in-sample fit — comparisons are
load-bearing, absolute R² optimistic (stated once in the notebook).

## Corrected numbers used (2026-07-08 pass) — the reason this consolidation exists
- **Velocity is nonlinear-INSTANTANEOUS, not temporal.** Single-frame MLP ≈ 2-frame MLP (Δ ≤ 0.015
  all-t, ≤ 0.007 late-t on both models); `dh` differencing is *worse*. The old "temporal 0.47→0.76"
  reading was a linear-vs-MLP confound → **RETIRED**. (§2)
- **RSSM det-only fiber residual ≈ GRU.** det-only ≈0.368 ≈ GRU ≈0.337; the full-320 ≈0.602 was
  inflated by the stochastic `s` (≈0.891). Do NOT call the RSSM "less canonical." (§3)
- **Position lives in the deterministic `h`,** not the stochastic `s`. (§2/§3)

## Sections (headline per section)
- **§0 Premise:** constant-velocity sim ⇒ minimal sufficient statistic `(pos,vel)` = 8-dim.
- **§1 Geometry:** intrinsic dim ≈5–7 (TwoNN 5.2 / MLE 6.9, cited) brackets 8 DOF; fat linear hull
  (recomputed scree GRU ~38/256, RSSM ~35/320 @90%); strongly curved (tangent ~56°/65° at NN
  spacing, cited). Fig 1.
- **§2 Recoverability:** position ~linear (lin ~0.84 / MLP ~0.96); velocity 2×2 (nonlinear-
  instantaneous); RSSM position split det≈full≫s. Fig 2.
- **§3 Canonicality:** GRU h ~34% non-canonical (fiber resid ≈0.337); RSSM det ≈0.368 ≈ GRU; s holds
  none (≈0.891). KL buys no canonicity. Fig 3.
- **§4 Editing head-to-head (CENTREPIECE):** five editors — GT / Unsteered / Manifold-global /
  PCA-geodesic (iterative local-tangent walk) / MLP-gradient (obs-driven Adam). Unified dark-theme
  waterfall overlay (green=target, red=ghost) + metrics table (→target, obs-change vs GT-swap %,
  ghost, per-step persistence, honest leave-out off-manifold residual). **Reversion example:** the
  MLP-gradient edit reaches target at step 0 and reverts by ~step 4 (Fig 6). RSSM echo confirms
  readable≠controllable is architecture-independent (§4b). Figs 4–6.
- **§5 Synthesis:** predictively-sufficient but non-canonical; curved `(pos,vel)→h`; velocity
  nonlinear-instantaneous; readable≠controllable; architecture-independent (RSSM replicates, KL
  delivers no canonicity/controllability). Frames — as *hypothesis* — editability ⟺ canonical,
  factored, predictively-sufficient state. Fig 7.

## Caveats preserved (honest)
- "obs-change % of a full swap" small-k geodesic uses a weak pseudoinverse denominator; the notebook
  additionally prints a **teacher-forced GT true-post-edit** 100% reference for a proper baseline.
- Naming kept distinct: "local-tangent projection" (one-shot) ≠ "PCA geodesic" (iterative walk).
- Recomputed RSSM scree @90% here ≈35 vs cited 34 (within rounding of a different bank subsample) —
  not a disagreement, just sampling. Any numbers that differed from source notebooks are flagged in
  the notebook's §5 auto-summary; the load-bearing corrected values (velocity 2×2, det-only fiber)
  reproduced the `diagnostic_corrections` results.

## Open
None new — consolidation only. Feeds the pending promotion of `findings/editability.md`,
`findings/state-geometry.md`, and a new architecture-independence finding (see `candidate-*.md`).

## v2 revision — intro / S0 / S1 (2026-07-09, Sevan section-by-section review)
*(Addendum written by the orchestrator — the worker executed the notebook but stopped before writing
this / verifying; orchestrator finished it.)* Re-ran clean (0 errors, 8 figs). Changes, intro/S0/S1 only
(S2–S5 left intact, just regenerated):
- **Intro/tier:** transient results moved OUT of the header into per-section `Current results (updated
  YYYY-MM-DD)` blocks; header now states purpose (single source of truth for editable causal structure
  across architectures; provisional proposal for `pim`) + source-notebook list. (New CLAUDE.md
  "Synthesis notebooks" tier standard.)
- **§0:** added a PROVISIONAL conceptual note distinguishing the **physical** minimal statistic (8-dim
  `(pos,vel)`) from the **causal/belief state** (computational mechanics / ε-machine; under noise the
  predictively-sufficient statistic is a *belief*, legitimately ≥8-dim; storing >8 dims is expected).
  Fig 0 **redrawn** (was broken: colinear World→GRU→RSSM) → clean architecture-agnostic pipeline with a
  multi-model World-Model box + three probes off `h`.
- **§1:** Fig 1 rebuilt to hold **N world models** (estimator categories × color-coded per-model bars,
  8-DOF line, no results-in-title). **RSSM intrinsic dim now COMPUTED** (was "~same"):
  **NEW RESULT — RSSM TwoNN 9.6 / MLE 10.0, HIGHER than GRU (5.2 / 6.9) and ABOVE the physical 8**
  (GRU brackets 8; RSSM sits above). Hull@90% GRU 38 / RSSM 35; tangent 56°/65° (cited). This **updates**
  the earlier "geometry replicates / intrinsic dim ~same" hand-wave — the RSSM's state is *higher*
  intrinsic-dim, consistent with the belief/stochastic-latent view. **NOT yet promoted** —
  `findings/architecture-independence.md` should gain this once Sevan judges it (he's mid-review).
- Orchestrator fixed unfilled `@TOKEN@` placeholders the worker left in the §0/§1 `Current results`
  blocks (surgical JSON edit; figures preserved).
- **Pending:** §2–§5 review (Sevan resumes tomorrow); the §4 waterfall/colormap/line-plot items are
  still queued for when his pass reaches §4.

## v3 revision — §2 / §3 (2026-07-15, Sevan section-by-section review)
Re-ran clean **synchronously in-turn** (0 error cells, 8 figs regenerated: fig0–fig7). **Changed ONLY
S2/S3 + the definitions table**; S0/S1/S4/S5 source is byte-identical to the pre-edit backup (verified by
diff — the only changed cells are `defn_table, 10, 11, 12, 13, 14, 15, 16`). No new science — this is a
legibility/presentation pass under the strengthened CLAUDE.md standard (clearly-demarcated tables, plain
language, results out of titles). Changes:
- **Results out of figure titles → dated `Current results (updated 2026-07-15)` blocks** for §2 and §3.
  §2/§3 section headers now pose the *question* ("can `(pos,vel)` be read out of a single `h`?", "is the
  hidden state a *function* of `(pos,vel)`?") with a plain "What this section measures" paragraph; Fig 2
  suptitle → "Recoverability of (pos, vel) from a single hidden state h"; Fig 3 suptitle → "is each
  hidden-state block a function of the 8-dim (pos, vel)?".
- **Loose aligned-`print()` tables → clearly-demarcated rendered tables** (velocity 2×2 `# [7]`, position
  `# [8]`, fiber `# [10]`). Implemented as `display(Markdown(...))` tables **matching the §1 `# [5]`
  style** — NOT pandas: pandas is **not installed** in the `.pim` venv and the existing `# [5]` table
  deliberately avoids it ("rendered markdown; no pandas dependency"); a first attempt with a pandas
  DataFrame `ModuleNotFoundError`'d, so I matched the repo's own convention. Each table keeps a plain-text
  mirror for agent self-verification; jargony headers (`=== VELOCITY 2x2 … => nonlinear-INSTANTANEOUS ===`)
  rewritten into plain sentences.
- **early-t vs late-t (replacing all-t).** Defined both in the definitions table + a §2 note: **early-t =
  frames t < 15** (recursive filter/belief not yet converged ⇒ velocity underdetermined), **late-t =
  t ≥ 15** (converged). The full single-vs-two-frame 2×2 for **both** regimes stays in the `# [7]` table;
  Fig 2 bars simplified: **2a** = single-frame only, categories {linear, MLP}, bars = GRU early / GRU late /
  RSSM early / RSSM late (dropped the `[h-1,h]` and `dh` groups from the bars — kept in the table); **2b** =
  single-frame per component only (dropped 2f bars); **2c** unchanged.
- **Fig 3** — plain-language suptitle, value labels on **both** linear and MLP bars (was MLP only), y-limit
  raised to 1.15 so the legend (`loc="upper left"`) clears the bars.

**Verified numbers this run (unchanged story, just recomputed):**
- Velocity (late-t, GRU): single-frame linear **0.588** → single-frame MLP **0.944**; two-frame MLP **0.951**
  (Δ vs single-frame **+0.007**); dh-MLP **0.723** (strictly worse). RSSM late-t: sf-lin 0.573 → sf-MLP 0.931,
  2f-MLP 0.931 (Δ −0.000). early-t is lower on both (GRU sf-MLP 0.867, RSSM sf-MLP 0.749). ⇒ velocity is
  **nonlinear-instantaneous, NOT temporal**; the old 0.47→0.76 "temporal" reading (a linear-vs-MLP confound)
  stays retired.
- Position: GRU linear **0.837** / MLP **0.974**; RSSM linear **0.851** / MLP **0.955**. RSSM position split
  (linear): full **0.851** ≈ det **0.835** ≫ s **0.583** ⇒ lives in the deterministic core.
- Fiber (MLP `g`): GRU h **0.337** ≈ RSSM det-core **0.368**; RSSM full **0.602** inflated by stochastic s
  (**0.891**). ⇒ `h` ~non-canonical vs the 8-dim physical statistic; KL structure buys no extra canonicity.
  (All reproduce the `diagnostic_corrections` corrected values.)

**Workflow note (mechanics only):** the notebook is ~1 MB with embedded figures and its cleared-output
source (~38k tokens) still exceeds the 25k Read cap, so NotebookEdit could not be unlocked directly. I
temporarily stashed the non-target cells' source to a scratch JSON (size-mechanics only, the fallback the
brief authorized), did all substantive S2/S3 edits via **NotebookEdit**, restored the stashed cells
verbatim, then executed. Backup + stashes live in the session scratchpad. PNGs in
`/tmp/master_editability/` (fig0–fig7).

---

## v4 addendum (§4 rebuild, 2026-07-15) — correctness fixes + full editor line-up on BOTH models

**Scope executed:** REVISION PASS v4 only (S4 rebuilt in place; §5 got only the mechanical renames needed
to keep executing — its redesign is v5). S0–S3 code cells byte-identical to the pre-edit backup; one
sentence in the intro outline was updated because it named §4 "THE CENTREPIECE" and "MLP-gradient
reversion" (both retired by this pass). Notebook re-executed end-to-end on GPU, 26/26 code cells, 0 errors.

### The two correctness bugs (both confirmed, both fixed)
1. **The old Fig-5 "GT" column was not ground truth.** It plotted `roll_obs["GT"]` — the model's own
   autoregressive rollout from the teacher-forced post-edit state `h_gt` — which is why it showed ghost
   traces/extra streaks. GT is now the **simulation's clean observations** `edits.clean_obs[smp, ef:ef+15]`
   (never a model output). The model rollout from the true state is kept as its own column, renamed
   **"True-state swap (model rollout)"** — and its imperfection is itself a finding (below).
2. **The old per-step →target metric was wrong.** `dist_to_target(obs, s)` compared step-s generated obs
   against the STATIC edit-frame render for every s, so even a perfect edit "drifted away" as objects moved.
   Replaced by **per-step GT-trajectory RMSE** = RMSE(gen obs @ step s, sim clean obs @ frame ef+s). The
   static render survives only as the labeled step-0 direct-edit check. Consequence: the old "MLP-gradient
   reverts by ~step 4" reading was partly an artifact of the static target; see the collapse verdict below.

### Editor line-up (5 editors × 2 models; references = Unsteered, True-state swap; GT = sim)
Headline numbers (64 edits, edit_frame 20; full tables in notebook cells [17]–[18]):

| GRU | readout RMSE | GT next-step RMSE | % of swap (obs-chg) | ghost-ray ratio | leave-out local-PCA resid (real 0.58) |
|---|---|---|---|---|---|
| Unsteered (ref) | 1.839 | 0.279 | 0 | 1.000 | 0.62 |
| True-state swap (ref) | 1.607 | 0.203 | 100 | 0.665 | 0.77 |
| Readout injection | 0.000 | 0.276 | 15.7 | 0.985 | 0.68 |
| MLP-probe gradient | 1.347 | 0.276 | 43.5 | 0.996 | 0.80 |
| Global-PCA projection | 0.026 | 0.267 | 74.5 | 0.928 | 0.88 |
| PCA geodesic | 1.237 | 0.277 | 55.6 | 0.914 | 0.12* |
| Decoder gradient (oracle) | 6.992 | 0.237 | 214.3 | 0.087 | 0.99 |

| RSSM | readout RMSE | GT next-step RMSE | % of swap (obs-chg) | ghost-ray ratio | leave-out local-PCA resid (real 0.63) |
|---|---|---|---|---|---|
| Unsteered (ref) | 1.839 | 0.279 | 0 | 1.000 | 0.67 |
| True-state swap (ref) | 1.818 | 0.271 | 100 | 0.884 | 0.72 |
| Readout injection | 0.000 | 0.279 | 0.1 | 1.000 | 0.67 |
| MLP-probe gradient | 15.971 | 0.273 | 61.5 | 0.960 | 0.80 |
| Global-PCA projection | 1.846 | 0.283 | 61.4 | 0.984 | 0.76 |
| PCA geodesic | 1.814 | 0.280 | 39.0 | 0.965 | 0.08* |
| Decoder gradient (oracle) | 209.824 | 0.131 | 466.9 | 0.088 | 0.99 |

*The geodesic's tiny leave-out local-PCA residual is partly by construction (its last operation IS a
local-PCA projection onto bank neighbours) — footnoted in the notebook; don't read it as "most on-manifold."

**New findings this pass:**
- **The true-state swap itself is sluggish** — the model's belief barely updates from ONE teleport frame:
  swap obs-change 0.129 (GRU) / 0.059 (RSSM), ghost-ray ratio 0.665 / 0.884, and its readout RMSE stays
  1.61 / 1.82 (≈ un-edited 1.84). This is the honest 100% that all %-of-swap numbers now use — the old
  pseudoinverse denominator gave e.g. "1362%" (GRU) and is demoted to a footnote.
- **Readout injection is decoder-inert**: readout 0.000 on both, obs-change 15.7% (GRU) / **0.1%** (RSSM)
  of the swap; on the RSSM its rollout is pixel-identical to unsteered.
- **MLP-probe gradient (new editor, `gradient_steer` on a frozen MLP (pos,vel) probe, in-sample R² 0.94 /
  0.91)** also fails to transfer: GT next-step ≈ unsteered on both models; on the RSSM its optimum drives
  the LINEAR readout to 16 — the MLP-probe optimum lies off-distribution.
- **No non-oracle editor beats the true-state swap on GT next-step RMSE on either model.** Best non-oracle:
  global-PCA projection GRU 0.267 vs unsteered 0.279 (swap 0.203) — moves a lot (74.5%) but scrambled.
- **Decoder gradient (renamed from "MLP-gradient", marked oracle) COLLAPSES, it does not revert** — verified
  two ways: (i) its distance to the unsteered rollout stays flat at ≈0.31 (GRU) for all 15 steps — it never
  dissolves back into the no-edit trajectory — while its GT-trajectory RMSE climbs from 0.011 to above the
  unsteered level by ~step 4; (ii) the Fig 5a waterfall shows the rollout fragmenting into speckled,
  non-scene-like output. On the RSSM the failure is milder/slower: step-0 0.039, next-step 0.131 (the only
  state better than the swap's 0.271 — oracle only), then the target streak smears and the ghost re-emerges
  by ~step 12–14, again without re-matching the unsteered rollout (distance ≈0.26–0.31).

### PCA-geodesic K=600 budget extension (32 samples, k=64, plateau early-stop <1%/50 iters) — Fig 6c
**Verdict: the readout gap ASYMPTOTES — the geodesic did not just need longer.**
- GRU: 1.75 → 1.08 @ iter 120 → **1.03 at plateau**; 32/32 samples early-stopped (median stop iter 135);
  mean curve flat from ~iter 200 to 600. Note the plateau is *below* the true-state swap's readout (1.61)
  while the geodesic's observations stay ≈unsteered — readout and observation accuracy are largely decoupled.
- RSSM: **no descent at all** — 1.80 → 1.75, median stop iter 51 (earliest possible). Cause: the injection
  distance is tiny on the RSSM, so the matched constant step is 0.011 (vs 0.20 GRU); but even the GRU's
  healthy descent asymptotes far above 0.

### Also in this pass
- §4 intro table is now *name | mechanism | needs | oracle?* (no results in the table); all results live in
  dated `Current results (updated 2026-07-15)` blocks (§4, §4a, §4b, budget-extension).
- §4 metric definitions table: every metric with formula, units, ↑/↓, and a concrete reference value;
  ghost rays defined (911 available across the 64 samples); "honest leave-out local residual" renamed
  **leave-out local-PCA residual** everywhere ("honest" removed).
- Figures rebuilt: **Fig 4** = one row per model × (readout-vs-next-step scatter, per-step GT-trajectory
  RMSE curves, leave-out local-PCA resid bars + real-state dashline), figure-level legend, no "reverts"
  annotation. **Fig 5a/5b** = per-model 8-column full-size dark waterfalls in the `world_model_eval` style
  (gray cmap, #0a0a14, orange dashed edit-frame line), 6 sim clean-obs context rows in every column,
  green/red target/ghost vlines, one figure-top legend, single-line titles. Old Fig 6 + cell `# [21]`
  dropped; **Fig 6a/6b** = per-model step-0 1D scans (geodesic_walk_k150 style, ghost/target zones shaded,
  static target render dashed black); **Fig 6c** = geodesic budget extension.
- §5 / `# [24]`: mechanical renames only (metrics4 keys, editor-name strings, per-model summary loop);
  redesign deferred to v5 as instructed.

**Open questions / caveats (say-it-plain):**
- Decode conventions differ by one frame across architectures (GRU decoder = next-frame prediction, RSSM
  decoder = current-frame reconstruction), so reference rollouts can carry a ±1-frame offset against the
  sim trajectory (footnoted in §4). It is small relative to teleports (~6–7 world units) but nonzero; a
  follow-up could re-align per architecture.
- The RSSM decoder-gradient next-step 0.131 is intriguing: the decoder's preferred directions clearly CAN
  express the edit and its imagination step propagates it briefly — but the state is wildly off-manifold
  (linear readout 210) and degrades. Worth a targeted look in the learn-to-edit direction.
- The RSSM geodesic's non-descent may be step-size-limited (const step matched to the tiny injection
  distance); a step-size sweep on the RSSM was NOT run here.
- metrics change note: §4 numbers are NOT comparable to the pre-v4 notebook (`->target` was redefined
  against the moving GT trajectory; % of swap switched to the true-state-swap denominator).

**PNGs:** /tmp/master_editability/{fig4_editor_metrics, fig5a_waterfalls_gru, fig5b_waterfalls_rssm,
fig6a_scans_gru, fig6b_scans_rssm, fig6c_geodesic_budget}.png (fig0–fig3, fig7 regenerated unchanged in
story). Old fig5_unified_waterfalls.png / fig6_reversion.png deleted.

---

## v5 addendum (2026-07-15) — §5 Summary rebuild + prose sweep

Revision pass v5 (per `research/directions/master-editability-notebook.md`) applied to
`notebooks/experiments/editability/00_master_editability.ipynb`; full synchronous re-execution, 0 error
cells, all 11 figures regenerated. **S0–S4 computation byte-identical** (verified by source diff against
the pre-v5 notebook; only prose strings in two prints changed — see sweep below).

- **§5 retitled** "Summary — what these experiments say about the learned state" (no results in the
  title). Body split into a dated quantities recap and a clearly-marked **"Our reading (interpretation)"**
  block — the only editorializing in the notebook. Calibrated language throughout: "≈34% of ‖h‖ is not
  explained by any g(pos,vel) we fit (R² of g on h 0.867) — largely but not fully a function of the
  physical state" (no "non-canonical" verdict); intrinsic dim corrected to "close to, slightly below, the
  physical 8 for the GRU (TwoNN 5.2 / MLE 6.9); the RSSM sits above (9.6 / 10.0)" (the old "~5–7 brackets
  8 DOF" claim was GRU-only and stale); decoder-gradient failure consistently described as **collapse
  off-distribution** (GT-trajectory RMSE back above unsteered by ~step 4 while distance to the unsteered
  rollout stays ≈0.31 — degeneration, never a revert). v4 nuances carried in: true-state-swap sluggishness
  (step-0 obs-change 0.129 GRU / 0.059 RSSM, ghost 0.665 / 0.884) and the K=600 geodesic asymptote
  (GRU plateau 1.03, all 32 samples early-stop; RSSM 1.80→1.75, no descent — more budget does not close
  the readout gap).
- **Fig 7 redesigned** as "Fig 7 — Summary": (a) capability bars per world model (position R² MLP,
  velocity R² single-frame MLP late-t, 1 − fiber residual; RSSM canonicality uses the det core; RSSM
  position bar switched from det-core-linear to full-state MLP so both models show the same metric);
  (b) ONE cross-architecture scatter — x = readout RMSE (symlog, linear ≤ 2), y = GT next-step RMSE,
  colour = editor (§4 colours), circle = GRU / square = RSSM, references included, legend outside the
  axes. Old 7b (%-gap-closed bars with large negative values) and 7c (duplicate of Fig 3a) removed.
  PNG renamed fig7_synthesis.png → **fig7_summary.png**.
- **Cell `# [26]` rebuilt** as demarcated `display(Markdown)` tables (no pandas): §1 geometry, §2
  recoverability, §3 fiber residuals (+ R² on block), §4 per-model editor tables with per-model reference
  scales, §4 rollout-behaviour/budget table (decoder-gradient step-0/step-4 vs unsteered + distance-to-
  unsteered; geodesic iter-120 vs K=600 plateau), PNG manifest. All framed "Current results (updated
  2026-07-15)".
- **Prose sweep (prints/markdown only, S0–S4 logic untouched):** cell [4] `=>`→`→` twice ("PHYSICAL"
  lower-cased); cell [5] "Read:" line — stale "~5-7 brackets the 8 DOF" corrected to the per-model
  statement + `=>`→`→`. No other `~=`/`=>`/`!=`-as-prose, wrong-referent "reverts", or stale editor names
  found in S0–S4 (remaining `!=` hits are code/formulas; "reverts" occurrences are the legitimate
  revert-vs-collapse definitions).
- Numbers spot-checked against this run's outputs: all §5 quantities match ([25]/[26] mirrors); run-level
  jitter confined to the third decimal (e.g. real-state leave-out local-PCA residual RSSM 0.625 this run
  vs 0.63 in the v4-dated §4 text).
- Residual nit (not fixed, would touch S1 computation): the unused citation dict key
  `honest_local_resid_real` in cell [5] still carries the banned word in its *identifier* (never printed).

→ FLAG FOR PROMOTION — pointer only: this notebook is the consolidated visual synthesis; promotion
decisions live in the `candidate-*.md` files.
