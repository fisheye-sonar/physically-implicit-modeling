# Direction: Master Editability Notebook — clean visual synthesis (GRU + RSSM)

**Tag:** `[in-frame]` · **Sub-question:** 1+2+3 (synthesis) · **Status:** v1 built; REVISIONS PENDING (2026-07-09) ·
**Complexity:** high (large, presentation-grade notebook; consolidation, not new science)

> ## REVISION PASS v4 — S4 rebuild (Sevan section-by-section review, 2026-07-15)
> *(Absorbs the 2026-07-09 pending items: waterfall correctness, colormap, next-step line plots.)*
> Apply all CLAUDE.md legibility/synthesis standards (incl. the NEW ones: mechanism names, reference
> scales + units, PCA-prefixed estimator names, revert-vs-collapse precision, calibrated claims).
> **Change ONLY §4/§4b (+ definitions table additions). §5 gets ONLY the mechanical renames needed to
> keep it executing** (its redesign is pass v5). S0–S3 untouched.
>
> **A. Correctness bugs (fix FIRST):**
> 1. **Fig 5 "GT" column is NOT ground truth** — it plots the model rollout from the teacher-forced
>    post-edit state (`h_gt`), which is why it shows ghost traces / extra streaks. The GT column must be
>    the **simulation's clean observations** `edits.clean_obs[smp, ef:ef+K]`. Keep the model-rollout-from-
>    true-state as its OWN column, labeled **"True-state swap (model rollout)"** — it is the upper bound
>    any editor could achieve, and its imperfection vs GT is itself informative. Label the unsteered
>    column "Unsteered (model rollout, no edit)".
> 2. **`→target` metric semantics:** currently `dist_to_target(obs,s)` compares the step-s generated obs
>    against the STATIC render of the edit-frame target (`tgt_render_int`) — objects keep moving, so even
>    the true-state swap "drifts away" from it. Replace the per-step metric with **RMSE(generated obs at
>    step s, sim clean obs at frame ef+s)** ("distance to the true post-edit trajectory"). Keep the static
>    target render ONLY for the step-0 direct-edit comparison, labeled as such.
>
> **B. Editor line-up (run on BOTH GRU and RSSM; design tables/figures to absorb future additions):**
> References: **GT (sim clean obs — not a model output)** · **Unsteered (model rollout)** ·
> **True-state swap (model rollout from teacher-forced post-edit h)**.
> Editors: **Readout injection** (linear-probe pseudoinverse `inject_state`, NO manifold projection — it
> exists in the code as `h_pinv` but was never shown as an editor) · **MLP-probe gradient** (NEW here: use
> `pim.editors.gradient_steer` with a frozen **MLP** (pos,vel) probe — train an `MLPExtractor` per model;
> this is the repo's established meaning of MLP steering) · **Global-PCA projection** (POCS
> `manifold_steer`) · **PCA geodesic** (iterative constant-step local-tangent walk, k=64, K=120) ·
> **Decoder gradient** (RENAMED from "MLP-gradient": Adam on `h` minimizing ‖decode(h) − GT obs at edit
> frame‖²; mark **oracle — uses GT observations**).
> The §4 intro table columns = *name | mechanism | needs | oracle?* — **NO "one line" column, no
> results/hypotheses in the table**; results live only in `Current results (updated 2026-07-15)` blocks.
> Naming note: either name ALL methods (the table does this) — keep only a one-line footnote explaining
> that "local-tangent projection" (one-shot) ≠ "PCA geodesic" (iterative) because they were previously
> conflated.
>
> **C. Metric definitions table for §4** (formula, units, ↑/↓, and a REFERENCE value for every magnitude):
> readout RMSE (probe readout of `h_edit` vs target, state-space, at the edit step, before any rollout);
> GT next-step RMSE (generated obs at step 1 vs sim clean obs at ef+1); per-step GT-trajectory RMSE;
> obs-change (step-0 RMS vs unsteered) reported as **% of the true-state-swap obs-change** (the proper
> 100%; drop the pseudoinverse denominator to a footnote); ghost-ray ratio (define "ghost rays": rays
> where the edited object was pre-edit AND must not be post-edit — `pre_render_id==edit_obj &
> tgt_render_id!=edit_obj`; explain the "ghost rays available: N" count); **global-PCA hull residual**
> (raw ‖h−proj‖, real-state reference ≈1.7); **leave-out local-PCA residual** (FRACTION
> ‖q−proj‖/‖q−local mean‖, k=64, neighborhood excludes the query's nearest neighbor to avoid the
> projection tautology — one-line rationale in the table; real-state reference printed beside every use).
> The word **"honest" is banned** — the metric is named by what it is.
>
> **D. Figures:**
> - **Fig 4 — one ROW per world model** (GRU row, RSSM row) × 3 panels: (a) scatter, x = readout RMSE,
>   y = **GT next-step RMSE** (lower = better; place the legend outside the axes so points near the
>   top-left can't collide); (b) per-step GT-trajectory RMSE curves per editor — REMOVE the dotted
>   "MLP-grad reverts" vline/annotation; y-axis label "RMSE(generated obs, true post-edit obs)";
>   (c) leave-out local-PCA residual bars + dashed real-state reference.
> - **Fig 5 → one waterfall figure PER world model** (Fig 5a GRU, Fig 5b RSSM): dark waterfall scheme
>   **matching `notebooks/world_model_eval.ipynb`** (check and copy its exact waterfall styling/cmap —
>   not the current magma-on-dark); columns = GT | Unsteered | True-state swap | Readout injection |
>   MLP-probe gradient | Global-PCA projection | PCA geodesic | Decoder gradient; make the whole figure
>   **WIDER** to fit 8 columns at full size (do not shrink panels); include **~6 pre-edit frames** in
>   every column (sim clean obs as shared context, with a horizontal line marking the edit frame; say so
>   in the caption); keep the green target / red ghost vlines; put the green/red **legend at the top of
>   the figure** (fig.legend), not inside a panel; **single-line title** (drop the "GT = a perfect…"
>   sentence).
> - **Fig 6 (old) + cell `# [21]` DROPPED.** Replace with per-WM **first post-edit observation line
>   plots** (Fig 6a GRU, Fig 6b RSSM): 1D obs trace at rollout step 0 for every editor overlaid, GT
>   target render dashed black, ghost/keep zones shaded — compact and clean, style of the
>   `geodesic_walk_k150` scans. Rewrite cell `# [18]` as a small demarcated per-step table with correct
>   language (see E).
> - **PCA-geodesic budget check (answers "did it just need longer?"):** at k=64 run the constant-step
>   geodesic to **K=600** on ~32 samples with plateau early-stop (stop if readout-RMSE improves <1% over
>   50 iters); plot readout-RMSE vs iteration; report in `Current results`: does it keep descending
>   toward the target readout or asymptote short of it?
>
> **E. Language:** the decoder-gradient rollout does not "revert" — inspect the waterfall: the output
> **collapses off-distribution** (degenerates), which is why its target error climbs back to baseline.
> Use "collapses" and describe what is seen. Remove "THE CENTREPIECE" (section is "§4 — Editing
> head-to-head (the core comparison)"); remove "GRU primary" (§4a GRU / §4b RSSM, equal treatment,
> §4c+ reserved for future models); strip results/jargon from all §4 prose outside `Current results`.
>
> ## REVISION PASS v5 — S5 (run ONLY after v4 lands and is verified)
> - §5 retitled **"Summary — what these experiments say about the learned state"**; Fig 7 retitled
>   "Fig 7 — Summary". Calibrated prose per CLAUDE.md: quantities not verdicts ("~34% of ‖h‖ is not
>   explained by (pos,vel); R²(h)≈0.86 — largely but not fully a function of the physical state" — NOT
>   "non-canonical" as a binary); fix "intrinsic ~5–7 bracketing 8 DOF" → "close to, slightly below, the
>   physical 8 (GRU); the RSSM sits above (~9.6–10)"; revert→collapse; interpretations allowed here ONLY,
>   clearly marked.
> - **Fig 7 redesign:** (a) keep the capability bars (neutral title); (b) REPLACE old 7b and 7c with ONE
>   cross-architecture takeaway panel: the Fig-4a scatter with BOTH world models overlaid (x = readout
>   RMSE, y = GT next-step RMSE, color = editor, marker shape = world model) — this is the genuine
>   synthesis plot (old 7b produced uninterpretable large negative %-gap bars; old 7c duplicated Fig 3a).
> - Consolidated summary `# [24]` → demarcated markdown tables, plain language, updated metric/editor
>   names.
> - Whole-notebook prose sweep for leftover shorthand/jargon ("honest", "MLP-gradient", "reverts",
>   `~=`/`=>`/`!=`) in markdown/prints — without touching S0–S3 computation.

> ## REVISION PASS v2 — INTRO / S0 / S1 (Sevan section-by-section review, 2026-07-09)
> Purpose of this notebook (make it the opening framing): a **single source of truth for how we
> investigate editable causal structure in trained world models, for comparison across architectures,
> and for solidifying the language + metrics we propose** — itself **provisional** (a proposal for what
> may enter `pim`), so don't over-index on details. Apply the new CLAUDE.md **"Synthesis notebooks"** +
> **"Notebook legibility"** standards. **Change ONLY intro/S0/S1 structurally; leave S2–S5 logic intact**
> (re-run to regenerate their outputs). GPU re-run required (RSSM intrinsic-dim compute is new).
>
> **INTRO** — Move transient results (e.g. "~35% non-canonical") OUT of the header. Header keeps only:
> the purpose above + the **source-notebook/checkpoint list** (that's good; keep it current). Any current
> numbers go into per-section `Current results (updated YYYY-MM-DD)` blocks.
>
> **S0** —
> - Add a concise **conceptual note** distinguishing the **generative/physical minimal statistic**
>   (`(pos,vel)` = 8-dim) from the **causal / belief state** the world model actually needs. Because of
>   process noise (pos σ0.04) + obs noise (σ0.2), `(pos,vel)` is *not* always confidently inferable from
>   history, so the predictively-sufficient statistic is a **belief** over `(pos,vel)` (recursive
>   Bayesian / Kalman-filter view): legitimately **≥8-dim** (mean + uncertainty), collapsing toward ~8
>   only at the filter's steady state. Frame in **computational-mechanics** terms (Shalizi/Crutchfield
>   causal states / ε-machine: the causal state = minimal predictively-sufficient partition of pasts;
>   under noise it is a belief, and statistical complexity Cμ exceeds the physical state's entropy).
>   **Implication to state plainly:** storing >8 dims is *expected*, so "non-canonical vs the 8-dim
>   physical state" ≠ "non-canonical vs the causal/belief state." Keep it SHORT and mark it
>   **PROVISIONAL (framing under refinement by Sevan)** — do not overreach.
> - **Redraw Fig 0.** Current diagram is broken: World→GRU→RSSM are colinear so the World→RSSM arrow
>   passes *through* the GRU box. Replace with a clean, architecture-agnostic overview:
>   `World state (pos,vel) → Renderer → 1D Observations → [World Model: GRU | RSSM | …] → hidden state h`,
>   then three labeled probes off `h`: **Recoverability** (h→(pos,vel)), **Editability** (Δh→rollout→obs),
>   **Geometry** (manifold of visited h). Proper non-overlapping boxes/arrows (no colinear pass-through);
>   the world-model box visibly holds a slot for multiple architectures. Simple and legible.
>
> **S1** —
> - Add a `Current results (updated YYYY-MM-DD)` block; move woven-in numbers there.
> - **Fig 1 must hold N world models.** (a) Remove hull counts from the title. (b) Restructure the
>   intrinsic-dim panel: x-axis categories = estimators (**TwoNN, MLE, hull@90%**), **one color-coded
>   bar per world model side-by-side**, shared legend, dashed line at the physical **8 DOF**. **Compute
>   TwoNN and MLE for the RSSM** on its own state bank (cheap model-free estimators) — NO "~same"; label
>   every model.
> - The printed S1 metrics table: add a right-hand **"how computed / meaning"** column (brief) per metric
>   (dims@90/95, intrinsic dim, tangent angle).
> - **Tangent-rotation provenance** (state it in the table/caption): principal angles between **local PCA
>   tangent subspaces** (k=64 nearest neighbors, top-8 components) of pairs of nearby states, over **80
>   anchors × 60 stratified targets** (method from `manifold_geometry_diagnostic`).

> ## REVISION PASS v3 — S2 / S3 (Sevan section-by-section review, 2026-07-15)
> Apply the strengthened CLAUDE.md legibility rules (clearly-demarcated tables; plain language, no
> shorthand; results out of titles into dated `Current results` blocks; define every implementation
> detail). **Change ONLY S2 and S3; leave S0/S1/S4/S5 intact** (re-run to regenerate). Synchronous
> in-turn GPU execution (per WORKER.md) — do NOT background it.
>
> **Whole-notebook (apply to S2 & S3 now):** every figure `suptitle`/panel title must state **what is
> plotted, not the result** ("Recoverability of (pos,vel) from h", not "velocity nonlinear-instantaneous").
> Move the result sentences into a per-section `Current results (updated 2026-07-15)` block.
>
> **S2 (Recoverability):**
> - Convert the loose aligned-`print()` tables (cell `# [7]` velocity-2×2, `# [8]` position) into
>   **clearly-demarcated tables** matching the style of the S1 metrics table `# [5]` (use a `display()`'d
>   pandas DataFrame). Rewrite the jargony headers (`=== VELOCITY 2x2 … => nonlinear-INSTANTANEOUS ===`)
>   into plain sentences.
> - **Use early-t vs late-t (not all-t).** Define both in the definitions table + a note: **early-t =
>   frames t < 15** (belief/filter not yet converged, velocity underdetermined); **late-t = frames t ≥ 15**
>   (converged, velocity well-estimated). Switch the current "all-t" series to "early-t" (cheap recompute).
>   Keep the FULL 2×2 (single vs 2-frame, early & late) in the printed table for reference.
> - **Simplify Fig 2 bars** (they're too crowded):
>   - **2a:** single-frame `h_t` only — categories **{linear, MLP}**, bars = **GRU early-t, GRU late-t,
>     RSSM early-t, RSSM late-t**. Drop the `[h-1,h]` and `dh` groups from the bars (they stay in the table).
>   - **2b:** **single-frame (1f) only** per component (vx0,vy0,vx1,vy1) — GRU vs RSSM; drop the 2f bars.
>   - **2c:** unchanged.
>
> **S3 (Canonicality / fiber):**
> - Add the dated `Current results` block; move the woven result sentences out of the Fig 3 title and the
>   `# [10]` `HEADLINE:` prints. Convert the `# [10]` fiber print into a clearly-demarcated table.
> - **Fig 3 fixes:** (i) plain-language `suptitle` stating what's shown (no `~=`, no baked-in results);
>   (ii) put value labels on **both** the linear and MLP bars (currently only MLP) — or neither; do both;
>   (iii) raise the y-limit / add headroom so the **legend no longer overlaps the leftmost bar**. Otherwise
>   Fig 3 is good.

> Build ONE clean, **presentation-grade, visual-heavy** notebook that is the single readable synthesis
> of the whole editability / canonical-state investigation, **GRU and refined RSSM side by side**.
> Location: `notebooks/experiments/editability/00_master_editability.ipynb` (the `00_` marks it the
> primary/entry notebook of this pillar). Use **NotebookEdit**; do not modify other notebooks.
> Number every cell `# [N]` and every figure `Fig K — …` with lettered sub-panels `(a)/(b)/(c)`
> (CLAUDE.md). Produce BOTH rich plots (Sevan judges from these) AND printed metric tables (self-verify).
> Light academic theme for metrics/analysis (`style_ax`, Okabe-Ito, white bg); **dark theme**
> (`#0a0a14`, `dark=True`) for simulator/observation/waterfall outputs. PNGs → `/tmp/master_editability/`.

## Purpose & stance
This is a **consolidation** notebook — the numbers already exist in the source notebooks (cited below);
recompute the cheap artifacts cleanly (states, probes, edits, waterfalls) and *cite* the expensive ones
(intrinsic dim on the 200k bank). **No `pim` changes** — keep code in the notebook (integration is
Sevan's call, later). Structure it like a paper walkthrough: one idea per section, each with a headline,
a figure, and a table. Every claim citable as "cell [N] / Fig K".

## Bootstrap (cold-start)
3-deep paths (`../../..` repo, `../../../runs`, `../../../datasets`); helpers via `../..`. Load BOTH:
GRU `runs/gru/3_dset3_gru_persistentids_inview_400epochs/best_model.pt` (H=256) and refined RSSM
`runs/rssm/4_dset4_refined_best/best_model.pt` (`model.sample=False`, flat=cat[h_det256,s_stoch64]=320).
Data `datasets/4_fixed_refl_inview`, `n_obj_keep=2`. Teacher-force test → `states_tf` each model.
Velocities from HDF5 `velocities` (`[:, :-1, :2, :]`). Mirror the working notebooks:
`canonical_state_editing.ipynb`, `geodesic_walk_k150.ipynb`, `diagnostic_corrections.ipynb`,
`../rssm_structure/rssm_state_geometry.ipynb`. Model-agnostic via `HiddenStateModel`.

## Sections (mirror the science; GRU + RSSM in each where applicable)

**§0 — Premise.** State the sim is constant-velocity ⇒ minimal sufficient statistic `(pos,vel)`=8-dim
for 2 objects. One clean schematic/text cell. Verify velocity temporal-std ≈ 0.

**§1 — Geometry (sub-Q1).** PCA scree (GRU 38/256 @90%, RSSM 34/320); intrinsic dim vs physical 8 DOF
(TwoNN 5.2 / MLE 6.9 — cite from `manifold_geometry_diagnostic`); curvature (tangent rotates ~56° GRU /
65° RSSM at NN spacing). Fig: scree + a curvature bar. Table: dims @70/90/95, intrinsic dim, tangent angle.

**§2 — Recoverability (sub-Q2).** Position (lin 0.84 / MLP 0.96) and the **corrected velocity 2×2**
(from `diagnostic_corrections`): velocity is **instantaneously readable but only NONLINEARLY**
(single-frame MLP ≈ 2-frame MLP; GRU late-t 0.94), **NOT temporal** — show the 2×2 table for both models
and call out that the old 0.47→0.76 "temporal" reading was a linear-vs-MLP confound (RETIRED). For the
RSSM also show h-only / s-only / full (position lives in det `h`, not stochastic `s`).

**§3 — Canonicality / fiber-collapse (sub-Q2).** `‖h − g(pos,vel)‖/‖h‖` (MLP g): GRU 0.337; RSSM
**det-only 0.368 ≈ GRU** (full-320 0.602 inflated by the stochastic `s` at 0.891 — show the split).
Headline: `h` is non-canonical (~35% not a function of the sufficient statistic) and the RSSM's det core
is **no more canonical** than the GRU — KL structure buys nothing here. Fig: residual bars (linear vs MLP,
GRU/RSSM/det/s).

**§4 — Editing head-to-head (sub-Q3) — THE CENTREPIECE.** Compare editors on the SAME edits
(edit_frame=20, roll out ~15), GRU primary + RSSM echo:
- **GT** (true post-edit rollout) · **Unsteered** · **Manifold-global** (global-PCA projection) ·
  **PCA geodesic** (iterative local-tangent walk, constant-step) · **MLP-gradient** (obs-driven Adam on `h`).
  *(Naming, keep distinct: "local-tangent projection" = one-shot; "PCA geodesic" = the iterative walk.)*
- **Unified waterfall comparison** (the deliverable Sevan wants): the clean **1D-line overlay** style from
  `geodesic_walk_k150.ipynb`, but **bigger and clearer** — one multi-panel figure with a column per editor
  and rows = rollout steps (or an overlaid-lines-per-step panel), **green = target loc, red = ghost/original**.
  Include the true observation waterfalls too (dark theme). **Explicitly include the reversion example**:
  show the MLP-gradient edit reaching the target at step 0 and **reverting by ~step 4** (Sevan asked to see
  this) — a per-step →target and ghost curve, plus the waterfall showing the streak snapping back.
- **Metrics table** per editor: →target render dist, obs-change (% of a full swap — and note the weak
  pseudoinverse denominator caveat), ghost ratio, **persistence** (per-step revert curve), off-manifold
  residual (global + honest leave-out local). This is where "readable ≠ controllable" is shown: probe/
  pseudoinv hits the readout but not the obs; obs-gradient hits the obs but off-manifold and reverts.

**§5 — Synthesis.** One text cell + one summary figure/table: predictively-sufficient but non-canonical;
curved `(pos,vel)→h`; velocity nonlinear-instantaneous; readable≠controllable; **architecture-independent**
(RSSM replicates, KL delivers no canonicity/controllability gain, world-state in the det core). Tie to the
organizing hypothesis (editability ⟺ canonical, factored, predictively-sufficient state) — as *hypothesis*.

## Deliverables
- Executed notebook (GPU, numbered cells/figures), plots + tables per section, PNGs to
  `/tmp/master_editability/`. Keep outputs lean enough to stay under the `Read` cap where feasible.
- Dated note `research/scratch/2026-07-08-master-editability.md` (`→ FLAG FOR PROMOTION` only as a pointer:
  "this is the consolidated visual synthesis; promotion decisions live in the `candidate-*.md` files").
  Do NOT edit `findings/` or `RESEARCH.md`, do NOT mark directions done.
- Structured report: section-by-section, which figures landed, any number that disagreed with the source
  notebooks (flag loudly), PNG paths.

## Context / caveats to preserve honestly
- The small-k geodesic obs-change "% of swap" uses a weak pseudoinverse denominator (`diagnostic_corrections`
  §3) — caption it as "moves a lot," not a clean 0–100%. If cheap, build a teacher-forced true-post-edit
  reference for a proper 100% baseline; otherwise state the limitation.
- All probes are in-sample fit — comparisons are load-bearing, absolute R² optimistic. Say so once.
- Sources of truth: `canonical_state_editing`, `geodesic_walk_k150`, `manifold_geometry_diagnostic`,
  `diagnostic_corrections`, `../rssm_structure/rssm_state_geometry`; corrected numbers in
  `research/scratch/2026-07-08-diagnostic-corrections.md` and the `candidate-*.md` files.
