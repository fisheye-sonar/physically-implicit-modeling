# Direction: Master Editability Notebook — clean visual synthesis (GRU + RSSM)

**Tag:** `[in-frame]` · **Sub-question:** 1+2+3 (synthesis) · **Status:** v1 built; REVISIONS PENDING (2026-07-09) ·
**Complexity:** high (large, presentation-grade notebook; consolidation, not new science)

> ## REVISIONS PENDING (Sevan, 2026-07-09) — address on the next master-notebook pass
> 1. **§4 waterfalls (Fig 5/6) are disliked AND possibly WRONG.** The **"Unsteered" panel looks like a
>    model's output, not an unsteered rollout** — **investigate as a potential bug** (is it plotting the
>    model's *generated* obs instead of the true unsteered/GT rollout? check the data feeding each column)
>    before/while restyling. Correctness first.
> 2. **Drop the purplish colormap → classic academic style** (the light / Okabe-Ito theme Sevan prefers),
>    even for the waterfalls. The old classic look was preferred.
> 3. **Add the next-step 1D line plots** (the style from `geodesic_walk_k150.ipynb` that Sevan liked —
>    make them clearer than that version).
> 4. Fuller notebook feedback from Sevan is still to come — do not do a full rebuild until it lands; this
>    list is the confirmed subset.

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
