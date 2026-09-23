# PROGRESS.md — Session Handoff

> Agent-owned, rewritten freely each session. Answers **"where is the work right
> now?"** — *not* "what's true" (that's `findings/`). Git history is the backstop.

_Last updated: 2026-09-23 09:55 PT — three appendix catch-ups running on the 4090 (nn_r2, grid categorical IM, token rescore with the mean-frame guard) before final_tables; last training job on the lab ~16:00_

## 2026-09-21 (evening) — PAPER TEXT session (appendix tables; no other edits)

Sevan asked for the appendix TABLE STRUCTURE (values to be refreshed later): eight tables added to `paper/paper_draft.tex`
under their existing `\subsection`s, in Tables 1/2's style (heat column types P / E / F, `G` gaps, `\multirow` + `\shortstack`),
every value read through the canonical readers (`pim.figures.tables.collect` under `select="index"` and `"fidelity"`, cartesian
basis; `tables.prediction_rows`; `runs/_baselines/*/{baselines,bayes_floor}.json`; `paper/figs/editability_trends/by_point_values.md`)
with `% source:` comments above each table. Blanks left where no number exists: Rayworld Bayes floors (queue `appendix_prediction`),
nearest-neighbour latent R² (`nn_r2` not folded into scores.json), categorical IM on smooth / obs5 / standard / blink (no arm),
the token model's mean-frame IM. Two tables are PROVISIONAL and marked `\red{}`: the token model's mean-frame editability
(`experiments/dw_tokens/bridge`, 2026-09-06, frustum, pre-guard) and legal-vs-illegal Othello targets
(`experiments/adjacent_flip_ablation/scores/ceiling_symdiff_*.json`, 2026-09-15; canonical re-run = `scripts/index_ceiling.py
--reachability --editors`). The grid-layout table was restyled (ND dropped as in the main tables, IM column added, guarded arms).
Main text untouched (verified byte-identical outside `\appendix`). Draft NOT committed (Sevan's working tree).
Noted at session start: 10 peer sessions on this repo, one busy; `qualitative_main/` round-4 scripts edited 19:55 with the
composites not yet regenerated — left alone.

## 2026-09-21 (day) — PAPER FIGURES session (Sevan: finish all figures today; this session = paper help only, not the queue)

**Plan agreed with Sevan 10:30 PT.** Figure 1 = (a) mini teaser + (b) probe-derived editing vs inverse map, hand-drawn by
Sevan (`paper/figs/mini_teaser.pdf`; the wrapfigure goes); the network diagram (`paper/figs/teaser.pdf`) moves to
§Experimental Setup with its symbols aligned to the text (h / z / s; it shows only the edited reference for editability).
Five WORKERS launched ~10:50 PT, one folder each under `paper/figs/`, briefs in each folder's `BRIEF.md`, common rules in
`paper/figs/FIGURE_BRIEF_COMMON.md`, shared look in `paper/figs/paper_style.py` (Times New Roman, TrueType-embedded vector
PDFs, Okabe-Ito, one colour per editor):
`environments_overview/othello/` (rule-set boards, legal moves, flips; options O1–O3 + composite),
`environments_overview/rayworld/` (frustum + rays + discs + velocity arrows + faded trajectory, linked waterfall, blink /
N-ray / appearance-cell panels + composite), `qualitative_main/` (editorial main-text figure from the cached guarded
writes: Rayworld options R1–R3 with the 128-ray model as the hero, Othello options T1–T3 incl. a zoom on the
symmetric-difference squares), `editability_trends/` (Edit Index by residual point; Edit Index by ray count with seed SD
via `tables.collect`), `history_rewrite/` (compute split from drawing; 3-edit × 4-column paper waterfall). Pieces AND
composites; Sevan assembles. Workers write only in their folders + `.scratch/`; the GPU is shared with the queue's
training job (one model at a time). Results land in each folder's README; this entry is updated when they report.
**Landed and verified on disk (11:05–11:30 PT), four of five:** `history_rewrite/` (`history_rewrite_top3` recommended: 3 largest
teleports × GT | Unedited | Single-point edit | History rewrite, K = 15; variants hist-alone / K = 10 / random-3; `pieces/`;
stretch `prediction_quality` = GT vs free-run, no edit; arms from the saved cards: IM +0.61 / 0.36, hist+IM +0.63 / 0.36,
hist +0.65 / 0.48, n = 32 one seed; the PDF now embeds all 128 rays — matplotlib's default ppi had resampled to 125).
`environments_overview/othello/` (O1 recommended: four boards after move 14 with legal-move dots, chosen move, flipped discs
ringed; O2 before/after pairs; O3 three positions; 52 vector pieces; stretch `predictive_composite` = legal set vs model
distribution per variant, visually identical at the canonical tint). `editability_trends/` (A2: Edit Index by residual
point for oth-standard | dw-noiseless with hollow = outside guard + an R² / MLP-skill row — IM on Othello lands only at
points 4–5, on Rayworld climbs to +0.59 by point 6 with g R² ≈ 0.3; B2 / B1_half: Edit Index by ray count, n = 3 pooled
means, SD bars smaller than the markers; every value matches the ledger; categorical IM = the new categorical map).
`qualitative_main/` (recommended composite `composite_R3_T3_typical_2row`: (a) Standard model × 3 scenarios × PI / GS / IM
— PI / GS smear, IM renders the teleported disc; (b) 5×5 zoom on the edited tile + changed squares, Standard vs
Adjacent NoFlip × Unedited / GT / PI / IM; `typical` = cases whose per-case index is nearest the population mean; R4 adds
a 5-ray foil column; T1 / T2 alternatives; 401 pieces). Still running: `environments_overview/rayworld/`.
**Fifth landed 11:40 PT — `environments_overview/rayworld/`** (composite recommended: (a) frustum with observer, 128 rays — hits
dark and stopping at the disc, misses light — discs filled with their reflectivity grey, velocity arrows, seven fading ghosts,
the frame-t* strip laid along the far plane, beside the 40 × 128 waterfall with a t axis and a t* pointer; (b) blink waterfall
with the hidden span bracketed and the 0.5 edge markers; (c) 8-ray frustum + the SAME world through 16 / 8 / 5 rays via
`render_scene` under each sibling's config; (d) the 30-cell appearance partition on dw-8ray with the discs' cells filled;
21 vector pieces, light and dark variants; sequences by a stated `default_rng(0)` rule in `selection.json`).
**ALL FIVE DONE in ~50 min wall clock; GPU freed (only the queue's training job remains).** Committed with the scripts, READMEs,
sidecars, top-level PDFs / PNGs and piece PDFs; piece PNG previews are gitignored (`paper/figs/.gitignore`, regenerable).
**ROUND 2 (Sevan's review, 11:50 PT):** ALL figure text → **Arial** (was Times New Roman; `paper_style.py` switched, maths in
Arial via the custom mathtext set; the two appendix qualitative scripts get the same) and `ps.save` crops with ZERO outer
padding (spacing is set in LaTeX). Asked: Othello O2 layout with ONE board reachable under standard / adjacent-flip /
adjacent-noflip for (a)–(c) (reachability search over standard bench prefixes with the vendor board class; (d) stays);
Rayworld pieces: fewer drawn rays in the big frustum (64 / 43 / 32 variants, strips stay 128), wider N-ray waterfalls,
"(signaled)" under "hidden", (c) without the small frustum, (d) bigger with translucent discs, a flatter composite with
letters set apart and less whitespace; main qualitative figure `composite_final` (Rayworld 2 × continuous + 2 × categorical
columns — A1 dw-noiseless with BLANK categorical IM cells (no arm by Sevan's own spec) vs A2 the 128-ray ray-family model
where the categorical map exists; Othello + GS, new `typical` rank-2 cases, CYAN on the flipped tile + legality-switching
squares in Unedited and PINK on the same squares elsewhere, boards +10 %, larger GT→PI gap, bigger panel letters) and
`composite_final_sidebyside` (variants as columns incl. Adjacent Flip, no GS, thinner rims); the appendix qualitative figures
regenerated in Arial with the CURRENT arms (categorical IM = the categorical map on 5/8/16/128-ray, blank on Standard /
Blink — the cached old-map frames matched no scored arm); history figure: original observed history above the line in EVERY
column, "origin / destination position". Trends re-rendered (done, 12:00). History round 2 DONE 12:10 (context rows asserted
identical across columns; `history_frames_*` = original | rewritten | counterfactual render as a separate appendix piece).
Othello / Rayworld / qualitative round-2 workers running (brief `qualitative_main/BRIEF_ROUND2.md`).
⚠ 14:10 PT all three round-2 workers were KILLED by the account's API session limit (reset 14:30); resumed 14:35 with their
transcripts intact (partial script edits were on disk). **Rayworld round 2 DONE 14:45:** `composite_v2_split` recommended
(5.46 × 2.96 in; every-3rd ray drawn = 43 of 128, strips and waterfall at full 128; N-ray strips 1.25 in wide; "(signaled)";
(c) strips only; (d) `categorical_frustum_8ray_crop` with discs at alpha 0.65); round-1 composites kept as `composite_v1*`. **Othello round 2 DONE 14:50:** a board reachable under
all three rule sets EXISTS only early-game — standard bench case 182 after move 6 (10 discs; adjacency placement order
`f3 d3 f2 d2 e3 c3`, flip-free, asserted equal to the standard board; nothing after moves 7–14 survives in 1000 games, exact
DFS for adjacent-noflip, flip-free sufficient condition for adjacent-flip); `composite_O2` = shared board (a)–(c), one move
c4 legal under all three (flips d4 in a, b; nothing in c), (d) own checkerboard game; `composite_O2_alt` = per-rule moves
(d1 flipping three in a row in a). Search ~45 lines, orders in `boards.json`. **Qualitative round 2 DONE 15:25 — ROUND 2 COMPLETE.**
`composite_final_A2` recommended (5.5 × 4.78 in): (a) the 128-ray ray-family model (`ray_ablation/L-dw-128ray-20m`) × 2 scenarios
× continuous + categorical columns — every one of the 12 edit cells a scored arm (cartesian PI −0.02 / GS −0.10 / IM +0.57;
appearance-fac PI −0.31 / GS +0.31 / categorical IM +0.64); `_A1` = dw-noiseless with the two categorical IM cells blank (no arm).
(b) 5×5 zoom, Standard (case 342) / Adjacent NoFlip (case 39), `typical` rank 2, Unedited / GT / PI / GS / IM, CYAN outlines
(flipped tile + legality-switching squares) in Unedited, PINK on the same squares elsewhere, key "pre-edit / post-edit";
boards +6 % (a five-column row at 5.5 in cannot give +10 %). `composite_final_sidebyside_A1/_A2` (5.5 × 3.26 in; Othello variants
as columns incl. Adjacent Flip case 261, no GS, rims 1 pt; the Rayworld strips there are 0.44 in wide — readable, not pretty).
APPENDIX figures regenerated in Arial with the CURRENT arms: `qualitative_edits/` categorical IM through the canonical
categorical map (cache hits only; 16 / 8 / 5-ray at pt 6 / 6 / 5), BLANK cells on Standard and Blink (no arm since 2026-09-20),
new caches `.scratch/qualitative_edits_catim_*`; `qualitative_edits_othello/` with the cyan / pink marking (shared helper).
All five folders committed; GPU freed.
**ROUND 3 (Sevan's second review, 15:40 → 17:10 PT) COMPLETE, all committed.** Othello: per-rule moves, (c) plays the would-flip move c4
that no longer flips (alt f5), bold letters. Rayworld: bigger frustum, t* row outlined, "hidden" centred, centre dots in (d), tighter
(composite 5.46 × 3.26 in). Qualitative: "Unedited Pred", "Example 1/2", dot key top-right above the colour bar, 128-ray model
throughout (Sevan: "effectively the standard model", named in the caption), A1 dropped; appendix Rayworld no-arm categorical IM
spots left fully empty (no frame). Trends: legend 0.18 in under the x labels; renamed by_point / by_rays / by_rays_half. History:
random-3 among the 15 of 32 cases with displacement ≥ 20 rays (seed 0 → cases 14 / 18 / 25), hist+IM default, histonly alt,
history_frames appendix piece. NEW `paper/figs/predictive_quality/` = rayworld (transposed, GT / Prediction × 3 examples) +
othello (moved from environments_overview/othello). **PRUNED per Sevan** ("drop the million extra versions"): every folder's top level
= the chosen figure(s) + at most one alternative + script(s) + README + sidecars; per-element PDFs under pieces/ (PNG previews
gitignored); dropped variants regenerable via script flags (git history keeps them). Worker briefs moved to `paper/figs/briefs/`.
**Open for Sevan:** categorical IM for dw-noiseless / dw-blink would need two ~40-min map fits (a scoring job, not started).
**ROUND 4 (17:15 → 19:40 PT) COMPLETE, committed.** Main qualitative figure `composite_final`: columns Standard (continuous) Examples
1–2 (dw-noiseless), 128-ray (categorical) Example 3 (dw-128ray, categorical IM), 5-ray (categorical) the SAME Example 3 (dw-5ray);
every drawn scenario passes the 5-ray visibility filter (the scorer's differing-ray zone under the dw-5ray renderer non-empty;
27 of seeds 0–39 pass; Examples = seeds 0 / 1 / 2). Appendix Rayworld figure re-seeded to the first six passing seeds
(0, 1, 2, 5, 7, 8 → seed0 + more_seeds/1,2,5,7,8; seeds 3 / 4 dropped); both READMEs carry the matching explanation (one world,
rendered under each instance's own radius / ray count). Trends: `by_rays` = one full-width overlaid panel, no ND. Predictive
quality: rayworld = (a) Standard (b) Blink (c) 5-ray × 2 sequences × GT / Prediction / Difference (canonical signed-error map;
Blink and 5-ray free-runs via the canonical bench calls, cached `.scratch/predictive_quality_<inst>.npz`, `compute_rayworld.py`)
— ⚠ the random 5-ray pair (cases 20, 26) shows large free-run misses (the disc never moves to its new ray); flagged to Sevan,
seed is the knob; othello = three examples per variant stacked (full page).
**ROUND 5 (20:00 PT):** the 5-ray visibility filter was too weak (it passed a ONE-ray change; seed 8 looked unchanged to Sevan).
New rule in `qualitative_edits/make_figure.py::passes`: at least 2 changed rays of the clean 5-ray frame AND at least 2 with
|Delta| >= 0.2, both off the scorer's own `build_edit_zones`. Seeds 0-59: 29 pass, 31 fail (15 that passed the old rule now fail).
⚠ The intensity clause never binds on discworld — fixed reflectivities make every changed ray differ by exactly 0.4 or 0.8, so the
ray-count clause decides every case; it is a guard for a future fainter instance, not a live discriminator. Main figure re-assigned
(Sevan's swap): Example 1 = seed 0, Example 2 = seed 2 (both Standard continuous), Example 3 = seed 1 (128-ray + 5-ray categorical).
Appendix more_seeds made disjoint from the main figure's seeds: 1, 2, 8 dropped, now 5, 7, 9, 10, 12; primary stays seed 0 so the
tex path resolves (its Standard column still shares seed 0 with the main figure's Example 1 - Sevan's call, left as is).
**Figure work is DONE pending Sevan's two hand-drawn figures. NEXT: the paper text** — Table 2 + §Results to the guarded arms and the
categorical map (+0.637 / +0.823 / +0.866 / +0.913), seed ± from the families, "one seed / 780k steps" wording, captions from the
folder READMEs (selection rules, arms, guard, which model each panel uses), the empty appendix sections the figures now fill.
**Next for the paper:** Sevan assembles Fig 1 (mini teaser + probe-vs-inverse panel) and the network diagram; then the text:
Table 2 + §Results to the GUARDED numbers, the categorical IM column to the categorical map, the ± from the seed families,
captions from each folder's README (selection rules, arms, guard). Figure text is minimal by design — every fact goes in the caption.
**Paper facts noted for the text (not yet applied):** Table 2 / §Results quote the pre-guard numbers; the categorical
IM column is now the categorical inverse map (5/8/16/128-ray parents +0.913 / +0.866 / +0.823 / +0.637) so "IM is unchanged
by the target" is false; five Rayworld families at n = 3 and oth-standard at n = 2 for the ± values.

## 2026-09-19 (night) — follow-up to the scoring audit (Sevan): ONE change made, the rest deliberately left

**Changed:** `master_eval.ipynb` SETTINGS `dw_bases` default `"frustum"` → `"frustum,cartesian"` (Sevan: both by default,
cartesian is the de facto basis). Frustum stays FIRST on purpose — categorical probes are keyed under `bases[0]`; putting
cartesian first would orphan every cached categorical probe and skip those blocks on newly scored runs. Proven inert: a
dry run of `score_all` + `score_all_baselines` gives the IDENTICAL to-do list under both defaults (every run and floor
file already has both blocks), every queue job that runs the scorer sets `PIM_DW_BASES` itself, and with that variable
set SETTINGS is identical before / after. Edited while the lab was mid-training (128ray_s1, ~6 h from its scoring stage);
the remote's copy is unchanged until its next pull (harmless, same reason). **Sevan: NO other fix from the audit now** —
none is worth any risk to the queue; revisit after it drains (list in the entry below).
**⚠ For Sevan's decision (paper wording, not code):** in the CATEGORICAL rows IM does NOT invert from the categorical grid.
`dwa.inverse_arms` feeds g the CONTINUOUS 8-d post-edit state (`full_state_pair`, exact target position + velocity) in
every block; a categorical block contributes only its bench (cell-changing cases) and its basis label (frustum). So in
those rows PI / GS are told the target CELL and IM the exact POSITION. The draft already says "IM … writes the full state
either way"; Sevan expected a categorical inverse. Options: keep + say so in the caption (no compute), or add a
categorical inverse map g_cat(labels → residual) as a new arm (an instrument change; ~IM's cost per run).
**Variance on the guard:** the tables' ± on every fidelity cell = SD over training seeds (`pool_replicates` pools
`<ed> fid`; e.g. 8-ray cartesian PI 0.921 ± 0.041, GS 0.886 ± 0.017, IM 0.265 ± 0.008, n = 3). There is NO case-level SD
of the guard anywhere (only a bootstrap CI, and only on IM / IM-NN + Othello arms scored since this evening); the Edit
Index has both (seed SD in the tables, `edit_index_case_sd` / `_se` on every arm scored since 2026-09-18).
**Seen in the executed notebook:** every pass retries the replicates' inherited-but-never-fitted extra targets (8-ray
replicates inherit 10 grid targets; 23 runs + 2 floor files are perpetually "adding … SKIPPED") — seconds each, harmless,
noisy; fix later by letting a replicate inherit only the targets its probes exist for.

## 2026-09-19 (evening) — AUDIT of `pim/scoring` (Sevan: is it pure plumbing? any bugs?) — read in full, nothing changed

**No metric is defined there** (no numeric reduction in the package; ruff F clean; every recorded `best` = the true argmax
on all 157 blocks of 51 runs, no NaN index anywhere; arm counts match SETTINGS exactly: 293 regression / 282 categorical).
**It does make choices:** `best` = UNGUARDED argmax (Othello: on the UNION index, and the bare `edit_index` of an Othello arm IS
the union) — not what the tables report since today (`pim.metrics.selection`); `scripts/index_ceiling.py --editors` and
`paper/figs/history_rewrite` still read it. Categorical blocks live in `dw_bases[0]`, so Table 2's categorical IM is the
FRUSTUM-basis inverse map beside cartesian continuous rows (8-ray +0.703 vs +0.711; ~0.02). The scored bases come from the
env var `PIM_DW_BASES` (default frustum only — CLAUDE.md's plain nbconvert command does not produce the cartesian blocks
the paper reads). Floors' large-corpus recipe (250k / 50 epochs), seed, which floor point is reported live in `baselines.py`.
**Bugs / hazards (none changes a reported number):** (1) `scan_runs` excludes `_`-TOPICS only → quarantined
`ray_ablation/_R-dw-8ray-20m` is still scored (cartesian + IM were added to it) while its SETTINGS entry (`R-dw-8ray-20m`)
no longer matches; (2) `except RuntimeError` around `fit_probes` also swallows CUDA OOM on a REGRESSION block → run
written without its canonical block, rc 0 (self-heals next pass); (3) discworld PI / ND / GS arms do NOT carry
`fidelity_ci95_*` (only IM does — `score(…)` is called without the unsteered card; REGISTRY / 37edca0 say otherwise);
(4) `score_all_baselines`: an arch on file with no scanned run (`recurrent_l` on dw-noiseless, dw-pn04) → KeyError the day
a basis is added; baselines.json written non-atomically; (5) a metrics.jsonl line caught mid-write aborts `scan_runs`;
(6) `inverse_discworld`'s print dereferences `best["IM"]` (None if no IM arm); (7) scores.json is read-modify-write with
an hour-long window and a full rescore drops foreign blocks (`prediction`): the remote's pull of the shared seed-0 member
`L-dw-8ray-20m__seed0_s512000` overwrote the lab's copy (prediction block lost; `score_prediction.py` re-adds it).
**Found on the way (upstream, `pim.probes`):** the LINEAR probe at residual point 0 is numerically broken on every
regression block (held-out R² −0.1 … −2.2; SAME checkpoint: lab −1.12 vs remote +0.10) — rank-deficient embedding point;
invisible in the tables (max over points; no reported PI arm at pt 0). Same-checkpoint lab-vs-remote scoring otherwise:
PI identical, IM ±0.011, GS up to 0.06 per arm (best GS −0.074 vs −0.032) = measurement noise of refitting the MLP probes.

## 🔄 Paper seed-replicate queue (`experiments/paper_ci/`) — LAUNCHED 2026-09-18 20:48 PT, ETA ≈ 2026-09-23 03:00 PT

**2026-09-19 12:15 check:** healthy. Done: `rep_dw-5ray_s1` (lab), `rep_dw-16ray_s1` (remote), both corpus
pushes. Running: `rep_dw-8ray_s1` (lab, master_eval since 11:59, ~14:00), `rep_dw-8ray_s2` (remote, 300k/512k,
~16:30). **⚠ Gap: both dw-16ray members (`__seed1`, `__seed0_s512000`) have NO `appearance-fac` block** — their
fac probes were fitted (stage C, cached) but `master_eval`'s `dw_extra_targets` on this branch has no
`ray_ablation/L-dw-16ray-20m` entry (it lives only on the unmerged `dw_16ray` branch, where the parent was
scored), so the replicates inherit nothing and the ledger shows the 16-ray categorical row at n = 1 — a paper
Table 2 row. **FIXED by Sevan 2026-09-19:** entry added (762faae); `push_inputs` keeps the trailing slash on
directory pushes (it was nesting `scripts/scripts/` on the remote) and a held job adds the 16-ray members' fac
block (507d623).

**2026-09-19 (Sevan) — ⚠ THE TABLES' ARM-SELECTION RULE CHANGED: best Edit Index INSIDE the fidelity guard** (≤ 1.0; the
unguarded best only where an editor has no guarded arm) — `pim/metrics/selection.py`, read from the arms already in
scores.json, no re-scoring. **16 of the paper's 48 cells move** (list in the REGISTRY row): every discworld PI number on the
128-ray-class instances drops to ≤ 0 (noiseless +0.20 → −0.10, 128ray +0.24 → −0.02, blink +0.17 → +0.02), oth-adjacent-flip
PI +0.40 / 2.09 → +0.35 / 0.83 (now INSIDE the guard), GS there → −0.06. **The draft's Table 2 and §Results text quote the
OLD numbers — to be redone from the notebook** (and the seed ± re-pool under the new rule automatically; the CI ledger /
dashboard already show it). The qualitative figure scripts now draw the SAME guarded arms (re-rendered on the lab GPU beside the
queue's training job, + `more_seeds/seed1…5/` for each; write caches renamed `_guarded`). Appendix notebook gained Table A2 / A2b =
the paper's editability table with every editor at its LOWEST FIDELITY RATIO (`collect(select="fidelity")`,
`selection.best_arm_by_fidelity`): under it only IM (and standard Othello) keeps a positive index — PI / GS fall to −0.15 … −0.60
on Rayworld.
Same session: `pool_replicates` / `ci95_halfwidth` / `t975` → `pim/metrics/replicates.py` and the best-point / gap rules →
`pim.metrics` with EVERY table value bit-identical under the old rule (58 rows × 2 bases + 10 replicate sets); Tables 1d/1e
basis bug fixed (`_rand_perdim` matched frustum caches for cartesian rows); `table_alignment`, `table_bayes`, `fig_capacity`
REMOVED (they read `experiments/`) with their long-list cells (Table 4 there is now `table_prediction`). Notebooks NOT
re-executed (final_tables will; or run them by hand).

**2026-09-19 (Sevan) — THE PAPER NO LONGER DEPENDS ON `experiments/`** (audit: 5 of 37 experiment folders fed a paper
claim). Moved: the Edit-Index CEILING → `scripts/index_ceiling.py` (+ `pim/environments/othello/counterfactual.py`: the
counterfactual-game search; writes `runs/<run>/index_ceiling.json`; `L-oth-20m` re-run on CPU reproduces +0.909, 13 cases,
new CI 0.854–0.956; discworld ceilings on the 1000-case bench STILL TO RUN — in the proposed queue job; 24-case CPU smoke on
8-ray +0.87); flips per move → `scripts/othello_corpus_stats.py` (run: 2.245 / 0.269 / 0 / 0 → `runs/_baselines/<inst>/
corpus_stats.json`); the seed-0 member layout → `scripts/layout_checkpoint_replicate.py` with a FORWARDER at the old path
(replicate.sh untouched while the queue runs — switch its stage B after the drain, see `experiments/bayes_floor/
QUEUE_HANDOFF.md`); `build_paper_tables_and_figs` cell [5] (old Table 4 from `experiments/bayes_floor`) DELETED;
`experiments/history_rewrite` → `paper/figs/history_rewrite/` (script moved + paths fixed, compiled, NOT re-run — GPU).

**2026-09-19 (Sevan) — PREDICTIVE LOSS vs BAYES FLOOR made canonical; NOTHING RUN YET (queue owns the GPUs).**
`pim/metrics/prediction.py`, `pim/environments/prediction.py` (a run's `prediction` block),
`pim/environments/discworld/bayes.py` (floor by posterior sampling over the initial state; blink's marker process
analytic), `pim/environments/othello/bayes.py` (exact), `scripts/bayes_floor.py` →
`runs/_baselines/<inst>/bayes_floor.json`, `scripts/score_prediction.py` (fold-in), `tables.table_prediction`,
`notebooks/build_appendix_tables_and_figs.ipynb` (Table A1; renders `—` until the files exist), 7 tests, REGISTRY
rows. Scope (Sevan): 4 Othello + dw-noiseless / blink / 16 / 8 / 5-ray, main-table runs + the token run read two
ways (+ dw-128ray, added the same day); no smooth / pn04 / training curves / per-position curves; columns loss · floor ·
excess, drawn as a FIGURE in the master tables' style (heat panels, grouped rows). A sampled floor is stored as a
bracket lo … hi and SHOWN as midpoint ± (half-width + one SE); Othello's exact floor has no ± (Sevan). A TRIVIAL-predictor
column (Sevan): the best history-blind constant from the probe split, stored in each floor file (`trivial`); measured on
CPU: oth-uniform 4.094 (= log 60), dw-8ray 0.1007 / token CE 5.18, dw-noiseless 0.0726. ⚠ Repeat-the-current-frame is a far
stronger naive reference on discworld (dw-8ray 0.0078 vs floor ≈ 0.0056, model 0.00577) — recorded as `persistence_mse`,
printed not tabulated; put to Sevan. CPU pilot, dw-8ray: floor 0.0055–0.0057 vs model 0.00577 (`scratch/2026-09-19-bayes-floor-pilot.md`);
the old Table 4 "floor = 0" was a state oracle, not a Bayes floor. Verified on CPU: renderer parity 100% on all five
instances, blink marker model calibrated within 1.4% on 10k sequences. **FILLED ON CPU 2026-09-19 15:10 (lab in a training phase, no scorer running):** the 4 exact Othello floors + every
instance's trivial predictor (`bayes_floor.py`; discworld files are `+trivial-only` until the sampler runs), the
`prediction` block in 22 runs (paper runs + their scored replicates; CPU, 26 s a run), Table A1 rendered — Othello rows
complete (excess +0.0179 / +0.0052 / +0.0023 / +0.0015 nats), discworld rows show trivial + loss, floor pending the GPU.
Token run: CE 0.4638 nats/frame; mean-frame MSE 0.00573 vs the frame model's 0.00574 on the same sequences.
**To run:** one queue job after
`final_tables` — `experiments/bayes_floor/QUEUE_HANDOFF.md` (for the session operating the queue; job file in
`queue_proposal/`, NOT installed). `master_eval` untouched. UNCOMMITTED. When it lands: check the brackets, write
`findings/predictive-quality.md`, fill the paper's red prediction-quality placeholders, retire Table 4.

**2026-09-19 (Sevan) — `tokens/` trimmed to the minimum:** the discworld tokenizer now writes ONLY `train.i16`,
`vocab.npz`, `meta.json`; every small split is encoded from its h5 at read time (`tokens.encode_h5`, new).
`experiments/dw_tokens/scripts/{ce_by_position,ngram_floor}.py` and `tests/test_token_bench.py` /
`test_discworld_tokens.py` switched off `tokens/test.npy` / `edits.npy` (re-encoding verified byte-identical;
21 token tests pass on CPU under an audit hook that fails any open of the two files). REGISTRY, layout spec,
`layout.py`, `make_discworld_tokens.py` updated. UNCOMMITTED. The two `.npy` files (lab + remote) are Sevan's to
delete; the existing `tokens/meta.json` still lists the retired files under `"files"` (nothing reads it).

**Live:** 28 jobs, all at 512k; first launches `rep_dw-5ray_s1` (lab), `rep_dw-16ray_s1` (remote), the
dw-8ray corpus push (lab cpu lane). The dispatcher timer (`pimci-dispatch`, 2 min) and both watchdogs
are the memory — nothing to re-arm after a session ends or a reboot. **How to check:** the dashboard
(`https://sevan-ubuntu-lab.tail9a3a96.ts.net/ci/`), `logs/paper_ci/dispatch.log`, `systemctl --user
list-units 'pimci-*'` on either host, per-job `logs/paper_ci/<id>/`, `experiments/paper_ci/README.md`
(pause / hold / reset / add). Alerts on the usual ntfy topic; 2-h digest on `pim-ci-digest-ai691k`.
**When it drains (`final_tables` done):** read `dashboard/ledger.md`, check Table 5 (SD + CI panels) and the
± cells in the paper notebook, update `findings/seed-variance.md` (one entry per family: n, budget, SD,
guard verdicts k/n), the REGISTRY run rows, the brief → `done`; move the smoke artefacts
(`experiments/paper_ci/state/_smoke/`, `logs/paper_ci/smoke_*`) to an archive folder.
**Status 2026-09-19 13:35 PT — 4 of 31 jobs done, all verified on disk.** `rep_dw-5ray_s1` (lab, 8.05 h),
`rep_dw-16ray_s1` (4090, 10.75 h), `rep_dw-8ray_s1` (lab, 8.44 h), `score_16ray_fac` (9 min). Pooled n = 2 so
far (member + seed 1): decodability SD 0.000, IM ±0.002–0.007, GS ±0.003–0.02, PI ±0.02–0.03. _(Corrected 19:20: the
"PI guard 1.056 on the 512k members" remark made here at 13:35 was under the OLD unguarded arm selection; the tables' rule
changed that afternoon to the best arm INSIDE the guard (`pim.metrics.selection`), under which all three 8-ray members
have a passing PI arm, guard 0.89 / 0.90 / 0.97 at index +0.18 / +0.23 / +0.21.)_ Measured rates: lab 5090 under its 450 W cap 23.4 steps/s on 8/5-ray (−15%), the 4090 20 steps/s on
discworld (= the uncapped 5090) but 53 min per factorised fit and 45 min per `master_eval` run (lab 31 / 29);
`plan.py` estimates recalibrated, ETA ≈ 2026-09-23 12:00. **Fixes made in flight:** the watchdog reads a tick
heartbeat file (a quiet tick logged nothing → false "not ticking"); progress = the newest write among a job's
outputs, stall threshold 45 min (the scoring stages write outside the training metrics → false stall 03:22);
directory pushes keep their trailing slash (scripts/ was nesting on the remote). **The 16-ray gap:**
`master_eval`'s `dw_extra_targets` had no entry for `ray_ablation/L-dw-16ray-20m` (it lived on the unmerged
`dw_16ray` branch), so the 16-ray members were scored without `appearance-fac`; Sevan added the line 12:20
(commit 762faae), an in-flight execution's in-place write dropped it, it was restored from git 13:19 and
`score_16ray_fac` added the block to both members (values on the parent's). ⚠ A notebook edited while
`master_eval` is executing is OVERWRITTEN when that execution ends — commit the edit and restore after.
**Added 2026-09-19 (Sevan):** the probe-corpus-size control (`experiments/probe_corpus_size/`, README) —
queue jobs `ctrl_corpus_dw` (lab, ~3.5 h, after the ray family) and `ctrl_corpus_oth` (either host, ~1.5 h):
probes / inverse map refitted on up to 200k sequences / 100k games, editors re-swept, canonical numbers
untouched. Write its result into `findings/probe-capacity.md` when it lands. Owed still: the Othello PI/ND/GS
guard CI line in `master_eval` cell [4]; the Othello ceiling bootstrap (deprioritised by Sevan).
**Added 2026-09-19 15:00 (Sevan, via another session's hand-off `experiments/bayes_floor/QUEUE_HANDOFF.md`):** job
`appendix_prediction` — Bayes floors + trivial predictors per instance (`scripts/bayes_floor.py` →
`runs/_baselines/<inst>/bayes_floor.json`), a `prediction` block folded into every scored run
(`scripts/score_prediction.py`), appendix Table A1 re-rendered. Lab only, LAST (deps `final_tables`, which now waits
for EVERY other job so nothing runs on either host while scores.json files are rewritten). CPU smoke passed (parity
1.0 on all six discworld instances, four exact Othello floors). When it finishes: check each `bayes_floor.json`
against the hand-off's accept/reject list, record in `findings/predictive-quality.md`, tell Sevan Table A1 is ready.
**2026-09-19 18:45 — `master_eval` refactor CUT OVER on the lab (merge 108ed90, branch `master_eval_module`, another
session's work; `experiments/master_eval_refactor/` README + CUTOVER.md).** The scorer moved out of the notebook into
`pim/scoring/` as a PURE MOVE (notebook 1,251 → 174 code lines; SETTINGS cell hash-identical incl. the 16-ray entry;
no driver, `paper_ci`, trainer, metric, probe or editor file touched; `EVAL_VERSION`s and the scores.json schema
unchanged; `best_arm` unchanged — the one-rule consolidation waits for the queue to drain). Gate (theirs): Othello
fixture 14,610 leaves and `L-dw-5ray-20m__seed1` 17,103 leaves bit-identical, decision equivalence over 32 runs, real
nbconvert entry point. Audit (mine, independent): 37 old-notebook functions → 20 byte-identical, 17 differ only by the
threaded `s` / `runs` argument (21 lines), the two driver loops differ only by that threading and inert `dry_run`
branches, module constants identical; suite 301 pass on the merged tree; dry listing under the queue's environment =
0 full rescores + the 7 known skip-when-uncached add-block items. **Still to do:** `git pull` on the 4090 during a
training stage (before its next scoring pass); verify the first post-cut-over scoring pass on EACH host (CUTOVER.md's
three checks: one `=== scoring` per new run, the same seven `nothing added` lines, blocks + sibling-consistent numbers).
Rollback: `git revert 108ed90`. Now possible (NOT done, Sevan's call): `nn_r2` in `pim/scoring/blocks.py::attach_inverse`
(+ the Othello inverse_map dicts), the guard CI in `pim/scoring/othello.py::othello_arms`.
**19:10 — dw-8ray is the first complete family (n = 3, all members with frustum / cartesian / pos@appearance / appearance-fac):**
cartesian IM +0.720 ± 0.006 (guard 0.265 ± 0.008), PI +0.207 ± 0.022 (0.921 ± 0.041), GS −0.074 ± 0.030; appearance-fac IM +0.718 ±
0.013, GS +0.445 ± 0.014, PI +0.392 ± 0.006; MLP skill SD 0.000. The 4090 was pulled to 84c2e78 at 19:09 during
`rep_dw-16ray_s2`'s training stage (refactor + guard CI; no execution running, no executing script changed). ⚠ The ledger
reads through `pim.figures.tables`, whose arm selection became GUARDED on 2026-09-19 (another session, `pim/metrics/selection.py`):
ledger / ping / dashboard numbers for PI in particular differ from anything quoted before that change.
**20:53 — dw-5ray complete (n = 3) and the refactored scorer VERIFIED on the lab.** Its first pass (`rep_dw-5ray_s2`, 20:22–20:52) met CUTOVER.md's checks: one `=== scoring` (seed 2), 38 skips, no `stale`, no output errors; eight `nothing added` lines = the note's seven + the 8-ray seed-2 member synced back after the note was written; seed 2's eval version / blocks unchanged, MLP skill identical to its siblings to three decimals, scoring 29.3 min (seed 1: 28.8 under the old scorer); notebook still 6 cells / 174 lines with the 16-ray entry after the in-place write. dw-5ray (guarded selection): cartesian IM +0.808 ± 0.009 (guard +0.230 ± 0.007), PI +0.132 ± 0.036 (+0.898 ± 0.051), GS -0.093 ± 0.015; appearance-fac IM +0.849 ± 0.005, GS +0.559 ± 0.012, PI +0.538 ± 0.017; MLP skill SD 0.000 / 0.000. Lab now on `rep_dw-128ray_s1`. STILL OWED: the same three checks on the 4090's first pass (`rep_dw-16ray_s2`, early 2026-09-20), which must also produce `appearance-fac` without help.
**2026-09-20 04:26 — dw-16ray complete (n = 3) and the refactored scorer VERIFIED on the 4090** (`rep_dw-16ray_s2`, 9.26 h; the cut-over's last owed check). Its first pass there: one `=== scoring` (seed 2), 38 skips, no `stale`, no output errors, 5 `nothing added`, notebook still 6 cells with the 16-ray entry, 50.0 min; stage C fitted `appearance-fac` for both members UNAIDED (the 2026-09-19 `dw_extra_targets` fix holds) and the block is scored. In family: cartesian IM +0.663 ± 0.007 (members +0.661 +0.656 +0.670), MLP skill +0.964 ± 0.000, IM guard 0.288 ± 0.004. 4090 now on `ctrl_corpus_oth`; lab in `rep_dw-128ray_s1`'s scoring stages (training ended 04:04).
**2026-09-20 05:00 — `ctrl_corpus_oth` killed by its 40 GB memory cap (4090; cgroup OOM inside the DENSE Othello inverse-map fit at 60k games; host fine).** Written before the kill: 20k → 60k games is flat (LIN 0.9747 → 0.9752, PI +0.819 → +0.819, ND +0.748 → +0.746; IM +0.810 at 20k). The requeued retry is redefined to those sizes (`--sizes 20000 60000 --im-max-n 20000`, cache hits) so it cannot fail twice; `final_tables` no longer depends on the two controls. NOT run: 100k games, IM above 20k (`experiments/probe_corpus_size/README.md`). 4090 went straight on to `rep_oth-standard_s1`.
**2026-09-20 06:12 — `rep_dw-128ray_s1` done (lab, 9.30 h): dw-128ray at n = 2.** Two members scored in one pass (the new `__seed0_s512000` and `__seed1`, 29.7 min each, three blocks each incl. `appearance-fac`, no errors, no `stale`, notebook 6 cells). cartesian IM +0.573 ± 0.008 (+0.568 +0.579), guard 0.315; MLP skill +0.981 / +0.981; fac (old continuous arm) IM +0.577 / +0.585. 10 of 32 jobs done. Lab → `rep_dw-128ray_s2` (training until ~13:20, then scoring until ~14:45 — the window in which a categorical-IM side unit on the lab cannot meet a scoring stage); 4090 on `rep_oth-standard_s1` at 8.2 steps/s (on plan, ~17.3 h of training).
**2026-09-20 13:25 — dw-128ray complete (n = 3): all four ray families are done** (`rep_dw-128ray_s2`, lab, 7.19 h; one `=== scoring`, 29.1 min, three blocks, no errors / `stale`, notebook 6 cells). cartesian IM +0.574 ± 0.006 (guard 0.317 ± 0.008), GS −0.097 ± 0.027, PI −0.003 ± 0.009, MLP skill +0.981 ± 0.000; appearance-fac ND +0.528 ± 0.016, GS +0.286 ± 0.014, PI −0.331 ± 0.024, IM (OLD continuous arm, to be deleted on deployment) +0.580 ± 0.005. 11 of 33 done. Lab → `rep_dw-noiseless_s1` (extension to 512k); 4090 still on `rep_oth-standard_s1`. Sevan checked in 11:03 — no decision on the categorical-IM deployment yet.
**2026-09-20 17:01 — `rep_dw-noiseless_s1` done (lab, 3.60 h): the replicate-EXTENSION path's first production run, verified.** Seed 1 resumed 390k → 512k (1.49 h), its stale scores parked as `scores.s390000.json` with a `runs/MOVES.md` line; a new `__seed0_s512000` member laid out (the 421,875-step one kept, outside ±10% so not pooled); both scored in one pass (two `=== scoring`, no errors / `stale`, 6 cells). Ledger pools n = 2 at 512k (seed 2, still at 390k, is correctly excluded until its own extension, now running): cartesian IM +0.594 ± 0.003, MLP skill +0.971. 12 of 33 done. Categorical-IM deployment: still no decision from Sevan.
**2026-09-19 23:55 — CATEGORICAL INVERSE MAP: branch built, NOT deployed (Sevan decides in the morning).** Sevan found that every
categorical discworld block's "IM" was its basis's CONTINUOUS full-state map scored on the categorical bench (identical `g_r2` vector
to the frustum block; 52 blocks in 22 runs). His spec: on a categorical block g inverts the block's own one-hot labels + the discs'
Cartesian velocity (deliberately asymmetric to the forward probes), recipe = the forward probe's (200k sequences, 50 epochs, streamed),
IM only, scope dw-128ray / 16ray / 8ray / 5ray (+ 8ray-tok) × `appearance-fac`; old arms DELETED from every scores.json; he will
likely move one categorical-IM run per ray family to the TOP of the queue (it shapes the writing). **Where:** staging clone
`../pim-master-eval-refactor`, branch `categorical_inverse` (commit 70685a7, pushed, unmerged); `experiments/categorical_inverse/README.md`
there holds the design, the evidence and the 5-step deployment. **Running from the clone, writing ONLY inside it:** unit
`catinv_preview` (the four parents at the production recipe → `scores/preview_<run>_appearance-fac.json`, ~45 min each sharing the lab
GPU, 5-ray first; point 0 of 5-ray: held-out R² +0.463 = in-sample, no overfit gap) then unit `catinv_gate` (parity gate on
`L-dw-8ray-20m__seed1`: only the factorised block's old IM arms may differ). Logs `../pim-master-eval-refactor/logs/categorical_inverse/`.
The live tree, the queue and every scores.json are UNTOUCHED. Morning: report the four numbers + the gate, get Sevan's go, then deploy
per the README (merge in a gap → `clear_continuous_im.py --apply` → catch-up jobs at the top of the queue with `PIM_ADD_CAT_IM=1`).
**2026-09-20 03:00 — the staging evidence is in (branch `categorical_inverse` @ 0f076a3+, pushed, STILL unmerged; live tree / queue / scores untouched).**
Previews on the four parents, production recipe, full bench, best arm inside the guard — Edit Index / fidelity, NEW categorical map vs the
OLD continuous map on the same bench: **5-ray +0.913 / 0.25 vs +0.836 · 8-ray +0.866 / 0.27 vs +0.703 · 16-ray +0.823 / 0.29 vs +0.684 ·
128-ray +0.637 / 0.29 vs +0.574** (PI +0.513 / +0.380 / −0.061 / −0.308; GS +0.598 / +0.458 / +0.280 / +0.308). g's held-out R² = in-sample
to three digits everywhere (0.61–0.71): no overfit at 200k × 50 epochs; ~37 min per run sharing the lab GPU. Landing (the run's own
categorical probes read the target cell off the written residual) sits at the probes' natural-residual ceiling on 5 / 8 / 16-ray; ⚠ on
128-ray the exact-cell criterion is near-unreachable (LIN 0.06 natural) and the MLP reads the write as the exact cell in 0.12 vs 0.42
natural — a coarser write than the 1,028-class grid. **What the waterfalls show** (`experiments/categorical_inverse/outputs/`, four
families): the Edit Index scores rollout step 0 (the edit frame), and the new map's gain is THERE — edit-frame RMSE drops (8-ray 0.106 →
0.090), whole-rollout RMSE to the edited truth does not move (0.234 → 0.233). How long the edit holds is the family's property, identical
under both maps: it holds for all 15 steps at 5 / 8-ray and returns to the unsteered trajectory within a frame or two at 128-ray.
Parity gate on `L-dw-8ray-20m__seed1`, both bases (03:15): 22,386 of 22,390 leaves bit-identical, max |Δ| 0; the cartesian block —
continuous IM / IM-NN included — is identical to disk, the ONLY differences are the factorised block's old IM arms (282 → 264 arms,
`best.IM*`, `inverse_map`); the run's `probes/` untouched (27 files). The token path (`L-dw-8ray-tok-20m`) was never exercised by the
frame previews: its smoke passes, and its production-recipe preview (03:52) gives **8ray-tok +0.780 / 0.20 vs the old map's +0.623 / 0.30** (PI +0.228, GS +0.351; R² 0.641 = in-sample). Catch-up queue jobs are DRAFTED in the branch (`experiments/categorical_inverse/
queue_drafts/`, not installed): parents (arms only, with the previews' cached maps copied in), then the 5 / 8 / 16-ray members (~1.9 h a
family on the lab), then 8ray-tok. ⚠ `score_pending.sh` holds no lock — a catch-up must never overlap a replicate's scoring stage on
the same host (in the `gpu` lane it cannot).
**2026-09-20 04:26 — dw-16ray complete (n = 3); the refactored scorer VERIFIED on the 4090 too.** `rep_dw-16ray_s2` (9.26 h) was its first pass on `pim/scoring`: one `=== scoring` (seed 2), 38 skips, no `stale`, no output errors; five `nothing added` lines there (its tree lacks the lab-scored 5-ray members and 8-ray seed 1 — the lab shows eight); the remote notebook is still 6 cells / 174 lines with the 16-ray entry after its in-place write; seed 2 came back with frustum / cartesian / **appearance-fac** on its own (the `dw_extra_targets` line works — no catch-up job), eval version unchanged, MLP skill identical to its siblings, case-level fields present. Stage B's moved layout script (forwarder) worked on that host as well. dw-16ray (guarded selection): cartesian IM +0.663 ± 0.007 (guard +0.288 ± 0.004), PI +0.153 ± 0.021 (+0.925 ± 0.020), GS -0.121 ± 0.010; appearance-fac IM +0.689 ± 0.007, GS +0.255 ± 0.006, PI -0.050 ± 0.014. Factorised IM across the ray family so far: 5-ray +0.849 ± 0.005 > 8-ray +0.718 ± 0.013 > 16-ray +0.689 ± 0.007 — 8 vs 16 is the closest contrast (gap 0.029, ~2–3 SD). The 4090 moved on to `ctrl_corpus_oth` (the Othello probe-corpus-size control, its first run through the queue); the lab is in `rep_dw-128ray_s1`'s scoring stages. 9/32 done.
**2026-09-20 05:45 — corpus-size controls redefined after a memory kill; TWO SESSIONS were editing the queue.** `ctrl_corpus_oth`
was OOM-killed on the 4090 at 05:00 (40.7 GB anonymous in the Othello inverse-map fit at 60k games; GOTCHAS 2026-09-20). Kept and
pulled to the lab: 20k games LIN 0.9747 · PI +0.819 · ND +0.748 · IM +0.810 (canonical 0.9751 / +0.818 / +0.749 / +0.806 — a different
20k games reproduces everything to ±0.004); 60k games LIN 0.9752 · PI +0.819 · ND +0.746 — flat at 3× the rows. Another session
redefined the retry in the queue JSON at 05:05 (commits 4735a94, 7053ddc) while this one was rewriting the scripts; reconciled at 05:40
into `plan.py::control_jobs` (the single source): Othello 20k/40k/60k probes + IM at 20k/40k; discworld 30k/100k + IM 30k/60k; 200k as
its own single-attempt job; all lab-only; controls are not `final_tables` deps; both scripts skip recorded parts and accept either flag
spelling. ⚠ Sevan has been asked to name ONE operator for `experiments/paper_ci/`. The 4090 moved on to `rep_oth-standard_s1` at 05:02.
**2026-09-20 13:25 — dw-128ray complete (n = 3); the first EXTENSION job is running.** `rep_dw-128ray_s2` (lab, 7.19 h): seed 2 with frustum / cartesian / appearance-fac, eval version unchanged, MLP skill identical to its siblings. dw-128ray (guarded selection): cartesian IM +0.574 ± 0.006 (guard +0.317 ± 0.008), PI -0.003 ± 0.009 (+0.985 ± 0.010), GS -0.097 ± 0.027; appearance-fac IM +0.580 ± 0.005, GS +0.286 ± 0.014, PI -0.331 ± 0.024. Under the guarded rule NO probe-derived PI arm edits 128-ray (the scorer's unguarded best is +0.23 at guard 1.55 on all three). Factorised IM over the ray family: 5-ray +0.849 ± 0.005 > 8-ray +0.718 ± 0.013 > 16-ray +0.689 ± 0.007 > 128-ray +0.580 ± 0.005. `rep_dw-noiseless_s1` launched 13:25: resumed from `ckpt/latest.pt` at 390,000 → 512,000 (config `replicate.steps` 512000, `resumed` recorded). ⚠ The config says `batch_order_exact: False`: the discworld block stream has no `skip()`, so the batches after the resume are not the ones an uninterrupted 512k run would have drawn (weights, optimiser state and lr ARE exact at the resume point; the corpus had already been seen ~5× by 390k). The extended members are valid seed-1 / seed-2 runs at 512k, not bit-identical to uninterrupted ones — say so where the extension is described. When it finishes, check: stale `scores.json` parked as `scores.s390000.json` (+ a `runs/MOVES.md` line), a NEW `__seed0_s512000` member laid out (the 421,875 one is outside ±10%), three blocks on both. 11/33 done.
**2026-09-20 14:20 — QUEUE-SIDE NOTES for the categorical-IM deployment (branch `categorical_inverse`, staging clone; NOT installed — Sevan's call).**
Written by the session operating `experiments/paper_ci/` after reading that branch's README + `queue_drafts/`. (1) Every factorised-block "IM" number this
queue has reported (ledger, pings, the ray-family entries above: 5-ray +0.849, 8-ray +0.718, 16-ray +0.689, 128-ray +0.580) is the OLD continuous-state arm that
deployment deletes; the cartesian-block IM numbers are unaffected; the branch's preview of the NEW arm on the parents: +0.913 / +0.866 / +0.823 / +0.637. (2) The
draft note "dw-128ray has no members yet" is stale: all three 128-ray members were scored by 13:25, so 128-ray needs a catch-up job like the other three families
(16 in-scope runs in all: 4 parents + 12 members; ~37 min each, ~10 h of lab gpu lane, ~7.5 h if the preview's cached maps are copied into the parents' `probes/`).
(3) Do NOT hand-edit `queue/final_tables.json` to add the catch-up ids: `plan.py` regenerates every still-queued job it builds, so the ids must be registered in
`plan.py` (as `appendix_prediction` and the controls were) or the next `plan.py` run drops them. (4) Remaining discworld jobs are all lab-only (noiseless
extensions, blink, the controls) and out of the new arm's scope; the 4090 runs only Othello from here, so its pull is for code identity, not behaviour.
(5) Install procedure this session will follow when Sevan says go — the refactor cut-over's: audit the branch, merge between scoring passes, run the clear
script, register the catch-up jobs in `plan.py`, pull on the 4090 during a training stage, verify the first pass on each host. ⚠ STILL OPEN: Sevan has been
asked (05:45, repeated) to name ONE operator for the queue — two sessions have been editing it and both write this file.
**2026-09-20 17:15 — the EXTENSION path verified end to end (`rep_dw-noiseless_s1`, lab, 3.60 h); one transient fixed.** Seed 1 resumed 390,000 → 512,000, its 390k `scores.json` was parked as `scores.s390000.json` (MOVES line), the layout script found the 421,875 member outside ±10% and laid out `__seed0_s512000`, both runs were scored fresh with frustum / cartesian / appearance-fac (eval version unchanged), and the tables now pool the 512k set: dw-noiseless n = 2 at [512000], left out [421875] — cartesian IM +0.594 ± 0.003, GS -0.198 ± 0.024, PI -0.093 ± 0.051; factorised GS +0.311 ± 0.008. `rep_dw-noiseless_s2` launched 17:01 (resumed at 390,000; n = 3 when it is scored, ~20:30). ⚠ TRANSIENT found and closed: the trainer writes the NEW budget into config.json when an extension STARTS, but `replicate.sh` parks the old scores only when training ENDS, so for the training hours the tables would pool 390k scores as a 512k member (it happened silently to seed 1 13:25–14:54). Seed 2's stale scores were parked by hand at 17:10 (the driver's own step then no-ops), and `dispatch.py::park_stale_snippet` now does it at the LAUNCH of every `extend` job, on the launching host (tested in scratch: parks scores + variance once, idempotent, leaves a run already at budget alone) — `rep_oth-adjflip_s1` (390k scores AND variance.json) is the one remaining case. The driver itself is NOT edited while bash executes it; after the queue drains, move the parking in `replicate.sh` to BEFORE the training call. 12/33 done.
**2026-09-20 19:36 — dw-noiseless complete (n = 3 at [512000], left out [421875]): FIVE of the six discworld families are done.** `rep_dw-noiseless_s2` (lab, 2.55 h): the driver's own parking step no-opped after the hand-parking, stage B found the 512k member, seed 2 scored fresh with three blocks. dw-noiseless (guarded selection): cartesian IM +0.592 ± 0.005 (guard +0.334 ± 0.020), PI -0.080 ± 0.042, GS -0.186 ± 0.027; factorised GS +0.309 ± 0.006, PI -0.403 ± 0.009. Across the five families (n = 3 each, 512k) — CARTESIAN IM: 5ray +0.808 ± 0.009 · 8ray +0.720 ± 0.006 · 16ray +0.663 ± 0.007 · 128ray +0.574 ± 0.006 · noiseless +0.592 ± 0.005; factorised GS: 5ray +0.559 ± 0.012 · 8ray +0.445 ± 0.014 · 16ray +0.255 ± 0.006 · 128ray +0.286 ± 0.014 · noiseless +0.309 ± 0.006. (The factorised IM column is the old continuous arm — see the 14:20 note.) Remaining discworld: dw-blink (2 re-trainings, lab). The lab started `ctrl_corpus_dw` at 19:35 (probes 30k / 100k, inverse map 30k / 60k; its first real run; results persist per part); `ctrl_corpus_dw_200k` and `ctrl_corpus_oth` follow. 4090: `rep_oth-standard_s1` ~425k / 512k. 13/33 done.
**2026-09-20 20:10 — `ctrl_corpus_dw` OOM-killed at 100k sequences (46.8 GB anonymous, lab cap 45 GB); redefined, 200k job DELETED.** The dense regression
fit was written for 30k (GOTCHAS 2026-09-20 evening). Recorded before the kill: 30k of `probe_250k` reproduces the canonical noiseless numbers to ±0.007 (LIN 0.8718 /
MLP 0.9727 / PI +0.197 / GS −0.063 / IM +0.592); the inverse map at 60k gives IM +0.604 (g R² 0.742 → 0.755). Held the job before its retry, removed the dead process's
72 GB scratch memmap (+ a 2.4 GB orphan from 18:26; `fuser` clean), redefined in `plan.py`: probes 30k / 60k, IM 30k / 60k, ONE attempt, starts after `ctrl_corpus_oth`
(running on the lab since 19:57). Sevan's "30k vs 200k" question is answered by the capacity sweep's streamed 250k fits for decodability and by this control's 2× point
for editability; a 200k editability point needs a streamed regression fit — not mid-queue. 32 jobs, 13 done.
**2026-09-20 20:35 — both corpus-size controls DONE and recorded; `rep_oth-standard_s2` launched on the lab (20:30).** `ctrl_corpus_oth` (lab, 0.25 h, 41 GB peak; inverse map
at 40k games fitted) and the redefined `ctrl_corpus_dw` (lab, 0.24 h; 30k part skipped as recorded) exited 0, no scratch leaked. Result = dated entry in
`findings/probe-capacity.md`: at 2–3× the canonical probe rows LIN / MLP skill move ≤ 0.001, PI / ND keep the same arm (≤ 0.006), Othello IM is flat, discworld GS drifts
down inside its seed SD, and the ONE mover is the discworld inverse map (IM +0.592 → +0.604 per doubling, ~2 seed SDs; canonical IM a slight under-estimate, no ordering
changes). Not answered: the literal 200k point (streamed regression fit, post-deadline) and discworld's GUARDED PI / GS cells (the scripts record the unguarded arm).
Running: `rep_oth-standard_s1` (4090, ~23:30) and `rep_oth-standard_s2` (lab). 32 jobs, 15 done. Note for whoever touches the notebook: `master_eval.ipynb` carries the
UNCOMMITTED `dw_bases` default edit of 2026-09-19 night — do not `git checkout` it.

**2026-09-23 09:55 — three APPENDIX catch-ups queued on the idle 4090 (Sevan 09:30 / 10:00), all before `final_tables`.** From a read of `paper/paper_draft.tex` against what the queue produces: the main text and most of the appendix are filled or land today (Table A1's Rayworld floors from `appendix_prediction`, the variance section from Table 5); the blanks that needed new work were the NN column of the IM-vs-NN table, the IM cells of the bins-vs-grids table, the legal/illegal table (handcrafted — Sevan hands it to another agent) and the token table's mean-frame rows. Done: (1) `inverse_map.nn_r2` is now STORED (computed since 2026-09-16, never written) and `PIM_ADD_NN_R2=1` re-adds the cached inverse arms of blocks predating it → job `nn_r2_shortlist` (ten parents; launched 09:26, ~1 min per Othello run; verified on L-oth-adjacent-20m: nn_r2 per point 0.18…0.93 beside g 0.36…0.98, 282 arms intact, IM arms equal to the lab's to 1e-4); (2) SETTINGS `dw_cat_im` targets += appearance / grid-6x5 / grid-10x3 / grid-16x8 → `catim_grid_8ray` (four ~40-min categorical maps on the 8-ray parent); (3) the token model's mean-frame readout gets a guard — `token_bench.scorecard` `fidelity_ratio_expected` (identity 1.0; a push toward the edited frame 0.17) — and `PIM_FORCE_RESCORE=<runs>` rescores a named run from scratch → `token_rescore_8ray` (every arm of the 8-ray token model recomputed under one rule; the current tex row's mean-frame PI/GS cells match no stored arm, so the rescore replaces them; the `prediction` block it drops is re-added by `appendix_prediction`). Commits ad3195f, 1748cef; the 4090 pulled to HEAD (its dirty rsync copies reverted, the killed control's untracked scores parked in `.scratch/`). `final_tables` depends on all three. Mean-frame INDEX cells already on disk (`zone_edit_index_expected`): continuous PI +0.12 / GS −0.08 / IM +0.84, categorical (appearance-fac) +0.33 / +0.48 / +0.92 at the reported frame-set arms.

**2026-09-23 08:15 — all five `pseed_*` extras done; the 4090 is idle for the rest of the queue.** The three oth-adjflip members, 10 probe refits each + 10 inverse-map refits, at the canonical arm (`variance.json` in each member): seed 0 (`__seed0_s512000`) LIN 0.9479 ± 0.0007 · PI +0.375 ± 0.014 · ND +0.392 ± 0.006 · IM +0.642 ± 0.036; seed 1 LIN 0.9485 ± 0.0007 · PI +0.337 ± 0.008 · ND +0.337 ± 0.003 · IM +0.650 ± 0.025; seed 2 LIN 0.9485 ± 0.0006 · PI +0.238 ± 0.016 · ND +0.270 ± 0.016 · IM +0.638 ± 0.027. Reading across the whole set (oth-standard, dw-8ray, adjflip × 3): refitting the LINEAR probe moves the Edit Index by ≤ 0.016 and skill by ≤ 0.0007; refitting the INVERSE MAP moves IM by 0.004 (oth-standard) to 0.036 (adjflip) — on adjflip that is the size of the training-seed spread (± 0.028), so the IM ± there is instrument as much as model; everywhere else the ± is the model. Six `stall` lines on the `pseed_*` jobs, all false (progress never read from these jobs). Remaining: `rep_oth-noflip_s2` (lab, 270k at 07:48, scored ≈ 16:00) → `final_tables` → `appendix_prediction`.

**2026-09-23 05:10 — `pseed_dw-8ray` done (4090, 2.83 h): the discworld probe-seed spread is as small; `pseed_oth-adjflip_s0` fixed and requeued.** `runs/ray_ablation/L-dw-8ray-20m/variance.json`: full state, 10 probe seeds — LIN skill 0.9503 ± 0.0005, PI at the seed-0 arm +0.220 ± 0.001; `appearance-fac`, 6 seeds — skill 0.9355 ± 0.0003, PI at the canonical arm +0.381 ± 0.005 (ratio 0.96 ± 0.006), ND +0.474 ± 0.000. Training seeds gave PI ± 0.022 on this family. Every `stall` line on the `pseed_*` jobs (three so far) was the blind spot — the check never saw progress on these jobs at all (166 min at 05:00 while the log advanced every ~27 min). `pseed_oth-adjflip_s0` launched at 05:04 against `__seed0_s492188`, a member name guessed in `plan.py` before the extension (the real one is `__seed0_s512000`) → FileNotFoundError; fixed (7658024) and the file regenerated at its requeue, before attempt 2. Running: `pseed_oth-adjflip_s0` retry (4090), `rep_oth-noflip_s2` (lab, ~15:30). 34 of 37 done.

**2026-09-23 02:15 — `pseed_oth-standard` done (4090, 0.86 h, attempt 2): PROBE-seed spread on `L-oth-20m` is an order of magnitude below the TRAINING-seed spread.** Ten refits of the linear probes (20k games) + ten refits of the inverse map, `runs/initial_othello_comparison/L-oth-20m/variance.json`: LIN skill 0.9753 ± 0.0002 (best point 7 in 10/10); at the canonical edit arm PI +0.817 ± 0.003 (ratio 0.296 ± 0.002), ND +0.747 ± 0.002; IM +0.809 ± 0.004 (ratio 0.376 ± 0.007, g R² 0.825 ± 0.003), IM-NN +0.017 ± 0.007. Training seeds gave PI ± 0.015, IM ± 0.034 — so the ± in the tables is the model, not the instrument. The script's `own_best` entries (pt 7, α 100, ratio 6.3) are its pre-guard skill-best-point rule and are not a reported number. Attempt 1 of every `pseed_*` job had died at rc 127 (bare `python` in a systemd unit); fixed in `plan.py` (b4a6788), queue regenerated before any second attempt. The 02:07 `stall` line on this job was the known blind spot (progress not read from `attempt_2.log`). `pseed_dw-8ray` running on the 4090.

**2026-09-23 01:20 — oth-adjacent complete (n = 3 at 512k); `pseed_oth-standard` launched on the 4090 (the first probe-seed extra).** `rep_oth-adjacent_s2` (4090, 17.80 h; one pass, 282 arms, no errors; `runs/MOVES.md` intact after the sync-back — the merge fix holds). Ledger (guarded): LIN 0.987 ± 0.001, GS −0.176 ± 0.016, ND +0.270 ± 0.094, PI −0.121 ± 0.196 (members −0.280 / −0.180 / +0.098), IM −0.118 ± 0.304 (−0.119 / +0.186 / −0.422), GS fidelity −4.9 ± 0.4. The wide cells are again the failing editors on an inert world: the guarded arm lands on a different alpha per seed and the index straddles zero. For Sevan's main-table question (2026-09-22 17:30): the seed SD is ≤ 0.004 on every decodability cell, ≤ 0.04 on every editability cell the text relies on (discworld IM ≤ 0.012; oth-standard / adjflip IM ≤ 0.034), and > 0.1 only on Othello cells where the editor fails (adjflip PI; adjacent PI and IM; adjacent GS fidelity) — mark those † in the main table, spreads in Table S5. Three of four Othello families at n = 3; noflip's seed 2 is the last training job (lab, ~15:30). 32 of 37 done.

**2026-09-23 00:05 — `rep_oth-noflip_s1` done (lab, 15.53 h): oth-noflip at n = 2; `rep_oth-noflip_s2` launched on the lab (the LAST training job).** Two members scored in one pass (eval_version 2026-09-12.1, 282 arms each, no errors). Ledger (guarded): LIN 1.000 ± 0.000, PI −0.967 ± 0.002, ND −0.970 ± 0.002, GS −0.495 ± 0.050, IM −0.877 ± 0.034 — vs the canonical parent's IM −0.553 / fidelity −1.58: on this inert world NO arm of the parent passes the guard, so its cell is the unguarded best (a degraded arm), while both members have an arm inside the guard that does essentially nothing (fidelity ≈ 0.00, index ≈ −0.9). Same selection rule, different branch — the canonical-vs-members gap here is the guard rule on a row where every editor fails, not a seed effect; state it in the Table 5 caption. Overnight before this (2026-09-22): `rep_oth-adjacent_s1` done (4090, 07:29), `rep_dw-blink_s2` done (lab, 08:29) → dw-blink n = 3 (all six discworld families complete); fidelity reported as 1 − ratio since e359c2b (11:00). 31 of 37 done; running: `rep_oth-adjacent_s2` (4090, ~02:00) and `rep_oth-noflip_s2` (lab, ~15:30); then the five `pseed_*` extras, `final_tables`, `appendix_prediction`.

**2026-09-21 13:25 — `rep_oth-adjflip_s2` done (4090, 13.74 h): the Othello EXTENSION path verified; oth-adjflip at n = 2 (512k); `rep_oth-adjacent_s1` launched on the 4090.** Seed 2 resumed 160k → 512k, the new `__seed0_s512000` member was laid out and both scored in one pass (eval_version 2026-09-12.1, 282 arms each, no errors); the 421,875 member is left out of the pool. Ledger so far (guarded): LIN 0.948 ± 0.000, IM +0.607 ± 0.002 (canonical +0.664), GS −0.064 ± 0.018; PI +0.083 ± 0.284 (members +0.284 / −0.117 — the guarded PI arm flips between a landing and a failing alpha across seeds) and ND +0.441 ± 0.155 — the wide cells are selection jumping between arms, to be read against `variance.json` when seed 1 lands (≈ 2026-09-22 05:00). **`runs/MOVES.md` erased a SECOND time** by this job's sync-back (the 4090's copy lacked the 13:06 adjflip-s1 park lines) → restored from git and the dispatcher now MERGES `runs/MOVES.md` from a remote (appends lines it lacks, never overwrites; 3365814, installed atomically between ticks — the dispatcher is a fresh process every 2 min, unlike the bash drivers). 22 of 37 done; gpu lanes: adjflip_s1 (lab, extension from 390k) and adjacent_s1 (4090).

**2026-09-21 13:10 — oth-standard complete (n = 3 at 512k): the FIRST Othello family done; `rep_oth-adjflip_s1` (extension) launched on the lab.** `rep_oth-standard_s2` (lab, 16.58 h; scoring 12:47–13:05, one pass, eval_version 2026-09-12.1, 282 arms, no errors). Pooled, mine/theirs symdiff, guarded selection: LIN 0.969 ± 0.003, MLP 0.971 ± 0.002, PI +0.808 ± 0.015 (members +0.798 / +0.801 / +0.826), ND +0.732 ± 0.027, GS +0.808 ± 0.012, IM +0.778 ± 0.034 (+0.795 / +0.801 / +0.738), IM-NN +0.029 ± 0.007; guards PI 0.52 ± 0.05 (canonical 0.30 — the members' guarded PI arm sits at a milder alpha), ND 0.34, GS 0.34, IM 0.40 ± 0.04. Seed 2 is the outlier on both sides (highest PI, lowest IM) — Othello's seed spread is 3–7× discworld's on the same editors; the canonical parent (+0.818 / +0.828 / +0.806) sits at the top of every member range. Extension launch verified: the dispatcher parked seed 1's 390k `scores.json` + `variance.json` as `.s390000.json` with two `runs/MOVES.md` lines at 13:06:32 and the trainer resumed from `ckpt/latest.pt` at step 390000. `rep_oth-adjflip_s2` is in its scoring stage on the 4090 (done ≈ 13:20). 21 of 37 done.

**2026-09-21 09:45 — MONITORING RESUMED by this session at Sevan's word ("you should be the only active monitoring agent now"); overnight verified.** Events since the hand-back: the five `catim_*` catch-up jobs all done (21:32 → 05:01, ~1.9 h each; every 46-min `stall` line beside them was a false alarm — a fit in progress writes nothing the stall check sees); `rep_oth-standard_s1` done on the 4090 (18.33 h) → oth-standard at n = 2 (PI +0.800 ± 0.002, GS +0.815 ± 0.002, IM +0.798 ± 0.004, LIN 0.970 ± 0.003); `rep_oth-adjflip_s2` launched on the 4090 at 23:22 as an EXTENSION (seed 2 resumed from its 160k checkpoint, config rewritten to 512000, `batch_order_exact` absent = default; at 440k at 09:34, scoring done ≈ 13:00); `rep_oth-standard_s2` on the lab at 400k (≈ 14:30). **Categorical IM landed on all 16 ray-family members** (appearance-fac block, IM only): 5-ray +0.909 / +0.913 / +0.913 / +0.915, 8-ray +0.866 / +0.869 / +0.874 / +0.879, 16-ray +0.819 / +0.821 / +0.823 / +0.823, 128-ray +0.634 / +0.637 / +0.638 / +0.640 (parent first), guards 0.25–0.29. **One regression fixed:** `runs/MOVES.md` in the lab tree had LOST the two noiseless parking lines (a 0/2 diff) — the sync-back of `rep_oth-standard_s1` pulled the 4090's OLDER copy (its checkout predates 3a7d266) over the lab's, because `runs/MOVES.md` is in every replicate job's outputs. Restored from HEAD and the HEAD copy scp'd to the 4090 (its `replicate.sh` only ever appends), so the next sync-back carries it forward; a proper fix (pull on the 4090, or drop MOVES.md from remote outputs in favour of a per-job snippet) waits for a training stage / the drain. Digest last sent 08:23; dashboard ETA 09-23 16:19. Still owed from the 21:35 entry (not started): REGISTRY run rows, `findings/ray-ablation.md` / `inverse-probe.md` numbers, the four waterfalls.

**2026-09-20 21:35 — CATEGORICAL INVERSE MAP DEPLOYED (Sevan's go 21:15, option A: catch-up tonight, sharing the lab GPU).** In order, each step verified: (1) the pending `dw_bases` notebook edit committed (f588e6f) — it sat beside the branch's new cell; (2) `categorical_inverse` merged in the LIVE tree (637ea41; the one conflict was the nbconvert-re-serialised notebook: live copy kept, cell [2b] inserted with the notebook editor; suite 307 pass on the merged tree); (3) `clear_continuous_im.py --apply`: the old continuous-state IM / IM-NN arms removed from **57 categorical blocks in 27 runs** (dated backup in each `scores_backup/`; idempotent; regression blocks untouched; the ledger copes with the blank cells); (4) the previews' cached maps copied into the five parents' `probes/` (routed by model fingerprint, add-only); (5) SMOKE of the production path on `L-dw-5ray-20m`: arm added in 20 s from cache, **+0.9133 / 0.25 (pt 5) = the preview exactly**; (6) catch-up registered in `plan.py` as five CPU-LANE jobs (`catim_parents`, `_5ray`, `_8ray`, `_16ray`, `_128ray`; wrapper `experiments/categorical_inverse/drivers/catchup_job.sh` never starts beside another `master_eval` execution or within 2.5 h of the gpu-lane job's scoring stage). `catim_parents` done 21:32: **5-ray +0.913 / 0.25 · 8-ray +0.866 / 0.27 · 16-ray +0.823 / 0.29 · 128-ray +0.637 / 0.29 · 8ray-tok +0.780 / 0.20**, each block recording `state onehot-labels+cartesian-velocity, n_seq 200000, epochs 50`, IM only. The 12 members follow (~37 min each, ~7.7 h; `catim_5ray` running). ETA model: `gpu_share` (dispatch.py `gpu_share_extra`, the current rate off the last two log lines) + MEASURED Othello rates in `plan.py` (capped lab 9.6 steps/s = 14.8 h per 512k run, not 12.7; 4090 8.54) — the dashboard ETA moved from 09-23 14:30 to ~18:10 because it had been optimistic, not because work was added to the gpu lanes. NOT done, deliberately: the 4090 pull (it runs only Othello from here; its tree is dirty — see the hand-back entry — and no pull is needed for correctness). Owed when the members land: REGISTRY run rows, `findings/ray-ablation.md` / `inverse-probe.md` numbers, the four waterfalls into the record.
**2026-09-20 20:50 — HAND-BACK: the launching session (PID 4145414, local VSCode window) STANDS DOWN from `experiments/paper_ci`; the session Sevan types into
(`physically-implicit-modeling-8f`, Remote-SSH) is the ONE operator (Sevan 20:43, relayed by that session).** The two were the same conversation in two processes;
nothing Sevan typed after 2026-09-19 23:14 reached the launching one, and nothing it wrote reached him. Everything below is what is NOT visible in git.
- **Nothing is held, nothing is half-done.** No `PAUSED` flag, no `hold` on any job, no job file or state file edited by hand since `aab7874`. `plan.py` is exactly as
  committed (controls redefined, `ctrl_corpus_dw_200k` deleted) and `queue/*.json` was regenerated from it. No pending edit to `plan.py`, `dispatch.py`, `run_job.sh`,
  `replicate.sh` or `score_pending.sh`. Last commit of that session: `a30cee2` (the corpus-size control's record), pushed.
- **Uncommitted in the lab tree:** only `notebooks/master_eval.ipynb` — NOT that session's edit (it is the `dw_bases` default change of 2026-09-19 night plus
  in-place execution outputs). It never ran `git checkout` on it after 2026-09-19 12:25.
- **Nothing depends on that session's process:** dispatcher + watchdog timers, the running units and the dashboard (`pimci-dashboard.service`, `tailscale serve /ci`)
  are systemd user units. Its only background waiter was a 30-min `tail -F logs/paper_ci/dispatch.log | grep` Monitor — stopped at hand-back.
- **The 4090 vs a plain pull:** checkout `84c2e78`. Dirty there: `experiments/paper_ci/scripts/ledger.py` (the dispatcher's `push_inputs` rsync of `scripts/` — the
  lab's version, not a hand edit), `notebooks/master_eval.ipynb` (execution outputs), and UNTRACKED `experiments/probe_corpus_size/scores/` (left by the 05:00 killed
  attempt; its content was pulled to the lab at 05:10 and is committed). A plain `git pull` there will REFUSE (the untracked scores file is now tracked upstream;
  `ledger.py` differs): first move that `scores/` directory aside and `git checkout -- experiments/paper_ci/scripts/ledger.py`, and only during a TRAINING stage (the
  notebook is mid-write during a scoring stage). No pull is needed to keep the queue correct: among files a remote job executes, HEAD differs from `84c2e78` only in
  `dispatch.py` (lab-only) and `ledger.py` (already pushed). No stray `.scratch/*.npy` on either host (checked 20:45); the gitignored probe cache of the killed control
  is still under `experiments/probe_corpus_size/probes/` on the 4090 (harmless, a few GB).
- **Intentions that session carried, none started:** (1) categorical-IM install — waits for Sevan's go; the 14:20 notes above stand, plus one sentence: MERGE IN THE
  LIVE TREE ONLY, never inside the staging clone (`runs/` is a symlink there and `runs/MOVES.md` is tracked). (2) `nn_r2` stays on hold (Sevan). (3) `rep_oth-adjflip_s1/s2`
  are EXTENSIONS: the dispatcher parks the stale 390k `scores.json` / `variance.json` at launch (a78f386) — verify the park line in `runs/MOVES.md`, that a new
  `__seed0_s512000` member is laid out, that the ledger pools the 512k set, and read `batch_order_exact` in the resumed `config.json` (False on discworld; never
  checked on Othello); seed 2 resumes from ~160k, so it is nearly a full run. (4) On `appendix_prediction` done: the accept list in
  `experiments/bayes_floor/QUEUE_HANDOFF.md`, then `findings/predictive-quality.md`, then tell Sevan Table A1 is ready. (5) When the queue drains: ledger → Table 5
  (SD + CI panels) → `findings/seed-variance.md` (one entry per family: n, budget, SD, guard k/n; extension members are not bit-identical to uninterrupted runs on
  discworld; 5-ray factorised GS canonical +0.598 at 780k sits ~3 SD above its 512k members — a budget mismatch to state) → REGISTRY run rows → brief `done`; archive
  the smoke artefacts (`state/smoke_*.json`); move `replicate.sh`'s parking step to BEFORE training and point it at `scripts/layout_checkpoint_replicate.py`.
  (6) Deprioritised by Sevan: the Othello ceiling bootstrap.
- **Shared memory directory:** that session appended a 20:40 state paragraph to `paper-ci-queue-in-flight.md` and, at hand-back, replaced its own 19:45
  "working arrangement" paragraph with the decision above. It touches nothing further — repo, queue, hosts or memory — unless Sevan types to it.
_(Build record follows.)_

**Sevan (2026-09-18):** a training-seed spread on every main-table number (10 shortlist runs), n = 3,
mean ± SD as the readout and a t-based 95% CI secondary, on the two GPUs only, in ≤ 5 days; the
supplemental runs and the gridified layouts get none. Brief: `research/directions/paper-confidence-intervals.md`.
Protocol written generically as `harness/MULTIDAY.md` (queue + timer dispatcher + per-host watchdog +
dashboard + two ntfy channels); `experiments/paper_ci/README.md` is the operator's manual.
**Plan (Sevan 20:40: ONE budget, 512k everywhere; `appearance-fac` only):** 22 replicate/extension jobs
(dw-noiseless and oth-adjacent-flip's 390k sets are extended) + 2 corpus pushes (dw-8ray / dw-5ray → remote)
+ 5 probe-seed extras + final tables; greedy schedule ≈ 4.2 days (`plan.py --show`). 780k everywhere ≈ 6 days.
**State:** lab timers installed and ticking (`pimci-dispatch` every 2 min, `pimci-watchdog` every 5 min,
`pimci-dashboard` server); remote watchdog timer installed; `tailscale serve` publishes the dashboard at
`https://sevan-ubuntu-lab.tail9a3a96.ts.net/ci/`; smoke jobs ran through the real launch path on BOTH
hosts (the remote one failed once — the job file was not on the remote — fixed: the dispatcher now
pushes config/scripts/job file before every remote launch). Code changes this session: case-level
spread beside every arm (`edit_index.case_stats` / `ratio_ci95`, wired in both scorecards; Othello's
PI/ND/GS guard CI needs one notebook line — owed), `tables.pool_replicates` gains `_ci95` / `_values`
and pools every editor's guard, Table 5 has SD and CI panels, `layout_checkpoint_replicate.py` lays
out a new member when the existing one is > 10% off the budget, `replicate.sh` extends a finished
replicate and parks its stale scores, `score_pending.sh` honours `PIM_SCORE_ONLY=1`. Tests: 281 + 4 pass.
**Launch = `rm experiments/paper_ci/state/PAUSED`.** Remote synced (runs, non-train splits, Othello
corpora, code at HEAD after the pull). Hardening done by Sevan 20:40: nvidia packages held,
apt-daily-upgrade.timer stopped, 5090 at 450 W. Digest topic subscribed. Dashboard: fidelity / decodability views + PWA insets.
_Previous: 2026-09-18 18:30 PT — orientation session; 2026-09-15/17 paper-session work committed as 4c725b5_

## 🔄 oth-adjacent-flip seed variance — launched 2026-09-16 17:35 PT, unit `oth_adjflip_seeds` + watcher (`scripts/drivers/oth_adjflip_seeds.sh`, logs `logs/oth_adjflip_seeds/`)

**⏸ PAUSED 2026-09-17 09:25 PT at Sevan's request (he wants the GPU).** Stages A–C are DONE: the
parent, `__seed0_s421875` (the parent's own checkpoint at a matched budget) and `__seed1` are all
scored AND carry 10 probe seeds each. Seed 2 stopped at step 160,000/390,000 with `ckpt/latest.pt`
written. **To resume, relaunch both unit commands from the driver's header** — A–C skip, seed 2
continues from its checkpoint (~5.8 h train + ~35 min probe seeds).
**Results so far (symdiff, best arm):** parent +0.399/2.09 PI, +0.138/1.75 GS, +0.664/0.51 IM ·
seed-0 ckpt +0.364/1.90, +0.077/2.54, +0.639/0.50 · seed 1 +0.379/1.86, +0.116/2.86, +0.616/0.40.
Training-seed spread at the matched budget: PI 0.015, GS 0.039, IM 0.023; every editor keeps its
best point. Probe-seed spread (10 seeds): skill ±0.0008, PI ±0.007–0.014, ND ±0.003–0.005,
IM ±0.017–0.026 — so refitting g moves IM about as much as retraining does. GS is NOT probe-reseeded
(that needs the MLP grid refitted, ~90 min/member).

Sevan: the run's quoted contrasts (symdiff IM +0.66/0.51, ND +0.58, PI +0.40) carry no spread. Two
sources, the `experiments/seed_variance` convention: RUN seeds (re-train at 390k with `--seed 1` /
`--seed 2` via `scripts/drivers/replicate.sh`, plus the parent's step-421,875 checkpoint as the
seed-0 member — val at 390k is within 0.1% of the 780k best, and the tables pool replicates into the
parent row's ± at a matched budget) and PROBE seeds (10 seeds of the linear grid **and of the
inverse map g**, which also reseeds IM-NN's retrieval bank → `runs/<run>/variance.json`).
**Order (Sevan): seed 1 is trained, scored AND probed before seed 2 starts.** Stages: A parent probe
seeds (~30 min) → B seed 1 train 390k (~10 h) + seed-0 ckpt member + score → C probe seeds on both
(~1 h) → D seed 2 (~10 h) → E probe seeds. ETA ~14:00–17:00 on 2026-09-17 (rate read from the
heartbeat once training starts; the parent ran 8.4 steps/s on the 4090, `L-oth-20m` 11.2 here).
**Sevan may stop this in the morning to use the GPU and resume later:** `systemctl --user stop
oth_adjflip_seeds oth_adjflip_seeds_watch`, then relaunch the same `systemd-run` command — finished
stages skip, and an interrupted training resumes from `ckpt/latest.pt` (replicate.sh now passes
`--resume`; it previously died on train.py's guard). The driver exports
`PIM_SKIP_TOPICS=training_curve`.
Prep (commit 7d08991): IM/IM-NN section in `probe_seeds_othello.py` (`--im-seeds`, `--im-points`,
`--out`); its canonical edit point is now the best arm under **symdiff** — `scores.json["best"]` is
keyed on the UNION index, which named pt 4 where the tables quote pt 5; `RetrievalBank.r2`
subsamples to 20k held-out rows (Othello's full set is ~450k against a 1.8M-row bank). Smoke:
1 linear seed + 1 IM seed on the parent (3.0 min / 6 s per seed; artefact in
`experiments/seed_variance/scores/adjflip_probeseeds_smoke_2026-09-16.json`), and a 200→400-step
Othello replicate train + resume in `runs/_pipeline_smoke/L-oth-adjflip-smoke__seed1`.
_Previous: 2026-09-16 13:55 PT

## ✅ dw-128ray chain — 2026-09-15 20:57 → 2026-09-16 13:45 PT, unit `dw_128ray` — DONE (`scripts/drivers/dw_128ray.sh`, logs `logs/ray_ablation/dw_128ray/`, `logs/dw_128ray_fac/`)

**Outcome (13:45):** dw-128ray lands on dw-noiseless on every column (cartesian IM +0.57/0.32, PI +0.24/1.57; appearance-fac LIN 0.26 / MLP 0.71, PI −0.00, ND +0.54, GS +0.31, IM +0.57) — the coarse-ray gains are a RAY-COUNT effect, not disc size. Recorded: `findings/ray-ablation.md` addendum, REGISTRY run row, both table notebooks re-executed. Hardware: two hard freezes during stage D (07:22, ~10:20; journal ends abruptly, no oops/OOM/Xid) → Sevan replaced the PSU; the GPU whined at 530 W on the new PSU, so the last relaunch ran under `nvidia-smi -pl 400` (Sevan raised it back later). One slip: relaunch #5 ran master_eval without `PIM_SKIP_TOPICS=training_curve`, which scored `L-dw-20m_s001000` / `s004000` (complete, valid, unrequested) before I stopped it; the driver now exports the guard.
**Same day, parked:** Table 1b (inverse-map R² beside retrieval R²) is in the paper notebook; the retrieval column is BLANK until `inverse_map.nn_r2` is folded into the scores — patch script ready (`scratchpad/patch_master_eval_nn_r2.py`, NOT applied; ~2 h GPU for all runs, ~50 min for the listed ones). Sevan: hold. Also this session: `experiments/waterfall_gallery/` (six edit waterfalls), `experiments/history_rewrite/` (IM at every history step on dw-noiseless: rewritten history alone carries the edit, +0.65/0.48; +IM at EF +0.63/0.36 vs canonical +0.61/0.36).

Sevan: a 128-ray entry that matches the 5/8/16-ray family EXACTLY — dw-8ray geometry (radius 1.0, wall rays
dropped, max-edit-attempts 2000) with 130 cast / 128 kept rays. dw-noiseless is radius 0.5 / 128 cast / no drop,
so until now ray count and disc size were confounded between it and the coarse-ray family. Stages: B generate
(eval / edits / probe 120k + 250k / 20M corpus 410 GB / edit selection, ~2 h) → C train
`runs/ray_ablation/L-dw-128ray-20m` 780k (~8 h on the 5090) → D master_eval in BOTH bases + both table
notebooks (~40 min) → E appearance-fac (`probe_target_fit.sh`, `logs/dw_128ray_fac/`, ~1 h). **ETA ~09:00 PT
2026-09-16.** Every stage idempotent; relaunch the same unit command to resume (train `--resume`).
Prep this session: `bigcorpus.INSTANCES["dw-128ray"]` (+ the `dw-16ray` block pulled from the unmerged
`dw_16ray` branch, identical text, so its seed ranges are forbidden here too); instance.json; run listed in
`build_paper_tables_and_figs` (before 16-ray) and `build_full_tables`; master_eval `dw_extra_targets` row
(nbformat — Read cap); REGISTRY instance + run rows. Disk: 680 GB free after deleting three orphaned 15 GB
`.scratch/tmp*.npy` (arms.py scratch memmaps from a 2026-09-10 crash). Generator smoke with the family flags:
128-wide obs, 100/100 edits placed. Heartbeat gpu column shows an NVML driver/library mismatch — cosmetic (torch
sees the 5090); a reboot clears it. **When done:** read `scores.json`, update `findings/ray-ablation.md`
(128 vs 16 vs noiseless: ray count or disc size?), flip the REGISTRY run row from IN FLIGHT, re-render tables.
_Previous: 2026-09-14 08:00 PT — dw-8ray-obs5 chain DONE (first section); before that 2026-09-11 ~05:00 PT — oth-adjacent-flip chain on the WSL remote DONE_

**2026-09-17 (paper session) — clip-to-[0,1] pilot on dw-noiseless + dw-8ray:** clipping the
predicted frame changes no conclusion. Largest index move +0.046 (PI, dw-noiseless, cartesian);
the effect is concentrated in PI's FIDELITY (1.93 → 1.52, 1.71 → 1.41) because only PI's write
leaves the feasible range by a lot ([−2.16, +2.61] at α=100 vs the unedited model's [−0.02, +0.82]).
IM / IM-NN untouched (≤0.001). Tables stay UNCLIPPED (decision recorded). Every raw number
reproduces scores.json. Script `experiments/clip_output_pilot/scripts/clip_pilot.py`, scratch
`2026-09-17-clip-output-pilot.md`.

**2026-09-17 (paper session) — Othello edits LAND in probe space and still do not edit:** the α=1
exact-landing PI write reads the requested board (tile 98.4–100%, all 64 squares on standard-noflip)
and leaves the move distribution at the unedited floor on ALL FOUR instances (−0.70 vs −0.93 standard;
−0.958 vs −0.98 standard-noflip); standard Othello's +0.82 needs a 3× overshoot, adjacent-noflip's
+0.18 needs 60×. Identical read-out at α=1 and α=3 spans −0.700 → +0.818, so probe-space landing
carries no information about behavioural landing. GS converges perfectly on the MLP probe (100% /
99.6%) exactly on the two no-flip variants where it is most destructive (fidelity 4.3–6.7). Validated
by recomputing PI's canonical symdiff index (matches scores.json on all four). Script
`experiments/probe_readout_landing/scripts/othello_landing.py`, scores `.../scores/landing_*.json`,
scratch `2026-09-17-othello-probe-space-landing.md`. Owed: same check on the discworld categorical
arms; decide whether the per-α landing table goes in the appendix.

**2026-09-15 (evening) — near-teleport pilot (dw-noiseless):** teleports INSIDE the 1.6-unit exclusion zone
edit as well as canonical ones (IM +0.63/0.32 vs +0.60/0.33; PI +0.26/1.52 vs +0.23/1.61; GS ≈ 0), so a hole in
the joint support does not break IM on discworld; the adjacent-noflip failure is a code-factorisation matter,
not off-support per se. `experiments/near_teleport_pilot/`, `findings/inverse-probe.md`, scratch note. The
repulsion variant is OFF the table as the missing "nothing edits" discworld cell; an equality coupling remains.

**2026-09-15 (later) — IM by reachability on Othello (paper-theory check):** on adjacent-noflip IM lands on the
ordinary reachable single-flip cases (+0.53 symdiff, n 29) and fails on cases with no counterfactual board
(−0.17, fid 1.41, n 30); on standard and adjacent-flip IM lands on both subsets (+0.70…+0.76). The inverse map
generalises past reachability only where colour varies independently in training (flips). Recorded in
`findings/inverse-probe.md` (dated entry) and the scratch addendum; JSONs `ceiling_symdiff_*.json`.
Candidate discworld cell for the paper's matrix: couple the two discs' positions in the generator.

**2026-09-15 (paper session) — Othello Edit-Index CEILING under the symdiff headline measured:** ≈ **+0.91** on
all three instances with reachable counterfactual boards (standard 13 ordinary cases: +0.909 symdiff / +0.692
union; adjacent 29: +0.916; adjacent-flip 32: +0.908), unedited −0.94 … −0.98. Standard ND recovers 91%, PI 83%.
⚠ GS (best full-bench arm) is destructive on the 13-case subset (fid 2.89) — check its per-case fidelity.
Script `experiments/adjacent_flip_ablation/scripts/ceiling_symdiff.py`, scores `…/scores/ceiling_symdiff_*.json`,
scratch `2026-09-15-othello-ceiling-symdiff.md`, findings addendum in `edit-direction-alignment.md`, REGISTRY row.
Owed: bootstrap the ceiling; re-measure the discworld overwrite-oracle ceiling (+0.91, 2026-09-07) on the
1000-case protocol bench before the paper pairs the two.

## ✅ dw-8ray-obs5 chain — 2026-09-13 17:50 → 2026-09-14 07:29 PT, unit `dw_8ray_obs5` — DONE (`scripts/drivers/dw_8ray_obs5.sh`, logs `logs/observer_ablation/dw_8ray_obs5/`)

Sevan: "launch the whole thing now" (N = 5, otherwise dw-8ray's setup; discs may be invisible to some observers).
Stages: B generate (eval/edits/probes + 20M corpus, 128 GB memmap, ~3 h) → B' smoke-train → C train 780k (~8 h)
→ W wait for `experiments/multi_observer/SCORER_READY` (≤ 12 h, hourly warning pings) → D appearance-fac probes
(`--basis cartesian`) + floors → master_eval → test loss → full tables → ALL DONE (~05:30 2026-09-14).
**Relaunch 17:50** after stage B1b failed on 1 seed in ~1,500: the edits generator drew teleport targets from
observer 0's frustum (its own sampler, not `sim.sample_position`); fixed in `edits_dataset._sample_in_frustum`
(commit 53af51c, GOTCHAS 2026-09-13), partial `edits.h5` moved to `_unused/` (ledgered), eval split kept.
**Scorer pieces DONE 18:40 (commit 733b32c), marker NOT yet touched:** per-view factorised appearance target
(`grid_target`: `_view_cells`, mixed-radix joint code, factors (centre+1, length+1) per view; one view is
bit-identical; `tests/test_multiview_target.py`); master_eval `dw_bases_by_instance` (obs5 → cartesian),
instance-aware `discworld_blocks`, baselines loop, `dw_extra_targets` entry; `tables.reg_key` so a lone
cartesian block is the canonical row; both table notebooks execute. Geometry fact: every arena position lights
≥ 1 kept ray in every view (the 7 % invisibility is occlusion). **Generation DONE 21:54** (corpus VERIFIED 20M / 0 dup / 128 GB; selection 1000/1000 from 1040 scanned, 47/4000
identical frames, mean 11 entries changed, teleport mean 1.9 — the round arena is tighter than the frustum); smoke-train
OK; **training started 21:55** (ETA ~06:00). **Scorer smoke PASSED 22:10 and the marker is touched** — it needed one
more fix: `arms.fit_probes` refused categorical targets in any basis but frustum (the guard now covers only the
snapped target, whose cell centres ARE frustum coordinates). Stage D will run fac probes (`--basis cartesian`) →
master_eval (cartesian regression block + fac) → tables; ALL DONE ~07:30.
**DONE 07:29 (chain complete with marker; monitors stopped).** Training 21:55 → 05:51 (476 min; val 0.00302 vs
dw-8ray 0.00574); stage D fac probes + floors 05:51 → 07:05; master_eval + tables → 07:29. **Result: five observers
make the categorical read-out MORE decodable (MLP 0.99) and LESS editable — appearance-fac PI +0.16 / ND +0.42 /
GS +0.37 vs dw-8ray's +0.38 / +0.49 / +0.46; cartesian regression block inert (PI +0.19 / 1.60). Lands on
dw-smooth's numbers.** `findings/observer-ablation.md` (observed), REGISTRY run row, run added to
`build_full_tables`. Post-chain fix: the baselines cell hardwired basis "frustum" for categorical floors → the
obs5 fac floors were fitted but not recorded; now `basis0 = dw_bases_for(inst)[0]`, master_eval rerun (skips all
runs; adds the floor block), tables re-executed. **Open:** waterfall; N = 1 on the circle (arena vs observers
control); second seed; whether obs5 enters the paper shortlist.
**Owed while it runs (original list, kept for the record):** (1) master_eval per-instance
regression basis — dw-8ray-obs5 scores the **Cartesian** block, not frustum (also the baselines cell);
(2) the **per-observer factorised appearance target** in `grid_target.py` — for `n_observers > 1` the
factorisation is (centre, length) per observer per disc (5 × 20 classes per disc; the joint cell is NOT
enumerated); with one observer it must reproduce dw-8ray's `appearance-fac` labels exactly (test);
`master_eval dw_extra_targets` gains `"observer_ablation/L-dw-8ray-obs5-20m": ("appearance-fac",)`.
Smoke before launch: 64-seq suite with the flags re-renders bit-exactly through both config paths, discs inside
the circle every frame, ray zones build on the 40-entry frames (~10 differing entries per case, all observers
involved). Storage: ~150 GB (378 GB free before). Monitors armed in the orchestrator session.

## Multi-observer discworld — implemented 2026-09-13 afternoon (superseded by the launch above)

Sevan's last control before writing: N observers on a ring about the frustum's depth midpoint,
discs confined to the circle tangent to every near/far plane. Landed as `pim/environments/discworld/observers.py`
+ `SimConfig.n_observers` / `region` (default-off, bit-identical; 5 tests) + three one-line dispatches
(renderer, position sampler, containment test) + generator / `sim_config_from` passthrough. Animations and
visibility statistics in `experiments/multi_observer/` (README). Awaiting Sevan's look at the renders before any
instance (`dw-8ray-obs5`?) is registered or generated.

## ✅ OVERNIGHT CHAIN 2026-09-12 20:00 → 2026-09-13 11:10 PT — units `dw_smooth_gen` (CPU) + `rescore_protocol` (GPU) — DONE

**Sevan's order (2026-09-12 evening):** profile H (power dome, (1 − u²)²) for the anti-aliased
instance; "kick off the big run now, observe all typical overnight protocol". Launched:

- **`dw_smooth_gen`** (`scripts/drivers/dw_smooth_gen.sh`, MemoryMax 40G, logs
  `logs/smooth_ablation/dw_smooth_gen/`): dw-smooth eval / edits / probe 120k / probe 250k (done
  19:56–19:59) → 20M corpus (started 19:59, ~1 h) → edit selection (1000 cases, ≥ 2 rays).
- **`rescore_protocol`** (`scripts/drivers/rescore_2026-09-12.sh`, MemoryMax 45G, logs
  `logs/rescore_2026-09-12/`): stage 0 waits for the gen unit (≤ 3 h) → 1 `train.py --smoke` on
  dw-smooth (`runs/_pipeline_smoke/dw-smooth-smoke`) → 2 master_eval (EVERY discworld run at
  `2026-09-12.2`, every Othello run at `2026-09-12.1`, ~9–11 h) → 3 Table 3 alignment refresh →
  4 Haufe editability (`table3_haufe_edit.json`) → 5 test_loss → 6 both table notebooks
  [ping RESCORE DONE, ~07:00–09:00] → 7 train `smooth_ablation/L-dw-smooth-20m` 780k (~8 h) →
  8 appearance-fac probes + random-init + observation floors → 9 master_eval (new run) →
  10 test_loss + tables [ping ALL DONE, ~16:30–18:30 2026-09-13].
- **Progress (2026-09-13 01:05 PT):** gen unit DONE 22:03 (corpus VERIFIED 20M / 0 dup / 24 ranges; selection
  1000/1000, every teleport changes ≥ 9 rays); smoke-train OK 22:04; master_eval 22:04 → 00:55 (2 h 51 min,
  all 33 runs at the new versions — far under the 9–11 h budget); alignment refresh, Haufe, test loss, tables
  DONE 00:58 (**RESCORE DONE pinged**). Stage 7 training started 00:58 → ETA ~09:00; ALL DONE ~10:30–11:00.
  Draft reading of the rescore: `scratch/2026-09-13-protocol-rescore.md` (discworld: nothing moves; Othello:
  symdiff lowers floors to ≈ −0.93…−0.98 and lifts standard Othello to +0.72…+0.83, ordering unchanged; Haufe
  helps only oth-adjacent and, slightly, the discworld regression blocks).
- Monitors armed in the orchestrator session: stage watcher (both driver logs + unit states, every
  60 s) and a 30-min heartbeat (stage, scored-file count, unit memory, disk, training metrics).
  `nvidia-smi` is broken (driver/library mismatch, GOTCHAS 2026-09-12); torch verified the GPU first.
- **Landed before launch (this commit):** `SimConfig.soft_shading="power"` + `soft_profile_power`
  in both renderer backends (`soft_render._profile`; default bit-identical; 4 new/extended tests),
  `sim_config_from` + `generate_dataset.py` passthrough, `bigcorpus` instance `dw-smooth`
  (seed blocks 200e9 / 225e9 / 1040e9 / 1050e9, forbidden everywhere), master_eval
  `dw_extra_targets` + `build_full_tables` long list gain the smooth run, `score_pending.sh`
  repointed at the renamed table notebooks, REGISTRY instance + run rows, `instance.json`.
- Smoke evidence: a 128-seq legacy suite with the power flags re-renders bit-exactly (float32)
  through BOTH config paths; lit runs carry ~33 distinct values (the dome); 13/13 soft-render
  tests pass; the driver passes `bash -n`; the gen unit's first three stages ran clean.

**DONE 11:10 PT (chain complete, unit ended with its marker; both monitors stopped).** Stage 7 training 00:58 → 09:35
(516 min; slowed 27 → 20 steps/s 03:23–05:00 while the OTHER session ran `inlp_othello.py` on the nodrop-20m
extension — not touched); stage 8 fac probes + floors 09:35 → 10:46; stage 9 scoring 10:46 → 11:10; tables rebuilt.
**Result: `L-dw-smooth-20m` predicts 18× better than dw-noiseless (val 5.9e-5 vs 1.06e-3) and edits WORSE** —
frustum PI +0.11 / 2.62 (noiseless +0.23 / 1.54), appearance-fac ND +0.38 (noiseless +0.61); decodability unchanged
in kind (MLP ≈ 1.0 on trained and random-init). `findings/smooth-ablation.md` (observed), REGISTRY run row, scratch
note completed. Smoke run stays in `runs/_pipeline_smoke/dw-smooth-smoke` (the smoke topic). Recorded in commit.
**Open after the chain:** waterfall panel for dw-smooth's frustum PI arm; decide whether dw-smooth enters the paper
shortlist (long list only now); findings banners for the OTHELLO numbers (symdiff headline; floors now ≈ −0.93…−0.98,
standard Othello best arms +0.72…+0.83) still quote the pre-protocol union numbers — sweep owed; Table 3 Haufe
columns filled (after-Haufe hurts categorical PI; helps oth-adjacent).
**(Original wrap-up plan follows.)** read `logs/rescore_2026-09-12/headline.txt` + the ALL DONE ping; compare
against the parked pre-rescore scores (`scores.pre-alignment-2026-09-12.json` etc.); update the
findings banners, REGISTRY run rows (numbers), `findings/probe-target-type.md` for dw-smooth;
move `runs/_pipeline_smoke/dw-smooth-smoke` artefacts as usual; commit; recap for Sevan.
**If a stage fails:** the ping carries the log tail; fix forward and relaunch under the same unit
name (`--collect`); stages 0–1 and the gen stages are idempotent (present files are skipped;
corpus shards resume via `_done_NNN`).


## oth-adjacent-flip chain on the WSL remote — DONE 2026-09-11 04:22 PT (`experiments/adjacent_flip_ablation/drivers/oth_adjacent_flip.sh`, unit `oth_adjacent_flip` on wsl-sevan)

**Outcome:** pilot → corpus → cases → 780k-step training (25.7 h on the RTX 4090) → master_eval
(+ oth-adjacent-flip floors) → tables, all clean; unit result success, 26.7 GB peak under the
40 GB cap; Windows hardware-error count flat at 40 throughout. Result: Bayes-optimal; skill
0.947 / 0.970 above the observation floor 0.888 / 0.914; ND +0.24 / fid 0.62, PI +0.17 / 0.85,
GS +0.03 / 2.62 vs unedited −0.70 — editability returns at a third of Othello's level.
`findings/adjacent-flip-ablation.md` (`observed`), REGISTRY run row, instance row updated.
Pulled to this box: run dir (2.2 GB) + `runs/_baselines/oth-adjacent-flip/` + the instance's
splits/cases (moved into layout v2 on arrival); `build_full_table` rebuilt with the new row.
Remote (`~/research/physically-implicit-modeling` on wsl-sevan) is at 268f6c6 on this branch
with its nbconvert outputs stashed; its `runs/_smoke/` rename is ledgered in its `runs/MOVES.md`.
Remote setup notes live in memory (`wsl-remote-host`, `wsl-remote-hardware-check`).
**Follow-up done 2026-09-11 (self-contained, `experiments/adjacent_flip_ablation/scripts/`):** true-edit-direction
alignment (least-aligned Othello model: 4× generic raw, 2.6× Haufe), Haufe-corrected edits (PI +0.29 / ND +0.33, fid ~0.5),
and the ceiling on ordinary counterfactuals: +0.655 / +0.679 / +0.697 (flip / adjacent / standard) with ND at 55% / 30% / 75%
of it. ⚠ A first pass reported a +0.14 ceiling and blamed the index — wrong (swap-built histories through a toothless
legal-mass filter), caught by Sevan, retracted the same day; GOTCHAS entry rewritten.
**Presence edits (2026-09-11 evening, `scripts/presence_edit.py`):** standard Othello +0.447 / 0.37 (reproduced); oth-adjacent and oth-adjacent-flip NOT presence-editable (guarded −0.12 / +0.05) — the enclosure-vs-local-legality difference is now the leading candidate for the gap.
**Masked probes (2026-09-11 late, `scripts/masked_probes.py`):** flipped-row colour probe reads recoloured tiles at 7% error and edits them WORSE than the canonical probe (ND −0.27 vs +0.08); parity-only probe edits best (+0.28); flip-bit probe 78% decodable, inert. Computed colour = decodable + causally inert at the last position (the discworld pattern inside Othello); the register theory now needs "writable" as a separate property.
**INLP (2026-09-12 early, `scripts/inlp_othello.py`):** copies of a tile's colour 30 / 50 / 85–232 (standard / flip / adjacent) = the editability ordering; writing 64 copies at once edits oth-adjacent to +0.47 / 0.47 where single copies do nothing; standard peaks at K=16 (+0.65); flip saturates at K=8 (+0.24). Vocabulary: materialised vs fused variable (compiler sense); materialisation follows fan-out.
**Dropout ablation (2026-09-11 evening):** `scripts/train.py` gained `--resume` (exact continuation from `ckpt/latest.pt`, incl. optimizer/RNG/history; extendable; token sources fast-forward the batch stream so a resumed run reproduces an uninterrupted one — `tests/test_training_resume.py`) and `--dropout`. `L-oth-adjacent-nodrop-390k` launching on wsl-sevan (unit `oth_adjacent_nodrop`; oth-adjacent data built there from scratch in layout v2): dropout 0, 390k steps (~13 h on the 4090), then INLP + K-copy edits vs the dropout run. `experiments/dropout_ablation/`.
**Dropout ablation DONE (2026-09-12 ~12:00):** `L-oth-adjacent-nodrop-390k` trained (12.5 h, same optimum); INLP copies per tile 283 / 227 / 187 / 161 / 150 / 148 / 146 / 162 vs 232 / 138 / 99 / 88 / 90 / 85 / 83 / 83 with dropout — MORE redundant without it; K ≤ 16 edits inert; best K-copy edit +0.45 / 0.37. The fused code is the rule's. Canonical master_eval scores on the remote pending at write time; run dir pulled to the lab box.
**Extension DONE (2026-09-13 03:21):** `dropout_ablation/L-oth-adjacent-nodrop-20m` = the no-dropout run continued 390k → 780k via `--resume` (interrupted once at 400k when a Microsoft Store WSL update shut the VM down; resumed exactly; the keepalive task is now a self-restarting loop). INLP copies per tile 272 / 210 / 158 / 132 / 122 / 119 / 118 / 137 vs 283 / 227 / 187 / 161 / 150 / 148 / 146 / 162 at 390k and 232 / 138 / 99 / 88 / 90 / 85 / 83 / 83 with dropout — the tail prunes 10–20 % with doubled training but stays 1.2–1.6× the dropout run's; K-copy edit curves unchanged (K ≤ 16 inert; +0.42 at pt 1, K=128); canonical PI −0.21 / ND +0.05 / GS inert, skill 0.973 / 0.978. The redundancy is the regime's steady state. Run dir + scores pulled to the lab box; `experiments/dropout_ablation/scripts/inlp_compare.py` builds the N-run comparison figures/tables from the score files. Written up in `findings/adjacent-flip-ablation.md` §Dropout ablation → Extension.
**Dropout 0.3 DONE (2026-09-13 17:46) and rescored on one protocol:** `dropout_ablation/L-oth-adjacent-drop03-390k` — same optimum (val 2.4354); at eval 2026-09-12.1 **PI +0.172 / fid 1.04 and ND +0.159 / 0.73 land inside the guard** (dropout 0.1: ND +0.107 / 1.02 only; dropout 0 at 390k and 780k: nothing). INLP copies at point 1 283 (dropout 0) → 232 (0.1) → 185 (0.3), plateau-then-cliff cascades at 0.3, K-copy window moving deeper (pt 1 → 2 → 3). Compression check (`experiments/dropout_ablation/scripts/covariance_dim.py`) REFUTES the stated prediction: participation ratio of the residual covariance RISES with dropout (44 → 81 → 84) — dropout decorrelates the stream and confines colour to a block of sufficient copies rather than compressing anything. All four arms now scored at eval 2026-09-12.1 (`canonical_table.py`, `inlp_compare.py`). Written up in `findings/adjacent-flip-ablation.md` §Dropout ablation → Dropout 0.3; REGISTRY rows.
**Dropout 0.7 DONE (2026-09-14 09:37, rescored 09:40):** `dropout_ablation/L-oth-adjacent-drop07-390k` — val 2.4403 (0.008 above the floor; eval-mode val stuck at 2.66–2.75 for 110k steps before dropping), skill 0.981 / 0.986, guarded PI −0.168 / ND +0.033 — editability FALLS BACK from the 0.3 peak (inverted U in dropout). INLP point-1 copies 169 (283 → 232 → 185 → 169 for 0 → 0.1 → 0.3 → 0.7) but wide redundant plateaus at every depth and colour computed later (pt-1 R² 0.72; K-copy window at pt 4). Five arms on one protocol: `experiments/dropout_ablation/scripts/canonical_table.py`, `inlp_compare.py` (`outputs/*_dropout_0_01_03_07.png`), `covariance_dim.py`. Written up in `findings/adjacent-flip-ablation.md` §Dropout ablation → Dropout 0.7; REGISTRY row. Both GPUs idle. Open: second seed at 0.3 (→ `replicated`), dropout sweep on standard Othello, 0.5 to locate the peak.
**Still open:** the recoloured-tile split
(only 2/42 clean cases have one — needs a different counterfactual construction), extended-α + landing sweep, a second seed.


## Overnight chain 2026-09-09 18:49 → 2026-09-10 04:47 PT — `scripts/drivers/probe_targets.sh`, unit `probe_targets`, logs `logs/probe_targets/`

**Outcome:** stages 1–5 and two of the three extra variants complete; every block and floor
persisted and scored; tables rebuilt. The unit was **OOM-killed at the 45 GB cap on the first
fit of `grid-32x16`** (512 cells, 1,536 logits — its (200k × 39 × 512) label arrays plus the
16 GB residual memmap's page cache exceeded the cap); nothing partial was written. To run it:
raise the cap (the box has 59 GB; the desktop has been taken down by OOM before — prefer
making `label_frames` lighter first) and `scripts/fit_probes.py --run … --target grid-32x16`
for both 8-ray runs, then `master_eval` + `build_full_table` (they add only what is missing).
Monitors: both exited on their own. Findings: `findings/probe-target-type.md` (complete tables).
⚠ At 04:48 PT a SECOND unit, `probe_targets_2` (`scripts/drivers/probe_targets_2.sh`, not
this session's), started fitting further grid variants on the 8-ray runs (`grid-6x5` first;
`build_full_table` cell [1] had been edited to order `grid-4x2 / 6x5 / 10x3 / 64x32`). Its
blocks will be absorbed by `master_eval` like the others; this session did not launch or
monitor it.

Sevan's order (2026-09-09 evening): (1) Othello read by a REGRESSION probe — `mine_signed`
(+1 mine / 0 / −1 theirs, 64 outputs; his decision: mine/theirs frame, not absolute colour)
on all three Othello runs + floors, ND included; (2) the dw-8ray APPEARANCE target — the
observation-exact partition: exactly 30 single-disc appearances (runs of lit rays), 90
logits — on `L-dw-8ray-20m` and `L-dw-8ray-tok-20m` + floors; (3) a resolution sweep run
unconditionally: `appearance-d2`, `appearance-d3`, `grid-16x8`, and if everything finishes
before 06:00 PT, `appearance-lat`, `grid-8x4`, `grid-32x16` (one probe set per model, no
floors). All land as extra probe-target blocks / rows in the main table.

**Interim results (all but `grid-32x16` landed by 04:45 PT; `findings/probe-target-type.md`
carries the tables):** Othello read by a REGRESSION probe is exactly as editable as by the
categorical one (PI +0.62 / ND +0.63 vs +0.61 / +0.62); **dw-8ray under the observation-exact
APPEARANCE target is editable at Othello's level — GS +0.61 / fid 0.39 (frame), +0.58 / 0.31
(token), PI and ND +0.43 with guards < 1, at every residual point, persisting over the
rollout** (regression on the same stream: GS −0.06, ND n/a, PI +0.28). The sweep is
unimodal at the observation-exact resolution: lat (15 cells) GS +0.42, exact (30) +0.61, d2
(60) +0.57, d3 (90) +0.51, grid-16x8 (128) +0.37 — and the 16 × 8 grid on dw-8ray reproduces
the noiseless grid's weak numbers, so the earlier grid control's failure was the TARGET's
misalignment with the frame, not the instance. Waterfall:
`runs/ray_ablation/L-dw-8ray-20m/figures/waterfall_edits_appearance.png`. Fig 3 (the sweep)
added to `build_full_table` as cell [8]; ⚠ two cosmetic defects to fix in the morning
(tick labels "30"/"32" collide; per-panel legends crowd the Edit Index panels — the fixed
cell is written in this session's transcript; the notebook grew past the reader's size
limit so it could not be re-written tonight). ⚠ Another session edited
`build_full_table` cell [1] (BASIS_ORDER gained grid-4x2/6x5/10x3/64x32, INSTANCE_ORDER
`oth-adjacent-flip`) and REGISTRY (an `oth-adjacent-flip` instance row) during the night —
not this chain's doing; left as found.

Stages: 1 master_eval (Othello mine_signed inline + floors, ~1.5 h) → 2 appearance probes
(2 models) → 3 floors → 4 master_eval + tables → 5 variants ×3 → 6 extra variants if before
06:00 PT. Monitors armed: stage watcher + 30-min heartbeat (this session). ntfy pings at each
stage. If the session is lost: `systemctl --user is-active probe_targets`,
`logs/probe_targets/driver.log`, then re-arm both monitors. Findings skeleton:
`findings/probe-target-type.md`. Code committed before launch (see `git log`).

**23:30 PT update.** Stages 1–4 DONE (Othello `mine_signed` on 4 runs + 8 training-curve
checkpoints; dw-8ray `appearance` on both models + floors; tables). Headline in
`findings/probe-target-type.md`: **dw-8ray IS editable through the observation-exact
categorical target** (frame model GS +0.61 / fid 0.39, PI +0.43 / 0.76, ND +0.43 / 0.91;
token model GS +0.58 / 0.31; regression target on the same models: GS negative everywhere);
by-step stays positive for 15 steps; every residual point lands; Othello `mine_signed`
reproduces the categorical PI/ND to a few hundredths. Chain 1 is in its variant sweep
(appearance-d2 → d3 → grid-16x8 → if before 06:00 PT: appearance-lat, grid-8x4, grid-32x16).
**Chain 2 queued** (`scripts/drivers/probe_targets_2.sh`, unit `probe_targets_2`, logs
`logs/probe_targets_2/`): waits for chain 1's `chain complete`, then Sevan's additions —
the MISALIGNED 30-cell controls `grid-6x5`, `grid-10x3` (same count as appearance, not the
runs of rays) and the gradient ends `grid-4x2`, `grid-64x32` — ~75 min each. Both chains are
systemd units: they do not depend on any agent session. The session died once (~22:30) and
the chain did not notice; monitors re-armed 23:25. ⚠ master_eval SETTINGS were extended with
the four new targets via nbformat (the Read/NotebookEdit tools refuse the notebook above the
size limit) — the one deliberate exception, flagged to Sevan.

**04:55 PT update.** Chain 1 scored appearance-d2 / -d3 / grid-16x8 / appearance-lat /
grid-8x4 (sweep so far: the observation-exact target is a PEAK — coarser (lat 15) edits
worse at equal decodability, finer loses decodability fast and editability gradually;
grid-8x4 at ~the same cell count edits at about half the appearance level → alignment, not
count) and was **OOM-killed at 45 GB on grid-32x16** (`fit_probe_stream` built 2 × 25 GB
int64 label matrices for the in-sample stats). Fixed (streamed counts, gated identical,
commit after ee556cd); chain 2 (`probe_targets_2`, started 04:48: grid-6x5, grid-10x3,
grid-4x2, grid-64x32) imports the fix fresh at each stage; **chain 3** (`probe_targets_3`,
`logs/probe_targets_3/`) is queued behind chain 2 to redo grid-32x16. Expected: chain 2
done ~10:00 PT, chain 3 ~11:30 PT. Then: fill `findings/probe-target-type.md` with the full
sweep table + a resolution figure, REGISTRY rows for the new targets' numbers, GOTCHAS entry
for the OOM.

**2026-09-15 — IM (inverse-map overwrite) becomes a CANONICAL EDITOR; ND dropped from the tables by default
(Sevan).** `pim/probes/inverse.py` (g: state → residual, the mirror of MLP-128, cached per point per basis;
`RetrievalBank` for IM-NN), `pim/editors/inverse.py` (overwrite h′ = g(s_post); delta and retrieval kept
non-default), IM / IM-NN arm functions for frame (`dwa.inverse_arms`), token (`tkb.inverse_arms`) and Othello
(`oa.inverse_arms`) models — one g per basis serves every block of a run. `master_eval` scores IM + IM-NN on every
block; an EDITOR-LEVEL fold-in appends them to already-scored runs (dated backup under `runs/<run>/scores_backup/`,
atomic write; a 3-run smoke left every existing arm byte-identical and reproduced the 2026-09-14 experiment: token
model IM +0.650 / 0.29 at pt 8 vs +0.65 / 0.29; no-flip −0.554 vs −0.553). Tables: `EDITORS = (PI, GS, IM)`,
`EDITORS_ALL` keeps ND / IM-NN in the frame; `T.set_basis(BASIS)` switches EVERY discworld number to one regression
basis (knob at the top of both table notebooks, default frustum; grid-defined targets unchanged; dw-8ray-obs5 falls
back to cartesian and is starred under frustum). Registry rows for INVERSE MAP and IM. Chain `score_im` launched
~11:30 to fold IM into all runs in both bases (ETA ~4 h); tables to be re-rendered when it completes.

**2026-09-14 19:47 → 21:18 — CARTESIAN blocks added to all 21 discworld runs (Sevan: validated cartesian scores
"in our back pocket" for the paper; tables and figures NOT switched).** Unit `score_cartesian` + watcher, overnight
protocol (smoke of the cartesian path, capped unit, staged driver, heartbeat, pings); 91 min because the pre-09-11
runs still had cached cartesian probes. `master_eval` `dw_bases` now reads `PIM_DW_BASES` (default frustum —
unchanged behaviour; applied with nbformat, the notebook exceeds the Read cap). Frustum vs cartesian canonical PI:
noiseless +0.23 / 1.54 vs +0.20 / 1.71; L-dw-20m +0.18 vs +0.16; blink +0.21 vs +0.17; smooth +0.11 vs +0.10; 8-ray
+0.26 / 0.99 vs +0.21 / 1.00; 5-ray +0.24 / 0.94 vs +0.14 / 0.85; token model +0.00 both; GS negative in both bases
everywhere. No conclusion moves; the coarse-ray runs lose the most, as the 2026-09-04 8-ray comparison predicted.
Cartesian floors added for dw-smooth (the one instance without them). Verify table `logs/score_cartesian/verify.txt`.

**2026-09-14 (evening) — INLP SWEEP on discworld, per-variable cascades (Sevan; new experiment
`experiments/inlp_sweep/`, 8 run × target cells).** Redundancy (dimensions removed per variable to mean R² < 0.4,
points 2–8): 5-ray appearance-fac 39–48 > 8-ray fac 29–34 ≈ 5-ray 26–32 > smooth 24–29 ≈ 8-ray 22–24 > 8-ray token
14–23 > noiseless-fac 7–8 ≈ noiseless 5–7 — fewer rays and coarser targets mean MORE copies. Bets: Sevan (8-ray
slightly less redundant; fac meaningfully less) loses all three; Claude right in direction, wrong in size. The
K-copy write (joint weighted-ridge step over every variable's first K copies, shrunk targets; K = 1 ≡ PI in z-space,
verified) beats canonical PI by +0.08–0.10 inside the guard on 8-ray (+0.34 / 0.93) and 5-ray (+0.31–0.34), matches
PI on the factorised targets, and moves nothing on noiseless, smooth, the token model or noiseless-fac. Redundancy
predicts editability in NEITHER direction across models (smooth: as redundant as 8-ray, least editable; Othello
ran the other way) — it is a symptom of code geometry. Sevan caught a wrong first write (independent per-variable
steps summed; cross-talk between the eight variables) — superseded, rescored from the saved cascades.
`findings/inlp-sweep.md`; figures `experiments/inlp_sweep/outputs/inlp_r2_by_iteration*.png`.

**2026-09-14 (late) — inverse probe on dw-blink by subset (Sevan): reappearance frames edit exactly like visible
frames.** Reappearance (edited object hidden through EF−1, visible at EF; 642 cases, staleness 1–12) delta
+0.69 / 0.39, overwrite +0.53 / 0.39; visible (no blink on either object through the 15-step rollout; 630 cases)
+0.67 / 0.43, +0.53 / 0.38; canonical PI re-searched on each +0.24 / 1.57 vs +0.21 / 1.56, GS ≤ 0 on both. Per-point
profiles coincide. Carried-through-blackout position and frame-readable position are equally writable by the
state-conditional mean. `findings/inverse-probe.md` "dw-blink by subset".

**2026-09-14 (late) — the adjacent failure put to Sevan's two tests; the positional/bypass hypothesis is
NOT supported, the generalisation reading stands.** (1) Last-tile cases (the just-placed disc recoloured — the
only tile whose square and colour both enter through the written position; adjacent 864 / standard 792):
adjacent +0.41 / 0.51 at point 1, +0.22 / 0.72 at point 2, destruction from 3 with landing ≥ 0.94 — the same
profile as the canonical cases; canonical editors re-searched on these cases: ND guarded +0.19 / 0.69, PI and
GS nothing. Standard edits the same tile at +0.86 / 0.25 (ND +0.72, PI +0.87, GS +0.83 inside the guard).
(2) Reconstruction control, overwrite with g(s_pre), no edit: adjacent's output preserved at points 0–5 (guard against
the pre-edit truth 1.2–2.2), standard's damaged at 0–3 and 6–8 (guard 5–16) and preserved only at 4–5 (1.8–2.0), where its edits work. My "the output consumes the 3% the board does not explain" claim is withdrawn
(Sevan: R² on reachable boards says nothing about g off-manifold). Adjacent's output IS a function of the board
on reachable boards; the failure is confined to unreachable ones. Open: why standard's board function extends
off-manifold and adjacent's does not (INLP redundancy; placement-parity features). `findings/inverse-probe.md`
"The two tests"; scores `experiments/inverse_probe/scores/othello_*_{lasttile,recon}.json`.

**2026-09-14 (evening) — INVERSE PROBE COMPLETE on twelve runs (batches 2–3: adjacent-flip, L-dw, smooth,
blink, 8-ray token model, obs5, no-flip, 5-ray).** Every discworld instance edits through the state → latent
write inside the guard: 5-ray +0.87 / 0.24, 8-ray +0.83 / 0.28, blink +0.67, smooth +0.67, noiseless +0.66,
8-ray-tok +0.66 (frame-set EI), L-dw +0.65, obs5 +0.61 (cartesian) — canonical PI ≤ +0.26 on all of them.
Othello under the fixed write follows the flipping toggle: standard +0.88, adjacent-flip +0.81 / 0.35
(canonical +0.35), adjacent +0.41 / 0.97 at point 1 only, no-flip nothing (control by the checkerboard
theorem — landing 0.00–0.03, the probe reads colour off occupancy; my "no-flip fails" prediction was right
for a reason that makes it uninformative). Retrieval edits on discworld (+0.30 to +0.69), never on Othello.
Sevan challenged the "lookup vs computed" account of the adjacent failure — discworld position is a lookup
too and edits — and it is withdrawn; the replacement hypothesis (is the fact available from context
positions other than the one written) and its test (write at every context position on adjacent) are
recorded in `findings/inverse-probe.md` reading 6, UNTESTED, held at Sevan's request until agreed.
`adjacency-ablation.md` reading 3 carries a dated addendum (discworld clause contradicted). Scores:
`experiments/inverse_probe/scores/*_mirror128.json`; table printer `scripts/table.py`.

**2026-09-14 — inverse probe on `L-oth-adjacent-20m` (Sevan): the state → latent write does NOT edit the
adjacency model** — overwrite +0.00 / 1.48, delta's only guard-passing arm +0.41 / 0.97 (pt 1), deeper points
destructive (guards 2–8), retrieval inert — although the board explains 97–98% of its residual (an input
lookup). Standard Othello edits at +0.88 with R² 0.8. The flipping toggle orders the inverse-map editor as it
orders the canonical ones: the environment effect survives the instrument change (finding, reading 5).

**2026-09-14 (afternoon) — INVERSE PROBE, all three runs, mirrored map (128 hidden, 200 epochs — Sevan): the
state → latent write EDITS DISCWORLD's regression state.** Best arms (EI / guard): Othello delta +0.88 / 0.23
(canonical best +0.83); noiseless delta +0.66 / 0.46, overwrite +0.60 / 0.34 (canonical PI +0.23 / 1.54, GS
−0.08); 8-ray delta +0.84 / 0.28, overwrite +0.73 / 0.27, and RETRIEVAL (10 nearest training states, no
training) +0.63 / 0.43 (canonical PI +0.26 / 0.99). State explains only 0.29–0.57 of the residual on discworld
yet the conditional mean edits — the alignment result closed from the other side; the probe-derived write
was the wrong write, not the representation. Sevan's bet loses in both environments. Wide map (1024/40)
dropped from the write-up (files kept). `findings/inverse-probe.md` rewritten with the 4-form table +
canonical editors per run. Next candidates: 5-ray / blink / token model / adjacent Othello under the same
write; a waterfall for the discworld delta arm.

**2026-09-14 — INVERSE PROBE on Othello (Sevan's idea; contained, `experiments/inverse_probe/`): a learned
state → latent MLP EDITS BETTER than every canonical editor.** g: one-hot board → residual (1024 hidden, 40
epochs, 943k rows); delta write h + α(g(s_post) − g(s_pre)) at point 5, α 1.5: **+0.917 / 0.20** vs PI +0.82 /
0.30, ND +0.72 / 0.31, GS +0.83 / 0.28; plain overwrite +0.90 / 0.22; nearest-neighbour retrieval ≤ +0.15;
mean-residual control inert. Sevan's bet ("does not edit or poorly") loses on Othello. Works where the probes
work (pts 4–6, g R² ≥ 0.85); early points land the read-out without moving the output; the last point
destroys. Unit ran 4 min (after one failed launch: a 4 GB broadcast in the NN search hit the NVML-mismatch
OOM assert — fixed as a one-hot matmul). `findings/inverse-probe.md`. Discworld (8-ray, factorised state)
held for Sevan's go.

**2026-09-14 — scoring audit (Sevan: are the remote dropout runs comparable; is everything on disk current?).**
Every non-quarantined run is at the current version (discworld 2026-09-12.2, Othello 2026-09-12.1) on the
standardised bench (Othello: 1000 flips at a 20-move prefix from the instance's own edits games; discworld:
the instance's ≥2-differing-rays selection, 1000 cases, dims "all"), floors complete for every (instance,
arch, target) at b4. The four `dropout_ablation` runs (imported from the WSL box) were SCORED HERE (local
probe_dir, current version, mine_signed present, checkpoints + metrics intact) → directly comparable.
⚠ Their baseline is the CANONICAL dropout: Transformer-L's class default is 0.1 (no `dropout` key in the
older configs), so the sweep reads 0.0 (nodrop) / 0.1 (L-oth-adjacent-20m) / 0.3 / 0.7 — `nodrop-20m` is a
regime change, not a replicate of the canonical run (PI −0.05 vs +0.18). Gaps: (1) `ray_ablation/R-dw-8ray-20m`
was re-quarantined to `_R-dw-8ray-20m` with no ledger entry, and its 2026-09-12.2 scores hold only the
frustum block — the recurrent `appearance-fac` result in `probe-target-type.md` / `recurrent-l.md` is at the
OLD protocol (Sevan's call: un-quarantine + rescore fac, 2 min with cached probes, or mark old-protocol);
(2) `_architecture_gate/R-dw-*` at 2026-09-01.4 (quarantined; recurrent finding's numbers are old-protocol);
(3) regression blocks recorded `bench_selection: None` while the selection WAS applied — fixed in
`bench_arrays` (record only; fills on the next rescoring); (4) three never-fitted `grid-64x32` requests removed.

**2026-09-14 — replicate machinery made generic (Sevan: "make those changes; guard by default, override
available").** (1) `master_eval`: a replicate inherits its parent's extra targets (`extra_targets_of`); the
three hand entries removed. (2) `pim.figures.tables.pool_replicates`: pooled per (parent, basis) at a
matched budget (±10%, nearest checkpoint joins; other budgets listed as "not pooled"); `collect(pool_budgets=
True)` / `POOL_BUDGETS` in both table notebooks overrides (Table 5 says "mixed budgets"); 4 tests.
(3) `scripts/drivers/replicate.sh <topic/parent> <seed> <steps> [targets]` — one replicate end to end;
`layout_checkpoint_replicate.py … nearest:<steps>`. (4) `probe_seeds_othello.py`. WORKFLOW/REGISTRY updated.

**12:15 PT 2026-09-12 — seed-variance pilot CLOSED (stopped after the canonical run's probe seeds, Sevan's call;
my ETA had slipped 08:00 → 18:00 because the probe refits ran at all 9 points and each seed re-collects its
stack — recorded).** Probe seeds: skill 0.9587 ± 0.0006 (20 seeds, regression) / 0.4319 ± 0.0007 (6, fac); same best
point every seed; ND +0.505 ± 0.000, PI ±0.001; the across-points profile reproduces seed for seed (SD of the
per-seed spread 0.006 / 0.0002). Training seed: Edit Index SD 0.005–0.017; guard SD up to 0.15. Tables rebuilt by
hand: ± cells in Tables 1–2 for the noiseless row (n = 3), Table 4a/4b. `findings/seed-variance.md` complete.
Nothing running.

**09:15 PT 2026-09-12 — seed-variance: run-seed half in.** Two 390k replicates + the seed-0 422k checkpoint,
pooled n = 3: decodability SD ≤ 0.005; Edit Index SD 0.005–0.017 on every editor / target (fac ND +0.624 ±
0.017, fac GS +0.322 ± 0.014); the GUARD is where the spread lives (GS fid 0.68–1.09, SD 0.15). Half budget
vs full: same size as the seed spread. `findings/seed-variance.md` drafted; probe-seed stage running
(3 runs × ~2 h) → ~15:00 PT.

**23:00 PT 2026-09-11 — seed-variance pilot launched (Sevan: "retrain L-dw-noiseless-20m twice at half the steps,
refit a ton of linear probes, report fluctuation; replicates as ± not rows").** Decisions: pool seed 0 (as the
run's own 421,875-step checkpoint, matched budget) with the two 390k replicates, n = 3, SD; probe refits at
every point, each seed's own best point reported alongside the seed-0 point, plus the spread across points
per seed. Code: `fit_probes(seed=)` (key already carried the seed; seed 0 keys unchanged), `train.py
--replicate-of` → `config.json["replicate"]`, `build_full_table` folds replicates into ± cells of Tables 1/2 +
Table 4 (run-seed and probe-seed spread), `experiments/seed_variance/scripts/{probe_seeds,layout_checkpoint_replicate}.py`,
driver `scripts/drivers/seed_variance.sh`. Smoked (train --smoke with the flag; probe_seeds --smoke; layout;
tables with zero replicates); 248 tests. Unit `seed_variance` started 22:59: train seed 1 (~4 h) → seed 2
(~4 h) → score + fac ×3 (~2 h) → probe seeds ×3 (~2 h each: 20 regression + 6 factorised, the latter 16 min
a seed) → tables. Honest ETA **~15:00 PT 2026-09-12** (the factorised probe seeds dominate; trim to 4 seeds
if wanted — decide at stage E). Watchers + heartbeat armed; pings per stage. Convention written into
`harness/WORKFLOW.md`, REGISTRY, `experiments/seed_variance/README.md`.

**15:55 PT — loss-matched control for the GRU result: the 8-ray transformer at step 32k (val 0.00600 vs the
GRU's best 0.00595) under `appearance-fac`: ND +0.57 / 0.73, GS +0.46 / 0.52, PI +0.33 / 1.09 (guarded +0.32)
— edits at its fully-trained level, where the GRU at the same loss does not (+0.13 / +0.07). Architecture, not
exposure. `runs/training_curve/L-dw-8ray-20m_s032000` (training-curve view; out of the main tables by design);
unit `l8ray_s32k_fac`, 43 min. Recorded in `probe-target-type.md` and `recurrent-l.md`. Nothing running.

**15:40 PT — Othello centre tiles (Sevan's quick check; `experiments/othello_centre_tiles/`).** Li's bench
never intervenes on d4/e4/d5/e5; 98 cases synthesised with the bench recipe. L-oth-20m, cached probes:
PI +0.64 / 0.20, ND +0.65 / 0.20, GS +0.57 / 0.27 vs whole bench +0.61 / +0.62 / +0.65 — as editable as
the rest of the board (linear editors' best point 5 rather than 4).

**15:00 PT — `appearance-fac` on the RECURRENT 8-ray run (R-dw-8ray-20m, un-quarantined by Sevan 12:45;
unit `r8ray_fac`, 59 min): the GRU does NOT edit under the read-out that unlocks the transformer.**
skill 0.94 / 0.94 (identical to L-dw-8ray-20m; floors alike), PI +0.25 / 0.73 at point 0 only (input
projection; every carried hidden destructive), ND +0.13 / 0.65, GS +0.07 / 0.86 vs the transformer's
+0.41 / +0.50 / +0.46. The categorical-target result is architecture-dependent; recomputation is back
in play as a help (addendum in `recurrent-l.md`). Finding, registry, Table 2c updated. GPU idle;
nothing queued.

**12:30 PT — cartesian basis dropped (Sevan, 10:20; done once no unit was executing the notebooks).**
`master_eval` `dw_bases = ("frustum",)` (no EVAL_VERSION bump — the scorer only adds missing blocks);
`build_full_table` `HIDDEN_BASES = ("cartesian",)` hides the old blocks in Tables 1–3 and 1b–1e; Table 3
re-lettered (3a = frustum). Executed clean. REGISTRY note added. Nothing running; GPU idle.

**12:20 PT — joint-cell `appearance` on dw-5ray scored (12:14): PI +0.54 / 0.78, ND +0.47 / 0.85, GS +0.64 / 0.40
vs 8-ray's +0.43 / +0.43 / +0.61 — the quantisation bump replicates on the second read-out; GS at Othello's
level.** All units done; GPU idle. Now: dropping cartesian from `dw_bases` and hiding it in the tables.

**11:15 PT 2026-09-11 — dw-5ray chain COMPLETE (11:09): Sevan's bet holds — a bump on every editor.**
`L-dw-5ray-20m` best val 0.00694; canonical regression rows inert as everywhere (PI +0.30 / 1.12,
GS −0.11); **`appearance-fac`: skill 0.93 / 0.93, PI +0.49 / 0.68, ND +0.56 / 0.90, GS +0.58 / 0.47** vs
8-ray +0.41 / +0.50 / +0.46 — the best-edited discworld row on PI and GS, decodability unchanged.
Along 128 → 8 → 5 rays PI and GS rise monotonically under the factorised target; ND flat. Finding,
registry, Table 2c, sweep figure updated. Joint-cell `appearance` on 5-ray running (unit
`dw_5ray_appearance`, ~12:10). Then: drop cartesian from `dw_bases` + hide it in the tables (Sevan,
10:20 — agreed; deferred until no unit is executing the notebooks). Note for the record: dw-5ray has
no filtered edit selection (identical-frame teleports in the canonical bench; unedited −0.86).

**06:40 PT 2026-09-11 — training on course; NVML broken by an unattended driver upgrade (harmless).**
dw-5ray corpus VERIFIED 01:41 (20M seq, seeds 160e9…179.5e9, obs.f32 16 GB); training started 01:41,
~1,650 steps/min, step 480k at 06:34, best val 0.00694 → ends ~09:45, scoring ~10:15, fac ~11:15.
At 06:15 unattended-upgrades moved the NVIDIA userspace to 595.91 (kernel module 595.84): `nvidia-smi`
fails until reboot; CUDA still initialises in fresh processes (tested), so stages D/E are unaffected.
GOTCHAS entry added. Reboot at Sevan's convenience after the chain completes.

**23:35 PT — token-model `appearance-fac` scored (63 min): skill 0.94 / 0.94, PI +0.24 / 0.54, ND +0.30 / 0.50,
GS +0.40 / 0.43 — below its joint-cell row (+0.26 / +0.44 / +0.58): factorisation costs the token model,
whose output is a softmax over whole frames (one token per run). Finding, registry, Table 2c updated.
`dw_5ray` chain released from its wait stage; generation running.

**22:45 PT — overnight: token-model `appearance-fac` running; the dw-5ray instance + training chain
queued behind it (Sevan's call, 22:25: "run the 70 min L-dw-8ray-tok-20m evaluation under appearance
factorized, then kick off a heavier quantization run — 5 rays, radius 1.0; standard overnight protocol").**
Sevan's ranking correction on record: dw-8ray is the best-edited discworld run under `-fac` (all three
editors land, GS highest), not noiseless. Unit `appearance_fac_tok` started 22:30 (~70 min). Unit
`dw_5ray` (`scripts/drivers/dw_5ray.sh`): stage A waits for `appearance_fac_tok` (4 h timeout), B
generates dw-5ray (7 cast / 5 kept rays, radius 1.0 — 4 kept rays leaves positions lighting no ray;
14 appearance cells, fac 9 + 4 classes; fresh seed block 160e9 / 185e9 / 1020e9 / 1030e9 registered in
`bigcorpus.INSTANCES`; ~2 h), C trains 780k steps (~8 h), D scores canonically (~30 min), E fits
`appearance-fac` + floors and scores (~1 h). ETA ~11:30 PT 2026-09-11. Generator path smoked on
`_smoke_5ray` (layout v2 --role; moved to `datasets/archive/`). Watchers: stage + heartbeat on both
units; ntfy pings at every stage. Both units are systemd, independent of this session.

**20:15 PT — `appearance-fac` on L-dw-blink-20m scored (unit 72 min): ND +0.53 / 0.94, GS +0.33 / 0.95,
PI +0.02 / 2.44; skill 0.46 / 0.68 (rand 0.22 / 0.45, obs 0.07 / 0.39).** Blink's first positive edit —
and NOT above noiseless under the same target (+0.63 / 0.78): carrying position does not make it more
editable; the "integration pressure" account of a discworld toggle is not supported. ND ranking under
`-fac`: noiseless +0.63 > blink +0.53 > 8-ray +0.50. Also today: hidden-frame floors on blink
(`experiments/blink_ablation/scripts/hidden_frame_floors.py`): trained MLP 0.96–0.98 at ten frames
hidden vs observation 0.46–0.78 / random-init 0.58–0.74 — position IS computed and carried there.
GPU idle; tonight's training instance still Sevan's call (5-ray quantisation was my recommendation
before this result; see the discussion — the result weakens the case for integration pressure).

**18:25 PT — `appearance-fac` on L-dw-noiseless-20m scored (unit 72 min): ND +0.63 / 0.78 — OTHELLO'S
LEVEL, on the 128-ray run whose regression rows never edited.** 233 + 29 classes, 1,048 logits;
skill 0.43 / 0.79 vs rand-init 0.30 / 0.59 and obs-right 0.12 / 0.52 (the first discworld target
where training's margin is clear); GS +0.35 / 0.68; PI +0.01 / 1.95 (pseudo-inverse ill-conditioned
at 1,048 rows — pipeline unchanged, noted). Finding, registry, Table 2c, sweep figure updated.
Sevan's next-step discussion (17:45): a new instance tonight (5-ray quantisation vs anti-aliasing);
my recommendation — first `appearance-fac` on `L-dw-blink-20m` (no training; the "computed
variable" test with the working target), then the 5-ray instance. Awaiting the call.

**17:30 PT — alignment of the factorised probe with the true edit direction (Sevan's contained
test; `experiments/edit_direction_alignment/scripts/fac_probe_alignment.py`, no canonical change).**
66/192 valid counterfactuals on dw-8ray. fac rows hold 0.070 of Δ vs 0.025 generic / 0.006 random
rank (2.8× / 12×); ND direction cos² 0.056 vs 0.002 random (28×), joint cell 0.035, regression rows
0.003 (= generic). Haufe: fac 0.034 vs 0.039 random-pattern floor — washed to chance, as in Result 5.
`findings/edit-direction-alignment.md` Result 6. Next (approved): `appearance-fac` on
`L-dw-noiseless-20m`.

**17:00 PT — `appearance-fac` scored (unit 65 min): the categorical gain SURVIVES factorisation.**
skill 0.94 / 0.94 (above the joint cell's 0.89 / 0.90; rand-init 0.92 / 0.94), PI +0.41 / 0.71
(joint: +0.43 / 0.76), **ND +0.50 / 0.86** (best discworld ND; joint +0.43 / 0.91), GS +0.46 / 0.53
(joint +0.61 / 0.39 — GS loses the joint move, still ≫ regression −0.06 and the grid band +0.35).
So alignment + categorical read-out + per-tile class-swap edit are what make discworld edit like
Othello, and `<partition>-fac` is the cheap form that keeps all three → the one to run on
noiseless (2 × (233 + lengths) classes instead of 2,889 × 3). Finding, registry updated; Table 2c
has the row. Open with Sevan: the Othello converse (a fuller regression set-up).

**15:55 PT — the FACTORISED categorical target `appearance-fac` launched (Sevan's pick after the
snapped result).** Per object one softmax per factor of the appearance partition — run centre (15
classes) and run length (5) — 4 tiles × 20 classes instead of 30 cells × 3; scales to 128 rays as
2 × (233 + n_len) against 2,889 × 3. Edits are per-tile class swaps (PI: swap old ↔ new at each
moved tile; ND: summed row contrast; GS: CE toward the new labels); the bench is the partition's
cell-changing 192 cases. `grid_target.FactorisedTarget` (`<partition>-fac`, works for product
grids too), `Bench.moves`, `arms.categorical_direction`, token bench mirrored; 3 tests (247 pass);
real-model smoke through every editor. Unit **`appearance_fac`** (`scripts/drivers/probe_target_fit.sh`
— model probes, random-init + observation floors, then `score_pending.sh`), started 15:53,
GRID_PROBE_RECIPE, ~1 h. Commit ae99a1f. Sevan's next idea on record: the CONVERSE on Othello
(classification → regression already done as `mine_signed`; a fuller regression set-up needs
thought — "not just swapping the optimiser").

**15:40 PT — `pos@appearance` scored (unit done in 15 min): the snapped regression target IS the
regression row.** skill 0.96 / 0.99 (gap ≤ 0.001, not overfit), PI +0.34 / 0.92 at α 100 (the
canonical row: +0.28 / 0.90 at α 100), GS −0.13 / 0.84 (−0.06 / 0.82); the categorical
appearance row stays alone at +0.43 / +0.61. Reading in `findings/probe-target-type.md`: the
categorical gain is the read-out + logit-swap edit, not the target's frame-alignment; the
regression side of this axis is closed. Table 2c shows the row under its reference. Committed.

**15:20 PT — layout v2 verified from my side; Table 2c; the SNAPPED regression target launched.**
The other agent's migration (commits ebf6a99, f3a404f, 2de841d): 240 tests green, `--verify`
gate passed per the spec, probe caches re-keyed logically (not orphaned), my sweep-figure script
updated by them. Nothing blocks. **Part 1 (Sevan):** `build_full_table` now splits every extra
probe target out of Tables 1–3 (`DF_X`; `CANONICAL_BASES`) into **Table 2c — the probe-target
sweep**: per run, the canonical reference row, then the extra targets by cell count; Fig 3 and
the 3c… floor tables read `DF_X`. Executed clean. **Part 2 (Sevan):** `pos@<partition>` — the
gridification kept as 4-D position regression (`grid_target.SnappedTarget`, cell centroids in
the frustum basis with a medoid fallback, `bench_arrays` regression branch on the partition's
filtered cases, `_targets` snap, canonical 30k recipe, floors inline; 4 new tests, 244 pass).
`master_eval` SETTINGS: `pos@appearance` on `L-dw-8ray-20m`. Running as unit
**`snapped_appearance`** (`scripts/drivers/score_pending.sh`, `logs/snapped_appearance/`),
started 15:19 — fits LIN + MLP inline, PI/GS (ND recorded, not reported), the dw-8ray floors
for the target, then the tables. ~1 h expected. Watcher on. Prediction on record: LIN skill
below the regression row's 0.95 (piecewise-constant target), MLP ≈ 0.9; if PI/GS rise from
+0.28 / −0.06 toward the categorical appearance row's +0.43 / +0.61, alignment is the story.
Notebook rule exception taken again (nbformat; the two notebooks exceed the reader's cap).

**14:20 PT — chain 5 STOPPED at Sevan's call; the overnight probe-target programme is closed.**
grid-64x32 on noiseless needed ~6 h (20 min per probe point at 6,144 logits; LIN at the majority
at points 0–2); Sevan: "that's crazy, let's cancel it now". `systemctl --user stop probe_targets_5`;
no partial probe or score artefact (the cache writes only after a full fit). **Nothing is running
or queued; GPU free; the tree is clear for the other agent's cache-key migration** (pinged).
Noiseless reading written into `findings/probe-target-type.md`: decodability survives finer
grids (MLP 0.44 at 512 cells vs 0.04 on 8-ray), editability does not follow (ND / GS flat across
32 → 512, PI degrading); `appearance-lat` the outlier (ND +0.51 / 0.79, linearly unreadable).
Re-queue grid-64x32 (noiseless) or the full 2,889-cell appearance partition only if wanted.
Sevan is moving to a new idea next.

**13:45 PT — chain 5 on its last variant, grid-64x32; ETA revised.** appearance-lat (233
cells: LIN 0.04 / MLP 0.63, PI +0.04 / 1.68, **ND +0.51 / 0.79**, GS +0.30 / 1.08) and
grid-32x16 (0.20 / 0.44, +0.17 / 1.13, +0.34 / 0.92, +0.29 / 0.90) are in the tables and the
finding (commits 932089f, 4549455); figure re-rendered. grid-64x32 started 13:19: ~20 min per
probe point at 6,144 logits, so ~5–6 h for LIN + MLP → done **~19:30 PT**, not 15:30. Unit
memory 36 GiB steady under the 45 GiB cap (label tensor 7.8 M × 2,048), GPU 18.6 GiB. LIN at
the majority at point 0 (as on 8-ray); the MLP is what the noiseless question needs (it read
grid-32x16 at 0.44 where 8-ray was at 0.04), so the stage runs on. So far the noiseless
gradient: decodability survives finer grids, editability does not follow — ND +0.34…+0.37 and
GS +0.27…+0.30 flat across 32 → 512 cells, PI degrading; appearance-lat is the one outlier
(ND +0.51, linearly unreadable).

**11:35 PT — chain 4 OOM-killed at appearance-lat; chain 5 relaunched with the fix.**
Chain 4 scored grid-8x4 (noiseless: skill 0.71 / 0.89, PI +0.27 / 1.05, ND +0.34 / 0.92, GS
+0.27 / 1.11 — in the tables) and was killed by systemd-oomd 50 s into appearance-lat:
`AppearanceTarget.cell_of` labelled all 15.6 M probe positions in one shot (~60 GB at 128
rays). Fixed: chunked labelling + snapping of unseen grazing runs (commit after 5c35ba7;
gated: 2 GB peak, 16 s, all 233 / 2,889 cells realised; 18 tests). **Chain 5**
(`probe_targets_5`, `scripts/drivers/probe_targets_5.sh`, `logs/probe_targets_5/`) runs the
three remaining variants — appearance-lat, grid-32x16, grid-64x32 — started 11:35; watcher +
heartbeat re-armed on it. Expected done ~15:00 PT (grid-64x32 the long one, ~1.5 h).

**10:56 PT — chain 3 complete; chain 4 running.** grid-32x16 on both 8-ray models is in the
tables (frame: skill 0.01 / 0.04, PI +0.14 / 1.22, ND +0.31 / 0.97, GS +0.42 / 0.68; tokens:
0.01 / 0.05, +0.02, +0.10, +0.18 / 0.58) — the product-grid GS band holds with the probe at
the majority, i.e. editability there is insensitive to decodability (findings §4). Chain 4
(`probe_targets_4`) started 10:57 with grid-8x4 on the noiseless run; heartbeat + watcher on.

**09:10 PT — records caught up while chain 3 runs.** Consolidated 8-ray sweep tables and
reading in `findings/probe-target-type.md` (commit bec3776), the OOM in GOTCHAS, REGISTRY rows,
and the sweep figure `experiments/probe_targets/outputs/probe_target_sweep.png` via
`pim.figures.sweep_figure` (commit 3649a83; re-run `experiments/probe_targets/scripts/sweep_figure.py`
when chains 3/4 land — the noiseless row fills in by itself). Chain 3 started 08:57, ~2.5 min
per probe point on grid-32x16; expected done ~10:30 PT, chain 4 then ~10:30 → ~15:00 PT.
Sevan asked (09:00) why 32x16 is redone (chain 1 OOM, never scored) and how much is left
(chain 3: 32x16 × 2 models; chain 4: 4 noiseless variants) — answered.

**08:58 PT — chain 2's last variant STOPPED by me.** grid-64x32 on dw-8ray: the linear
probe sat exactly at the majority error (skill 0.00) at points 0–2 and the variant would have
taken ~8 h for both models (≈15 min per point at 6,144 logits). The far end of the gradient
is thereby measured (decodability collapses); the editors on a probe that reads nothing are
noise, not a result. Stopped via `systemctl --user stop probe_targets_2` so chain 3 (grid-32x16
redo) and chain 4 (the noiseless sweep Sevan asked for) run today instead of tonight. Chain 2's
scored variants: grid-6x5, grid-10x3, grid-4x2 (all in the tables). Re-queue grid-64x32 on
dw-8ray only if Sevan wants the number in the table.

**05:40 PT — chain 4 queued** (Sevan: "run similar grid sweeps on L-dw-noiseless-20m; I
predict it benefits from a finer grid than 8-ray"): `scripts/drivers/probe_targets_4.sh`,
unit `probe_targets_4`, behind chain 3 — grid-8x4, appearance-lat (233 cells), grid-32x16,
grid-64x32 on the noiseless run (~40 min each, one model, no floors); the full appearance
partition there (2,889 cells) is left out (22 GB label tensor). Expected done ~14:30 PT.
Heartbeat now follows whichever of chains 2/3 is active; each chain has its own watcher.

## 2026-09-12 (evening) — ONE EDIT PROTOCOL built; tables rebuilt as a module; overnight rescore QUEUED (not launched)

Sevan's spec, all done except what is queued: **benches** — Othello 1000 single-tile flips at a fixed 20-move
prefix per instance from a dedicated `edits` index range (old cases in `_unused/`); discworld 1000-case
selections (≥ 2 differing rays) on all five instances (`scripts/make_edit_selection.py`); **protocol** — full-state
writes only, shared α grids + GS layers, `EVAL_VERSION_BY_ENV` dw 2026-09-12.2 / oth 2026-09-12.1; **Othello
headline = symmetric difference** (union kept). **Tables**: `pim/figures/tables.py` + `notebooks/build_paper_tables.ipynb`
(shortlist) / `build_full_tables.ipynb` (long list, renamed from build_full_table) — Table 1 = decodability with floors
(right-aligned obs, listed archs only), 1b–1e per-cell optima, 2 (+2b arms as an image), 2c gridified only coarse→fine,
**Table 3 alignment** (worker: `experiments/edit_direction_alignment/scores/table3_alignment.json`, all 16 rows;
PI-after-Haufe columns blank), **Table 4 test loss vs Bayes floor** (`experiments/bayes_floor`), Table 5 seed
replicates; Fig 1 (paper only, L-oth-20m + L-dw-20m, zigzag fixed), Fig 2 capacity; old Fig 3 and tables 3d/3j/3l/3m/3n
dropped. Both notebooks execute clean on the CURRENT scores (old benches, new headline) — they refresh after the rescore.
**QUEUED — `scripts/drivers/rescore_2026-09-12.sh`** (launch: see its header; ~9–11 h GPU): 5-ray fac floors →
master_eval both envs → Table 3 alignment refresh → Haufe editability (`table3_haufe_edit.py`, smoked OK) → test loss →
both notebooks. Anti-aliasing instance: NOT started (Sevan holding). GPU is free right now.

## 2026-09-12 — METRICS COMMITTED; discworld RESCORED under the pre-dynamics target — nothing moved (branch `new_metrics`)

Sevan's decisions: (1) the Edit Index keeps the SIMULATOR references (Bayes-optimal given the exact state; the
model-referenced v2 is dropped — pilot: ≤ 0.03 change, magnitude entanglement); (2) **the discworld write target
is the PRE-dynamics state** — `pos[EF] − v·dt` for the edited object, the current state for the rest — because the
probe reads the state of the LAST consumed frame and the model's output is one dynamics step ahead (GOTCHAS
2026-09-12; commit b7f741f; test `tests/test_bench_target_alignment.py`). Probes and Othello untouched.
**DONE:** all 19 discworld runs rescored (`EVAL_VERSION_BY_ENV['discworld'] = 2026-09-12.1`, unit
`rescore_dw_alignment`, 65 min), tables rebuilt. **Result: every block within ≤ 0.06 of its pre-fix value
(median 0.005), floors identical, no conclusion changed** — the offset only matters near the ceiling and no editor
is near it (`scratch/2026-09-12-alignment-rescore.md`). Pre-fix scores parked as
`scores.pre-alignment-2026-09-12.json`; every finding quoting discworld editor numbers carries a one-line banner.
The pre-run prediction ("128-ray numbers up modestly") was wrong. NEXT: write-up; the no-flip realisable-edit
result (pilot) as a separate legal-vs-illegal finding if wanted; ceiling row per run (cheap) if wanted.

## 2026-09-11 (night) — PILOT: the model-referenced Edit Index on paired counterfactuals (branch `new_metrics`)

`research/scratch/2026-09-11-edit-index-v2-pilot.md`; `experiments/edit_index_v2_pilot/`. Nine
conditions × 48 cases × one canonical arm per editor. **v2 ≈ v1 (±0.03) on every frame / Othello
row** — the reference change alone moves nothing on these near-Bayes models (token model +0.04..+0.10).
What moves is the EDIT TYPE: realisable (paired-history) edits are 2-tile occupancy changes on the
no-flip / adjacency instances and ~5-tile changes on standard Othello. **⭐ oth-noflip becomes
EDITABLE (ND +0.58, guard 0.27) on realisable edits** — the canonical bench asked it for an
impossible one-tile recolouring; the adjacency instances stay inert on the same edits. Standard
Othello: GS +0.68 holds, PI/ND collapse at their single-flip arms (not re-tuned). Discworld unchanged.
Sevan sleeps on the metric decision (v2 vs p_A + S(B)); next: magnitude-stratified rerun, α re-tune
for multi-tile Othello edits, then the `edits/v2/` spec.

## Where the work is (2026-09-09)

**Done this session (Sevan's ask): the grid target lives in `pim`, not in an experiment.**
`pim.environments.discworld.grid_target` (`"grid-16x8"`), a categorical branch in
`discworld/bench.py` + `discworld/arms.py`, ONE class-swap helper shared with Othello
(`pim.editors.pinv.swap_class_logits`), kind-agnostic Probe Skill / tripwire
(`pim.metrics.probe_skill_from_stats`). No probe was fitted: the 18 grid probes + the two
floors were RE-KEYED from the experiment into `runs/noise_ablation/L-dw-noiseless-20m/probes/`
and `runs/_baselines/dw-noiseless/`, and `master_eval` ADDS a missing probe-target block to a
current `scores.json` instead of rescoring (`dw_extra_targets`, `require_cached` — the scorer
can never start a grid fit). `build_full_table`: one row per (run, probe target); the grid
row reports ND; Table 3c. Canonical numbers = the experiment's to 4 decimals (ND +0.373 /
0.91, GS +0.289 / 0.87, PI +0.126 / 1.58). ⛔ **Sevan's constraint stands: no new probe
training.** The Othello REGRESSION-target ablation he intends is NOT implemented (no probes;
the arm wiring would be dead code until they exist) — when it runs, it is a `kind="regression"`
block on the Othello side, keyed like the grid one.

**Also done (2026-09-09, later): Transformer-L is ONE class with two interface parameters.**
`input` linear | embedding and `head` regression | categorical (`output_kind` logits | raw),
the two old names kept as presets (`TransformerL`, `TransformerLTokens`); parameter names and
registration order preserved, so every probe-cache fingerprint is unchanged. Gated on all 28
canonical L checkpoints: identical fingerprints, state-dict keys, residual streams and head
outputs (bit-identical). `scripts/train.py` now takes the interface as a run parameter
(`--repr frames|tokens`, `--objective ce|mse_onehot`, either environment; Othello has only a
token corpus so `frames` is rejected there), and `master_eval` picks the scorer by what the
model EMITS (frame → ray-zone rollout; distribution → set-based step-0), not by environment.
No retraining, no probe fits, no rescoring; 221 tests. Decided in discussion the same day:
the mixed discworld cells (linear-in/categorical-out, embedding-in/regression-out), the
discworld MSE-on-one-hot cell and any Othello continuous-input cell are NOT pursued — an
embedding on a one-hot IS a linear layer, so "Othello under discworld's input" is degenerate;
the control logic (each environment invariant to the interface swap meaningful for it) is
complete with `L-dw-8ray-tok-20m` and `L-oth-20m-mse`. Flagged as the real gaps: a second
SEED of the flagship pair (the record is `observed`, one seed per cell), bootstrap intervals
on the Edit Index, and — as a research call — a discrete-state discworld instance (lattice
positions, jump dynamics) as the one environment-side test of the constitutive hypothesis.

**The claim the paper can make.** Othello is editable, discworld is not, under an
identical pipeline (probes, editors, α grids, Edit Index + guard). Everything we varied to
explain the gap came back negative, so the honest framing is **"decodability is not
editability"** in a matched cross-environment design — NOT "the conditions for
editability". (Framing discussion 2026-09-07; retreat ratified after blink failed.)

**Runs since 09-01** — each has a findings note; numbers there, not here.

| run | what it varied | editable? | note |
|---|---|---|---|
| `objective_ablation/L-oth-20m-mse` | CE → MSE-on-one-hot head | YES +0.68 | `othello-mse-head.md` |
| `interface_ablation/L-dw-8ray-tok-20m` | frames → tokens, Othello arch | no (+0.006) | `interface-ablation.md` |
| `flip_ablation/L-oth-noflip-20m` | Othello without recolouring | no (~0) — but colour is causally IRRELEVANT there (checkerboard theorem), so it is a "decodable ≠ used" control, not a rules ablation | `flip-ablation.md` |
| `blink_ablation/L-dw-blink-20m` | position must be CARRIED (blackouts) | no (+0.27 reappearance ≈ +0.22 visible) | `blink-ablation.md` |
| `adjacency_ablation/L-oth-adjacent-20m` | colour USED without enclosure geometry | **no** (PI −0.05 / ND +0.12 / GS 0), read-out lands 100% | `adjacency-ablation.md` |

**Excluded as explanations:** objective, interface, observation resolution, noise,
decodability, causal use, carried state, and target type (categorical vs continuous —
`grid-target-control.md`: a 128-cell 3-way discworld target gives ND +0.37 / GS +0.29,
one-frame and reverting, vs Othello's +0.63). What remains is constitutive: a discrete
combinatorial state read through a categorical head vs continuous geometry through a
regression head.

**The geometry line** (`edit-direction-alignment.md`, `experiments/edit_direction_alignment/`).
With oracle counterfactual states, Δ = h_cf − h, the fraction of Δ inside the probe's
read-out subspace tracks editability: oth-uniform 26×/30× its generic baseline,
oth-adjacent 8×, every discworld variant 0.5–1.2× (at chance). Two results worth carrying:
patching discworld's last-position residual with the FULL Δ produces the edit (+0.94), so
the residual IS load-bearing and position's code is high-rank/nonlinear; and the canonical
Othello editors reach 91% of the true-counterfactual ceiling, which validates the metric.
⚠ Alignment relative to the GENERIC baseline predicts editability; ABSOLUTE alignment does
not (Haufe correction raises absolute overlap everywhere and reorders nothing).
⛔ Scope: that analysis modifies the INSTRUMENT — see `RESEARCH.md` "The independent
variable is the ENVIRONMENT, not the editor" before extending it.

**Housekeeping done 2026-09-09.** dw-8ray's edit bench was silently degenerate — 20% of
its teleports render an identical frame and 22% move one ray of eight — so both 8-ray runs
now score a stored filtered case list (`datasets/discworld/dw-8ray/edits_selection.json`,
≥2 differing rays, 192/192 scoreable, shared by the ray-zone and token benches so the
interface pair stays matched). Conclusions unchanged; PI's guard on `L-dw-8ray-20m`
improved from 1.11 to 0.90, i.e. non-destructive. Old numbers parked as
`scores.pre-selection-2026-09-09.json` in each run dir (`runs/MOVES.md`).

**Open decisions.**
- (2026-09-10) **Dataset layout v2 — DONE** (`research/specs/DATASET_LAYOUT_SPEC.md`,
  `scripts/migrate_datasets.py`; verify PASSED, 240 tests green). Every `datasets/` path now comes
  from `pim/environments/layout.py`: `train/ probe/ eval/ edits/v1/ (edits/v2 reserved) tokens/
  _unused/` + `layout.json`; probe caches re-keyed logically (372 blobs, bytes unchanged, 5
  duplicates parked in `_superseded/`); producers stamp new instances v2 at birth. Ledgers in
  `datasets/MOVES.md`, `runs/MOVES.md`; GOTCHAS entry 2026-09-10. Nothing rescored. NEXT (Sevan's
  order): the model-referenced Edit Index + paired-counterfactual bench (`edits/v2/`) — design still
  under discussion (legal vs illegal edits; references p_A / S(B); the ceiling row); spec not yet
  written; then full re-evaluation when the GPU is free.
- (2026-09-09, Sevan) Re-reference the Edit Index and guard to the MODEL'S OWN predictions —
  unedited prediction and prediction on the true counterfactual history — instead of GT, so the
  floor is −1 by construction and +1 is the model's own counterfactual. Claude's read: the right
  object, but it demands a counterfactual per bench case (dw ≈110/192 in-frustum; Othello ≈5%
  of flips are reachable boards), so the bench must be rebuilt from counterfactual PAIRS, or the
  GT index kept as headline with the true-counterfactual ceiling reported beside it. Undecided.
- Table 1b per-component point-selection rule (options given 2026-09-05, undecided).
- Untrack the four `runs/*/config.json|scores.json` files on the remote? (`runs/` is
  gitignored; these predate it and were force-added.)
- Rewrite history to drop ~315 MB of old smoke-run checkpoints under
  `logs/archive/_smoke_runs/` (needs force-push across ~20 branches).

**Repo state.** `main` = `origin/main` = 484428a (housekeeping_big merged). 196 tests pass.
The nbstripout git filter now runs with `-W ignore` (it was flooding the terminal with
nbformat MissingIDFieldWarning on every git command).

_Previous update: 2026-09-01 — THE GUARD unified: one fidelity definition, one polarity, both environments_

_2026-09-01 (later) — **The guard is now one metric everywhere** (EVAL_VERSION 2026-09-01.1,
all five runs rescored):_

    fidelity = RMSE(edited prediction, edited-world GT) / RMSE(unsteered prediction, same GT)
               evaluated at the EDIT STEP ONLY.   > 1 = degraded, not steered.

Three changes, all Sevan's calls:

1. **Step 0, not the rollout.** An activation edit touches step 0 alone; later steps are
   recomputed from a window holding the *unedited* history plus one edited frame, so
   averaging 15 of them dilutes exactly what the guard is for. Measured: the step-0 form
   flags discworld PI at **1.63 / 1.69** where the rollout form managed 1.11 / 1.16, and
   ND at **2.23 / 2.82** against 1.14 / 1.12. `gt_traj_rmse` stays in the scorecard, so
   the rollout ratio — *can an edited frame survive re-entry into unedited context?* — is
   one division away under its own name, never as this guard.
2. **Othello gets one at last** (`move_fidelity_ratio`), RMSE over **all 64 squares**
   against uniform-over-legal-post. RMSE not Li error: Li error is a top-N *set* mismatch,
   integer-valued, blind to probability magnitude and compressive at the good end (it
   renders GS-mine vs PI as 0.013 vs 0.041 — a 3× spread on a handful of misordered
   squares — where RMSE gives the honest 0.207 vs 0.243). All-64 not union-support,
   because a guard restricted to the edit's own zone cannot see collateral damage.
   RMSE is also the distance `edit_index_legal` is already built from, so the two
   environments now share one construction rather than a shared name.
3. **`li_error_vs_pre` is retired from the guard column** — it is a DIAGNOSTIC (one half
   of the pair the Edit Index already uses) and its "higher is better" holds only up to
   the pre→post separation, 2.763. It stays in `scores.json` as the anchor to Li et al.'s
   published units.

**The table now reads on one axis** (EI / fid, ↓ good):

| run | env | unedited | PI | ND | GS | GS-mine |
|---|---|---|---|---|---|---|
| L-oth-20m | othello | −0.713 | +0.610 · **0.24** | +0.447 · **0.34** | +0.471 · **0.38** | +0.646 · **0.21** |
| L-dw-20m | discworld | −0.700 | +0.175 · **1.63** | −0.038 · **2.23** | −0.195 · 0.94 | — |
| L-dw-noiseless-20m | discworld | −0.924 | +0.218 · **1.69** | −0.037 · **2.82** | −0.114 · 0.97 | — |

Every Othello editor moves the prediction 3–5× CLOSER to the post-edit world than doing
nothing; every positive-EI discworld editor moves it 1.6–2.8× FURTHER. That contrast is
the finding, and it is now a single number rather than two metrics with opposite
polarity sharing a column.

⚠ **Never quote an Edit Index without the guard.** The index is *relative* (which world
is the output nearer) so a wrecked output still scores mildly positive — discworld PI,
EI +0.22 at fidelity 1.69. The guard is *absolute* and is the only thing that says so.

_Previous update: 2026-09-01 (early) — noise ablation done: removing ALL noise does NOT make discworld editable_

_2026-09-01 — **Two results, both from the new canonical pipeline, both overnight.**_

**1. GS-mine: a wrong negative retired.** Li-style gradient steering "failing" through
mine/theirs probes on Othello (EI −0.0014, on record since 08-24) was a **target-frame
bug** — the arm fed mine/theirs probes the benchmark's *absolute-colour* labels. Isolated
at identical point/α, varying only the frame and the probe set: mine targets **+0.6451**,
absolute **−0.0534**, and the OLD and NEW probes agree to four decimals in both frames
(probe quality was never involved; the old sweep did include α 0.05 at all nine points,
so not a grid artefact either). **GS-mine +0.6459 is now the strongest Othello editor**,
Li error **0.036** — below Li et al.'s published best intervention of 0.12 — guards clean.
All four best Othello arms now read mine/theirs probes: Nanda's frame wins for editing as
well as decoding. `grad_steer_arm` takes an explicit `target_labels`, and every
`scores.json` records `probe_sources`. → `scratch/2026-08-31-gs-mine-frame-mismatch.md`

**2. Noise ablation — the headline, and it is negative.**
`noise_ablation/L-dw-noiseless-20m` = dw-pn04 with `obs_noise_std` AND
`position_noise_std` both **0.0**, everything else byte-identical (Transformer-L
25,371,776 params, 20M sequences, 780k steps, matched recipe; 8.03 h vs 8.04 h).

| | dw-pn04 | dw-noiseless | L-oth-20m |
|---|---|---|---|
| best val | 0.022873 | **0.001063** (21.5× better) | 2.02798 |
| unedited EI | −0.700 | **−0.924** | −0.713 |
| PI | +0.175 · fid 1.11 | **+0.218 · fid 1.16** | **+0.610** ✓ |
| ND | −0.038 | −0.037 | +0.447 / +0.622 ✓ |
| GS | −0.195 | −0.114 | +0.471 / **+0.646** ✓ |

Every editor lands where it did with noise; PI's +0.218 is again **destructive**
(fidelity 1.16, collateral 0.513 vs 0.118 unedited = 4.3×), not an edit. **A model 21.5×
better at its own objective is no more editable** — prediction quality is not the gate.

Two side findings worth more than the headline: (a) the unedited EI floor moves −0.700 →
**−0.924**, confirming the ±0.82 compression noted 08-25 was an artefact of scoring a
noise-trained model against clean renders — the editors now have MORE headroom and still
do not use it; (b) **noise was making the code more linear**: LIN skill 0.944 → 0.872
while MLP *rises* 0.977 → 0.979 (velocity worst: o1·vy LIN 0.526 → 0.173, MLP 0.651 →
0.542) — same information, less linearly accessible without noise.

**Where this leaves the paper.** Three "artefact" explanations for the discworld negative
are now closed: not data scale (20M), not architecture (Li et al.'s own minGPT), not
noise (both channels off). The live structural differences vs Othello are **continuous vs
discrete state**, **regression vs classification**, and **the observation being a
projection** (occlusion, perspective) rather than full information.
→ `scratch/2026-09-01-noise-ablation.md`

**Open:** the floor for a noiseless world is not zero (a 128-ray quantised render limits
how precisely the initial state is inferable), so part of the 0.00106 is irreducible —
cheap CPU work, not yet run. Also pending: whether to drop `state` probes from the Othello
grid now that no editor needs them (72 → ~29 fits), and Sevan's call on committing to
sequence-split only.

_Previous update: 2026-08-31 (evening) — THE HOUSECLEANING: canonical core built, old tree deleted, both canonical runs scored_

_2026-08-31 — **The housecleaning.** The project re-centred on editability and the repo was
rebuilt around a small canonical core so every number is reviewable line by line. Everything
pre-existing is recoverable at the git tag **`pre-cleanup-2026-08`**; nothing was deleted from
`runs/`, `logs/`, `outputs/`, or `datasets/` (moves only, ledgered in each tree's `MOVES.md`)._

- **The canonical core is `pim/`** — environments (discworld + vendored othello, each instance
  packaged with its data + `instance.json`), models (Transformer-S/L × regression/token heads —
  the four old minGPT bridges collapsed to one), probes (LIN + **MLP-128** canonical + nullspace
  cascade, fingerprint-keyed caches), editors (**PI z-space+y-affine** / ND / GS workhorses +
  nullspace + two oracle editors), metrics (Probe Skill; the two Edit Index constructions get
  distinct names), training (ONE loop, two objectives; defaults = the matched BIG20M recipe).
  Index: **`research/REGISTRY.md`** (replaces METRICS_AND_EDITORS.md). Every port was
  equivalence-gated on the real canonical checkpoints before the old copy was deleted —
  bit-identical forwards, bit-identical probe fits, editor arms reproducing stored numbers to
  full precision (PI +0.6104 to 4 decimals; ND −0.0380; GS −0.0014; gates .9925/.9980).
- **The y-affine bug** (see `scratch/2026-08-31-pi-y-affine-bug.md`): every pre-08-31 discworld
  PI number solved standardised-y read-outs against raw-unit targets (Othello was structurally
  immune). Fixing it ~tripled the best EI (+0.05→+0.18) **and sharpened the negative**: the α=1
  write that provably lands the read-out leaves the generation at the unedited floor. Canonical
  PI is `"zspace"`; `"legacy"` reproduces old numbers and is never quoted as PI. The landing
  check now goes through `probe.forward` itself, so the bug class is unrepresentable.
- **Canonical scoring**: `notebooks/master_eval.ipynb` scans `runs/**` (excluding `archive/` and
  `_`-topics), scores every run identically — no metric math in notebooks — and writes
  `scores.json` + the fitted probes into **the run's own dir** (`probes/`; re-tests never
  refit). `build_full_table.ipynb` renders Table 1 decodability (FIRST, deliberately),
  Table 1b discworld per-component decodability, Table 2 editability with guards.
- **First master table** (`initial_othello_comparison`, the only two runs passing the
  canonical-runs rule): decodability ≈ **0.94–0.98 on both sides**, tripwire clean — so
  decodability is NOT what separates the environments. Editability: **Othello editable by every
  workhorse editor** (PI **+0.610**, ND **+0.622** single-point / +0.447 plain, GS **+0.471**
  through state probes = Li §4.1 verbatim; li-vs-pre guards all ≥1.8) vs **discworld by none**
  (best PI +0.175 with fidelity 1.11 = destructive; ND/GS negative). Two documented method
  deltas vs old records: ND is now swept per-point (beats the every-layer form +0.62 vs +0.37),
  and GS steers through *state* probes (works, +0.47) where the old mine-target arm failed.
- **Runs**: `runs/<topic>/<run>/` with config.json + commit_sha + probes/ + scores.json;
  W16, the S rungs and the L90 pair are archived (non-canonical data/recipes). Fresh canonical
  Transformer-S runs on both instances are the first post-cleanup trainings to schedule.
- **Pending** (unchanged, now cheaper on the clean base): canonical S runs, the fixed-noise 900k
  DW rung, frustum baselines + the depth-coordinate question, optional Othello LR anneal.

_Previous update: 2026-08-31 (recording the 08-24/25 discworld-at-scale work; see the 08-25 section first)_

_2026-08-22 (evening) — **the environment is what flips editability, and it
replicates.** Li et al.'s architecture, ~900k sequences, same optimiser, at both 4 and 14 epochs:
Othello reaches Edit Index **+0.24 / +0.23** with every guard clean (Li error 2.76 → **0.43**,
legal mass → **0.973**); discworld under identical conditions reaches −0.11 / −0.18 and fails its
guards. Architecture, editor, probe implementation, probe data, data volume and training length are
all excluded. ⚠ Exactly ONE editor works — Nanda's `target − current` on the mine/theirs basis;
the PI-injection claim from the 4-epoch pass is withdrawn._

## 2026-08-25 — Discworld at 20M: at the Bayes floor, and still not editable

Full write-up **`research/scratch/2026-08-25-discworld-at-scale.md`** — read it before acting on
any of this. Thread `notebooks/experiments/editability/discworld_scale/`; runs
`runs/discworld_scale/`. (Recorded 08-31; the run finished 08-25.)

### The headline

`BIG20M_discworld_L` — Transformer L (25,371,776 params) on **20,000,000 discworld sequences** at
`position_noise_std=0.04`, 780,000 steps = 11.09 epochs, every optimiser setting matched to the
Othello 20M run. **Best val 0.022873 @ step 660,000**, 8.04 h, 22 checkpoints.

**It sits 3.16% above a state-omniscient oracle and is still not editable.**

| bound (400 seqs x 39 frames, scored exactly as val_loss is) | MSE | RMSE |
|---|---|---|
| knows `clean_{t+1}` exactly — obs-noise floor | 0.018866 | 0.1374 |
| knows true `(pos, vel)` at t — **strict lower bound on Bayes** | 0.022171 | 0.1489 |
| the model | 0.022873 | 0.1512 |

82.5% of the loss is observation noise, 14.4% process noise, **3.1% everything else**. Bayes risk
brackets to [0.022171, 0.022873], so **further training can buy at most 0.000702 MSE (1.5% RMSE)**.
The curve agrees: plateaued at **step 50,000 of 780,000**; the last 610,000 steps bought 0.000105
with a train-val gap of ~0 throughout. Othello was still improving at step 775,000.

### Editability — unedited baseline EI -0.6998

| editor | cart `pos` | cart `full` | inv_y `pos` | inv_y `full` |
|---|---|---|---|---|
| PI injection | +0.0475 | +0.0522 | **+0.0874** | **+0.0854** |
| Nanda addition | -0.0380 | -0.0915 | -0.0172 | -0.0513 |
| MLP grad steering | -0.2043 | -0.1895 | *(linear-only)* | — |

⚠ **PI is destructive, not a weak success.** Best arm at **alpha 175, the top of the sweep**;
target RMSE **1.22x worse**, **collateral 5.4x worse**, edit-frame 1.43x worse — only ghost
improves, which any destruction achieves. EI rises monotonically with collateral across the alpha
sweep: the output moves away from the unedited world without moving toward the edited one, so
`d_u ~ d_e` and the index drifts to ~0. **EI near zero is the ambiguity point, not partial
success.** `fidelity_ratio` cannot see this (0.993 cartesian; **1.09-1.35 in frustum, i.e. WORSE
than doing nothing**) — always read target/ghost/collateral beside it.

Linear probes read position at mean R^2 **0.9461 (cart) / 0.9716 (inv_y)**; velocity 0.6878 /
0.6998. So "undertrained" and "no world model" are both dead as explanations. What blocks editing
is the **representation's geometry** or the **editors**.

### Three traps recorded in the write-up

1. ⛔ **Never quote the "overall" probe R^2.** `othello_probe._r2` pools over all dims, so it is
   variance-weighted, and position holds **99.9%** of the variance. Cartesian's unweighted mean is
   0.817 against the pooled 0.9440; within frustum, `u` outweighs `1/y` by **120x**, so `inv_y`'s
   0.9836 is nearly blind to the depth coordinate. **Quote per-dimension numbers.** Fitting is fine
   (`fit_probe` divides by `y_std`) — only the summary statistic is skewed.
2. ⚠ **The frustum result is UNCONTROLLED.** No random-init or observation-space baseline exists in
   that basis; the cartesian ones are on `W16`, and both control scripts have zero `basis`
   references. `u = x/(k*y)` is essentially the ray index, so the gain may be "easier function of
   the raw observation" rather than anything about the model. **~15 min to fix; blocking.**
3. ⚠ **The Edit Index scale is really +0.82 to -0.80, not +/-1** — the scorecard scores against the
   CLEAN render while the model is trained on the NOISY one, and clipped noise puts the optimal
   background at **+0.0798** across **74% of rays**. No conclusion to date is affected.

### Not done — a power outage killed the chain at the analysis stage

Training had already finished and every checkpoint survived.

1. **The 900k rung** `L90_discworld_pn04` (`runs/discworld_scale/chain.sh` stage 5,
   `--limit 900000 --steps 316406`), ~3.2 h + analysis. Never started.
2. MLP probes / grad steering in frustum (~8 min).
3. The frustum baselines in trap 2 above (blocking).
4. **The frustum depth coordinate was never settled** — five candidates, and
   `2026-08-23-frustum-basis.md`, cited by both `editability-scaling-sweep.md` and `frustum.py`,
   was never written. `inv_y` was used because it is the module default and what the derivation
   argues for, **not** because it won a comparison. Linear-only makes all five ~30 min.

### Machinery (202 tests green)

`discworld_scale/{corpus,train}.py`, `runs/discworld_scale/{gen,chain,hb}.sh`. Corpus
`datasets/20_dwscale_20m` (410 GB, 20M seqs, seeds 10,000,000..19,510,499,999, **0 duplicates,
disjoint from dset4 and dset17**); probe corpus `datasets/21_dwscale_probe` (120k, seeds 9e11) —
dset 4's test split holds only 10,000, far short of what keeps an MLP probe honest.

⛔ **The generator retries at a DIFFERENT seed** (`pim/simulator/dataset.py:111`,
`seed + attempt * 1_000_000`, up to 300 attempts), so any shard wider than 1M samples can silently
duplicate sequences. Shards are 500k with a 500M stride, making collisions structurally impossible;
`corpus.verify()` asserts it. Measured retry rate at these settings: **0.0000%**.

⛔ `train.py`'s `ckpt_base=0` used to spin forever (`s *= 2` never advances from 0) — no output,
GPU idle, indistinguishable from a stalled loader. Guarded. A ">230x thread-thrashing slowdown"
claimed while debugging that was **wrong**; it is retracted in the file's comments, and capped vs
uncapped thread pools are indistinguishable.

## 2026-08-24 — Overnight: the 20M Othello cell, and what it says about the sweep plan

**Run `runs/scaling/BIG20M_othello_L`** — Transformer L (25.3M), 20M unique Othello games,
780,000 steps at constant lr 1e-3 after 2,000 warmup, log-spaced checkpoints. Launched 18:51 on
08-23. **COMPLETE** 14:10 on 08-24, 19.4 h. Best val **2.02798** at step 775,000, excess over Bayes **+0.01878**, final train-val gap **+0.0005** (no overfitting). 11 checkpoints, 1.1 GB. Driver `runs/ours_on_othello/
bignight.sh`; heartbeat `runs/ours_on_othello/hb.sh`.

### The finding: the 900k Othello model was memorising, and my "saturation" numbers were turnarounds

Full write-up in **`research/scratch/2026-08-24-saturation-is-overfitting.md`**; figure
`runs/scaling/loss_regime.png` (regenerate with `notebooks/experiments/editability/scaling/
loss_regime_figure.py`).

Same architecture, same environment, same optimiser — only the pool size differs:

| | 900k games (`L90_theirs_othello`) | 20M games (running) |
|---|---|---|
| best val | 2.0881 @ step 58,000 | **2.02798 @ 775,000 (final)** |
| excess over Bayes 2.0092 | +0.0789 | **+0.01878 — 4.2x closer** |
| final train loss | **1.9204 — BELOW the floor** | 2.02755 — above it |
| train–val gap | +0.2585 and climbing | **+0.0005 — flat, never overfit** |

A train loss below `E[log |legal|]` is only reachable by memorising which legal move was drawn.
**The 900k run is partly a lookup table over its pool.** Its val turns around at step 58,000 and
gives back +0.0908.

⚠ **The sweep's compute budget rested on a broken premise, and it was mine.** "Saturation ~6k
steps at 90k games, ~58k at 900k → 20M saturates near 1–1.5M" extrapolated *overfitting
turnarounds*, not convergence. The 20M cell has no turnaround. `directions/
editability-scaling-sweep.md` is annotated at the top; do not trust its cost tables.

Practically: **"train the 20M cell to saturation" is not well-posed.** Excess follows roughly
`step^(-0.5)`, so each halving of excess costs ~4× the steps. A stopping point is a declared
tolerance or budget, not a discovered fact — **an open decision for Sevan.**

Second open decision: the curve now bends slightly *above* its own early power law (exponent
drifted −0.526 → −0.465), most likely the **constant LR** settling at its noise floor. The repo's
own reference is that annealing alone bought 0.0120 on the 14-epoch run. **Appending a short
anneal after step 780k** would preserve every property constant LR was chosen for (all the
constant-LR checkpoints are already on disk). Not done; awaiting a call.

### Why this bears on editability

`L90_theirs_othello`'s `best_model.pt` **is** the step-58,000 checkpoint, so the **+0.241 Edit
Index in the 08-22 table was measured inside the memorising regime.** This does not invalidate the
environment-flip result — both arms were matched on architecture, epochs and pool size, so the
regime issue applies equally — but the Othello arm was never tested at its best. Ten checkpoints
spanning a 512× range of training now exist for exactly that test.

⚠ Checkpoints (powers of 2 from 1,000) and val passes (multiples of 5,000) are on **incommensurate
grids**; no checkpoint has an exact val. Compute a per-checkpoint val during the sweep rather than
reusing the nearest-neighbour numbers.

### Machinery added overnight (202 tests passing, ruff clean)

| what | where | guards |
|---|---|---|
| `MLP ≥ linear` tripwire | `pim/extractors/standard.py` (the canonical estimator, so every call site) | the 08-22 probe-starvation failure. Branches on the in-sample score to name **memorising** vs **under-trained** — opposite fixes. `mlp_r2_insample` is NaN when skipped, never the held-out value in disguise. |
| provenance-verified probe cache | `editability.fit_probes` → `runs/othello_arch/probe_cache/` | the 08-21 bug where a random-init control was served the trained model's probes. Key includes the **weights**; a hit is verified against stored provenance and raises on mismatch; writes are atomic. |
| **Probe Skill** | `pim/eval/probe_skill.py` | one axis for regression *and* classification probes. Regression branch is **exactly** R² against the train mean (asserted to 1e-12), so no existing number changes. Trivial predictor comes from **train**; majority is **per output dimension**. Registry row added. |

Tests: `tests/test_probe_cache.py`, `tests/test_probe_skill.py`, and four tripwire tests in
`tests/test_standard_probes.py`.

### Editability on the finished 20M model (2026-08-24, later)

`research/scratch/2026-08-24-othello-editability-20m.md`. Target `mine`, 20k probe games.
**PI injection +0.6104** (pt 4), **Nanda target−current +0.3746**, Nanda addition +0.2403,
grad steering −0.0007. Unedited −0.7126. Guards clean on the top three (Li 2.763 → 0.096–0.194
with Li-pre staying 2.49–2.87; legal 0.857 → 0.99). Against the 900k model's +0.241, on identical
architecture and environment — the model is the only difference.

⚠ **PI injection is back and is now the strongest editor**, having been withdrawn on 08-22 when its
4-epoch +0.138 did not survive to 14 epochs.

The MLP probe story resolved in two parts: MLP-512 at the old 6,000-game default was **memorising**
(0.98 rows/param, held-out worse than linear), and fixing capacity (MLP-128, 13.04 rows/param)
collapses the in-sample gap ~10× — but it *still* sits 0.04–0.07 pp under linear at mid-depth,
which cannot be memorisation. **mine/theirs is linearly decodable; the MLP adds nothing.** So
grad steering's −0.0007 is **not** a probe artifact — re-run through clean MLP-128 probes gives
**−0.0014**, identical within noise, guards failing the same way (Li 2.763 → 5.656, legal → 0.747).
Gradient steering genuinely fails on Othello mine/theirs while the linear-probe editors reach +0.37
and +0.61.

⚠ Fixed on the way: `envctrl_eval.py` shared one probe-cache dir across all checkpoints with no
model in the key — running it on BIG20M would have loaded L90's probes. Now per-fingerprint.

## 2026-08-22 — The environment control, and the confound ladder finishing

Thread `notebooks/experiments/editability/othello_arch/`; brief
`directions/our-architecture-on-othello.md`; scratch `2026-08-21-ours-on-othello.md` Results 4–6;
`findings/editability.md` (`observed`, ⭐-candidate).

### The headline

Their architecture (25,312,768 params, 8 blocks, `d_model` 512), ~900k sequences, 4 epochs, same
optimiser and schedule — **environment as the only variable**:

| epochs | environment | unedited EI | **best EI** | editor · target | Li ↓ | legal mass |
|---|---|---|---|---|---|---|
| 4 | **Othello** | −0.482 | **+0.241** ✓ | Nanda t−c · `mine` | 2.915 → **0.969** | 0.824 → 0.923 |
| 14 | **Othello** | −0.591 | **+0.231** ✓ | Nanda t−c · `mine` | 2.763 → **0.432** | 0.849 → **0.973** |
| 4 | **discworld** | −0.699 | −0.113 ⚠ | Nanda · `pos` | — | fidelity 1.042 |
| 14 | **discworld** | −0.689 | −0.182 ⚠ | grad steering · `full` | — | fidelity 1.005 |

Stable across 3.5× training; both arms improved as models and neither saturated. On Othello the
edit **sharpens** while the index saturates (Li 0.969 → 0.432) — report Li error and legal mass
beside the index, never the index alone.

⚠ **One editor, one basis.** Nanda's `target − current` on **mine/theirs**. The same editor on
absolute colour is catastrophic (Li 15.8, legal 0.091). **PI injection read +0.138 at 4 epochs and
−0.013 at 14 — withdrawn.** Gradient steering fails on both targets, so its −0.010 was *not* the
target artifact I suspected.

### What was excluded on the way

| confound | ruled out | how |
|---|---|---|
| probe implementation | 2026-08-20 | our code reproduces Li's intervention on Li's model |
| probe training data | 2026-08-21 | saturates 140× short of theirs |
| our editor | 2026-08-21 | the pseudoinverse is the **best** editor on their model, at a single mid-depth point |
| data volume | 2026-08-22 | 222× on the Othello ladder moves absolute editability +0.059 → +0.098 |
| architecture + volume + epochs | 2026-08-22 | the table above |

### ⚠ Corrections made today, both caught by Sevan

1. **I led with "gain over own null" for several messages.** It rises 3.8× across the Othello
   ladder while *absolute* post-edit Edit Index is flat (+0.059 → +0.098) — the whole rise is the
   null falling as the model becomes a better predictor of the *unedited* world. Report the null
   and the absolute value together, never the gain alone.
2. **The Othello row's three editors did not share a probe target** (Nanda/PI on `mine`, gradient
   steering on `state`). Faithful to the two papers, but it makes the within-row editor comparison
   uninterpretable. Both evaluators now sweep target as an explicit axis; re-runs queued.

### Also today

- **`ours_on_othello` full-ladder notebook: complete**, 15 cells, 0 errors, 3 figures, `results.json`.
- **Discworld 10× data ladder:** `S0c_90k` 0.02112 → `S1_900k` **0.02034** at fixed 95,100 steps —
  10× data buys 3.7%.
- **`directions/discworld-at-scale.md` marked superseded**: its trigger did not fire.
- **Harness hardened** (`ORCHESTRATION.md`): periodic monitoring must use a **self-re-arming**
  scheduler, and `pgrep -f <name>` is banned as a liveness check — it cost 6 h of idle GPU when a
  wait loop matched its own monitoring shell. Matching `GOTCHAS.md` entry added.

### All of it ran; only `F_w40` is still going

14-epoch rematch of both arms → editability of each → confound-free re-evaluation of the 4-epoch
pair → `F_w40` (in flight). Results in `runs/othello_arch/*_editability.json`.

### ⚠ Monitoring: `CronCreate` does not fire here — measured

`harness/ORCHESTRATION.md` was edited this morning to prescribe a self-re-arming scheduler over
hand-rolled watchers. **That was wrong and is reverted.** Over a **2 h 39 min idle window**
(10:44 → 13:22) the cron produced **zero** wake-ups while background-task completions fired
reliably all night. The prescribed pattern is now a **staggered bank** of `run_in_background`
watchers (T+18, T+36, T+54 …), which removes the dependence on the agent re-arming without
depending on a primitive that does not deliver. Verify a mechanism has fired; registration is not
delivery.

---

## 2026-08-21 (later) — Nanda's write replicated, the pseudoinverse vindicated, and two review notebooks shipped

Scratch `2026-08-21-linear-direction-interventions.md` (with a marked CORRECTION) and
`2026-08-21-composition-random-baseline.md`; `findings/editability.md` and
`findings/state-geometry.md` both updated. Code:
`othello_transfer/{linear_intervention,single_layer,nanda_on_discworld,controls_lib}.py`,
`latent_linearity/composition_lib.py`. **No models trained.**

### 1. Nanda's linear-direction intervention replicates on their model, with our probe

`x ← x + α·p_d` along **our** linear probe's weight column. Reproduced null **2.723**, identical to
their Table 2. At α = 0.12: Li error **0.108** against their published **0.10**. The
`target − current` variant reaches **0.026** / Edit Index **+0.691** — the best result anything has
posted on this benchmark. Their Figure 7 shape reproduces (error collapses only once ≥ 6 residual
points are written).

### 2. ⚠ A correction I made the same day: our pseudoinverse is the *strongest* editor on their model

I first reported that `inject_state` fails on Othello-GPT (never below Li 1.461) and started
walking back 2026-08-20's "the editor is cleared". **That was wrong.** Sevan asked for the
single-layer variant. Written at **one** residual point instead of nine:

| point | 0 | 1 | 2 | 3 | 4 | **5** | 6 | 7 | 8 | all 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Li error ↓ | 2.719 | 2.721 | 2.581 | 0.763 | 0.296 | **0.052** | 0.695 | 1.872 | 2.164 | 1.461 |
| Edit Index ↑ | −0.829 | −0.823 | −0.579 | +0.404 | +0.559 | **+0.697** | +0.206 | −0.443 | −0.136 | −0.275 |

**The mechanism, with its control.** Multi-layer application *helps* Nanda's fixed direction
(0.236 → 0.062) and *hurts* our pseudoinverse **28×** (0.052 → 1.461). A recomputed injection
re-reads the probe against the already-edited stream and re-imposes "hold the other 63 tiles",
undoing itself each layer; a fixed direction does not depend on `x` and cannot. 2026-08-20's
clearance **stands unqualified**, and the 2 × 3 table is now all-green on their model:

| mechanism | Othello-GPT (theirs) | discworld `W16` (ours) |
|---|---|---|
| Nanda linear-direction addition | **+0.603 ✓** | −0.118 ✗ (3.6× collateral) |
| our gradient editor (`_descend`) | **+0.656 ✓** | −0.194 ✗ |
| our pseudoinverse injection | **+0.697 ✓** (point 5 only) | −0.66 ✗ |

**One live consequence.** A **depth confound** this cannot settle: `W16` has 5 residual points,
Othello-GPT 9, and the error only collapses at ≥ 6 — a direct argument for run A.

⚠ **I also claimed a second consequence — that a single-point pseudoinverse write on `W16` had never
been tried — and that was wrong.** Sevan caught it. `transformers/transformer_world_state.ipynb` §4
(2026-08-04) writes `h + (target − (Wh+b))W⁺` at **each residual point individually** on `W2`/`W4`/
`W16` and finds it **inert at every one**: Edit Index −0.65…−0.68, equal to each model's own
unsteered value, at fidelity ratio 1.00. So the multi-layer pathology is an Othello fact and does
**not** contaminate the discworld numbers.

### 2b. The one genuinely untried piece — the α sweep — is now run, and it closes the question

`othello_transfer/pinv_alpha_discworld.py`, 9 s. 2026-08-04 took the full jump (α = 1) and never
swept the step size; Othello's single-point optimum is α = 1.5 with a ~50× spread. Adding α as the
only new axis, everything else held to 2026-08-04:

| α | 0.05 | 0.25 | 1.0 | 2.0 | 4.0 | 6.0 |
|---|---|---|---|---|---|---|
| point 1 | −0.684 | −0.683 | −0.676 | −0.654 | −0.575 | −0.493 |
| **point 2** (mid-depth) | −0.684 | −0.682 | −0.669 | −0.634 | −0.538 | **−0.443** |
| point 4 (last) | −0.684 | −0.683 | −0.681 | −0.677 | −0.667 | −0.656 |

**α = 1 reproduces 2026-08-04's −0.683…−0.669 against its published −0.68…−0.65** — the anchor holds.
**Nothing crosses zero**, and the α response is monotonic with no optimum, the same signature as
Nanda's addition here and the opposite of Othello's sharp peak. ⚠ The best cell is not an edit
anyway: ‖Δh‖/‖h‖ = 0.909, read-out left 15.96 sim units past the target (5× the original 3.19 error),
collateral RMSE +29% against target RMSE −8%. Where the write lands the read-out (α ≈ 1) the index is
inert to three decimals.

**The full Othello recipe — single point, mid-depth, α swept — has now been applied to `W16` and
fails.** Nothing about editor parameterisation explains the discworld negative.

### 3. Latent object-composition is mostly architectural (`latent_linearity/random_baseline.ipynb`)

Sevan asked whether `delta_h_analysis` §7's composition result holds for an **untrained** model. It
largely does — additivity is a first-order Taylor property of any smooth map. By composed cosine,
random init at the identical config reads **+0.890** against trained **+0.904** (linear) and
**+0.853** against **+0.835** (nonlinear — the untrained net is *higher*). Read against the
**renderer's own** non-additivity (the two objects share rays; ceiling 0.406 → 0.207 across
displacement scales), the trained *linear* model tracks that ceiling to ±0.05 at every scale while
random init misses it by 0.03–0.10. **The strong claim is dead; a ceiling-tracking claim survives,
on the linear family only.** Three flaws in my first pass had to be fixed to get here — uniform
displacement direction, a shuffled floor that permuted only one delta, and measuring displacement
from `positions[ef]` where the teleport is already in the data. All three inflated the effect.

### 4. Two notebooks shipped, both executed clean

- **`othello_transfer/controls.ipynb`** — 12 cells, 0 errors, ~4 min. Re-derives every number
  quoted above and in `2026-08-21-probe-reality-checks.md`: probe-data scaling (with an assertion
  anchoring to the published 0.9349), the observation-window baseline, the random-init baseline,
  the intervention sweep, and the single-layer comparison. Figures `fig_controls_probe.png`,
  `fig_controls_intervention.png`; results `runs/othello_transfer/results_controls.json`.
- **`latent_linearity/random_baseline.ipynb`** — 10 cells, 0 errors, ~15 s. Figure
  `fig1_composition_random_baseline.png`; results `runs/latent_linearity/random_baseline_results.json`.

⚠ **A tooling limitation worth knowing** (recorded in `GOTCHAS.md`): both notebooks exceed the
`Read` token cap once figures are embedded, and `NotebookEdit` requires a prior `Read`. Post-execution
cell edits must go through small JSON helpers that touch only `source` and assert the cell inventory
is unchanged.

---

## 2026-08-21 — Four controls on what our probes read; planning the architecture port

`othello_transfer/probe_scaling.py`; scratch `2026-08-21-probe-data-scaling.md`;
`findings/editability.md` updated (`observed`). ~3 min, no models trained.

**Result:** discworld probe quality is **data-saturated**. Sweeping probe training data over a 60x
range on `W16` (48k → 2.88M rows) moves MLP position R² **0.9315 → 0.9604**, +0.029, with successive
steps of +0.015 / +0.007 / +0.006 / **+0.001** — flat by ~1.5M rows, short of Othello's ≈6.7M. The
test-split fit at 1500 sequences reproduces the published **0.9349** exactly. So the ~140x
probe-data gap against Li et al. is **not** an explanation for the editability negative; the 0.96
ceiling belongs to `W16`'s residual stream, not to probe fitting.

**Three further controls the same day** (`scratch/2026-08-21-probe-reality-checks.md`), prompted by
Sevan's low-dimensional-observation-manifold hypothesis and his occluded-disc proposal:

- **Raw-observation baseline.** Linear probe on the observation the model receives: R² **0.292**
  (1 frame) → **0.323** (16-frame window); MLP **0.851**. Trained latent: 0.803 linear / 0.944 MLP.
- **Random-weight baseline** (Li et al.'s own `--random` control). Random init reads **0.559 linear
  / 0.819 MLP**. ⚠ **This corrected an overstatement I made earlier the same day** — of the trained
  latent's +0.48 linear gain over the raw observation, **about half is the architecture**, and
  training contributes +0.244. What survives: the depth *trend* separates them cleanly (random
  declines with depth, trained rises), and **the model's real achievement is linearisation** —
  position is already nonlinearly present in the observation (MLP 0.851) and the model makes it
  linearly accessible. **Nonlinearly, our probe is largely reading the observation, not a learned
  state.** Against Li et al., where training removes ~93% of the random model's error versus ~71%
  of its unexplained variance here.
- **Occluded discs — inconclusive, and the dataset cannot fix it.** A single frame reads a fully
  occluded disc at RMSE 0.87–0.98 (floor 1.83), because occlusion here constrains a disc to the
  *angular shadow* of the visible one rather than hiding it. The trained latent is no better than
  random init on object 1, and there is **no decay with time hidden** — the signature of a per-frame
  constraint, not a carried state.

**Two traps recorded in `GOTCHAS.md`:** `is_visible` means frustum overlap and is **identically
True** on every current dataset, so the visibility masks in `othello_gpt/pipeline` have never
filtered anything; and object index is confounded with brightness under `fixed_reflectivities=True`.

**Where the confound elimination stands.** Editor implementation: cleared (2026-08-20). Probe
training data: cleared (today). Still open: **model** training data (3.6M unique frames against
their 1.2B unique transitions) and **architecture** — which the planned run folds together
deliberately.

**Planned next (Sevan, 2026-08-21): "run A" — their architecture on discworld at their data scale.**
minGPT verbatim with two substitutions (`nn.Embedding` → `Linear(128, 512)`; logit head →
`Linear(512, 128)`, CE → MSE), 8 blocks / 8 heads / `d_model` 512 / full causal / learned absolute
positions / dropout 0.1, `block_size` 39, trained on ~25M freshly generated dataset-#4-config
episodes. Sevan has explicitly authorised violating the registry's `d_model` = 256 convention for
this run, and accepted that multiple variables move at once: the bet is that the negative survives
all of them, in which case none of them was the cause. Run B (their architecture at our current data
scale) is **not** to be run unless A shows editability.

**Feasibility, measured:** generation 645 episodes/s/core → **25M in ~30 min on 32 cores**; storage
**1.19 TB** all fields, **0.54 TB** dropping `obs_depth`/`obs_id` (regenerable from seeds),
0.27 TB at fp16; training ~975M transitions ≈ their 1.2B, one pass, order 0.5–2 h.

~~**New run-A decision surfaced 2026-08-21:** `always_in_frustum` for the corpus.~~ **Not a
decision.** Sevan rejected the inference it rested on ("I'm also not convinced that this means we
need discs that leave the frustum — that seems like a jump in logic I don't follow") and has settled
it: run A uses the standard world, `always_in_frustum = true`, matching dataset 4. I kept carrying
it as open after it had been answered; it is struck here and removed from the brief.

~~**Seven decisions raised with Sevan and awaiting their call**~~ — **all seven were settled the
same day** and are recorded in the brief's *Provenance of the decisions* section: noisy training
target · **no** ReLU after the input projection · dropout 0.1 kept · their Trainer and schedule ·
seed-range asserts (generating from base_seed 0 would regenerate #4's test and edits episodes as
training data and silently destroy the eval) · reuse #4's test/edits splits · 200k-episode probe
split. Left stale here for two days; struck 2026-08-21.

---

## 2026-08-20 (later) — NEW thread `othello_transfer`: our probe and our editor on THEIR model — **REPRODUCED, and the implementation question is closed**

New thread `notebooks/experiments/editability/othello_transfer/` (`probe_transfer.ipynb`,
`othello_shim.py`, `othello_data.py`, `transfer_pipeline.py`, `board_grid.py`, `README.md`,
`OTHELLO_TRANSFER_RUNS.md`). `othello_gpt/othello_probe.py` extended **in place** with a 3-way
classification head. `METRICS_AND_EDITORS.md` §6 registers the new metrics; note
`scratch/2026-08-20-othello-transfer.md`; `findings/editability.md` updated (`replicated`,
**★-candidate**) and its Current understanding point 1 rewritten. Outputs in
`runs/othello_transfer/` (5 figures, `results.json`, `probe_cache/`). **Status: complete.**

**HEADLINE — our editor reproduces their intervention, by a wide margin.** Null 2.723 (their 2.68) →
best **0.016** (their 0.12): a **170x** error reduction against their 22x. Probe: **0.57%** error at
the best layer against their 1.7%. Stable across three independent full runs, two probe widths, two
step counts, two step sizes, nine applied layers and both held-out conventions. **The discworld
editability negative is not a bug in our probe fitting, edit objective, descent, or multi-layer
schedule** — that exact code, unmodified, works on the model the result was published on.

**On the Edit Index, the axis that compares the two worlds:** Othello **−0.829 → +0.656** (crosses
zero, 81% of available headroom); discworld `W16` −0.684 → −0.194 (never crosses, 29%). On the
symmetric-difference support the sweep reaches +0.868 against a −0.943 floor.

**Three secondary results.** (1) **Nanda's linear finding reproduces exactly** — linear probe 23.90%
in absolute colour vs **0.72%** in mine/theirs, so Li's "linear probes fail" is a coordinate-frame
artifact. (2) **Their frame-level split does not inflate their number** — 0.57% vs 0.66% by
convention, against the +0.34 R² inflation the same change causes on discworld. (3) **The probe
constraint does not pin down the write on their model either**: `hit_target` reads 1.000 at every
alpha over a 50x range and the edit objective reaches ≥99% of its best reduction throughout, while
the outcome moves by a factor of **83**. That makes 2026-08-18's "the optimiser decides which
probe-satisfying write you land on" a property of probe-derived writes generally, not of our world.

**What it does NOT settle:** the two remaining explanations — the *world* (discrete board consumed
directly by the legal-move computation vs continuous positions reaching the output only through a
renderer) and the *read-out* (their probe predicts a quantity the computation consumes; ours one
merely correlated with it). Both survive. The cleanest next test is the read-out one, flagged on
2026-08-18 and still unbriefed.

**A reproducibility bug found and fixed on the way out.** The corpus generator seeded each
multiprocessing worker from its **pid**, so `SEED = 0` produced a different 20k-game corpus every
run (1,179,692 / 1,179,508 / 1,179,665 rows across three runs) and the probe cache — keyed on row
count — could never hit, silently refitting a 37-minute grid each time. Now seeded per work item and
verified reproducible both ways; recorded in `GOTCHAS.md`. The reported numbers were stable across
the pre-fix runs (null baseline 2.723 identical, probe error 0.56–0.58%), so no conclusion changes,
but those runs are not bit-reproducible. **The committed artifact is the post-fix run** (5th
execution, 72 min, `probe_cache/probe_grid_e8664d50f504.pt`), which re-runs bit-identically.

**Process note worth reading.** The first two runs were scored at the wrong operating point. Run 1
selected the step size by read-out convergence — which **saturates** here — and so picked the
smallest alpha, understating the method ~100x. Run 2's replacement (objective convergence)
saturated too. Run 3 stops selecting: it sweeps applied layer at **both** ends of the range, labelled
"smallest probe-satisfying write" and "best of the swept range", and reports both. This is the
2026-08-19 correction's failure mode in mirror image, and it is now written into the notebook and
`METRICS_AND_EDITORS.md` rather than only into a run.

**The direction is the point.** `othello_gpt/` (2026-08-18) ran *their method on our model*. This
runs *our probe and our editor, unmodified, on their model*, against their own 1001-case benchmark.
It is the positive control for the one explanation of the editability negative that no experiment in
the thread could rule out: **that our editor implementation is simply wrong.** Every editability
number in the repo comes from that code.

**Sevan's spec, in his words:** keep as much of their architecture and environment fixed as
possible, keep as much of our probe and intervention code fixed as possible, build the minimal
bridge, and make data and design decisions that mimic Li et al. Explicitly **not** their
intervention code — ours, "because I need to be sure it wasn't my implementation for how to do the
editing that was failing."

**The bridge is 135 lines and contains no editing logic.** `othello_shim.py` supplies the seven
names our editing code calls over their unmodified minGPT `GPT`. Gated as **bit-identical** to their
`GPT.forward` and to `GPTforProbing` at all nine residual points. `build_edit_spec`,
`make_intervention_hook` and `_descend` run byte-identical.

**Recovered the dead artifacts before any of this was possible.** All four of the paper's Google
Drive links are **404** (verified against a known-public Drive folder as a control). The checkpoint
came from the third-party mirror `sbentley/othello-world-ckpts` and was verified three ways:
loads into their own `GPT` (141/141 keys), **functionally identical to the authors' TransformerLens
conversion** (max probability difference 2.1e-6), and reproduces the paper's legal-move property
(99.98% of mass on the legal set). The championship dataset is ~93% reconstructible from the public
WTHOR archive if ever needed — 136,055 games parsed and validated against their own rules engine.

**Findings already banked from triage, before the main run:**
- The shipped `intervention_benchmark.pkl` is the paper's **natural** subset: our reproduced
  null-intervention baseline is **2.723** against their published 2.68 natural / 2.59 unnatural.
  1001 cases, all integrity checks clean. The flip changes **2.1 of 64 squares** on average.
- **The Edit Index translates, and the reference is exact rather than approximate.** Their generator
  draws moves uniformly from the legal set, so uniform-over-legal *is* the true conditional
  distribution. The unedited model sits **0.0016** RMSE from it per square against a **0.0193**
  separation between the two worlds — a 12x margin. Floors: **−0.829 (union support)**,
  −0.943 (symdiff); a perfect predictor of the unedited world scores exactly −1.
- **Two things about their released code**, both flagged in the notebook: `train_probe_othello.py`
  never calls `model.eval()`, so their probe harvest ran with dropout live at p=0.1; and it
  hardcodes `data_root="data/othello_championship"` even for the synthetic model, on a frame-level
  `random_split` — the leakage convention `harness/ANALYSIS.md` §2 exists to avoid. Their 1.7% is a
  frame-split number. We report both conventions, never merged.

**Substrate:** `seaborn`, `psutil`, `pgn` added to `.pim` (their `data/othello.py` imports them at
module level); `pandas` came along with `seaborn`, so the standing GOTCHAS entry asserting its
absence was rewritten as a convention rather than an environment fact. Verified additive only —
numpy and torch unchanged, 178 tests pass. `othello_gpt/README.md` gained the required pointer to
this thread and back.

**Owed / awaiting Sevan:** the `★` call on this entry; whether the read-out explanation gets a
direction brief; and whether the persistence question (step 0 only here, deliberately) is worth
designing. Two substrate items flagged for a decision rather than fixed silently: **notebook cells
here were edited through a narrow script, not `NotebookEdit`** — this notebook is 26k tokens and
`Read` caps at 25k, so the sanctioned tool chain is blocked (the `GOTCHAS.md` "figure-heavy
notebooks exceed the Read token cap" case); and **`pandas` is now in `.pim`** as a transitive
dependency of `seaborn`, which their `data/othello.py` imports at module level (verified additive
only; the standing GOTCHAS entry was rewritten from an environment fact into a convention).

---

## 2026-08-20 — Session start: repo moved, three substrate breakages fixed; an unrecorded 08-19 correction pass found on disk

Orchestrator re-onboard. No science this entry — substrate and record-keeping only.

**The repo moved** from `/home/sevan/research/physically-implicit-modeling` to
`/home/sevan/research/PIM/physically-implicit-modeling`, which silently broke three things:

1. **`git`'s `nbstripout` filter pointed at the old absolute path** (`filter.nbstripout.clean`
   and `diff.ipynb.textconv` in `.git/config`), so every notebook `git diff` printed
   `external filter ... failed 127` and diffed raw JSON. Repointed at the new path; notebook
   diffs strip correctly again. Anything committing a notebook before this would have failed.
2. **`harness/check.sh` reported every line of every harness file as a violation.** The awk
   pass prefixed each line with the absolute filename before grepping, and the deny-list
   contains `\bpim\b`, matched case-insensitively — so the new `/PIM/` path component made
   every line a hit. Now matched against the line only, with the filename added to the output
   afterwards. Verified both ways: clean on the real harness (16 files), still catches a
   planted violation.
3. **`CLAUDE.md` claimed Python 3.12**; the venv is 3.13.5. Corrected.

Health check: **178 tests pass**; `torch 2.11.0+cu130`, CUDA available (RTX 5090, 32 GB).

**Found on disk and not in this file: a correction pass on the `othello_gpt` thread**, dated
2026-08-19 in the artifacts, uncommitted. Written into `findings/editability.md` and
`GOTCHAS.md` but never recorded here and with **no scratch note**. What it holds:

- **A probe fit on mixed-scale targets barely trains the small dimensions.** An unweighted MSE
  in raw target units gives each output dimension a gradient share proportional to its
  variance — position 3.0–3.6 vs velocity 0.0033 in sim units, a ~1000x imbalance. Fix in
  `othello_probe.fit_probe`: take the loss in standardised target space. Mean velocity R²
  **0.158 → 0.276** on `W16`, matching a dedicated velocity-only probe (0.272); position pays
  0.938 → 0.927. **The tripwire is `MLP ≥ linear`, per dimension** — the mis-trained MLP read
  velocity *below* the linear probe (0.158 vs 0.200), which is impossible for a strictly more
  expressive model that trained. Second instance of this class after 2026-08-11.
  `pim.extractors.fit_readability_probes` now warns on >100x target-variance spread; no
  published repo number is affected, because the repo has always fit position and velocity as
  separate probes.
- **This strengthens the full-state-probe null.** The "a probe reading the entire world state
  changes nothing" result was originally hedged as a weak completeness test because velocity
  was barely readable. That was the probe bug, not the model: with a genuinely informative
  velocity read-out the edit arm still moves only −0.539 → −0.553.
- **Two Sevan-flagged corrections to the 2026-08-18 entry**, neither changing the qualitative
  conclusion: the reported gain used the read-out-convergence operating point (α = 0.05) and
  **understated the method ~3x** — the step-size sweep reaches −0.194 at fidelity 1.014
  (α = 0.3), inside the 1.05 guard, a gain of **+0.49**; and the backing waterfall was drawn
  from the four largest teleports, which sit at the **98th percentile** of the Edit Index
  distribution (+0.07 vs a −0.54 mean). Panels are now randomly sampled, with the extreme-case
  panel kept as a separately titled addition (cell [9b]).

**Owed for that pass** (not done by whoever ran it): a scratch note, and a commit. The working
tree carries 7 modified files (`METRICS_AND_EDITORS.md`, both `othello_gpt` notebooks,
`othello_probe.py`, `pim/extractors/standard.py`, `GOTCHAS.md`, `findings/editability.md`) on
branch `latent_linearity`.

**Still awaiting Sevan** (carried forward): the `★` calls on the 2026-08-18 and 2026-08-19
entries; whether the latent DiT's above-chance probe visibility is worth a look; whether
`directions/edit-direction-causality.md` goes `active`; and the standing offer to compress this
file to current state plus a dated archive.

---

## 2026-08-19 (latest) — `latent_linearity`: do the edits that work agree on a direction?

New thread `notebooks/experiments/editability/latent_linearity/` (notebook `latent_edit_directions.ipynb`,
`edit_directions.py`, `figures.py`, `README.md`, `LATENT_LINEARITY_RUNS.md`); note
`scratch/2026-08-19-latent-edit-directions.md`; findings updated in `editability.md` (`replicated`,
**★-candidate**), `architecture-independence.md` (`replicated`) and `object-individuation.md` (`observed`,
**★-candidate**); metrics registered as `METRICS_AND_EDITORS.md` §5; two entries added to `GOTCHAS.md`.
Figures in `runs/latent_linearity/figures/` (19 PNGs). **No models trained.** Notebook runs end-to-end in ~35 s.

Sevan's spec: extend `delta_h_analysis`'s ground-truth edit-direction analysis to GRU / RSSM / transformer / DiT,
then ask whether the two *learned* pathways (action channel, single post-edit frame) write the same displacement
as the training-free oracles. Not about editors, and explicitly not about compositionality — that comes later.

**HEADLINE 1 — the two oracles agree in every architecture.** `cos(counterfactual overwrite, freeze-time TF)`,
edit-only Δh, N=256: pixel DiT **+0.910 (25°)** · GRU **+0.808 (36°)** · transformer **+0.806 (36°)** · latent
DiT **+0.667 (48°)** · RSSM **+0.593 (54°)**, shuffled controls +0.00 ± 0.22. The GRU/RSSM values **replicate**
2026-08-03's +0.799 / +0.569 on an independently built Δh, and three new architectures agree.

**HEADLINE 2 — the trained action channel lands on the oracle's displacement.** On `XG_A_H256` all four
mechanisms edit the generation (+0.643 / +0.563 / **+0.645** / +0.216 vs unsteered −0.641), and the tightest pair
measured anywhere is **counterfactual overwrite vs the trained action channel: +0.872 (29°), 5.9× chance**. First
evidence here that "train something that emits Δh" targets a well-defined object. Correlational — the falsifying
test is in the new direction brief.

**HEADLINE 3 — Sevan's prediction was right, and sharper than expected.** Whether one **uncued** post-edit frame
persists is a fact about the training distribution: step-0 Edit Index **−0.002** (never saw a teleport) →
**+0.216** (`XG_A`, teleports always cued by an action) → **+0.532** (`XG_C`, same data and recipe, action input
removed). Being told about interventions in training makes a model *less* willing to believe an unexplained one.

**The negatives replicate too.** No shared "an object moved" axis (cross-episode cosine +0.00 … +0.04 in every
model and mechanism, chance 0), and the direction stays at or below chance visibility to a linear position probe
(GRU 0.73× · transformer 0.49× · pixel DiT 0.14× · RSSM 0.03×; the action write 0.91×). **One exception worth
pulling on: the latent DiT's 64-d carried code at 1.17× chance** — the only state object above it, and also the
least linearly readable state in the study (position R² 0.220 vs 0.74–0.86).

**Part 2 is GRU-only, and that is a checkpoint-inventory fact.** Audited every checkpoint under `runs/`: the only
teleport-action-conditioned or teleport-observing world models in this repo are GRUs. The action-conditioned
RSSMs (`runs/endogenous_rssm/R*`) take **forces**, which cannot express a teleport. Recorded in the thread
registry with the table.

**Machinery.** `scripts/editability_metrics.py` gained `shift_zones` (score an arm that leads by k frames) and
absorbed `representative_samples` / `random_samples` from `othello_gpt/pipeline.py`, which now re-exports them.
Two `GOTCHAS.md` entries: the RSSM prior/posterior round-trip in `delta_h_analysis`'s `continue_from` (real, but
**measured and cleared** as the cause of its weak freeze-time arm: +0.097 corrected vs +0.091 legacy), and the
DiT family's hedged decode reading as an off-by-one when it is not.

**Awaiting Sevan:** `★` calls on the three headlines; whether the latent DiT's above-chance probe visibility is
worth a look; and whether `directions/edit-direction-causality.md` (proposed today — the projection-decomposition
causal test, plus a teleport-trained transformer and RSSM) should go `active`.

**Standing substrate note:** this file is 220 KB and has become an append-only log rather than a handoff. Offered
to compress it to current state plus a dated archive; awaiting a go-ahead.

---

## 2026-08-18 (part 2) — The same edit, applied to the observed history instead of the latent, works

Sevan's follow-up spec, after reading part 1: (a) the edit frame looks nearly right, so plot it on
its own — unedited vs post-edit model output against both ground-truth worlds — and **double-check
the Edit Index**; (b) in a new notebook, apply the edit **backward through the whole observation
history**, using the model's own decoded positions and never ground truth, since constant velocity
makes a constant displacement a valid rewrite.

Notebook `othello_gpt/history_rewrite.ipynb` (+ `history_edit.py`); note
`scratch/2026-08-18-history-rewrite.md`; `findings/editability.md` updated (`observed`,
**★-candidate**). Edit-frame diagnostic added to `othello_gpt_probing.ipynb` as cells [16]–[17]
(Fig 9).

**The Edit Index audit came back clean — the metric is right and the intuition was also right.**
Three independent computations agree to 4 decimals. The frame *does* look nearly correct: the
unsteered output is 0.276 RMSE from the edited world over the **full frame**. But the index scores
only the rays where the two ground-truth worlds differ — **median 28 of 128, ~22% of the frame** —
and there the unsteered output is **0.564** from the edited world against **0.093** from the
unedited one, about **6× closer to the world without the teleport**. The model is an excellent
predictor of the *wrong world*. Both readings are correct; recorded in `GOTCHAS.md` along with a
second, genuinely silent issue: episodes whose differing mask is empty (3/256 here) are dropped by
`nanmean`, so the effective N moves by step.

**HEADLINE — the history rewrite crosses zero and holds.** Edit Index **−0.684 (unsteered) →
+0.626 at step 0, +0.351 at step 14**, fidelity **0.674** (the rollout ends **33% closer** to the
true post-edit world than doing nothing). Against its own `δ=0` reconstruction control (−0.569) the
gain is **+1.195** at step 0 and **+0.727** at step 14 — where the latent write gains +0.146 → +0.010
on the same episodes. **No ground truth is used**: positions come from the model's own probe
read-out, and rendering needs only radius and reflectivities, which are world constants on a
`fixed_reflectivities` dataset.

**The most informative number: decode error barely matters.** Handing the method *ground-truth*
positions instead of the probe's raises it from +0.626 to **+0.640** — a gap of **0.014** — despite
a 0.49-sim-unit decoded position error. An *inconsistent* write is rejected however accurately it
hits the probe target; a *consistent* one is honoured even when substantially inaccurate. That
relocates the barrier from **precision of the write** to **coherence of the evidence**, and it is
falsifiable (corrupt the rewrite's internal consistency at fixed accuracy).

**Depth sweep.** Step-0 index +0.265 (rewrite depth 1) → +0.594 (depth 5), flat from depth 8;
step-14 index keeps climbing +0.080 → +0.302 (8) → **+0.355 (16)**, flattening near the model's
16-frame per-layer attention window. Placing the object and *holding* it need different amounts of
history. **A single rewritten observation frame (+0.265) beats every latent write (−0.538).**

⚠ **It uses the renderer**, so it is not a pure latent intervention and must never be quoted beside
the latent editors as if it were one. Flagged in the notebook, the registry row, and the finding.

**Harness work:** `sim_config_from` / `object_constants` extracted into `editability_metrics.py` as
the one place the reference-render config and world constants are built (zones verified identical,
178 tests pass). A hand-rolled frustum test halved `x_near` — which is already a half-width — and
called **28% of ground-truth frames** out of view on an always-in-frustum dataset; replaced with the
simulator's own `frustum_half_width`, now asserted to read 0.0% on GT.

**Awaiting Sevan:** `★` on both 2026-08-18 entries; and whether the coherence-vs-precision reading
deserves a direction brief (the falsification test above, plus the W2/W4 runs to turn the
attention-window correspondence into a claim).

---

## 2026-08-18 (part 1) — Othello-GPT's method, ported exactly: probing replicates, editing does not

Sevan's spec: implement Li et al.'s (ICLR 2023, arXiv:2210.13382) decoder and editor **exactly as they
did**, on the transformer — the MLP probe, its accuracy across layers, then gradient steering from that
probe to move one object's position while holding the other, repeated at **every layer at and after the
applied layer**, reported across applied layers, with qualitative results plus Edit Index at step 0 and
across the rollout; and then the same again with a probe predicting the **entire** world state
(positions + velocities) under the identical edit objective.

Thread `notebooks/experiments/editability/othello_gpt/` (notebook + `othello_probe.py` + `pipeline.py`
+ README); note `scratch/2026-08-18-othello-gpt-method-port.md`; `findings/editability.md` updated
(`replicated`, **★-candidate**). No new world models — uses `runs/transformers/{W2,W4,W16}`.

**HEADLINE — the probing half of the paper replicates, the intervention half does not.** Best position
R² **linear 0.798 → MLP 0.934** (+0.136), MLP rising monotonically with depth: their §3 result holds
here. But the intervention lands the read-out **completely** (3.35 → **0.007–0.018** sim units, a 99.5%
reduction, at every applied layer) while the generation barely moves: Edit Index **−0.684 → −0.538**,
gain **+0.146** on a ±1 scale. Fidelity **0.993–0.999** — the write is **ignored, not destructive** — and
the arms collapse onto the unsteered curve **by step 1** (gap +0.146 → **+0.010** by step 14). The
waterfall is unambiguous: the object stays on the ghost locator and never reaches the target.

**Earlier applied layers propagate further** (−0.538 at points 0/1 → −0.622 at point 4), matching the
structural prediction that an edit at residual point ℓ changes block inputs for layers > ℓ only.

**The full-state probe changes nothing** (−0.539 vs −0.538) — but that is a *weak* completeness test,
because velocity is barely readable here at all (per-dim R² −0.04 to 0.45). Stated as such rather than
as "completeness does not help".

**Two findings that came out of the port itself.** (1) The **oracle observation ceiling on this model is
only +0.126**, so the probe write achieves ~18% of what a perfect single-frame intervention achieves —
any claim about the size of the failure has to be read against that. (2) **The optimiser decides which
probe-satisfying write you land on**: at a matched selection rule, Adam's write is 1.7–4.9× larger in norm
and moves the generation, while plain gradient descent lands the read-out with a smaller write and
moves it essentially not at all (point 0: read-out 0.192, Edit Index −0.680 = unsteered). The set of
activations satisfying the probe is large and the probe constraint does not pin down a member the
dynamics honour — the same shape as the 2026-08-05 tangent-constrained result, from a new direction.

**Flat across attention windows**: gain +0.153 (W2) / +0.137 (W4) / +0.146 (W16).

**Why it matters for the thread:** the probe-derived-write failure is **not an artefact of this repo's
editor implementations**. The strongest published version of that method — its schedule, its loss, its
multi-layer write, its own baseline — fails here too.

**READING (not established):** does not contradict Li et al., it locates the difference, and this
notebook does **not** separate the two candidates: (a) *the world* — their board state is discrete and
exactly determined by the move sequence and the flipped tile is consumed directly by the legal-move
computation, ours is continuous and reaches the output only through a renderer (consistent with
2026-08-05 putting `readable ≠ grabbable` in the world, not the model); (b) *the read-out* — their probe
predicts a quantity the next-token computation consumes, ours one merely correlated with it.

**Deviations from the paper, both deliberate and both stated in the notebook's opening table:**
regression + R² instead of 3-way classification (our state is continuous; probe *shape* unchanged), and
held out **by sequence** rather than by frame (their split leaks a trajectory-constant velocity label —
`GOTCHAS.md`, +0.34 inflation). Adam rather than plain GD for the activation update is an
implementation choice the paper explicitly sanctions; plain GD is run alongside and reported.

**Awaiting Sevan:** the `★` call on this entry, and whether the two candidate explanations above are
worth a direction brief (the cleanest next test is a probe trained on a quantity the decoder provably
consumes). Also still open from 2026-08-17: the `★` pass over the backfilled findings.

---

## 2026-08-17 — Harness restructuring: portable working standards, and the findings gate replaced

**Not a research session.** Substrate work, at Sevan's direction, in preparation for reusing
these working standards on a second project (a CV project on vesicle detection in
neuroscientific data).

**What changed, and where to look now:**

1. **`harness/` — portable working standards** (README · STYLE · ANALYSIS · WORKFLOW ·
   ORCHESTRATION · WORKER · COLLABORATION · UPSTREAM · templates · theme.py). Universal prose
   only; each file ends with a `## Local instantiations` section holding one-line pointers to
   this project's concrete specs. **`harness/check.sh` enforces the quarantine** (project nouns
   in portable prose = failure) and fires from a `PostToolUse` hook on every write into
   `harness/`. Known limit, recorded in `harness/README.md`: it catches project *nouns*, not
   rules phrased generically that are only true here — an actual port is the only test, which
   is what `harness/UPSTREAM.md` §2 exists to record.
2. **`CLAUDE.md` cut 492 → 133 lines.** Now a router: the **role fork first** (orchestrator by
   default, every spawned subagent is a worker), action-triggered pointers into `harness/`,
   then project mechanics and the four architecture invariants. Conventions moved to
   `harness/`; the 1D waterfall spec moved to
   `notebooks/experiments/editability/WATERFALL_SPEC.md`; traps moved to `research/GOTCHAS.md`.
3. **The findings promotion gate is gone.** Findings are now **agent-written and continuously
   updated**, with an explicit `status` per entry (`observed` / `replicated` / `established`)
   and evidence links; **Sevan marks `★` for significance**, which is orthogonal to status.
   Corrections are new dated entries that `supersede` or `retract`. Rationale, with the
   measurement that motivated it, is in `harness/WORKFLOW.md` and `research/README.md`.
   *The trigger: the newest entry in any findings file was 2026-07-17 while 25 scratch notes
   queued behind the gate.*
4. **A month of findings backfilled** (2026-07-18 → 2026-08-14) across `editability.md`,
   `state-geometry.md`, `architecture-independence.md`, `object-individuation.md`,
   `predictive-quality.md`, plus a **new `trained-editors.md`** carrying the 2026-08-14 headline.
   Every `Current understanding` header was resynced with its log. Entries marked
   **★-candidate** are the ones flagged for Sevan's significance pass — see below.
5. **Standardization pass.** Audit found **18 separate waterfall implementations** and 8 scripts
   hand-rolling probe fits while only one used the standard. Wrote
   **`pim.figures.waterfall_grid`** — the single implementation, with the two recurring
   violations made *structurally impossible* (a shared teacher-forced row is unrepresentable;
   fixed `vmin`/`vmax` on every cell) and a warning below three sample rows. 8 tests;
   170 + 8 tests pass. **Migration debt recorded, not yet paid** — the 18 copies are unmigrated.
6. **Auto-memory retired.** Content moved to `harness/COLLABORATION.md` (working with Sevan) and
   `research/PROJECT_INTENT.md` (the long-horizon destination). `MEMORY.md` now says only "not
   used; read these repo files instead". Backup of the originals in this session's scratchpad.

**Awaiting Sevan:**
- **The `★` significance pass** over the backfilled findings. Six entries are marked
  `★-candidate`: renderer-inherited `readable ≠ grabbable` (2026-08-05), the Δh reachability
  ceiling (2026-08-03), the transformer two-state result (2026-08-04), observation noise as a
  regulariser (2026-07-30), and the trained editor crossing zero (2026-08-14).
- **Whether `established` is set correctly** on the backfilled entries — that is the one status
  I was deliberately reluctant with.
- Directions statuses were normalized against artifacts on disk; several "executed" briefs could
  now move to `directions/done/`, which is a tidy-up pass worth doing after the `★` read.

**Not done / open:**
- The 18 waterfall copies are not migrated to `waterfall_grid`.
- The probe-fitting drift is recorded in `GOTCHAS.md` but not fixed in the 8 scripts.
- `scripts/eval_editability_endogenous.py` still computes the metric set retired 2026-07-30.
- The export to the CV project has not happened; when it does, fill in `harness/UPSTREAM.md` §2.

---

## 2026-08-14 — Trained editors on exogenous-action models: the first probe-free latent editor to cross zero

Sevan's spec: two world models trained where **objects teleport during training** — one told the
teleport (actions in), one observer — plus a **control** with no actions/teleports; a thorough
ablation over **all** edit types; a fine-tuning variant and an **MLP editor taking
`(h, start_pos, target_pos)`**; each under a **next-step** and a **k=8 rollout** loss; plus (added
mid-run) a fine-tuning variant using the **un-whitened pseudoinverse** from `metric_corrected_edits`.
Thread `notebooks/experiments/editability/trained_editors_actions/` (notebook + `TRAINED_EDITOR_RUNS.md`);
`scripts/train_action_editors.py`, `scripts/eval_action_editors.py`; note
`scratch/2026-08-14-trained-editors-actions.md` (**FLAG FOR PROMOTION**). **18 trained arms**; no new
world models needed (`XG_A_H256`/`XG_C_H256` from the hidden-size sweep, `CTRL_H256` = `controls/H256`).

**HEADLINE — `E_θ(h, start, target) → Δh` with the world model FROZEN reaches Edit Index +0.204
(control) / +0.111 (actions given) / +0.117 (observer)** against unsteered −0.671/−0.669/−0.578 —
gains **+0.875/+0.780/+0.695** at fidelity 0.84/0.89/0.82, Target RMSE ≈0.48 → 0.25–0.28, and **zero
prediction cost** (the model is untouched). For scale, the published best *learned* mechanism was
**−0.14** and the best training-free structural editor here is metric-corrected injection at
−0.42…−0.52. The waterfall confirms a real relocation (object on the green locator, ghost dimming),
with streaking the oracle lacks. **The only change from the published amortized editor is the extra
input**: giving it the *starting* positions hands it the displacement instead of making it infer the
world from `h` — and that moves the mechanism from −0.14 to +0.20.

**Sevan's addition — the un-whitened (Σ¹) write is a BETTER fine-tuning target, 6/6 cells.** Gains
over own unsteered, pseudoinverse → un-whitened: k=1 control +0.233→**+0.341**, actions
+0.187→**+0.292**, observer +0.182→**+0.302**; k=8 +0.141→**+0.263**, +0.126→**+0.178**,
+0.109→**+0.241** — and marginally cheaper in prediction. It also reproduces as the best training-free
structural editor (−0.516/−0.516/−0.423 vs pseudoinverse −0.656/−0.649/−0.552; published −0.51).
Σ_hh condition number 8.85e3, so the un-whitening gate passes.

**Adapting the EDITOR beats adapting the MODEL.** Every fine-tune arm degrades next-step RMSE
(control 0.1041 → 0.1218–0.1253; actions 0.1071 → 0.1205–0.1356) and still ends **negative**; the MLP
arms cost nothing and end positive. All fine-tunes carried retention (weight 1.0, confirmed with
Sevan), so this is not the known no-retention degeneracy.

**The two losses do NOT simply trade index for fidelity — the k=8 arms OVERTAKE** (added with the
by-step curves Sevan asked for, and it corrects the step-0-only reading). Control MLP editor:
`k=1` +0.204 → +0.215 (step 4) → +0.127 (step 14); `k=8` **−0.035 → +0.267 → +0.230**. The `k=8`
arms cross above their `k=1` counterparts by ~step 4 in every mechanism and every model, ending
higher on the index *and* lower on GT-traj RMSE — the rollout loss produces an edit that takes a
few steps to materialise and then holds. ⚠ The **unsteered** curve also climbs on its own
(−0.671 → −0.438) as the free-run drifts from both reference worlds, so the gap to each arm's OWN
unsteered curve is what is reported. Decoder Grad k=8 is the strongest oracle across the whole
rollout (+0.80 → +0.38), matching the gallery's "lands AND persists".

**Actions/teleports in world-model training buy nothing for latent editing** — the control is the
*best* cell for both new mechanisms. The action model's advantage shows up only in its **action
interface** (+0.618), which bypasses the latent entirely.

**READING (not established):** this does not overturn the negative, it **locates** it. Every failing
editor is a *probe-derived* write — a correlational direction, inverted. What was missing was never
reachability (`full_rowspace_edit`) or capacity (`action_hidden_size`) but a **map from the intended
change to the state change**, which a pseudoinverse estimates badly and a small network can learn.
That also explains the un-whitened result: a better approximation of that map. Honest limit: +0.20 is
just past equidistant, not the oracle's +0.63 — **it edits, not yet cleanly.** ⚠ The editor is given
**ground-truth** start/target, and generalisation to a withheld object or out-of-range displacement
is **untested** — exactly where the published fine-tuning arms failed ("a button, not a handle").

**Harness:** fixed `eval_action_sweep.xg_data`'s separation filter (hardcoded 15-step window even
when fewer steps were rendered — 3462 vs 6000 usable episodes); documented in code that **cuDNN
cannot backprop through an RNN in `eval()` mode**, so editors optimising `h` through the dynamics
must flip to `train()` (identical here — no dropout) and restore.

## 2026-08-13 — Hidden-size ablation on ACTION models: the negative DEEPENS with capacity

## 2026-08-13 (latest) — Hidden-size ablation on ACTION models: the negative DEEPENS with capacity

Sevan: *"run an additional ablation of hidden state sizes but all on the action models and make sure
that our findings replicate or are proven to change."* Thread dir
`notebooks/experiments/editability/action_hidden_size/` (notebook + `ACTION_SWEEP_RUNS.md`); driver
`scripts/train_action_hidden_sweep.sh`; harness `scripts/eval_action_sweep.py`; note
`scratch/2026-08-13-action-hidden-size.md` (**FLAG FOR PROMOTION**). **15 new runs, ~90 min GPU.**

**Design — two families chosen to differ on whether the action space contains the intervention.**
`XG_A_H*` exogenous **teleport-to-absolute-coordinate** actions (its action space *contains the edit*,
so "issue the action" is a **built-in ground-truth handle** — the strongest positive control the thread
has had) · `XG_C_H*` identical recipe with **actions withheld** · `EN_H*` endogenous L3 (forces, death,
**REINFORCE survival into the same GRU trunk that predicts**; forces cannot teleport, so it has no
action-interface arm by construction). `H ∈ {8,32,128,256,512}`, hidden size the only variable. The
passive `runs/controls/H*` curve is **recomputed** with the identical estimator.

**F1 (prediction saturates by H≈128) REPLICATES everywhere** — passive 0.1499→0.1040, XG_A
0.1681→0.1071, XG_C 0.2016→0.1772, EN 0.2080→0.1553. **F3 (canonicality moves the other way)
REPLICATES everywhere** — MLP fiber residual rises in all four (passive 0.288→0.637, XG_A 0.410→0.695,
EN 0.270→0.500). **F2 (readability rises) replicates for both exogenous families** (XG_A 0.195→0.786)
but **PARTLY CHANGES for endogenous**: 0.274→0.636 @256 then **falls to 0.546 @512** (velocity
0.496→0.196) — most likely under-training (every EN run got the same 6000 iterations), flagged not
explained.

**F4 — REPLICATES AND IS STRONGER THAN "FLAT".** The legitimate readout-injection gain over its own
unsteered row **shrinks toward zero as capacity grows**: passive +0.479⚠/+0.181⚠/+0.016/+0.008/+0.004 ·
XG_C +0.382⚠/+0.234⚠/+0.025/+0.026/+0.016 · XG_A +0.419⚠/+0.278/+0.026/+0.020/**+0.007** · EN
+0.194/+0.057/+0.001/+0.001/**−0.001** (⚠ = fidelity > 1.05). **Every large gain at H=8–32 is
degradation** (fidelity 2.3–3.1; the waterfall shows saturated bands, not a relocated object); by
H≥128 the editors are **inert** (fidelity 1.00, sitting exactly on the unsteered line). The two failure
modes trade places as capacity grows and neither is an edit. **Meanwhile the action interface RISES
over the same range: +0.216 → +0.455 → +0.582 → +0.618 → +0.608 at fidelity 0.71–0.83**, and the
decoder-gradient oracle rises too (+0.521 → +0.984).

**→ Capacity improves prediction, readability AND action-channel controllability while making the
latent readout channel *less* effective.** A dissociation, not a plateau — and internally consistent:
the capacity that makes position more linearly decodable (F2) makes `h` less a function of the physical
state (F3). A model **trained to perform this exact edit on command** gets better at it with capacity,
and none of that transfers to writing the target into `h`.

**F4 IN TIME (added 2026-08-14, Sevan asked why the by-step figure was missing).** The by-step Edit
Index was absent because `eval_action_sweep.py` **stripped every list when writing its JSON**, and the
notebook stripped them again when copying the passive family's published editability — **the same
list-stripping bug in three places** (it was also in `eval_action_editors.py`). All fixed; eval re-run
for all 15 runs; new notebook §4 / Fig 4. Result: **readout injection never separates from its own
unsteered curve at any step** — gap at step 14 is −0.002/+0.007/+0.052/+0.002 at H=256 across the four
families (≤ +0.007 at H=512) — indistinguishable from the curve it should be steering away from. The apparent rise of the injection curve is entirely the
*unsteered baseline climbing* as the free-run drifts from both reference worlds. **CORRECTION (2026-08-14): capacity does NOT buy edit
persistence** — an earlier entry said it did, reading the *gap to unsteered*; the **raw** step-14
decoder-gradient index is flat at ≈0.02–0.23 at every `H` in every family (it starts at +0.45…+0.99 and
reverts to ≈0 within 15 steps, reproducing the published +0.94 → +0.08), and the waterfall shows it
drifting back and streaking. The gap grew only because the *unsteered* index falls as the predictor
improves. Sevan caught it from the figure. New `CLAUDE.md` rule: gap-to-unsteered answers
"distinguishable from doing nothing"; only the raw curve answers "does the edit hold".

**⛔ TWO MEASUREMENT BUGS, all values recomputed (2026-08-14, both caught by Sevan).** (a) **The ground
truth carried more than one intervention** — dataset 7 fires random teleports on ~30% of transitions and
the *other* object's teleports landed inside the scored window, so every trajectory metric scored the
model on events it was never told about and nothing was comparable to a single-teleport edits split.
Both reference worlds are now **constructed** by rolling the frame-`ef` state forward under the passive
dynamics. (b) **Velocity R² was all-t, and the split convention differs from older notebooks**: same
probe and window, **by sequence 0.565 vs by row 0.905** on `controls/H256` (+0.34), vs 0.924 → 0.971 for
position — velocity is nearly constant within a sequence so a row split leaks the answer. Every
pre-2026-08-06 velocity number in the repo is on the leaky convention.

**Harness work:** new `eval_action_sweep.py` (canonical §4 metrics for both families + a passive
reference recomputed with the same probes; validated by reproducing the published passive sweep).
**BUG FIXED — `scripts/gen_continuous_dataset.py` was broken outright**: it rebuilt `SimConfig` from
*every* dataclass field of dataset 4's stored `sim` dict, which predates the soft-render/omni2d fields
→ `KeyError: 'soft_edge'`, so **no continuous-action dataset could be regenerated at all**.
**⛔ `scripts/eval_editability_endogenous.py` is STALE** — it still computes the metric set retired
2026-07-30 (`reach`/`collat`/`ghost`/`select`) that `CLAUDE.md` forbids; deliberately not used here,
and it should be migrated or marked superseded. Also caught: `eval_controls.py`'s readability block
splits **by row** and uses a **1×128** MLP, so its published F2/F3 numbers are not comparable to
`fit_readability_probes` — hence the recomputation.

## 2026-08-13 — History editing: the complement is the past, and that is why the edit fails

## 2026-08-13 (latest) — History editing: the complement is the past, and that is why the edit fails

Sevan's hypothesis: *the reason our edits fail is that the extra information in the latent — the part
outside the probe's row space, which we never edit — is information about the previous frames.* First
hypothesis in the thread to name a **content** for the complement rather than describing it geometrically.
Thread dir `notebooks/experiments/editability/history_editing/` (README, `history_tools.py`, two notebooks,
both executed clean); brief `directions/history-editing.md`; note `scratch/2026-08-13-history-editing.md`
(**FLAG FOR PROMOTION**).

**Design — the one thing that makes it decisive.** Every editor that works supplies a velocity-consistent
**translated history through observations**; every editor that fails writes one frame's position into the
**latent**. Both notebooks hold the content, `δ` and the frame count `n` fixed and vary **only the channel**.
Interpretation pre-registered in the brief before running.

**GRU (`controls/H256`).** (1) The past is readable **only nonlinearly**: linear probes match the
no-stored-history null to **+0.0008 at every lag**, while the MLP probe beats it by **+0.146 at lag 20**
(direct 0.883 vs learned null 0.737; imposed and learned nulls agree to ≤0.006, so it is not extrapolation
inefficiency). (2) **The complement is observation content, not past positions** — regressed on predictors
residualised against the present `(pos,v)`, held-out R²: past **positions** ≈ **0.00** at any depth; past
**observations** **0.609** (one frame) … **0.636** (ten), obs(t) alone 0.659, decaying with age
(obs(t−2) 0.550, obs(t−5) 0.364), shuffled controls ≈ 0. (3) The editor: latent −0.665 (n=0) → −0.585 (n=8)
→ −0.477 (n=16, fid 1.06) vs observation +0.028 → **+0.635** → **+0.671** (fid 0.61), unsteered −0.670 —
and **a matched-norm RANDOM write scores −0.585, exactly the latent n=8 value**, so the entire latent
"gain" is write size. (4) Structurally, the stacked lag probe's **effective rank saturates at 8** (the
`(pos,vel)` core) however many lags are stacked, and `Δh_true` sits in its row space at **0.49–0.60×
chance** — at/below chance and *falling*.

**Transformer (`transformers/W4`, span 13) — the sharpest version.** Position is readable at **every**
window position and every residual point (mean linear R² 0.591/0.769/**0.797**/0.773/0.765 for ℓ=0…4, peak
mid-stack as published). Writing the translated history at **all 5 × 13 sites** drives the probe readout
error **3.289 → 0.000 sim units** with ‖Δr‖/‖r‖ = 0.102 — and the Edit Index moves **−0.667 → −0.631 at
fidelity 1.00**. Every escape route closed: depth n=0→12 saturates by n=4; single residual points
−0.666…−0.647; layers ≥1/≥2/≥3 all ≈ all-layers; **re-applied at every rollout step: −0.631, identical**;
scaled ×2/×4 still fidelity 1.00; matched-norm random −0.661 (so a *small* real content effect, 0.036 index
points ≈ 3% of the observation result). Same content through observations: **+0.681**.

**READING (not established).** The premise is **right** and its implication is **wrong**: the complement is
history, but **observation-shaped** history, so the pre-registered fork resolves onto its second branch —
**the channel is the barrier, not the content.** The transformer makes it airtight: it *has* per-frame
slots, the probe reads each at R² ≈ 0.8, the write succeeds **exactly**, and the world does not move. A
frame's representation is not a *handle* on that frame. This gives the through-line (*no successful edit is
free of dynamics*) a mechanism — the working editors are the only ones writing **in the format the
complement is stored in** — and explains why `orthogonal_edits`' `∫gg′=0` bites.

**New code (additive, default-off, tested).** `pim/world_models/transformer/model.py`: `_run`'s `edit` now
also accepts a **callable** `fn(layer_idx, x) -> x` at every residual point (the tuple form wrote the **last
position only**, which cannot express a history edit); plus `residual_stack(state)` exposing the
`(n_layers+1, B, S, d_model)` write surface. +6 tests (suite **170 green**), default path asserted
bit-identical, ruff clean. **Registry updated** — `METRICS_AND_EDITORS.md` gained the history-editor pair,
the mandatory matched-norm + landing-diagnostic controls, and the effective-rank warning.

**Trap now pinned by a test:** *a constant offset is invisible to a pre-norm transformer* (LayerNorm's null
space), so a naive "shift the residual stream" write reads as a null result from the **editor** rather than
the **model**. Cost one debugging cycle.

**SAME-DAY FOLLOW-ON — full row space at H=8** (`editability/full_rowspace_edit/h8_full_rowspace_edit.ipynb`,
Sevan-directed). *Two orthogonal INLP probes on the 8-dim GRU, both reading current position — their row
space is the whole hidden state, so "the edit direction lies outside the row space" is removed by
construction.* **The construction works exactly:** two rank-4 probes, orthogonal to 7.9e-13, spanning 8 of 8;
reachable fraction of `Δh_true` **0.5897** with probe 1 (chance 0.7071 → 0.83×, *below* chance) → **1.0000**
with both. **And nothing improves:** cos(write, `Δh_true`) = **+0.040 (88°)** vs −0.011 (91°). At *identical*
displacement (‖Δh‖/‖h‖ = 0.791 = ‖Δh_true‖): full-rank **−0.277**, probe-1 −0.210, **random −0.172**,
unsteered −0.489, oracle **+0.529 at fidelity 0.81** — the full-row-space write is **not better than noise of
the same size**, and all three degrade. The literal ask is additionally degenerate: the stacked map is square
with **cond 14,373**, so it needs ‖Δh‖/‖h‖ = **2137** → fidelity 261, saturated garbage. *(Mechanism: the two
probes disagree by 0.923 sim units on real states, so exact agreement is off-manifold. Structural note: INLP
orthogonality means the min-norm probe-1 write **already** holds probe 2 fixed, so the second probe adds
nothing unless asked for a different value — which is the ill-posed request.)*
**→ Reachability was never the binding constraint.** The row-space fraction is a valid ceiling but not the
active one; a probe tells you which state would *read* as the target, not which state *is* it. Caveat: H=8 is
a much weaker model, all comparisons internal; **the H=256 analogue (64 probes) is not run** and is the
follow-on that would generalise it.

**Harness note:** figure-heavy notebooks exceed the `Read` tool's token cap once executed, so this thread
used a jupytext-style source file + an `nbformat` builder with **round-trip assertions** (cell count, exact
source, newline count) — the safe version of the programmatic build that silently no-op'd on 2026-08-11.

**Owed:** one seed / one model each / position only; §2's residual decomposition is **linear** (the MLP
fiber residual 0.467 is the tighter target); the transformer write uses **linear** probes only, so an
MLP-gradient version of the all-position write is the obvious next arm; observation arms are fed **clean**
renders (optimistic for the channel that already wins). The `n`-saturation is predicted by this world's
constant velocity — a world with acceleration/bounces is where the literal hypothesis could still win.

_Previously updated: 2026-08-11 (branch `orthogonal_edit_analysis`: **omniscient-2D thread opened** — the editability negative SURVIVES full observability, but the omniscient latent is less readable/canonical/editable; occupancy-dilution hypothesis pre-registered)_

## 2026-08-11 (latest) — Omniscient 2D: full observability does NOT rescue editability

Sevan's request: *"train the GRU not on the 1D observations but on the whole omniscient 2D world
which is fully observable."* The whole thread's negative had only ever been measured through a 1D
perspective scan, which is lossy twice — it **projects** and it **occludes**. `orthogonal_edits`
relocated the negative to the *world* via `∫gg′ = 0`; that argument is stated for a 1D scan, so this
tests it. Thread dir `notebooks/experiments/editability/omniscient_2d/` (notebook, runs registry,
`frame_grid.py`, `WATERFALL_SPEC_2D.md`); note `scratch/2026-08-11-omniscient-2d.md`.

**New code, additive and default-off** (the `soft_render.py` pattern): `pim/simulator/render2d.py`
— 48×64 orthographic raster over the world rectangle, no projection/occlusion/perspective, flattened
row-major so the whole stack still sees `(N, T, R)` with R=3072 and needs **no changes**;
`obs_dim(cfg)` in `config.py` as the single source of truth for R; `--omni2d` flags; per-split seed
overrides on `generate_dataset.py`; `--n-train-limit` on `train_gru.py`. `tests/test_render2d.py`
(18, incl. a bit-identical pin), suite **182 green**.

**Design — a one-variable swap.** `datasets/12_omniscient2d` uses split base seeds matched to
dataset 4, so positions, velocities, reflectivities, edit objects and edit values are **bit-identical**
(verified, 200 rows/split). Runs `runs/omniscient_2d/{2D_H256_s0, 2D_H256_s1, 1D_H256_30k_s0}` — the
`controls/H256` recipe verbatim; the 1D arm is **sample-matched** via `--n-train-limit 30000`, which
selects precisely the 2D suite's scenes.

**0. The 1D control validates the pipeline** — it reproduces published `controls/H256` numbers (90k)
to within 0.03 index points on every editor at 30k. No 1D↔2D difference is a sample-size effect.

**1. THE NEGATIVE SURVIVES.** Best standard editor gain over its own unsteered row: **+0.11/+0.14**
(2D) vs **+0.13** (1D); Pseudoinverse Injection inert in both (+0.02 vs +0.03). Removing projection
*and* occlusion does not make the latent grabbable.

**2. THE SURPRISE — the omniscient latent is WORSE on every axis.** Position R² **0.634/0.686**
linear (1D: 0.797), **0.752/0.762** MLP (0.877); fiber residual **0.881/0.836** vs **0.583** (much
less canonical); every oracle weakens — Counterfactual **+0.38/+0.24** vs **+0.68**, Decoder Grad k=1
**+0.62** vs **+0.96**, Freeze-time **+0.34/+0.27** vs **+0.52**. Geometry moves the other way: PCA
hull @90% 74/70 vs 44, TwoNN 2.3/2.5 vs 3.2. Seeds agree to ≤0.14 index points.

**Interpretation (not established): occupancy dilution through the objective.** An object covers
**25.5%** of the 1D scan but **1.45%** of the omniscient frame, so under a plain per-pixel MSE ~98.5%
of the gradient is about background and the pressure to encode position is ~**18× weaker per unit of
loss**. The 2D model reaches a *lower absolute* next-step RMSE (0.0875 vs 0.1051) at a similar
ratio-to-noise-floor (0.62× vs 0.68×) largely by predicting empty space; Fig 5 shows the consequence
— soft blobs where the 1D model is crisp. Same blur explains the less-negative unsteered index
(−0.54/−0.52 vs −0.66).

**This makes result 1 provisional in a specific way:** it shows the negative is not caused by
projection or occlusion, **not** that it is independent of observation *sharpness*.
**PRE-REGISTERED follow-on** (guide fixed before running): an **occupancy-matched** omniscient arm
(reweight the loss by occupancy, or enlarge objects). If readability + oracles recover while the
standard editors stay inert → result 1 stands and "omniscient is worse" is an objective-weighting
artifact. If the standard editors improve too → result 1 was confounded by blur and must be re-run.

**⚠ Cross-channel comparability rule, now enforced in three places** (registry, notebook definitions
table, `editability_metrics` docstring): an object is ~13% of a 1D scan but 0.73% of the omniscient
grid, so whole-frame averages (next-step RMSE, Edit-frame RMSE, GT-traj RMSE) are **not**
cross-channel comparable. Only the **Edit Index**, R²/fiber residual, and ratios to each arm's own
reference are.

**2D WATERFALL SPEC — APPROVED 2026-08-12 and promoted to `CLAUDE.md`.** It is now the sanctioned form
for **any** 2D-raster observation, not just this thread: `CLAUDE.md` § Waterfalls carries the rule and
points at `omniscient_2d/WATERFALL_SPEC_2D.md`; `METRICS_AND_EDITORS.md` carries the same pointer (it is
named in `CLAUDE.md` as a past leak path for exactly this drift). Defaults accepted as-is (5 steps,
3 context frames). Added on approval: **`frame_animation`** — the optional third view, obeying the
existing animation rules (numbered persistent title, 3.03 fps, **990 ms holds** on the last pre-edit
frame and the edit frame; verified on `anim1_editors_2d.gif`, notebook cell [19]). It is an addition,
never a substitute — a claim still ships with grid + trails, since a GIF cannot be read in a notebook
diff or a paper. Two guards worth knowing: the `anim_num` ↔ filename rule is **enforced** (raises), and
the GIF encoder collapses identical hold frames into one carrying the summed duration, so **check the
duration list, not the frame count** (22 slots store as 18 frames).

**Superseded note — the spec was previously:** — a literal waterfall cannot be drawn when a frame is 2D.
Proposed: `frame_grid` (arms × time, every content rule of the 1D spec preserved; time subsampled)
+ `frame_trails` (all 15 steps composited). Validated against known answers (unedited world scores
exactly −1.00, synthetic collapse +0.16) and it earned its keep — it caught MLP Grad Steering's
respectable-looking −0.47 as *ringing artifacts at fidelity 1.11*, which the scorecard alone did not
separate from a real gain.

**Bugs caught:** (a) `build_inmemory_dataloaders` moved the full tensor to GPU *then* split, peaking
at ~2× the dataset (29.5 GB) — OOMs a 32 GB card that needs 14.8; now splits on the host. (b)
`generate_dataset.py` derives split seeds sequentially, which would have put the 2D test/edits scenes
inside dataset 4's *train* range — leaking against the 1D baseline; fixed with explicit overrides +
an overlap check. (c) **A silent no-op notebook**: building an `.ipynb` with `source = s.split("\n")`
drops trailing newlines, `.ipynb` joins `source` with `""`, so every cell collapsed onto one line and
— since every cell starts with a `# [N]` comment — became a comment. nbconvert reported success,
exec counts ran 1→20, **zero outputs, zero errors**. Use `splitlines(keepends=True)`; and treat *an
executed notebook with no outputs as not having run*.

## 2026-08-11 (later) — standard MLP probe was undertrained (Sevan-flagged; FIXED)

Sevan flagged the impossible pattern "linear R² 0.70, MLP R² negative" on the DiT probe grid. Diagnosis:
`train_extractor` batches over **sequences** (batch 512), so at N=500–1500 sequences the standard 30 "epochs"
is only ~30–90 Adam steps — the 2×256 MLP never converges; linear lstsq (exact solve) is unaffected. Validated
on GRU H256 h: MLP R² 0.17 @30ep → **0.89 @300ep, ABOVE linear 0.81** (as a strictly-more-expressive probe must
be); z-scoring inputs unnecessary once training is adequate. **Fix: `STD_EPOCHS` 30 → 300 in
`pim/extractors/standard.py`** (documented in-module). ⚠ Every MLP R² reported from `fit_readability_probes`
between 2026-08-06 and 2026-08-11 under-reads (incl. transformer thread MLP 0.58–0.68 and all of today's DiT/
input-grad notebooks); linear R² everywhere is unaffected. DiT notebooks re-executed with the fix same day.

## 2026-08-11 — DiT thread + Input Grad Steering (both Sevan-directed)

## 2026-08-11 (latest) — VAE + latent DiT BUILT, TRAINED, ANALYSED: the compression hypothesis is REFUTED

`directions/latent-dit-vae.md` executed end-to-end (Sevan: "treat the latent DiT as a wholly separate
architecture"). New code: `pim/world_models/vae.py` + `scripts/train_vae.py`; `pim/world_models/latent_dit/` +
`scripts/train_latent_dit.py` (frozen VAE + concat DiT core via a new `data_transform="identity"` flag;
implements `HiddenStateModel` in observation space so the whole eval suite runs unchanged); loader dispatch;
`tests/test_latent_dit.py` (17) + DiT-core tests (40). **The owed API fix landed too**:
`predict_mode="sample_fresh"` (per-sample fresh noise, seedable `model.noise_gen`) replaces the `_eps_bank`
mutation hack. Runs: `runs/vae/{vae_z16,vae_z8}`, `runs/latent_dit/{0_z16_w4,1_z16_w2,2_z8_w4}`; registry
`editability/latent_DiT/LATENT_DIT_RUNS.md`; scratch note `scratch/2026-08-11-latent-dit.md`.

**Gate passed** (`latent_DiT/latent_dit_world_state.ipynb`): VAE recon 0.1294 vs noisy / **0.0986 vs clean**
(noise floor 0.1541 — it denoises); 16-d code retains position as well as the raw 128-d obs (MLP R² 0.540 vs
0.515); decoded next-step **0.02517 vs noisy** against a **VAE floor of 0.01672**, and **0.01174 vs clean vs the
pixel DiT's 0.01186 on identical data** (equal-or-better structure); `sample_fresh` rollouts stable; probe grid
linear R² **0.81** at late layers (pixel 0.70), again **flat across τ**.

**RESULT** (`input_grad_steering/input_grad_steering_latent_dit.ipynb`): compressing the write surface 128-d →
16-d does **not** make probe gradients semantic. cos(δ, Δ_true) in latent space **+0.118…+0.168** vs
observation space +0.146…+0.212 vs pixel DiT +0.11…+0.22 (chance ≈ −0.03). Edit Index band unchanged
arm-for-arm (obs-grad −0.30…−0.44, latent-window grad −0.38…−0.48, iterate grad −0.13…−0.23), and **both
oracles reproduce their pixel values exactly** (Render write @1 **+0.12**, velocity-consistent Counterfactual
window write **+0.71**). The most readable state in the thread (linear 0.810 / MLP 0.905) is not one bit more
controllable. **So the failure is NOT representation geometry — it is belief dynamics** (the ghost is carried
by the clean context; only window-consistent evidence removes it). Next: objectives that synthesise multi-frame
velocity-consistent evidence (SDEdit-style window re-noise, render-space objective, repeated small edits).

**1. DiT thread opened** (`notebooks/experiments/editability/DiT/`, registry `DIT_RUNS.md`). New code:
`pim/world_models/dit/single_frame.py` (**SingleFrameDiTModel** — vanilla diffusion forcing, single-frame
tokens; `--variant` in `train_dit.py`; loader dispatch; `tests/test_dit_single_frame.py`, 37/37 with existing
DiT tests). Four d256 runs launched (concat W2/W4 = span-matched to transformer W2/W4; single-frame W3/W5):
concat W2 **0.02480** / W4 **0.02445**; single-frame W3 **0.02504** / W5 **0.02424** — the tokenization
package (clean-channel concat vs vanilla diffusion forcing) is a **wash** on single-step quality at matched
frame span.
**Sample-mode rollout collapse SOLVED:** the vertical-stripe collapse in autoregressive Euler sampling was the
fixed ODE start-noise reused every step; fresh noise per step restores coherent stochastic rollouts
(free-run RMSE 0.257 → **0.186**, vs mean mode 0.192). API follow-up owed: a supported fresh-noise mode
(currently done by mutating `_eps_bank` in the eval script).

**2. Input Grad Steering — clean negative, both architectures** (`editability/input_grad_steering/`, README +
2 executed notebooks; scratch note `scratch/2026-08-11-input-grad-steering.md`). Backprop a frozen standard
linear probe through the network to the input observation: the readout is always fully driven, but δ is
adversarial fuzz — cos(δ, Δ_true) ≈ 0.25 (transformer) / 0.12 (GRU), Edit Index moves −0.69→−0.50 /
−0.68→−0.44, fidelity ≈ 1.0 (ignored). Oracle on the same write surface (newest frame ← clean edited render):
transformer **+0.27**, GRU **−0.01** (= belief inertia; matches First Obs TF). Readable≠controllable now also
holds at the input. **DiT leg (`input_grad_steering_dit.ipynb`, run `9_…w4_d256`): the guidance hypothesis
FAILS naive** — probe-guided Euler sampling at any effective strength degrades (index plateaus at −0.05 with
collateral 0.13→0.55, saturated frames), and **early-τ-only guidance = full-schedule**, localizing the failure
to the gradient direction, not kick timing. The rollout DOES re-cohere corrupted frames into valid-looking
objects (the projector is real) but to the nearest coherent world, not the target. DiT oracle render write:
+0.12. **Same-day extension:** (ℓ × τ) probe grid (`DiT/dit_world_state.ipynb`, new tested `resid_sink` hook)
— linear R² depth-monotone 0.26→0.76 and **flat across τ** (position belief is context-driven, not in the
iterate); and **pause–optimize–resume Latent Grad Steering @(L3, τ)** — drive the iterate to an exact verified
probe readout mid-ODE, resume: **−0.51 → −0.18** (best probe-only editor in the whole thread, collateral only
0.28–0.35), but waterfall shows **duplication not relocation** (new persistent target band + surviving ghost;
the ghost lives in the untouched clean context tokens). Whole-window history arm (n=4): −0.52, = n=1.
**MLP-probe variants (post-fix probes, R² 0.85–0.90): Input Grad · MLP −0.31 vs linear −0.52** — best
history-write on any architecture in the thread (cos +0.22; waterfall shows ghost dimming + new target band
from a clean-frame edit) — while **latent steering is probe-capacity-invariant** (≈ −0.2 plateau either way):
the plateau is belief dynamics (clean context tokens), not readout quality. All verdicts also invariant under
full fresh-noise Euler rollouts.
Escalations: SDEdit-style history re-noising, repeated latent edits over frames, render-space objective.
**Whole-window arms:** Input Grad · MLP n=4 = n=1 (−0.31; width doesn't help probe gradients). But
**Counterfactual window write (n=4, oracle): Edit Index +0.71 — the DiT edit essentially fully lands**
(target/ghost RMSE ≈ 0.09, collateral at baseline, GT-traj RMSE 0.206 < unsteered 0.303): all 4 window frames
← velocity-consistent counterfactual renders (own velocity, constant offset to hit target at ef). So the
Render write @1 cap (+0.12) was **conflicting velocity evidence, not belief inertia** — the DiT is fully
editable through consistent history; +0.71 is the ceiling probe-only editors (best −0.31) have to chase.

**1. Readability probes are now standardised in code.** Two different MLP probes had been in use and their R²
values are **not comparable**: `MLPExtractor` on defaults (**1×128, in-sample**) in `00_master_editability` and
`controls/`, versus a hand-rolled **2×256 held-out** probe in `iterative_probing` / `nonlinear_gru`. Two axes
conflated — capacity, and in-sample-vs-held-out. New `pim/extractors/standard.py::fit_readability_probes`:
linear lstsq + **2×256 ReLU MLP**, both fit on the same 80% of **sequences**, both scored on the same held-out
20% against the train mean. `MLPExtractor` gained `n_hidden_layers` (**default 1 = unchanged, asserted
bit-identical**) because the **MLP Grad Steering** editor writes through a *frozen 1×128* probe whose published
results are tied to it — reporting probe and steering probe are now explicitly different objects.
`tests/test_standard_probes.py` (7 tests, suite 107 green). Documented in `METRICS_AND_EDITORS.md` §2 (⭐ block)
and `CLAUDE.md`. **Pre-2026-08-06 MLP R² numbers are on a mixture of the two probes and need re-fitting before
cross-notebook comparison.**

**2. Transformer: uneditable against 5 activation editors, not 1.** Extended
`transformers/transformer_world_state.ipynb` (§4d, 2 cells, 30s) with **Global PCA Projection (PI)**,
**MLP Grad Steering**, **INLP @29 (R² corrected)** at the last residual point. All leave every transformer run
on the unedited side at **fidelity 0.99–1.00** (genuine inertness, not degradation): best is INLP at
**−0.39…−0.48** vs unsteered −0.66…−0.68 — same ordering and nearly the same values as the GRU. **Window size
is irrelevant** (W2/W4/W16 agree within 0.09 on every editor).

**The transformer-specific finding is upstream of editability:** the write itself lands less well. Global PCA
Projection drives the linear readout error 3.23 → **0.062** on the GRU (succeeds, then is ignored) but only
3.22 → **1.98–2.20** on the transformers at comparable ‖Δ‖. So the transformer negative is **weaker and more
confounded** than the GRU's — part of the failure is that the last-residual-point activation resists being
written to at all while staying on-manifold. Said plainly in the notebook rather than reporting the index alone.
Skipped by design: Local PCA Geodesic (cost); Multistep Steering and Decoder Grad k=15 are **ill-posed** on a
transformer activation edit, which cannot survive into the next step by construction.

_Last updated: 2026-09-23 09:55 PT — three appendix catch-ups running on the 4090 (nn_r2, grid categorical IM, token rescore with the mean-frame guard) before final_tables; last training job on the lab ~16:00_

## 2026-08-05 (later 6) — editor gallery: three slide waterfalls, and canonical editor names

Sevan asked for three slide-quality waterfalls (standard / learned / oracle families) replacing the master's
single all-editors waterfall, plus a fixed naming scheme. New notebook
`editability/editor_gallery/editor_gallery.ipynb` (12 cells, 0 errors), **17 editors on `runs/controls/H256`**,
N=64. Names are now canonical in `METRICS_AND_EDITORS.md` (⭐ block with the full old→new mapping).

**Corrections made to Sevan's proposed list before building** (he confirmed all): Multistep Steering feeds the
model's **own decoded** obs, never a render — that is precisely what separates it from freeze-time; `PCA
geodesic` refits a **local** tangent, so it is now **Local PCA Geodesic** (the old name collided with the global
one); Global PCA Projection is **alternating (POCS)**, not one-shot. He dropped the metric-corrected and
tangent-constrained editors from the gallery, added **Decoder Grad Steering k=15**, and chose **INLP @29 with R²
correction** (the uniform k=29 variant is degenerate — fidelity 1.57).

**RESULTS.** *Standard (all 7 fail):* best is **INLP @29 R²-corrected −0.37** (fidelity 0.93) vs unsteered
−0.68; Pseudoinverse Injection is visually identical to unsteered at −0.66. **Multistep Steering (PI) @16 is a
trap** — best-looking index (−0.22) at **fidelity 1.32** and collateral 0.429 vs 0.127; it drags both objects.
*Learned:* **Trained Editor (frozen world model) −0.68 → −0.14**, fidelity 0.68 — largest gain of any mechanism,
still short of 0. `no retention`'s unsteered index *rises* to −0.39 from degraded prediction alone, so its −0.30
is mostly scale movement. *Oracle — the headline:* **Decoder Grad k=1 = +0.97 at the edit frame → +0.08 by step
14** (visible striping: off-manifold, dynamics reject it), while **Decoder Grad k=15 = +0.83 → +0.77 at fidelity
0.20** — **the only editor that both lands and persists**. Counterfactual Overwriting +0.70 → +0.45; Freeze-time
+0.52 → +0.26; First Obs. TF only −0.08.

**THROUGH-LINE the figures make visible:** nothing that writes to `h` from a *readout* works. The mechanisms that
work either feed **externally rendered observations** (freeze-time, counterfactual) or optimise `h` against the
**full future** (k=15). Every one-shot readout-derived write leaves the object put or degrades the rollout.

**Master notebook updated:** Fig 5a now *displays* the three gallery figures with provenance rather than
recomputing (CLAUDE.md requires the master stay lightweight); Fig 5b (RSSM) retired as a no-op — the learned arms
are GRU fine-tunes with no RSSM counterpart. **Model provenance flagged:** the gallery uses `controls/H256`, not
the master's `3_dset3_...` (trained on dataset 3, evaluated on dataset 4), because H256 matches the eval dataset
**and** is the `base:` of every fine-tuned arm. Master §4 numbers therefore differ from the gallery's.


## 2026-08-05 (later 5) — metric-corrected edits: the whitening hypothesis is real, and insufficient

Sevan's derivation: a least-squares probe is `W = Σ_ph Σ_hh⁻¹`, so if `h ≈ h₀ + Jp` then `Wᵀ = Σ_hh⁻¹ J Σ_pp` —
**the probe's row space is `J` whitened by the inverse state covariance, not `J`.** With anisotropic `Σ_hh` the
two can be near-orthogonal even for a perfect probe, which would explain every probe-derived editor failure in
this thread as a metric artifact. Notebook `editability/metric_corrected_edits/metric_corrected_edits.ipynb`
(12 cells, 0 errors, 5 figs), note `scratch/2026-08-05-metric-corrected-edits.md`. GRU `controls/H256`, 78k-state
bank, N=256 edits, four pre-registered tests with the interpretation guide fixed *before* the run.

**GATE PASSED decisively:** `Σ_hh` condition number **1.79e4**. It also **independently corroborates the
derivation** — `iterative_probing` found the position code sits in *below-average-variance* directions, exactly
what `Σ⁻¹` does to a least-squares probe. Two routes, same structural claim.

**TEST 1 — the direction really improves.** cos to `Δh_true` **+0.079 → +0.236** (85° → 76°); reachable-subspace
mass **0.098 (0.78× chance) → 0.380 (3.04× chance)**. **The first probe-derived subspace in this thread that is
meaningfully enriched rather than at chance.** Monotone in α; the constraint-satisfying family and Sevan's
literal `Σ^α W⁺δ` agree to ±0.003.

**TEST 2 — best training-free structural editor the thread has produced, and still not an edit.** Only the metric
differs from the failing editor: unsteered −0.67 → Euclidean −0.65 → **Mahalanobis −0.51**, fidelity **0.98**,
Target 0.488→0.432, Ghost 0.589→0.534. **+0.14 index points at zero fidelity cost** — for scale, heavy
fine-tuning bought +0.13 *and* cost 13% of next-step prediction.

**TEST 3 — magnitude is not the answer either.** α=1 has a genuine optimum where ‖Δ‖ matches the oracle's (Target
and Ghost RMSE both minimise at ×3 = 3.39× a dynamics step vs the oracle's 3.75×), but fidelity crosses 1 there,
so the best legitimate arm is **×2 → −0.33 = 25% of the oracle's gain**. Past that the index rises only by
degrading (×8: +0.01 at fidelity **1.57**, striped garbage in Fig 5). Scaling the *Euclidean* direction is much
weaker (×8 → −0.45, Target barely moves). **Neither wrong-direction nor wrong-magnitude alone.**

**TEST 4 — the local metric is WORSE.** 1024-NN local `Σ_hh`: cos +0.143 vs global +0.236. Its Edit Index (−0.38)
looks better than global (−0.51) but that is **entirely magnitude** — local makes ‖Δ‖ = 2.27× a step vs global
1.13×, and at matched displacement global wins (×2 → −0.33). The anisotropy that matters is **global**, not local
curvature. Exactly the confusion the scale sweep exists to prevent.

**THE PATTERN NOW ACROSS THREE UNRELATED CONSTRUCTIONS:** tangent projection 57% captured → ~33% of the gain;
116-dim position-code projection 57% → 33%; metric correction ~24–38% → 25%. **Partial capture buys
sub-proportional effect** — the all-or-nothing reading, in graded form.

**WHAT IT CHANGES:** the whitening account is mechanistically real and *solves the specific puzzle* of why
probe-derived directions looked orthogonal to successful edits — it was a metric artifact, not a fact about the
representation. **Every past row-space/orthogonality number in the thread was measured in the wrong (Euclidean)
metric** and should be re-expressed. But it does not rescue editing, and the obstacle is now excluded from: the
probe's 4-dim slice, the linear position code, manifold-tangency, the metric, and magnitude.

**OWED:** characterise `Δh_true`'s complement directly (four misses now share a signature); resolve the ×2–×3
optimum (coarse grid, fidelity crosses 1 inside the bracket); re-express old orthogonality numbers in the
Mahalanobis metric; neighbourhood-size sweep for the local metric (only k=1024 tried); one model, one seed,
position probe only.


## 2026-08-05 (later 4) — iterative probing: the linear position code is 116 dims, and editing in it still fails

Sevan's experiment: fit a linear position probe, **project its 4-dim row space out of `h` entirely**, refit on
what remains, repeat to chance — how many probes, and is the total `4 × #probes`? This is INLP (Ravfogel et al.
2020) used to *measure* a code rather than erase an attribute. Notebook
`editability/iterative_probing/iterative_probing.ipynb` (11 cells, 0 errors, 2 figs), note
`scratch/2026-08-05-iterative-probing-position-dimensionality.md`. GRU `controls/H256`, 78k aligned states from
2k test sequences, split **by sequence**, float64 throughout.

**ANSWER: 29 probes, every one exactly rank 4 → 112 dims.** The arithmetic holds, and for a stateable reason:
`lstsq` returns the **minimum-norm** solution, so each new probe's rows land inside the already-deflated row
space, hence orthogonal to everything removed. Rank and orthogonality (max |inner| < 1e-6) are **asserted every
step**. Decay is gradual — R² 0.822 → 0.479 (24 dims) → 0.236 (44) → 0.091 (68) → 0.020 (112) — so "the"
dimensionality is threshold-dependent; half the readability is gone by 24 dims.

**The controls are where it gets sharp.** (a) **Random-ablation:** removing **112 random** dims leaves position
at **R² 0.767** vs **0.020** for the chosen 112 — the collapse is about *which* directions, not how many.
Shuffled-label floor −0.003. (b) **The position directions carry BELOW-average energy**: after 112 removals the
real track keeps **63.4%** of state energy, random keeps **56.9%** (≈ isotropic 56.3%) — position is not written
along the dominant variance axes. (c) **112 dims is the LINEAR code, not the information**: an MLP refit on the
fully deflated states still reads position at **R² 0.544** (from 0.909). (d) Scale: the states occupy 40/75/**172**
PCA dims at 90/95/99%, so the code spans **65% of the subspace the states actually occupy**.

**WHY THIS MATTERS TO THE THREAD.** Every §4 row-space number is measured against **one probe's 4 dims**, which
is also the entire reachable set of readout injection. The linear position code is **28× larger**. It does not
overturn the negative, but it reframes what "the edit direction is at chance in the row space" means.

**THE FOLLOW-ON THIS SETS UP (not run, and it is the interesting one):** measure the counterfactual Δh's overlap
with the **112-dim** accumulated subspace against matched chance `√(112/256)` = 0.661. If still at chance, a
successful edit is orthogonal to the *entire linear position code* — much stronger than the current claim. If
enriched, the failure is that one probe exposes the wrong 4-dim slice, and a **multi-probe injection editor
writing in all 112 dims** becomes the obvious thing to try.

**OWED:** greedy removal order (112 is an upper bound on a minimal spanning set); arbitrary R²<0.02 stop; one
model, one seed, **position only** — velocity untested.

## 2026-08-05 (later 3) — nonlinear GRU: superposition confirmed for real, editability negative survives a fifth axis

Sevan spotted that `delta_h_analysis` §7's object-**superposition** result might be nothing but a linear-decoder
identity — `decode(h0+d1+d2) = decode(h0+d1)+decode(h0+d2)−decode(h0)` holds for *any* vectors when `decode` is a
single `nn.Linear`. He was right. Then: *train a GRU with a nonlinear encoder and decoder and see which findings
survive.* Notebook `editability/nonlinear_gru/nonlinear_gru_findings.ipynb` (21 cells, 0 errors, 10 figs),
registry `.../NONLINEAR_GRU_RUNS.md`, note `scratch/2026-08-05-nonlinear-gru-decoder.md`.

**New code, additive and default-off.** `pim/world_models/gru/model.py` gained `enc_hidden_layers` /
`dec_hidden_layers` / `mlp_activation`; the extra blocks live in `enc_trunk` / `dec_trunk` submodules **absent
from `state_dict` at depth 0**, so every existing checkpoint loads unchanged and produces bit-identical output
(max-diff exactly `0.0` on `forward` / `step` / `decode`). Encode/decode routed through single `_enc` / `_dec`
choke-points. `tests/test_gru_mlp_depth.py` (10 tests; suite now 100 green). Runs `runs/nonlinear_gru/
{NL_enc2dec2_s0, NL_dec2_s0, NL_enc2dec2_s1}`, identical recipe to `controls/H256`, ~6 min each.

**THE ARTIFACT IS CONFIRMED — but it was the *evidence*, not the *result*.** Affine-decoder models: composed
decode equals the affine prediction to **6.6e-08**, so their composed Edit Index **+0.46** was algebraically
determined and never evidence about the latent. **On the nonlinear decoders (affine gap 8.5e-02–8.8e-02) the
result holds and is object-specific**: composed **+0.43…+0.44** against nulls of **−0.18…−0.20** (same
composition with object 1's delta from a *different* sample) and **−0.45…−0.46** (random Δ of matched norm),
with unedited −0.74 and direct +0.72. So compositionality is **confirmed, on decoders where it can actually be
tested** — a stronger position than before. It is *partial*: composed recovers ~82% of the gain to `direct` and
Fig 7 sample 47 shows it banding where direct stays clean. **Correction to my first framing:** I led with
"composed doesn't beat the affine prediction, so the nonlinearity is a tax" — but the affine prediction is not a
null model (it already assumes each single edit works). The real nulls are the two above. State-space cosine is
real everywhere but *weaker* with a nonlinear read-out (+0.873/29° → +0.784…+0.801/37–38°), floors +0.31…+0.37
and ~+0.05 — opposite to a "shallow decoder was hiding the structure" story.

**EVERYTHING ELSE SURVIVES.** Nonlinear variants are *slightly better* predictors (next-step 0.1029–0.1033 vs
0.1041), position is still linearly readable (held-out R² 0.803–0.805 vs 0.815), readout injection is still
**inert** (−0.65 vs its own unsteered −0.67), counterfactual overwrite still hits **+0.68 on every model**, and
the row-space fraction is still chance (1.05–1.24×). Fifth independent axis, after capacity, noise, actions and
architecture. Fig 4 shows injection pixel-indistinguishable from unsteered while all three oracles land the object.

**One new result worth carrying:** the **decoder-gradient oracle weakens on a nonlinear decoder**, +0.97/+1.00 →
+0.68…+0.72. Against an affine decoder it solves a *convex* least-squares problem in `h`; through an MLP it does
not. Its near-perfect score on the linear models was partly the decoder's convexity — relevant wherever it is
quoted as an upper bracket.

**Two method errors caught in-flight**, both mine: linear probe R² was **in-sample** while the MLP's was held-out
(which had produced velocity MLP R² *below* linear); and the §5 waterfall's GT column went black where displaced
objects leave the frustum. Both fixed, both stated in the notebook. **Harness bug fixed:**
`METRICS_AND_EDITORS.md` still mandated the shared teacher-forced `ef` waterfall row that `CLAUDE.md` banned on
2026-07-30 — and `CLAUDE.md` names that file as the leak path into the `controls/` notebooks. Registry corrected.

**OWED:** no depth/activation sweep (fixed at 2 blocks, ReLU); §5 uses one displacement pair and N=67; baselines
are single-seed. `delta_h_analysis` §7's Edit Index columns are not evidence on that affine-decoder model and
must not be cited without its §7b control — but the claim itself is now independently confirmed here.

## 2026-08-05 (later 2) — tangent-constrained injection: a new direction, still no edit

Sevan's construction: `WᵀΔ = δ` has a 252-d solution set; plain injection takes the minimum-norm member,
so take instead the member lying in the **local tangent space** — `Δ_tan = B(WᵀB)⁺δ` with `B` a local-PCA
basis. Notebook `editability/orthogonal_edits/tangent_constrained_injection.ipynb` (14 cells, 0 errors,
4 figs), note `scratch/2026-08-05-tangent-constrained-injection.md`. GRU `controls/H256`, N=256, bank 58.5k.

**RESULTS.** (a) Local manifold is **~22-d** at 90% variance (33 at 99%), not 5–10. (b) `Δh_true` **is**
enriched in `span(B)`: 0.568 at d=8 vs 0.177 chance = **3.2× chance** — the first subspace in this thread
that is meaningfully enriched (the probe row space is at/below chance). (c) The editor is genuinely new
(cos **+0.069** with plain injection) and moves the Edit Index −0.644 → **−0.290** — **but that is
degradation, not editing**: Target RMSE unchanged at 0.488, Ghost RMSE worse, fidelity ratio **1.14**.
Scale sweep reaches +0.007 at α=32 with Target RMSE **3.997** and fidelity **16.7**.

**Decisive control:** project the *working oracle* onto `span(B)` — the ceiling for any tangent-constrained
editor. Keeps 57% of `Δh_true`, cos **+0.568 (55°)**, no degradation (fidelity 0.91), and still scores only
**−0.197**. **Keeping 57% of the true edit yields essentially none of the effect.**

**Two takeaways:** the binding constraint is *not* manifold membership; and the edit looks close to
**all-or-nothing**. The latter is new and directly testable — sweep `h₀ + β·Δh_true` for β ∈ [0,1] and look
for a threshold. That is follow-on #1.

**Waterfalls added (Fig 5) and they settle the failure mode.** unsteered / plain injection /
inject-then-project are visually identical, object parked on the ghost locator. The tangent editor
**streaks**, and at ×4 / ×32 it is vertical-stripe garbage. Only the counterfactual oracle puts the object
cleanly on the green target. Correct word per `CLAUDE.md`: the tangent editor **collapses** (degenerates
off-distribution), it does not *revert*; plain injection is the *inert* one. Four scalar figures had not
separated these.

**Harness updated** on the back of this: `CLAUDE.md` now makes a waterfall **mandatory** for any claim about
an effect on generations (with the arm's headline metric in each column title, and degenerate settings shown
as their own columns), and adds a legibility rule — horizontal bars for long series names, and eyeball the
rendered PNG rather than trusting that the cell ran.

**Process note:** my first summary logic called this a success off the Edit Index alone. The fidelity ratio
is exactly what `METRICS_AND_EDITORS.md` requires to gate a success claim, and it caught it. Also fixed: an
alignment check that varied warm-up length instead of the compared frame (k=0 and k=+1 differed by 0.5%,
testing nothing).

## 2026-08-05 (later) — soft/differentiable renderer: the result survives

Sevan asked for the full differentiable-renderer implementation plus a smoothed-but-not-differentiable
control ("closer to a standard simulation a world model would be trained on"), a fresh dataset, and a
retrained GRU with the rendering as the only variable. All done. Notebook
`editability/orthogonal_edits/soft_render_geometry.ipynb` (13 cells, 0 errors, 4 figs).

**New code, all optional and additive** — `renderer.py` is untouched and every knob defaults to off, pinned
by `test_soft_render.py::test_defaults_are_bit_identical`: `pim/simulator/soft_render.py` (numpy + torch
backends), 10 tests, 4 new `SimConfig` fields (`soft_edge`, `soft_shading`, `soft_psf_sigma`,
`soft_occlusion_temp`), CLI flags on `generate_dataset.py`, dataset `datasets/5_soft_render`, model
`runs/soft_render/H256_soft` (400 epochs, identical protocol to `controls/H256`).

**RESULT — survives.** `N_eff` of the change under a nudge went 1.00 → **15.43** (a 15× spread of the
derivative off the silhouette), and the geometry did not move: cos(required, pseudoinverse) 85.8° → **87.0°**,
row-space fraction 0.77× → **0.74× chance**, injection closes −0.1% → **−2.0%** of the gap. With the **exact
Jacobian** from the differentiable backend it is **0.37× chance** — further below. Hard- and soft-occlusion
backends agree to 3 decimals, so the realistic antialiased-simulator control behaves identically. In the
retrained GRU (quality gate 0.73× its own noise floor; position R² 0.824): cos **86.9°**, row-space
**1.11× chance**, injection **−0.585 → −0.567**, inert.

**Prediction I got wrong, recorded:** I said shading would be the structural knob and antialiasing/blur inert.
Reverse is true — soft edge does nearly all the spreading (1.0 → 10.4), lambert adds nothing (→ 9.2), because
a dome is steepest at its rim and flat at its apex. Sevan's instinct was right.

**Bugs found and fixed** (would have invalidated the soft numbers): (a) `clean_obs` is *reconstructed* as
`reflectivities[obs_id]`, exact only for a flat renderer — soft datasets now store `obs_clean` and the loader
prefers it; (b) `build_edit_zones` built its own `SimConfig` and would have rendered the reference worlds
**hard** while the model was trained **soft**; (c) the soft renderer's relaxed visibility gate reported a hit
on nearly every ray, corrupting `obs_id`/`obs_depth`.

## 2026-08-05 — NEW branch `orthogonal_edit_analysis`: the negative is a property of the WORLD, not the models

Sevan's question, from a conversation about whether the renderer is a function and whether it is linear.
Every `readable ≠ grabbable` result so far was measured **inside a trained model**. Transformer §6 sharpened
it to geometry (87–90°, row-space fraction at/below chance) but still in-network, leaving open whether the
network **chose** an awkward layout or **inherited** one. Brief `directions/orthogonal-edits.md`, notebook
`editability/orthogonal_edits/observation_space_geometry.ipynb` (11 cells, 0 errors, 3 figs), note
`scratch/2026-08-05-observation-space-geometry.md` (**FLAG FOR PROMOTION**). **No model is loaded anywhere
in this notebook** — that is the design.

**The mechanism, statable in one line:** an object covers `n` rays at constant intensity; nudging it one ray
changes only the **two edge rays** and leaves the interior untouched. So `cos ≈ −√(k/2n)` — **a linear probe
reads the plateau, moving the object changes the edges, and a plateau is nearly perpendicular to the spikes
at its own edges.** Underneath: for `f(p) = g(·−p)`, `∫ g g' = ½∫(g²)' = 0` exactly. Predicted −0.125,
measured −0.151 over 1996 samples.

**RESULT — inherited** (N=2000, per-sample then averaged, structural analogue of §6 with `h∈R^256` → raw
`o∈R^128`, chance `√(4/128)=0.177`):
- `cos(required change, pseudoinverse direction)` = **+0.073 (86°)** teleport, **+0.011 (89°)** nudge;
  shuffled controls +0.001 / −0.000. Inside the null.
- Row-space fraction **0.135 (0.77× chance)** teleport, **0.097 (0.55× chance)** nudge — *below* chance.
- These land on top of the in-network numbers (86–89° here vs 87–90° there).
- **The direct demonstration §6 could not make:** apply the injection to the observation itself. The probe
  reads the target to **1.25e-06 sim units** — a perfect write by its own objective — and it closes
  **−0.1%** of the RMSE gap to the target world, rendering as a diffuse ripple while the plateaus stay put.

**Why it matters:** this relocates the thread's central negative from the models to the world. It retires
better-probes / longer-training / different-architecture as fixes *for this world*, and it explains why
every editor that DOES work (counterfactual overwrite, freeze-time, history overwrite) acts through the
**observation sequence** — the only channel that can produce an edge-shaped change.

**Caveat that must be closed before promotion:** this renderer has **hard silhouettes** (`obs_intensity` is
the first-hit reflectivity, flat, no antialiasing — the clean render is *piecewise constant* in position:
only {0, 0.4, 0.8} appear, and 14 of 25 small steps changed nothing). The `∫gg'=0` argument predicts
orthogonality survives smoothing, but **that is untested and is the first thing a referee will ask.**
Follow-on #1 is a soft-renderer replication.

**Gotcha recorded:** the repo's `_fit_mlp` does not converge on position targets without standardising them
(returns R² ≈ −0.5, worse than the mean — a failed fit, not a result). Standardise `y` before fitting.

## 2026-08-04 — Transformer world model, end to end (branch `michael_controls`)

Sevan's request: implement the transformer as a full architecture arm — model, training script, sweep, and a
notebook covering every section of `00_master_editability`, with a multi-layer investigation. Direction brief
`directions/transformer-world-state.md` (`[reframe]`), registry
`notebooks/experiments/editability/transformers/TRANSFORMER_RUNS.md`, notebook
`transformers/transformer_world_state.ipynb` (17 cells, 0 errors, 6 figures), note
`scratch/2026-08-04-transformer-world-state.md` (**FLAG FOR PROMOTION**).

**New code (all tested + linted):** `pim/world_models/transformer/model.py`, `scripts/train_transformer.py`,
`tests/test_transformer.py` (6 tests), transformer dispatch in `pim/world_models/loader.py`.

**Why it is a `[reframe]`.** Every §4 result assumed the model has **one** state — a vector that is both
carried and readable — which is what makes "edit the world state" well-posed. A causal transformer has **two**,
and they come apart: the **carried** state is the observation buffer (persists, but each slot is one frame);
the **readable** state is the residual stream (history-dependent, but recomputed every step). A write to the
readable state is transient *by construction*, not by failure — reporting it as the GRU's reversion would be
wrong.

**Load-bearing structural fact, established first:** the carried state spans `n_layers×(window−1)+1` frames,
**not** `window`. Pinned by `test_buffer_rollout_matches_full_sequence` — a one-pass banded forward and a
step-by-step buffer rollout agree only at `state_span`, diverging from exactly `t = window` otherwise. Sizing
the buffer by `window` would have understated the history an edit must overwrite by a factor of `n_layers`.

**Training:** `W2`/`W4`/`W16` (spans 5/13/61), `d_model=256` matched to the GRU's hidden size, 3.23M params
each, 300 epochs. **12.6 + 12.9 + 12.7 = 38.2 min total** on the local 5090 (~2.5 s/epoch). They **overfit** —
val bottoms at ~epoch 40 then rises; best-checkpoint selection is doing real work, unlike the GRU. ~60 epochs
would suffice next time.

**RESULTS (N=192 held-out edits, canonical §4 metrics, quality gate passed):**
1. **Quality gate passes.** Next-step RMSE 0.1039 (`W16`) vs GRU 0.1041, noise floor 0.1539. `W16` also beats
   the GRU on val loss (0.02359 vs 0.02362). Like-for-like.
2. **Readability peaks mid-stack**, not at the decoder: position R² 0.60 → 0.81 (middle) → 0.76 (last), GRU
   0.83. Velocity R² at the middle point separates by window — 0.13/0.20/0.32 for W2/W4/W16 — the expected
   mechanism (longer window = more velocity evidence).
3. **"Readable ≠ grabbable" is not a recurrence artifact.** Readout injection is inert at *every* depth and
   *every* window (Edit Index = each model's own unsteered value, fidelity ratio 1.00). It survives to attention.
4. **Transient vs persistent, measured.** Activation edit (readable state): **+0.86 → +0.04** over 14 steps —
   the strongest step-0 edit in the notebook, gone in ~2 steps. History overwrite (carried state):
   +0.63…+0.67 → **+0.27…+0.28**. Unsteered −0.68 → −0.43.
5. **A registered prediction resolved — and NEITHER of us was right.** Sevan predicted a fixed *fraction*
   (≲50% of window); I predicted a fixed *count* (~2–4 frames). These are the endpoints of one scaling law
   `n_sat ∝ span^β` (β=1 Sevan, β=0 me). **Measured β = 0.47** — saturation grows like the **square root** of
   available history: 3/5 frames (60%) at window 2, 4/13 (31%) at window 4, 6/20 (30%) at window 16. *(3-point
   fit — order-of-magnitude only.)* The **crossover** point (Edit Index > 0) is n=1 for every model and is a
   useless discriminator; do not quote it.

**Candidate finding:** *on an architecture whose readable state is not carried, editability is not a property
of the latent at all — it is a property of the observation history.* The single-`h` framing is an
architectural coincidence, not a general fact. This sharpens rather than complicates Sevan's through-line
(*no successful edit is free of dynamics*): here the only channel that persists **is** the history.

**Follow-up same day (Sevan's review):** (a) **readout injection does NOT work on the transformer** — Fig 4
combined both editors and the injection line sat exactly on the unsteered line, hiding the null. Now split
into Fig 4a/4b with landing diagnostics (Table 3): probe error 3.2 → 1e-6, ‖Δh‖/‖h‖ up to 0.15, but
‖Δrender‖/‖render‖ 0.007–0.036 and Edit Index −0.684 → −0.681. A working editor, a null result.
(b) **New result — retention keeps improving after the step-0 index saturates** (Fig 7/Table 4): window 16
lands at n≈6 but retention (step-14 ÷ step-0) climbs 0.37 (n=2) → 0.62 (n=16). **~30% of span to land the
edit, ~the whole span to hold it**; β = 0.47 applies to landing only. (c) The counterfactual history is
**velocity-oracle** — a straight line arriving at the target at the true post-edit velocity (which equals
the pre-edit velocity, since the teleport preserves it exactly). Documented; estimating velocity from
observations instead is a follow-on. (d) **No leakage**: edits seeds 110000+ vs train 0–89999, nearest train
scene L2 0.55 vs median 5.31; and the "drift" is ballistic — direction/speed noise are 0, so 84% of the
post-edit motion is fixed by (position, velocity), which the overwrite explicitly supplies. Added Fig 8 (six
unselected samples) and a predictability audit (Table 5). Notebook now 23 cells, 0 errors, 9 figures.

(e) **§6 added — WHY the injection is inert, as geometry** (Fig 9, Table 6). The pseudoinverse direction lies
in the probe's row space by construction; the decoder's own descent direction
`−∇_h‖decode(h) − gt_obs‖²` is **orthogonal to it** — cosine +0.007…+0.050 (**87–90°**), every value inside
the shuffled-pair control band, per-sample then averaged, N=192. The row-space fraction of the decoder's
direction is **2.31× chance at the middle residual point but 0.57× chance (BELOW chance) at the last**, the
one the decoder reads. That fraction is the **hard ceiling** on any injection-style editor. Matches the GRU
reachability ceiling from `delta_h_analysis` (0.096 vs 0.125 chance) and explains the only non-zero signal in
Table 3 (the middle point perturbs the render most, and is where row-space enrichment peaks).
(f) Waterfalls corrected: both now carry an explicit **pseudoinverse injection** column (the old
"activation edit" column was the decoder-gradient oracle, labelled by *site* not *editor* — that label is
what let the oracle's success read as the injection's). Observation-space confirmation of the null: the
injection columns are indistinguishable from unsteered in all nine samples. **Rule: an editor column must be
named by its editor, never by its edit site.** Notebook now 25 cells, 0 errors, 10 figures.

**Sharpest follow-on:** the **KV-cache view** — carried *and* history-dependent, the true transformer analogue
of a GRU `h` write. `state_view="kv_cache"` already exposes it.

**Harness fixes made in passing:** `SCORECARD_COLUMNS` listed `gt_traj_rmse` twice (duplicate column in every
scorecard table repo-wide); `load_checkpoint` never moved the transformer to the GPU; the
`METRICS_AND_EDITORS.md` "Oracle observation" rename had overwritten its own former name ("true-state swap").
Registry gained the two transformer editors + the **saturation point** metric.

## 2026-08-03 — NEW branch `delta_h_analysis`: what does a *successful* edit look like in latent space?

Sevan's request, discussed and scoped before building. Every §4 negative so far describes what *fails*; this
characterises what **works**. Two mechanisms reliably edit, and both need oracle access — **counterfactual state
overwrite** and **freeze-time teacher forcing** — so they hand us ground truth for `Δh = h_post − h_pre`.
Notebook `notebooks/experiments/editability/delta_h_analysis.ipynb` (17 cells, 0 errors, 5 figs);
brief `directions/delta-h-analysis.md`; note `scratch/2026-08-03-delta-h-analysis.md` (**FLAG FOR PROMOTION**).
N=256 held-out edits (the prior version of this measurement was N=64, GRU only, one construction); GRU + RSSM.

**Sevan's framing, now the thread's through-line:** *no successful edit is free of dynamics.* Every mechanism that
works operates by making the model **consume observations over time**; none writes to `h` directly.

**The framing that makes it a measurement.** Readout injection produces `Δh ∈ row(A)` **by construction**, so the
row-space fraction of Δh_true is the **hard ceiling** on what that editor could ever achieve — and
`‖P_row Δh‖/‖Δh‖` is exactly the best cosine it could reach with the truth.

**RESULT — a successful edit is large, edit-specific, and invisible to the probe.**
1. **Both oracles succeed, and Sevan's prediction held.** Counterfactual **+0.68** (holds to **+0.44** at step 14),
   freeze-time **+0.54** (→ +0.26), vs unsteered −0.67 and readout injection −0.66 (inert). Counterfactual is
   stronger *and* more persistent — as he reasoned, a full overwrite has no pre-edit remnant to revert to.
   (Contrast the decoder-gradient oracle: +0.94 → −0.12, a single-frame success.)
2. **Reachability ceiling — row-space fraction 0.096 (GRU) / 0.005 (RSSM) against a chance level of 0.125 / 0.112.
   Both at or BELOW chance.** A successful edit is *less* aligned with the probe's row space than a random
   direction. Readout injection could match at best ~10% (GRU) / ~0% (RSSM) of the true edit direction — not
   because it is weak, but because it is confined to a 4-dim subspace the edit provably avoids.
3. **Adding velocity to the probe does not help** (0.096 → 0.110 while chance rises 0.125 → 0.177, so relative
   alignment *falls*). The content the edit moves is **not physical state** — consistent with the fiber residual
   (~0.87 of ‖h‖ is not a function of (pos,vel)); the edit lives in that 87%.
4. **The two oracles agree strongly: cos = +0.799 raw / +0.816 edit-only** (shuffled control +0.023, random +0.062).
   Two unrelated constructions land on nearly the same displacement, so "the edit direction" is well-defined.
   Meanwhile cos(oracle, readout injection) = **+0.078** — the failing editor is nearly orthogonal to what works.
5. **Magnitude: ‖Δh‖/‖h0‖ = 0.97** — as large as the entire state; **14×** the injection it replaces (RSSM 275×),
   **3.6×** one ordinary dynamics step.
6. **No shared edit direction across edits:** mean pairwise cosine **+0.011** vs random +0.062 — indistinguishable
   from zero. Every edit has its own direction; magnitude is far more stable than direction (CV 0.28).
7. **Same displacement, different starting states (Sevan's follow-up).** Holding the object's positional change
   fixed (5 canonical δ, n=64 each) and varying everything else raises the mean pairwise cosine from **+0.011 to
   +0.071** (GRU) / +0.008 → +0.084 (RSSM) — a **6.6×/10.3× effect**, so displacement genuinely carries information
   about Δh — **but the absolute level is still ≈0.08**, nowhere near determining it. *There is no displacement→Δh
   lookup table*, which kills the most attractive remaining hypothesis (that readout injection was merely using the
   wrong basis) and explains the memorisation result below. **Sub-finding:** purely *lateral* displacements are
   ~2.5× more consistent than purely *depth* displacements (0.129/0.106 vs 0.050, GRU) — a perspective signature,
   since a sideways move changes which rays are hit but a depth move changes apparent size by a start-dependent
   amount. My sharper prediction (alignment decays with depth *mismatch*) was **not** supported — Fig 6b is flat,
   r ≈ +0.02.
8. **Learning from oracle Δh memorises, cleanly diagnosed:** MLP **train R² 0.951 → held-out R² 0.088**, applied
   Edit Index **+0.01** vs the +0.68 oracle it imitates. Even ground-truth supervision on a working edit does not
   transfer — and that is exactly what (6) predicts.
9. **Probe accuracy does NOT buy reachability — hypothesis refuted.** Across the 8 controls GRUs (probe R²
   0.19→0.87) the enrichment `f/chance` stays at 0.46–0.89× with no trend. *The raw fraction appears to fall
   steeply (0.632 → 0.079) but that is almost entirely the changing chance level* (`√(d/H)` = 0.707 at H=8 vs 0.088
   at H=512) — a bug I caught and fixed mid-build; correcting it removed the apparent effect.

**Reading:** the successful-edit displacement is well-defined per edit, enormous, and lives almost entirely in the
part of the latent no probe over physical state can address. That is the *mechanism* behind "readable ≠
controllable", stated as a measurement rather than an inference, and it predicts both learned results we already
have (amortized editor plateaus; fine-tuning wires a button).

**§7 COMPOSITIONALITY — added at Sevan's request, and it is the thread's strongest POSITIVE.** Sevan asked whether
`Δh_comp = Δh2 − Δh1` is testable; as literally stated it is a **tautology** (pure vector arithmetic on states
defined by subtraction). The non-tautological version requires the second edit to be **constructed independently by
re-running the oracle**. Two tests, with the composed state applied and rolled out:
- **Sequential (path-independence), freeze-time only** — counterfactual overwrite is *vacuous* here since it
  discards history by construction. GRU cos **+0.904**, composed recovers **94%** of the direct edit's Edit-Index
  gain (RSSM +0.742 / 77%). The latent is substantially path-independent.
- **Object superposition** `[move obj0] + [move obj1]` vs `[move both]` — *not* tautological for either mechanism.
  GRU cos +0.873 (counterfactual) / +0.881 (freeze-time), composed recovers **83% / 87%** of the direct gain;
  RSSM +0.815 / 79%. So the configuration→latent map is close to **additively separable across objects**.
**Waterfalls (Fig 8a/8b, 3 random samples, both models)** confirm it in observation space: the composed column
visibly reproduces the direct both-moved column — both objects at their target locators, both ghosts vacated —
with a slight overshoot matching `‖composed‖/‖direct‖ = 1.13`.
**I predicted both would fail; both largely succeed.** Reconciles with §2: residuals are large (0.39–0.69) while the
Edit Index retains 77–94%, i.e. *the part of Δh that matters for the observation composes even though the whole
vector does not*. The latent is structured and additively organised — just not addressable by a position probe.

> **Flaw caught and fixed mid-build:** the first sequential test used the **midpoint** as waypoint, which under
> linear interpolation with matched frame counts makes the two-step route traverse *exactly* the direct route
> (verified: max difference 0.0) — vacuous, and it reported cos +0.979. A 2-unit **perpendicular detour** gives the
> real +0.904. Conclusion survives but is materially weaker; the first version would have overstated it.

**HARNESS updated after Sevan's review of Fig 7 (four new rules, all from real failures this session).**
`CLAUDE.md` notebook-legibility now carries:
1. **One quantity per axis** — a shared axis asserts the bars mean the same thing. Fig 7 put `sequential` and
   `superposition` together, where "composed" meant a *two-stage endpoint* in one and a *literal vector sum* in the
   other, so a cosine in one bar was not the same object as in the next. Test: *could a reader subtract two bars and
   get something meaningful?* Outcome metrics common to both (Edit Index) may stay shared, stated explicitly.
2. **No derived duplicates** — never report a number that is an algebraic function of two already shown.
   `residual² = r² + 1 − 2·r·cos θ`, so the residual panel was fully determined by the cosine and magnitude columns
   and *read as a contradiction* because the identity was invisible. Also added to `METRICS_AND_EDITORS.md` as a
   gate on adding new metrics.
3. **Multi-dimensional comparisons need multi-dimensional labels** — `sequential (freeze-time)` vs
   `superposition (freeze-time)` let the *mechanism* (parenthetical) visually dominate the *test type* (leading
   word), so arms testing different things looked like one family and Sevan read the wrong bar.
4. **High-dimensional intuition must be stated** — cos 0.9 is a **26° angle** (differing by ~0.45 of the length);
   the mean cosine of random vectors is **0**, not `1/√H` (that is the per-pair sd); and a random vector already
   holds `√(d/H)` of its norm in any d-dim subspace, so plot **enrichment** not raw fraction when `H` varies.
   All three bit this session.

**A framing correction made mid-build:** `1/√H` is the **per-pair standard deviation** of the cosine between random
vectors, *not* a baseline the **mean** should sit at (that is 0). I had been quoting it as if it were a floor for
the mean; the empirical across-displacement / shuffled-pair control is the right reference for a mean, and the
notebook and note now say so.

**Caveats:** RSSM ±1 alignment is ambiguous at measurement precision (k=−1 0.1059 vs k=0 0.1067, 0.8%) because its
prior decode is blurry — its Δh numbers carry that. Freeze-time is far weaker on RSSM (+0.09) than GRU (+0.54),
uninvestigated. Row-space ceiling applies to *linear* probes only. One checkpoint per architecture.

**Also this session:** confirmed for Sevan that the interleaved observe-and-settle steering in `multistep_steering`
§1a was only ever run with the **vanilla pseudoinverse** (plus one `+manifold` variant) — PCA geodesic, MLP-probe
gradient and decoder gradient were never put through that loop. The nearest existing coverage is
`controls/encoder_editing.ipynb`, which runs the multi-step self-observation idea across four editors but at the
**encoder port**, not in `h`-space. That gap (interleaving the other editors in `h`-space, on the canonical
metrics) is still open.

## 2026-07-30 (night) — NEW branch `more_trained_editability`: can editability be INDUCED BY TRAINING?

Sevan asked for a much more extensive test of **trained** editability, including fine-tuning the world model to
induce it. New topical dir `notebooks/experiments/editability/trained_editability/` with `learn_to_edit.ipynb`
**moved into it** (paths repointed) and a new notebook alongside. Registry `TRAINED_EDITABILITY_RUNS.md`; brief
`directions/trained-editability.md`; note `scratch/2026-07-30-trained-editability.md` (**FLAG FOR PROMOTION**).
**This pays the "heavier fine-tune still OWED" debt** that has sat in `METRICS_AND_EDITORS.md` since `learn_to_edit`.

**New: `scripts/train_editable_gru.py`.** Two mechanisms, one evaluation, 5 arms, all from `runs/controls/H256`:
- **fine-tune the world model** so a **fixed, frozen** readout-injection probe works — nothing about the editor is
  learned, so all adaptation is in the model, which must learn to honour writes along `A⁺`. Loss
  `edit + λ·retention`, where retention is ordinary next-step prediction; λ is what separates "became editable"
  from "was destroyed and now renders whatever it is asked for".
- **amortized editor** `E_θ(h,target)→Δh` against a **frozen** world model.
Arms: `FT_light` (300 steps) · `FT_heavy` (3000) · `FT_heavy_noret` (λ=0) · `FT_heavy_obj0` (object-0 edits only,
the content control) · `AMORT`. Trained on `edits[2000:]`; **everything reported on the held-out `edits[:64]`**, the
same samples the `controls/` notebooks use. `eval_controls.py` gained `--root` so the identical §4 suite scores them.

**RESULT — training moves the edit, and wires a BUTTON.** Δ Edit Index of the *trained interface* vs each arm's own
unsteered: base **+0.01**, light **+0.04**, heavy **+0.13**, no-retention +0.10, object-0-only +0.10,
**amortized +0.54**.
- **The light-budget negative was partly a budget artifact** (+0.04 → +0.13 from 300 → 3000 steps) — worth knowing,
  since that negative is currently cited in `findings/editability.md`.
- **But even the best arm only reaches "equidistant".** Amortized absolute Edit Index **−0.14**, against its own
  unsteered −0.68 and the decoder-gradient oracle's +0.94. It never arrives at the edited world.
- **No mechanism generalisation.** The *same* mechanism with a freshly-fit probe moves only **+0.01…+0.04** on every
  arm; the other standard editors are unmoved. The model obeys the interface it was trained for, not the class.
- **No content generalisation.** `FT_heavy_obj0` has an obj1−obj0 gap of **−0.08** vs the both-objects control's
  **+0.09** — withholding an object costs ≈0.17 index points. A per-object button.
- **Cost.** Fine-tuning costs 13% of next-step prediction even with retention (0.1041 → 0.1173). Without retention
  it degrades to **0.1486**, essentially the observation noise floor (0.1539) — the world model is destroyed —
  **and editing gets no better**.
- **The cleanest new fact is an asymmetry:** learning a bespoke editor for a *frozen* latent (+0.54, zero cost to
  the model) works far better than making the latent obey a *fixed* editor (+0.13, 13% prediction cost). Consistent
  with the standing reading that the obstacle is the **reachability of the edit map**, not the representation.

**Also answered (Sevan's question about the Edit Index).** Why is unsteered ≈ −0.7 rather than −1 for `H256`?
Computed: the references are **clean** renders (not noisy), and the index is evaluated **only on the differing
rays**, so shared background is already excluded (`d_edited` = 0.547 there, near full object contrast). The gap is
entirely the model's own blur — `d_unedited` = 0.090, which is its one-step prediction error rather than 0.
Decisive detail: **the split is by observation noise, not model quality.** At obs noise 0.2 the index *saturates at
−0.72* from H=128 on (H128/256/512 all have `d_unedited` ≈ 0.09 — the noise-limited floor), while the noise-free
models reach −0.84. So `H256` is at its best achievable value; more capacity cannot lower it. Matching the *true*
unedited world still gives exactly −1.0 (asserted).

**Owed / next:** train-from-scratch with an edit objective in the loss is the one version of "train for editability"
this does not cover; RSSM untested; one seed per arm.

## 2026-07-30 (latest) — Sevan's review: a real rendering bug, a rename, and the Edit Index over the rollout

**BUG (mine, and it came from a stale spec): every controls waterfall painted a shared teacher-forced `ef` row
across ALL columns.** Only the **Oracle observation** reference actually sees that frame, so every other column
looked teacher-forced when it wasn't — and it **hid the exact frame the §4 scorecard scores** (step 0). It also
displayed the *clean* render while the model that legitimately sees that frame is fed the **noisy** `edits.obs[ef]`.
Root cause: `CLAUDE.md`'s waterfall spec *mandated* the shared row — but that convention had already been caught and
removed in `eval_editability_endogenous.py` v2 (2026-07-28) for exactly this reason. I followed the stale spec.
**Fixed:** every column now shows its own free-run from step 0, GT column = `clean_obs[ef:ef+K]`, and the
`CLAUDE.md` spec now carries an explicit **⛔ never paint a shared teacher-forced row** block explaining why.

**RENAMED `True-state swap` → `Oracle observation`** (Sevan: the old name doesn't match what it does). It is not a
state swap — the model is teacher-forced **one extra frame**, the real **noisy** `edits.obs[ef]`, i.e. it simply
gets to *see* the teleport. Renamed in the registry, master notebook, controls notebooks, eval script and this
thread's notes; historical notes on the retired metric scale were left alone.

**Confirmed the metric is computed correctly (Sevan asked why unsteered isn't −1).** It is correct, and the offset
is interpretable: `d_unedited` is the model's **own one-step prediction error**, not 0, so a perfect predictor
would score exactly −1 and a real one falls short by its blur. Boundary controls asserted: scoring `gt_unedited`
returns exactly **−1.0**, scoring `gt_edited` exactly **+1.0**. And across the 8 controls models the unsteered
index tracks next-step RMSE with **Pearson r = +0.987** (−0.85 for the best predictor, −0.52 for the worst) — so
the unsteered row is effectively a readout of predictive quality and **must appear in every table**. Also verified
the counterfactual render is built on the right frame: velocity is constant in this sim
(`velocities[ef-1] == velocities[ef]`), so there is no off-by-one; the residual is one step of position diffusion.

**NEW — Edit Index over the whole rollout (Sevan's suggestion, and it pays off immediately).** Added
`edit_index_by_step` to `scripts/editability_metrics.py`: the counterfactual world is now rendered **forward**
(edited object continuing along its own velocity, other object on its true trajectory), so the bounded index can be
evaluated at every step. Sevan's prediction was right — **the decoder-gradient oracle's success is a single-frame
success**: on `H=256` it scores **+0.94 at step 0, +0.15 by step 5, −0.12 by step 14**, i.e. it decays past
"neither world". A step-0 scorecard alone would have called that a clean win. New **Fig 3b** in all three controls
notebooks plots it; **GT-traj RMSE** was also added as a panel to Fig 3 as requested.

**Plot fixes:** noise-ablation Fig 3 now uses rotated (tilted) editor labels like the other notebooks, and the
`1.0` reference line was removed from its RMSE panels — 1.0 is not a meaningful level for an RMSE (it was a
leftover intuition from the percentage metrics). It is kept **only** in the hidden-size sweep, where editors
actually cross it and it marks a real threshold (observation intensity is bounded in [0,1], so RMSE > 1 means the
scan was pushed out of range) — flagging that as a judgement call to overrule if you'd rather it go everywhere.

All re-run: `00_master_editability` (0 errors, 11 figs) and the three controls notebooks (0 errors, 6–7 figs each).

## 2026-07-30 (later) — §4 EDITABILITY METRICS REDESIGNED; master + controls re-run on the new set

**Sevan asked to replace `reach %` / `collateral %` with plain RMSE-vs-GT, which surfaced a deeper problem, and we
designed the replacement together before implementing.** The old §4 metrics measured **change away from the
unsteered rollout**, normalised by the oracle observation. Two fatal flaws, both visible in this thread's own data:
(1) they scored *change*, not *correctness* — a scrambling editor posted `reach` of **400–440%** at `H=8`/`H=32` and
the decoder-gradient oracle posted 209–327%, where 100% was supposed to be the ceiling; (2) the denominator was a
**soft, model-dependent** reference whose own strength varied widely (swap ghost ratio 0.315–0.868 across the noise
cells), so the same physical edit scored differently on different models — fatal for cross-model sweeps.
Sevan also noted `selectivity` becomes meaningless once both terms are errors, and that ghost ratio is really just
a zone-restricted RMSE.

**THE NEW CANONICAL SET** — prose in `notebooks/experiments/editability/METRICS_AND_EDITORS.md` §4, implemented
**once** in **`scripts/editability_metrics.py`** (imported everywhere, never re-derived; that drift is what produced
five incompatible versions of "reach"):
- **Layer 1 — absolute error vs ground truth, decomposed by ray zone**, all at rollout step 0, all lower-is-better,
  no normalisation: **Target RMSE** / **Ghost RMSE** / **Collateral RMSE** / **Edit-frame RMSE**, plus **GT-traj
  RMSE** over the rollout and the **fidelity ratio** (`GT-traj RMSE(editor)/GT-traj RMSE(unsteered)`; > 1 = the edit
  left the rollout further from the truth than doing nothing).
- **Layer 2 — the Edit Index ∈ [−1,+1]**, the calibrated headline. Both ground-truth worlds at the edit frame are
  *rendered*: `gt_edited` (the teleport happened) and `gt_unedited` (the counterfactual where it did not — the
  edited object continued along its own velocity). On the rays where they differ,
  `(d_uned − d_edit)/(d_uned + d_edit)`: **+1** = the output *is* the edited world, **−1** = the unedited world,
  **0** = equidistant. **It cannot be gamed by destroying the output** — garbage is far from both worlds and scores
  ≈ 0. "Dim everything toward background" also cancels (the differing rays include target rays as well as ghost
  rays). And the repo's dominant failure — *paint a copy at the target while keeping the ghost* — correctly reads
  ≈ 0 where the old reach reported >100%.

**Everything re-run on the new set: `00_master_editability.ipynb` (0 errors, 11 figs) and all three
`controls/` notebooks (0 errors).** The eval script re-ran across all 8 checkpoints. Retired-metric numbers
anywhere in the repo are flagged as not comparable.

**The redesign sharpened, and in one case corrected, the readings:**
- **Master.** Readout injection now reads in one line: readout RMSE **0.000** (the probe reads the target
  *exactly*) with Edit Index **−0.66** (GRU) / **−0.64** (RSSM) — indistinguishable from doing nothing. Unsteered
  −0.68/−0.64; no probe-directed editor escapes the unedited end (−0.50 to −0.66); oracle **+0.97/+0.87**. The
  "Current results" block was rewritten and re-dated.
- **Hidden-size sweep — the old metric had inverted the low-capacity reading.** At `H=8`/`H=32` the structural
  editors' zone RMSEs exceed **1.0** (intensity is bounded in [0,1]) with fidelity up to 2.2× — they destroy the
  observation. The Edit Index scores that ≈ 0 ("neither world"), not 400%. At `H ≥ 128` structural editors sit
  within 0.08 of unsteered while the oracle reaches +0.87…+0.99. New clean trend: **the oracle's Edit Index rises
  monotonically with capacity (+0.58 → +0.99)** — a bigger latent makes the target state more precisely reachable
  by decoder optimisation, though not by probe-directed writes.
- **Noise ablation — conclusion unchanged, plus a new incidental result.** Structural editors −0.63…−0.67 vs oracle
  +0.91…+0.97 in all four cells. New: **belief inertia is governed by sensing noise** — the oracle observation (no
  editing at all, just one frame of real evidence) reaches Edit Index **+0.54** with clean observations but
  **−0.40** with sensing noise. Suggestively the world-noise-only cell accepts the jump furthest, as if training on
  a jittery world loosens the prior over motion. Flagged as n=1-per-cell, worth a dedicated test.
- **Encoder editing — headline softened and made precise.** Hidden-state injection **−0.67** (1% of the achievable
  span) vs the same pseudoinverse at the encoder port **−0.43** (21%); best probe-directed **−0.08** (50%); render
  oracle **+0.52**. So the interface genuinely matters — but no probe-directed write crosses to the edited side,
  and the best one **triples the collateral error** (0.127 → 0.335) with fidelity 1.15: it repaints rather than
  relocates, exactly as Fig 6's intermediates show.

**Also fixed: `notebooks/experiments/controls/` was created as a SIBLING of `editability/` — wrong.** Migrated to
`notebooks/experiments/editability/controls/`. `CLAUDE.md` now carries the protocol: every experiment lives inside
the research thread it serves (controls/ablations/side-quests as subdirectories), never beside it.

> **Note on mechanics:** the master notebook could not be opened with the `Read` tool (54k tokens, over the cap), so
> its cells were patched via a JSON round-trip with exact-match assertions rather than `NotebookEdit`, then verified
> by AST-parsing every cell and executing the notebook end to end. Flagging because it deviates from the standing
> "never touch .ipynb outside NotebookEdit" rule.

## 2026-07-30 — branch `michael_controls`: Michael's three controls, all COMPLETE

Sevan relayed three control/side experiments from a conversation with his postdoc Michael. All three are **built,
trained, evaluated and written up**; all three scratch notes are **FLAGGED FOR PROMOTION** and awaiting Sevan's
artifact-or-signal call. Notebooks + registry live in `notebooks/experiments/editability/controls/`
(`CONTROL_RUNS.md`). **Uncommitted** (held per commit-only-when-asked).

**ENABLING INFRASTRUCTURE (this is what made the thread runnable).** GRU training was **CPU-bound on gzip HDF5**:
68 s/epoch at H=256 with the GPU idle → 7.5 h per 400-epoch run, and this thread needs 8 runs. Added
`build_inmemory_dataloaders` + `InMemoryLoader` (`pim/world_models/dataloader.py`) and a `--in-memory` flag to
`scripts/train_gru.py`: the observation tensor (1.8 GB) lives on the GPU, identical split/batching/optimizer.
**0.50 s/epoch — 136× faster**, loss curves matching the lazy path (epoch-2 train loss 0.0267 both). 400 epochs is now
~3.5 min. Also new: `scripts/eval_controls.py`, one pass per checkpoint computing all four affordance families
(predictive / recoverability / canonicality / editability) into `runs/controls/eval/<code>.json` + `_rollouts.npz`, so
the notebooks only load, plot and tabulate — which is how they stayed short.

**3 new datasets** (matched to `4_fixed_refl_inview` except the noise flags): `9_obsnoise0_posnoise0`,
`10_obsnoise0_posnoise004`, `11_obsnoise02_posnoise0`. **8 new runs** in `runs/controls/`.

### D1 — encoder-space editing (`directions/encoder-space-editing.md`, `encoder_editing.ipynb`)
Michael's premise: all world information enters the latent through one channel, `x_t = relu(W_enc·obs_t + b_enc)`; every
editor so far writes to `h` instead. So probe `x`, edit `x`, and spread the write over `N` frozen steps the way
freeze-time teacher forcing does — but **without the renderer**.
**RESULT — the interface is a real variable, but the write still repaints rather than relocates.** The *same* linear
pseudoinverse edit is inert on `h` (ghost **0.996**) and moves the needle at `x` (ghost **0.803**); the best
probe-directed encoder write reaches **0.670**, i.e. 27–45% of the way from unsteered to the render oracle. Spreading
helps (0.838 at N=1 → 0.650 at N=12). **But every probe-directed encoder write fails the fidelity guard** (GT-traj RMSE
1.15× unsteered) while the **freeze-time render oracle through the identical port passes** (ghost 0.266, fidelity 0.72).
**Fig 6 (the intermediate decoded observations, Sevan's explicit ask) is the money panel:** the oracle shows one
coherent object *translating*; the probe-directed write shows a *cross-fade* — a new blob brightening while the old one
stays. Also confirmed exactly as predicted: **velocity R² at the port is 0.005 vs 0.474 at `h`** — the encoder output
has no memory, so there is no velocity there to write.

### D2 — hidden-size sweep (`directions/hidden-size-sweep.md`, `hidden_size_sweep.ipynb`)
`H ∈ {8, 32, 128, 256, 512}`, one variable, dataset 4. (`H=8` = the world's true state dimensionality; `H=128` = the
observation resolution.)
**RESULT — capacity moves prediction and readability a lot, grabbability not at all.** Prediction saturates by `H=128`
(next-step RMSE 0.1495 → 0.1167 → 0.1054 → 0.1041 → 0.1042). Linear readability rises **monotonically** — position R²
0.175 → 0.855, velocity 0.002 → 0.531 — **refuting my pre-registered guess** that a squeezed latent would be more
linearly readable; it simply fails to represent the state. Canonicality moves the *opposite* way (MLP fiber residual
0.215 → 0.601), so capacity trades canonicality for readability. §4 numbers restated on the canonical metric set —
see the section above.

### D3 — noise ablation (`directions/noise-ablation.md`, `noise_ablation.ipynb`)
The 2×2 of observation noise (sensing) × position noise (the world itself), at `H=256`.
**RESULT — neither noise source is what blocks editing**; the negative holds in the fully deterministic,
perfectly-sensed world (§4 numbers restated on the canonical set above). **Both pre-registered
recoverability predictions refuted**, and one clean positive: **observation noise is a linearising regulariser** —
position R² (linear) 0.596 → 0.819 when sensing noise is turned on. Velocity readability is invariant to both sources
(0.451–0.471 across the whole 2×2). Canonicality: the linear and MLP fiber estimators **disagree in sign** (sensing
noise off moves linear −0.026 but MLP **+0.193**), so both are reported — reporting only the linear one would have
produced the opposite headline.

### METHODOLOGICAL ADDITION — the fidelity guard (now part of the canonical set)
Ghost ratio alone is **not sufficient** and can invert a conclusion: at `H=8`/`H=32` structural editors reported
"good" ghost values while their GT-traj RMSE was up to **2.2×** unsteered — the edit destroys the observation and the
vacated rays dim as a side effect. The **fidelity ratio** was introduced here and is now part of the canonical §4 set;
finding it is what led to the full metric redesign recorded in the section above.

**Awaiting Sevan:** artifact-or-signal + promotion call on all three notes; whether D2+D3 merge into one "the §4
negative is robust to capacity and to stochasticity" findings entry; whether to re-run the full §4 editor line-up at the
encoder port on the `object-individuation` models; whether to fold the fidelity guard into the metrics registry.

## 2026-07-28 — Waterfall cleanup tied off; NEW experiment thread opened (endogenous actions)
**Waterfall honesty pass DONE (source edits only, no full re-runs per Sevan).** Added the **teacher-forced edit-frame
row** to every relevant waterfall — one shared row = the TRUE post-edit obs / edit target (`clean_obs[ef]`, identical
across columns), each column's model rollout below it = its **free-run from `ef+1`**. Applied to
`actions/action_space_object_individuation.ipynb` (Fig 5), `actions/action_conditioned_structure.ipynb` (Fig 5), and
converted `learn_to_edit.ipynb` (Fig 3b/5d) from `magma`→gray + added context frames + figure-top legend (kept GT
column). **Caught + fixed a real off-by-one** I'd introduced pre-compaction: `warm_up_to_edit` teacher-forces
`obs[0..ef-1]` so the predict-next GRU rollout **step-0 ↔ `clean_obs[ef]`** (confirmed by the §4 scorecard's
`gt_traj_obs=clean_obs[ef:]`); the shared true-`ef` row therefore requires **dropping each model column's step-0**
(`ROLL[...][1:]`) and slicing GT to `clean_obs[ef+1:ef+K]` (both length `K-1`) so columns align — earlier version left
GT `ef+1`-aligned while model was `ef`-aligned. All 4 cells AST-parse + shape-check clean; the 4 edited waterfall cells
have outputs cleared (need a re-run to regenerate — the stale images contradicted the new code). **Harness locked:**
CLAUDE.md waterfall spec + `editability/METRICS_AND_EDITORS.md` now make the shared true-`ef` row mandatory with the
exact alignment rule. Also fixed the E2 `render_scene` NameError.

**NEW THREAD (proposed, design-only so far): endogenous-action interactive world** →
`directions/endogenous-action-interactive-world.md`. The promised follow-up to object-individuation (which was scoped
to EXOGENOUS actions and left endogenous open). Hypothesis (strong enactivist): actions must be **generated by the
latent world and self-predicted** (closed sensorimotor loop / efference copy), not merely observed, to induce a
factored/editable latent. Central control = **actor** (action-OUTPUT head, acts on the world) vs **observer** (identical
arch, action-INPUT, same obs+actions, never acts) — actor-yes/observer-no would isolate the effect to agency. Three
levels: L1 random position-shift, L2 force/momentum, **L3 goals** (collision/wall avoidance — the anticipated payoff).
Key design reality surfaced: the sim is non-differentiable, so **L1/L2 give no learning signal to the action head
(efference-copy ablation only); L3 needs a policy-learning loop (REINFORCE / Dreamer-in-imagination)** — this is why L3
is where agency actually enters. First concrete build (after discussion): a **stateful `InteractiveWorld.step()`** (the
one genuinely new primitive — everything today is offline `simulate()`) + a **keyboard `play.py` emulator** (2D ∥
waterfall, key overlay) that a human plays now and the trained model drives later through the SAME view. Design doc
addresses Sevan's A–D + additions E–I (embodied-vs-god-hand agent, degeneracy/anti-freeze, discrete-vs-continuous,
on-policy nonstationarity, efference-copy sanity probe).

**Discussion round 2 done + decisions locked** (in the direction doc's "Decisions locked" block): god's-hand first
(embodied later), GRU-only first, discrete keys, action head decodes from `h_t`, no action-input on the actor except
the efference copy of its own *sampled* action (needed once RL makes the policy stochastic), start REINFORCE then the
SMiRL-style survival-from-unpredictable-death variant, empowerment steered away from (circular / teaching-to-the-test).
Clarified the "dark room problem" (prediction alone rewards boredom → freeze/no-op, so L1/L2 are efference-copy
ablations and L3 needs a policy objective).

**BUILD 1 DONE + VALIDATED (2026-07-28, branch `endogenous_actions`): the interactive sim + keyboard emulator
(human-playable; no models yet).** New files, all reuse `pim/simulator` and touch no existing path:
- `pim/simulator/interactive.py` — `InteractiveWorld` (the one genuinely new primitive: stateful `reset`/`step`; there
  was no online world before) + `InteractiveConfig`. Two dynamics modes: **`shift`** (L1 position-delta, drift base,
  frustum/collision-guarded) and **`force`** (L2 F/m momentum, intrinsic anti-freeze drift, friction, speed clamp,
  bounce/clamp/death walls). God's-hand per-object actions `(n,2)` (also accepts the `(n,3)` `[active,a1,a2]` model
  schema). **Death → rebirth** built in (reset to fresh IC, optional pure-noise frames = the SMiRL substrate). Key
  design property (tested): deaths are a **force-mode** phenomenon — the `shift` guard makes collisions/frustum-exit
  impossible by action (matches the prior collision-free datasets), so the L3 avoidance game lives in force mode.
- `scripts/play.py` — the emulator: `Driver` protocol (`HumanKeyboardDriver` via WASD=obj0 / arrows(or IJKL)=obj1;
  `RandomDriver`; `HeuristicAvoidDriver`), a 2D-world ∥ grayscale-waterfall dual panel + **keyboard overlay** (pressed
  keys highlighted; a model/driver's continuous action is discretised back onto the same keys — the "see what it's
  doing as key-presses" feature) + status (frame/deaths/survived). Live `plt.show()` loop; headless `--save` GIF path.
  Reuses viz.py's frustum/waterfall style. Toggle dynamics live with `M`, reset `R`, pause `SPACE`.
- `tests/test_interactive.py` — 12 tests (both modes, determinism, bounce-containment, shift-moves/force-accelerates,
  `(n,3)` coercion, shift-guard-prevents-collision, force death+rebirth). **Full suite 43 passed; ruff clean; black
  formatted.** Validated the render pipeline headlessly (Agg) → demo GIFs for force+avoid and shift+random look correct
  (frustum, discs+reflectivity labels, action arrows, gray waterfall bands, key overlay). Live keyboard path is
  untested here (no display) — Sevan runs `python scripts/play.py` locally to play.
**BUILD 1 fixes (2026-07-28, after Sevan playtested):** (1) matplotlib keymaps cleared so `s`/etc. no longer trigger
toolbar actions (the save dialog was eating key-releases → stuck keys); (2) `--death-on-collision` now defaults ON so
deaths register; (3) collision threshold fixed from the offline generator's `collision_margin·2r=1.6` to true disc
contact `2·radius=1.0` (added `collision_slack`/`spawn_clearance`, split `_contact` vs `_spawn_sep`); (4) `M` now
toggles dynamics IN PLACE (keeps positions) instead of calling reset; (5) wall-death decoupled from bounce — walls
always bounce, `death_on_wall` is an independent toggle (`--death-on-wall` / live `B`; `C` toggles collision-death).
2 new tests (contact-distance, wall-death); **45 passed, ruff/black clean;** re-validated headlessly (deaths increment,
M in-place, `s` freed, C/B toggle). Sevan's verdict: "simulator looks good and ready to go."

**BUILD 2 DONE + RUN COMPLETE (2026-07-28 overnight, branch `endogenous_actions`): actor-vs-observer L1→L3 trained +
evaluated.** Sevan dispatched with two difficulty tweaks (wall-death on, `init_speed=0.28` momentum). New:
`pim/world_models/actor_gru.py` (`EndogenousActorGRU` = obs-only-encoder GRU + categorical policy head {−1,0,+1}/axis +
value head + action-conditioned decoder; HiddenStateModel-conformant passive no-op decode — the OBSERVER is the same
class fed the actor's actions); `scripts/train_endogenous.py` (batched on-policy rollout in `InteractiveWorld`;
predictor loss for actor+observer; **REINFORCE+value baseline into the actor's SHARED trunk** at L3 — the mechanism
under test); `scripts/eval_endogenous.py`. Runs `runs/endogenous/{L1,L2,L3,L3b}` (L3b=seed 1), ~55 min GPU, launched
detached + watcher (fired clean). Notebook `notebooks/experiments/editability/actions/endogenous_actor_observer.ipynb`
(0 err, 4 figs); scratch `2026-07-28-endogenous-action-actor-observer.md` (**FLAG FOR PROMOTION**).

**RESULT — clean positive on identifiability, localized to GOAL-DIRECTED agency:** L3 actor learned the survival goal
(survival 12→~1536–3072, deaths ~250→0–2, reward −0.03→+0.10; both seeds). **Passive-latent recoverability: L3 actor ≫
observer — pos R² lin 0.76 vs 0.59 (Δ+0.17), vel R² lin 0.56 vs 0.39 (Δ+0.17), replicated (L3b Δ+0.14/+0.17).** L1
(shift, no goal) actor≡observer (Δ≈0.00); L2 (force, no goal) actor marginally worse (Δ≈−0.01) → it is **not**
self-generating actions nor momentum, but **acting toward a goal** (policy grad into the shared trunk) that reshapes the
latent. Velocity (historically hard-to-read) gains most — collision-avoidance forces motion-tracking. Gain is
**legibility, not prediction/canonicality**: L3 actor is a slightly worse predictor (next-step RMSE 0.131 vs 0.109) and
LESS canonical (fiber MLP 0.40 vs 0.34 — carries extra control state). The **observer is a strong control** (same
obs+actions, no agency → no gain) = the enactivist prediction; extends object-individuation's "readable ≠ grabbable"
with a big *readability* gain from agency.

**§4 GRABBABILITY FOLLOW-UP (same night, Sevan asked): is the more-readable L3 latent an editable object HANDLE? Mostly
NO** (`scripts/eval_editability_endogenous.py`; passive latent, foreign latent-surgery editors → object-0 teleport
target; N=64; `editability_metrics.json` + waterfalls `runs/endogenous/edit_figs/`). The genuine structural editor
(MLP-probe gradient) reaches **2× further on the actor than the observer (75–83% vs 35–45%)** — readability *does* buy
obs-space reach — **BUT the object-handle hallmarks fail for both:** ghost **0.91–1.16** (vs oracle 0.01, true-swap ≈0 —
the object never *leaves* its old spot) and non-selective (collateral ~100%, selectivity ~0.45 — drags the other
object). Waterfalls confirm the reach is *painting a copy at the target while keeping the ghost*, not moving the object;
readout injection is inert; only true-swap + the off-manifold decoder-gradient oracle move it cleanly. **Verdict:
agency buys legibility + steerability, NOT a clean grabbable handle — "readable ≠ grabbable" holds under endogenous
goal-directed action, sharpened.** Keeps pointing at explicit object scaffolding (RESEARCH.md endgame). Note updated
(nuanced FLAG FOR PROMOTION). **Still owed:** action-interface controllability (edit *through* the trained action
channel); non-action auxiliary-task control; embodied; RSSM. **Uncommitted** (branch `endogenous_actions`, held per
commit-only-when-asked). **Awaiting Sevan:** artifact-or-signal + promotion call; next-move pick.

## 2026-07-28 (late) — IN FLIGHT: stronger-predictor rerun of the §4 grabbability test
Sevan reviewed the §4 negative and pushed back on two things — **(i) a waterfall bug** and **(ii) "the predictions are
so messy it's hard to tell whether it's really failing or just a bad predictor."** Both were legitimate:
- **Waterfall bug (FIXED).** v1 injected the TRUE target-obs row into *every* column and dropped each editor's own
  step-0 decode (`ROLL[...][1:]`), so every column looked teacher-forced on the edit frame **and the exact frame the
  scorecard scores was hidden**. Only True-swap legitimately sees that frame. v2: each column shows **its own free-run
  from step 0**, GT is its own column.
- **Predictor quality was genuinely poor (CONFIRMED).** Measured: weak-model free-run RMSE **0.24**, sharpness
  **TV ratio 0.59** (only ~60% of GT sharpness). A new **quality gate** (free-run RMSE + TV ratio + next-step) now runs
  for every model, so editability is only interpreted for models that pass.
- **Off-distribution rollout (checked, NOT the driver).** v1 rolled out with no-op actions though the actor always
  acts; v2 adds a `self` mode (model's own policy acts on its imagined world). Results are near-identical to `noop`.
- **Editor line-up widened:** + Global-PCA projection, + PCA geodesic (reusing `pim/editors/manifold_steering`).
- **NEW action-channel control (informative already):** a PD controller in the REAL sim closes **94%** of the distance
  to the target (the channel genuinely has authority), but the weak model's *imagination* of those same actions barely
  moves the object (imagined reach **2.1%**, ghost **0.987**; model-vs-real RMSE 0.29) → the weak model cannot even
  simulate its own action channel off-policy, i.e. its editability failure IS partly a predictor failure. This is the
  cleanest evidence that Sevan's objection was right and the v1 verdict must be re-tested at higher model quality.
- **Stronger models TRAINING (detached + watcher):** `runs/endogenous/{L3s0,L3s1}` (L3, seeds 0/1) and `L2s0` —
  hidden **512**, **2-layer MLP encoder + residual MLP decoder** (added to `EndogenousActorConfig` as `enc_layers` /
  `dec_layers`, defaults preserve the old architecture so **old checkpoints still load strictly**), a **5-step free-run
  (multistep) objective** to fight rollout blur, 25k iters. Early signal: at it 1000 the strong model already matches
  the weak model's *final* prediction RMSE and reaches survival 768.
- Built + validated: `scripts/eval_editability_endogenous.py` (v2, all of the above) and the comparison notebook
  `notebooks/experiments/editability/actions/endogenous_grabbability.ipynb` (9 cells, valid, 0 syntax errors).
**COMPLETE (2026-07-29 00:05).** All 3 strong runs trained (`L3s0`,`L3s1` 25k it; `L2s0` 12k it), both evals re-run
across all 7 checkpoints, notebook `endogenous_grabbability.ipynb` executed (0 err, 7 figs), scratch note revised.

**RESULT 1 — §4 grabbability CONFIRMED, and now NOT a predictor artifact (the control the first pass lacked).**
Structural editors are inert on the strong models: ghost **0.998–1.010** (1.0 = the object never leaves), reach 0.3–6%.
But on the **same model / decoder / rollout**, the **decoder-gradient oracle** (ghost 0.004–0.012, reach 89–93%) and
the **oracle observation** (ghost ≈ 0, reach 100%) succeed completely. **If blur caused the failure the oracle would fail
too** → a state rendering the target exists and rolls out fine; probe-directed writes cannot reach it. Failure = the
**edit map's reachability**, not the predictor. Replicated 2 seeds × 2 rollout modes. Counter-intuitively the editors
got *more* inert as the predictor improved (PCA geodesic reach 28% → 4%).

**RESULT 2 — the 2026-07-28 identifiability headline is DOWNGRADED (do not cite the old magnitudes).** Δ(actor−observer)
position R² **+0.155 → +0.017** at strength — **identical to the no-goal control (+0.018)**, so the position advantage is
no longer goal-specific at all (the observer catches up, 0.589 → 0.863). Velocity survives but ~3× smaller (**+0.052**
vs control −0.015); canonicality **flips sign** to a cleaner positive (fiber MLP **−0.074**, actor now *more* canonical;
control −0.026). Revised reading: goal-directed agency mainly **accelerates** the emergence of readable structure; what
durably survives is a modest velocity-readability + canonicality gain.

**Honest limitations found:** (a) the stronger models did **not** fix the blur (sharpness 0.607 → 0.633; free-run RMSE
slightly worse) — capacity + multistep were insufficient; (b) the **action-channel control is not a clean "button"
result** — the real sim closes 93–95% of the distance but the model's imagination of those (OFF-POLICY) actions barely
moves the object (reach 2–6%), conflating "doesn't transfer to the state" with "poor off-policy generalization". The
earlier "button, not a handle" phrasing **overclaimed and is retracted** pending an on-policy action-intervention test.
Corrected framing: the model is an **on-policy predictor, not an intervention-supporting simulator** — no tested
intervention route works in imagination except decoder optimization and fresh observational evidence.

**Awaiting Sevan:** artifact-or-signal + promotion call on the revised note; next-move pick (on-policy action-
intervention test / non-action auxiliary-task control / embodied / RSSM / go constructive with explicit scaffolding).

## 2026-07-29 — Sevan's notebook review (12 items): legibility fixes, metric corrections, animations
Sevan reviewed both endogenous notebooks. Two **methodological corrections**, one **harness fix for a recurring
failure**, and a new qualitative notebook.

**HARNESS (recurring failure — Sevan: "you are still reintroducing terms and inconsistent idiosyncratic naming
conventions that I can't follow").** Added a hard `CLAUDE.md` rule: every experiment thread keeps a **canonical run
registry**; every notebook copies the rows it uses into its own definitions table; **figures use descriptive labels,
never bare codes** (`L3 force+goal · strong · seed 0`, not `L3s0`); a suffix encoding a variable must state what it
encodes; adding a run means adding its registry row in the same commit. Created the first registry:
`notebooks/experiments/editability/actions/ENDOGENOUS_RUNS.md` (every run + role + level + architecture + seed +
purpose, plus the metric caveats below). Both notebooks now carry full definitions tables.

**METRIC CORRECTIONS (both were real):**
- **`survival` is capped + quantized at 3072** = `batch·rollout / max(deaths,1)`, i.e. bounded by the **measurement
  window** (64×48 frames/iteration), NOT by the world (episodes are unbounded; only death ends one). 0 or 1 deaths both
  read 3072. This is why the curve looked spiky/saturated. Fig 1 now leads with **deaths per 1000 frames** (unbounded,
  linear) and marks the 3072 cap explicitly on the survival panel.
- **`mean reward` is per STEP, not per episode** (+0.1 survive / −1.0 death) so **+0.1 is the ceiling**, not "survives a
  few frames". Documented, with the return-scale note (γ=0.99 ⇒ survival stream ≈ 10, so death ≈ −11 in return terms).
- **Dropped the sharpness/TV metric** from the grabbability notebook per Sevan's preference; Fig 1a is now next-step
  RMSE with the repo's standard **dashed baselines** (`pim/eval/baselines.py`): copy-previous-frame 0.160 and
  observation noise floor 0.066 — models sit at 0.10–0.13, so below the trivial baseline but well above the floor.
- **Added the per-step "does the edit land and hold" curve** (RMSE vs the post-edit target render vs rollout step),
  replacing the old panel; **observer waterfalls** now rendered alongside the actor's for every run.

**Answered in-notebook (Sevan's Q5/Q7/Q9):** (a) the actor's loss is a fixed weighted sum
`pred + 1.0·policy + 0.5·value + 0.01·entropy` and **those weights were NEVER swept** — the prediction-vs-control
balance is an arbitrary unvalidated hyperparameter and the contrast is by construction sensitive to it (flagged as the
most obvious missing control); prediction is not strictly needed for survival — it is there because it is the research
subject and to keep actor/observer objectives comparable (the Dreamer/RSSM pattern). (b) **The "death = unpredictable"
idea does not remove the need for RL**, but the clean version is to keep REINFORCE and make **reward = −(prediction
error)** (the SMiRL / free-energy formulation) — one self-consistent objective instead of an arbitrary λ; recommended
as the next experiment. (c) The **static GT column** is deliberate: each editor changes the latent → changes the policy
→ would induce a *different* true future, so there is no single common reference; the frozen target is editor-
independent but only correct at step 0 (step-0 metrics unaffected; the per-step curve reads as "how long does the edit
keep resembling the intended scene", not prediction error).

**NEW notebook `endogenous_agent_animations.ipynb`** (Sevan's item 4): play.py-style animations of every trained agent —
2D world + **keyboard overlay showing the model's actions as key presses** + white force vectors + real observation
waterfall + **the model's predicted-observation waterfall**. Built by **importing the same `Emulator` class
`scripts/play.py` uses** (extended with a `ModelDriver` and support for N predictor panels), with world settings read
from each checkpoint so the visualisation matches training (death-on-collision/wall, death noise frames, momentum).
Covers L1/L2/L3 weak + both strong seeds + the strong no-goal control, an **actor-vs-observer** 4-panel comparison, and
**three training stages** (barely trained → partway → trained) from a checkpointed rerun (`L3s0_ckpt`, `--ckpt-every
2500`, running). GIFs → `runs/endogenous/animations/`.

## 2026-07-29 (later) — Sevan's second review: two real bugs + a hygiene failure
- **BUG (mine): the "deaths per 1000 frames" panel was INVERTED.** I plotted `1000/deaths` instead of
  `1000·deaths/frames`, so 0–1 deaths rendered as **1000** and 252 deaths as **4** — the curve rose as the agent
  *improved*. Sevan caught it ("all of the plots are going UP over time, even reaching 1000"). Fixed to use the raw
  per-iteration death counts (`deaths_curve`, now exported by `eval_endogenous.py`). Corrected values: L3 strong ends at
  **0.33 deaths/1000 frames**, the strong no-goal control at **79.1**.
- **Why 3072 is a cap, explained properly:** training is on-policy with a **fixed budget of 64 worlds × 48 steps = 3072
  frames per iteration**; `survival` is estimated inside that budget as frames ÷ deaths, so zero deaths is
  indistinguishable from immortality and reads 3072 — **right-censoring**, plus quantization (3072/1536/1024…). The world
  has no frame limit. The rate statistic is unbiased; the notebook now leads with it and marks the censoring limit.
- **Plot bloat (my regression):** adding all 7 runs to every figure made them unreadable. Reverted to a **3-run main
  comparison** (L3 goal weak · L3 goal strong · L2 no-goal strong control) with short two-line labels; L1/L2-weak and the
  second seeds are footnotes appearing only in the full table. Fig 1 now shows **one seed each** (weak vs strong).
- **HYGIENE FAILURE (Sevan was right):** every endogenous run used **`obs_noise_std=0.05`** while the repo standard —
  every dataset 0–8, including dataset 4 behind the exogenous-action work — is **0.2**. It leaked from a `play.py`
  *display* default into the science. Internal comparisons remain valid (all runs share it) but **absolute RMSE / noise
  floor / probe R² are not cross-citable with earlier notebooks**. `train_endogenous.py` now exposes `--obs-noise` and
  **defaults to 0.2**; the deviation is documented at the top of `ENDOGENOUS_RUNS.md` and both notebooks. **A matched
  re-run at 0.2 is OWED** before any cross-thread numeric comparison.
- **Q6 answered with evidence:** the action-channel test finds a PD-controller action sequence in the **real** sim
  (closes 93–95%), then replays those exact actions in the model's imagination with the policy head **bypassed** (the
  action enters via the decoder conditioning, the same pathway used in training). Its poor showing is now explained by a
  new per-step panel: model-vs-real RMSE is **0.12 at step 1** (matching the teacher-forced animations) rising to
  **0.35 by step 15** — i.e. the animations show *one-step* prediction, the test is a *15-step closed-loop* rollout, and
  the controller's actions are additionally off-policy (the two are confounded — why "button, not a handle" stays retracted).
- **New + a genuinely informative RESULT:** autoregressive ("dreaming") animations (`AutoregressiveModelDriver`) for L3
  weak + strong — after a 15-frame warm-up the model consumes only its own predictions while still acting on the real
  world. **Quantified: the goal-trained actor dies 2.8 times per 1000 frames teacher-forced, but 87.8 (weak) / 85.0
  (strong) closed-loop — ≈31× worse, and essentially the same as the NO-GOAL control's 79.1.** So *acting inside its own
  imagination is about as bad as having no policy at all*, and the strong configuration does not help. This is the
  cleanest statement yet of the thread's through-line: **the model is an on-policy predictor, not a simulator you can
  act inside** — and it is the same regime the editability + action-channel tests operate in, which is why their numbers
  look so much worse than the one-step-ahead panels suggest. Animation notebook's training-stage cell also made robust
  to a missing final checkpoint.
- **Animation notebook size:** embedding 12 GIFs made it 284 MB; regenerated at dpi 55 / 100 frames → **70 MB** with
  legibility preserved (GIFs also on disk in `runs/endogenous/animations/`, ~4 MB each, for easy saving). Note
  `nbstripout` strips outputs on commit, so the on-disk GIFs are the durable artifact.

## 2026-07-29 (evening) — action-in-transition ablation: a real bug, but NOT the dominant cause
**What was wrong.** The endogenous actor fed its action **only to the decoder**, never to the recurrence:
`h_t = GRU(enc(o_t), h_{t-1})`, `ô_{t+1} = dec([h_t, proj(a_t)])`. Measured consequence: feeding *opposite* actions
produced a **bit-identical** next state (‖Δh‖ = 0.0000) — the action could not influence the imagined state at all,
only the decoded observation, so its effect had to re-enter via decoder→predicted-obs→encoder (a lossy bottleneck).
Every standard action-conditioned world model (including this repo's own `action_gru_continuous`) puts the action in
the transition. This was my design error, not a property of endogenous action.

**Fix (`action_in_transition`, default False so old checkpoints load strictly).** The **previous** action is
concatenated to the GRU input — `h_t = GRU([enc(o_t), proj_trans(a_{t-1})], h_{t-1})` — using a *separate* projection
from the decoder's, so decoder behaviour is untouched. Previous (not current) because `a_t = π(h_t)` is produced *from*
`h_t`; `a_{t-1}` is what caused the transition into `t`. Threaded through `collect()` (tracks `prev_a`),
`predict_sequence` (right-shifts the action sequence), and the multistep free-run loss. Verified: ‖Δh‖ = 1.008 with the
flag on, 0.0000 off.

**A second bug caught while writing this up (would have invalidated the whole comparison).** Every *eval* path
(`ModelDriver`, `AutoregressiveModelDriver`, `AutoregressivePredictor`, the Emulator's predictors, `collect_eval`,
`warm`/`rollout`/`quality_gate`/`action_interface_test`) called `gru_step` **without** `prev_action`, so the new model
would be evaluated with a **no-op in its transition** (‖Δh‖ = 0.345 vs correct). The completion watcher ran the
comparison 28 s **before** the fix landed and reported a spurious teacher-forced rate of 23.3 for `L3s0_ait`. All paths
are now patched (harmless for flag-off models — verified `L3s0` numbers unchanged) and the comparison was re-run.

**RESULT (`L3s0_ait` = `L3s0` + action-in-transition, single variable, 25000 it):**
| | teacher-forced | closed-loop | imagined-vs-real RMSE @ step 1 / 10 / 20 |
|---|---|---|---|
| `L3s0` (action NOT in transition) | 2.8 | **85.0** | 0.159 / 0.397 / 0.457 |
| `L3s0_ait` (action IN transition) | 2.8 | **72.2** | 0.186 / 0.319 / 0.391 |
*(deaths per 1000 frames; no-goal control = 79.1; copy-previous-frame baseline RMSE 0.160, random-frame 0.393)*

**Verdict: the missing action pathway was a genuine bug but is NOT the dominant cause of the closed-loop collapse.**
Fixing it buys ~15% fewer closed-loop deaths (85.0 → 72.2) and slightly slower drift, but 72.2 is still barely better
than having **no policy at all** (79.1), and the imagination still reaches **random-frame-level error (≈0.39) by ~10–20
steps** — i.e. the dream decouples from reality rather than merely degrading. Teacher-forced metrics are identical
(2.8 both), confirming teacher forcing is blind to this change. Remaining suspects, in order: **no latent-space
consistency objective** (nothing ties the imagined latent to observation-informed latents — this is exactly RSSM's
KL(posterior‖prior)), the **hidden-state reset every 48 frames** (still present here, so this null is partially
confounded — flagged before the run), and a **5-step imagination horizon trained vs 100+ evaluated**.
→ Strengthens the case that the fix needed is a *training signal*, not more plumbing. **Next: RSSM**, aligned with
standard practice (free bits / KL balancing, actor trained in imagination, state carried across boundaries), keeping
the actor/observer contrast *inside* RSSM so agency and architecture stay separable (Sevan's constraint).

**Throughput profile (measured, batch 64):** simulator stepping ~39% (over half of it rendering), model forward during
collection ~45%, gradient update only ~16%. So my earlier "the Python loop is the bottleneck" claim was **wrong** —
collection is 84% of wall-clock but the *model's* 48 sequential latency-bound GPU calls are the largest slice. Batch
scaling: model forward is ~flat (16× batch for 1.6× time) while the per-world Python simulator is strictly linear —
so **vectorizing the sim is what would unlock large batches** (~10× env-frames/s), not the ~1.6× direct saving.
Recommended *after* the RSSM build (Dreamer-style imagination training reduces real-env demand).

## 2026-07-29 (evening 2) — VECTORISED (GPU) SIMULATOR + parity suite
Sevan asked to make training faster before the RSSM build, and to validate the change by
re-running an existing training run and checking the results are unchanged.

**Why the simulator was the right target (measured, not assumed).** Per iteration at batch 64:
simulator ~39 %, model forward during collection ~45 %, gradient update ~16 %. So the simulator is *not* the biggest
slice — but it is the only **linear** one. Batch scaling: the model forward is latency-bound and nearly flat (16× batch
for ~1.6× time) while the per-world Python simulator is strictly linear. The simulator is therefore what *prevents*
using the large batches the GPU is idle-waiting for.

**New: `pim/simulator/interactive_batched.py` — `BatchedInteractiveWorld`.** World state as `(B, n_obj, 2)` tensors,
device-agnostic (CPU or **CUDA**, observations stay on-device). Vectorises physics, wall handling, the collision test,
the death→noise→rebirth state machine, and the ray-casting renderer. Two scalar-world subtleties preserved
*deliberately*: shift-mode's accept-guard is **sequential over objects** (object 1 sees object 0's already-shifted
position — kept as an inner loop over `n_obj`), and wall handling **resolves y before x** (the x half-width uses the
updated y). The scalar `InteractiveWorld` is untouched and remains the parity reference.

**Speed (48 steps, obs_res 128, 2 objects):**
| batch | scalar Python loop | batched CPU | batched GPU | GPU speedup | GPU env-frames/s |
|---|---|---|---|---|---|
| 64 | 165 ms | 30 ms | 51 ms | 3.2× | 60k |
| 256 | 655 ms | 56 ms | 58 ms | 11.2× | 211k |
| 1024 | 2641 ms | 84 ms | 65 ms | **40×** | 752k |
| 4096 | 10592 ms | 227 ms | 72 ms | **148×** | **2.7M** |
GPU time is nearly **flat** in batch size (51 → 72 ms for 64× the worlds), i.e. the simulator is now latency-bound like
the model instead of linear — which is exactly what unlocks large batches.

**Parity suite: `tests/test_interactive_batched.py` (11 tests; whole suite now 56 passed).**
- **Bit-exact in float64 with noise off** (`drift_force_std=0`, `obs_noise_std=0`), given the same initial state and
  actions: positions and velocities for **both** dynamics modes; observations exact after the scalar world's own
  float32 cast (asserted as *equality*, stronger than a tolerance); shift-mode `blocked` flags and positions.
- **Event parity** (collision / wall / died / alive) with `reset_on_death=False`.
- **Death→rebirth TIMING parity** — compared only up to each world's first rebirth. Writing this test surfaced a real
  property (not a bug): **after a rebirth the two implementations legitimately diverge**, because each resamples fresh
  initial conditions from its own RNG stream. Trace confirmed identical behaviour through death and all noise frames,
  divergence starting exactly at the rebirth frame. Consequence: **any training comparison can only be statistical**,
  never bit-identical, once a death occurs.
- **Statistical parity with noise on**: matched noise σ and matched death rate.
- **CUDA path matches the CPU path** in float64.

**Integration:** `scripts/train_endogenous.py --batched-sim` adds `collect_batched()`, which keeps the whole rollout
on-device (no numpy round trip). Default off, so every existing result is reproducible by the original code path.

**Validation run COMPLETE — the vectorised simulator reproduces the training outcome.**
`runs/endogenous/L3s0_ait_batched` (identical to `L3s0_ait` except `--batched-sim`), 25000 iters in 1942 s.
| metric | `L3s0_ait` (scalar) | `L3s0_ait_batched` | seed-noise reference (`L3s0` vs `L3s1`) |
|---|---|---|---|
| final train pred RMSE (actor/obs) | 0.0825 / 0.0747 | 0.0815 / 0.0737 | — |
| position R² linear (actor) | 0.781 | 0.803 | 0.783 vs 0.869 (Δ 0.086) |
| velocity R² linear (actor) | 0.526 | 0.551 | 0.537 vs 0.452 (Δ 0.085) |
| fiber residual MLP (actor) | 0.492 | 0.463 | 0.453 vs 0.451 |
| next-step RMSE (actor) | 0.1252 | 0.1203 | 0.1188 vs 0.1015 |
| deaths/1000 frames, teacher-forced | 2.8 | 3.9 | — |
| deaths/1000 frames, **closed-loop** | **72.2** | **72.8** | — |
**Verdict: every difference is smaller than the seed-to-seed variation of the same config** (e.g. position R² differs
by 0.022 between simulators vs 0.086 between seeds), and the headline closed-loop failure is unchanged (72.2 vs 72.8).
Bit-identical agreement is impossible by construction — the two implementations diverge the moment a rebirth resamples
initial conditions from different RNG streams — so this is the correct form of validation, and it passes.

**Speed: 2.81x end-to-end at batch 64** — 25000 iterations in **1942 s vs 5455 s** for the identical scalar-sim run
(same config, same iteration count; the cleanest available comparison). Implied simulator share of the scalar iteration:
141/218 = **~64%**, consistent with the standalone sim benchmark (165 ms at batch 64).

> **Two of my own measurements were wrong and are retracted.** (1) The profile claiming "sim 39% / model forward 45% /
> update 16%" is invalid — its model-forward reading (196 ms at batch 64) was warmup/contention noise, and the true value
> is ~20-25 ms; a 39% share is also arithmetically incompatible with an observed 2.8x speedup (Amdahl caps it at 1.6x).
> (2) A "controlled interleaved" benchmark reporting only 1.24x was **not** controlled: the scalar path is CPU-bound and
> the batched path is GPU-resident, so running it while another job held the GPU penalised the batched path far more.
> Interleaving equalises *exposure* to contention, not *sensitivity* to it. **Rule going forward: quote full-run
> comparisons, not micro-benchmarks taken while other jobs hold the GPU.**

**Batch size — both framings, since only one was given earlier.** For a **fixed frame budget** a larger batch is much
faster: 76.8M frames needs 25000 iterations at batch 64 (~32 min) but only 1562 at batch 1024 (~7 min, estimated) —
roughly **5x** on top of the 2.8x already banked. For a **fixed number of gradient updates** a larger batch instead costs
modestly more wall-clock and sees ~16x more data. The genuine caveat is update count (1562 policy updates vs 25000), but
**the survival task was solved by iteration ~1000 in both runs**, so there is large headroom and large-batch training is
very likely sufficient. Per-iteration costs at batch 256/1024 still need a clean measurement on an idle GPU.

**Next run STARTED automatically (2026-07-29 16:00): `runs/endogenous/L3s0_ait_state`** — the **fair GRU baseline**,
differing from `L3s0_ait_batched` by **exactly one flag** (`--carry-state`), so it is a clean single-variable test of the
hidden-state-reset flaw. The recurrent state is now carried across iteration boundaries (detached => truncated BPTT)
instead of being zeroed every 48 frames while the world continues; `predict_sequence` gained an `h0` argument so the
update starts from the state collection started from, and the actor/observer carry separate states (the actor's from
collection, the observer's from its own teacher-forced pass). The state is deliberately **not** reset on death — the
worlds are one continuous stream and rebirth is already observable through the noise frames.

## 2026-07-29 (night) — RSSM build: world model WORKS, imagination-based actor DOES NOT (yet)
Brief written and approved: `research/directions/endogenous-action-rssm.md` (hypotheses stated up front; Sevan agrees
he'd *like* emergent editability but doesn't expect it). Built:
- **`pim/world_models/rssm_actor.py`** — subclasses `RSSMModel` (base untouched, verified: its `gru_cell` input size is
  still stoch-only). Adds (a) **action in the transition**: `h_t = GRUCell([s_{t-1}, proj(a_{t-1})], h_{t-1})` — the base
  RSSM had no action input at all; verified opposite actions now change the next state (‖Δh‖ = 1.70 vs the GRU's
  historical 0.0000); (b) policy + value heads on `[h,s]` (same factored discrete space, so the `play.py` key overlay
  still works); (c) **reward + continue heads** — required because training the actor inside imagination has no simulator
  to query; (d) `imagine_for_actor` (differentiable imagination; verified gradients reach policy, reward head and the
  **prior net**).
- **`scripts/train_rssm_endogenous.py`** — online loop on `BatchedInteractiveWorld`, standard objective: recon +
  KL-balanced (0.8/0.2, DreamerV2) with **free bits**, reward/continue heads on real data, actor via **λ-returns over
  imagined rollouts** (REINFORCE + value baseline, discrete actions), critic regressed on the same returns. Observer twin
  trained on the **world-model loss only**. State carried across chunks with dead worlds cleared (GRU-thread lesson).
  **`obs_noise_std=0.2`** — the repo standard, clearing the 0.05 debt.

**WORLD MODEL: healthy.** recon RMSE 0.37 → 0.22, KL rises 0.008 → 0.18 and sits **above** the free-bits floor (0.094),
so the KL term is active and there is **no posterior collapse** — I checked this explicitly because an early KL of 0.029
looked like collapse and turned out to be an untrained-model transient.

**ACTOR-IN-IMAGINATION: fails, and not merely by entropy collapse.** Sweep at 1500 iters:
| ent_coef | final entropy | final reward | imagined return |
|---|---|---|---|
| 0.003 | 0.04 (dead) | −0.058 | −0.72 (falling) |
| 0.03 | 1.21 (alive) | −0.053 | −0.73 (falling) |
Reward ends **worse than initialisation** (−0.024 → −0.058) and the **imagined return falls monotonically**, so this is
not just under-regularised exploration — the policy is optimising a bad objective. Policy-gradient sign verified correct.
**Diagnosis:** imagined latents drift off the visited-state distribution, the reward head extrapolates nonsense there,
and the actor faithfully maximises that nonsense. (Note this is a *different* failure from the GRU's carry-state
collapse, which was stale dead-world state.)

**Overnight hedge launched** (`runs/endogenous_rssm/`): (1) **`R2s0`, level 2, 10000 it — no actor loss at all**, so it
cannot hit the bug; guarantees a trained action-conditioned RSSM world model, which makes **closed-loop coherence and
§4 editability testable in the morning regardless**. Then (2) `R3s0_warm` / (3) `R3s1_warm`, level 3 with
`wm_warmup=4000` (imagination trustworthy before the policy optimises against it) and `ent_coef=0.05`.

**OVERNIGHT RESULTS (all three runs completed).**

**World model: trains well.** recon RMSE 0.166–0.168 at `obs_noise_std=0.2`, KL 0.15 (active, above the free-bits
floor), no posterior collapse. The long warm-up + `ent_coef=0.05` **did fix the entropy collapse** — policy entropy
ends at 3.93–4.02 instead of 0.04.

**Actor: still does not learn the task.** `R3s0_warm` reward −0.016, `R3s1_warm` −0.022 versus the no-goal control's
−0.033; deaths 72.5 / 76.0 per 1000 frames versus the no-goal 83. So the policy went from *actively worse than nothing*
to *marginally better than nothing* — nowhere near the GRU actor, which solved survival outright (2.8 deaths/1000
teacher-forced). Imagined return still drifts negative. **Hypothesis 3 (agency effect) cannot be tested until this works.**

**Closed-loop coherence: hypothesis 1 is NOT supported.** Warm on real observations (posterior), then imagine forward
with the prior under the model's own actions while the real world receives the same actions. Absolute RMSE is not
comparable across threads (GRU ran at noise 0.05, RSSM at 0.2), so compare each to its OWN baselines:
| model | step-1 error ÷ copy-previous-frame | late error ÷ random-frame |
|---|---|---|
| GRU `L3s0` | 0.99 | **1.16** (worse than a random frame) |
| RSSM `R2s0` | 0.98 | **0.77** |
| RSSM `R3s0_warm` | 0.91 | 0.82 |
| RSSM `R3s1_warm` | 0.87 | 0.87 |
**Reading:** the RSSM is *relatively* better — its imagination stays below the random-frame baseline out to 40 steps,
whereas the GRU's exceeded it by step ~20 — but it is still only **≈ copy-previous-frame quality from step 1 onward**.
That is not a usable simulator. Note the sharp **prior/posterior gap**: the same model reconstructs at 0.166 from
observations but its prior-only imagination sits at 0.30–0.34. The KL term did not close that gap.

**So: adding latent consistency (KL) + a proper imagination path did not rescue closed-loop rollout — in these runs.**

> ### ⚠ RETRACTED OVERREACH (2026-07-30, Sevan pushed back and he is right)
> I originally wrote here that suspicion should move "OFF the objective and ONTO the observation channel", speculating
> that a 1D 128-ray scan may be too impoverished for long-horizon self-consistency. **That conclusion is not supported by
> this evidence and is withdrawn.** It generalises from two *under-engineered* attempts to a claim about what is
> *achievable* — precisely the "bug reframed as insight" failure mode `RESEARCH.md` names as the one to guard against.
> Against it: (1) **teacher-forced next-step prediction is good** in both architectures, so the observation demonstrably
> carries the needed information — the failure is that our models do not *propagate* it; (2) the RSSM actor **never
> learned the task at all**, which indicates implementation/tuning problems, not an information limit; (3) the
> prior/posterior gap (recon 0.166 vs imagination 0.30–0.34) is a **classic symptom of an undertrained/under-tuned
> RSSM**; (4) Dreamer-class models routinely achieve long-horizon imagination on far harder, more ambiguous
> observations, on training budgets far larger than our ~40-minute first attempt.
> **Correct status: "not achieved by our implementation yet", NOT "not achievable".** Separating those two would need a
> working reference implementation or an information-theoretic argument, and we have neither. Sevan's read — that the
> task should be achievable and the open question is how much engineering it takes — is the better-supported one.

**Owed / next:** consolidated notebook (predictive + animations + editability — Sevan's explicit request) still to
build; §4 editability on the RSSM latent not yet run (the editor script is written against the GRU API — `gru_step`,
`decode_action` — and needs an RSSM adapter, though `RSSMActor` does satisfy `HiddenStateModel`). Actor fixes if we
continue that line: if the warm-start actor still fails, the candidate fixes are (a) train the reward head on *imagined*
as well as real latents, or regularise imagination to stay near the visited-state manifold; (b) shorter imagination
horizon early, annealed up; (c) fall back to REINFORCE on real rollouts for the policy while keeping RSSM's KL for the
world model — a hybrid that abandons "actor in imagination" but keeps the latent-consistency term that motivated RSSM.
Consolidated notebook (predictive + animations + editability, per Sevan's request) still to build.

> **⏳ OWED / REMINDERS FOR SEVAN (deferred — surface these in catch-ups):**
> 0. **Re-run the endogenous thread at the standard `obs_noise_std=0.2`** (deviation found 2026-07-29; see above).
> 1. **Pure-latent-overshooting RSSM re-run** — our RSSM-multistep result used a HYBRID objective (latent-overshoot
>    KL + an added observation-overshoot reconstruction term that pure PlaNet/Dreamer omits, and which drives the
>    blur). The RSSM-multistep finding is **HELD** until we re-run with **pure** latent overshooting to confirm the
>    "objective harms the RSSM" sub-claim isn't our added term. Brief: `directions/multistep-objective-rssm-pure-overshoot.md`.
>    (The §4 editability null is structural and robust to this.) **PING Sevan to schedule.**
> 2. **Tangent-curvature metric not distance/scale-normalized** — absolute degrees are a density/scale artifact
>    (56° vs 20° across notebooks is not real). Deferred fix + options: `directions/curvature-metric-normalization.md`.
>    Does not change any finding's conclusion (intrinsic dim + hull are load-bearing).

## 2026-07-27 — Catch-up after Sevan's week away; loose ends
Everything from 2026-07-17 was committed by Sevan (branch `action_conditioning`, clean). This session: renamed the
head commit to describe the experiment; **PROMOTED the object-individuation finding** →
`findings/object-individuation.md` (Sevan-approved), **scoped explicitly to EXOGENOUS actions** (endogenous action
untested — flagged as the natural next question), with Exp-2 (`action-conditioned-structure`) **folded in** as the
earlier/weaker version. Set up the two OWED reminders above. **Still HELD (Sevan):** RSSM-multistep promotion (pending
the pure-overshoot control). **Ready to draft on request:** counterfactual-history-state (metric fixed) + multistep-
steering (freeze-time-editing win) findings. **Infra:** `nvidia-smi` fails (NVML driver/library mismatch from the
week's update) but **torch/CUDA compute WORKS** — only the monitoring tool is broken; re-runs are fine.
**multistep_steering notebook cleaned (2026-07-27):** clarified η/S/+manifold/N definitions; **all waterfalls
rewritten to the master Fig-5a spec** (gray cmap, 6 noisy context frames, edit-frame line, figure-top legend) via a
reusable `waterfall_grid` helper (they had shipped `magma` + no context frames); added two **behind-the-scenes**
expository waterfalls (fig0a interleaved self-decoded-obs process; fig2b freeze-time teacher-forced-frames process).
**CLAUDE.md waterfall spec strengthened** (hard requirement, "one helper, route every waterfall through it",
recurring-violation warning). Re-ran clean (0 err). Note for the eventual freeze-time finding: 1b (freeze-time WINS)
replicates on RSSM; 1a (interleaved latent steering FAILS) is GRU-only.

## 2026-07-17 — NEW branch `action_conditioning`: action-space → object-individuation experiment
Prior editability_multi_exploration work is **committed + merged** (PR #9; RSSM multistep negative replicated).
Sevan set up clean branch `action_conditioning`. After a long design discussion (the reframe below), launched the
follow-up to Exp-2's actions.

**The reframe (important — this is the through-line now):** the real target is **object individuation**, not
"editability" per se. Question: does training a world model on an **interaction affordance** (moving objects)
reorganize its *passive* latent into a **separable, grabbable object handle** that **generalizes to interventions it
was never trained on** — vs just wiring a trained "button"? "Realism" of the latent world (structural-realist /
pragmatic stance) = the structure supports untrained interventions + persistence. Editability was always a probe for
objecthood. Sevan's framing shift: treat the GRU+latent **as the world** ("the latent world"), not a model *of* one.

**Brief:** `directions/action-space-object-individuation.md` (active). Independent variable = **action-space type**:
`dxdy` (large relative), `teleport` (absolute in-frustum placement — saturates content, forces ghost-removal),
`axis_x` (x-only restricted — the **content-generalization** probe: train x, test y). Confound triad kept (baseline
`7_dset4` / perturbed-passive-teleport control / action-conditioned). **All eval on the PASSIVE latent (action OFF)**
with the master §4 editors — so the test is **interface generalization** (does the affordance live in the *state*,
grabbable by a foreign write-mechanism, or only in the input→dynamics pathway?). Headline readouts: **object-handle
selectivity** (reach / collateral / ghost / persistence), **content generalization** (M_axis y-vs-x), interface
generalization, + light §1–§3 + an exposition (show the affordances; confirm they're perceptually large this time).
A **clean negative** is a strong result (motivates explicit scaffolding — RESEARCH.md endgame). GRU only first pass;
RSSM later. **Worker LAUNCHED** (uses the fixed WORKER.md decoupled-execution rule: train via foreground script
calls, keep the notebook light). Awaiting completion → verify artifacts → scratch note review with Sevan.

**RESULT — DONE + VERIFIED (0 error cells, 4 ckpts, 14 figs; note `scratch/2026-07-17-action-space-object-individuation.md`,
FLAG FOR PROMOTION). Worker did NOT orphan (decoupled-execution fix held).** CLEAN NEGATIVE on the primary readout:
**no action space individuates a grabbable object handle in the passive latent.** With the canonical structural editor
(PCA geodesic, an untrained write-mechanism) targeting object k on the passive/no-op latent: **ghost never clears
(0.90–0.93 for ALL five models** vs oracle-observation 0.44–0.67, decoder-gradient oracle 0.09), and edits are **non-selective
(≈0.56–0.58** — the other object is disturbed nearly as much). Holds for every affordance (dxdy/teleport/axis_x) + the
confound triad; **baseline actually has the best reach (36.7%)** so the affordances don't help the handle at all. Actions
were genuinely large this time (|Δobs| 0.19–0.22, 2–7× Exp-2). Content generalization moot (M_axis ≈ baseline; the y>x
reach asymmetry is a lateral-vs-depth geometry artifact in baseline too). **Weaker POSITIVE, localized to action-knowledge:**
large affordances make the passive latent more canonical / linearly-readable (fiber Pert-pass 0.488 → M_teleport 0.316 →
M_axis 0.282 vs baseline 0.395; vel-linear R² up) — replicates+strengthens Exp-2, but that's representation *legibility*,
not *manipulability*. **Interpretation:** objecthood lives in the input→dynamics pathway (a button), not the state (a
grabbable handle) — the affordance doesn't transfer across write-mechanisms. Readable ≠ grabbable. This is the
"you-can't-lose" negative that **motivates explicit object scaffolding** (RESEARCH.md endgame). Awaiting Sevan: read +
artifact-or-signal + promotion call; whether to (a) probe a manipulation-type reach / persistence test, (b) go
constructive (explicit-slot architecture), or (c) an RSSM check. Caveats: GRU only, N=48 edits, in-sample probes.

> **⏳ OWED / REMINDER FOR SEVAN (deferred, do when back — needs thought):** the **tangent-rotation
> "curvature" metric is not distance/scale-normalized**, so its absolute degrees are a sample-density &
> latent-scale artifact (this is why master says 56° and the newer notebooks say ~20° — NOT a real
> difference). Fix spec + options in `directions/curvature-metric-normalization.md`. Does **not** change any
> current finding's conclusion (intrinsic dim + linear hull are the load-bearing geometry numbers). Also
> OWED: the same **static-target-render / target-fill metric inflates** as the edited object moves away
> (frozen target rays) — fixed in counterfactual + multistep_steering this round, but `00_master_editability`
> likely has it too and is under a no-edit hold until Sevan re-opens it.

## 2026-07-16 (review round) — Sevan's feedback on the 3 experiments: promotion, fixes, harness

**Clearance delivered (gates promotion): ALL models trained on NOISY obs (`obs_noise_std=0.2`).** Verified:
dataset 3 (counterfactual/master GRU), dataset 4 (multistep + action baseline), dataset 5 (action models 2/3,
confirmed noise-matched to dataset 4 — the `obs_noise_std=0.0` in the action notebook is only cell 18, a
demo/edit render, NOT training data). So multistep + counterfactual are cleared, and there is **no noise
confound** in the action baseline-vs-treatment comparison.

**Sevan's two methodological catches — both CONFIRMED correct:**
- **Static-target / target-fill metric confound.** `target-fill(s)=mean(rollout@targetrays)/mean(GT@targetrays)`
  with **target rays frozen at the edit frame** → as the object moves away, `GT@targetrays→background≈0`, the
  ratio inflates >1 and trends upward for every method (explains h*_shared 1.4, unsteered 0.655). Same flaw in
  `multistep_steering`'s "RMSE vs static target render" (`s_target`) and probably in master. Sound metrics
  (obs-RMSE vs the **moving** clean GT, ghost ratio) are unaffected. → fixing the metric to track the object.
- **Curvature not normalized** (see reminder above).

**PROMOTED (Sevan's explicit approval):** multi-step-objective **NEGATIVE** result → `findings/editability.md`
2026-07-16 entry (multi-step rollout training buys rollout accuracy + GT-matched sharpness/no-blur, but NO
editability/canonicality gain — editing failure is structural, not a next-step-loss artifact). RSSM
replication noted as OWED in the entry.

**HARNESS FIX (root cause of the worker orphan-the-run failure).** Diagnosis: a **subagent is NOT re-invoked
when a background job finishes** (that notification goes to the parent), and the **10-min Bash cap** makes a
30-min/3-training notebook impossible to run as one foreground blocking call → workers are structurally pushed
into "background it and stop → orphan." `WORKER.md` rewritten: (1) **decouple training into standalone
foreground script calls** (a GRU is ~9 min < cap) + keep the analysis notebook light (loads checkpoints only);
(2) if a run must exceed the cap, **poll in-turn with back-to-back foreground calls, never return while
pending** (ending the turn early = task failure). This is a design fix, not a sterner warning.

**RSSM multistep replication:** brief written `directions/multistep-objective-rssm.md` — **HOLDING for Sevan's
go-ahead** (he'll greenlight an overnight run after the small fixes land).

**NOTEBOOK-EDIT PASS — DONE + VERIFIED (all 4 re-ran, 0 error cells, no retraining):**
- (a) **counterfactual** — frozen-target metric fixed to **track the object per-step** (target-fill now →1
  sanely, no >1 inflation) + **h*_shared W-sweep (W=1..10)** added (`fig4_Wsweep.png`): more counterfactual
  context monotonically lowers RMSE→GT (0.240→0.183), ghost (0.77→0.39), raises target-fill (0.53→0.88); only
  **~7–9% of the displacement is reachable by linear position injection** (the reachability point, quantified).
- (b) **multistep_objective** — "Fig S"→"Fig 0", §S→§0, 30° reference line + panel-c legend removed; stale
  `figS_sharpness.png` deleted.
- (c) **multistep_steering** — confounded static-target curve replaced with an object-tracking metric
  (panel 1a; 1a conclusion unchanged — interleaved doesn't beat one-shot, heavy collateral).
- (d) **action** — **exposition section E1–E3 inserted before §1**: E1 actions↔obs effect (0.7-unit nudges
  visible as step-jumps in object x/y + marked waterfall), E2 **change-the-action sanity** (flip the token at
  t0 → rollout shifts, mean|Δobs| 0.027 obj0+x / 0.134 obj1+x → **the action channel is causally used**,
  answers the item-12 leakage question), E3 2D world **GIF** (`action_demo.gif`). Action scratch note updated
  with a validity addendum (noise-matched, no confound; causal-use confirmed; shallow-shortcut caveat).

**Action promotion still HELD by Sevan** pending his read of the exposition. Tangent-curvature fix still OWED
(deferred, see top-of-file reminder + directions brief). Master notebook untouched throughout.

## 2026-07-16 (late) — RSSM multistep replication RUNNING (Sevan green-lit; he's out)
Sevan green-lit the RSSM multistep run + left. Executing autonomously (orchestrator-driven for reliability, not a
worker). **Objective:** PlaNet-style **latent overshooting** — new script `scripts/train_rssm_multistep.py`
(standard ELBO + imagine W steps through the prior from each posterior state, obs-recon of the future + KL(sg(post)‖
imagined-prior); starts subsampled n_start=8). **W∈{1(pure ELBO),2,5}**, matched **150-epoch** budget (reduced from
the refined RSSM's 500 to fit the 2-3h cap; det 256 / stoch 64; ~11s/ep baseline). Training 3 RSSMs sequentially in
bg (~112 min) → `runs/rssm_multistep/w{1,2,5}_dset4`. Analysis notebook **built + validated**:
`notebooks/experiments/editability/multistep/multistep_objective_rssm.ipynb` (adapted from the GRU multistep notebook: RSSM
checkpoints, `sample=False` prior-mean eval, §0/§1/§2/§3/§4 + a NEW **§3b det-vs-stoch split**; all cells compile;
core RSSM editor pipeline + det/stoch logic validated against the refined RSSM — det carries ~all pos/vel code &
is far more canonical than stoch, as expected). Pending: training finish → run notebook → verify → scratch note.
Caveat baked into the notebook: 150-epoch undertraining (cross-W is the load-bearing comparison) + the un-normalized
curvature metric.

**RSSM RESULT — DONE + VERIFIED (0 error cells, 12 figs; note `scratch/2026-07-16-multistep-objective-rssm.md`).**
Training done (w1 recon 0.0247 / w2 0.0323 / w5 0.0365; 109 min). Notebook ran clean after a one-line ckpt patch
(added `val_loss` key the loader needs). **Verdict: the GRU negative REPLICATES on the RSSM, and the objective is
additionally HARMFUL there** — no editor reaches the oracle observation for any W (readable≠controllable, unchanged);
AND multi-step overshoot **blurs the decoder** (rollout TV/GT 1.23→0.43 — objects fade; OPPOSITE the GRU's no-blur),
**worsens** single-step (next-step RMSE 0.113→0.166) and open-loop (0.204→0.247) prediction, **collapses the linear
hull** (36→10 dims @90%), and reduces linear readability (pos 0.82→0.64) + canonicality (MLP fiber 0.42→0.52). det
h carries ~all (pos,vel) (det≈full, intrinsic ~4); overshoot de-canonicalises the det core. Caveat: overshoot
best-recon ckpts are early (w2 ep64, w5 ep25) → harm understated if anything. Finding `editability.md` OWED-RSSM
line updated (marked done-scratch, pending Sevan's review). **This completes ALL work Sevan assigned; awaiting his
return** — promotion calls (multistep RSSM leg; action-conditioning still held pending his exposition read; the
counterfactual metric is now fixed and ready to draft as a finding on his word) + the deferred curvature-metric fix.



## 2026-07-16 (later) — NEW BRANCH `editability_multi_exploration`: 3 parallel experiments briefed

Sevan opened branch `editability_multi_exploration` to run **three editability lines at once**. Master
notebook `00_master_editability` is OFF-LIMITS (no new edits/results); all work goes in NEW notebooks,
scratch-only, promotion deferred to Sevan's review. Focus: **GRU primary, RSSM where cheap** (RSSM: examine
deterministic `h` — primary world-state carrier — and stochastic `s` separately). NOT the DiT.

**Key feasibility fact:** a GRU trains 400 epochs in **~8.5 min** on this GPU (dataset 4, 256 hidden) →
retraining for Exp 2/3 is cheap; the RSSM is the only expensive leg. Dataset 4 = `4_fixed_refl_inview`
(T=40, R=128, edit_frame=20, 2 obj, 90k train). Master baselines: GRU
`3_dset3_gru_persistentids_inview_400epochs`, RSSM `4_dset4_refined_best`; matched dataset-4 GRU baseline =
`7_dset4_gru_400epochs`.

**Three briefs written (status `proposed`, awaiting Sevan to mark active):**
1. `directions/multistep-steering.md` `[in-frame]` — Exp 1: (1a) interleaved closed-loop latent steering
   (push a little → decode → feed back → push, re-asserting the unedited object's target) vs one-shot;
   (1b) freeze-time teacher forcing (interpolate the edit over N∈1..15 frames, TF, then unfreeze). No
   retraining. Deliverable = editability success/failure only (NOT full master spread).
2. `directions/action-conditioned-structure.md` `[reframe]` — Exp 2. **Reframed by Sevan:** the question
   is whether **training on (random) discrete-token actions** with real causal effect **induces
   causal/editable latent structure**, tested by **discarding the action channel (no-op) and re-running
   the master latent editors** — NOT editing via the action channel (that's a secondary completeness
   check, expected limited since nudges ≪ edit teleports). Discrete tokens {no-op, obj0±x/±y, obj1±x/±y},
   no-op dominant/sparse actions. Requires new sim nudge + action-augmented dataset + action-conditioned
   GRU (must conform to `HiddenStateModel` protocol at no-op so the master suite runs unchanged) + train.
   Proposed optional control (perturbed-passive: same nudged trajectories, token withheld) to separate
   "perturbation diversity" from "action-knowledge" — the enactivist crux. Replicates master §1–§4.
3. `directions/multistep-prediction-objective.md` `[reframe]` — Exp 3: multi-step rollout training
   objective (free-running w-step BPTT), w∈{2,5} vs single-step baseline. Watch blur/mode-collapse.
   GRU primary; RSSM nice-to-have, **≤2–3h cap, cut if slower**. Replicates master §1–§4.

**ALL THREE COMPLETE + VERIFIED (2026-07-16). Nothing promoted — all scratch, awaiting Sevan's review.**
Three workers launched in parallel (Sevan approved incl. the Exp 2 perturbed-passive control). Each verified
on disk (0 error cells, notes, PNGs). Consolidated results:

- **Exp 1 — `notebooks/experiments/editability/multistep_steering.ipynb`** (10 cells, 0 err; note
  `scratch/2026-07-16-multistep-steering.md`; PNGs `/tmp/multistep_steering/`). **Freeze-time TF (1b) is a
  clean WIN on GRU+RSSM** — rendering the edit in over N frames (sweet spot N≈3–8) monotonically lands the
  edit + removes ghost (GRU ghost 0.333→0.123; RSSM 0.485→0.130), deployable (we render the target). **Interleaved
  latent steering (1a) does NOT win** — closed-loop push only eats ghost by dragging BOTH objects (collateral
  explodes); one-shot latent inject is inert → reproduces *readable≠controllable*. Velocity artifact from
  freezing is real (bends GRU RMSE→GT back up past N≈5) but degrades dynamics, not placement. Caveat: N=64.

- **Exp 3 — `notebooks/experiments/editability/multistep/multistep_objective_structure.ipynb`** (15 cells, 0 err; note
  `scratch/2026-07-16-multistep-objective-structure.md`; script `scripts/train_gru_multistep.py`; ckpts
  `runs/gru_multistep/w{2,5}_dset4_gru_400epochs`; 11 PNGs `/tmp/multistep_objective/`). **Clean NEGATIVE:** a
  free-running w-step rollout objective (w∈{2,5}) buys open-loop rollout accuracy (0.208→0.188) and GT-matched
  sharpness (**no blur** — watch-item cleared) but **no editability and no canonicality gain** — §4 pathology
  (decoder-inert probe, belief sluggishness, off-manifold oracle collapse) replicates unchanged across w; if
  anything canonicality mildly *degrades* (fiber resid 0.357→0.457, pos-linear R² 0.84→0.76). RSSM leg CUT
  (per cap). Refutes the brief's "coherence-under-iterated-dynamics ⇒ editable state" intuition.

- **Exp 2 — `notebooks/experiments/editability/actions/action_conditioned_structure.ipynb`** (22 cells, 0 err; note
  `scratch/2026-07-16-action-conditioned-structure.md`; substrate `pim/simulator/actions.py` +
  `pim/world_models/action_gru.py`; dataset `datasets/5_action_augmented`; ckpts `runs/gru/8_action_cond_gru_400ep`
  + `runs/gru/9_perturbed_passive_gru_400ep`; 8 PNGs `/tmp/action_conditioned/`). **NUANCED (partial positive):**
  three GRUs on byte-identical trajectories (baseline / perturbed-passive control / action-cond). **Action-training
  improves the PASSIVE latent's identifiability + canonicality — localized to action-KNOWLEDGE (3→2), not
  perturbation (1→3):** pos-linear R² 0.838→0.890, vel-linear R² 0.582→0.659, MLP fiber resid 0.379→0.324.
  **BUT editability did NOT follow** — §4 editors still fail on all three (readable≠controllable persists); the
  canonicality gain is necessary-direction but not sufficient-magnitude. Side result: *unexplained* perturbations
  (model 3) reduce belief inertia (true-swap obs-change 0.121→0.202, ghost 0.680→0.347) — a coherent-rollout effect.
  **Worker FAILURE (recovered):** the Exp 2 worker built the pipeline + dataset and launched the full nbconvert
  but **backgrounded it and stopped** (the recurring orphan-the-run failure — 3rd occurrence) *and* never wrote
  its scratch note. Orchestrator **adopted the running nbconvert** (rather than kill+restart), watched it to
  completion (0 err), verified all 3 models/figures, and **wrote the scratch note by reconstructing from the
  notebook's printed tables** (per ORCHESTRATION "reconstruct from artifacts"). Harness upgrade candidate: the
  synchronous-execution rule in WORKER.md is being ignored a 3rd time → escalate to enforcement.

**AWAITING SEVAN:** (1) read-through + artifact-or-signal calls on all three (esp. Exp 2's identifiability/
canonicality-yes-but-editability-no nuance, and whether Exp 1's freeze-time win + Exp 3's clean null warrant
`findings/` entries); (2) promotion decisions; (3) **commit** — the branch `editability_multi_exploration` holds
all of it, uncommitted (3 notebooks, 2 pim modules, 1 script, 3 briefs, 3 notes, PROGRESS/README edits). Master
notebook untouched throughout (per Sevan's constraint). One long-lived GPU kernel (PID 946778, ~3.4h) left
alone — predates the session, likely Sevan's VSCode review kernel.



## 2026-07-15 (later) — master-notebook S4/S5 review (Sevan, 31 items)

**Bugs CONFIRMED by code inspection (Sevan caught both):**
- **Fig 5 "GT" column was NOT ground truth** — it plotted the model rollout from the teacher-forced
  post-edit state `h_gt` (hence ghost traces / extra streaks). Fix: GT column = sim `edits.clean_obs`;
  the model rollout from `h_gt` stays as its own labeled "Oracle observation (model rollout)" column.
- **"MLP-gradient" was a misnomer** — it is the DECODER/obs-gradient editor (Adam on h vs GT obs). The
  repo's actual MLP-probe steering primitive (`pim.editors.gradient_steer`, from the mlp_steering PR) was
  never in the line-up. Renamed → "Decoder gradient"; "MLP-probe gradient" ADDED as a new editor.
- Also: the per-step `→target` metric compared against the STATIC edit-frame target render (so even the
  oracle observation "drifts" from it) — redefined vs the time-evolving sim clean obs at ef+s.
- Sevan's read of the decoder-gradient failure is right: it **collapses off-distribution**, it does not
  "revert" — language fixed everywhere + a revert/collapse/drift precision rule added to CLAUDE.md.

**HARNESS (durable, CLAUDE.md):** mechanism-based method names (no repo-name collisions); reference scale
+ units for every magnitude; PCA-prefixed estimator names ("honest" banned); revert-vs-collapse-vs-drift
language; calibrated claims (quantities in body, interpretation only in Summary); comparison sets grow
(new editor/model = added column/row, not a redesign).

**v4 (S4 rebuild, both models) — DONE + VERIFIED** (26/26 cells, 0 errors, sync execution, S0–S3
byte-identical; Fig 5a visually verified: GT column = clean sim render, 8 full-size cols, 6 shared
context frames, top legend, decoder-gradient collapse visible). Editor line-up per model: Readout
injection / MLP-probe gradient (new, `gradient_steer`) / Global-PCA projection / PCA geodesic / Decoder
gradient (renamed, oracle) + GT(sim)/Unsteered/True-state-swap refs. Figs: 4 (row/model), 5a/5b, 6a/6b
(step-0 scans), 6c (geodesic budget). **NEW SCIENCE from v4 (held for Sevan; feeds candidates/findings
after his read):**
- **The oracle observation itself is sluggish** — obs-change only 0.129 (GRU) / 0.059 (RSSM) with ghost-ray
  ratio 0.665 / 0.884: a single-frame belief update barely moves the rendered scene, so *every* editor's
  ceiling is low. Reframes "editing fails": even reality's own state, injected, doesn't visually teleport
  the object in one frame.
- **Geodesic K=600: ASYMPTOTES** (GRU 1.75→1.03 plateau by ~iter 135, flat to 600; RSSM no descent).
  Resolves Sevan's "did it just need longer?" — NO. And GRU's plateau readout (1.03) is *better* than the
  true-swap's readout (1.61) while its obs stay ≈unsteered → readout and obs accuracy nearly decoupled.
- **No non-oracle editor beats the oracle observation on GT next-step RMSE, on either model.**
- **Old "reverts by ~step 4" was partly a metric artifact** (static-render target); decoder-gradient on
  GRU **collapses off-distribution** (distance-to-unsteered stays flat ≈0.31 — never returns); RSSM's is
  milder (best next-step 0.131, smears by ~step 12).
- Worker-flagged caveats: ±1-frame decode-convention offset (GRU predicts next / RSSM reconstructs
  current) footnoted; geodesic's tiny leave-out residual partly tautological (last op is a local-PCA
  projection); RSSM geodesic non-descent may be step-size-limited (no sweep run).
**v5 (S5 Summary redesign) — DONE + VERIFIED** (0 errors, sync ~9 min, S0–S4 code unchanged — diff-checked;
only print-string fixes in two §0/§1 cells). §5 = "Summary — what these experiments say about the learned
state" with a clearly-marked "Our reading (interpretation)" block; calibrated phrasing ("≈34% of ‖h‖ not
explained by any g(pos,vel) we fit — largely but not fully a function of the physical state"; "close to
the 8 physical DOF (GRU slightly below, RSSM above)"); collapse-not-revert throughout. **Fig 7 — Summary**
visually verified: (a) capability bars (both models, values on both bars), (b) ONE cross-architecture
scatter (readout RMSE symlog × GT next-step RMSE; color=editor, circle=GRU/square=RSSM, legend outside) —
replaces the old 7b (negative-% bars) and 7c (Fig-3a duplicate). Consolidated summary cell → demarcated
markdown tables under "Current results (updated 2026-07-15)". `fig7_summary.png` (stale fig7_synthesis
removed).

**ENTIRE S4/S5 31-item feedback batch: COMPLETE.** Master notebook now fully review-passed §0–§5.

**2026-07-16 — correction + proposed experiment (from discussion):** Sevan refuted my "oracle observation =
editing ceiling" claim, and he's right: the one-frame-evidence state is the optimum of *observation-
mediated single-frame* belief updating — a LOWER bound for latent editing, not a ceiling (editors have
direct write access to `h`, unconstrained by filter dynamics). **Proposed: counterfactual-history state**
— back-extrapolate the edited object from the target with preserved velocity (other object true history),
render clean obs 0..ef, teacher-force → `h*`; inject at the edit frame. Since rollout is fully determined
by `h`, this should render the teleport cleanly and persist → existence proof that a clean-edit state
exists in h-space; the failure then localizes entirely to the **edit map's reachability**, sharpening the
learn-to-edit negative result. Caveat: back-extrapolation may exit the frustum early (train data was
always-in-frustum) — teacher-force only last ~10 counterfactual frames as mitigation. Also re-files the
sluggish-swap result as a **belief-inertia** measurement (dynamics/coherent-rollout thread; natural
K-frames-of-evidence convergence curve).

**LAUNCHED as a SEPARATE reference notebook** (Sevan: keep it out of the master to avoid bloat; expected
to succeed ~tautologically; promote only if surprising). Worker RUNNING →
`notebooks/experiments/editability/counterfactual_history_state.ipynb`; brief
`directions/counterfactual-history-state.md`; note will be `scratch/2026-07-16-counterfactual-history-state.md`.
The belief-inertia / K-frames-of-evidence convergence curve remains a separate future idea (dynamics thread).
**Awaiting Sevan:** (1) read-through of the rebuilt §4/§5 (esp. the sluggish-swap + geodesic-asymptote
results and the ±1-frame decode-convention caveat); (2) promotion calls — the v4 results likely warrant
updating `candidate-editability` / `candidate-rssm-replication` (sluggish swap reframes the editing story;
RSSM intrinsic-dim 9.6–10 still unpromoted); (3) **commit** — the branch holds all master-notebook
revisions (v2–v5), learn_to_edit v1+v2, harness upgrades, briefs, scratch addenda — none committed yet.

## 2026-07-15 — master-notebook review continues (Sevan; S2/S3)

- **HARNESS (durable):** strengthened `CLAUDE.md` "Notebook legibility" — (i) **clearly-demarcated tables**
  (display'd DataFrame / markdown, NOT aligned-monospace prints); (ii) **plain language, no shorthand**
  (`~=`/`=>`/`<<`/ALL-CAPS jargon banned; titles state what's shown, not the result); (iii) **define every
  implementation detail** (thresholds/subsets like "late-t = t≥15") where used.
- **NOTEBOOK v3 (S2/S3) — worker RUNNING (with the new synchronous-execution rule):** `directions/
  master-editability-notebook.md` "REVISION PASS v3". S2: loose print-tables → demarcated DataFrames; plain
  titles + `Current results` block; **switch all-t → early-t (t<15) vs late-t (t≥15)** with definitions;
  simplify Fig 2 bars (2a single-frame {lin,MLP} × GRU/RSSM early/late; 2b single-frame only; 2c same),
  keep single-vs-2-frame in the table. S3: `Current results` block; demarcated fiber table; Fig 3 plain
  title + value labels on BOTH bars + headroom so legend doesn't overlap. S0/S1/S4/S5 untouched.
- **Answered (chat):** item-4 early-t/late-t definition; flagged the all-t→early-t interpretation for Sevan.
- §4/§5 review still pending Sevan's continued pass.

## 2026-07-09 (eve) — master-notebook section-by-section review (Sevan; intro/S0/S1)

Sevan is reviewing `00_master_editability.ipynb` section by section (today: intro, S0, S1; **S2+ tomorrow**).
Two kinds of action taken:
- **HARNESS (durable):** added a **"Synthesis notebooks (source-of-truth tier)"** standard to `CLAUDE.md`:
  separate the invariant spine from dated **`Current results (updated YYYY-MM-DD)`** blocks; build every
  figure/table to hold **N world models** (color-coded per WM, no results-in-titles, compute don't-"~same");
  keep lightweight. This tier = provisional proposals for `pim`.
- **NOTEBOOK v2 (intro/S0/S1) — DONE + VERIFIED.** `00_master_editability.ipynb` re-ran clean (0 errors, 8
  figs; S2–S5 intact). Verified: Fig 0 redrawn (clean architecture-agnostic pipeline, no colinear arrows);
  Fig 1 rebuilt for N models; S0 belief-state/CM note; dated `Current results` blocks. **NEW RESULT (held):
  RSSM intrinsic dim COMPUTED — TwoNN 9.6 / MLE 10.0, HIGHER than GRU (5.2/6.9) and ABOVE 8** (GRU brackets
  8; RSSM above) — updates the old "geometry ~same" hand-wave; consistent with the belief/stochastic-latent
  view. NOT promoted — `findings/architecture-independence.md` should gain this once Sevan judges it.
- **Recovery:** the v2 worker **orphaned its notebook execution** (`run_in_background` nbconvert) and stopped
  early — orchestrator watched it to completion, fixed unfilled `@TOKEN@` placeholders it left (surgical JSON
  edit, figures preserved), and wrote the scratch addendum it skipped.
- **HARNESS FIX (root cause):** added a hard rule to `WORKER.md` — workers must run notebook execution to
  completion **synchronously in-turn**; NEVER `run_in_background`/`setsid nohup` the execution and stop
  (orphans the run + GPU kernels). This is the 2nd such failure; now explicit.
- §2–§5 review + §4 waterfall items still pending Sevan's continued pass.
- **Answered (in-chat):** the computational-mechanics question (causal/belief state) and the tangent-rotation
  method — see chat; the CM point may later refine RESEARCH.md framing (Sevan's call).

## 2026-07-09 (pm) — learn-to-edit launched, nbstripout flood fixed

**Branch:** `learn_to_edit` (Sevan made it; the 2026-07-09 promotions/rename are committed+merged via PR #7).

**HARNESS UPGRADE (durable, from Sevan's learn_to_edit review — points 1 & 5):** added a **"Notebook
legibility" hard standard** to `CLAUDE.md` (workers read it) + a pointer in `WORKER.md`. Requires, in every
experiment notebook: a **definitions table up front with each metric's explicit formula** (not buried); the
**same metric set + units across anything compared** (RMSE, not MSE); **tables for dense value sets**; inline
**data-source provenance** for borrowed constants; and a **GT/reference column in every comparison figure**.
This is the fix "for the long run, not just this notebook."

**learn_to_edit v2 REVISION — DONE + VERIFIED.** `editability/learn_to_edit.ipynb` revised in place (27 cells,
0 errors, 8 figs; RMSE now used throughout; definitions table added; note addendum in
`scratch/2026-07-09-learn-to-edit.md`). Verified on disk incl. the GT column now in the FT waterfalls (Fig 5d)
and the new Variant-B data-scaling figure (Fig 4B). **Verdict UNCHANGED** — the new B fine-tune budget sweep
reinforces v1: held-out d_gt improves only slowly (0.287→0.273 over 128→1024), ghost drops modestly,
**sel_err gets monotonically WORSE with budget** (all worse than ORIG's 0.129), h_edit stays off-manifold
(~2.7–2.9 vs real ~1.75), and fine-tuning slightly **de-canonicalizes** (fiber 0.382→0.407). Editability
still not cleanly induced. Deeper follow-ups (heavier FT, λ sweep, RSSM) remain parked for Sevan's call.

**Note (not acted on):** a ~35-min-old ipykernel (7 procs, ~part of 4.6 GB GPU) persists — most likely
Sevan's own VSCode review kernel (predates the worker; stable kernel file), so NOT killed. GPU has headroom;
kill it only if it's a stray.

**nbstripout terminal flood — FIXED.** The `BrokenPipeError` was flooding Sevan's terminal (git prompt kept
re-invoking the clean filter, which printed a Python traceback on every early-closed pipe). Fix (local
`.git/config`, persists across branches): clean filter now runs python with `signal.signal(SIGPIPE,
SIG_DFL)` so a broken pipe dies silently instead of printing a traceback, and `filter.nbstripout.required
= false` so a filter hiccup can't hard-fail git. Verified: early-closed pipe exits with no traceback;
stripping still works (0 outputs in cleaned stream). Also killed an orphaned 88-min Jupyter kernel
(leaked from earlier `setsid nohup` worker runs) — GPU now clear (1.5/32 GB). Discarded a stray
kernelspec-only diff on `editability_structure.ipynb`.

**Learn-to-edit — DONE + VERIFIED (both variants working end-to-end).** `editability/learn_to_edit.ipynb`
(15 cells, 0 errors, 7 figs), note `scratch/2026-07-09-learn-to-edit.md` (→ FLAG FOR PROMOTION). Verified
on disk (numbers present, no orphaned kernels). **RESULT: NEGATIVE — editability could NOT be cleanly
induced on this GRU**, neither by a frozen learned editor (A) nor a light fine-tune (B); both show the
**memorization signature** (train obs-loss collapses, held-out barely beats unsteered, selectivity gets
WORSE). Nuance: the info IS present (A overfits train; the obs-gradient oracle solves per-sample but
off-manifold at resid 6.8; more data helps d_gt/ghost *slowly*) — it's just **not reachable by a fixed/
amortized function few-shot, and only off-manifold per-sample**. B also **failed to canonicalize** (fiber
flat 0.382→0.383; readability down; dims up) → doesn't falsify editability⟺canonical, just fails to
support it. Strength: **medium**, not "impossible." **HELD for Sevan (judgment call on a negative
result):** the main threats-to-validity / flip-tests are (i) a **heavier Variant B fine-tune** (current was
light, 1.5k iters — a stronger intervention could still induce editability = a positive result), (ii) a
**λ sweep** mapping the on-manifold↔reach-the-edit tension, (iii) the **RSSM pass**. Did NOT auto-launch
these — interpreting/extending a negative result is the human judgment call. Offer stands to launch on request.

**Master notebook — REVISION FEEDBACK from Sevan (deferred behind learn-to-edit; captured in
`directions/master-editability-notebook.md` REVISIONS section):**
- §4 waterfalls are disliked AND possibly **wrong** — the "Unsteered" panel "looks like a model's output,"
  not an unsteered rollout. **Investigate as a potential bug**, not just aesthetics.
- Drop the purplish colormap → **classic academic style** (the light/Okabe-Ito theme).
- **Add the next-step line plots** (the 1D-line style Sevan liked from `geodesic_walk_k150`).
- Sevan will give fuller notebook feedback later.

## 2026-07-09 — Promotions, folder rename, git/nbstripout triage

**Folder rename DONE:** Sevan renamed `notebooks/experiments/manifold_editing/` → `editability/`.
Swept **all** downstream path references (findings, directions, scratch, PROGRESS, folder README) —
`grep manifold_editing` now returns 0. Notebook internals unaffected (relative paths, same depth).

**4 candidates PROMOTED to `findings/` (Sevan-approved), with preliminary/scoped hedging:**
- `findings/editability.md` (was too conclusive about "the GRU" → now scoped to *this
  pure-next-step-prediction GRU checkpoint*, not GRUs in general) ← `candidate-editability`.
- `findings/state-geometry.md` ← `candidate-state-geometry`.
- `findings/architecture-independence.md` (NEW) ← `candidate-rssm-replication`.
- `findings/predictive-quality.md` (NEW) ← `candidate-predictive-quality`.
Each opens with a **Scope (preliminary)** banner: claims are about *these trained checkpoints* at this
stage, not the architectures in general. Candidates marked ✅ PROMOTED, kept as backing detail.
`findings/README.md` index updated.

**git / nbstripout triage (Sevan's error):** nothing corrupted. (1) The `BrokenPipeError` is benign —
git ran the `nbstripout` clean filter on the large notebooks during the branch switch and closed the
pipe early; the checkout completed (clean tree, right HEAD `2dc6b4f`). (2) **Real consequence:**
`nbstripout` (clean filter, `required=true`) strips notebook outputs on commit, so after the
main→branch roundtrip the working-copy notebooks now have **0 embedded figures** (`00_master_editability`
+ `diagnostic_corrections`). The figures survive only as `/tmp` PNGs → **copied to gitignored
`runs/_review_figures/{master_editability,diagnostic_corrections}/`** so they're not lost to a /tmp wipe.
DECISION FOR SEVAN: to review the master notebook *with* figures you must either re-run it, view the
saved PNGs, or (if you want persistent inline figures) exempt presentation notebooks from nbstripout /
export an HTML. Nothing to fix in the repo itself.

**learn-to-edit: HELD** (Sevan's call). Brief stays `proposed`; not launched.

**Dynamics thrust — reframed (Sevan):** velocity lives in the *state* (nonlinear/entangled), so the
next question is **how the GRU *uses* positions and velocities to update its state** (mechanism of the
transition), not "state vs dynamics." This is the natural successor thread once editability is banked.

**Uncommitted now:** the rename sweep + 4 promotions + 2 new findings + folder README fix +
`runs/_review_figures/` (gitignored). Ready to commit on request.

## 2026-07-08 — Editability reorganization session

**Branch:** `editability_reorganization` (off the merged RSSM work; HEAD 6bcc3a9). NB: the prior
`2026-07-02` RSSM-investigation PROGRESS section lived on `editability_rssm_replicate`'s working tree,
not this branch — but the substantive artifacts are all HERE (notebook `rssm_structure/
rssm_state_geometry.ipynb`, restored scratch note, and `candidate-rssm-replication.md`).

**Corrections worker — DONE (2026-07-08):** `directions/diagnostic-corrections.md` → notebook
`editability/diagnostic_corrections.ipynb` + note `scratch/2026-07-08-diagnostic-corrections.md`
(verified on disk). Results folded into findings + candidates:
1. **Velocity 2×2 → "velocity is temporal" RETIRED (both models).** single-frame MLP ≈ 2-frame MLP
   (Δ ≤ 0.007 late-t; GRU sf-MLP R² 0.94), `dh` worse than single-frame. Velocity is instantaneously
   readable, just **nonlinearly** — the old 0.47→0.76 gap was the linear→MLP axis, not single→temporal.
   *Strategic:* undercuts the planned "velocity-in-the-dynamics" thrust — velocity is in the STATE
   (nonlinear/entangled), not the transition. Reframe that thrust before running it.
2. **RSSM det-only fiber = 0.368 ≈ GRU 0.337** (full-320 0.602 was the stochastic `s` at 0.891). The
   "RSSM less canonical" claim is DEAD — det cores are on par; KL structure buys no canonicity.
3. **Small-k geodesic — SHAKY (worker-flagged), overturned my brief's expectation.** Walked states hug
   the manifold MORE than real states (honest leave-out resid 0.11–0.30 < real 0.58–0.79); obs moves a
   lot but readout only 30–46% reached. Reinterpretation: bottleneck is reachability *along* the curved
   manifold, not off-manifold ejection. Weak swap denominator → exploratory, NOT promoted.

**Master notebook worker — DONE + VERIFIED (2026-07-08):** `directions/master-editability-notebook.md`
→ `editability/00_master_editability.ipynb` (primary/entry notebook; note
`scratch/2026-07-08-master-editability.md`; PNGs `/tmp/master_editability/fig0–7`). Verified on disk:
33 cells, **0 error outputs**, 8 embedded figures; every corrected number present in outputs (velocity
0.944/0.951, fiber 0.337/0.368/0.602/0.891, reversion 0.011→0.275). Visually spot-checked Fig 5 (unified
5-editor waterfall, dark theme, green=target/red=ghost — reads clearly) and Fig 6 (the reversion: GT
sticks, MLP-gradient reaches target @step0 then climbs back by ~step4 with the quantitative curve). Repo
clean — the worker's OOM-recovery runner scripts stayed in `/tmp`, none leaked into git. Only nit: RSSM
scree @90% recomputed = 35 vs cited 34 (subsample; noted in-notebook). **Aesthetic caveat for Sevan:**
§4 uses **waterfalls**, not the 1D-line overlay you said you liked from `geodesic_walk_k150` — waterfalls
show more, but the 1D-line version can be added if you prefer it.

**Directory reorg (Sevan item 4d) — judgment call:** did NOT rename `editability/`. Sizing showed
~17 markdown files reference the path (incl. provenance scratch notes, which shouldn't be rewritten to a
new path). Instead expressed structure via a **primary/working/scratch convention** documented in
`notebooks/experiments/editability/README.md` (primary = `00_master_editability.ipynb`), plus a
naming fix (local-tangent projection [one-shot] ≠ PCA geodesic [iterative]). A full pillar rename remains
available as a coordinated reference sweep if Sevan wants it.

**Done this session (orchestrator, CPU, in parallel with the worker):**
- **Findings corrected** (Sevan-authorized, NOT promotions): `editability.md` summary → non-canonical /
  readable≠controllable (supersedes "target unreachability") + 2026-07-08 log entry flagging the
  velocity 2×2 in-progress (do not cite "velocity is temporal" until it lands); `state-geometry.md`
  summary → intrinsic dim ~5–7 + curvature ~56° + local-resid tautology retraction; fixed stale
  editability notebook ref.
- **Scratch consolidated** → 4 self-contained candidates (kept separate, not squished):
  `candidate-editability`, `candidate-state-geometry`, `candidate-rssm-replication`,
  `candidate-predictive-quality`. Raw dated notes retained as provenance; scratch/README points at them.
- **Learn-to-edit brief** written (`directions/learn-to-edit.md`, `[reframe]`, status **proposed**):
  Variant A frozen editor (information-presence test), Variant B light fine-tune (inducibility +
  re-measure canonicality). Ready for Sevan to mark active and kick off next turn.
- **Recovered** `scratch/2026-07-02-rssm-state-geometry.md` (was untracked + disturbed during a scratch
  tidy; restored from commit 7719825, now staged/tracked — no longer at risk).

**Promotions HELD for Sevan's post-lunch read** — the 4 candidates. Recommendation: promote editability
(after the velocity 2×2), state-geometry, rssm-replication (hold the fiber *magnitude* pending the
det-only refit), and the RSSM generative-quality gap. Each candidate ends with its own recommendation.

**RSSM eval refinement (Sevan item 2):** specific case (non-canonicality measured on the full 320-d
incl. stochastic `s`) handled NOW by the worker's det-only refit; broader note (RSSM evals should
report h-only / s-only / full consistently, not default to full-state) captured in
`candidate-rssm-replication`.

**Session tasks — all DONE** (Sevan's 6 items): (1) small-k geodesic ✓, (2) velocity-MLP-on-h_t ✓
[temporal retired], (3) master notebook ✓ + reorg [light-touch, flagged], (4) learn-to-edit brief ✓
[proposed], (5) unified waterfall comparison ✓ [Fig 5/6], (6) git [Sevan's earlier push/merge/branch].

**Uncommitted:** a clean body of finished work on `editability_reorganization` (master + corrections
notebooks, 4 candidates, 3 briefs, findings corrections, folder README, restored 2026-07-02 note staged).
NOT committed — waiting on Sevan (harness rule: commit only when asked). Ready to commit on request.

**HELD for Sevan (decisions):** promotion calls on the 4 `candidate-*.md`; mark `learn-to-edit` active to
launch next; the pillar-rename; the §4 waterfall-vs-1D-line aesthetic choice; and — the strategic one —
**reframe the dynamics-identifiability thrust** now that velocity is shown to live in the state
(nonlinear/entangled coordinate), not the transition.

## 2026-06-29 — RSSM refinement (engineering, branch `rssm_refinement` off main)
Good-faith predictor-tuning of the RSSM (item #4 of the 2026-06-24 sequence). Full write-up:
**`research/scratch/2026-06-29-rssm-refinement.md`**. Headline: best RSSM now competitive —
near-horizon clean-obs MSE 0.01726 vs GRU 0.01515 (~14% gap), **beats GRU at long horizon**,
recoverability fell 0.55→0.32 as a byproduct (no position supervision). Fixed a real bug
(best-checkpoint was selected by total ELBO → froze on an undertrained warm-up epoch; now
by recon loss). Levers = lr(3e-4)+free_nats(3)+epochs; architecture is NOT the lever (plateau).
Qualitative gap confirmed (Sevan's eye-test): RSSM-mean fades the 2nd object; RSSM-**sampled**
rollout jitters/forks — analyze in prior-mean mode. Best ckpt: `runs/rssm/4_dset4_refined_best/`
(gitignored — reproducible from config+seed0). NEXT: parallel GRU tuning pass for a fair compare.
New substrate (committed): `scripts/sweep_rssm.py`, `scripts/compare_rollouts.py`, RSSM `sample`
toggle + enc/dec depth, recon-based ckpt selection. Watcher-heartbeat used for monitoring (see
auto-memory `feedback-watcher-heartbeat`; ScheduleWakeup did not fire in this env).

## Current state

- **Branch:** `edits_investigate_structure`
- **Active thread:** causal editability of GRU hidden states (sub-Q3), reframed around the
  **canonical sufficient statistic**. Verified from the sim: dynamics are constant-velocity
  (no accel/vel-noise; tiny pos noise 0.04), so the **minimal sufficient statistic is
  `(positions, velocities)` = 8-dim** for 2 objects — which sits right at the variance elbow.
- **Synthesis reached this session (the "why can't we edit it" answer):** the GRU state is
  **predictively sufficient but non-canonical**, and the world state is embedded in `h` as a
  **curved, history-entangled, non-snapshot** manifold. The *readable* code ≠ the *controllable*
  code: a probe reads position off a linear slice, but rendering a moved object needs an `h` that
  is on the curved ~6-dim manifold, carries the consistent ~35% "extra" state, and encodes
  velocity *temporally* — no low-dim probe-targeted edit produces such an `h`; the only state that
  renders the target is off-manifold and the dynamics reject it. Two independent experiments
  (keystone + geometry) corroborate this.
- **Candidate unification (framing, not a finding):** **editability ⟹ canonical (snapshot,
  factored, on-manifold) state ⟹ recoverability + coherent rollout + persistence.** This GRU has
  the *dimensionality* (~6–8) but not the *canonicality* — the gap explicit physical scaffolding
  would close. Language to ground in: causal representation learning (observational vs
  interventional identifiability; Locatello disentanglement impossibility); observability-vs-
  controllability (control theory). Read into these before committing vocabulary.

## Done this session (2026-06-24) — all 4 verified on disk

All notebooks under `notebooks/experiments/editability/`. Each worker wrote its own scratch
note + numbered notebook (plots + printed tables). Orchestrator verified every headline number
against the notebooks' printed outputs (not the sign-offs).

1. **Canonical-state keystone** — `canonical_state_editing.ipynb`; note
   `scratch/2026-06-24-canonical-state-editing.md`; PNGs `/tmp/canonical_state/`. **Hypothesis held
   strongly.** (A) Position linearly readable (R² 0.84, MLP 0.96); **velocity NOT readable from a
   single `h_t`** (R² 0.47) — it's a **temporal feature** (2-frame MLP → 0.76). (B) **Fiber NOT
   collapsed:** best `g(pos,vel)→h` leaves **34.7% residual**; linear→MLP drop 0.53 ⇒ strongly
   curved embedding. (C) **Completing the target to `(pos,vel)` does NOT fix editing** (1.4% gap,
   ghost 0.99, identical to position-only) ⇒ kills the velocity-incompleteness hypothesis. (D)
   **Obs-driven edit = readable≠controllable, localized:** reaches the target obs but lands 15.7
   off-manifold / 16.7 from canonical and reverts by ~step 4 (sequence target sticks better).
2. **Geometry diagnostic** — `manifold_geometry_diagnostic.ipynb`; note
   `scratch/2026-06-24-manifold-geometry-diagnostic.md`; PNGs `/tmp/manifold_geometry/`.
   (i) **Intrinsic dim ~5–7** (TwoNN 5.2, MLE 6.9) — brackets the physical 8 DOF; 38–73-dim global
   hull is the curved embedding, not DOF. (ii) **Strongly CURVED**: tangents rotate ~56° at the
   nearest-neighbor spacing; local tangent never aligns with global. (iii) **The geodesic's
   "strictly on-manifold" local-resid ≈0.0002 was a projection tautology** + coarse `LOCAL_K=512`;
   honest local residual never collapses (~0.75–0.84 at all k).
3. **Geodesic K=150 confirmation** — `geodesic_walk_k150.ipynb`; note
   `scratch/2026-06-24-geodesic-walk-k150.md`; PNGs `/tmp/geodesic_k150/`. K=30 "curvature barrier"
   was a **schedule artifact** (fractional step decays geometrically): constant-step control
   descends ~2× faster (RMSE→0.35), readout *is* reachable; obs still doesn't move. NOTE: its
   "stays strictly on-manifold (local resid 0.0002)" sub-claim is **retracted** by experiment 2
   (tautology). Its core point (readout reachable, obs unmoved) is now subsumed by the keystone.
4. **PCA component → position** (earlier) — `pca_component_position.ipynb`; note
   `scratch/2026-06-23-pca-component-position.md`. Metric-dependent; parked for interactive refine.

## Awaiting Sevan (human-gated — I did NOT touch `findings/` or `RESEARCH.md`)

- **Promotion calls** on 4 scratch notes: canonical-state-editing (proposed new *core* editability
  finding), manifold-geometry-diagnostic, geodesic-walk-k150, pca-component-position.
- **Finding corrections** (these contradict established entries):
  - `findings/editability.md` current-understanding — *target unreachability under manifold
    constraint* is **superseded** by non-canonicality / curved embedding / readable≠controllable.
  - **Local-residual numbers** in `findings/editability.md` + `findings/state-geometry.md` are
    **projection tautologies**; honest local residual floors ~0.75–0.84 (dated correction owed).
  - Trivial: `findings/editability.md:6` still has the pre-move notebook path (left for you).
- **Caveats for the artifact-or-signal calls:** N=64 edits; tiny |v|≈0.05 depresses velocity R²;
  "canonical" reference is teacher-forced (soft oracle). Curvature + fiber-collapse use the full
  200k–390k bank, so those are robust.

## Proposed sequence (saved 2026-06-24 EOD — pick up tomorrow)

Agreed direction, in order. Revised by the EOD discussion (velocity-in-dynamics + the
"don't-integrate-yet" decision).

1. **Velocity probe check (cheap, do first):** retrain the velocity probe on **late timesteps
   only** (t≥~15, where velocity is actually inferrable) and plot **probe-R²-vs-rollout-step** (as
   `world_model_eval` does for position). The current keystone probe used ALL timesteps incl. early
   frames where velocity is undetermined — a real confound. Prediction: single-frame R² rises but
   plateaus below the 2-frame 0.76 ⇒ velocity is encoded temporally, not as a snapshot coordinate.
2. **Velocity-in-the-dynamics (the bridge to dynamics-identifiability):** probe the GRU **update-
   network activations** (gate/candidate pre-acts `z,r,n`), not just `h`. Hypothesis (Sevan): the
   state stores *position*; the update recomputes effective velocity from `obs_t` vs `h_{t-1}` and
   discards it. If velocity is decodable from the update activations but not `h`, velocity is
   identified in the **dynamics**, not the representation — a dynamics-identifiability result, and
   the natural zoom-out from editability. NB: this reinterprets the keystone's 34.7% residual — it
   conflates *spurious history* with *legitimate dynamics scaffolding* (a reason it's underbaked).
3. **Sevan's promotion + finding-correction calls** on the 4 scratch notes (still owed) + the doc
   edits below.
4. **RSSM refinement (engineering, autonomous, NEW BRANCH off `main`):** the RSSM works but predicts
   worse than the GRU and its probe is weaker — diagnosing it now would confound undertrained-vs-
   architecture. Refine training/hyperparameters largely autonomously, **with a defined target +
   compute budget** (e.g. match GRU rollout-prediction quality within X%, or N trials) so it can't
   spin. Branch off `main` (not this diagnostic branch); checkpoint lands in gitignored `runs/`.
   Decoupled from integration — can run in parallel.
5. **Re-run the diagnostic on the refined RSSM** → the generalization result for the editability /
   canonical-state story.

**Integration decision (revised this session): DO NOT integrate the day's instruments into main
yet.** Metric-bloat is the anti-goal — the story should live in a *few principled values*, not 30
metrics. The fiber-collapse residual (conflates history vs dynamics-scaffolding) and the geometry
diagnostics (curvature metric needs a cleaner definition; not yet deeply owned) stay **exploratory**
until they are (a) formalized into something principled and (b) understood well enough to present.
Integration bar = **principled + deeply understood + paper-worthy.** Manifold-projected editing may
later be kept as the *reference editor* (honestly captioned: best of a set that all largely failed).

## Docs to edit from this session (OWED)

- **`RESEARCH.md` (Sevan):** add the organizing-principle *hypothesis* — affordances may be
  downstream of a single property, a **canonical, factored, predictively-sufficient state**;
  editability is its sharpest test. Human-authored; mark as hypothesis not result.
- **`findings/state-geometry.md` (correction owed):** local off-manifold residual ≈0 was a
  **projection tautology**; honest local residual never collapses (~0.75–0.84 all k). Intrinsic dim
  ~5–7 brackets the physical 8 DOF; 38–73-dim hull = curved embedding, not DOF.
- **`findings/editability.md` (correction owed):** supersede "target unreachability under manifold
  constraint" → non-canonical state / curved `(pos,vel)→h` embedding / velocity-in-dynamics /
  "readable ≠ controllable." Fix stale notebook path on line 6 (→ `…/editability/`).

## Meta / strategy (in discussion, 2026-06-24 EOD — not yet ratified)

- **Depth-first per criterion** (dig into one affordance, make it precise, then zoom out) — endorsed.
  **Zoom-out triggers:** (i) understanding plateaus (experiments refine numbers, not the mental
  model); (ii) you have a paper-section-worthy claim; (iii) the live leads point *outward* (to
  another architecture or another sub-question). **Editability is at a zoom-out point now** — its
  remaining threads (RSSM generalization, velocity-in-dynamics) are already outward moves; likely
  next criterion = **dynamics-identifiability.**
- **Scaling stance:** the bottleneck is **Sevan's understanding (serial)**, not compute. So:
  parallelize the **engineering substrate** (training/infra/datasets — objective targets) across
  autonomous branches; keep **diagnostic science serial + interactive**; use worker agents for
  legwork *within* a criterion, synthesis stays human. Automation should *feed* understanding, not
  flood it.
- **Educational gate (proposed):** nothing promoted/integrated without a short "mechanics & meaning"
  explainer (how computed, what it means, assumptions, failure modes) that Sevan has read and could
  present. Exploration ungated; *integration* gated on *understanding*. This is the prerequisite for
  scaling automation safely.
- **Findings-gate evolution — RATIFIED + DONE (2026-06-25):** the bright line moved from *typing*
  to *commitment*. The **orchestrator may now draft `findings/` edits as a diff for Sevan's
  approval** (workers stay scratch-only); the promotion decision + approval stay human;
  `RESEARCH.md` stays fully human-authored. Encoded in `research/README.md`, `ORCHESTRATION.md`,
  `WORKER.md`. (A PreToolUse hook to enforce remains a future upgrade, still prose-only.)

## Substrate / harness state

- **Notebooks reorganized** into `notebooks/experiments/editability/` (Sevan's move). All
  internal relative paths normalized to the new 3-deep location; KB markdown refs updated. New
  convention in CLAUDE.md: **number every cell (`# [N]`) and every figure (`Fig K`)**.
- **Briefs written this session:** `directions/canonical-state-editing.md` `[reframe]`,
  `directions/manifold-geometry-diagnostic.md` `[in-frame]`; backlog index updated.
- **Multi-agent orchestration used for the first time (worked):** 3 background workers executed
  end-to-end this session; ownership boundaries held (wrote only `scratch/` + own notebooks); the
  verify-on-disk discipline caught nothing fabricated but is the reason we trust the numbers.
  Restraint still applies — one execution-heavy worker at a time, judgment-heavy work stays
  interactive.

## Open decisions / parked

- **Background-agent Edit/Write — FIXED 2026-06-23** (`settings.local.json`: added
  `Write`/`Edit`/`NotebookEdit` to `permissions.allow`; `worktree.bgIsolation: "none"` because
  `datasets/`+`runs/` are gitignored so a worktree has no data). Verified this session — workers
  used `NotebookEdit` cleanly.
- **`Read` token cap on figure-heavy notebooks:** a fully-executed notebook with embedded PNGs can
  exceed the `Read` limit (hit on `geodesic_walk_k150.ipynb`), which blocks `NotebookEdit`'s
  read-precondition. Workaround used: surgical JSON edit via Bash for that one file; otherwise keep
  outputs lean / edit setup cells before outputs accrue.
- **Harness enforcement:** PreCompact hook wired (reminds to update PROGRESS before compaction).
  Still prose-only (deferred until a failure demands): scratch→findings promotion gate, the
  `RESEARCH.md` write-block, the worker-reads-orchestrator-files guard.
- **Shared notebook setup → `pim/`:** `rollout_from_flat`, `decode_pos`, `sigma`, the
  load/teacher-force/subspace/warm bootstrap are duplicated per notebook. Factoring into `pim/`
  would kill the cold-start burden. (Code change — Sevan's call.)
