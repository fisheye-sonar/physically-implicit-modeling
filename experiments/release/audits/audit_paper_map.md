# Audit: which code produces each item in paper/paper_draft.tex (2026-09-23, read-only)

Flags: **SHIP** = pim/, scripts/, or notebooks/{master_eval, build_paper_tables_and_figs, build_appendix_tables_and_figs}.ipynb ·
**OUT:figs** = paper/figs/ · **OUT:exp** = experiments/ · **OUT:full** = build_full_tables.ipynb · **NO CODE** = scratch or hand-computed ·
**NO RENDERER** = the number is in an artifact, but no table or notebook prints it.
I checked every table cell below against the artifacts myself, running `pim.figures.tables.collect` read-only in the cartesian basis. All cells match unless noted.

## Headline findings
1. **No code writes LaTeX.** Every paper table was copied by hand from a notebook image, printed output, or JSON.
2. **Five paper tables have no ship notebook today:** tab:seed_spread, tab:tokens_decodability, tab:tokens_editability, tab:additional_rw, and tab:im_by_point. The NN-editor cells of tab:im_vs_nn are also unrendered. tab:legal_illegal and tab:two_flip come from scripts in scripts/, which print summaries only.
3. **build_full_tables.ipynb uses BASIS='frustum'.** So its Table 5 is NOT tab:seed_spread for the continuous Rayworld rows, although the paper comment claims it is. The real source is experiments/paper_ci/scripts/ledger.py (cartesian).
4. **Every data-driven figure is built by paper/figs/*.py.** None of the three notebooks makes a paper figure. Several of those scripts depend on gitignored `.scratch/` caches.
5. **The paper depends on uncommitted working-tree changes:** pim/metrics/selection.py (the highest-fidelity fallback), tables.py, and ledger.py.

## 1. Tables

| paper item | producing code | artifact read | flag |
|---|---|---|---|
| tab:decodability | build_paper cell [2] `T.table_decodability(F,'1')` gives Observation / Random init / Trained; its panel (b), the overfit check, is not in the paper. Cell [2b] `T.table_inverse_r2(F,'1b')` gives the IM columns: Best Point = `g max`, Edit Point = `g@IM`. | scores.json `probe_skill["mine\|{linear,mlp}\|sequence"]` (Othello), `bases.cartesian.probe_skill_{linear,mlp}` (best point), `inverse_map.g_r2` read at the IM best_arm point; `runs/_baselines/<inst>/baselines.json` `archs.<arch>.bases.<basis>.{observation_right_large,random_init}` | SHIP. Caption says the baselines are fit "on the same sequences", but the observation floor is `observation_right_large` (170k games / 250k seqs) against 20k / 30k for the trained probes. |
| tab:editability | build_paper cell [3] `table_editability` gives the Othello and continuous rows. Cell [4] `table_gridified(F,'3')` gives the categorical rows (the `appearance-fac` block); the bin counts in the row labels come from `grid_target.target_cells`. | arms in scores.json (Othello top-level `arms`, key `edit_index_symdiff`; dw `bases.{cartesian,appearance-fac}.arms`, key `edit_index`) and `unedited`; selection by `pim.metrics.selection.best_arm` (uncommitted highest-fidelity fallback). Daggers = `F.rep_sd` SD > 0.1. | SHIP. Table 2 prints ±SD under each Othello and continuous cell. Table 2c is a large superset (about 25 unused blocks). The dagger comment points to ledger.md (OUT:exp). |
| tab:predictive_skill | build_appendix cell [2] `T.table_prediction(...,'A1')`; cell [3] prints `T.prediction_rows` | scores.json `prediction.readings.{moves,frames,tokens,expected-frame}.loss_paired` (scripts/score_prediction.py → `pim.environments.prediction.score_run`); `runs/_baselines/<inst>/bayes_floor.json` (scripts/bayes_floor.py; `pim.metrics.prediction.floor_estimate` / `excess_estimate`) | SHIP. The ×10⁻³ scaling was applied by hand. |
| tab:seed_spread | experiments/paper_ci/scripts/ledger.py → experiments/paper_ci/dashboard/ledger.md (`T.set_basis('cartesian')`, `T.collect(...).rep_sd`, SD with n−1) | scores.json of `<run>__seed0_s512000`, `__seed1`, `__seed2`, pooled by `pim.metrics.replicates.pool_replicates` | **OUT:exp**. Partly in ship: build_paper Table 2 shows ±SD for Othello and continuous EI / Fid. Probe Skill SDs and categorical-row SDs appear in no ship notebook. OUT:full Table 5 is frustum and prints mean±SD with ND. |
| tab:legal_illegal | scripts/reachability_table.py. Step 1: `pim.environments.othello.reachability` → `runs/_baselines/<inst>/reachability.json`. Step 2 (GPU): re-runs PI / GS / IM at best_arm on the full bench → `runs/<run>/editability_by_reachability.json`. | `split.{reachable,unreachable}.{n,unedited_index,editors.*.{index,fidelity}}` | SHIP script; printed summary only, no notebook. |
| tab:two_flip | scripts/two_flip_editability.py: default run adjacent-noflip; `--run initial_othello_comparison/L-oth-20m`; `--run flip_ablation/L-oth-noflip-20m --no-legal` | `runs/<run>/two_flip_editability.json` `reported.{legal,illegal}.{PI,GS,IM}.{edit_index_symdiff,fidelity_ratio,index_se}`, `groups.<g>.unedited_index`, `partners_searched` | SHIP script; printed only. |
| tab:tokens_decodability | Frames rows: the 8-ray row of Tables 1 / 1b. Token rows: no cartesian notebook lists L-dw-8ray-tok-20m (only OUT:full does, in frustum). **Categorical rows:** the trained skill is in `table_gridified`, but the observation floor (0.477 / 0.935), the random-init floors (0.922 / 0.935; tok 0.921 / 0.925) and the categorical IM R² (0.618 / 0.615; tok 0.641 / 0.641) are drawn by NO table function, because table_decodability and table_inverse_r2 read canonical blocks only. | baselines.json `archs.{transformer_l,transformer_l_tokens}.bases.{cartesian,appearance-fac}`; scores.json `bases.appearance-fac.inverse_map.g_r2` | **NO RENDERER** |
| tab:tokens_editability | Frame-set rows: `table_editability` / `table_gridified` would draw them (marked †) if the tok run were listed in cartesian. **Mean-frame rows: no renderer.** I checked that `best_arm(arms with fidelity_ratio := fidelity_ratio_expected, ed, 'zone_edit_index_expected')` reproduces all six cells. Unedited = `unedited.zone_edit_index_expected`. | tok scores.json `bases.{cartesian,appearance-fac}` | **NO RENDERER** (mean-frame rows are scratch) |
| tab:additional_rw | smooth and obs5 are missing from the build_paper and build_appendix run lists. OUT:full lists them, but in frustum: smooth ≠ paper; obs5 is correct via the `BASIS_BY_INSTANCE` *. A cartesian `collect([], [noiseless, smooth, obs5])` plus Tables 1 / 1b / 2 would produce it. | scores.json `bases.cartesian`; baselines.json dw-smooth / dw-8ray-obs5 cartesian | **no ship renderer** (OUT:full has the wrong basis) |
| tab:im_by_point | paper/figs/editability_trends/by_point.py → by_point_values.{md,json} (Fid = 1 − fidelity_ratio) | L-oth-20m (`probe_skill["mine\|mlp\|sequence"]`, `inverse_map.g_r2`, IM arms); L-dw-noiseless-20m `bases.cartesian` | **OUT:figs** |
| tab:categorical | build_paper cell [4] `table_gridified`: the L-dw-8ray-20m rows `appearance`, `appearance-fac`, `grid-6x5`, `grid-10x3`, `grid-16x8`, `pos@appearance`. The position row comes from Tables 1 / 2. | L-dw-8ray-20m scores.json blocks, including categorical-IM arms | SHIP (superset) |
| tab:im_vs_nn | Latent R² from build_paper [2b] `table_inverse_r2` (`g@IM`, `nn@IM`). IM Index / Fid from Table 2. **The NN (IM-NN) Index / Fid cells are rendered by no notebook** (`T.EDITORS` = PI, GS, IM). They exist in `F.df["IM-NN EI"/"IM-NN fid"]` and in ledger.md. | `inverse_map.{g_r2,nn_r2}`; `IM-NN` arms | partial; NN cells OUT:exp or NO RENDERER |
| tab:fidelity_selected | build_appendix cell [4] `collect(select='fidelity')` + `table_editability(F_fid,'A2')`; cell [5] `table_gridified(F_fid,'A2b')` | same arms, `best_arm_by_fidelity` | SHIP (A2b is a superset) |

Tables that are commented out in the .tex (rw_smooth_dome, probe_alignment, frustum_basis, and the column-form tab:additional_rw) are not in the paper.

## 2. Figures

The paper's filenames (figs/*_overview.pdf, figs/appendix/...) do not exist in the repo. They are renamed copies of paper/figs outputs, and which variant each one is still needs pinning.

| paper item | producing code | reads | flag |
|---|---|---|---|
| fig:teaser (teaser.pdf) | hand-drawn; PDF only | — | NO CODE (diagram) |
| fig:experimental_setup (setup_overview.pdf) | hand-drawn; PDF only | — | NO CODE (diagram) |
| fig:othello_and_variants (othello_overview.pdf) | paper/figs/environments_overview/othello/make_figure.py → composite_O2.pdf, or composite_O2_altmove.pdf | bench games (`load_benchmark`), vendored engine; loads draw_board from qualitative_edits_othello | OUT:figs, CPU |
| fig:rayworld_and_variants (rayworld_overview.pdf) | paper/figs/environments_overview/rayworld/make_figure.py → composite.pdf, or composite_onerow.pdf | datasets/discworld/{dw-noiseless,dw-blink,dw-8ray}/eval/test.h5, renderer, `categorical_target("appearance")` | OUT:figs, CPU |
| fig:qualitative_edits (qualitative_edits_overview.pdf) | paper/figs/qualitative_main/composite_final.py (+ common.py, rayworld_panel.py, othello_panel.py) → composite_final.pdf, or one of the sidebyside variants | `.scratch/qualitative_edits_catim_seed{0,1,2}_ctx8.pkl` and `…_128ray_seed1…` (GPU, from qualitative_edits/make_figure.py `build()` and `common.py --build-128ray 1`); `.scratch/othello_edits_guarded_cache.pkl` (GPU, 2026-09-19) | OUT:figs + .scratch. The README's regenerate steps (`--set`) build seeds 0, 5, 7, 9, 10, 12, not 1 or 2. The Othello adjacent-noflip GS is drawn at the old highest-index fallback (pt2 α1.5, ratio 6.68), not Table 2's arm, and the main caption does not say so. |
| fig:rayworld_predictions | paper/figs/predictive_quality/rayworld.py. Panel (a) needs `.scratch/history_rewrite_arrays.npz` from history_rewrite/make_figure.py (GPU). Panels (b) and (c) need compute_rayworld.py (GPU) → `.scratch/predictive_quality_{dw-blink,dw-5ray}.npz`. | free-runs on bench cases 20, 26 | OUT:figs + .scratch |
| fig:othello_predictions | paper/figs/predictive_quality/othello.py | `.scratch/othello_edits_guarded_cache.pkl` (`probs["Unedited"]`, `legal_pre`) | OUT:figs + .scratch |
| fig:editability_over_res | paper/figs/editability_trends/by_point.py → by_point.pdf | scores.json of L-oth-20m and L-dw-noiseless-20m (cartesian), `best_arm` per point | OUT:figs, CPU. Unaffected by the fallback change for these two runs. |
| fig:history_rewriting | paper/figs/history_rewrite/make_figure.py (GPU; writes scores.json + npz) → draw_paper.py → history_rewrite.pdf | L-dw-noiseless-20m checkpoint, the cached g in runs/.../probes, the dw-noiseless bench | OUT:figs + .scratch |
| fig:more_qualitative_edits_othello_1–5 | paper/figs/qualitative_edits_othello/make_figure.py `--seed k --layout cols --out-dir more_seeds/seed<k>` | the `.scratch` guarded cache (2026-09-19 arms). Running `--recompute` today would pick the new fallback for adjacent-noflip GS and for standard-noflip GS / IM, so the drawn boards would change. The appendix text describes the old boards. | OUT:figs + .scratch |
| fig:more_qualitative_edits_rayworld_1–2 | paper/figs/qualitative_edits/make_figure.py `--set` → more_seeds/seed{5,7,9,10,12}/…_paired.pdf | `.scratch/qualitative_edits_catim_seed<k>_ctx8.pkl`; probes and categorical g read from run probes/ caches only | OUT:figs + .scratch |

paper/figs outputs that are not in the paper: by_rays.py (its values back the "seven SD" sentence), the primary seed-0 appendix figures, history_frames, and history_rewrite_histonly.

## 3. Quoted numbers that are not simply a table cell

| number | producing code | artifact | flag |
|---|---|---|---|
| 0.27 vs 2.2 flips per move | scripts/othello_corpus_stats.py | `runs/_baselines/{oth-adjacent-flip,oth-uniform}/corpus_stats.json` `flips_per_move` = 0.2687 / 2.2449 (10k test games) | SHIP. The .tex `% source` comment is stale: it cites the experiments pilot and an unstored scratch check. |
| blink 1–12 frames; p 0.05, mean 7, warm-up 3 | pim/environments/discworld/bigcorpus.py `_BLINK`; blink.py (`min(cap, geometric)`) | config | SHIP |
| 128 rays, 40 frames, 0.4 / 0.8, radius 0.5 / 1.0, depths 3–12, speed 0.05–0.12 | pim/environments/discworld/{config,bigcorpus}.py | train/corpus.json | SHIP |
| 15 and 5 classes | pim/environments/discworld/grid_target.py (FactorisedTarget, "2 × (15+5)") | — | SHIP (not printed anywhere) |
| 3736 / 77 / 30 / 14 bins | `grid_target.target_cells`, shown in build_paper Table 2c labels (checked) | edits h5 `config_json` | SHIP |
| ~25M params; 780k / 512k steps; batch 256; 20M seqs; seeds 0 / 1 / 2; AdamW settings | pim/training (TrainConfig); runs/*/config.json (`n_params`) | — | SHIP |
| ~1.2M probe pairs; 200k categorical sequences; 100 GS steps; weight 0.2; α / η grids; GS start layers; k = 10; epochs 200 / 50; batch 4096 | master_eval SETTINGS (`dw_probe_seqs` 30k × 40, `oth_probe_games` 20k; `*_gs_steps`, `*_gs_beta`, ALPHA_REG / ALPHA_CAT, GS_LAYERS); `arms.GRID_PROBE_RECIPE`; pim/probes/{base,inverse}.py | — | SHIP |
| 1000 cases, step 20, unchanged edits skipped | SETTINGS `dw_bench_n`; scripts/make_edit_selection.py; scripts/make_othello_edits.py | edits/v1/selection.json | SHIP |
| **IM beats NN by 53.6%** | mean over the six Rayworld runs of `IM EI` / `IM-NN EI` − 1 from `T.collect` (cartesian). I reproduced it (80.3 / 89.0 / 48.7 / 35.9 / 32.9 / 35.0 %). | scores.json | **NO CODE** |
| Results seed sentence (≤0.004 skill, <0.04 for landing editors, 4 daggers, "the setting flips at the cutoff") | ledger.md; the arm-flip attribution exists only in research/findings/seed-variance.md | replicate scores.json | OUT:exp + NO CODE (attribution) |
| MLP ≥ 0.88; random-init within 0.03 except blink; ≤ +0.35; +0.51 / +0.60 | derived from tab:decodability / tab:editability | — | SHIP |
| **PI lands in probe space at α = 1** | `arms[PI, α=1].readout_err_after` in scores.json (≈1e-6 at points 1–8), written by `pim.environments.discworld.arms` / `readout_error` via pim/scoring | scores.json | NO RENDERER. Caveat: at point 0 it is **1.62** on dw-noiseless, so "exactly" holds only at points 1–8. |
| discussion R² 0.32–0.58 | tab:im_vs_nn / tab:decodability Edit Point | — | SHIP |
| "closes at least 99%" | build_appendix cell [3] `gap_closed` (minimum 0.991, L-oth-20m); printed, not drawn | — | SHIP |
| 15-frame rollouts; "a ray or two" | paper/figs/predictive_quality README | — | OUT:figs |
| 95% CI = ±2.48 SD | t(0.975, 2)/√3 by hand; `pim.metrics.replicates.t975` / `ci95_halfwidth` | — | SHIP constant, hand-computed |
| no skill > 0.004; landing EI ≤ 0.034; 4 cells > 0.1 | tab:seed_spread | ledger.md | OUT:exp |
| **each IM step as rays coarsen is ≥ 7 SD** | none. Recomputed from `F.rep_sd`: the smallest steps are 7.9 σ (continuous, 16→8 rays) and 7.4 σ (categorical, 8→5 rays) when divided by the larger member SD. Divided by √(s1² + s2²) they are **6.1 and 6.5**. | replicates | **NO CODE**; true only under the larger-SD convention |
| fixed-setting check (+0.28 / +0.33 / −0.40, −0.12, SDs 0.05–0.07, 0.32) | research/findings/seed-variance.md only | replicate arms | **NO CODE** |
| adjflip IM 0.62 ± 0.03, fidelity 0.40–0.59 | ledger.md (`F.rep_sd` `IM EI`, `IM fid_values` = 0.397 / 0.590 / 0.577) | replicates | OUT:exp |
| seed means within 0.05; fidelity differs by up to 0.26; Othello PI fidelity 0.48 vs 0.70 "at a larger step" | ledger.md canonical vs mean; the step size read from replicate arms by hand | replicates | OUT:exp + NO CODE (step) |
| probe refit ≤0.0007 skill / ≤0.016 EI; IM refit 0.004 / 0.036; 6–10 seeds | experiments/seed_variance/scripts/probe_seeds.py (dw-8ray appearance-fac, 6 seeds) and probe_seeds_othello.py (L-oth-20m and the three adjflip members, 10 seeds, with IM) → `runs/<run>/variance.json` `probe_seeds`; maxima taken by hand | variance.json | OUT:exp + NO CODE (aggregation) |
| reachable counts 441 / 555 / 4, 335 / 614 / 51, 0 / 1000 ×2 | reachability_table.py step 1 | `reachability.json` `counts` (budget 5M) | SHIP (checked) |
| +0.78 vs +0.61 (fidelity 0.51 / 0.48); "more than half unreachable" | tab:legal_illegal; reachability counts | — | SHIP |
| 480 searched; SE ≤ 0.11 | two_flip json `partners_searched` (noflip 480); `reported.*.index_se` (max 0.113, adjacent-noflip legal PI) | — | SHIP |
| 421 frames + UNK | scripts/make_discworld_tokens.py | datasets/discworld/dw-8ray/tokens/meta.json `vocab_size` 422 | SHIP |
| 49 of 1000 removed | `pim/environments/discworld/token_bench.py` | tok scores.json `bases.appearance-fac.n_cases_kept` = 951 | NO RENDERER |
| tokens within 0.013; mean frame 5.73 vs 5.74 | tab:tokens_decodability; tab:predictive_skill | — | see those tables |
| smooth 0.62→0.96 ("largest gain"); obs5 0.98, >0.97; −0.19…+0.02 at fidelity 0.02 | tab:additional_rw (+ tab:decodability) | — | no ship renderer |
| IM trends (R² 0.39→0.83→0.73; ≈0.3; +0.59 at pt 6, fid 0.66; >+0.54; PI / GS peak at pt 4) | by_point_values.md | — | OUT:figs |
| **reconstruction test** (≈2× the model's own error at pts 4–5; 5–16× elsewhere) | experiments/inverse_probe/scripts/othello_inverse.py (recon mode) → scores/othello_L-oth-20m_mirror128_recon.json; lasttile_table.py | — | **OUT:exp**. The "mirror128" g dates from 2026-09-14; check it is the current canonical g. |
| grid-ablation numbers | tab:categorical | — | SHIP |
| NN +0.27…+0.60 at 0.22–0.54; trails IM by 0.17–0.26; NN latent R² 0.73–0.90 | tab:im_vs_nn | — | partial |
| history rewrite (+0.61 / 0.64, −0.63, +0.59, +0.23, −0.74, +0.65 / 0.52, +0.63 / 0.64; RMSE 0.114 vs 0.264; 32 cases) | paper/figs/history_rewrite/make_figure.py | its scores.json `cards.{IM,hist,hist+IM}` (fid = 1 − ratio), `history_rmse` | OUT:figs |
| fidelity rule: IM moves ≤ 0.10 (+0.66→+0.56); "PI / GS take smaller steps" | build_appendix cell [4] printed comparison; the step sizes are only in `F_fid.df` arm columns, which are not printed | — | SHIP (steps are scratch) |
| training time 8–12 h / 19–26 h | metrics.jsonl `elapsed_s` | — | NO CODE |
| "saturating at 2%" | `TINT_SCALE` 0.02 in qualitative_edits_othello/make_figure.py | — | OUT:figs |

## 4. What the notebooks produce today

**build_paper_tables_and_figs** (cartesian; 4 Othello + 6 Rayworld runs; no figures despite the name):
- [2] Table 1, panel (a) → tab:decodability ✓. Panel (b), the overfit check ✗.
- [2b] Table 1b: g@IM and g max → tab:decodability ✓; nn@IM → tab:im_vs_nn ✓; nn max ✗.
- [3] Table 2 → tab:editability ✓, plus its ±SD (partial tab:seed_spread). `table_arms` is commented out.
- [4] Table 2c (tag '3') → categorical rows of tab:editability ✓ and tab:categorical ✓. It also draws about 25 blocks not in the paper: noiseless grid-16x8 / 8x4 / 32x16 / appearance-lat / appearance-fac; blink appearance-fac; 16-ray appearance / grid-8x4 / grid-16x8; 8-ray appearance-d2 / d3 / lat and grid-8x4 / 32x16 / 4x2; 5-ray appearance / grid-8x4 / grid-16x8.

**build_appendix_tables_and_figs** (no figures):
- [2] A1 → tab:predictive_skill ✓.
- [3] printout → "99%" ✓; the persistence-MSE and sampler diagnostics are not in the paper.
- [4] A2 plus the printed comparison → tab:fidelity_selected ✓ and "≤0.10" ✓.
- [5] A2b → categorical rows of tab:fidelity_selected ✓ (superset).

**build_full_tables** (BASIS='frustum'; its long list includes L-dw-20m, smooth, tok and obs5):
- Tables 1, 1b / 1c, 1d / 1e, 2, 2b, 2c, 4, 5, and Fig 1 (training curve of L-oth-20m and L-dw-20m).
- None of these is a paper item as configured. Table 5 would match tab:seed_spread only for the Othello and categorical rows.
- It is also run by scripts/drivers/score_pending.sh.

**make_waterfalls**: two figures for one run (default L-dw-20m, not a paper run), written to runs/<run>/figures/*.png: §1 is a free-run waterfall, §2 is an editing waterfall with PI / ND / GS and **no IM**. Not in the paper. training_curves.ipynb (per-run loss curves) is not in the paper either.

**Paper tables that no notebook produces:** tab:seed_spread (ship set), tab:tokens_decodability, tab:tokens_editability, tab:additional_rw, tab:im_by_point, the NN cells of tab:im_vs_nn. tab:legal_illegal and tab:two_flip come from scripts only.

## 5. Gaps

**Producing code sits outside the planned ship set:**
- tab:seed_spread, the tab:editability daggers, and the seed-spread prose → experiments/paper_ci/scripts/ledger.py. Fix: call `T.table_seed_variance(F, which='sd')` under cartesian in the appendix notebook.
- tab:im_by_point, fig:editability_over_res and the IM-trend prose → paper/figs/editability_trends/by_point.py.
- Every data-driven figure → paper/figs/*. These also need the gitignored `.scratch` caches (history_rewrite_arrays.npz, predictive_quality_*.npz, qualitative_edits_catim_*.pkl, othello_edits_guarded_cache.pkl), and some caches need a GPU to rebuild.
- History-rewrite numbers → paper/figs/history_rewrite/scores.json.
- Probe and IM refit spreads → experiments/seed_variance/scripts/probe_seeds{,_othello}.py.
- IM reconstruction test → experiments/inverse_probe/scripts/othello_inverse.py.
- IM-NN cells of tab:im_vs_nn → only ledger.md prints them.
- tab:tokens_* and tab:additional_rw → only build_full_tables lists those runs, and in the wrong basis.

**No code at all:** 53.6%; "≥ seven SD", which fails under the √(s1² + s2²) convention; the fixed-setting check; the "setting flips at the cutoff" attribution; "PI at a larger step at 512k"; training hours; the token mean-frame rows; the categorical floors and categorical IM R² of tab:tokens_decodability; the PI α=1 probe-space landing (not exact at point 0).

**Other release blockers:**
- Uncommitted selection.py / tables.py / ledger.py changes (the paper's fallback rule).
- Stale `% source` comments: the flips comment (now scripts/othello_corpus_stats.py) and "ledger.md = build_full Table 5".
- The tab:decodability caption ("same sequences") does not match the observation-floor corpus.
- The Othello figures were drawn with the old fallback: rerunning with `--recompute` changes the boards, and the main-text caption does not mention it.
- The qualitative_main regenerate recipe misses seeds 1 and 2.
- The figure variant is not pinned (composite_O2 vs altmove; composite vs onerow; composite_final vs sidebyside).
- master_eval scores about 25 extra blocks (grids, appearance-d2 / d3 / lat, mine_signed, frustum, ND) that nothing in the paper reads. Candidates to trim.

**Pipeline notes for the release:**
- Scorer inputs: extra-target probes and floors need `scripts/fit_probes.py` (plus `--random-init` / `--observation`) before master_eval; the scorer never fits them.
- Env-var gates: categorical IM and nn_r2 are added to an already-scored run only under `PIM_ADD_CAT_IM=1` / `PIM_ADD_NN_R2=1`; a fresh score gets both.
- Replicates: they go through scripts/drivers/replicate.sh (train.py `--seed --replicate-of`, layout_checkpoint_replicate.py, fit_probes.py, score_pending.sh), which is in scripts/.
