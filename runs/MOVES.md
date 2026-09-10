# runs/ move ledger — housecleaning 2026-08-31
# Rule: NOTHING in runs/ is ever deleted; every move is recorded here.

## Canonical runs (the ONLY two that pass the canonical-runs rule:
## canonical dataset AND canonical architecture AND canonical training setup)
- `scaling/BIG20M_othello_L` → `initial_othello_comparison/L-oth-20m`
- `discworld_scale/BIG20M_discworld_L` → `initial_othello_comparison/L-dw-20m`

## Everything else → archive/ (incl. W16, the S-on-othello rungs, and the L90 pair
## — L90_theirs_discworld trained on dset17's position_noise=0.0 and fails the rule)
- `_review_figures` → `archive/_review_figures`
- `action_editors` → `archive/action_editors`
- `action_sweep` → `archive/action_sweep`
- `controls` → `archive/controls`
- `discworld_scale` → `archive/discworld_scale`
- `dit` → `archive/dit`
- `endogenous` → `archive/endogenous`
- `endogenous_rssm` → `archive/endogenous_rssm`
- `gru` → `archive/gru`
- `gru_multistep` → `archive/gru_multistep`
- `latent_dit` → `archive/latent_dit`
- `latent_linearity` → `archive/latent_linearity`
- `nonlinear_gru` → `archive/nonlinear_gru`
- `omniscient_2d` → `archive/omniscient_2d`
- `othello_arch` → `archive/othello_arch`
- `othello_transfer` → `archive/othello_transfer`
- `ours_on_othello` → `archive/ours_on_othello`
- `rssm` → `archive/rssm`
- `rssm_multistep` → `archive/rssm_multistep`
- `rssm_sweep` → `archive/rssm_sweep`
- `rssm_sweep2` → `archive/rssm_sweep2`
- `scaling` → `archive/scaling`
- `soft_render` → `archive/soft_render`
- `trained_editability` → `archive/trained_editability`
- `transformers` → `archive/transformers`
- `vae` → `archive/vae`

## Kept in place: `_smoke/` (new-scheme pipeline smoke runs), `probe_cache/` (the
## canonical fingerprinted probe cache — no config.json, so the master scan skips it)

## Probe relocation (2026-08-31, per Sevan: probes live with their runs)
- canonical probe fits moved from `probe_cache/{discworld,othello}/` into
  `initial_othello_comparison/{L-dw-20m,L-oth-20m}/probes/` (cache keys are
  provenance hashes, directory-independent — every future lookup hits in place)
- smoke-run probe fits moved from probe_cache pools into runs/pipeline_smoke/*/probes/ (same relocation rule as the canonical runs)
- `runs/pipeline_smoke/` → `runs/_pipeline_smoke/` (underscore prefix = skipped by the master scan; these were 600-step pipeline tests, 0.077% of a canonical run, never interpretable as results)

- 2026-09-09  runs/{ray_ablation/L-dw-8ray-20m,interface_ablation/L-dw-8ray-tok-20m,ray_ablation/_R-dw-8ray-20m}/scores.json -> scores.pre-selection-2026-09-09.json  (rescored on the filtered dw-8ray edit-case selection; old numbers kept for comparison)

## 2026-09-09 — the grid probe target canonicalised (experiments/grid_target_control removed)
Probe blobs were LOADED and RE-STORED under canonical cache keys (no refit); the source
files were in the experiment's gitignored `probes/` dirs, which are gone with the folder.
- `experiments/grid_target_control/probes/` (18 per-point fits, model `8e615e18076c`) →
  `noise_ablation/L-dw-noiseless-20m/probes/probes_37be083f519e4f5f.pt` (LIN, points 0-8) and
  `…/probes_d04c9d9450fedf73.pt` (MLP-128, points 0-8): target `grid-16x8`, 200k seq, 50 epochs
- `experiments/grid_target_control/probes/` (18 random-init fits, model `a295a3758336` =
  `random_init_model("transformer_l", seed 0)`) → `_baselines/dw-noiseless/probes/probes_4c7bd6cdc06de02e.pt`
  (LIN) and `…/probes_b35092e3ec37f522.pt` (MLP)
- `experiments/grid_target_control/probes/` (2 observation floors, right-aligned, 200k, 50 epochs) →
  `_baselines/dw-noiseless/probes/probes_b19531c23a65e3cf.pt` (LIN), `…/probes_ec44d4ed567d5e2a.pt` (MLP);
  their skills added to `_baselines/dw-noiseless/baselines.json` under `bases["grid-16x8"]`
- `experiments/grid_target_control/_pn04_partial/probes/` (4 real fits on `L-dw-20m`, points 0-1 only;
  the fit was stopped 2026-09-08) → `initial_othello_comparison/L-dw-20m/probes/probes_68d1ea01cd0fd06c.pt`
  (LIN) and `…/probes_bde36cc476e7da5f.pt` (MLP), provenance marked `partial`. The four 2-epoch /
  3k-sequence smoke fits beside them were not kept.
- `experiments/grid_target_control/outputs/waterfall_grid_edits.png` →
  `noise_ablation/L-dw-noiseless-20m/figures/waterfall_edits_grid-16x8_experiment-2026-09-08.png`
- `runs/noise_ablation/L-dw-noiseless-20m/scores.json` gained the `grid-16x8` block (`blocks_added`
  records the date and commit); nothing else in it changed.

## Probe-cache re-key — 2026-09-10 (layout v2: cache keys name the corpus logically, `data=discworld/<inst>`, `split=probe_<size>`, instead of a filesystem path; bytes unchanged; full old→new list in `research/scratch/2026-09-10-layout-migration-log.json`)
- `experiments/dw_tokens/obsfloor/probes`: 64 re-keyed
- `runs/_architecture_gate/R-dw-20m/probes`: 4 re-keyed
- `runs/_architecture_gate/R-dw-noiseless-20m/probes`: 4 re-keyed
- `runs/_baselines/dw-8ray/probes`: 50 re-keyed
- `runs/_baselines/dw-blink/probes`: 20 re-keyed
- `runs/_baselines/dw-noiseless/probes`: 44 re-keyed
- `runs/_baselines/dw-pn04/probes`: 40 re-keyed
- `runs/_pipeline_smoke/S-dw-smoke/probes`: 4 re-keyed
- `runs/_pipeline_smoke/dw-tok-smoke/probes`: 4 re-keyed
- `runs/blink_ablation/L-dw-blink-20m/probes`: 4 re-keyed
- `runs/initial_othello_comparison/L-dw-20m/probes`: 16 re-keyed, 4 relative-path duplicates parked in `_superseded/`
- `runs/interface_ablation/L-dw-8ray-tok-20m/probes`: 24 re-keyed
- `runs/noise_ablation/L-dw-noiseless-20m/probes`: 16 re-keyed, 1 relative-path duplicates parked in `_superseded/`
- `runs/ray_ablation/L-dw-8ray-20m/probes`: 42 re-keyed
- `runs/ray_ablation/_R-dw-8ray-20m/probes`: 4 re-keyed
- `runs/training_curve/L-dw-20m_s001000/probes`: 4 re-keyed
- `runs/training_curve/L-dw-20m_s004000/probes`: 4 re-keyed
- `runs/training_curve/L-dw-20m_s016000/probes`: 4 re-keyed
- `runs/training_curve/L-dw-20m_s064000/probes`: 4 re-keyed
- `runs/training_curve/L-dw-20m_s128000/probes`: 4 re-keyed
- `runs/training_curve/L-dw-20m_s256000/probes`: 4 re-keyed
- `runs/training_curve/L-dw-20m_s512000/probes`: 4 re-keyed
- `runs/training_curve/L-dw-20m_s780000/probes`: 4 re-keyed
