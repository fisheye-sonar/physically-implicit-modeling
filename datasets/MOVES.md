# datasets/ move ledger — housecleaning 2026-08-31
# Rule: NOTHING in datasets/ is ever deleted; every move is recorded here.

- `20_dwscale_20m` → `discworld/dw-pn04/train`
- `21_dwscale_probe` → `discworld/dw-pn04/probe`
- `4_fixed_refl_inview` → `discworld/dw-pn04/eval`
- `0_initial` → `archive/0_initial`
- `1_fixed_refl` → `archive/1_fixed_refl`
- `2_fixed_refl_inview` → `archive/2_fixed_refl_inview`
- `3_fixed_refl_inview_brighter` → `archive/3_fixed_refl_inview_brighter`
- `5_action_augmented` → `archive/5_action_augmented`
- `5_soft_render` → `archive/5_soft_render`
- `6_cont_dxdy` → `archive/6_cont_dxdy`
- `7_cont_teleport` → `archive/7_cont_teleport`
- `8_cont_axis_x` → `archive/8_cont_axis_x`
- `9_obsnoise0_posnoise0` → `archive/9_obsnoise0_posnoise0`
- `10_obsnoise0_posnoise004` → `archive/10_obsnoise0_posnoise004`
- `11_obsnoise02_posnoise0` → `archive/11_obsnoise02_posnoise0`
- `12_omniscient2d` → `archive/12_omniscient2d`
- `13_cont_teleport_eval` → `archive/13_cont_teleport_eval`
- `14_cont_teleport_edittrain` → `archive/14_cont_teleport_edittrain`
- `15_teleport_eval_single` → `archive/15_teleport_eval_single`
- `16_teleport_edittrain_single` → `archive/16_teleport_edittrain_single`
- `17_scale_900k` → `archive/17_scale_900k`

17_scale_900k is archived with prejudice: position_noise_std=0.0 (dset4 has 0.04),
so results trained on it are uninterpretable against the canonical eval — the trap
that motivated the environment-instance manifests.
- `runs/ours_on_othello/corpus/probe_20000.npz` → `datasets/othello/oth-uniform/corpus/probe_20000.npz`
- `runs/ours_on_othello/corpus/test_10000.npz` → `datasets/othello/oth-uniform/corpus/test_10000.npz`
- `runs/ours_on_othello/corpus/train_1000000.npz` → `datasets/othello/oth-uniform/corpus/train_1000000.npz`
- `runs/ours_on_othello/corpus/train_20000000.npz` → `datasets/othello/oth-uniform/corpus/train_20000000.npz`
- `runs/ours_on_othello/corpus/train_5000000.npz` → `datasets/othello/oth-uniform/corpus/train_5000000.npz`
- `runs/ours_on_othello/corpus/train_90000.npz` → `datasets/othello/oth-uniform/corpus/train_90000.npz`
- (train_90000.npz existed in both — byte-identical, cmp-verified; the regenerated copy kept as train_90000.npz.regen-dup, the runs/ original is canonical)
- `runs/ours_on_othello/corpus/` (now empty) removed

## Layout v2 migration — 2026-09-10 (`scripts/migrate_datasets.py --apply`; spec `research/specs/DATASET_LAYOUT_SPEC.md`; log `research/scratch/2026-09-10-layout-migration-log.json`)
# Role-named splits: train/ probe/ eval/ edits/v1/ ; dead files -> _unused/ (kept). Per instance:
- `discworld/dw-8ray`:
  - `eval/dataset.json` → `edits/v1/edits.json`
  - `eval/edits.h5` → `edits/v1/edits.h5`
  - `edits_selection.json` → `edits/v1/selection.json`
  - `eval/val.h5` → `_unused/eval/val.h5`
  - `eval/train.h5` → `_unused/eval/train.h5`
  - `probe/test.h5` → `probe/probe_120k.h5`
  - `probe/dataset.json` → `probe/probe_120k.json`
  - `probe/train.h5` → `_unused/probe/train.h5`
  - `probe/val.h5` → `_unused/probe/val.h5`
  - `probe/edits.h5` → `_unused/probe/edits.h5`
  - `probe_250k/test.h5` → `probe/probe_250k.h5`
  - `probe_250k/dataset.json` → `probe/probe_250k.json`
  - `probe_250k/train.h5` → `_unused/probe_250k/train.h5`
  - `probe_250k/val.h5` → `_unused/probe_250k/val.h5`
  - `probe_250k/edits.h5` → `_unused/probe_250k/edits.h5`
  - `tokens/probe.npy` → `_unused/tokens/probe.npy`
  - `tokens/val.npy` → `_unused/tokens/val.npy`
  - copy `eval/dataset.json` → `eval/test.json`
- `discworld/dw-blink`:
  - `eval/dataset.json` → `edits/v1/edits.json`
  - `eval/edits.h5` → `edits/v1/edits.h5`
  - `eval/val.h5` → `_unused/eval/val.h5`
  - `eval/train.h5` → `_unused/eval/train.h5`
  - `probe/test.h5` → `probe/probe_120k.h5`
  - `probe/dataset.json` → `probe/probe_120k.json`
  - `probe/train.h5` → `_unused/probe/train.h5`
  - `probe/val.h5` → `_unused/probe/val.h5`
  - `probe/edits.h5` → `_unused/probe/edits.h5`
  - `probe_250k/test.h5` → `probe/probe_250k.h5`
  - `probe_250k/dataset.json` → `probe/probe_250k.json`
  - `probe_250k/train.h5` → `_unused/probe_250k/train.h5`
  - `probe_250k/val.h5` → `_unused/probe_250k/val.h5`
  - `probe_250k/edits.h5` → `_unused/probe_250k/edits.h5`
  - copy `eval/dataset.json` → `eval/test.json`
- `discworld/dw-noiseless`:
  - `eval/dataset.json` → `edits/v1/edits.json`
  - `eval/edits.h5` → `edits/v1/edits.h5`
  - `eval/val.h5` → `_unused/eval/val.h5`
  - `eval/train.h5` → `_unused/eval/train.h5`
  - `probe/test.h5` → `probe/probe_120k.h5`
  - `probe/dataset.json` → `probe/probe_120k.json`
  - `probe/train.h5` → `_unused/probe/train.h5`
  - `probe/val.h5` → `_unused/probe/val.h5`
  - `probe/edits.h5` → `_unused/probe/edits.h5`
  - `probe_250k/test.h5` → `probe/probe_250k.h5`
  - `probe_250k/dataset.json` → `probe/probe_250k.json`
  - `probe_250k/train.h5` → `_unused/probe_250k/train.h5`
  - `probe_250k/val.h5` → `_unused/probe_250k/val.h5`
  - `probe_250k/edits.h5` → `_unused/probe_250k/edits.h5`
  - copy `eval/dataset.json` → `eval/test.json`
- `discworld/dw-pn04`:
  - `eval/dataset.json` → `edits/v1/edits.json`
  - `eval/edits.h5` → `edits/v1/edits.h5`
  - `eval/val.h5` → `_unused/eval/val.h5`
  - `eval/train.h5` → `_unused/eval/train.h5`
  - `probe/test.h5` → `probe/probe_120k.h5`
  - `probe/dataset.json` → `probe/probe_120k.json`
  - `probe/train.h5` → `_unused/probe/train.h5`
  - `probe/val.h5` → `_unused/probe/val.h5`
  - `probe/edits.h5` → `_unused/probe/edits.h5`
  - `probe_250k/test.h5` → `probe/probe_250k.h5`
  - `probe_250k/dataset.json` → `probe/probe_250k.json`
  - `probe_250k/train.h5` → `_unused/probe_250k/train.h5`
  - `probe_250k/val.h5` → `_unused/probe_250k/val.h5`
  - `probe_250k/edits.h5` → `_unused/probe_250k/edits.h5`
  - copy `eval/dataset.json` → `eval/test.json`
- `othello/oth-adjacent`:
  - `corpus/probe_20000.npz` → `probe/probe_20000.npz`
  - `corpus/probe_20000_labels_20000.npz` → `probe/probe_20000_labels_20000.npz`
  - `corpus/probe_large_170000.npz` → `probe/probe_large_170000.npz`
  - `corpus/probe_large_170000_labels_170000.npz` → `probe/probe_large_170000_labels_170000.npz`
  - `corpus/test_10000.npz` → `eval/test_10000.npz`
  - `corpus/train_20000000.npz` → `train/train_20000000.npz`
  - `edits/cases_1001.pkl` → `edits/v1/cases_1001.pkl`
  - `edits/cases_1001.json` → `edits/v1/cases_1001.json`
- `othello/oth-adjacent-flip`:
- `othello/oth-noflip`:
  - `corpus/probe_20000.npz` → `probe/probe_20000.npz`
  - `corpus/probe_20000_labels_20000.npz` → `probe/probe_20000_labels_20000.npz`
  - `corpus/probe_large_170000.npz` → `probe/probe_large_170000.npz`
  - `corpus/probe_large_170000_labels_170000.npz` → `probe/probe_large_170000_labels_170000.npz`
  - `corpus/test_10000.npz` → `eval/test_10000.npz`
  - `corpus/train_20000000.npz` → `train/train_20000000.npz`
  - `corpus/train_90000.npz` → `_unused/corpus/train_90000.npz`
  - `edits/cases_1001.pkl` → `edits/v1/cases_1001.pkl`
  - `edits/cases_1001.json` → `edits/v1/cases_1001.json`
- `othello/oth-uniform`:
  - `corpus/probe_20000.npz` → `probe/probe_20000.npz`
  - `corpus/probe_20000_labels_50.npz` → `_unused/corpus/probe_20000_labels_50.npz`
  - `corpus/probe_large_170000.npz` → `probe/probe_large_170000.npz`
  - `corpus/probe_large_170000_labels_170000.npz` → `probe/probe_large_170000_labels_170000.npz`
  - `corpus/test_10000.npz` → `eval/test_10000.npz`
  - `corpus/train_1000000.npz` → `_unused/corpus/train_1000000.npz`
  - `corpus/train_20000000.npz` → `train/train_20000000.npz`
  - `corpus/train_5000000.npz` → `_unused/corpus/train_5000000.npz`
  - `corpus/train_90000.npz` → `_unused/corpus/train_90000.npz`
  - `corpus/train_90000.npz.regen-dup` → `_unused/corpus/train_90000.npz.regen-dup`
  - copy `pim/environments/othello/vendor/intervention_benchmark.pkl` → `edits/v1/cases_1001.pkl`

## 2026-09-10 — dw-5ray generator smoke
- `datasets/discworld/_smoke_5ray/` (layout-v2 `--role` smoke for the dw-5ray chain: 40 eval, 60 edits, 40 probe sequences at 7 cast / 5 kept rays, radius 1.0) → `datasets/archive/_smoke_5ray_2026-09-10/` — a smoke artefact, never read by code.

## 2026-09-11 — oth-adjacent-flip pulled from the WSL remote (layout v1 there) and moved into layout v2 on arrival
  - `datasets/othello/oth-adjacent-flip/corpus/probe_20000.npz` → `datasets/othello/oth-adjacent-flip/probe/probe_20000.npz`
  - `datasets/othello/oth-adjacent-flip/corpus/probe_20000_labels_20000.npz` → `datasets/othello/oth-adjacent-flip/probe/probe_20000_labels_20000.npz`
  - `datasets/othello/oth-adjacent-flip/corpus/probe_large_170000.npz` → `datasets/othello/oth-adjacent-flip/probe/probe_large_170000.npz`
  - `datasets/othello/oth-adjacent-flip/corpus/probe_large_170000_labels_170000.npz` → `datasets/othello/oth-adjacent-flip/probe/probe_large_170000_labels_170000.npz`
  - `datasets/othello/oth-adjacent-flip/corpus/test_10000.npz` → `datasets/othello/oth-adjacent-flip/eval/test_10000.npz`
  - `datasets/othello/oth-adjacent-flip/corpus/train_20000000.npz` → `datasets/othello/oth-adjacent-flip/train/train_20000000.npz`
  - `datasets/othello/oth-adjacent-flip/edits/cases_1001.pkl` → `datasets/othello/oth-adjacent-flip/edits/v1/cases_1001.pkl`
  - `datasets/othello/oth-adjacent-flip/edits/cases_1001.json` → `datasets/othello/oth-adjacent-flip/edits/v1/cases_1001.json`

## Othello edit benches replaced — 2026-09-12
# One recipe for every instance: 1000 single-tile flips at a FIXED 20-move prefix, cut from the instance's own
# `edits` games (a new index range [93M, 93M+10k), disjoint from train / test / probe). Li's shipped 1001 stays in
# git as the appendix anchor (pim/environments/othello/vendor/intervention_benchmark.pkl).
- `othello/<inst>/edits/v1/cases_1001.{pkl,json}` → `othello/<inst>/_unused/edits_v1_cases_1001/` (all four instances)
- new: `othello/<inst>/edits/edits_10000.npz` (the games), `othello/<inst>/edits/v1/cases_1000.{pkl,json}`

- 2026-09-13 17:50 — `discworld/dw-8ray-obs5/edits/v1/edits.h5` (PARTIAL: the generator died mid-write on a seed the frustum-drawn teleport sampler could not place) → `discworld/dw-8ray-obs5/_unused/edits_v1_partial_2026-09-13/`. Regenerated by the relaunched `dw_8ray_obs5` unit after the sampler fix (edits_dataset._sample_in_frustum follows region="circle").

## 2026-09-15 (wsl-sevan) — old-layout leftovers parked after the two-way sync with the lab
# The lab is authoritative for every split; its layout-v2 probe/eval/edits files were synced here (training corpora never move).
# The remote-born v1 files are duplicates or superseded benches; moved, not deleted. Training corpora moved into the v2 train/ dirs.
- `datasets/othello/oth-adjacent-flip/corpus/train_20000000.npz` → `datasets/othello/oth-adjacent-flip/train/train_20000000.npz`
- `datasets/othello/oth-uniform/corpus/train_5000.npz` → `datasets/othello/oth-uniform/train/train_5000.npz`
- `datasets/othello/oth-adjacent-flip/corpus` → `datasets/othello/oth-adjacent-flip/_unused/corpus_v1`
- `datasets/othello/oth-uniform/corpus` → `datasets/othello/oth-uniform/_unused/corpus_v1`
- `datasets/othello/oth-adjacent-flip/edits/cases_1001.json` → `datasets/othello/oth-adjacent-flip/_unused/edits_v1_cases_1001/cases_1001.json`
- `datasets/othello/oth-adjacent-flip/edits/cases_1001.pkl` → `datasets/othello/oth-adjacent-flip/_unused/edits_v1_cases_1001/cases_1001.pkl`
- `datasets/discworld/dw-8ray/eval/train.h5` → `datasets/discworld/dw-8ray/_unused/v1_partial_2026-09-09/eval_train.h5`
- `datasets/discworld/dw-8ray/eval/val.h5` → `datasets/discworld/dw-8ray/_unused/v1_partial_2026-09-09/eval_val.h5`
- `datasets/discworld/dw-8ray/eval/edits.h5` → `datasets/discworld/dw-8ray/_unused/v1_partial_2026-09-09/eval_edits.h5`
- `datasets/discworld/dw-8ray/eval/dataset.json` → `datasets/discworld/dw-8ray/_unused/v1_partial_2026-09-09/eval_dataset.json`
- `datasets/discworld/dw-8ray/probe/dataset.json` → `datasets/discworld/dw-8ray/_unused/v1_partial_2026-09-09/probe_dataset.json`
- `datasets/discworld/dw-8ray/probe/train.h5` → `datasets/discworld/dw-8ray/_unused/v1_partial_2026-09-09/probe_train.h5`
- `datasets/discworld/dw-8ray/probe/val.h5` → `datasets/discworld/dw-8ray/_unused/v1_partial_2026-09-09/probe_val.h5`
- `datasets/discworld/dw-8ray/probe/edits.h5` → `datasets/discworld/dw-8ray/_unused/v1_partial_2026-09-09/probe_edits.h5`
- `datasets/discworld/dw-8ray/probe/test.h5` → `datasets/discworld/dw-8ray/_unused/v1_partial_2026-09-09/probe_test.h5`
- `datasets/othello/oth-adjacent/edits/v1/cases_1001.{json,pkl}` → `datasets/othello/oth-adjacent/_unused/edits_v1_cases_1001/` (old bench; identical to the lab's parked copy)
