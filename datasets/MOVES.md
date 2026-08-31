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
