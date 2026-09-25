### [blocker] (static) generative-models-as-simulators/.git/config (not tracked)
- issue: Committing or pushing from the release repository would identify the authors.
- evidence: `cat .git/config`: origin url = git@github.com:<author's personal account>/generative-models-as-simulators.git. `git config --show-origin --get user.name` resolves to ~/.gitconfig with the author's real name, and the global email is personal. The repo has no commits yet, so the first commit would record that identity.
- fix: Before the first commit: `git -C RELEASE config user.name Anonymous` and `git -C RELEASE config user.email anonymous@example.invalid`. Repoint or remove origin to an anonymous host (an anonymous account, or a squashed export to anonymous.4open.science). Check `git log --format='%an %ae %cn %ce'` before any push.

### [blocker] (anon-lexical) /home/sevan/research/PIM/generative-models-as-simulators/.git/config
- issue: The git remote names the author's GitHub account, and the effective git identity (from ~/.gitconfig) is the author's real name and personal email with a -0700 timezone offset. The repo has no commits yet, so the first commit and push would publish them.
- evidence: .git/config:7 `url = git@github.com:SevanBrodjian/generative-models-as-simulators.git`. `git var GIT_AUTHOR_IDENT` gives `Sevan Brodjian <s***@gmail.com> 1790269763 -0700`, with user.name and user.email coming from file:/home/sevan/.gitconfig.
- fix: Before any commit (a human step): `git -C RELEASE config user.name Anonymous`; `git -C RELEASE config user.email anonymous@example.com`; `git -C RELEASE remote remove origin`, or set-url to a repo under an anonymous account; commit with `TZ=UTC`. Check with `git log --format='%an <%ae> %ad | %cn <%ce> %cd'` before pushing. Either publish from an anonymous account, or keep the named repo private and serve it through an anonymizing mirror. Never zip the working dir with .git in it.

### [blocker] (anon-semantic) generative-models-as-simulators/.git/config
- issue: The release repo's remote is the author's personal GitHub account, and the global git identity is his real name and personal email. Pushing or committing would publish the anonymous code under a named account with named commits.
- evidence: [remote "origin"] url = git@github.com:SevanBrodjian/generative-models-as-simulators.git; `git config --get user.name` returns the author's real name; user.email is a personal gmail address (masked).
- fix: Run `git remote remove origin`. Publish from an anonymous account or org, or through anonymous.4open.science with redaction terms SevanBrodjian/Sevan/Brodjian/sevanbro. Set a repo-local `git config user.name Anonymous` and `git config user.email anonymous@example.invalid` before the first commit, then check `git log --format='%an %ae %cn %ce'`. Create <ANON_HF_REPO> under an anonymous HF account and upload with that account's token.

### [major] (tables) paper/paper_draft.tex
- issue: The release cannot reproduce the Othello inverse-map reconstruction test at line 825: error 'within about twice the model's own only at points 4 and 5 ... five to sixteen times the model's own elsewhere'.
- evidence: `grep -rni reconstruct pim scripts notebooks README.md` in RELEASE finds only the Rayworld reconstruct_clean_obs helper. No shipped artifact stores the ratios. The LaTeX comment at line 822 cites the private research/findings/inverse-probe.md. SPEC's in-scope analyses do not name this test.
- fix: Either add scripts/im_reconstruction.py (on othello/standard, at each point write g(pre-edit board) with no edit and report the error against the pre-edit legal set divided by the model's own), or cut the two ratios from the sentence and keep the qualitative claim, which the tab:im_by_point fidelity column supports.

### [major] (pipeline) scripts/build_rayworld_corpus.py:29; pim/environments/rayworld/bigcorpus.py:18,223; scripts/train.py:73
- issue: There is no small-scale path to a Rayworld training corpus or frame model. build_rayworld_corpus.py has no size flag (SHARD_N, N_SHARDS = 500_000, 40 are fixed). bigcorpus.open_obs always maps (N_TOTAL=20M, 40, R), so train.py rejects any smaller corpus, and --limit cannot help because the map is opened first. tokens.tokenize_instance reads n from corpus.json instead, so the two readers disagree.
- evidence: train.py --env rayworld --instance 8-ray on a 20k-sequence obs.f32 fails with 'ValueError: mmap length is greater than file size'. It works only through the external wrapper work/pipeline/tiny.py, which sets bc.SHARD_N=10000 and bc.N_SHARDS=2.
- fix: Add --shards N and --shard-n N to build_rayworld_corpus.py, pass them to bc.use_instance, and record them in corpus.json. Make bigcorpus.open_obs and verify() read N from train/corpus.json when it exists, instead of N_TOTAL.

### [major] (pipeline) pim/training/sources.py:27-28; pim/training/stream.py:25,31
- issue: train.py --limit N (documented as the 'quick checks' flag) hangs forever for Rayworld frames when N < 6144. n_val = max(2*2048, 0.1n) leaves n_train shorter than one 2048-sequence block, so BlockStream.starts is empty. Its worker then spins in while True and batches() blocks on q.get().
- evidence: 'train.py --env rayworld --instance 8-ray --run rayworld/limit-test --steps 20 --limit 5000' printed nothing in 90 s and was killed (rc 124). The same run without --limit takes about 5 s.
- fix: In rayworld_source: if n_train < block: raise ValueError(f'--limit {n_total} leaves {n_train} training sequences, fewer than one {block}-sequence block; use at least {3*block}'). In BlockStream.__init__: assert len(self.starts), 'range shorter than one block'.

### [major] (pipeline) pim/environments/rayworld/arms.py:54; scripts/fit_probes.py:45-46; pim/scoring/rayworld.py (_fit), pim/scoring/baselines.py (_rw_categorical_floors)
- issue: The categorical probe recipe is fixed (GRID_PROBE_RECIPE: 200k sequences of probe_250k, 50 epochs) and no SETTINGS key changes it. Fits made with fit_probes.py --n-seq/--epochs land in cache keys the scorer never reads, so every categorical block and floor is silently skipped.
- evidence: Tiny master_eval log: 'appearance-fac: SKIPPED (no cached probes ...)', and the same for appearance, grid-6x5, grid-10x3 and grid-16x8 on every run. The floors print 'transformer_l/appearance-fac: floors SKIPPED'. The scores.json files lack those blocks.
- fix: Add SETTINGS rw_cat_probe_seqs and rw_cat_probe_epochs and pass them through probe_recipe(target, inst, n_seq, cat_n_seq=..., cat_epochs=...) in scoring/rayworld.py and baselines.py. Give fit_probes.py the same defaults, or have it print that overridden fits are ignored by the scorer.

### [major] (pipeline) pim/environments/rayworld/arms.py:121,168,251; pim/scoring/baselines.py:30
- issue: Full-scale corpus sizes are hard-coded (LARGE rw_n_seq=250_000, the 200k categorical recipe, and the 170k-game / 50-epoch Othello large floor). _read_corpus slices [:n_seq] and then indexes permutation(n_seq) without checking the file size, so a smaller corpus crashes with a cryptic error.
- evidence: master_eval with a 3000-sequence probe_250k.h5 fails with 'AcceleratorError: CUDA error: unspecified launch failure' at baselines.py:83. fit_probes.py with its default recipe raises 'IndexError: index 3000 is out of bounds'. probe_refit_variance.py raises 'IndexError: index 6333 ... size 3000'. The Othello floors took 12 min even in the tiny run.
- fix: In _read_corpus: if len(obs) < n_seq: raise ValueError(f'{h5_path.name} holds {len(obs):,} sequences; this fit needs {n_seq:,}'). Expose LARGE as SETTINGS keys (rw_large_seqs, large_epochs).

### [major] (pipeline) scripts/reachability_table.py:35; scripts/two_flip_editability.py:36; scripts/probe_refit_variance.py:28
- issue: The analysis scripts hard-code the scorer's settings (PROBE_GAMES=20_000, GS_STEPS=100, RW_PROBE_SEQS=30_000, RW_BENCH_N=1000) instead of reading what the run was scored with. On any run scored with other SETTINGS they refit, then fail their own check against scores.json.
- evidence: On the tiny othello/standard run: reachability_table.py refit its probes at 20k games (6 min) and then exited with 'GS: ... MISMATCH ... nothing written'. two_flip_editability.py raised 'AssertionError: GS: the single-tile path does not reproduce scores.json (-0.0171 vs +0.0015)'. probe_refit_variance.py --run rayworld/8-ray raised IndexError.
- fix: In pim/scoring/driver.py, write scores['settings'] = {k: s[k] for k in ('oth_probe_games', 'oth_gs_steps', 'oth_gs_beta', 'rw_probe_seqs', 'rw_bench_n')}. Have the three scripts read it, falling back to the current constants for the shipped files.

### [major] (pipeline) scripts/demos/demo.py:4
- issue: The documented usage of demo.py fails (its docstring, its --help text and the stage-B command list). The same command fails in PRIVATE. The README does not list it.
- evidence: 'python scripts/demos/demo.py --seed 7 --n-objects 4 --fixed-reflectivities --save outputs/demo.gif' raises 'RuntimeError: Could not generate a collision-free scene after 300 attempts'. Without --fixed-reflectivities it raises 'ValueError: refl_min_sep ... exceeds refl range'. '--seed 0 --n-objects 4 --fixed-reflectivities' and '--seed 7 --n-objects 3' both work.
- fix: Change the usage line to: python scripts/demos/demo.py --seed 0 --n-objects 4 --fixed-reflectivities [--save outputs/demo.gif]

### [major] (pipeline) STAGING datasets/rayworld/*/*/*.h5 (28 files)
- issue: The HDF5 config_json attribute keeps a 'generated_at' timestamp, against the SPEC rule of no dates or timestamps. This confirms verify-anon-lexical M2.
- evidence: 28 of 28 Rayworld .h5 files carry it, e.g. rayworld/standard/probe/probe_120k.h5 has generated_at '2026-08-31T17:38:23'. Regenerated files and the JSON sidecars do not carry it.
- fix: Rewrite the attribute into fresh copies of the files (not in place), then regenerate SHA256SUMS and MANIFEST.json.

### [major] (static) pim/probes/base.py:192 (also pim/probes/baselines.py:198, pim/metrics/decodability.py:44,63, paper/paper_draft.tex:226)
- issue: The paper defines the classification trivial predictor as the most common class of each cell, but the reported Probe Skill uses the majority class pooled over all cells. The pim.metrics functions that follow the paper's definition are dead and give different numbers.
- evidence: majority_check.py on standard Othello: stored majority_class_error_rate 53.119% equals the pooled recomputation 53.119%; the per-cell majority error is 46.104%. MLP skill 0.9756 as reported vs 0.9718 per cell; at a weak point (error 34.96%), 0.342 vs 0.242. Nothing calls probe_skill_classification or trivial_error_rate.
- fix: Numbers are fixed by invariant 1, so change the paper sentence to 'the single most common class over all cells (from the training split)'. Delete probe_skill_classification and trivial_error_rate and their exports, or make them pool over cells and have base.fit_probe and baselines.fit_probe_stream call them. Reword the probe_skill_from_stats docstring to say the majority is pooled over cells.

### [major] (static) pim/figures/waterfall.py:30
- issue: waterfall_grid is dead code, not a paper item; the tables report's reason for keeping it is stale.
- evidence: Reachability from every script and notebook never reaches it; its only reference is the re-export in pim/figures/__init__.py:6. reports/tables.md:50 says it is kept for history_rewrite, but history_rewrite.py:117 draws with style.waterfall. The live parts of the module are DARK_BG, DIFF_CMAP and EDIT_LINE (imported at scripts/figures/style.py:26); GHOST_C and TARGET_C only build DIFF_CMAP.
- fix: Move EDIT_LINE, TARGET_C, GHOST_C, DIFF_CMAP and DARK_BG=viz.BG_HEX into scripts/figures/style.py. Delete pim/figures/waterfall.py, make pim/figures/__init__.py docstring-only, and drop the viz.py:18 comment.

### [major] (static) README.md:48
- issue: The documented download commands contain unresolved placeholders, so they fail as written.
- evidence: `<ARTIFACTS_URL>` (line 48) and `hf download <ANON_HF_REPO> ...` (lines 56-64) appear literally.
- fix: Fill in the anonymous artifact URL and repo id before publishing, or replace the commands if the artifacts are hosted elsewhere.

### [major] (anon-lexical) /home/sevan/research/PIM/generative-models-as-simulators/.ruff_cache/0.15.7/*
- issue: All 12 ruff cache files contain the absolute project path, including the username. The directory is gitignored, but any archive or upload of the working directory ships it.
- evidence: e.g. .ruff_cache/0.15.7/312607013437321849 contains `/home/sevan/research/PIM/generative-models-as-simulators/pim`. Found by a raw GNU grep for `sevan` and `/home/` (12 files).
- fix: Delete RELEASE/.ruff_cache/, plus .scratch/ and any __pycache__/, before packaging. Package only via `git archive --format=zip HEAD`.

### [major] (anon-lexical) /home/sevan/research/PIM/gms-release-artifacts/datasets/rayworld/*/{eval/test.h5,edits/edits.h5,probe/probe_120k.h5,probe/probe_250k.h5}
- issue: 28 HDF5 files keep a plaintext `generated_at` date and local clock time in the root config_json attribute. This violates the spec's 'no dates or timestamps anywhere', disagrees with the JSON sidecars (which dropped it), and fingerprints working hours and timezone.
- evidence: The values run from `"generated_at": "2026-08-31T17:38:17"` to `"2026-09-15T20:58:20"` (28 distinct). `grep -rac generated_at --include=*.h5` finds 28 files. export.md:169 left them in on purpose.
- fix: Per file, write a fresh file rather than editing in place (tested: an in-place rewrite leaves the old bytes): `with h5py.File(src,'r') as s, h5py.File(tmp,'w') as d: [s.copy(s[k], d, name=k) for k in s]; c=json.loads(s.attrs['config_json']); c.pop('generated_at',None); d.attrs['config_json']=json.dumps(c,indent=2)`, then `os.replace(tmp, src)`. Verified on 5-ray/eval/test.h5: data bit-identical, no residue. Readers use only ['dataset']['sim']. Then regenerate SHA256SUMS and the MANIFEST sha256/bytes for the 28 files, and re-grep for `generated_at|2026-0`.

### [major] (anon-semantic) gms-release-artifacts/runs/othello/standard/config.json (also runs/othello/standard__seed0/config.json and both best_model.pt)
- issue: An internal run name and leftover training-config fields are shipped. The train block's d_model 256 / n_layers 4 / n_heads 4 contradicts the actual 8-layer, 512-wide model. The export's identity scan missed this.
- evidence: config.json train: {"rung": "D", "window": 16, "arch": "theirs", "run_name": "BIG20M_othello_L", "d_model": 256, "n_layers": 4, "n_heads": 4, "warmup_frac": 0.05, ...}, plus top-level rung and w16_reference_steps 95100. The torch.load metadata of both best_model.pt shows the same train_config with run_name BIG20M_othello_L.
- fix: In the STAGING copies, rewrite both config.json to the standard schema: arch, model {vocab 61, block_size 59}, train = TrainConfig fields (steps 780000, batch 256, lr 1e-3, wd 1e-4, clip 1, constant, warmup 2000, ckpt_base 1000, val_every 5000, seed 0), data with corpus datasets/othello/standard/train/train_20000000.npz, n_params, steps_per_epoch, epochs; keep replicate on __seed0. Re-save both checkpoints with train_config replaced and rung/best dropped, leaving model_state byte-identical (fingerprint hashes only the state dict). Regenerate SHA256SUMS and MANIFEST.json.

### [major] (anon-semantic) gms-release-artifacts/datasets/rayworld/*/*/*.h5 (28 files)
- issue: Every Rayworld HDF5 config_json attribute keeps a generated_at local wall-clock timestamp, which reveals the research timeline and violates 'No dates or timestamps anywhere'.
- evidence: standard/eval/test.h5 generated_at 2026-08-31T17:38:17 … 128-ray/probe/probe_250k.h5 2026-09-15T20:58:20. export.md:169 kept them deliberately. core-ray-fix.md:21-23 confirms no code writes or reads the field.
- fix: For each STAGING .h5, open with h5py mode 'r+', load config_json, delete the 'generated_at' key and write it back with json.dumps(indent=2). The datasets stay untouched. Regenerate SHA256SUMS and MANIFEST.json.

### [major] (completeness) paper/paper_draft.tex:226 vs RELEASE pim/probes/base.py:192
- issue: The paper defines the categorical Probe Skill baseline as the most common class of each cell; the code uses one majority class pooled over all cells.
- evidence: base.py computes majority_class_error_rate = 1 - bincount(y_tr.reshape(-1)).max()/y_tr.size. The othello/standard cached stats give 53.12% at every point, and 1-err/53.12 reproduces scores.json exactly. On 4000 probe games the per-cell majority error is 46.13%, which would make Table 1 standard Lin/MLP 0.971/0.972 instead of 0.975/0.976. This affects every Othello and categorical Probe Skill.
- fix: Paper line 226: 'with the most common class over all cells of the fit split as the trivial predictor'. Keep the code, which produced every number.

### [major] (completeness) paper/paper_draft.tex:200,465,469,519,1058; RELEASE README.md (step 2); pim/training/train.py:201; scripts/make_replicate_member.py:33
- issue: Runs are evaluated at their lowest-validation checkpoint, not at 780k / 512k steps, and seed-0 members use a different rule (exactly 512k) from seeds 1 and 2 (best at or below 512k).
- evidence: best_model.pt steps: 8-ray 555000, 5-ray 450000, standard-noflip 745000; 8-ray__seed1 390000, 5-ray__seed1/2 450000, adjacent-noflip__seed2 460000; every __seed0 is exactly 512000 (a copy of ckpt/step_000512000.pt). So line 469's 'Table 2 reports the same runs at 780,000 steps' is false for 8-ray and 5-ray.
- fix: Paper lines 200 and 1058: add 'we evaluate the checkpoint with the lowest validation loss (checked every 5,000 steps)'. Lines 465 and 519: 'the two new models, each at its best checkpoint within 512,000 steps, and the main run's checkpoint at exactly 512,000 steps'. Line 469: drop 'at 780,000 steps'. Add the same sentence to README step 2.

### [major] (completeness) paper/paper_draft.tex:209,213,1064 vs RELEASE pim/editors/grad_steer.py:77,105
- issue: GS is described as gradient descent with step η, but the code runs 100 Adam steps with learning rate η × probe.act_scale (the median per-dimension std of the activations).
- evidence: _descend: torch.optim.Adam([v], lr=alpha); the hook passes step = alpha * probe.act_scale. The master_eval SETTINGS note says 'GS's [alpha is] a fraction of the activation scale'.
- fix: Paper line 213: 'takes 100 Adam steps on the latent state with learning rate η times the median standard deviation of the latent state at that point'. State the scaling next to the η grid in line 1064.

### [major] (completeness) paper/paper_draft.tex:221 vs RELEASE scripts/make_edit_selection.py:35 and every datasets/rayworld/*/edits/selection.json
- issue: Edit cases are kept only if the edit changes at least 2 rays; the paper says only unchanged edits are skipped.
- evidence: --min-rays default 2, and 'min_rays': 2 in all 8 shipped selections. 5-ray pool: 6000 cases, 1501 identical, only 2489 kept, so about 2010 one-ray edits are excluded. 8-ray pool: 4000, 693 identical, 2414 kept.
- fix: Paper line 221: 'We skip cases whose edit changes fewer than two rays of the next frame (Rayworld) or leaves the legal moves unchanged (Othello)'.

### [major] (completeness) paper/paper_draft.tex:825 (no release producer)
- issue: The Othello IM reconstruction test (within about 2× the model's own error at points 4–5, 5–16× elsewhere) cannot be reproduced with the release.
- evidence: grep -rni recon in pim, scripts and notebooks finds only Rayworld reconstruct_clean_obs. The numbers come from PRIVATE experiments/inverse_probe/scores/othello_L-oth-20m_mirror128_recon.json, which is not shipped and not in SPEC scope.
- fix: Add scripts/im_reconstruction.py: write g(unedited board) at each point with pim.editors.inverse and the shipped IM cache, store the error against the pre-edit legal set relative to the model's own in runs/othello/standard/im_reconstruction.json, and add tables.im_reconstruction() plus an appendix cell. Or delete the reconstruction sentences from the paper.

### [major] (completeness) paper/paper_draft.tex:1075 and figs more_qualitative_edits_othello_1-5 vs RELEASE scripts/figures/qualitative_othello.py
- issue: The release Othello appendix figures differ from the paper's in 3 cells per figure, and the paper text describes the old boards.
- evidence: Pixel diffs against PRIVATE othello_edits_seed{1..5}_cols.png: 95,139 / 108,572 / 96,176 / 102,011 / 90,871, all in adjacent-noflip GS (pt2 α0.05 vs pt2 α1.5), standard-noflip GS (pt4 α0.05 vs pt0 α1.5) and standard-noflip IM (pt1 vs pt7). The paper says those cells show 'the setting with the highest Edit Index instead of the table's fallback'.
- fix: Regenerate the five paper figures with the release script (it matches Table 2's rule) and delete that sentence from line 1075.

### [major] (completeness) RELEASE README.md:84; pim/scoring/rayworld.py:66-75
- issue: The documented rescoring recipe ('delete its scores.json (and its probes/ to refit them) and run it again') silently drops the categorical blocks.
- evidence: _fit loads categorical probes with require_cached=True. On a miss it prints 'SKIPPED' and writes scores.json without that block, so Table 2's categorical rows and tab:categorical would show '—' for that run.
- fix: README:84: '…and its probes/ to refit them; then first rerun that run's scripts/fit_probes.py lines from step 3 (the scorer reads categorical probes and their floors from the cache and skips a block whose probes are missing)'.

### [major] (reviewer) README.md
- issue: The rescore recipe ('delete its scores.json (and its probes/ to refit them) and run it again') silently drops every categorical block. The scorer never fits categorical probes, it only reads them from the cache, so the Table 2 categorical rows, tab:categorical and tab:tokens_* cannot be rebuilt.
- evidence: README.md:84-86; pim/scoring/rayworld.py:66-75 catches RuntimeError and skips. With an empty cache, _fit(8-ray model, 'appearance-fac', ...) printed 'appearance-fac: SKIPPED (no cached probes for {...})' and returned None.
- fix: After '(and its `probes/` to refit them)' add: 'If you delete `probes/`, first refit its categorical probes with that run's lines from step 3 of From scratch (`scripts/fit_probes.py`).'

### [major] (reviewer) pim/scoring/summary.py
- issue: Nothing says how to read scores.json, and the obvious fields are not the paper's numbers. Othello's `edit_index` is the union construction (the paper reports `edit_index_symdiff`). `best` is the top arm without the fidelity cutoff. The master_eval summary's 'fid' column is the fidelity ratio (1 - Edit Fidelity).
- evidence: jq .best.PI runs/othello/standard/scores.json gives edit_index 0.552 and edit_index_symdiff 0.818 (paper +0.82). print_summaries shows 8-ray IM 'fid 0.265' against the paper's 0.73. summary.py:18-24 and 36-37.
- fix: summary.py: rename the header 'fid' to 'ratio' (or print 1-ratio as 'Fid.') and relabel EI(sd) as 'EI (reported)'. README: add a 'Reading scores.json' paragraph: tables apply pim.metrics.selection.best_arm to `arms`; `best` is the top arm without the cutoff; Edit Fidelity = 1 - fidelity_ratio; Othello's Edit Index is edit_index_symdiff; `bases` maps each block key to its scores.

### [major] (reviewer) scripts/demos/demo.py
- issue: The documented usage line fails.
- evidence: demo.py:4 (shown in --help): '--seed 7 --n-objects 4 --fixed-reflectivities' raises 'RuntimeError: Could not generate a collision-free scene after 300 attempts'. With 4 objects and 100 frames, seeds 0-2, 4-6 and 8-11 simulate; 3 and 7 fail.
- fix: Change the usage line to 'python scripts/demos/demo.py --seed 8 --n-objects 4 --fixed-reflectivities [--save outputs/demo.gif]'.

### [major] (reviewer) scripts/figures/qualitative_othello.py
- issue: The Othello appendix figures (and the overview's panel b) draw best_arm's highest-fidelity fallback. The paper says that where no setting reaches Edit Fidelity 0 (GS on both no-flip variants, IM on standard-noflip), they show the highest Edit Index setting. Already reported by figures and figures-recheck, still open.
- evidence: qualitative_othello.py:81 uses best_arm; the paper appendix 'Additional Qualitative Editability Visualizations' paragraph and its % figure facts comment say otherwise.
- fix: Edit the paper sentence and its comment to say the figures use Table 2's setting, including the fallback (paper, not a release file).

### [minor] (tables) paper/paper_draft.tex
- issue: Line 467 says the daggered spread 'comes mostly from selecting a different setting on each seed'. That holds only for IM on adjacent-noflip; for both PI cells two of the three seeds pick the same setting. The builder's report overstates this as 'a different setting on each seed | same'.
- evidence: My raw best-arm settings per seed: adjacent-flip PI pt2 α5, pt2 α5, pt1 α10; adjacent-noflip PI pt1 α10, pt1 α10, pt1 α20; adjacent-noflip IM pt2, pt0, pt1. T.dagger_cells shows the same.
- fix: '...and their spread comes mostly from the selected setting changing between seeds rather than from the models.'

### [minor] (tables) paper/paper_draft.tex
- issue: Line 348 says 'as the rays coarsen both probe-derived editors rise steadily, to +0.51 and +0.60 at 5 rays', but categorical GS falls from 128-ray to 16-ray.
- evidence: Categorical GS Edit Index for 128/16/8/5-ray: +0.308, +0.280, +0.458, +0.598. The seed means also fall (+0.286 to +0.255), with seed SDs 0.014 and 0.006, about 2 combined SDs. PI does rise steadily: -0.31, -0.06, +0.38, +0.51.
- fix: '...and as the rays coarsen PI rises steadily and GS rises from 16 rays on, to +0.51 and +0.60 at 5 rays.'

### [minor] (tables) notebooks/appendix_tables.ipynb
- issue: The notebook announces and prints numbers the edited paper no longer quotes: '49 of 1000 categorical cases removed' / 'categorical cases kept 951', and the two-flip partners searched (0/480/0).
- evidence: Cell tokens-numbers-md (line 330): 'Trained probes within 0.013 of the frame model; 49 of 1000 categorical cases removed'. pim/figures/tables.py:779 returns 'categorical cases kept'. tables.py:720 returns **d['partners_searched']. In the non-comment text of paper_draft.tex, grep finds no \b49\b outside table cells, and no 480 or 951; the sentences were removed in the uncommitted edit.
- fix: Change the header to 'Trained probes within 0.013 of the frame model'. Drop the 'categorical cases kept' entry from tokens_numbers, and drop **d['partners_searched'] from two_flip_numbers (keep n_cases and max Edit Index SE). Alternatively, restore the two phrases in the paper.

### [minor] (tables) gms-release-artifacts/runs/othello/adjacent-flip/variance.json
- issue: Two shipped variance.json files are read by no table and produced by no documented command: othello/adjacent-flip/variance.json and rayworld/standard/variance.json.
- evidence: tables.probe_refit_spread reads only othello/standard, othello/adjacent-flip__seed0/1/2 and rayworld/8-ray. The README refit loop is `for R in rayworld/8-ray othello/standard othello/adjacent-flip__seed{0,1,2}`. rayworld/standard/variance.json holds full (20 seeds) and appearance-fac (6 seeds), neither quoted in the paper.
- fix: Leave these two files out of the bundle and out of MANIFEST.json/SHA256SUMS in export_artifacts.py, or add the two runs to the README loop.

### [minor] (rescore) /home/sevan/research/PIM/generative-models-as-simulators/README.md
- issue: The rescore instruction ('delete its scores.json (and its probes/ to refit them) and run it again with the corpora bundle in place') silently loses categorical blocks. The scorer never fits categorical probes, and probe_250k for standard and blink is not in the corpora bundle, so their appearance-fac blocks cannot be rebuilt from the downloads.
- evidence: In treeB (standard, probes deleted) the notebook printed 'appearance-fac: SKIPPED (no cached probes for {...target: appearance-f...)'. The next dry run printed 'WOULD add to rayworld/standard: blocks ['appearance-fac']  IM on []' and will do so on every later run. qualitative_rayworld.py:147 then leaves the Standard categorical cells blank. The corpora bundle ships probe_250k only for 128/16/8/5-ray.
- fix: At README.md line 84-86, after '...to refit them)', add: 'The scorer never fits categorical-target probes: keep their files, or refit them first with scripts/fit_probes.py (From scratch, step 3). For standard and blink this needs scripts/generate_dataset.py --instance I --role probe --size 250k, which is not in the corpora bundle.'

### [minor] (rescore) /home/sevan/research/PIM/gms-release-artifacts/runs/rayworld/{16-ray,5-ray,8-ray,blink,smooth,standard}/scores.json (frustum) and obs5/scores.json (cartesian)
- issue: 7 shipped blocks carry bench_selection: null. That contradicts the schema comment at pim/scoring/blocks.py:111 ('None = the first n cases'): these blocks were scored on the selected cases.
- evidence: A fresh score of 8-ray and standard frustum writes a selection record (rule 'the FIRST 1000 cases ... >= 2 rays ...', n 1000, min_rays 2, pool 4000), and every frustum arm matches the shipped arms bitwise. The fresh frustum record equals the shipped cartesian.bench_selection for 8-ray and for standard.
- fix: In experiments/release/export_artifacts.py, set each null bench_selection to the fresh-score record: the same run's cartesian.bench_selection for the frustum blocks, and for obs5 cartesian the record from the scoring worker's obs5 rescore. Then regenerate MANIFEST and SHA256SUMS.

### [minor] (pipeline) README.md:107
- issue: The disk and time needed for a full rebuild are missing or understated. The '410 GB' for a 128-ray corpus counts obs.f32 only.
- evidence: A 128-ray obs.f32 is 409.6 GB, and meta.h5 adds about 26 GB per instance (26.08 GB for 8-ray). The eight corpora total about 2.1 TB. The PRIVATE 8-ray build took about 2 h (shard timestamps 18:24 to 20:28).
- fix: State 'about 2.1 TB for the eight Rayworld corpora (a 128-ray one about 440 GB) and hours per corpus on 32 cores; probe fits also use tens of GB under .scratch/'.

### [minor] (pipeline) scripts/train.py:97
- issue: Othello training without --limit silently starts regenerating the full 20M-game corpus when only a smaller train_<n>.npz exists, with no output for minutes.
- evidence: With train_20000.npz present, 'train.py --env othello --instance standard --run othello/nolimit-test --steps 200' printed nothing for 15 s and 25 s before timeouts, because oc.build(20M) was generating.
- fix: In _othello_tokens, if neither train_{n}.npz nor a larger file exists: raise SystemExit('no train split of n games; run scripts/make_othello_corpus.py --instance ... [--n-train N] and pass --limit N').

### [minor] (pipeline) README.md:84
- issue: The Scoring paragraph omits two consequences of a rescore. Deleting a Rayworld run's probes/ also drops its categorical probes, so their blocks are silently skipped unless step 3's fit_probes.py lines are rerun. The figure caches are keyed by (instance, seed) only, so they go stale after a rescore.
- evidence: master_eval prints 'SKIPPED (no cached probes ...)' when the categorical fits are missing. qualitative_rayworld.py columns() reuses outputs/cache/qualitative_rayworld.pkl keyed by (inst, seed).
- fix: Add: 'For a Rayworld run, rerun its step-3 fit_probes.py lines before rescoring, and pass --recompute to the figure scripts afterwards.'

### [minor] (pipeline) README.md:71
- issue: The README says the Othello probe games are regenerated 'in a few seconds'.
- evidence: qualitative_othello.py took 82 s without the corpora bundle and 24 s with it, so regeneration takes about 58 s.
- fix: Replace with 'in about a minute'.

### [minor] (pipeline) STAGING datasets/rayworld/*/{eval/test.json,edits/edits.json,probe/probe_*.json}
- issue: The shipped JSON sidecars are old four-split-suite manifests, while generate_dataset.py --role writes single-role manifests. The data are bit-identical.
- evidence: The shipped 8-ray eval/test.json has 'splits': train 100 / val 10000 / test / edits, no 'role', and a sim block without the blink, n_observers and region fields. The regenerated file has role 'eval' and one split.
- fix: Rewrite the sidecars in the regenerated format during export, or document that only 'sim' is read.

### [minor] (pipeline) STAGING datasets/rayworld/{standard,blink,smooth,obs5}/probe/
- issue: There is no probe_250k.h5 for these four instances, although their shipped observation_right_large floors were fit at n_seq 250000. Refitting their floors from the corpora bundle silently drops that floor.
- evidence: The standard baselines.json cartesian observation_right_large has n_seq 250000. The tiny run shows 'obs_right_large +nan' when the 250k file is absent.
- fix: Ship the four probe_250k.h5 files (about 1.5 GB each at 128 rays), or note in the bundle table that refitting those floors needs generate_dataset.py --role probe --size 250k.

### [minor] (static) pim/editors/pinv.py:41
- issue: Unreferenced definitions are shipped.
- evidence: Reachability plus grep, no caller anywhere: PinvMap and pinv_maps (pinv.py:41,50); floor_bracket (metrics/prediction.py:58, still named in the module docstring); layout.probe_dir/eval_dir/othello_split_file/INSTANCES (layout.py:52,74,121,16; the messages at lines 63 and 83 point callers to the dead othello_split_file); bigcorpus.instance_dir (106); othello/corpus.BLOCK (26); World.n_rays (rayworld/bayes.py:57); FrameVocab.obs_dim (tokens.py:54).
- fix: Delete each, with its __all__ and __init__ exports. Change the layout error messages to name othello_split_dir. Keep Table._repr_png_, which Jupyter calls.

### [minor] (static) pim/editors/pinv.py:58
- issue: Several parameters now accept only one value; they are left over from removed options.
- evidence: pinv_step(space='zspace') raises otherwise and is threaded through rayworld/arms.py:348,378, token_bench.py:173 and scoring/rayworld.py:100,153. frame_probs(kind='logits') (token_bench.py:78) and TransformerL.output_kind (transformer_l.py:40) are what is left of the raw head. Also linear_arm(mode='pinv') (othello/arms.py:252), _descend/make_intervention_hook(optimizer='adam') (grad_steer.py:69,93), and _seq_mask returning None with _run ignoring attn_mask (transformer_l.py:75,86).
- fix: Drop these parameters and output_kind; frame_probs becomes a softmax. Keep the 'PI[zspace]' editor label (scores.json schema) and fit_probe_grid's targets/splits (probe-cache key). Note that dropping pinv_step's space is a public signature change under invariant 4, though numerically a no-op.

### [minor] (static) pim/scoring/driver.py:69
- issue: missing_inverse/add_inverse only add IM to a scores.json written without it, and re-implement the IM record assembly.
- evidence: score_othello (othello.py:86) and score_rayworld (via inverse_rayworld) already compute IM in a full score. add_inverse's Othello branch (lines 71-91) filters records with np.isscalar and picks best with top_arm, where score_othello uses unfiltered records and max, so the two paths can write differently shaped records.
- fix: Delete missing_inverse, add_inverse and the missing_im branch of _complete; keep missing_blocks. If it stays, make it share the full scorer's assembly helper.

### [minor] (static) scripts/reachability_table.py:35
- issue: Scorer settings are copied as literals into six scripts.
- evidence: PROBE_GAMES, GS_STEPS, GS_BETA = 20_000, 100, 0.2 in reachability_table.py:35, two_flip_editability.py:36 and figures/qualitative_othello.py:38. PROBE_SEQS = 30_000 in figures/qualitative_rayworld.py:50 and history_rewrite.py:37. RW_PROBE_SEQS, RW_BENCH_N, OTH_PROBE_GAMES in probe_refit_variance.py:28. train.py:32 DEFAULT_INSTANCE duplicates layout.DEFAULT_INSTANCE.
- fix: Define the defaults once (e.g. pim/scoring/settings.py DEFAULTS), used by master_eval SETTINGS and every script; use layout.DEFAULT_INSTANCE in train.py.

### [minor] (static) pim/environments/layout.py:1
- issue: Paths are spelled outside layout, although its module docstring says no other module spells one.
- evidence: runs/_baselines paths are built by hand in scripts/fit_probes.py:73, reachability_table.py:48, othello_flip_rates.py:37 and bayes_floor.py:58. rayworld/tokens.py:86 defines its own tokens_dir(instance_dir), and lines 108 and 110 spell train/corpus.json and train/obs.f32.
- fix: Use layout.baselines_dir, layout.tokens_dir and layout.train_dir; delete tokens.tokens_dir.

### [minor] (static) pim/environments/othello/corpus.py:205
- issue: The module has a second, undocumented CLI and a constant named after the data-size ladder.
- evidence: The `if __name__ == "__main__"` block and the 'Usage: python -m ...' line (11) duplicate scripts/make_othello_corpus.py. LADDER = {"D": 20_000_000} at line 70.
- fix: Delete the main block and the Usage line. Rename LADDER["D"] to N_TRAIN at its 8 call sites (corpus.py 140/206, scoring/othello.py 24/75, othello/bayes.py 54/57, train.py:97, make_othello_corpus.py:26, figures/qualitative_othello.py:56).

### [minor] (static) pim/environments/othello/reachability.py:1
- issue: Many docstrings are longer than the spec's limits (modules 1-4 lines, functions 1-3).
- evidence: Module docstrings: reachability.py 24 lines, qualitative_overview 14, qualitative_rayworld 13, history_rewrite 12, qualitative_othello 11, othello/corpus 11, predictions 10, othello/__init__ 10, scoring/baselines 9; 38 modules exceed 4 lines. 19 function docstrings exceed 3 lines (othello/arms.py:332 inverse_arms has 11).
- fix: Keep the purpose line plus the usage line (the --help text) in scripts, and move figure descriptions to the captions. Trim reachability.py to its first paragraph plus the verdict definitions.

### [minor] (static) scripts/bayes_floor.py:1
- issue: CLI flag names and meanings differ across scripts.
- evidence: --instance takes ENV/INSTANCE (nargs +) in bayes_floor.py and bare names elsewhere. --n-workers (generate_dataset) vs --workers (build_rayworld_corpus, reachability_table, two_flip_editability). --obs-noise (play.py) vs --obs-noise-std (demo.py, generate_dataset). --seeds means values in the figure scripts and counts in probe_refit_variance. --force vs --recompute. Rayworld --instance has choices only in build_rayworld_corpus.
- fix: Use --instances for the env-qualified list, --workers everywhere, --obs-noise-std, --n-seeds for counts, one of --force/--recompute, and choices=sorted(bigcorpus.INSTANCES) on every Rayworld --instance.

### [minor] (static) scripts/figures/editability_by_point.py:1
- issue: The script has no argparse, so --help runs it and writes the figure.
- evidence: Running it with --help wrote outputs/figures/appendix/editability_over_res_point.{pdf,png} in the work copy.
- fix: In main(), add argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args().

### [minor] (static) scripts/make_othello_edits.py:51
- issue: British spellings appear in shipped prose and in four shipped artifacts.
- evidence: 'one occupied non-centre tile recoloured' is written into all four staged datasets/othello/*/edits/cases_1000.json (bench.py:138 says 'non-center'). 'labelled' at grid_target.py:1,30. Identifiers: CENTRE (othello/data.py:19, exported), synthesise_cases (bench.py:89), FactorisedTarget (grid_target.py:298), centres (grid_target.py:276).
- fix: Use 'non-center tile recolored' in the script, and have the export worker patch the four manifests' recipe strings (no code reads them). Use 'labeled'. Rename CENTER, synthesize_cases and FactorizedTarget if public-API renames are acceptable.

### [minor] (anon-lexical) /home/sevan/research/PIM/generative-models-as-simulators/{runs/othello,runs/rayworld,runs/_baselines,datasets/othello,datasets/rayworld}
- issue: The five test symlinks point to absolute paths that include the username. They are gitignored, but tar and zip keep link targets.
- evidence: `readlink runs/othello` gives /home/sevan/research/PIM/gms-release-artifacts/runs/othello; the other four are the same pattern.
- fix: Remove the symlinks before packaging, or package only via git archive.

### [minor] (anon-lexical) /home/sevan/research/PIM/gms-release-artifacts/runs/othello/standard/config.json (and standard__seed0/config.json, and both best_model.pt)
- issue: A legacy internal run name and research-ladder fields ship as history. The same train block also contradicts the model block (d_model 256 / n_layers 4 vs n_embd 512 / n_layer 8).
- evidence: config.json:3 and :37 `"rung": "D"`, :8 `"run_name": "BIG20M_othello_L"`, :45 `"w16_reference_steps": 95100`, plus `"arch": "theirs"`. The checkpoint train_config holds the same values.
- fix: Drop train.run_name, train.rung, the top-level rung and w16_reference_steps from both config.json files (grep over pim/, scripts/ and notebooks/ shows no reader). Optionally re-save both checkpoints without these keys: the fingerprint hashes only state_dict, so probe keys do not change. Regenerate SHA256SUMS and MANIFEST afterwards.

### [minor] (anon-semantic) generative-models-as-simulators/pim/ (package name)
- issue: The package name pim is the acronym of the private project/repo PhysicallyImplicitModeling (github.com/fisheye-sonar/PhysicallyImplicitModeling) and is never explained in the README. If that repo or any talk uses 'PIM' publicly, the name links the release to its authors.
- evidence: README line 37: 'installs the `pim` package'. The private repo remote is fisheye-sonar/PhysicallyImplicitModeling. All 1202 probe pickles reference pim.probes.base.
- fix: If the private repo stays private, keep the name and accept the risk. Otherwise rename to a neutral name (e.g. gms) and re-save the 1202 probe blobs under the new module path. Probe filenames hash only the provenance dict, so they do not change.

### [minor] (anon-semantic) generative-models-as-simulators/pim/environments/othello/corpus.py
- issue: Internal jargon for an unpublished data-scale ladder ('rung D') in shipped code.
- evidence: corpus.py:70 LADDER = {"D": 20_000_000}, used at corpus.py:140,206; othello/bayes.py:54,57; scoring/othello.py:24,75; scripts/train.py:97; scripts/make_othello_corpus.py:26; scripts/figures/qualitative_othello.py:56.
- fix: Replace it with N_TRAIN_GAMES = 20_000_000 and use oc.N_TRAIN_GAMES at every call site. The numerics are unchanged.

### [minor] (anon-semantic) generative-models-as-simulators/pim/environments/othello/corpus.py
- issue: A docstring describes the authors' machine, which the spec forbids.
- evidence: corpus.py:144 'CPU only, about 4.7k games/s on 32 cores (20M train games in about 70 min).'
- fix: Replace with 'CPU only; games are generated in parallel, one process per core.'

### [minor] (anon-semantic) gms-release-artifacts/datasets/rayworld/*/*/*.json and runs/othello/*/config.json
- issue: Layout versioning and old-layout paths remain in shipped JSON.
- evidence: 'layout': 2 in 18 manifests (e.g. datasets/rayworld/128-ray/edits/edits.json). data.corpus 'datasets/othello/<inst>/corpus/train_20000000.npz' in adjacent-noflip, adjacent-noflip__seed0, adjacent-flip, adjacent-flip__seed0, standard-noflip and standard-noflip__seed0 config.json.
- fix: Drop the 'layout' key, and rewrite 'corpus/' to 'train/' in those 6 config.json files.

### [minor] (anon-semantic) generative-models-as-simulators/scripts/make_othello_edits.py
- issue: British spellings in shipped prose and identifiers (a locale tic; the spec requires American), one of them written into shipped artifacts.
- evidence: make_othello_edits.py:51 'one occupied non-centre tile recoloured', which appears in all 4 datasets/othello/*/edits/cases_1000.json; othello/data.py:19 CENTRE; othello/bench.py:89 synthesise_cases; rayworld/grid_target.py:298 FactorisedTarget, :1,30 'labelled', :276 centres.
- fix: Change the prose to 'non-center tile recolored', 'labeled' and 'centers', including in the 4 STAGING cases_1000.json. Rename the identifiers to CENTER, synthesize_cases and FactorizedTarget; none are pickled.

### [minor] (completeness) paper/paper_draft.tex:213,348
- issue: 'At α = 1 the linear probe reads the edited state exactly' is false at residual point 0, and no notebook displays it.
- evidence: scores.json cartesian PI α=1 readout_err_after at point 0: 1.62 standard, 1.70 blink, 1.66 128-ray, 2.53 16-ray, 2.64 8-ray, 2.44 5-ray. At points 1–8 it is 1e-6 to 5e-6.
- fix: Paper: '…lands exactly at every point after the embedding'. Add a tables.pi_landing(F) cell to paper_tables.ipynb that prints readout_err_after of the α=1 arms.

### [minor] (completeness) paper/paper_draft.tex:170; RELEASE notebooks/paper_tables.ipynb
- issue: The main-text flip rates (0.27 vs 2.2 per move) have a producer and an artifact but no notebook renders them.
- evidence: runs/_baselines/othello/*/corpus_stats.json is read by no module or notebook (grep). My rerun of othello_flip_rates.py reproduces 2.2449 / 0.2687 byte for byte.
- fix: Add tables.flip_rates(runs), reading corpus_stats.json flips_per_move, and a paper_tables cell 'Flips per move (0.27 vs 2.2)'.

### [minor] (completeness) paper figures othello_overview.pdf / rayworld_overview.pdf (Figs. 3, 4)
- issue: Two data-driven paper figures have no release script.
- evidence: PRIVATE paper/figs/environments_overview/{othello,rayworld}/make_figure.py draw them from bench games and eval/test.h5. scripts/figures/ has no counterpart.
- fix: Port them as scripts/figures/environments.py, or say in the README that Figures 1–4 are illustrations with no script.

### [minor] (completeness) paper/paper_draft.tex:290 vs RELEASE pim/figures/tables.py floor_cells, pim/scoring/baselines.py:30
- issue: The Table 1 caption says the baselines are fit 'on the same sequences', but the observation floor is observation_right_large.
- evidence: That floor is fit on history aligned to the present, on 250k Rayworld sequences or 170k Othello games at 50 epochs. The trained probes use 30k / 20k at 200 epochs.
- fix: Caption: '…and an observation baseline fit to the history aligned at the present on a larger corpus (250,000 Rayworld sequences, 170,000 Othello games)'. Mention it in app:implementation.

### [minor] (completeness) STAGING datasets/rayworld/{standard,blink,smooth,obs5}/probe/; RELEASE README.md bundle table
- issue: The large observation floors of four instances were fit on probe_250k corpora that are in no bundle.
- evidence: STAGING ships probe_250k only for 128/16/8/5-ray, but baselines.json observation_right_large for standard, blink, smooth and obs5 is at n_seq 250000. The README says the corpora bundle is what refitting probes needs.
- fix: README: 'the large observation floors of standard, blink, smooth and obs5 need generate_dataset.py --role probe --size 250k first'. Or ship those corpora (about 8 GB).

### [minor] (completeness) paper/paper_draft.tex:471; RELEASE scripts/probe_refit_variance.py:92,109; pim/figures/tables.py:638
- issue: The probe-refit PI spread is measured at the unguarded highest-Edit-Index step size, not under the paper's selection rule.
- evidence: PI@canonical_edit_best = max(arms, key=edit_index). On the adjacent-flip seeds it picks α 10–35 with fidelity ratios 1.94 / 2.11 / 2.67 (Edit Fidelity -0.9 to -1.7).
- fix: Paper: '…PI's Edit Index at its highest-index step size at the main run's point'. Or switch the script to pim.metrics.selection.best_arm, which needs a rerun.

### [minor] (completeness) STAGING runs/othello/adjacent-flip/variance.json, runs/rayworld/standard/variance.json, runs/rayworld/8-ray/variance.json ('full'), runs/_baselines/othello/*/corpus_stats.json
- issue: These artifacts are read by no code, and README step 5 regenerates neither the two variance files nor 2 of the 4 corpus_stats files.
- evidence: tables.probe_refit_spread reads only othello/standard, adjacent-flip__seed0/1/2 and rayworld/8-ray appearance-fac. No reader of corpus_stats.json exists. othello_flip_rates.py defaults to standard and adjacent-flip.
- fix: Remove them from the export or render them. In README step 5 use --instance standard adjacent-flip adjacent-noflip standard-noflip.

### [minor] (completeness) paper/paper_draft.tex:695 (tab:tokens_editability caption)
- issue: The token model's categorical rows use 951 cases, while the paper implies 1000.
- evidence: 8-ray-tokens scores.json bases.appearance-fac.n_cases_kept = 951; the cartesian token block keeps 1000.
- fix: Add to the caption: 'The categorical token rows use the 951 cases whose edit changes the next frame's token.'

### [minor] (completeness) paper/paper_draft.tex:852,855
- issue: Two stale cells in tab:im_by_point.
- evidence: Othello IM Index at points 3 and 6: scores.json 0.2549 / 0.3849, paper +0.26 / +0.39.
- fix: Change them to +0.25 and +0.38.

### [minor] (completeness) paper/paper_draft.tex:996
- issue: 'IM … moves by at most 0.10 in Edit Index' understates the shift.
- evidence: fidelity_rule_shift: adjacent-flip +0.664 → +0.558, a shift of 0.106.
- fix: Say 'by at most 0.11'.

### [minor] (completeness) paper/paper_draft.tex:348
- issue: 'Both probe-derived editors rise steadily' as rays coarsen on the categorical target is not true for GS.
- evidence: Categorical GS: 128-ray +0.31, 16-ray +0.28, 8-ray +0.46, 5-ray +0.60. The first step drops by 0.03, above the seed SDs of 0.014 / 0.006.
- fix: '…PI rises steadily and GS rises from 16 rays on, to +0.51 and +0.60 at 5 rays'.

### [minor] (completeness) paper/paper_draft.tex:406 vs RELEASE pim/environments/rayworld/bayes.py
- issue: The paper describes the Bayes-floor sampler as simple rejection; the code uses SMC with Metropolis–Hastings rejuvenation and an exact frame-0 term, and none of its settings are in the paper.
- evidence: bayes.py: 512 particles, 500 initial sweeps, 40 per frame, first 1000 eval sequences. These are also the settings in the shipped bayes_floor.json files.
- fix: Add to app:implementation: 'sequential Monte Carlo over frames with 512 particles and Metropolis–Hastings rejuvenation, on the first 1000 held-out sequences'.

### [minor] (completeness) STAGING runs/othello/standard/config.json and runs/othello/standard__seed0/config.json
- issue: A legacy train block contradicts the shipped architecture.
- evidence: train.d_model 256, n_layers 4, n_heads 4, warmup_frac 0.05, run_name 'BIG20M_othello_L', window 16, rung 'D', w16_reference_steps, train.arch 'theirs'. The checkpoint's model_config is 8/8/512, and the other 41 configs use the TrainConfig schema.
- fix: In export_artifacts.py, drop those keys or rewrite the train block of these two configs to the TrainConfig schema.

### [minor] (reviewer) README.md
- issue: Internal jargon is used without a gloss: block, arm, bench, basis, guard/within_guard, zone, gates, floors (for the decodability baselines, where the paper uses 'floor' only for the Bayes floor), instance vs variant, and Transformer-L (never used in the paper).
- evidence: README:83,86,181 'block'; README layout 'edit benches and editor sweeps'; README:232 'Transformer-L'; notebook headings 'reported blocks'; fit_probes --basis 'the scorer's first basis'; within_guard columns in the appendix output; summary 'gates:'.
- fix: Add a 'Terms' table after Repository layout covering Transformer-L (8 blocks, width 512, minGPT body), instance, bench, arm, block, basis, guard/cutoff (Edit Fidelity >= 0, i.e. fidelity_ratio <= 1), zone, gates, floors and appearance-fac. In paper_tables.ipynb cell collect-md write 'every run's reported scores'. In the arms.py:1 docstring add 'an arm is one editor at one residual point and step size'.

### [minor] (reviewer) pim/environments/rayworld/sim.py
- issue: The simulator state is called 'latent state', which is the paper's (and the teaser's) word for the model's z. The same wording appears in on-screen demo titles.
- evidence: sim.py:1 and :24, dataset.py:1, viz.py:118 '2D environment  (latent state)', scripts/demos/play.py:115 '2D world  (latent state)'.
- fix: Replace 'latent state' with 'simulator state' in those five places.

### [minor] (reviewer) scripts/two_flip_editability.py
- issue: Othello pieces are called disc/tile/square where the paper says 'token'. 'Disc' also collides with Rayworld's discs.
- evidence: two_flip_editability.py:2,5 'Two-disc', 'disc counts'; othello_flip_rates.py:2,42 'Discs recolored', 'discs flipped per move'; README:188 'the two-disc edits'; tables.py:710 and the appendix_tables.ipynb heading 'Two-tile edits'.
- fix: Use 'two-token edits', 'token counts' and 'tokens flipped per move' in each place.

### [minor] (reviewer) pim/editors/grad_steer.py
- issue: The GS docstring gives plain gradient descent, but the code runs Adam at learning rate alpha*act_scale. The paper also says only 'gradient steps of size η'.
- evidence: grad_steer.py:3 'x' <- x - alpha * dL(p(x), B') / dx'; _descend uses torch.optim.Adam(lr=alpha); the hook passes step = alpha * probe.act_scale.
- fix: Docstring: '``n_steps`` Adam steps on the activation at learning rate ``alpha`` times the point's activation scale (median activation SD), minimizing the probe loss toward B'.' Paper Implementation Details: note that GS uses Adam and that η is relative to the median activation SD.

### [minor] (reviewer) pim/figures/waterfall.py
- issue: waterfall_grid is dead code. No script or notebook calls it; style.py imports only three constants from the module. The tables report's claim that it serves the history-rewrite script is stale.
- evidence: grep finds no caller of waterfall_grid; scripts/figures/style.py:26 imports DARK_BG, DIFF_CMAP and EDIT_LINE only; history_rewrite.py uses style.waterfall.
- fix: Delete waterfall_grid and its export in pim/figures/__init__.py:6 (or move the three constants into style.py), and change the __init__ docstring to 'The paper's tables (``tables``)'.

### [minor] (reviewer) README.md
- issue: The main text's Othello flip rates (0.27 against 2.2 per move) appear in no notebook, yet README says the appendix tables read them.
- evidence: README:188-189; nothing in pim/ or notebooks/ reads corpus_stats.json. Values are in runs/_baselines/othello/*/corpus_stats.json (0.2687, 2.2449).
- fix: Either add a tables.py reader of corpus_stats.json plus a paper_tables.ipynb cell, or change README:188-189 to '…and the Othello flip rates quoted in Section 3.1 (runs/_baselines/othello/<instance>/corpus_stats.json)'.

### [minor] (reviewer) scripts/figures/editability_by_point.py
- issue: It is the only runnable script without argparse, so --help runs it and writes the figure.
- evidence: Running it with --help printed '-> outputs/figures/appendix/editability_over_res_point.pdf'; main() at line 69 parses no arguments.
- fix: At the top of main() add argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args().

### [minor] (reviewer) README.md
- issue: Resource needs are understated or ambiguous. Four instances cast 128 rays, and the corpora total about 1.9 TB. Scoring memory-maps residual stacks of about 22 GB each into .scratch/.
- evidence: README:107 'a 128-ray Rayworld corpus takes about 410 GB'; 20M x 40 x 128 x 4 B = 410 GB for each of standard/blink/smooth/128-ray; probes/base.py collect_residuals docstring gives about 22 GB; arms.py _scratch_dir writes to .scratch/.
- fix: 'Each 128-ray corpus (standard, blink, smooth, 128-ray) takes about 410 GB, and all eight about 1.9 TB. Scoring needs about 25 GB free for residual stacks in .scratch/.'

### [minor] (reviewer) README.md
- issue: The <ARTIFACTS_URL> placeholder in prose is stripped by GitHub as an HTML tag, so the sentence renders as 'are at . The download'.
- evidence: README.md:44.
- fix: Fill it before release, or write it as `<ARTIFACTS_URL>` in backticks until then. Make sure the HF repo id belongs to an anonymous account.

### [minor] (reviewer) notebooks/paper_tables.ipynb
- issue: 'Landing editors' relies on an unstated threshold (tables.LANDS = 0.25 seed-mean Edit Index). The paper's 'every editor that lands' does not define it either.
- evidence: paper_tables.ipynb:98 and appendix_tables.ipynb:110 headings; pim/figures/tables.py LANDS = 0.25.
- fix: Append '(an editor lands when its seed-mean Edit Index is at least 0.25)' to both headings and to the paper's metric-spread appendix.

### [minor] (reviewer) README.md
- issue: The README does not say that paper Figures 1-4 (teaser, setup, environment overviews) are not generated, and it never shows how to run the in-scope demos.
- evidence: The README Figures section lists six scripts. The overview figures' make_figure.py exists only in PRIVATE paper/figs/environments_overview. The demos appear only in the layout line.
- fix: Add 'Figures 1 to 4 are illustrations and are not regenerated.' Add a Demos line: 'python scripts/demos/demo.py --seed 8 --n-objects 4 --fixed-reflectivities' and 'python scripts/demos/play.py'.

### [nit] (tables) notebooks/appendix_tables.ipynb
- issue: 'Each step of IM's increasing editability ... at least six standard deviations' holds only with seed means and the combined SD, and neither the paper nor the steps-md header (line 128) says which unit.
- evidence: T.im_steps: minimum 6.10 combined SDs (16-ray to 8-ray, continuous); 7.39 in larger-SD units. With the 780k main-run values the 16-to-8 step is 0.051, which is 5.39 combined SDs.
- fix: Paper line 467 and the header: 'at least six times the combined seed SD of the two seed means'.

### [nit] (tables) notebooks/appendix_tables.ipynb
- issue: The header promises each table is 'followed by the numbers the text quotes from it', but some quoted derived numbers come only from arithmetic on table cells.
- evidence: 'trails IM ... by 0.17 to 0.26' (IM minus IM-NN; im_vs_nn_gain prints only ratios), 'within 0.03 ... except blink' (trained minus random-init MLP) and 'the largest gain of any Rayworld variant' (smooth 0.347). All are correct, but no cell prints them.
- fix: Add an 'IM - IM-NN' column to tables.im_vs_nn_gain, or soften the header of cell 0.

### [nit] (tables) paper/paper_draft.tex
- issue: 'fit on about 1.2M pairs' (lines 202 and 923) counts the 20% held-out rows.
- evidence: Stored n_train_rows is 943,478 (Othello) and 936,000 (Rayworld); n_test_rows is 235,998 (Othello). About 1.18M is the fit and held-out sets together.
- fix: '...on about 1.2M pairs of latent state and simulator state, 80% of them for fitting'.

### [nit] (tables) pim/figures/tables.py
- issue: Some row labels differ from the paper's (rows are matched by order and variant name).
- evidence: tab:categorical 'Appearance (30)' vs 'Appearance (30 bins)' and 'Appearance, factorized (30)' vs 'Appearance, factorized'. tab:im_vs_nn and tab:fidelity_selected 'standard (128 rays)' vs 'standard'. tab:predictive_skill '128-ray (big discs)' vs '128-ray' and 'Rayworld (8-ray-tokens)' vs 'Rayworld (8-ray tokens)'.
- fix: Pass per-table label maps to row_label, or accept as cosmetic.

### [nit] (tables) notebooks/master_eval.ipynb
- issue: The Summaries cell shows each editor's top arm before the fidelity cutoff, ranked by the union Edit Index, which differs from the tables. The notebook does not say so.
- evidence: print_summaries shows adjacent-flip IM at pt4 with EI(sd) +0.558; Table 2 reports +0.66. Only the docstring in pim/scoring/summary.py explains this; the summaries-md header (line 84) is just '## Summaries'.
- fix: Header: '## Summaries (each editor's top arm before the fidelity cutoff; the tables apply the selection rule)'.

### [nit] (rescore) /home/sevan/research/PIM/generative-models-as-simulators/pim/scoring/summary.py
- issue: In the master_eval summaries (lines 18 and 24), the column headed 'fid' prints fidelity_ratio, not Edit Fidelity (1 - ratio), and the arm shown is the unguarded top arm. The printed numbers do not match the paper's cells.
- evidence: treeA summary: 8-ray cartesian 'IM all 6 1.0 +0.7112 0.265', where the paper has +0.71 / 0.73. treeB: standard cartesian 'PI all 2 100.0 +0.1974 1.715', where the paper has -0.10 / 0.04.
- fix: Change the header '{'fid':>7}' to '{'ratio':>7}', and add to the notebook's Summaries markdown: 'The unguarded top arm and its fidelity ratio; the tables apply the selection rule and report Edit Fidelity = 1 - ratio.'

### [nit] (rescore) /home/sevan/research/PIM/generative-models-as-simulators/pim/scoring/blocks.py
- issue: attach_inverse writes 'nn_r2': [NaN]*9 into categorical inverse_map blocks, which the shipped blocks lack. This adds non-standard NaN tokens to the JSON. It is harmless, because the tables read NaN either way.
- evidence: Fresh 8-ray appearance-fac inverse_map keys: [..., 'nn_r2', ...] = [nan x 9]. The shipped keys have no nn_r2. The fresh 8-ray scores.json has 45 NaN tokens.
- fix: In attach_inverse, write nn_r2 only when stats['nn_r2'] is present and finite (no retrieval bank means no key), or leave as is and accept the schema drift.

### [nit] (pipeline) scripts/figures/editability_by_point.py:92
- issue: The script has no argparse, so --help runs it and writes the figure.
- evidence: The --help run printed '-> outputs/figures/appendix/editability_over_res_point.pdf'.
- fix: Add argparse.ArgumentParser(description=__doc__).parse_args() in main.

### [nit] (pipeline) scripts/make_edit_selection.py:66
- issue: selection.json records the requested pool size, not the number of cases actually scanned.
- evidence: With a 3000-case edits split it records 'pool': 4000, while stats.pool is 3000.
- fix: Use 'pool': int(len(ok)).

### [nit] (pipeline) scripts/othello_flip_rates.py:27
- issue: The script defaults to standard and adjacent-flip, while STAGING ships corpus_stats.json for all four instances. README:197 therefore regenerates only two of the four files.
- evidence: STAGING has runs/_baselines/othello/{standard,adjacent-flip,adjacent-noflip,standard-noflip}/corpus_stats.json.
- fix: Default to sorted(oc.INSTANCES), or note that the paper quotes only two.

### [nit] (pipeline) pim/environments/rayworld/viz.py:167
- issue: The demos warn on every frame.
- evidence: 'UserWarning: Setting the 'color' property will override the edgecolor or facecolor properties.'
- fix: Pass facecolor=/edgecolor= to plt.Circle instead of color=.

### [nit] (static) pim/environments/rayworld/bigcorpus.py:86
- issue: Small consistency items.
- evidence: The RESERVED comment says 'data outside this release'. Lowercase rayworld/othello in comments (metrics/__init__.py:51,61; layout.py:22). 'canonical run' is written into every __seed0/config.json (make_replicate_member.py:49), and probe_refit_variance has --im-points canonical. fit_baseline_probe is an alias of fit_probe_stream (probes/baselines.py:229). Box-drawing banners in 7 files. Unused params in score(model) (rayworld/arms.py:278) and _decodability_values(F) (tables.py:399). scoring/baselines.py:165 writes a label cache into datasets/. Staged standard Othello splits lack flip/placement keys. Empty .scratch/ and .ruff_cache/ in the release root.
- fix: Reword RESERVED to 'seed ranges no split may use'. Capitalize Rayworld/Othello. Say 'main run'. Keep one name for the probe fit. Remove the banners and unused params. Cache the Othello labels under outputs/. Exclude .scratch/ and .ruff_cache/ from any archive.

### [nit] (anon-lexical) /home/sevan/research/PIM/generative-models-as-simulators/pim/environments/rayworld/grid_target.py
- issue: British spellings in prose: a weak geography signal, and the spec asks for American spelling.
- evidence: grid_target.py:1 and :30 'labelled'; :276 and :278 `centres`; scripts/make_othello_edits.py:51 'non-centre tile recoloured' (also in 4 shipped cases_1000.json recipe strings). The identifiers synthesise_cases and CENTRE are public API.
- fix: Change the prose to 'labeled', 'centers' and 'non-center tile recolored', and optionally update the 4 JSON recipe strings. Keep synthesise_cases and CENTRE, or rename them with the old names kept as aliases.

### [nit] (anon-lexical) /home/sevan/research/PIM/generative-models-as-simulators/README.md
- issue: The download placeholders are unresolved, and the publishing channel can add identity metadata.
- evidence: README.md:44 `<ARTIFACTS_URL>` and :57/59/62/63 `<ANON_HF_REPO>`. HF commits carry the uploader's account name, and tar headers store the owner name `sevan`. paper_draft.tex:379 still has the TODO for the anonymous links.
- fix: Upload from an anonymous HF account and fill in both placeholders. If any tarball is made, use `tar --owner=0 --group=0 --numeric-owner --mtime=1980-01-01`.

### [nit] (anon-lexical) /home/sevan/research/PIM/generative-models-as-simulators/pim/
- issue: The package name `pim` is the acronym of the private project (remote fisheye-sonar/PhysicallyImplicitModeling). It links the release to that project only if the private repo becomes public during review.
- evidence: 1198 probe pickles reference the global pim.probes.base.WorldStateProbe, so renaming would break unpickling.
- fix: Keep the name, and keep the private repo private until the review ends.

### [nit] (anon-semantic) gms-release-artifacts/runs/rayworld/standard/variance.json
- issue: An artifact no paper item or table reads is shipped, and it carries a removed dim set. There are other small history and unreleased-work hints.
- evidence: The paper's probe refits cover only standard Othello, adjacent-flip and 8-ray, but this file holds 20 seeds of 'full' with dims 'pos'. resumed records in 8 configs (adjacent-flip__seed1 from_step 390000). bigcorpus.py:86 'seed ranges of data outside this release'; config.py:72 'top-down raster observation; not implemented here'; metrics.jsonl elapsed_s and n_workers 16 hint at hardware.
- fix: Drop variance.json from the bundle or document it. Reword the comments to 'reserved seed ranges' and 'unused fields, kept so stored configs rebuild'. Optionally drop elapsed_s and resumed.

### [nit] (anon-semantic) generative-models-as-simulators/.ruff_cache/ (and runs/, datasets/ symlinks)
- issue: Gitignored local files contain the username in absolute paths and would leak in any non-git packaging.
- evidence: .ruff_cache/0.15.7/* contains '/home/sevan/research/PIM/generative-models-as-simulators/pim'. runs/{othello,rayworld,_baselines} and datasets/{othello,rayworld} are symlinks to /home/sevan/research/PIM/gms-release-artifacts.
- fix: Build any upload (supplementary zip, 4open upload) from a fresh clone or `git archive HEAD`, never by zipping the working tree.

### [nit] (anon-semantic) physically-implicit-modeling/paper/paper_draft.tex (outside the release)
- issue: The submission source and its figure PDFs carry identifying metadata that pdfTeX can copy into the paper PDF.
- evidence: paper_draft.tex:245 comment 'Sevan 2026-09-22' plus private run paths. figs/teaser.pdf has /Creator (Keynote), /Producer (macOS Version 26.6.2 (Build 25G83)) and CreationDate D:20260922201011, which pdfTeX writes into /PTEX.InfoDict. README:44 '<ARTIFACTS_URL>' renders as an invisible HTML tag on GitHub.
- fix: Add \pdfsuppressptexinfo=-1 to the preamble, never upload the commented .tex, and check the final PDF with `strings paper.pdf | grep -i -E 'keynote|macos|sevan|PTEX'`. Replace <ARTIFACTS_URL> before publishing.

### [nit] (completeness) paper/paper_draft.tex:197,1055
- issue: The paper omits two blink constraints: only one disc may be hidden at a time, and a disc cannot start a new blackout on the frame after it reappears.
- evidence: pim/environments/rayworld/blink.py:41-52.
- fix: Add 'one disc at a time' to the blink description.

### [nit] (completeness) paper/paper_draft.tex:1060-1064
- issue: The categorical PI target is not described in the paper.
- evidence: pinv.swap_class_logits swaps the current and target classes' logits at the edited cell.
- fix: Add one sentence to app:implementation.

### [nit] (completeness) RELEASE notebooks/master_eval.ipynb (Summaries); pim/scoring/summary.py
- issue: The summary prints the unguarded 'best' arm and the union Edit Index, which disagree with Table 2.
- evidence: adjacent-flip IM is printed as pt4 +0.558 while Table 2 has +0.66.
- fix: Add to the Summaries header: 'the arm shown is the unguarded top arm; the paper's rule is applied in the table notebooks'.

### [nit] (completeness) paper/paper_draft.tex:202,923
- issue: 'Fit on about 1.2M pairs' counts the held-out rows too.
- evidence: n_rows 1,179,476 in total, of which 80% (about 0.94M) are fit.
- fix: '…about 1.2M pairs, 80% of them for fitting'.

### [nit] (completeness) RELEASE scripts/figures/editability_by_point.py
- issue: There is no argparse, so --help runs the script.
- evidence: Running it with --help produced no usage text; it exited 0 and ran the full script.
- fix: Add argparse.ArgumentParser(description=__doc__).parse_args().

### [nit] (completeness) RELEASE scripts/othello_flip_rates.py:37; README.md:200
- issue: The flip-rate script spells its output path by hand, and README step 5 runs an unreported refit.
- evidence: othello_flip_rates.py builds out = REPO/'runs'/'_baselines'/… instead of calling layout.baselines_dir, against layout's 'no other module spells one'. README step 5 runs probe_refit_variance.py on rayworld/8-ray with its defaults, which includes the unreported frustum 'full' target (10 seeds).
- fix: Use layout.baselines_dir('othello', inst). Pass --targets appearance-fac --seeds 6 for rayworld/8-ray.

### [nit] (reviewer) pim/environments/othello/corpus.py
- issue: LADDER = {"D": 20_000_000} is left over from a size ladder and is used at 7 call sites.
- evidence: corpus.py:70; call sites in bayes.py, scoring/othello.py, train.py, make_othello_corpus.py and qualitative_othello.py.
- fix: Replace it with N_TRAIN = 20_000_000 and use it at the call sites (no numerics change).

### [nit] (reviewer) pim/environments/rayworld/grid_target.py
- issue: British spelling, and a vague appearance-fac gloss.
- evidence: grid_target.py:1 and :30 'labelled'; :4 'appearance-fac: those cells per object'.
- fix: Use 'labeled'. Gloss as 'each disc's appearance bin as a center label and a length label (15 and 5 classes on 8-ray)'.

### [nit] (reviewer) pim/environments/rayworld/bench.py
- issue: Single-value settings are left unexplained.
- evidence: bench.py:229-239 DIM_SETS = {"all": None}; master_eval SETTINGS 'rw_edit_dims': ('all',) and 'rw_target': 'full'.
- fix: Add a SETTINGS comment: 'full: positions and velocities, the 8-dim state; dims "all": every read-out'.

### [nit] (reviewer) pim/environments/othello/reachability.py
- issue: Several module docstrings exceed the spec's 1-4 lines.
- evidence: reachability.py 24 lines (a useful algorithm description, keep), othello/corpus.py 11, othello/__init__.py 10, scoring/baselines.py 9, figure scripts 10-14 (they double as --help).
- fix: Trim corpus.py, othello/__init__.py and scoring/baselines.py to 4 lines, or accept them as is.

### [nit] (reviewer) notebooks/master_eval.ipynb
- issue: 'Layer' is mixed with the paper's 'residual point'.
- evidence: GS_LAYERS, 'gs_layers', 'GS@L0', and 'layer by layer' in grad_steer.py:1.
- fix: Comment GS_LAYERS as '# GS start points (residual points)'.

### [nit] (reviewer) notebooks/paper_tables.ipynb
- issue: Tables render only as PNG (text/plain is '<tab:...: N rows>'), and the notebooks' language_info lacks file_extension, so nbconvert --to script writes .txt.
- evidence: jq of the executed notebook shows text/plain '<tab:decodability: 10 rows>'; nbconvert --to script produced master_eval.txt.
- fix: In the title markdown, mention that T.table_x(F).values / .text give the numbers as DataFrames.

### [nit] (reviewer) README.md
- issue: Small usability gaps: probes/ ships without INDEX.md (file names are hashes), sha256sum is not on macOS, and GPU memory for the figures (about 8 GB) is not stated.
- evidence: ls runs/rayworld/8-ray/probes shows 79 probes_<hash>.pt with no index; figures-recheck measured up to 8.0 GB.
- fix: Regenerate INDEX.md with ProbeCache(dir).write_index() at export, or say how to list the files. Mention 'shasum -a 256 -c' and the memory figure.
