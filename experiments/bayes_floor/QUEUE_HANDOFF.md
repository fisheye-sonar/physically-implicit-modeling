# Hand-off to the session operating the paper_ci queue — add ONE job at the end (2026-09-19)

You operate `experiments/paper_ci/` (the seed-replicate queue). Sevan wants one more job appended to it:
the appendix's predictive-loss-vs-Bayes-floor table. The code is written, tested and in the lab working
tree (UNCOMMITTED at the time of writing — see "Files" below). Nothing has been run. Your part is to
install the job so it runs on the LAB GPU after everything else, and to check its output.

## What the job does (lab only, ~1 h budgeted, probably 20–40 min)

1. `scripts/bayes_floor.py` — writes `runs/_baselines/<instance>/bayes_floor.json` for ten instances:
   four Othello (exact, seconds, CPU) and six discworld (dw-noiseless, dw-blink, dw-128ray, dw-16ray,
   dw-8ray, dw-5ray; posterior sampling on the GPU, 1000 sequences × 512 particles × 40 sweeps). Each file
   also carries the instance's TRIVIAL predictor (the best history-blind constant, fitted on the probe
   split — CPU, seconds). Model-free.
   Skips a file already at the current version.
2. `scripts/score_prediction.py` — adds a `prediction` block to every scored run's `scores.json`
   (one forward pass over the 10k held-out sequences per discworld run; Othello runs are read from their
   `gates`). One dated backup per run under `scores_backup/`, atomic replace, skips runs already done,
   honours `PIM_SKIP_TOPICS`.
3. `scripts/index_ceiling.py` on eight paper runs — the Edit Index CEILING (the paper's "+0.91" sanity check) →
   `runs/<topic>/<run>/index_ceiling.json`. Discworld: the overwrite oracle on the 1000-case bench (the old number was
   on the 192-case bench). Othello: a real game with the flipped board, `--reachability` adds IM on legal vs illegal
   targets. `L-oth-20m` is already done on CPU (+0.909, 95% CI 0.854–0.956, 13 cases — reproduces the experiment's
   file); re-running it is harmless.
4. Re-executes `notebooks/build_appendix_tables_and_figs.ipynb` (Table A1).

## Install

The job file is ready: `experiments/bayes_floor/queue_proposal/appendix_prediction.json`. It is
deliberately NOT in `experiments/paper_ci/queue/` — copy it there yourself:

```bash
cp experiments/bayes_floor/queue_proposal/appendix_prediction.json experiments/paper_ci/queue/
.pim/bin/python experiments/paper_ci/scripts/dispatch.py --dry      # confirm it is parsed and waiting on final_tables
```

It depends on `final_tables` (priority 95, lab, gpu lane), so it starts only when the queue has drained.
If you would rather register it through `plan.py`, mirror those fields; do not let `plan.py` rewrite
started jobs.

## Constraints — please keep these

- ⛔ **Never concurrently with `master_eval`** on the lab: both rewrite `scores.json` files. Depending on
  `final_tables` in the lab's gpu lane guarantees this. Do not move it to the cpu lane or to the remote
  (the parents, the 16-ray eval split and every replicate's pulled scores are on the lab; the tables read
  the lab tree only).
- Do not edit `scripts/drivers/score_pending.sh` or `replicate.sh` to call these scripts while the queue
  is live (bash reads a running script incrementally — GOTCHAS 2026-09-08).
- `master_eval.ipynb` is NOT changed by this work. New runs get their `prediction` block from
  `scripts/score_prediction.py` (it only adds what is missing, so re-running it is free). If Sevan later
  wants the scorer itself to write the block, the hook is one cell calling
  `pim.environments.prediction.score_run(run_dir)` and storing the result under `scores["prediction"]` —
  add it only when no `--inplace` execution of the notebook is in flight on either host, and sync the
  notebook to the remote the same way you synced the 16-ray `dw_extra_targets` fix.
- Smoke first (harness/OVERNIGHT.md). CPU-only, 1–2 minutes, writes `bayes_floor.smoke.json` files only:
  `.pim/bin/python scripts/bayes_floor.py --smoke --device cpu` — it must finish with a line per instance
  and no "parity FAILED". You can run this now; it does not touch a GPU or any scores.

## Accept / reject the result

- Each discworld `bayes_floor.json`: `diagnostics.parity == 1.0`; `diagnostics.reset_share` below ~1%;
  `mse.lo <= mse.hi` with the two within ~10% of each other — the table shows their midpoint ± (half their
  distance + one SE), so a wide bracket becomes a wide ± (if the bracket is wider, re-run that
  instance with `--particles 1024 --sweeps 80 --force` — it is cheap); on the coarse-ray instances
  `check_position_0.usable` is true and `exact` ≈ `sampler_lo` within 2–3 `sampler_lo_se`.
- Reference from the CPU pilot: dw-8ray floor 0.0055–0.0057, `L-dw-8ray-20m` loss 0.00577. No run's loss
  should sit BELOW its floor's `lo` by more than a couple of standard errors — if one does, the sampler is
  wrong for that instance, not the model good; flag it rather than quoting it.
- Trivial predictors (sanity): Othello ≈ 4.09 nats (log 60 = 4.094); discworld constant-frame MSE ≈ 0.07–0.11
  (dw-8ray 0.1007, dw-noiseless 0.0726), `persistence_mse` ≈ 0.008; dw-8ray token CE ≈ 5.18. Every model's
  loss must sit between its trivial predictor and its floor.
- Othello floors must equal each run's `gates["bayes_ce"]` (2.01069 / 2.29546 / 2.43264 / 1.67879 for
  oth-uniform / adjacent-flip / adjacent / noflip).
- Table A1 renders eleven runs (twelve rows: `L-dw-8ray-tok-20m` has two readings) with no `—` (a `—` under `trivial predictor` for a 128-ray token reading cannot occur — only dw-8ray is tokenised); floors and
  excesses read `value ± uncertainty` on discworld and a bare value on Othello (exact).
- Then: record the numbers in `research/findings/predictive-quality.md` (a dated entry, `observed`),
  update `research/PROGRESS.md`, and tell Sevan the table is ready. The paper's red placeholders in
  §Results ("within 0.02 nats", "below 0.006") and appendix `app:predict_skill` are filled from this table.

## After the queue has drained — one line in a driver

`scripts/drivers/replicate.sh` stage B still calls `experiments/seed_variance/scripts/layout_checkpoint_replicate.py`,
which is now a FORWARDER to the canonical `scripts/layout_checkpoint_replicate.py` (identical code; both paths tested on
an existing member). It was left alone because a live job is executing that driver. When nothing is running: point
stage B at `scripts/layout_checkpoint_replicate.py`, delete the forwarder, and make sure the remote has both changes
before its next replicate job (a git pull brings the new script and the forwarder together, so either order is safe).

## Files (all new unless marked)

`pim/metrics/prediction.py` · `pim/environments/prediction.py` · `pim/environments/discworld/bayes.py` ·
`pim/environments/othello/bayes.py` · `scripts/bayes_floor.py` · `scripts/score_prediction.py` ·
`tests/test_prediction_floor.py` · `notebooks/build_appendix_tables_and_figs.ipynb` (was empty) ·
`pim/figures/tables.py` (edited: `prediction_rows`, `table_prediction`) · `pim/metrics/__init__.py`
(docstring) · `research/REGISTRY.md` (two rows) · `experiments/bayes_floor/` (README, pilot, this file).
The job runs from the lab working tree, so it works uncommitted; commit them with your next commit if
Sevan has not already.
