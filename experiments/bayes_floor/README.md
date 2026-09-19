# bayes_floor — predictive loss beside the Bayes floor

**Question (Sevan, 2026-09-19).** Are the models we probe and edit near-optimal predictors of their
environments? Each run's held-out loss next to the best loss any predictor could reach from the same
observation history, for the paper's appendix (Table A1).

## Status — canonical code landed 2026-09-19; NOTHING HAS BEEN RUN (the seed queue owns both GPUs)

| piece | where |
|---|---|
| metric arithmetic (loss per sequence, mean frame of a token model, bracket, excess) | `pim/metrics/prediction.py` |
| discworld floor — posterior sampling over the initial state (+ dw-blink's marker process) | `pim/environments/discworld/bayes.py` |
| Othello floor — exact, E[log \|legal\|] | `pim/environments/othello/bayes.py` |
| the TRIVIAL predictor per instance (best history-blind constant, fitted on the probe split; + discworld's repeat-last-frame) — stored in the same floor file under `trivial` | `discworld/bayes.py::trivial_predictors`, `othello/bayes.py::trivial_ce` |
| a run's `prediction` block | `pim/environments/prediction.py::score_run` |
| write the floors → `runs/_baselines/<instance>/bayes_floor.json` | `scripts/bayes_floor.py` |
| fold the block into every scored run's `scores.json` | `scripts/score_prediction.py` |
| Table A1 | `pim.figures.tables.table_prediction` ← `notebooks/build_appendix_tables_and_figs.ipynb` |
| tests (7, CPU, 1 s) | `tests/test_prediction_floor.py` |

Instances: oth-uniform, oth-adjacent-flip, oth-adjacent, oth-noflip (exact); dw-noiseless, dw-blink,
dw-128ray, dw-16ray, dw-8ray, dw-5ray (sampled). Out of scope: dw-smooth, dw-pn04, obs5 (the sampler refuses them).
Verified without running the sampler at scale: the batched renderer reproduces 100% of stored frames on
all six discworld instances (300 sequences each), the stored initial state passes the acceptance rule
everywhere, and the blink marker model is calibrated on the full 10k blink split (expected vs seen markers
within 1.4%; mean −log p 0.3744 vs entropy 0.3726). Pilot (CPU, dw-8ray): `pilot/`,
`research/scratch/2026-09-19-bayes-floor-pilot.md` — floor 0.0055–0.0057, model 0.00577.

## To run (when a GPU is free) — see `QUEUE_HANDOFF.md`

```bash
.pim/bin/python scripts/bayes_floor.py --smoke --device cpu     # 1–2 min, writes bayes_floor.smoke.json only
.pim/bin/python scripts/bayes_floor.py                          # the ten floors (GPU; est. 20–40 min total)
.pim/bin/python scripts/score_prediction.py                     # the prediction blocks (GPU; ~1 min)
.pim/bin/python .pim/bin/jupyter-nbconvert --to notebook --execute --inplace notebooks/build_appendix_tables_and_figs.ipynb
```

## Retired

`scripts/test_loss.py` → `scores/test_loss.json` → `tables.table_bayes` (the old Table 4). Its discworld
"floor" renders the TRUE state one step ahead (0 on every noiseless instance) — a state oracle, not a
Bayes floor: the predictor never sees the state. Kept for the record; Table A1 supersedes it.
