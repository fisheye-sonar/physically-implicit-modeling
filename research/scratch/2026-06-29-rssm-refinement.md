# RSSM refinement — engineering sweep (2026-06-25 → 06-29)

Durable record of the RSSM good-faith engineering effort (branch `rssm_refinement`
off main). This note is git-tracked; the run artifacts under `runs/` are gitignored
(local-disk only), so the numbers + how-to-reproduce live here.

## Objective
Make the RSSM a strong *predictor* of observations under the project premise:
trained ONLY on the observation ELBO (no GT-position supervision). Optimize
near/long-horizon observation accuracy; treat recoverability + coherence as
diagnostics that should *fall out* of better prediction, not as objectives.
Reference = GRU r3 (already good on this task). Dataset = `datasets/4_fixed_refl_inview`.

## Headline result
Best RSSM = `runs/rssm_sweep2/FINAL/latest.pt` (recipe `r2_stoch64` @500ep:
det_size=256, stoch_size=64, hidden_dim=256, embed=128, enc/dec=1, lr=3e-4,
free_nats=3.0, kl_warmup=10, seed=0). Deterministic (prior-mean) eval, dset4 test:

| metric (clean-obs) | GRU r3 | best RSSM | old RSSM r3 |
|---|---|---|---|
| near-horizon MSE (H1–5) | 0.01515 | 0.01726 | 0.01915 |
| next-step MSE | 0.01088 | 0.01197 | 0.01405 |
| long-horizon MSE (last 5) | 0.09144 | **0.07128** | 0.08025 |
| recoverability (MLP probe) | 0.207 | 0.318 | 0.548 |

- RSSM ends ~14% above GRU near-horizon, ~10% next-step, but **beats GRU at long
  horizon** (crossover ~step 12; `runs/rssm_sweep2/figs/horizon_curve.png`). All
  models >> persistence.
- Did NOT meet the strict "match GRU" bar; cleared "within 25%"; beat prior RSSM.
- Recoverability fell 0.55→0.32 purely as a byproduct of better prediction — the
  "falls out of prediction" sign we wanted (no position supervision was used).

## What moved the needle (16 trials, 2 rounds)
1. **Checkpoint-selection bug (the big one):** best-by-total-ELBO froze on an
   undertrained epoch-~1 checkpoint under KL warm-up + free-nats (warm-up makes
   total loss artificially low; free-nats floor raises it afterward). Fixed to
   select by `val_recon_loss` in `scripts/train_rssm.py`. This alone flipped the
   RSSM from "looks broken" (probe MSE at the mean-predictor floor) to competitive.
2. **Fair eval:** score the deterministic prior-MEAN rollout, not sampled — added
   `model.sample` toggle in `pim/world_models/rssm/model.py`. Sampling noise had
   inflated horizon MSE and rollout jitter.
3. **lr is the dominant knob:** 3e-4 ≫ 1e-3 at det256; lr 1e-3 made bigger models
   worse. free_nats=3 is the sweet spot (1 and 6 worse). KL-balancing (0.8) hurt.
4. **Architecture is NOT the lever:** deep enc/dec, stoch64, det384 all cluster at
   near≈0.0175 once lr is right → honest plateau (the "wall" stop condition).
   Levers were lr + free_nats + epochs.

## Qualitative finding (the generative gap Sevan eyeballed)
`runs/rssm_sweep2/figs/waterfalls.png`: MSE-competitiveness partly hides a real
generative-quality gap. RSSM-mean tracks the bright object ~as well as GRU but
**fades the dimmer 2nd object** (mean-hedging on the hard object, not uniform blur).
RSSM-**sampled** rollout is **jittery and can fork the track** — visibly worse, and
the weakest model curve in Fig 1. Analyze the RSSM in prior-mean mode; treat the
sampled-generation weakness as itself a finding. (TV-sharpness metric was
inconclusive — don't rely on it.)

## How to reproduce / use
- Eval best ckpt: `python scripts/run_eval.py --checkpoint runs/rssm_sweep2/FINAL/latest.pt --data-dir datasets/4_fixed_refl_inview`
- Regenerate GRU-vs-RSSM figures: `python scripts/compare_rollouts.py --gru runs/gru/3_dset3_gru_persistentids_inview_400epochs/best_model.pt --rssm runs/rssm_sweep2/FINAL/latest.pt --data-dir datasets/4_fixed_refl_inview --out runs/rssm_sweep2/figs`
- Retrain best from scratch (reproducible, seed 0, ~90 min): `python scripts/train_rssm.py --dataset-dir datasets/4_fixed_refl_inview --run-dir runs/rssm/refined_best --run-name refined_best --det-size 256 --stoch-size 64 --hidden-dim 256 --lr 3e-4 --free-nats 3.0 --n-epochs 500`
- Re-run/extend the sweep (resumable, JSON ladder): `scripts/sweep_rssm.py` (see `--ladder-json`).

## Open / next
- NEXT SESSION: parallel GRU tuning pass (Sevan flagged) — tells us if the ~14%
  near-horizon gap survives a tuned GRU (current GRU is only lightly tuned).
- CHECKPOINT DURABILITY: `runs/` is gitignored → best ckpt won't travel via merge.
  It's reproducible from config+seed; if cross-machine access is needed without
  retraining, copy the 7MB `.pt` out-of-band or store it deliberately.
