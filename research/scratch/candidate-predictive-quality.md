# CANDIDATE FINDING — Predictive quality: refined RSSM is competitive but has a generative gap

**✅ PROMOTED → `findings/predictive-quality.md` (2026-07-09)**, with preliminary/scoped hedging. Kept as backing detail.

**Consolidates:** `archive/2026-06-29-rssm-refinement.md` (engineering + qualitative).
**Model/data:** best RSSM `runs/rssm/4_dset4_refined_best/` (det256+stoch64, lr 3e-4, free_nats 3,
500 ep, seed 0), vs GRU r3, on `4_fixed_refl_inview`. Deterministic (prior-mean) eval.
**Sub-question:** Predictive Quality / Observation Fidelity (affordance 1). **Status:** candidate —
partly an engineering record, partly a real generative-quality finding.

## The claim (one line)
Trained on the observation ELBO alone (no position supervision), the refined RSSM is a **competitive
next-step predictor and beats the GRU at long horizon**, but prior-mean rollouts **fade the dimmer
second object** and sampled rollouts **jitter/fork** — a generative-quality gap MSE hides.

## Key numbers (clean-obs MSE, deterministic eval)
| metric | GRU r3 | best RSSM | old RSSM |
|---|---|---|---|
| near-horizon (H1–5) | 0.01515 | 0.01726 (~14% above) | 0.01915 |
| next-step | 0.01088 | 0.01197 | 0.01405 |
| long-horizon (last 5) | 0.09144 | **0.07128 (beats GRU)** | 0.08025 |
| recoverability (MLP probe) | 0.207 | 0.318 | 0.548 |

- Crossover ~step 12 (RSSM wins long-horizon). All models ≫ persistence baseline.
- Recoverability fell 0.55→0.32 **as a byproduct of better prediction** (no position supervision) —
  the "recoverability falls out of prediction" sign we wanted.

## What moved the needle (engineering, durable)
1. **Checkpoint-selection bug (the big one):** best-by-total-ELBO froze on an undertrained epoch-~1
   checkpoint under KL warm-up + free-nats. Fixed to select by `val_recon_loss` — flipped the RSSM
   from "looks broken" to competitive.
2. Fair eval = deterministic **prior-mean** rollout (added `model.sample` toggle), not sampled.
3. **lr is the dominant knob** (3e-4 ≫ 1e-3); free_nats=3 sweet spot; KL-balancing hurt.
4. **Architecture is NOT the lever** — deep enc/dec, stoch64, det384 all plateau at near≈0.0175 once
   lr is right (honest wall).

## Qualitative finding (the generative gap)
Prior-mean RSSM tracks the bright object ~as well as the GRU but **fades the dimmer 2nd object**
(mean-hedging on the hard object, not uniform blur). **Sampled** rollout is jittery and can **fork the
track** — visibly worse. Analyze the RSSM in prior-mean mode; treat the sampled-generation weakness as
itself a finding. (A TV-sharpness metric was inconclusive — don't rely on it.)

## Caveats
- Did NOT meet the strict "match GRU" bar; cleared "within 25%"; beat prior RSSM.
- GRU is only lightly tuned — a parallel GRU tuning pass would tell us if the ~14% near-horizon gap
  survives a tuned GRU (flagged next).
- Best ckpt is gitignored (`runs/`) — reproducible from config+seed0; copy the 7MB `.pt` out-of-band
  for cross-machine use.

## Promotion recommendation
**PROMOTE the qualitative generative-gap** (mean-hedging on the dim object; sampled forking) as a
Predictive-Quality/Observation-Fidelity finding. Keep the engineering details (bug fix, knobs) as the
durable record — they are how-to-reproduce, not a scientific claim. Lower promotion priority than the
editability/geometry/RSSM-replication candidates.
