# Finding: Predictive Quality / Observation Fidelity

*Affordance 1 — does the model produce high-quality observations?*
Models: GRU r3 `3_dset3_gru_persistentids_inview_400epochs`, refined RSSM `4_dset4_refined_best`
(det 256 + stoch 64, lr 3e-4, free_nats 3, 500 ep, seed 0), dataset `4_fixed_refl_inview`.
Deterministic (prior-mean) eval. Engineering record: `research/scratch/2026-06-29-rssm-refinement.md`.

> **Scope (preliminary, 2026-07-09).** Concerns **these two trained checkpoints** on `dataset 4`. The
> RSSM was tuned in a bounded sweep; the GRU is only lightly tuned, so cross-architecture gaps are
> indicative, not final. Not a general ranking of GRU vs RSSM.

## Current understanding (mutable summary)

Trained on the observation objective alone (no position supervision), the refined RSSM is a
**competitive next-step predictor and beats the GRU at long horizon**, but a **generative-quality gap**
that MSE hides remains.

- **Near-horizon MSE** (clean obs): RSSM 0.01726 vs GRU 0.01515 (~14% higher); next-step 0.01197 vs
  0.01088. **Long-horizon (last 5): RSSM 0.07128 beats GRU 0.09144** (crossover ~step 12). All models ≫
  persistence.
- **Recoverability falls out of prediction:** the RSSM's probe recoverability *dropped* 0.55→0.32 purely
  as a byproduct of better prediction (no position supervision was used) — the sign we wanted.
- **Generative gap (qualitative):** prior-mean RSSM tracks the bright object ~as well as the GRU but
  **fades the dimmer second object** (mean-hedging on the hard object, not uniform blur); **sampled**
  rollouts **jitter and can fork the track**. Analyze the RSSM in prior-mean mode; treat the
  sampled-generation weakness as itself a finding. (A TV-sharpness metric was inconclusive.)

**Why it matters.** The RSSM is a fair predictive baseline for the editability/canonical-state
comparison (so architecture differences there aren't undertraining artifacts), and the generative gap
(mean-hedging / sampled forking) is a real observation-fidelity phenomenon worth its own thread.

## Log

### 2026-06-29 — Refined RSSM competitive; generative gap; engineering levers · `established`
Best RSSM `runs/rssm/4_dset4_refined_best` (gitignored; reproducible from config+seed0). Fixed a real
bug: best-checkpoint was selected by total ELBO → froze on an undertrained warm-up epoch under KL
warm-up + free-nats; now selected by `val_recon_loss`, which flipped the RSSM from "looks broken" to
competitive. Dominant knob is lr (3e-4 ≫ 1e-3); free_nats=3 sweet spot; **architecture is not the
lever** (deep enc/dec, stoch64, det384 all plateau at near≈0.0175 once lr is right). Fair eval requires
deterministic prior-**mean** rollouts (added `model.sample` toggle). Did not meet the strict "match GRU"
bar; cleared "within 25%"; beat the prior RSSM. NB: do not compare RSSM val_loss (incl. KL) to GRU's.
