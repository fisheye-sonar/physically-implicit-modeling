# Finding: Predictive Quality / Observation Fidelity

*Affordance 1 — does the model produce high-quality observations?*
Models: GRU r3 `3_dset3_gru_persistentids_inview_400epochs`, refined RSSM `4_dset4_refined_best`
(det 256 + stoch 64, lr 3e-4, free_nats 3, 500 ep, seed 0), dataset `4_fixed_refl_inview`.
Deterministic (prior-mean) eval. Engineering record: `research/scratch/2026-06-29-rssm-refinement.md`.

> **Scope (preliminary, 2026-07-09).** Concerns **these two trained checkpoints** on `dataset 4`. The
> RSSM was tuned in a bounded sweep; the GRU is only lightly tuned, so cross-architecture gaps are
> indicative, not final. Not a general ranking of GRU vs RSSM.

## Current understanding

> **Updated 2026-09-23.** For the paper's ten models the question is settled against the Bayes floor of each
> world (log entry 2026-09-23): every model closes ≥ 99 % of the trivial-to-floor gap, so editability
> differences between worlds are not prediction-quality differences.
>
> **Updated 2026-08-17.** Two capacity/data results now bound this concept. Predictive quality
> **saturates by `H=128`** and `H=512` buys nothing (2026-07-30, replicated across four model
> families 2026-08-13). And the repo's standard dataset is **not a neutral choice**: its
> observation noise acts as a regulariser that adds +0.22 to linear position readability, while
> removing world noise buys no velocity readability at all (2026-07-30) — both the opposite of
> what was pre-registered. Absolute RMSE is not comparable across noise cells; read each against
> its own baselines.

### Previous synthesis (mutable summary)

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

### 2026-09-23 — Every paper model predicts its world to within a few percent of the Bayes floor · `replicated`

**Evidence:** `scripts/bayes_floor.py` → `runs/_baselines/<instance>/bayes_floor.json` (ten instances);
`scripts/score_prediction.py` → the `prediction` block in every scored run (held-out `eval/test.h5` /
`test_10000.npz`, 10k sequences, paired 1000-sequence subset for the floor comparison);
`notebooks/build_appendix_tables_and_figs.ipynb` → Table A1. Queue job `appendix_prediction`
(experiments/paper_ci, lab, 2026-09-23 15:09–15:45, log `logs/paper_ci/appendix_prediction/`); accept list
in `experiments/bayes_floor/QUEUE_HANDOFF.md`, every item met (parity 1.0 on all six discworld instances,
reset share 0.6–0.7 % at 128 rays and < 0.1 % at 5–16 rays, bracket width 9 % / 2–5 %, Othello floors equal
each run's `gates.bayes_ce` to six decimals, no loss below its floor's lower bound, no `—` in the table).
Method and pilot: the other session's `experiments/bayes_floor/` (Othello floors exact — the generator is
uniform over the legal set; discworld floors by a particle sampler over the latent state, bracketed).

| run | trivial | Bayes floor | model | excess | gap closed |
|---|---|---|---|---|---|
| L-oth-20m (nats / move) | 4.094 | 2.0107 | 2.0286 | 0.018 | 0.991 |
| L-oth-adjacent-flip-20m | 4.094 | 2.2955 | 2.3007 | 0.005 | 0.997 |
| L-oth-adjacent-20m | 4.094 | 2.4326 | 2.4349 | 0.002 | 0.999 |
| L-oth-noflip-20m | 4.094 | 1.6788 | 1.6803 | 0.001 | 0.999 |
| L-dw-noiseless-20m (MSE ×10⁻³) | 72.6 | 0.80 [0.77, 0.83] | 1.06 | 0.26 | 0.996 |
| L-dw-blink-20m | 66.0 | 0.98 [0.96, 1.01] | 1.26 | 0.28 | 0.996 |
| L-dw-128ray-20m | 92.4 | 0.75 [0.72, 0.77] | 0.94 | 0.19 | 0.998 |
| L-dw-16ray-20m | 97.2 | 3.77 [3.67, 3.86] | 4.00 | 0.23 | 0.997 |
| L-dw-8ray-20m | 100.7 | 5.62 [5.55, 5.69] | 5.74 | 0.12 | 0.999 |
| L-dw-5ray-20m | 105.5 | 6.86 [6.78, 6.93] | 7.05 | 0.19 | 0.998 |
| L-dw-8ray-tok-20m, tokens (nats / frame) | 5.176 | 0.441 [0.434, 0.448] | 0.464 | 0.023 | 0.995 |
| L-dw-8ray-tok-20m, expected frame (MSE ×10⁻³) | 100.7 | 5.62 | 5.73 | 0.11 | 0.999 |

**Reading.** Every model closes ≥ 99 % of the trivial-to-floor gap. The Othello excesses (0.001–0.018 nats)
are below 1 % of the floor; the Rayworld excesses are 2–5 % of the floor at 5–16 rays and 26–32 % at 128 rays
(the 128-ray floors are small — 0.75–0.98 × 10⁻³ — so a fixed absolute excess of ≈ 0.2 × 10⁻³ is a large
fraction; it is the same absolute excess as at 8 rays). The token model matches the frame model on the same
world under both readings. So the editability differences the paper reports (`ray-ablation.md`,
`inverse-probe.md`) are not differences in how well the models predict: every model sits within a few
percent of what its world allows. Status `replicated`: two environments, ten models, two readings of the
token model, floors by two independent methods (exact / sampled) agreeing on the world both can compute.

**Caveats.** The discworld floor is a sampled bracket, not an exact value (the exact position-0 check is
usable only at low ray counts, where it agrees with the sampler within its SE); the 128-ray relative excess
depends on that bracket's lower edge. One seed per run here — the training-seed spread of the loss is the
`loss_sd` column of the notebook's table (≤ 0.0004 nats on Othello, ≤ 6 × 10⁻⁶ MSE on Rayworld), far below
every excess.


### 2026-07-30 — Observation noise is a regulariser that makes position *linearly* readable · `replicated` ★-candidate

**Evidence:** `scratch/2026-07-30-noise-ablation.md` ·
`notebooks/experiments/editability/controls/noise_ablation.ipynb` (+ `CONTROL_RUNS.md`) · three
new datasets matched to `4_fixed_refl_inview` in every respect except the two noise flags
(`datasets/9_obsnoise0_posnoise0`, `10_obsnoise0_posnoise004`, `11_obsnoise02_posnoise0`;
90k/10k/10k/10k, `ef=20`, seed 0) · four GRUs at `H=256`, identical recipe.

Every dataset in this repo carries **two independent noise sources** and no result had ever
separated them: **observation noise** (`obs_noise_std = 0.2`) corrupts the 1D scan — *sensing*
is imperfect while the world is exact; **position noise** (`position_noise_std = 0.04`) makes
**the world itself** stochastic. Conceptually opposite, and every prior finding was read off the
both-on cell of a 2×2 that had never been filled in.

**§1 Predictive quality**, each cell against its own baselines:

| cell | obs / pos noise | next-step RMSE | copy-previous | own noise floor |
|---|---|---|---|---|
| no noise | 0.0 / 0.00 | 0.0482 | 0.0940 | 0.0000 |
| world noise only | 0.0 / 0.04 | 0.0731 | 0.1028 | 0.0000 |
| sensing noise only | 0.2 / 0.00 | 0.0893 | 0.2109 | 0.1539 |
| both (repo standard) | 0.2 / 0.04 | 0.1041 | 0.2139 | 0.1539 |

**§2 Recoverability — both pre-registered predictions refuted:**

| | no noise | world only | sensing only | both |
|---|---|---|---|---|
| position R² (linear) | 0.596 | 0.693 | **0.819** | **0.828** |
| velocity R² (linear) | 0.451 | 0.451 | 0.469 | 0.471 |

1. Removing world noise does **not** buy velocity readability (Δ = −0.002).
2. **Observation noise makes position linearly readable** (+0.22 linear R²) and makes the state
   markedly more canonical under an MLP fiber map — the opposite of the pre-registered guess.

**Why it matters:** the repo's standard dataset is not a neutral choice — its sensing noise is
actively producing the linear readability that several findings are stated in terms of.

**Caveat that governs any re-reading:** absolute RMSE is **not comparable across cells** (the
noise-free cells have a ≈0 noise floor), so every predictive number is read against that cell's
own baselines. Each model is evaluated on **its own** dataset's edits split, so the cells contain
different scenes and every row needs its own GT column.

---

### 2026-07-30 — Prediction saturates by `H=128`; capacity buys readability, not grabbability · `established`

**Evidence:** `scratch/2026-07-30-hidden-size-sweep.md` ·
`notebooks/experiments/editability/controls/hidden_size_sweep.ipynb` · five GRUs,
`hidden_size ∈ {8,32,128,256,512}`, one variable.

Predictive quality saturates by `H=128`; `H=512` buys nothing over `H=256`. Linear readability
rises **monotonically** with capacity. Replicated across three action-conditioned families
2026-08-13 (next-step RMSE: passive 0.1499 → 0.1040 · exogenous-actions-given 0.1681 → 0.1071 ·
exogenous-observer 0.2016 → 0.1772 · endogenous 0.2080 → 0.1553). **Action knowledge is worth a
large constant** (0.1071 vs 0.1772 at H=256 — teleports are unpredictable without the action)
but does not move *where* the curve flattens.

The editability half of both sweeps is in `editability.md`.

---

### 2026-06-29 — Refined RSSM competitive; generative gap; engineering levers · `established`
Best RSSM `runs/rssm/4_dset4_refined_best` (gitignored; reproducible from config+seed0). Fixed a real
bug: best-checkpoint was selected by total ELBO → froze on an undertrained warm-up epoch under KL
warm-up + free-nats; now selected by `val_recon_loss`, which flipped the RSSM from "looks broken" to
competitive. Dominant knob is lr (3e-4 ≫ 1e-3); free_nats=3 sweet spot; **architecture is not the
lever** (deep enc/dec, stoch64, det384 all plateau at near≈0.0175 once lr is right). Fair eval requires
deterministic prior-**mean** rollouts (added `model.sample` toggle). Did not meet the strict "match GRU"
bar; cleared "within 25%"; beat the prior RSSM. NB: do not compare RSSM val_loss (incl. KL) to GRU's.
