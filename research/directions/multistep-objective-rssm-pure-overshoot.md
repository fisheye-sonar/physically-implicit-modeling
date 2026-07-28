# Direction: RSSM multistep — PURE latent overshooting (validity re-run of the negative)

**Tag:** `[in-frame]` · **Sub-question:** 1/2/3 · **Status:** proposed — **HOLD (Sevan: do NOT run yet; ping him
to schedule it later).** Not expected to change the verdict, but needed to make the RSSM-multistep negative airtight.

## Why this exists (the concern)
Our RSSM multistep result (`scratch/2026-07-16-multistep-objective-rssm.md`) trained an objective that is **not the
canonical PlaNet/Dreamer setup**. We verified (deep code analysis, 2026-07-16) that `scripts/train_rssm_multistep.py`
implements the standard ELBO + the correct multi-distance **latent-overshooting KL** — BUT it *also* adds an
**observation-overshoot reconstruction** term (`MSE(decode(imagined-prior state), future obs)`,
`train_rssm_multistep.py:118`). **Pure PlaNet "latent overshooting" does NOT reconstruct from the imagined prior** —
it reconstructs only from the posterior and lets the multi-distance KL do the predictive training. Reconstructing
from a *stochastic* multi-step prior sample trains the decoder to output the mean over the prior's uncertainty →
**blur**. So our headline RSSM result — "the multi-step objective *harms* the model via blur / worse prediction" —
is **entangled with our added obs-overshoot term**, not necessarily intrinsic to latent overshooting. (The §4
editability null is robust to this — it's structural — but the "harms the model" claim needs this control.)

## The re-run
Re-train the RSSM `W∈{1,2,5}` sweep with **pure latent overshooting**: drop the observation-reconstruction-from-prior
term entirely; keep standard ELBO (recon from posterior + 1-step KL) + the multi-distance overshoot **KL** only
(`KL[ sg(posterior_{t+d}) ‖ d-step imagined prior ]`, free-nats clamped). Everything else identical (det 256 / stoch
64, dataset 4 noisy, matched epoch budget, `sample=False` eval, subsampled starts). Then re-run the SAME analysis
notebook (`multistep_objective_rssm.ipynb` — parameterize the checkpoints) and compare:
- **Blur / §0 sharpness:** does the rollout-TV collapse (1.23→0.43) and the open-loop-RMSE degradation persist, or
  were they our obs-overshoot term? (Hypothesis: pure overshoot blurs *less*.)
- **§4 editability null:** expected to replicate (structural). If it doesn't, that's surprising and important.
- **§1–§3 structure + det/stoch:** does the canonicality/readability degradation persist?

## Deliverable & cost
New checkpoints `runs/rssm_multistep_pure/w{1,2,5}`, re-executed notebook (or a lightweight variant), dated scratch
note comparing pure-overshoot vs the hybrid. ~1h GRU-discipline (RSSM ~11s/ep; reuse the matched 150-epoch budget).
Follow WORKER.md decoupled execution (train via foreground script calls).

## Status note
Sevan's expectation: probably won't change the editability verdict, but the "objective harms the RSSM" sub-claim
should not be published until this control is run. **HELD** until Sevan schedules it — the orchestrator should
surface this reminder in a future catch-up (it is also pinned at the top of `PROGRESS.md`).
