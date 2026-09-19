# 2026-09-19 — discworld Bayes-floor sampler, CPU pilot on dw-8ray (PILOT, not canonical)

Script + raw output: `experiments/bayes_floor/pilot/` (`floor_pilot.py <inst> S P K INIT`, CPU only).
Method: posterior over the initial state x0 (p0, v of both discs) given frames 0..t =
generator prior × 1[generator accepts the 40-frame trajectory] × 1[x0 renders frames 0..t exactly];
SMC over t with MH rejuvenation (mixture of scales; velocity moves pivot about a random observed frame).
Floor_t = E[Var(frame t+1 | frames 0..t)], mean over rays; overall = mean over the 39 positions
(the training loss's averaging).

Checks: batched renderer reproduces 100% of stored test frames; stored (p0, v0) reproduce all 40 frames.
Exact population floor from 345k accepted prior draws grouped by observed prefix: t=0 0.00714,
t=1 0.00711; sampler 0.00699 ± 0.00023, 0.00676 ± 0.00024 (t=1 still ~1.5 SE low).

| S | P | K | CPU time | resets | posterior variance (biased LOW if under-mixed) | posterior-mean MSE vs truth (upper bound) | model L-dw-8ray-20m, same seqs |
|---|---|---|---|---|---|---|---|
| 100 | 64 | 4 | 3 s | 94/3900 | 0.00370 | 0.00490 (noisy) | 0.00585 |
| 300 | 256 | 16 | 4 min | 87/11700 | 0.00505 ± 0.00010 | 0.00582 ± 0.00015 | 0.00577 ± 0.00012 |
| 300 | 512 | 40 | 20 min | 28/11700 | 0.00547 ± 0.00010 | 0.00566 ± 0.00013 | 0.00577 ± 0.00012 |

Reading: the two estimators bracket the floor and close as mixing improves: dw-8ray floor ≈ 0.0055–0.0057,
model 0.00577 → excess ≈ +0.0001–0.0003 (2–5%). The old "floor = 0" (state oracle) was not a Bayes floor.
No position where the model beats the floor by > 2 SE at the largest setting. Unconditional frame variance 0.108.
Open: resets (all particles die on a < 1/P event) → more particles or a better proposal; 128-ray cost
(render ×13, narrower posterior); blink needs the schedule process; token CE floor from the same particles.
