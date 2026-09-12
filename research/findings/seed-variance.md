# Seed variance — how much do the numbers move across training seeds and probe seeds? (pilot, 2026-09-12)

**Status:** run-seed half measured 2026-09-12 09:10; probe-seed half running (unit
`seed_variance`, `experiments/seed_variance/`). One run family: `L-dw-noiseless-20m`.

## Set-up

Sevan's question (2026-09-11): before quoting contrasts of a few hundredths (8-ray vs 5-ray,
joint cell vs factorised), what is the seed-to-seed spread? Two sources, measured separately:

- **Run seeds.** Two re-trainings of the canonical run with `--seed 1` / `--seed 2` at **390k
  steps** (half the budget: the training-curve record shows loss within 5% of final and
  editability flat from 64k on discworld), plus the canonical run's own **step-421,875
  checkpoint** as the seed-0 member of the set at the same budget. n = 3, pooled; SD with n − 1.
  Each replicate is scored canonically (frustum regression) and under `appearance-fac`,
  through the unchanged scorer, with the instance's cached floors. Convention: replicates are
  `runs/<topic>/<parent>__seed<k>/` with a `replicate` block in `config.json`, folded into the
  parent row's ± in the tables, never rows (`harness/WORKFLOW.md`).
- **Probe seeds.** On the canonical run and both trained replicates, the LINEAR probe refitted
  with 20 seeds (frustum regression) and 6 seeds (`appearance-fac`) at every residual point —
  the probe seed drives init and the held-out split — with PI (and ND on the categorical
  target) swept at each seed's own best point and at the seed-0 best point.
  `runs/<run>/variance.json`, Table 4b. *(Pending at the time of writing.)*

## Run seeds: the replicate set at ~390k steps

| `L-dw-noiseless-20m` | val MSE | frustum skill LIN / MLP | frustum PI | frustum GS | fac skill LIN / MLP | fac PI | fac ND | fac GS |
|---|---|---|---|---|---|---|---|---|
| seed 0, 780k (the canonical row) | 0.00106 | 0.959 / 0.996 | +0.233 / 1.95 | −0.099 / 0.99 | 0.433 / 0.786 | +0.014 / 1.95 | +0.628 / 0.78 | +0.352 / 0.68 |
| seed 0, 421,875 (checkpoint) | 0.00110 | 0.957 / 0.996 | +0.217 / 2.23 | −0.104 / 1.00 | 0.425 / 0.775 | +0.007 / 1.81 | +0.614 / 0.79 | +0.330 / 0.85 |
| seed 1, 390k | 0.00110 | 0.956 / 0.995 | +0.227 / 1.80 | −0.109 / 0.99 | 0.424 / 0.782 | +0.016 / 1.96 | +0.644 / 0.72 | +0.306 / 1.09 |
| seed 2, 390k | 0.00110 | 0.954 / 0.996 | +0.230 / 2.01 | −0.121 / 1.00 | 0.428 / 0.785 | +0.007 / 1.77 | +0.616 / 0.80 | +0.331 / 0.83 |
| **pooled n = 3: mean ± SD** | | 0.956 ± 0.001 / 0.996 ± 0.000 | +0.225 ± 0.007, guard 2.01 ± 0.21 | −0.111 ± 0.009, guard 1.00 ± 0.01 | 0.426 ± 0.002 / 0.781 ± 0.005 | +0.010 ± 0.005 | **+0.624 ± 0.017**, guard 0.77 ± 0.04 | **+0.322 ± 0.014**, guard 0.92 ± 0.15 |

**Reading.**
1. **Training seed barely moves anything.** Decodability SDs are ≤ 0.005; the Edit Index SDs
   are 0.005–0.017 across every editor and both targets. The half-budget 780k-vs-390k
   difference (seed 0 at 780k vs its own 422k checkpoint) is of the same size as the seed
   spread: +0.628 vs +0.614 on ND, 0.786 vs 0.775 on MLP skill.
2. **The guard is where the variance lives.** GS's fidelity ratio ranges 0.68–1.09 across the
   set (SD 0.15) while its Edit Index sits at 0.31–0.35; PI's guard on the regression target
   ranges 1.80–2.23. A one-seed row can therefore land on either side of the guard at its best
   arm without the index moving — read guard verdicts near 1.0 as ±0.15.
3. **What this licenses.** Contrasts of ≥ 0.05 on the Edit Index between runs of this family
   are outside the seed spread (≈ 3 SD); the joint-cell vs factorised GS difference on 8-ray
   (+0.61 vs +0.46) and the 5-ray vs 8-ray bump (+0.58 vs +0.46 on GS, +0.49 vs +0.41 on PI)
   are well clear of it, if 8-ray's seed spread resembles noiseless's — that is the next
   instance to replicate. Differences of 0.02–0.03 (noiseless ND +0.63 vs blink +0.53 is
   clear; 8-ray fac ND +0.50 vs joint +0.43 is at the edge) should be quoted with the ±.
4. The regression rows' inertness reproduces exactly: PI at α 175–200 with the guard at 1.8–2.2
   on every seed, GS negative on every seed.

Provenance: `runs/noise_ablation/L-dw-noiseless-20m{,__seed0_s421875,__seed1,__seed2}/scores.json`;
driver `scripts/drivers/seed_variance.sh`, `logs/seed_variance/` (train 4 h 01 + 4 h 01, scoring
3 × ~40 min). Tables: ± cells in Tables 1–2, Table 4a.

## Probe seeds

*Pending — stage E of the chain (20 regression + 6 factorised linear seeds per run, every residual point).*
