# Seed variance — how much do the numbers move across training seeds and probe seeds? (pilot, 2026-09-12)

**Status:** measured 2026-09-12 (run seeds 09:10; probe seeds on the canonical run 12:00; the replicates' probe seeds skipped at Sevan's call). Unit `seed_variance`, `experiments/seed_variance/`. One run family: `L-dw-noiseless-20m`.

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

## Probe seeds: refitting the linear probe (canonical run, 2026-09-12 09:10–12:00)

20 seeds on the frustum regression probe and 6 on `appearance-fac`, each at EVERY residual
point (the seed drives the probe's init and the 80/20 sequence split), with PI — and ND on
the categorical target — swept at the seed's own best point and at the seed-0 best point.
Stopped after the canonical run at Sevan's call (12:00): the two trained replicates' probe
seeds were skipped as uninformative once these numbers were in.

| `L-dw-noiseless-20m` | seeds | best skill mean ± SD | best point | SD across seeds, per point | SD across POINTS, per seed | editors at the best point |
|---|---|---|---|---|---|---|
| frustum (regression) | 20 | **0.9587 ± 0.0006** | 6, every seed | 0.0006–0.0008 at points 1–8 (0.020 at point 0) | 0.364 (0.348–0.373) | PI +0.175 ± 0.001, guard 2.51 ± 0.00 |
| appearance-fac | 6 | **0.4319 ± 0.0007** | 1, every seed | 0.0001–0.0009 | 0.066 (0.0660–0.0666) | ND **+0.505 ± 0.000**, guard 0.78 ± 0.00; PI +0.005 ± 0.015 |

**Reading.**
1. **The probe seed contributes nothing measurable.** Skill SD 0.0006–0.0007 at the best
   point, and 0.0001–0.0009 at every other point but point 0 (the input embedding, where the
   regression probe is at chance and its seed matters: 0.020). The best point is the same
   for every seed on both targets. ND's best Edit Index is identical to three decimals across
   six seeds; PI's varies by 0.001 on the regression target and 0.015 on the categorical one,
   where it is at zero anyway.
2. **The spread ACROSS residual points is a property of the model, not the seed.** Per seed,
   the SD of skill over the nine points is 0.364 ± 0.006 on the regression target (driven by
   point 0's −0.15) and 0.0663 ± 0.0002 on the factorised one — the profile over points is
   reproduced seed for seed to three decimals. So "which point is best" and "how peaked the
   profile is" are stable facts about the run, not artefacts of one fit.
3. **A note on the editor numbers here.** By design they are evaluated at the probe's
   best-DECODABILITY point, so they are not the table's best-over-points arms: regression PI
   reads +0.175 at point 6 (the table's +0.233 is point 1), factorised ND +0.505 at point 1
   (the table's +0.628 is point 2). Both are the same arms every seed — the point is that the
   probe seed does not move them, not what their level is.

Provenance: `runs/noise_ablation/L-dw-noiseless-20m/variance.json`; the seeded probes are
cached in that run's `probes/` (key field `seed`); `logs/seed_variance/e_probe_seeds_L-dw-noiseless-20m.log`
(20 regression seeds 51 min, 6 factorised seeds 2 h — each refit at 9 points on the 200k
recipe). Table 4b.

## What the pilot settles

For this run family, a single seed's number carries roughly **±0.01–0.02 on the Edit Index
from training-seed noise and nothing from probe-seed noise**; the guard near 1.0 is the one
quantity that moves materially between seeds (±0.15). Contrasts of ≥ 0.05 on the index are
real at this scale. What is NOT yet measured: the same spread on dw-8ray (the best-edited
instance, where the contrasts we quote live) and on Othello; and whether GS — an MLP-probe
editor, not refitted here — carries a probe-seed spread of its own.
