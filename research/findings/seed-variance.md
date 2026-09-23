# Seed variance — how much do the numbers move across training seeds and probe seeds? (pilot, 2026-09-12)

> **Rescored 2026-09-12** under the PRE-dynamics write target (GOTCHAS 2026-09-12; `scratch/2026-09-12-alignment-rescore.md`): every discworld editor number quoted below moved by ≤ 0.06 (median 0.005), floors unchanged, no conclusion changed. `scores.json` holds the current values.


**Status:** `replicated` for the paper's ten families at n = 3 × 512k (2026-09-23, last section — THE record for Table 5); the 2026-09-12 pilot below stands as the first measurement. Pilot: measured 2026-09-12 (run seeds 09:10; probe seeds on the canonical run 12:00; the replicates' probe seeds skipped at Sevan's call). Unit `seed_variance`, `experiments/seed_variance/`. One run family: `L-dw-noiseless-20m`.

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

*Numbers below are at eval version 2026-09-12.2 (every run was rescored 2026-09-12/13 after the first write-up; the spread is unchanged in size — the first version's SDs were 0.005–0.017 too — and Table 5 in the table notebooks reads the same files).*

| `L-dw-noiseless-20m` | val MSE | frustum skill LIN / MLP | frustum PI | frustum GS | fac skill LIN / MLP | fac PI | fac ND | fac GS |
|---|---|---|---|---|---|---|---|---|
| seed 0, 780k (the canonical row) | 0.00106 | 0.959 / 0.996 | +0.231 / 1.54 | -0.082 / 1.07 | 0.433 / 0.786 | +0.007 / 1.93 | +0.612 / 0.80 | +0.326 / 0.71 |
| seed 0, 421,875 (checkpoint) | 0.00110 | 0.957 / 0.996 | +0.211 / 1.77 | -0.086 / 1.04 | 0.425 / 0.775 | -0.001 / 1.78 | +0.597 / 0.82 | +0.298 / 0.74 |
| seed 1, 390k | 0.00110 | 0.956 / 0.995 | +0.215 / 1.96 | -0.090 / 1.05 | 0.424 / 0.782 | +0.003 / 1.74 | +0.626 / 0.83 | +0.291 / 0.80 |
| seed 2, 390k | 0.00110 | 0.954 / 0.996 | +0.223 / 2.13 | -0.092 / 1.03 | 0.428 / 0.785 | -0.003 / 1.78 | +0.600 / 0.82 | +0.305 / 0.75 |
| **pooled n = 3: mean ± SD** | | 0.956 ± 0.001 / 0.996 ± 0.000 | +0.217 ± 0.006, guard 1.95 ± 0.177 | -0.089 ± 0.003, guard 1.04 ± 0.006 | 0.426 ± 0.002 / 0.781 ± 0.005 | -0.000 ± 0.003 | **+0.608 ± 0.016**, guard 0.82 ± 0.009 | **+0.298 ± 0.007**, guard 0.76 ± 0.032 |

**Reading.**
1. **Training seed barely moves anything.** Decodability SDs are ≤ 0.005; the Edit Index SDs
   are 0.005–0.017 across every editor and both targets. The half-budget 780k-vs-390k
   difference (seed 0 at 780k vs its own 422k checkpoint) is of the same size as the seed
   spread: +0.628 vs +0.614 on ND, 0.786 vs 0.775 on MLP skill.
2. **The guard moves more than the index, on the arms that fail it.** The regression PI
   guard ranges 1.54–2.13 across the set (SD 0.18) while its index sits at 0.21–0.23; the
   editors that pass the guard hold it tightly (factorised ND 0.80–0.83, GS 0.71–0.80). So a
   one-seed guard verdict on a FAILING arm is soft by ±0.2, a passing one is firm to ±0.03.
   (Before the 2026-09-12 rescoring, GS's guard on this target had spanned 0.68–1.09 across
   the same seeds — a reminder that a guard near 1.0 is the least stable number in a row.)
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

## 2026-09-23 — The paper's replicate set: every shortlist family at n = 3 training seeds, 512k steps · `replicated`

**Evidence.** `experiments/paper_ci/` (the multi-day queue, `harness/MULTIDAY.md`; launched 2026-09-18 20:48, drained
2026-09-23 15:45, 40 jobs on the lab 5090 and the WSL 4090): for each of the ten shortlist runs two re-trainings
(`--seed 1`, `--seed 2`) to 512,000 steps plus the parent's own step-512,000 checkpoint laid out as the seed-0 member
(`scripts/layout_checkpoint_replicate.py`), every member scored by the unchanged scorer (`pim/scoring`, frustum +
cartesian + `appearance-fac` where the parent has it), pooled by `pim.metrics.replicates.pool_replicates` at the
matched budget (±10 %; the dw-noiseless / oth-adjacent-flip 390k members were EXTENDED to 512k and their 390k scores
parked as `scores.s390000.json`, `runs/MOVES.md`). Ledger: `experiments/paper_ci/dashboard/ledger.{md,json,csv}`;
Table 5 (SD and CI panels) from `build_full_tables.ipynb` (the paper's appendix table `tab:seed_spread` is the SD panel); probe-seed refits in each run's `variance.json`.
Cells are the tables' REPORTED arm: the best Edit Index inside the fidelity guard, else — under the fallback in force
when Table 5 was rendered (2026-09-23 15:07, `pim.metrics.selection.best_arm`) — the lowest-fidelity-ratio arm.
Fidelity = 1 − the RMSE ratio (2026-09-22). n = 3 throughout; ± = SD with n − 1.

| family | n | LIN | MLP | PI index | GS index | IM index | PI fid | GS fid | IM fid |
|---|---|---|---|---|---|---|---|---|---|
| oth-standard | 3 | +0.969 ± 0.003 | +0.971 ± 0.002 | +0.808 ± 0.015 | +0.808 ± 0.012 | +0.778 ± 0.034 | +0.478 ± 0.054 | +0.665 ± 0.007 | +0.603 ± 0.038 |
| oth-adjflip | 3 | +0.948 ± 0.000 | +0.970 ± 0.001 | +0.165 ± 0.245 | −0.022 ± 0.073 | +0.623 ± 0.028 | +0.140 ± 0.031 | +0.034 ± 0.038 | +0.521 ± 0.108 |
| oth-adjacent | 3 | +0.987 ± 0.001 | +0.989 ± 0.001 | −0.121 ± 0.196 | −0.944 ± 0.005 | −0.498 ± 0.422 | +0.154 ± 0.085 | −0.020 ± 0.010 | +0.002 ± 0.077 |
| oth-noflip | 3 | +1.000 ± 0.000 | +1.000 ± 0.000 | −0.966 ± 0.001 | −0.959 ± 0.002 | −0.883 ± 0.026 | +0.001 ± 0.001 | −0.008 ± 0.002 | +0.003 ± 0.002 |
| dw-noiseless (cartesian) | 3 | +0.867 ± 0.002 | +0.971 ± 0.000 | −0.080 ± 0.042 | −0.186 ± 0.027 | +0.592 ± 0.005 | +0.016 ± 0.010 | +0.022 ± 0.026 | +0.666 ± 0.020 |
| dw-blink | 3 | +0.894 ± 0.002 | +0.975 ± 0.001 | −0.024 ± 0.060 | −0.097 ± 0.018 | +0.515 ± 0.012 | +0.019 ± 0.016 | +0.020 ± 0.020 | +0.557 ± 0.007 |
| dw-128ray | 3 | +0.904 ± 0.004 | +0.981 ± 0.000 | −0.003 ± 0.009 | −0.097 ± 0.027 | +0.574 ± 0.006 | +0.015 ± 0.010 | +0.029 ± 0.023 | +0.683 ± 0.008 |
| dw-16ray | 3 | +0.891 ± 0.002 | +0.964 ± 0.000 | +0.153 ± 0.021 | −0.121 ± 0.010 | +0.663 ± 0.007 | +0.075 ± 0.020 | +0.102 ± 0.009 | +0.712 ± 0.004 |
| dw-8ray | 3 | +0.888 ± 0.002 | +0.932 ± 0.000 | +0.207 ± 0.022 | −0.074 ± 0.030 | +0.720 ± 0.006 | +0.079 ± 0.041 | +0.114 ± 0.017 | +0.735 ± 0.008 |
| dw-5ray | 3 | +0.852 ± 0.002 | +0.883 ± 0.000 | +0.132 ± 0.036 | −0.093 ± 0.015 | +0.808 ± 0.009 | +0.102 ± 0.051 | +0.151 ± 0.016 | +0.770 ± 0.007 |
| dw-128ray (appearance-fac) | 3 | +0.253 ± 0.001 | +0.700 ± 0.001 | −0.331 ± 0.024 | +0.286 ± 0.014 | +0.637 ± 0.003 | +0.023 ± 0.020 | +0.196 ± 0.019 | +0.719 ± 0.005 |
| dw-16ray (appearance-fac) | 3 | +0.877 ± 0.002 | +0.917 ± 0.000 | −0.050 ± 0.014 | +0.255 ± 0.006 | +0.821 ± 0.002 | +0.059 ± 0.019 | +0.179 ± 0.065 | +0.716 ± 0.004 |
| dw-8ray (appearance-fac) | 3 | +0.935 ± 0.000 | +0.943 ± 0.000 | +0.392 ± 0.006 | +0.445 ± 0.014 | +0.874 ± 0.005 | +0.079 ± 0.027 | +0.451 ± 0.009 | +0.734 ± 0.001 |
| dw-5ray (appearance-fac) | 3 | +0.927 ± 0.000 | +0.932 ± 0.000 | +0.538 ± 0.017 | +0.559 ± 0.012 | +0.912 ± 0.003 | +0.182 ± 0.029 | +0.527 ± 0.016 | +0.750 ± 0.000 |

**What it settles.**
- **Decodability is seed-stable to the third decimal everywhere**: LIN SD ≤ 0.004, MLP SD ≤ 0.002 across all ten
  families and both targets.
- **Every editability cell the paper's claims rest on has SD ≤ 0.04**: discworld IM 0.003–0.012 on both targets;
  the categorical PI / GS on the ray family 0.006–0.024; Othello standard PI / GS 0.012–0.015 and IM 0.034; adjflip
  IM 0.028. The ray-count ordering of IM (5 > 8 > 16 > 128) is separated by 5–15 SDs at every step.
- **The wide cells are editors that fail on that world**, where the guarded arm lands on a different alpha per seed
  and the index straddles zero: oth-adjflip PI (± 0.245), oth-adjacent PI (± 0.196) and IM (± 0.422), plus their
  fidelities. Mark them † in the main table; nothing the text relies on is among them.
- **The instrument is not the spread.** Refitting the linear probe ten times moves the index by ≤ 0.016 and skill
  by ≤ 0.0007 on every run tested (oth-standard, dw-8ray, the three adjflip members); refitting the inverse map
  moves IM by 0.004 (oth-standard) to 0.036 (adjflip) — only on adjflip is the instrument's spread comparable to
  the model's. Corpus size is not the spread either (`probe-capacity.md`, 2026-09-23).
- **Othello's seed spread is 3–7× discworld's on the same editors** (standard: PI ± 0.015 vs 5-ray PI ± 0.036 is
  the exception; IM ± 0.034 vs ≤ 0.012), and the canonical parent sits at the top of every oth-standard member
  range — one draw, not a biased one, but worth remembering beside the ±.

**Caveats.** Discworld extension members (dw-noiseless seeds 1–2, resumed 390k → 512k) are not bit-identical to
uninterrupted runs (`batch_order_exact: False` on discworld); their spread is indistinguishable from the other
families'. The 5-ray canonical factorised GS (+0.598 at 780k) sits ~3 SD above its 512k members — a budget
mismatch the canonical row carries, not a seed effect. oth-noflip's canonical IM (−0.93) vs members (−0.85 … −0.90)
is the guard's fallback branching on an all-fail row, not variance.
