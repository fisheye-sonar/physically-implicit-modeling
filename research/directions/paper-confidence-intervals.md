# Seed spread on the paper's main numbers — `[in-frame]` — sub-questions 2, 3

**Status:** `active` (Sevan, 2026-09-18) — queue built and smoked 2026-09-18 evening; launch pending
Sevan's go. Reference implementation `experiments/paper_ci/` (protocol `harness/MULTIDAY.md`).

## The question

Every decodability, Edit Index and fidelity number in the paper's main tables (ten runs: four
Othello, six discworld) is a single training seed. What is the spread under re-training, so the
environment contrasts the paper leans on are read against it?

## Decisions (Sevan, 2026-09-18)

- **Readout: mean ± SD over training seeds, n stated** (Table 5 panel (a); the ± cells of Tables
  1–2). **Secondary: the t-based 95% CI of the mean** (Table 5 panel (b)). Both come from the same
  replicate sets (`pim.figures.tables.pool_replicates`; REGISTRY "replicate spread").
- **n = 3**: the parent's own checkpoint at the matched budget (seed 0) + two re-trainings
  (`scripts/train.py --replicate-of`, seeds 1 and 2). No third GPU is assumed.
- **One budget everywhere: 512k steps** (Sevan, 2026-09-18 evening; was Othello 512k / discworld
  390k). 780k everywhere would be ~250 GPU-hours at 5090 rates — six days on the two GPUs — and
  the window is 4–5 days. `L-oth-20m` has checkpoints only at 256k / 512k / 780k; every other run
  has a 512k or 492,188 checkpoint for its seed-0 member. The two sets that existed at 390k
  (dw-noiseless, oth-adjacent-flip) are EXTENDED to 512k by exact resume; their 421,875 members
  fall outside the pooling window and stay on disk. The paper says so in one sentence; the tables
  print the budget beside every ±. The Othello point estimate drifts +0.58 → +0.61 → +0.62 over
  256k / 512k / 780k, about one SD, so the 512k spread is a fair estimate of the 780k spread.
- **Probe seeds are not re-measured per run**: the pilot found them an order of magnitude below
  the training-seed spread (`findings/seed-variance.md`; skill ±0.0007, Edit Index ≤ 0.02). Each
  replicate is re-scored in full (probes refitted, editors re-swept), so one draw of probe noise
  sits inside every ±. Cheap extras at the tail of the queue: 10 probe seeds on `L-oth-20m` and
  `L-dw-8ray-20m`, and on adjacent-flip's 512k members (its 390k probe seeds are parked).
- **The guard gets a spread too** (every editor's `fid` column is pooled), and a guard verdict is
  read as "k of n seeds".
- **Case-level (bench-resampling) spread is recorded beside every arm from now on**
  (`edit_index.case_stats`, `ratio_ci95`; REGISTRY "case-level spread") so the option exists later;
  it is never added to the seed spread and is not quoted in the paper.
- **Dataset variance is not measured** (20M-sequence corpora; the probe corpora's split variance is
  the probe seed; one fixed bench per instance across seeds) — stated in the paper.
- The supplemental runs (obs5, smooth, 8ray-tok, oth-mse) and the gridified probe layouts get
  NO spread; the time goes to the main numbers.

## The plan (`experiments/paper_ci/plan.py --show`)

Twenty-two replicate / extension jobs + two corpus pushes + five probe-seed extras + one tables
rebuild. Greedy two-host schedule from a cold start (all at 512k): **see `plan.py --show`; ≈ 4.2 days**
at 100% duty (lab RTX 5090 rate 1.0; WSL RTX 4090 rate 0.77). Data locality: dw-128ray and
dw-blink on the lab only (406 GB corpora), dw-16ray on the remote only (its corpus lives there),
dw-8ray / dw-5ray either side after a ~8 min corpus push, every Othello family either side.
Priority: ray family → standard Othello and adjacent-flip → blink → adjacent / noflip → extras.

## Bootstrap

`experiments/paper_ci/README.md` (operate), `harness/MULTIDAY.md` (protocol). Launch =
`rm experiments/paper_ci/state/PAUSED`. Dashboard `https://sevan-ubuntu-lab.tail9a3a96.ts.net/ci/`.
Ledger `experiments/paper_ci/dashboard/ledger.md`. Tables: `notebooks/build_paper_tables_and_figs.ipynb`
(Table 5 panels via `T.table_seed_variance(F, which="sd")` / `which="ci"`).

## Decision rules

A contrast between two rows is "outside the seed spread" when it exceeds ~3 pooled SDs
(n = 3 each). Since 2026-09-19 the tables (and this queue's ledger, which reads through them) report each editor's best arm INSIDE the
fidelity guard (`pim.metrics.selection.best_arm`; the unguarded best only when no arm passes, flagged `within_guard` False),
so "fails the guard" now means NO arm passes on that run — read it as k of n seeds.
Replicates pool only at a matched budget (±10%); a set with n < 2 prints no ±.

## Owed / open

- DONE 2026-09-19: Othello's PI / ND / GS arms carry the guard's case-level CI (`pim/scoring/othello.py::othello_arms`, after
  the scorer moved into `pim/scoring/`); smoke on `L-oth-20m` reproduced the canonical PI arm to the last digit.
- Bootstrap CI on the Othello ceiling (+0.91 symdiff on n = 13 / 29 / 32 ordinary cases):
  `ceiling_symdiff.py` has the per-case values; a 5-minute addition, no GPU.
- Gridified discworld rows: only `appearance-fac` gets a spread (Sevan: the 8-ray grid sweep goes
  to the appendix without one).
- Machine hardening DONE 2026-09-18 evening (Sevan): nvidia packages held, `apt-daily-upgrade.timer`
  stopped, the 5090 capped at 450 W (two PSU freezes at 530 W on 2026-09-16).
