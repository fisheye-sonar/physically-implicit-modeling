# seed_variance — how much do the metrics fluctuate across training seeds and probe seeds? (2026-09-11 night)

**Question (Sevan).** Before quoting contrasts like 8-ray vs 5-ray (GS +0.46 vs +0.58) or joint
cell vs factorised, how big is the seed-to-seed spread of Probe Skill and of each editor's best
Edit Index / guard — from re-training the model, and from re-fitting the probe?

**Pilot.** `L-dw-noiseless-20m`. (a) RUN seeds: two re-trainings with `--seed 1` / `--seed 2` at
390k steps (half the canonical budget; the loss is within 5% of final by then and editability
is flat from 64k on discworld), plus the canonical run's own step-421,875 checkpoint as the
seed-0 member of the set at the same budget — n = 3, pooled. Each is scored canonically and
under `appearance-fac`. (b) PROBE seeds: on the canonical run and the two replicates, the
LINEAR probe refitted with 20 seeds (regression, frustum) and 6 seeds (`appearance-fac`) at
EVERY residual point — the probe seed drives init and the held-out split — then PI (and ND on
the categorical target) swept at each seed's own best point and at the seed-0 best point.

**Convention (also in `harness/WORKFLOW.md` and `research/REGISTRY.md`).** A replicate is a
run dir `runs/<topic>/<parent>__seed<k>/` whose `config.json` carries
`replicate = {"of": "<topic>/<parent>", "seed": k, "steps": n}` (written by
`scripts/train.py --replicate-of`); a checkpoint-of-the-parent replicate adds `"checkpoint": true`
(`scripts/layout_checkpoint_replicate.py`). Replicates are scored by `master_eval` like any run
(their extra targets listed in SETTINGS under their own key) and are NEVER table rows:
`build_full_table` pools them per (parent, basis) into the parent row's **± (SD, ddof 1)** with
n and the step budget in Table 4a. Probe-seed spread lives in `runs/<run>/variance.json`
(`scripts/probe_seeds.py`) and is shown in Table 4b. Probe seeds other than 0 are cached beside
the canonical probes (the seed is in the cache key).

**Chain.** `scripts/drivers/seed_variance.sh` (unit `seed_variance`, `logs/seed_variance/`):
train seed 1 → train seed 2 → lay out seed-0 ckpt → score + `appearance-fac` on the three
replicates → probe seeds on three runs → tables. Result: `research/findings/seed-variance.md`
(to be written when it lands).
