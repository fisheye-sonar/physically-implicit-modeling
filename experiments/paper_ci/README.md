# paper_ci — the paper's seed-replicate queue (2026-09-18)

**Question (Sevan).** The paper's main tables quote decodability, the Edit Index and the
fidelity guard for ten shortlist runs with no spread. Put a training-seed spread (n = 3: the
parent's checkpoint at a matched budget + two re-trainings) on every one of them, across the
two GPUs, in under five days, with everything landing in the canonical tables. The protocol is
`harness/MULTIDAY.md`; the direction brief is `research/directions/paper-confidence-intervals.md`.

## What runs

`plan.py` holds THE job list (families, budgets, hosts, estimates) and writes `queue/*.json`;
`python plan.py --show` prints the plan with a greedy two-host schedule and its makespan. The
jobs are `scripts/drivers/replicate.sh` calls (one replicate end to end: train / extend →
seed-0 member → categorical targets → master_eval in both bases), two corpus pushes, a few
probe-seed extras, and one final tables rebuild on the lab box.

## Operating it

| | |
|---|---|
| launch | `rm experiments/paper_ci/state/PAUSED` (the dispatcher timer is already ticking) |
| pause launches | `touch experiments/paper_ci/state/PAUSED` (running jobs continue; probing and the dashboard continue) |
| hold / release one job | add `"hold": true` to `queue/<id>.json` (or `state/<id>.json`) / remove it |
| reset a failed job | `rm state/<id>.json` (its run dir is left alone; `replicate.sh` resumes from its checkpoint) |
| add a job | write `queue/<id>.json` (copy a neighbour) — or edit `plan.py` and re-run it (started jobs are never rewritten) |
| stop a running job | `systemctl --user stop pimci-<id>` on its host; the dispatcher requeues it (attempts permitting) |
| one tick by hand | `.pim/bin/python experiments/paper_ci/scripts/dispatch.py` (`--dry` launches nothing) |
| timers | `systemctl --user list-timers 'pimci-*'`; logs `logs/paper_ci/dispatch.log`; per job `logs/paper_ci/<id>/` |
| dashboard | `https://sevan-ubuntu-lab.tail9a3a96.ts.net/ci/` (phone: add to home screen); files in `dashboard/` |
| ledger | `dashboard/ledger.md` / `.csv` / `.json` — rebuilt by `scripts/ledger.py` whenever a job finishes |
| channels | alerts: the usual ntfy topic; digest every 2 h: `pim-ci-digest-ai691k` (subscribe separately) |

Hosts: `lab` (source of truth; the dispatcher, watchdog and dashboard run here) and `remote`
(`wsl-sevan`; watchdog only). A remote job's inputs are rsynced there before launch and its
outputs pulled back when it finishes; the tables only ever read the lab tree.

## Where things land

Replicates: `runs/<topic>/<parent>__seed<k>/` (+ `__seed0_s<step>` members), scored like any
run (`scores.json`); the tables pool them into the parent row's ± (`build_paper_tables_and_figs`
Tables 1–2 cells, Table 5 SD and CI panels). Probe seeds: `runs/<run>/variance.json`. The
ledger mirrors the same numbers per (run, block, metric).

## Files

`config.json` hosts / channels / thresholds · `plan.py` · `queue/` job definitions (tracked) ·
`state/` runtime (gitignored) · `dashboard/` (page tracked, data gitignored) · `scripts/`
`dispatch.py` `dispatch_sim.py` `run_job.sh` `hostprobe.py` `watchdog.py` `ledger.py`
`serve.py` `install_units.sh`.
