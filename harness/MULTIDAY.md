# MULTIDAY.md — the protocol for work that spans days or machines

`OVERNIGHT.md` covers one job on one machine that a human will look at in the morning. This
file covers the next size up: a SET of jobs, on more than one machine, over several days, that
must keep running and keep reporting while no session is watching. Everything in
`OVERNIGHT.md` still applies to each job; this file adds what the scale demands. Every item
here was paid for by a chain that stalled, a session that ended, or a box that froze.

## 1. A queue, not a chain

- One job file per unit of work (a training, a scoring, a transfer): id, command, environment,
  the hosts allowed to run it (data locality decides), the lane it occupies (`gpu` or `cpu`),
  dependencies, priority, an estimate per host, the inputs it needs pushed and the outputs to
  pull back, the progress file to read for a live ETA, a retry budget.
- **Every job is idempotent and resumable on its own**: re-running it after a kill costs at
  most the in-flight stage. A job that cannot be re-run is not a job; split it.
- A plan script writes the queue and can SIMULATE it (greedy list schedule on the hosts) so the
  makespan is known before anything launches, and again on the live state for the ETA.
- Priorities follow what the result is FOR: the numbers a claim leans on first, the ones that
  decorate an appendix last. Independent work is never serialised by convenience.

## 2. A dispatcher that is an OS timer, not a session

- The dispatcher is a short, stateless TICK run by a `systemd --user` timer (with linger) on
  ONE host, the source of truth. Each tick: poll every host → fold finished jobs in → launch on
  free lanes → render the dashboard → alert / digest. No tick depends on the previous one
  having run in the same process; state lives in files; a lock prevents overlapping ticks.
- A remote job is launched as a DETACHED unit on the remote (`systemd-run` over ssh) so it
  survives the dispatcher, the network and the dispatcher's own host going down. Its inputs are
  pushed right before launch and its outputs pulled the moment it finishes; nothing is assumed
  to be in two places at once.
- A job's own wrapper records start / end / exit code on ITS host, in a file. A unit that
  disappears without an end record is a failure to retry, not a mystery to wait on.
- A `PAUSED` flag holds launches without stopping the polling and the dashboard; per-job
  `hold` does the same for one job. Resetting a job is deleting its state file.

## 3. A watchdog on every host, independent of the dispatcher

Each host runs its own timer that shares nothing with the dispatcher but the job files, and
speaks even when the dispatcher, the network or the other host is gone:

- a unit in the failed state; an active unit whose progress file and logs have not moved in N
  minutes (stall); disk below a floor; the GPU not answering;
- on the dispatcher's host: the dispatcher not ticking;
- on every other host: the dispatcher's host not answering (`tailscale ping` or equivalent),
  after K consecutive checks, and its recovery.

Every alert is rate-limited per key. A persistent condition pings once, not every five minutes.
The job wrapper pings its own failures with the log tail, so a failure is never silent even if
the dispatcher is.

## 4. Notifications — two channels

- **Alerts** (loud): job failed, unit died, host unreachable / back, disk low, launch failed,
  stall, dispatcher not ticking, and every job's completion WITH the numbers it produced (the
  ledger lines for its group) and how many jobs remain.
- **Digest** (quiet, its own topic, every N hours): per host what is running and how far along,
  counts, the global ETA, the dashboard link. The human mutes this one when they sleep.

## 5. A dashboard the human can open on a phone

Static files rendered by the dispatcher (a `state.json` the page fetches every minute), served
on the private network (`tailscale serve` gives HTTPS and an installable page). It shows: the
global ETA and when it was computed (stale = red); per host GPU / power / disk / uptime / what
is running with a progress bar and its own ETA; the queue with predicted start and end per job;
done and failed jobs with timestamps, durations and attempts; the scores LANDING (canonical
value, n, mean ± spread, every member) so a wrong number is seen while there is still time; the
event log. The same numbers are written to disk as a CSV and a markdown ledger.

## 6. Harden the machines for the duration

Before launch, for each host: power-cap a GPU that has frozen the box (a 5–10% slowdown beats a
reboot at 3 a.m.); hold the driver packages and pause unattended upgrades (a driver/library
mismatch mid-week fails every NEW job); confirm the keep-alive of a VM that dies without a
session; confirm linger; budget the disk (every replicate's size × count + headroom, on every
host); confirm the network path both ways.

## 7. Pre-flight checklist

1. Code is IDENTICAL on every host (one commit; pull, don't copy). Data splits, floors and the
   parent runs the jobs read are synced; training corpora move only when the plan says so.
2. **Smoke the real launch path on every host**: a trivial job through the dispatcher —
   launch, record, finish, sync-back, dashboard, ping. Then the first real job is watched by a
   person until its first stage completes.
3. The plan simulation is read and its makespan and host balance written into the brief with
   the date; the human knows the ETA before saying go.
4. The queue is written, `PAUSED` is set, the timers are installed and seen ticking, the
   dashboard opens on the phone, the digest topic is subscribed. Removing `PAUSED` is the launch.

## 8. While it runs

- The agent does not poll; it reads the dashboard and the pings like the human does, and
  verifies every completion against the artefacts on disk (scores files, table cells), never
  against the ping alone.
- Fix forward: a failed job is fixed and its state reset; the queue is never rewritten under a
  running job (the plan script leaves started jobs untouched).
- When a session ends or a box reboots, nothing has to be re-armed: the timers are the memory.

## 9. When it drains

The ledger is the first read; the tables are rebuilt once on the source-of-truth host; the
record (findings, registry, live state) is updated from the tables, not from the pings; the
smoke artefacts are moved to an archive folder; the human gets a recap that stands alone.

## Local instantiations (this project — not portable)

- Reference implementation: `experiments/paper_ci/` (2026-09-18, the paper's seed-replicate
  queue): `plan.py` (jobs + simulation), `scripts/dispatch.py` (tick), `scripts/run_job.sh`
  (wrapper), `scripts/hostprobe.py` (poll), `scripts/watchdog.py`, `scripts/ledger.py`,
  `scripts/install_units.sh` (systemd user units), `dashboard/`. Its `README.md` is the
  operator's manual.
- Hosts: the lab box (source of truth, dispatcher) and the WSL box (`wsl-sevan`; memory note
  `wsl-remote-host` for how it stays alive). Channels: `OVERNIGHT.md` §Local instantiations
  (alerts) plus the digest topic in `experiments/paper_ci/config.json`.
- Dashboard: `https://sevan-ubuntu-lab.tail9a3a96.ts.net/ci/` via `tailscale serve`.
