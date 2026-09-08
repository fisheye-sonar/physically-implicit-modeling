# OVERNIGHT.md — the protocol for any job that outlives a reply

Training runs, corpus builds, scoring chains: anything the human will not watch to the end.
This is the operational checklist; the reasoning and the failure history behind it are in
`ORCHESTRATION.md` ("Long-running jobs"). Every item here was paid for once.

## 1. Before launch

- **Smoke the exact code path first**, end to end, on a tiny configuration: train a few
  steps, save, reload, score with the real scorers. A chain that fails at the scoring stage
  at 3 a.m. wastes the night. If the scoring side is not ready yet, the smoke covers what
  IS ready and the driver waits for a ready-marker before the rest (see §3).
- **Every fitted artefact is persisted before it is used** — answer "where is it saved?"
  before launching anything that fits, pilots included. Recomputing is never the plan.
- **One heavy job at a time**, under a memory cap. Two of them have taken the desktop down.
- Estimate the wall-clock from a measured rate (a comparable run, or the first thousand
  steps) and tell the human the ETA.
- Acknowledge the launch order on the notification channel as soon as you read it.

## 2. Launch — a capped, detached unit running a staged driver

```bash
systemd-run --user --unit=<name> -p MemoryMax=<N>G --collect --working-directory=$PWD \
    /usr/bin/bash -c 'bash <driver>.sh > <logs>/unit.log 2>&1'
```

- The unit survives the launching session; the cap turns a leak into a killed unit
  instead of a frozen machine. `--collect` lets a failed unit be relaunched under its name.
- The driver is a **staged shell script**: `set -u`; one `stage` marker line per stage
  written to a driver log; a `fail()` that notifies with the tail of the failing stage's
  log and exits; a final `chain complete` marker. Each stage redirects to its own log.
  `export PYTHONPATH=$ROOT` — imports under a unit do not inherit the shell.
- **Python stdout under a unit is block-buffered.** Progress must come from something
  the job flushes — a metrics file written per validation pass, a checkpoint directory —
  or run the interpreter with `-u`. A log that stays empty for hours is not a stalled job.
- Never delete from the data trees to make room or to "clean up" a failed attempt: move.

## 3. Watch — two monitors, armed right after launch, both of which you can see fire

1. **A stage watcher** (persistent) that polls the driver log for `STAGE`, `FAILED`,
   `chain complete` AND checks the unit's state: a unit that is inactive without the
   completion marker is a failure and must be reported, not silence. Coverage means every
   terminal state emits a line — ask "if this crashed now, would the watcher say so?".
2. **A heartbeat** (persistent, every 30 min) that prints step, train/val loss, best
   so far, GPU memory and utilisation, the unit's memory, and disk free — from the
   flushed files, not from buffered stdout. Guard every arithmetic on a value that can be
   `[not set]`. It exits on its own when the unit ends.

- Prefer a script you wrote and can watch over a scheduling primitive; scheduled
  wake-ups have failed silently for hours, background-task completions have not.
- Liveness is `systemctl --user is-active <unit>`, a PID you captured, or the GPU's
  process list — never `pgrep -f <name>` (it matches the shell running the check).
- If a stage waits on a marker (§1), the wait has a timeout with a loud message.
- If the machine reboots or the session dies, the unit keeps running; the first thing a
  resumed session does is re-arm both monitors from the log and metrics files.

## 4. Notify — short, at key points only

Channel: see Local instantiations. Pings: order acknowledged · chain started (with ETA) ·
each long stage done · any failure (with the log tail) · all done (with the headline
numbers). One or two lines each. Heartbeats do not go to the channel.

## 5. While it runs

- Build what the results will need — scorers, registry rows, the write-up skeleton — so
  they land the moment the chain finishes. Do not end the session; reply to heartbeats
  in one line; do not poll.
- Do not restart a running job for cosmetics (an empty log, a nicer name). Fix forward.

## 6. When it finishes

Read the results and write them where they belong (registry row, findings note, memory),
ping the headline, stop the heartbeat, move smoke artefacts into an archive folder, and
give the human a recap that stands alone: what ran, what came out, what is left.

## Local instantiations

- Notification channel: `https://ntfy.sh/swirling-tornado-ai691k` (`curl -d "<text>" -H
  "Title: PIM <run>: <event>"`).
- Units: `systemd-run --user --unit=<name> -p MemoryMax=40G --collect
  --working-directory=$PWD …`; logs in `logs/<topic>/<run>/` (`unit.log`, `driver.log`,
  `<stage>.log`); drivers in `scripts/drivers/` (example: `scripts/drivers/oth_mse.sh`);
  the ready-marker pattern for a scorer built while the model trains: `scripts/drivers/dw_tokens.sh`.
- Monitors: the `Monitor` tool with `persistent: true` (stage watcher + 30-min heartbeat
  reading `runs/<topic>/<run>/metrics.jsonl`); `TaskStop` when done.
- Probe caches live inside the run (`runs/<topic>/<run>/probes/`); the memory note
  `never-discard-fitted-probes.md` holds the rule.
