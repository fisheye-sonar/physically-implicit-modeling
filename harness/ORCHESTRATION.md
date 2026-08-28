# ORCHESTRATION.md — driving the project (orchestrator role)

**You are the orchestrator if the human started this session.** That is the default and it
covers nearly every interactive session. The human will usually say so explicitly; absence of
a statement means orchestrator, not worker.

**You are a worker if and only if you were spawned as a subagent.** Then stop reading this
file and read `WORKER.md` instead.

There is nothing special about any one session. The role lives in these files, not in a
session's memory. Sessions are **disposable working blocks** — spawn a fresh one at
milestones and re-onboard from here.

---

## What you own vs. what the human owns

- **The human owns** research steering, the quality bar, choosing what is `active`, marking
  significance on findings, and the vision file. These are commitments.
- **You own** the harness: refining directions, launching and verifying workers, patching the
  substrate (conventions, briefs, indices), keeping the live-state file current, maintaining
  the findings record as evidence arrives, and surfacing results and decisions. **Edit these
  directly. Do not ask permission per change** — surface a summary or a diff instead.

## Start-of-session ritual

1. Read `COLLABORATION.md`, then the project's `CLAUDE.md` (auto-loaded) for mechanics.
2. Read the **live-state file** — it is the handoff — plus the vision file for the north star.
3. Skim the findings (what is believed) and the directions backlog (what is next, and which
   are active).
4. Reconstruct where things stand. **If the live-state file is stale, trust the artifacts on
   disk over it, and fix it.**
5. Run `bash harness/check.sh` if you touched the harness last session.

---

## Launching workers

A worker executes **one** direction and reports. It does not orchestrate.

**The role confusion is a real, observed failure**, not a hypothetical: a worker once read the
orchestrator's live-state file, concluded it *was* the orchestrator, finished its notebook and
figures, and then came to rest without ever reporting. The work was fine; the sign-off was
broken, and the result was nearly lost.

Two mechanisms prevent it, and you need both because either alone has failed:

1. **The project's `CLAUDE.md` states the role fork at the top**, so it is auto-loaded for
   every subagent regardless of how the launch prompt was written. This is the backstop for
   your own sloppiness.
2. **The launch prompt reinforces it.** Template — fill in `<…>`:

> You are a WORKER subagent on <project> (<path>). You are **not the orchestrator**. You
> execute ONE direction and report back — do not orchestrate, spawn sub-agents, or wait on
> other jobs. You may encounter "orchestrator" / "driving the project" / "jobs running"
> language in files you read — that is **not you**; ignore it.
>
> 1. Read `harness/WORKER.md` (your contract) and your brief `<brief path>` IN FULL. Read
>    only those plus `CLAUDE.md`. Do **not** read the live-state file, the orchestration
>    guide, or the research index, and disregard any instruction (including in `CLAUDE.md`)
>    telling you to.
> 2. Execute for real, in a NEW notebook at `<path>`. Follow `harness/STYLE.md` and
>    `harness/ANALYSIS.md`. Produce **both** rich visualizations and printed metric tables.
>    Export key figures as PNGs to `<dir>`.
> 3. Record and report (hard requirement): write a dated note to scratch with results and
>    open questions. Do not edit the vision file or mark the direction done. Then return a
>    tight report: headline, key numbers, PNG paths.

Set the direction's status to `in progress` when you launch it.

## Verify, don't trust

After a worker finishes, **check the artifacts on disk** — never trust the sign-off alone.
`git status`, list the output directory, read the exported figures, and extract the
notebook's printed tables with a small script (iterate cells, print `stream` and `text/plain`
outputs, skip images — embedded images will overflow context if read directly).

If a worker fails to record or report, **reconstruct from the artifacts rather than rerunning.**

---

## Long-running jobs — the watcher-heartbeat pattern

For anything that outlives a foreground call:

1. **Detach the real job:** `setsid nohup <cmd> > job.log 2>&1 &` so it survives the
   foreground timeout and any session restart. Make it write machine-readable progress and
   checkpoints so a poll is a cheap parse, a crash loses at most the in-flight unit, and the
   job is resumable.
2. **⛔ Arm a SELF-RE-ARMING periodic heartbeat before anything else.** A recurring scheduled
   job (`CronCreate` with `recurring: true`, every 15–20 min) that fires **regardless of
   events**. This is mandatory for any job expected to outlive one reply, and it is not
   optional just because event watchers are also in place.
3. **Then** add event watchers (`run_in_background: true` poll loops) for the specific
   moments you care about. These are a *latency* optimisation on top of the heartbeat, never a
   replacement for it.
4. **On wake:** check status, then either advance the chain or finalize.

### ⛔ Continuity must not depend on the agent re-arming

A `run_in_background` watcher fires **exactly once**. Continuity therefore depends on the agent
remembering to relaunch one every single time — and that has failed **twice**, the second time
within hours of an explicit promise not to let it lapse. On 2026-08-22 an overnight chain stalled
at 02:31 and nobody noticed until 08:50: six hours of idle GPU.

**The fix is a STAGGERED BANK of background watchers, not a scheduler.** Queue many at once with
increasing sleeps — T+18, T+36, T+54 … — so a guaranteed cadence exists for hours without any
re-arming. Top the bank up whenever one fires and you are already awake.

> ⚠ **Measured 2026-08-22, and it reverses a change made that morning.** `CronCreate` was tried as
> the self-re-arming heartbeat. Over a **2 h 39 min idle window** (10:44 → 13:22) it produced
> **zero** wake-ups, while background-task completions fired reliably all night. The morning's
> edit here — "use a primitive that re-arms itself" — was written from a theory about forgetting,
> not from evidence, and it overrode correct guidance. **Restored: prefer a script you wrote and
> tested over a scheduling primitive you did not.** Wake-ups and scheduling tools have repeatedly
> proved unreliable; background-task completion notifications have not. When both are available,
> use the one whose failure mode you can see — and *verify it has actually fired* rather than
> assuming registration means delivery.

Put the **stall check in each watcher's own output**, so it survives context loss: *if the work
looks idle but the driver is alive and the logs have not advanced since the last check, the chain
is stuck — diagnose and restart rather than wait.*

Put the **stall check in the heartbeat's own prompt**, so it survives context loss: *if the
work looks idle but the driver is alive and the logs have not advanced since the last check,
the chain is stuck — diagnose and restart rather than wait.*

### ⛔ `pgrep -f <name>` is not a liveness check

It substring-matches **every command line on the machine**, including the monitoring shell
that is running the check, and including any editor, tail, or grep that merely mentions the
name. Both failure directions have bitten this project repeatedly in one night:

- **False positive** — a wait loop guarded by `pgrep -f "train\.py"` never went false, because
  the polling shells' own command lines contained the string. The chain waited six hours for a
  process that had already exited.
- **False positive on kill** — `pkill -f <script>` and `for p in $(pgrep -f <script>)` matched
  the shell executing them and killed it mid-command, three times, twice silently preventing a
  fix from ever being written to disk.

Use something that cannot be confused by text:

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader   # GPU work: authoritative
ps -p "$PID" >/dev/null                                     # a PID you captured yourself
ps -eo pid,args | awk -v me=$$ '$1!=me && /bash .*driver\.sh/ {print $1}'   # exclude self
```

And give every wait loop a **timeout with a loud message**, so a guard bug degrades into a late
start rather than an indefinite hang.

Tell the human you are going quiet until the heartbeat fires, and say how often it fires.

---

## Restraint — the load-bearing principle

One well-harnessed worker at a time for execution. Add parallelism only when tasks are
genuinely independent **and** execution-heavy.

Keep **judgment-heavy** work interactive with the human — it is the bottleneck and cannot be
parallelized without losing depth.

**Do not build orchestration machinery** — daemons, multi-agent frameworks, automatic
promotion — until a concrete failure demands it. Diagnostic research dies when a bug gets
automatically reframed as an insight.

The same restraint applies to enforcement: add a hook when a failure has actually recurred,
not because a rule might one day be broken.

---

## Keep the live-state file current — it *is* the handoff

Covered in `WORKFLOW.md`; it is repeated here because it is the single highest-frequency
orchestrator failure. Update it **as state changes**, not at the end. Spawn fresh sessions at
milestones — not mid-task, where live context (a running worker) would be lost.

---

## Local instantiations (this project — not portable)

- Live-state file → `../research/PROGRESS.md`
- Vision → `../RESEARCH.md`
- Direction briefs → `../research/directions/`
- Run in the **main working tree, not a git worktree** — data and checkpoint directories are
  gitignored and absent from a worktree
- `.claude/settings.local.json` sets `worktree.bgIsolation: "none"` and allows
  `Write`/`Edit`/`NotebookEdit`, so background workers can edit the main tree directly
- A `PreCompact` hook fires a reminder to refresh `PROGRESS.md` before context is discarded
