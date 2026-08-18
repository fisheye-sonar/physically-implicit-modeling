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
2. **Launch a harness-tracked watcher** with `run_in_background: true`: a poll loop that sleeps
   and exits when either ~25 minutes elapse or the job finishes or dies (grep the log for a
   completion sentinel, or check the process). Its completion fires a task notification that
   **reliably re-invokes you** — that is the wake signal.
3. **On wake:** check status, then either relaunch a fresh watcher or finalize.

**Prefer a script you wrote and tested over a scheduling primitive you did not.** Wake-up and
scheduling tools have proved unreliable in practice — a pending wake can be cancelled by
ordinary session activity — while background-task completion notifications have fired every
time. When both are available, use the one whose failure mode you can see.

Tell the human you are going quiet until the watcher fires.

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
