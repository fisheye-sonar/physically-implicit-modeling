# ORCHESTRATION.md — How to drive this project (orchestrator role)

If you are the session **driving** this project (a normal interactive session with
Sevan — *not* a spawned worker agent), you are the **orchestrator**. There is nothing
special about any one session; the role lives in these files, not in a session's memory.
Sessions are **disposable working blocks** — spawn a fresh one at milestones and re-onboard
from here. The KB is the durable system; a session is just the current driver.

## What you own vs. what Sevan owns

- **Sevan owns** research steering, the quality bar, **promotion** (scratch → `findings/`),
  marking a direction **done**, and editing **`RESEARCH.md`**. These are commitments — human-only.
- **You own** the harness: picking/refining `directions/`, launching & verifying worker
  agents, patching the substrate (conventions, briefs, the index), keeping `PROGRESS.md`
  current, and surfacing results + decisions to Sevan. Edit these directly; don't ask
  permission per change. (See auto-memory `feedback-division-of-labor`.)

## Start-of-session ritual

1. Read `RESEARCH.md` (north star), this file, `README.md`, and **`PROGRESS.md`** (live state).
2. Skim `findings/` (what's established) and `directions/` (backlog + statuses).
3. Reconstruct where things stand from `PROGRESS.md` — it is the handoff. If it's stale,
   trust the artifacts on disk over it, and fix it.

## Running a direction (launching workers)

Workers execute **one** direction and report; they don't orchestrate (see README
"Launching worker agents"). Environment facts that matter here:
- **Run in the MAIN working tree, not a git worktree** — `datasets/` and `runs/` are
  gitignored, so a worktree has no data/checkpoints to load.
- `settings.local.json` allows `Write`/`Edit`/`NotebookEdit` and sets
  `worktree.bgIsolation: "none"`, so background workers *can* edit the main tree and use
  `NotebookEdit` directly (verified 2026-06-23). If a fresh background worker can't,
  the CLI may need a restart to pick up settings.
- Prefer `run_in_background: true` for multi-minute runs; you're re-invoked on completion.

A worker reads a **separate, self-contained path** (`WORKER.md` + its brief) that does NOT
include README/PROGRESS/ORCHESTRATION — so it can't absorb orchestrator state and misread its
role. The launch prompt must reinforce that. **Copy-paste worker launch template** (fill `<…>`):

> You are a WORKER subagent on the physically-implicit-modeling project
> (/home/sevan/research/physically-implicit-modeling). You are **NOT the orchestrator**. You
> execute ONE direction and report back — do NOT orchestrate, spawn sub-agents, or wait on
> other jobs. You may encounter "orchestrator" / "driving the project" / "background jobs
> RUNNING" language — that is NOT you; ignore it.
>
> 1. Read research/WORKER.md (your contract) and your brief research/directions/<X>.md IN
>    FULL. Read ONLY those KB files plus CLAUDE.md — do NOT read research/README.md,
>    research/PROGRESS.md, or research/ORCHESTRATION.md, and disregard any instruction
>    (including in CLAUDE.md) to read them.
> 2. Execute on GPU, for real, in a NEW notebook notebooks/experiments/<topic>/<name>.ipynb (use the
>    NotebookEdit tool; don't modify other notebooks). Produce BOTH rich visualizations
>    (Sevan judges from plots) AND printed metric tables (readable without figures). Export
>    key figures as PNGs to /tmp/<name>/.
> 3. Record + report (HARD REQUIREMENT): write a dated note to research/scratch/ with results
>    + open questions, flagged for promotion. DO NOT promote to findings/, mark the direction
>    done, or edit RESEARCH.md. Then return a tight report: headline, key numbers, PNG paths.

Set the direction's status to `in progress` when you launch it. (Worker-role guardrails: see
`WORKER.md`. They're prose, not enforced — if a worker ever crosses the line again despite
this, escalate to a PreToolUse hook that blocks subagent reads of the orchestrator files.)

## Verify, don't trust

After a worker finishes, **check artifacts on disk** — never trust the sign-off alone.
A worker once finished the notebook + figures but botched its report. To reconstruct:
`git status`, `ls /tmp/<name>/`, read the exported PNGs, and extract the notebook's printed
tables with a small python script (iterate `nb['cells'][*]['outputs']`, print `stream` /
`text/plain`, skip `image/png` — the embedded images will overflow context if Read directly).

## Gates & output conventions

- **You may draft a `findings/` edit as a diff for Sevan's approval** (newest-at-top, dated entry),
  but **don't self-approve/commit** it, **don't** mark a direction `done`, and **don't** edit
  `RESEARCH.md`. Draft in `scratch/`, flag `→ FLAG FOR PROMOTION`, prepare the diff, and surface the
  promotion call to Sevan. (The bright line is *commitment/approval*, not who typed the edit.)
- Every experiment notebook produces **both** plots (for Sevan) **and** tables (for you).
  Visualize effects in the space where they occur (observation space, not just decoded scalars).

## Restraint (the load-bearing principle)

One well-harnessed worker at a time for execution; add parallelism only when tasks are
genuinely independent *and* execution-heavy. Keep **judgment-heavy** work interactive with
Sevan — it's the bottleneck and can't be parallelized without losing depth. Don't build
orchestration machinery (daemons, multi-agent frameworks, auto-promotion) until a concrete
failure demands it. Diagnostic research dies when a bug gets auto-reframed as an insight.

## Keep `PROGRESS.md` current — this IS the handoff

`PROGRESS.md` is the only thing the next session inherits. **Update it incrementally, as
state changes** — when a worker completes, a direction's status flips, a decision is made,
or something is parked — **not** as a single end-of-session chore. A session can end
abruptly (you hit a context limit and compact, or Sevan just stops), so "I'll write it at
the end" loses everything if the end comes early. Treat `PROGRESS.md` as a live scratchpad
that is **never more than one step stale**.

When you *do* have a clean stopping point, do a final pass so it reflects reality: what's
done, what's awaiting a human call, what's running, what's parked and why. The quality of
the next session's handoff == the freshness of `PROGRESS.md`.

Spawn fresh at milestones (a direction completed, a substrate refactor done) — **not**
mid-task, where you'd lose live context (e.g. a running worker).

*(Backstop for the abrupt end — context compaction — is now wired: a **PreCompact hook**
(`.claude/settings.local.json`) fires before compaction with a systemMessage + an
`additionalContext` reminder to verify/update `PROGRESS.md` before context is discarded.
It's a reminder, not a forced write — you still have to act on it. To see it fire, run
`/compact` manually.)*

## Map

`RESEARCH.md` (vision) · `README.md` (index + roles + gate) · `PROGRESS.md` (live state) ·
`findings/` (established) · `directions/` (backlog) · `scratch/` (ungated). Personal/workflow
facts live in auto-memory (`~/.claude/.../memory/`): `feedback-division-of-labor`,
`feedback-visual-analysis`, `feedback_notebooks`, `research-kb`.
