# research/ — Research Knowledge Base (index)

**Read this file + `PROGRESS.md` at the start of every research session.**
This is the durable, version-controlled memory for the *science* of the project
(the engineering reference is `../CLAUDE.md`; the north star is `../RESEARCH.md`).
The filesystem is the memory — plain dated markdown, no database. Load detail
files lazily, only when the topic is relevant.

> **If you are *driving* this project** (an interactive session, not a spawned worker),
> you are the orchestrator — also read **`ORCHESTRATION.md`** for the operator's manual
> (how to launch & verify workers, the role split, session handoff). Sessions are
> disposable; the role lives in these files.

## The four roles (different owners, different edit cadences)

| Artifact | Where | Agent may read | Agent may write | Becomes durable when |
|---|---|---|---|---|
| **Vision** (north star) | `../RESEARCH.md` | yes | **no** (human-only) | human edits it |
| **Progress / handoff** ("where am I") | `PROGRESS.md` | yes | yes (free, rewritten each session) | — |
| **Scratch / observations** | `scratch/` | yes | yes (free, ungated) | — |
| **Findings ledger** ("what's true") | `findings/` | yes | **draft diffs only** (human approves) | human approves the diff + promotion |
| **Directions backlog** ("what's next") | `directions/` | yes | **propose only** | human marks it active |

## The one invariant

**Drafting, surfacing, and proposing are agent powers. Promotion and commitment
are human powers.** An agent may write a candidate finding into `scratch/` and
**may draft the corresponding `findings/` edit as a diff for human approval** — but
it may **not** self-approve or commit it; the promotion decision and the approval
stay human. An agent may propose a new entry in `directions/`; it may **not** mark a
direction active. The vision file (`RESEARCH.md`) is human-authored only. Keep the
bright line at *commitment*, not at *typing*.

## Promotion gate (scratch → findings)

A result crosses from `scratch/` into `findings/` only after it passes
**"artifact or signal?"** — an explicit human check that the effect is real and
not a bug, a confound, or a metric artifact. Until then it is an *observation*,
not a *finding*. This gate is what stops a reframed bug from compounding silently
across sessions. The agent may now **prepare the `findings/` diff** (newest-at-top,
dated entry) to save the human keystrokes — the gate is Sevan's review/approval of
that diff, *not* who typed it. (Enforced by discipline + this rule; a PreToolUse hook
is a planned upgrade — see `directions/` / TODOs, not yet wired.)

## Conventions

- **Findings** are organized one file per *concept*; within a file, **dated,
  append-only** entries (newest at top under a short mutable "current
  understanding" summary). Corrections are *new dated entries*, not edits to old
  ones — the record of how understanding changed is itself valuable.
- **Directions** are one file per candidate experiment, each tagged
  `[in-frame]` (a variation on the current approach) or `[reframe]` (changes the
  question/premise). This keeps the backlog from filling with near-variants
  wearing a novelty costume. A fresh session can be pointed at a single direction
  file: "read `directions/<x>.md` and execute."
- Dates are absolute (`YYYY-MM-DD`). Today's framing may be wrong tomorrow; the
  timestamp is how a future reader calibrates.

## Launching worker agents

An agent spawned to execute ONE direction is a **worker**, not an orchestrator. A worker
reads a **separate self-contained path** — `WORKER.md` + its assigned brief, and nothing
else from this KB (not this README, not `PROGRESS.md`, not `ORCHESTRATION.md`) — so it never
absorbs orchestrator state and misreads its role. Launch it with the template in
`ORCHESTRATION.md`. Require the worker to:
- execute only its assigned direction — **do not** spawn sub-agents, orchestrate, or
  "wait on other jobs." *(2026-06-23: a geodesic worker read the orchestration meta-state
  in `PROGRESS.md`, concluded it was the orchestrator, and came to rest without reporting
  — even though it HAD finished the notebook + figures. Work fine; sign-off broken.)*
- **end by writing its `scratch/` note AND returning a structured report** — hard
  requirement. The note is the durable record; the report is for the orchestrator.

Orchestrator side: **always verify artifacts on disk** (notebook, scratch note, exported
PNGs) — never trust a worker's sign-off alone. If a worker fails to record/report,
reconstruct from the artifacts (extract printed tables from the notebook, read exported
figures) rather than rerunning.

## Map

- `../RESEARCH.md` — vision / north star (human-owned)
- `../CLAUDE.md` — engineering reference (how the code is organized, how to work)
- `PROGRESS.md` — current session state + handoff
- `findings/` — durable established results, by concept
- `directions/` — backlog of candidate experiments (agent proposes, human disposes)
- `scratch/` — free observation surface (ungated; nothing here is "true" yet)
