# research/ — Research Knowledge Base (index)

The durable, version-controlled memory for the **science** of this project. The filesystem is
the memory — plain dated markdown, no database. Load detail files lazily.

- **How the record works** (roles, findings model, directions lifecycle) → `../harness/WORKFLOW.md`
- **How to drive the project** → `../harness/ORCHESTRATION.md`
- **If you are a spawned worker** → `../harness/WORKER.md` and your brief, and **not this file**
- **Engineering mechanics** → `../CLAUDE.md`
- **The north star** → `../RESEARCH.md`

## What is here

| File / dir | Answers | Who writes |
|---|---|---|
| `../RESEARCH.md` | what are we trying to learn? | **Sevan only** — agents read, never write |
| `PROGRESS.md` | where is the work right now? | agent, freely, continuously |
| `findings/` | what do we believe, and how strongly? | **agent, continuously** — graded, not gated |
| `directions/` | what should we do next? | agent proposes; **Sevan marks `active`** |
| `scratch/` | what did this run show? | agent, freely, ungated |
| `GOTCHAS.md` | what will silently waste a day? | agent |

## The model, in one paragraph

**Findings are written by the agent as evidence arrives** — there is no promotion queue.
Every entry carries a **status** (`observed` / `replicated` / `established`) reflecting how
strongly it is held, and its **evidence** (notebook, scratch note, run codes, dataset, n).
**Sevan marks `★` for significance**, which is orthogonal to status. Corrections are new dated
entries that `supersede` or `retract` earlier ones; nothing is ever rewritten or deleted.

*(Changed 2026-08-17. The previous design required human approval before anything entered
`findings/`. Measured outcome: the newest entry in any findings file was 2026-07-17 while 25
scratch notes queued behind the gate — a month in which the "what's true" record described a
project that no longer existed, and the real synthesis migrated into `PROGRESS.md`, which is
explicitly the volatile file. The gate's actual purpose — keeping "seen once" separate from
"established" — is now served by grading each entry rather than by blocking the write. Full
rationale in `../harness/WORKFLOW.md`.)*

## The two human powers

**The vision** (`../RESEARCH.md`) and **what to work on next** (marking a direction `active`).
That is all. Everything else the agent writes and Sevan reviews as a document.

The standard that replaces the old gate is unchanged in substance and now lives inside each
entry: **artifact or signal?** State scope, sample size, caveats, and what would falsify the
claim, in the same breath as the claim. Never soften that question to make a result land.

## Map

- `PROGRESS.md` — live state + handoff (read at the start of every orchestrator session)
- `findings/` — what we believe, by concept, graded and dated
- `directions/` — candidate experiments; `done/` for closed ones
- `scratch/` — raw observation surface; the provenance findings entries cite
- `GOTCHAS.md` — project traps and non-comparable historical numbers
