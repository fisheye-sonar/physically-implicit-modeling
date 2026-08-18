# Initializing the harness in a new project

Follow this once, when copying `harness/` into a project for the first time.

## 1. Copy

```
harness/                     → copy wholesale
.claude/settings.local.json  → copy, then retarget the permission allowlist
```

Delete, in the copy:
- every `## Local instantiations` section (they point at the old project)
- `UPSTREAM.md` §2's filled port record — keep the empty table and the staging plan

## 2. Create the research record

```
research/
  PROGRESS.md      # live state — start it with today's date and "project initialized"
  GOTCHAS.md       # from templates/gotchas.md
  findings/        # empty; add files per concept as they earn one
  directions/      # empty; add briefs as they are proposed
  scratch/         # empty
RESEARCH.md        # the vision file — HUMAN writes this, not you. Leave it for Sevan.
```

## 3. Write `CLAUDE.md`

Keep it short. It carries what the harness deliberately does not:

1. **The role fork, first section.** Orchestrator by default; every spawned subagent is a
   worker. This is the auto-loaded backstop against role confusion.
2. **The session ritual** — which harness files to read, in order.
3. **Action triggers** into the harness — before writing a figure, before computing a metric,
   before spawning a subagent. Phrase them as triggers on an observable action, not as topics.
4. **Project mechanics** — commands, environment, where data and outputs live.
5. **Architecture contracts** that cannot be inferred from any single file.
6. **Pointers** to the project's metric registry, run registries, and gotchas file.

Do **not** copy conventions into `CLAUDE.md` that already live in the harness. If you find
yourself explaining how to build a figure there, it belongs in `STYLE.md`.

## 4. Retarget the quarantine check

Edit the `DENY` list in `harness/check.sh`: remove the old project's vocabulary, add the new
project's domain nouns, model names, and dataset identifiers. Then run it.

## 5. Wire the hooks

In `.claude/settings.local.json`:
- `PostToolUse` on `Write|Edit` → run `harness/check.sh` when the path is under `harness/`
- `PreCompact` → remind to refresh the live-state file before context is discarded

## 6. Record the port

Fill in `UPSTREAM.md` §2 with **every edit you had to make** to a harness file to make it fit
this project. That diff is the measured evidence of which rules were secretly local, and it is
the only reliable test of the quarantine. Do not skip it — it is the main reason the file
exists.
