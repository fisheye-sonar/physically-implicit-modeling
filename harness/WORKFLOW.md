# WORKFLOW.md — the research record

The filesystem is the memory: plain dated markdown, version-controlled, shared across
sessions, worktrees, and agents. Load detail files lazily, only when the topic is relevant.

## The five artifacts

| Artifact | Answers | Who writes | Cadence |
|---|---|---|---|
| **Vision** | "what are we trying to learn?" | **human only** — agents read, never write | rarely; its drift should be legible in `git diff` |
| **Live state** | "where is the work right now?" | agent, freely | continuously; never more than one step stale |
| **Scratch** | "what did this run show?" | agent, freely, ungated | per experiment |
| **Findings** | "what do we believe, and how strongly?" | **agent, continuously** | whenever evidence changes |
| **Directions** | "what should we do next?" | agent proposes; **human chooses what is active** | as the backlog evolves |

Two things stay human, and only two: **the vision**, and **what to work on next**. Everything
else is written by the agent and reviewed by the human as a document, not as a queue.

---

## Findings — continuously maintained, explicitly graded

Findings are **written and updated by the agent as evidence arrives.** There is no promotion
queue and no waiting for approval. A record that lags the work is worse than useless: it
describes a project that no longer exists, and forces the real synthesis into volatile files.

What the old approval gate was actually protecting was not the human's signature — it was the
separation between *seen once* and *established*. That separation is preserved by making it
**explicit on every entry** rather than by making it a bottleneck.

### Structure

One file per **concept**. Each file has:

1. **Current understanding** — a short, mutable synthesis at the top. This is the live answer
   and is rewritten as understanding changes. It is the part most people read.
2. **A dated, append-only log** below it, newest first. Entries are never rewritten or
   deleted.

### Every entry carries two independent axes

**Status — how strongly is it held?** The agent assigns this from the evidence:

| status | meaning |
|---|---|
| `observed` | seen once, in one configuration. Real enough to record, not to build on. |
| `replicated` | holds across seeds, configurations, or an independent path to the same conclusion. |
| `established` | load-bearing. Has survived deliberate attempts to break it, and other work depends on it. |

**Significance — does it matter?** Marked **`★`** by the human. Orthogonal to status: a
result can be `established` and unimportant, or `observed` and pivotal. This is the flag that
replaces the promotion gate, and it costs a glance rather than a re-derivation.

Be **reluctant with `established`.** When claiming it, state the case in the entry itself —
"established because it replicates across A and B and survived C" — so the claim is auditable
in one read. Everything else should be freely and promptly written.

### Every entry carries its evidence

Non-negotiable, because the agent now writes directly into the record: the notebook or script
that produced it, the scratch note, the run codes, the dataset and split. A finding whose
evidence cannot be located is not a finding.

### Corrections and retractions

The agent will now write things that turn out to be wrong. That is an acceptable cost of a
current record, but only with an explicit mechanism:

- A correction is a **new dated entry** marked `supersedes YYYY-MM-DD`. The old entry stays
  exactly as written.
- A withdrawal is a new dated entry marked `retracts YYYY-MM-DD`, saying what was wrong and
  how it was caught.
- Update **Current understanding** in the same edit. An entry log that contradicts its own
  header is the failure this structure exists to prevent.

The trail of how understanding changed is itself valuable — it is often the most informative
thing in the file. Never tidy it away.

### The standard that replaces the gate

The old gate asked: *artifact or signal?* That question does not go away; it moves into the
entry. Every finding states its scope, its sample size, its caveats, and what would falsify
it. **Do not soften that question to make a result land.** If it is shaky, it is `observed`
with the shakiness written down, and that is a perfectly good thing to record.

---

## Directions — the backlog

One file per candidate experiment, each a self-contained brief a fresh session can be pointed
at: "read this and execute."

**Tagging**, to keep the backlog honest about novelty:
- `[in-frame]` — a variation or extension within the current approach.
- `[reframe]` — changes the question or a premise (rarer, higher value). If the backlog fills
  with `[in-frame]` items, that is a frame-lock signal: deliberately go find a `[reframe]`.

**Lifecycle:** `proposed` → `active` → `in progress` → `done` / `dropped`.
Only **`active` is a human call** — choosing what to work on is steering. The agent may
propose, may set `in progress` when execution starts, and may set `done` when the work is
finished and written into the record. On `done`, move the file to `directions/done/`.

**Every brief must be cold-start runnable.** It must work from a fresh session with no live
state from another notebook. Include a **Bootstrap** section naming exactly what to load or
compute — checkpoint, data, estimators, setup, helpers — from the paths in its context
section. **Define every metric and threshold**; if the brief asks a binary question, state the
decision rule; if a magnitude is interpreted, define its units and mandate a control.

---

## Scratch — the ungated surface

Free working space. Half-formed observations, raw numbers, "huh, that's weird" notes. Write
freely; nothing here is claimed to be true. Naming: `YYYY-MM-DD-<topic>.md`.

Scratch is the **raw provenance** that findings entries cite. It is disposable in principle,
but do not delete a note a finding still points at.

Under the current model, scratch is no longer a waiting room. Write the scratch note **and**
update the relevant findings file in the same session.

---

## Live state — this is the handoff

The live-state file is the only thing the next session inherits. **Update it incrementally, as
state changes** — when a run completes, a direction's status flips, a decision is made, or
something is parked — **not** as an end-of-session chore. A session can end abruptly, so "I'll
write it at the end" loses everything when the end comes early. Treat it as a live scratchpad
that is never more than one step stale.

At a clean stopping point, do a final pass so it reflects reality: what is done, what is
awaiting a human call, what is running, what is parked and why.

It answers **"where is the work"** — not "what is true", which is the findings' job. Git
history is the backstop.

---

## Dates and provenance

Dates are absolute (`YYYY-MM-DD`), always. Today's framing may be wrong tomorrow, and the
timestamp is how a future reader calibrates. Convert relative references ("last week") to
absolute dates when writing.

---

## Local instantiations (this project — not portable)

- Vision → `../RESEARCH.md`
- Live state → `../research/PROGRESS.md`
- Findings → `../research/findings/`
- Directions → `../research/directions/`
- Scratch → `../research/scratch/`
- Project traps → `../research/GOTCHAS.md`
