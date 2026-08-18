# <Thread> — canonical METRICS & METHODS registry

**One source of truth for the names, formulas, and definitions used across this thread.**
Goal: stop each deliverable re-inventing terms.

**How to use it.** A deliverable picks the **subset** it needs and copies the exact name,
formula, units, and better-direction into its own definitions table, so it still stands
alone. It is expected that a deliverable will not use every entry, and it may go outside this
list when testing something genuinely new — in which case fold the new entry back in here.

**Before adding a metric, check it is not derivable from ones already listed.** A metric that
is an algebraic function of two existing ones adds no information, grows the zoo, and reads as
a contradiction when the reader cannot see the identity linking them.

**Consistency rules that always apply here:** *(state the project's own — units, which
reference signal errors are scored against, RMSE not MSE, etc.)*

---

## Metrics

| name | formula | units | better | notes |
|---|---|---|---|---|
| | | | ↑/↓ | |

## Methods

| name | what it actually does (mechanism) | inputs | notes |
|---|---|---|---|
| | | | |

## Implementation

Every metric above is implemented **once** and imported — never re-derived per notebook.

- module → `<path>`
- functions → `<names>`

## Retired — do not reintroduce

| name | retired | why | what replaced it |
|---|---|---|---|
| | | | |

Numbers computed on a retired definition are **not comparable** to current ones. Say so
wherever an old number is quoted.
