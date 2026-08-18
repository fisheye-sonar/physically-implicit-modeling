# findings/ — what we believe, and how strongly

Established and provisional results, organized **one file per concept**, with **dated,
append-only** entries inside each. This is the "what's true" record, and it is expected to be
**current** — if the newest entry lags the work by weeks, the record is broken.

Full model: `../../harness/WORKFLOW.md`.

## Rules

- **Agent-written, continuously.** Write findings as evidence arrives. There is no approval
  queue. Write the scratch note **and** update the relevant findings file in the same session.
- **Every entry carries a status:**

  | status | meaning |
  |---|---|
  | `observed` | seen once, in one configuration. Real enough to record, not to build on. |
  | `replicated` | holds across seeds, configurations, or an independent path to the same conclusion. |
  | `established` | load-bearing. Survived deliberate attempts to break it; other work depends on it. |

  Be **reluctant with `established`** and state the case for it in the entry, so it is
  auditable in one read. Be free and prompt with the other two.
- **Every entry carries its evidence:** notebook or script, scratch note, run codes, dataset
  and split, n. A finding whose evidence cannot be located is not a finding.
- **`★` marks significance** and is placed by **Sevan**. Orthogonal to status — a result can
  be `established` and unimportant, or `observed` and pivotal.
- **Append-only.** Never rewrite or delete a dated entry. A correction is a *new* dated entry
  marked `supersedes YYYY-MM-DD`; a withdrawal is marked `retracts YYYY-MM-DD`, saying what was
  wrong and how it was caught. The trail of how understanding changed is itself a finding —
  e.g. the "decode ≠ generate" reversal in `editability.md`.
- Each file opens with a short **Current understanding** summary — the one mutable part, the
  live synthesis — followed by the immutable dated log. If the header and the log contradict
  each other, one of them is wrong; fix it in that edit.
- **State scope and caveats in the same breath as the claim.** Do not soften "is this signal
  or an artifact?" to make a result land.

## Concepts

- [`state-geometry.md`](state-geometry.md) — dimensionality, manifold structure, curvature of
  the visited-state set (sub-question 1).
- [`editability.md`](editability.md) — causal manipulability of hidden states: which edits
  succeed, which the dynamics reject, and why (sub-question 3).
- [`object-individuation.md`](object-individuation.md) — do object-moving actions individuate
  objects into a grabbable *state* handle? (sub-question 3).
- [`architecture-independence.md`](architecture-independence.md) — which failures replicate
  across architectures (cross-cutting).
- [`predictive-quality.md`](predictive-quality.md) — observation fidelity (affordance 1).
- [`trained-editors.md`](trained-editors.md) — learned edit mechanisms and what they reveal
  about why probe-derived writes fail (sub-question 3).

_(A standalone identifiability / probe-recovery file — sub-question 2 — is still folded into
`editability.md` and `architecture-independence.md`; split it out when it earns its own
concept.)_
