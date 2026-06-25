# findings/ — Durable Findings Ledger (index)

Established results, organized **one file per concept**, with **dated,
append-only** entries inside each. This is the "what's true" record.

## Rules

- **Gated:** nothing enters `findings/` by default. A result is promoted here from
  `scratch/` only after passing the **"artifact or signal?"** human check (see
  `../README.md`). Drafting is an agent power; promotion is a human power.
- **Append-only:** never rewrite or delete a dated entry. A correction is a *new*
  dated entry that supersedes the old one. The trail of how understanding changed
  is itself a finding (e.g. the "decode≠generate" reversal in `editability.md`).
- Each file opens with a short **Current understanding** summary (the one mutable
  part — the synthesis) followed by the immutable dated log.

## Concepts

- [`state-geometry.md`](state-geometry.md) — dimensionality, manifold structure,
  curvature of the GRU visited-state set (sub-question 1).
- [`editability.md`](editability.md) — causal manipulability of hidden states:
  which edits succeed, which the dynamics reject, and why (sub-question 3).

_(identifiability / probe-recovery findings — sub-question 2 — not yet a separate
concept file; add when there's a promoted result.)_
