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
- [`architecture-independence.md`](architecture-independence.md) — the
  non-canonical / readable≠controllable failure replicates on a refined RSSM; the
  KL-structured latent buys no canonicity (cross-cutting; preliminary, 2 checkpoints).
- [`predictive-quality.md`](predictive-quality.md) — observation fidelity: the
  refined RSSM is a competitive predictor with a generative gap (affordance 1).

_(A standalone identifiability / probe-recovery file — sub-question 2 — is still
folded into `editability.md` (recoverability + velocity) and `architecture-independence.md`;
split out when it earns its own concept.)_
