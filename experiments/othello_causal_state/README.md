# othello_causal_state — is the decoded board the full board or the task-relevant board?

**Question** (2026-09-03): does the probe read discs the game no longer depends on as
accurately (and as confidently) as discs it does depend on?

**Definitions** (exact, from the vendored rules; `scripts/label_relevance.py`):
* **dead** — the disc lies on no ray (row/column/diagonals) from any empty square, so it
  can never influence any future legal set or flip. Sufficient, conservative.
* **irrelevant (now)** — flipping the disc leaves the effective legal set unchanged; under
  uniform-random play it has zero influence on the model's next-move target at this
  position. dead ⊂ irrelevant.
* alive squares split into **frontier** (touches an empty) and **interior**.
* controls: age, moves since the disc last changed colour, number of flips, distance to
  the nearest empty.

**Pipeline** — `drivers/causal_state.sh` (unit `oth_causal`, MemoryMax 24G, log in
`logs/othello_causal_state/`): flags for 5,000 TEST-split games (24 workers, ~1 min) →
`probe_by_relevance.py` (cached canonical LIN / MLP-128 probes at every residual point;
accuracy, entropy and p(true) per square; by-move curves, paired within-position
differences with bootstrap CIs, control strata) → `qualitative.py` (the natural
positions with the most dead squares + one engineered "fill the bottom-left first" game,
flagged as off-distribution).

**Outputs** — `scores/relevance_test5000.{npz,json}`, `scores/probe_by_relevance_<run>.json`,
`scores/per_square_pt<k>_<fam>.npz` (focus points), `scores/qualitative_examples.json`;
figures in `outputs/`. No canonical code was changed and nothing was fitted.

**Result (2026-09-03, v2)** — dead discs are decoded worse than alive interior discs of the
same position (paired Δ −0.017 at the best points, CI excludes 0; −0.065 at points 1–2)
with higher entropy, and the gap survives age/flip-count strata; discs that some legality
computation can traverse are decoded equally well whether or not a flip of their colour
changes the legal set (Δ −0.001). The represented board is the full board minus, in part,
the discs nothing can read. **Criterion note:** the v1 "irrelevant-now" flag (single-flip
test) is blind to redundancy and conflates "not read" with "read but inert"; the
colour-blind "on no gap-free run from any empty square" criterion turns out to coincide
exactly with "dead" (the nearest empty on any line always has a gap-free run), so there
is no intermediate class. v1 artefacts are under `scores/v1_single_flip/` and
`outputs/v1_single_flip/`. Write-up: `research/findings/othello-causal-state.md`.
