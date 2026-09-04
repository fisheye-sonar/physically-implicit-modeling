# othello_edit_by_step — editability by move number

**Question** (2026-09-02): how does the Edit Index vary with where we are in the game?
Li's shipped benchmark only covers prefixes of length 5–30, so the full game needs
synthesised cases built with the same recipe.

**Pieces**

* `scripts/synth_cases.py` — Li-style cases at every move number 1–59 from the corpus
  TEST split (a length-t prefix of a held-out game + one occupied non-centre square
  flipped to the opposite colour; flips that leave the legal set unchanged or empty are
  rejected — the shipped 1001 have exactly these properties). 256 cases per move, seed 0,
  written in the shipped pkl's own format to `cases/`.
* `scripts/edit_by_step.py` — the master_eval Othello loop verbatim (cached canonical
  probes — a cache miss aborts, nothing is fitted; same α grids read from the run's
  scores.json; same arms; same scorecard) on a case set, kept PER CASE. Saves per case
  set the cases' legal sets, targets, unsteered probabilities and unedited index
  (`scores/<label>_cases.*`), and per editor every arm's per-case Edit Index, fidelity
  and Li error (`scores/<label>_arms_<editor>.*`) plus a per-move summary. `--editors GS`
  later adds the MLP-probe arms to the same store without recomputing anything else.
* `scripts/plot_by_step.py` — figures from the saved arrays (no GPU): Edit Index by
  move for the best arm at each move and for the pooled-best arm read per move, with the
  unedited floor and the shipped benchmark's own per-move values overlaid; fidelity and
  winning point underneath. With `--compare li` it also writes the validation table
  (synthesised vs shipped cases at moves 5–30).
* `drivers/edit_by_step.sh` — the whole pipeline as unit `oth_edit_step` (MemoryMax 16G,
  log in `logs/othello_edit_by_step/`). `EDITORS="GS" drivers/edit_by_step.sh` for the
  nonlinear-probe replication.

**Canonical changes this needed** (2026-09-02, behaviour-preserving, pinned by
`tests/test_othello_bench.py`): `bench.benchmark_from_cases` (the body of
`load_benchmark`, now callable on any case list) with `case_targets` served from the
Benchmark itself; `othello_moves.move_fidelity_ratio_per_case` (the guard before its mean).

**Result (2026-09-02)** — 14,848 cases, moves 1–58 (move 59 has no valid case: with one
empty square the next move is forced whatever the colours, so no flip changes the legal
set). Both editors land from move 3, peak at moves 4–8 (PI +0.8, fidelity 0.1) and
decline to ≈ +0.35 by moves 40–55; moves 1–2 are not editable (fidelity ≥ 1); the winning
point is 4 throughout. Validation vs the shipped 1001 at moves 5–30: −0.03 (PI) / −0.08
(ND), entirely case mix — stratified by the size of the legal-set change the two agree
within 0.02. Write-up: `research/findings/othello-edit-by-step.md`.
