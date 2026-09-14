# inverse_probe — can a map from STATE to LATENT supply the write? (2026-09-14)

**Question (Sevan).** Our editors write through probes fitted latent → state, inverted
(PI), contrasted (ND) or descended (GS). Fit the other direction — g: state → residual at
point ℓ, on the same probe corpus — and write g(target state) into the residual. Does a
function of the target state alone edit? Sevan's bet: no, or poorly, in both environments.

**Method** (`scripts/othello_inverse.py`; `scripts/discworld_inverse.py` for the frame models,
full state in the run's canonical regression basis; `scripts/discworld_tokens_inverse.py` for the
token model, frame-set Edit Index; `scripts/table.py` prints the summary table). The quoted g is
the mirror of the canonical MLP probe — hidden 128, 200 epochs, seeded 80/20 split by sequence —
from the state (Othello: one-hot mine/theirs board, 64 × 3; discworld: position + velocity of
both discs, 8) to the 512-d residual at point ℓ; one g per point. Four write forms at the edit
position, canonical 1000-case bench, canonical scorecards:
- **overwrite** h′ = g(s_post) — the conditional mean of the residual given the board;
- **delta** h′ = h + α (g(s_post) − g(s_pre)), α swept — keeps what h carries beyond the board;
- **nn** — both forms with g replaced by the mean residual of the k = 10 training rows nearest
  the board (Hamming over 64 tiles): retrieval, no training.
Controls: overwrite with the state-free mean residual; the canonical PI / ND rows. Also g's
held-out R² per point (how much of the residual the board explains) and whether the canonical
linear probe reads the written residual as s_post ("landed").

Contained: reads the run's cached probes and probe corpus; writes `scores/othello_<run>.json`;
nothing in `pim/` changes. Unit `inverse_probe_oth`, log `logs/inverse_probe/`.

**Result (2026-09-14, twelve runs, mirrored map 128/200 — the quoted variant; the 1024/40 map's files are
kept as `*_h1024e40.json`).** Best guarded arm (EI / guard): Othello +0.88 / 0.23 (canonical best +0.83),
adjacent-flip +0.81 / 0.35 (canonical +0.35), adjacent +0.41 / 0.97 at point 1 only (does not edit),
no-flip nothing (control: colour is square parity there, the write does not even land). Discworld,
all eight instances edit: 5-ray +0.87 / 0.24, 8-ray +0.83 / 0.28, blink +0.67, smooth +0.67
(overwrite), noiseless +0.66, 8-ray token model +0.66 (frame-set EI), L-dw +0.65, obs5 +0.61
(cartesian); canonical PI never above +0.26 on any of them. Retrieval (10 nearest training states)
edits on discworld (+0.30 to +0.69), never on Othello. `research/findings/inverse-probe.md`; the
reading of the adjacent failure is open (finding, reading 6).

**Sevan's two tests of the adjacent failure (2026-09-14 evening).** (1) Last-tile cases — the
just-placed disc recoloured, the only tile whose square and colour both enter through the written
position (adjacent 864 / standard 792 cases): adjacent +0.41 / 0.51 at point 1 then destruction,
canonical ND +0.19 / 0.69, PI and GS nothing; standard +0.86 / 0.25 (ND +0.72, PI +0.87, GS +0.83).
The positional/bypass hypothesis is not supported. (2) Reconstruction control — overwrite with
g(s_pre), no edit: adjacent's output preserved at every point through 5 (guard 1.2–2.2 against the pre-edit truth),
standard's damaged except at points 4–5 (reconstruction guard 1.2–2.2 vs 5–16). Adjacent's output is a function of the board on reachable
boards; the failure is off-manifold (Sevan's generalisation reading). `--cases last-tile`,
`--recon-only`, `othello_lasttile_gs.py`, `lasttile_table.py`; finding section "The two tests".

**dw-blink by subset (2026-09-14 evening, Sevan).** Reappearance (the edited object hidden through the
last context frame, visible at the edit frame; 642 cases) vs visible (no blink on either object through
the 15-step rollout; 630 cases): delta +0.69 / 0.39 vs +0.67 / 0.43, overwrite +0.53 on both, retrieval
+0.35 on both, canonical PI +0.24 / 1.57 vs +0.21 / 1.56 — indistinguishable, and equal to the whole bench.
`discworld_inverse.py --select {reappearance,visible} --canonical-on-subset`; `blink_subset_table.py`.
