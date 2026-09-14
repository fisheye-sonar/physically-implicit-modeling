# inverse_probe — can a map from STATE to LATENT supply the write? (2026-09-14)

**Question (Sevan).** Our editors write through probes fitted latent → state, inverted
(PI), contrasted (ND) or descended (GS). Fit the other direction — g: state → residual at
point ℓ, on the same probe corpus — and write g(target state) into the residual. Does a
function of the target state alone edit? Sevan's bet: no, or poorly, in both environments.

**Method** (`scripts/othello_inverse.py`, Othello first; discworld held until this lands).
g is an MLP (hidden 1024, 40 epochs, seeded 80/20 split by game) from the one-hot mine/theirs
board (64 × 3) to the 512-d residual at point ℓ; one g per point. Three write forms at the
edit position, canonical 1000-case bench, canonical scorecards (symmetric-difference Edit
Index, move fidelity):
- **overwrite** h′ = g(s_post) — the conditional mean of the residual given the board;
- **delta** h′ = h + α (g(s_post) − g(s_pre)), α swept — keeps what h carries beyond the board;
- **nn** — both forms with g replaced by the mean residual of the k = 10 training rows nearest
  the board (Hamming over 64 tiles): retrieval, no training.
Controls: overwrite with the state-free mean residual; the canonical PI / ND rows. Also g's
held-out R² per point (how much of the residual the board explains) and whether the canonical
linear probe reads the written residual as s_post ("landed").

Contained: reads the run's cached probes and probe corpus; writes `scores/othello_<run>.json`;
nothing in `pim/` changes. Unit `inverse_probe_oth`, log `logs/inverse_probe/`.
