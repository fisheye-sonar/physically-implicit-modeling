# Inverse probe — a learned state → latent map edits BOTH environments, better than any probe-derived write (2026-09-14)

**Status:** measured 2026-09-14 on `L-oth-20m`, `L-oth-adjacent-20m`, `L-dw-noiseless-20m`, `L-dw-8ray-20m`
(`experiments/inverse_probe/`; units `inverse_probe_oth`, `inverse_probe_dw`, `inverse_probe_mirror`;
minutes of compute per run). An INSTRUMENT experiment — it changes the write, not the
environment — quoted beside the canonical editors, never in their place.

## The question and the bets

Every canonical editor writes through a probe fitted latent → state, then inverted (PI),
contrasted (ND) or descended (GS). Fit the OTHER direction — g: state → residual at point ℓ,
on the same probe corpus — and write g(target state) into the residual. Sevan's bet: it does not
edit, or edits poorly, in both environments. Mine: the delta form lands near ND on Othello and
beats PI but stays well short of Othello on discworld.

## Method

g is the MIRROR of the canonical MLP probe (Sevan): one hidden layer of 128, the probes' own
200-epoch recipe, the same seeded 80/20 split by sequence, one g per residual point. Inputs:
Othello — the one-hot mine/theirs board (64 × 3 = 192); discworld — the FULL frustum state,
position and velocity of both discs (8), the canonical regression target. Outputs: the
512-d residual at point ℓ. Four write forms at the edit position, scored on each run's
canonical bench with the canonical scorecards (Othello: symmetric-difference Edit Index and
move fidelity; discworld: ray-zone Edit Index and the fidelity guard, the pre-dynamics target
state written for all dims):

- **overwrite**          h′ = g(s_post) — the residual replaced by its conditional mean given the target state;
- **delta**              h′ = h + α (g(s_post) − g(s_pre)), α ∈ {0.25 … 3} — keeps what h carries beyond the state;
- **retrieval overwrite** h′ = m(s_post), m(s) = the mean residual of the k = 10 training frames whose state is nearest s (Hamming on Othello, Euclidean in standardised frustum units on discworld) — a lookup, no training;
- **retrieval delta**    h′ = h + α (m(s_post) − m(s_pre)).

Controls: overwrite with the state-free mean residual (inert or destructive everywhere:
−0.04 … −0.59); the canonical PI / ND / GS rows from each run's `scores.json`. Also g's held-out
R² per point (how much of the residual the state explains). A wider map (1024 hidden, 40
epochs) was run first on Othello and noiseless and gave the same picture a few hundredths
higher; its files are kept (`*_h1024e40.json`) and not quoted.

## The table — best arm over residual points, Edit Index / fidelity guard

| run (bench) | unedited | **overwrite** | **delta** (best α) | **retrieval overwrite** | **retrieval delta** (best α) | canonical PI | canonical ND | canonical GS | state explains h (R², pts 1–8) |
|---|---|---|---|---|---|---|---|---|---|
| `L-oth-20m` (1000 flips) | −0.93 | +0.81 / 0.38 (pt 5) | **+0.88 / 0.23** (pt 4, α 3) | +0.04 / 2.54 | +0.15 / 1.34 | +0.82 / 0.30 | +0.72 / 0.31 | +0.83 / 0.28 | 0.68–0.88 |
| `L-oth-adjacent-20m` (1000 flips) | −0.96 | +0.00 / 1.48 (pt 3); guarded −0.03 / 0.74 | +0.50 / 1.12 (pt 1, α 2); **guarded +0.41 / 0.97** | −0.15 / 4.40 | −0.03 / 1.70 | +0.18 / 3.69 | +0.49 / 1.98 | −0.16 / 6.68 | 0.85–0.98 |
| `L-dw-noiseless-20m` (1000 teleports, frustum) | −0.93 | +0.60 / 0.34 (pt 6) | **+0.66 / 0.46** (pt 5, α 3) | +0.40 / 0.65 (pt 6) | +0.38 / 0.76 (pt 5) | +0.23 / 1.54 | n/a | −0.08 / 1.07 | 0.29–0.35 |
| `L-dw-8ray-20m` (1000 teleports, frustum) | −0.90 | +0.73 / 0.27 (pt 5) | **+0.84 / 0.28** (pt 1–2, α 3) | +0.63 / 0.43 (pt 7) | +0.68 / 0.50 (pt 2) | +0.26 / 0.99 | n/a | −0.03 / 0.90 | 0.37–0.57 |

Every inverse-map arm above passes the guard (fidelity < 1) except on `L-oth-adjacent-20m`
(where only the delta's guarded arm, +0.41 / 0.97 at point 1, does — every deeper point is
destructive, guards 2–8) and Othello's retrieval forms, which do not edit at all. Per point (`scores/*_mirror128.json`): on Othello the effect is
confined to points 3–6 and peaks at 4–5, exactly where PI and ND edit; on noiseless it is
positive at every point from 1 on and flat from point 3; on 8-ray it is positive at EVERY
point including the input embedding (overwrite +0.70 / 0.29 at point 0), peaks for the delta
at points 1–3 and for the overwrite at 5.

## Reading

1. **Sevan's bet loses in both environments, and the discworld half is the finding.** The
   regression rows of `L-dw-noiseless-20m` and `L-dw-8ray-20m` have never edited through a
   probe — PI only over the guard at the top of its α grid, GS negative, on every instance and
   seed we have. A map from the full state alone, fitted on the same 30k probe sequences with
   the probe's own architecture mirrored, edits them at +0.66 / 0.46 and +0.84 / 0.28, the
   latter at Othello's level. The residual was writable all along; the probe-derived write was
   the wrong write. This is the alignment result (`edit-direction-alignment.md` Result 2: the
   true Δ edits at +0.9, the position rows carry none of it) closed from the other side — the
   conditional mean E[h | state] carries the part of Δ the probe rows miss.
2. **It edits where the state explains little of the residual.** On noiseless g's held-out R²
   is 0.29–0.35 and on 8-ray 0.37–0.57, against 0.68–0.88 on Othello — most of discworld's
   residual is NOT a function of position and velocity (history, the other object's past,
   whatever the model keeps) — yet writing the conditional mean, or its difference, moves the
   output. The overwrite discards the unexplained 60–70% and still edits at +0.60 / +0.73 with
   the guard well under 1; the delta keeps it and edits a little better with a slightly worse
   guard. Editability does not require explaining the residual; it requires writing along the
   direction the state moves it.
3. **Retrieval works where states repeat and fails where they do not.** On 8-ray the ten
   nearest training frames are genuinely near (a coarse 5-ray-to-8-ray world visits similar
   states often) and their mean residual edits at +0.63 / 0.43 — a training-free editor better
   than every canonical one on this run; on noiseless it edits at +0.40; on Othello, where a
   20-move board is essentially unique, the ten nearest boards differ in many tiles and
   retrieval is inert. The learned map interpolates where the lookup cannot, and is never worse.
4. **The Othello result reproduces at probe width.** Delta +0.88 / 0.23 at point 4 — above
   PI (+0.82), ND (+0.72) and GS (+0.83) — with the same 128-unit hidden layer the canonical
   MLP probe uses. Width mattered only at the deeper points (6–8), where the 128-map explains
   less of the residual and its edits fall with it.
5. **The environment toggle survives the instrument change — `L-oth-adjacent-20m` (added at
   Sevan's request).** On the adjacency model the board explains 97–98% of the residual from
   point 2 on (colour there is an input lookup: the placing move's parity), so g is nearly an
   identity on the board code — and writing it does NOT edit: overwrite +0.00 at guard 1.5,
   the delta's only guard-passing arm +0.41 / 0.97 at point 1 (the same point and level as the
   canonical ND's +0.49 / 1.98 without the destruction), every deeper point destructive
   (guards 2–8), retrieval inert. Standard Othello, where the board explains LESS of the
   residual (0.8), edits at +0.88. So the inverse map is ordered by the same environment
   property that orders the canonical editors — flipping on / off — and the ordering is
   sharper, not weaker, under the better write: the conditional mean of the residual given
   the board is the board's code on both models, and the model that consumes that code to
   compute legality moves with it while the model that reads colour off the input token does
   not. This is the cleanest evidence so far that what the environment does to editability is
   a property of how the model USES the state, not of how any editor writes it.
6. **What this does to the project's negative.** "Discworld position is decodable but not
   editable" was true of the probe-derived writes. The state-conditional mean is an editor
   built from the same data, the same architecture and the same held-out discipline as the
   probe, and it edits discworld's regression state at +0.66 to +0.84. The open question
   moves: not whether the state is writable, but why a linear read-out's pseudo-inverse and a
   nonlinear read-out's gradient both miss the direction that E[h | state] finds — the write
   directions those editors produce live in the probe's row space, and the alignment work
   measured that space to hold 1–3% of the true displacement on discworld. The
   environment-side toggles (ray count, flipping) still order the canonical editors; whether
   they order the inverse map too is the next measurement (5-ray, blink, the token model,
   adjacent Othello).

Caveats: one seed of g and one α grid (0.25 … 3; the discworld delta peaks at the top of it on
several points, so its best may be slightly under-read); k = 10 unswept; the inverse write sees
the whole pre- and post-edit STATE per case, more than PI/ND see (the tile flip / the teleport
dims), which is part of the construction and part of the caveat; no waterfall yet for the
discworld writes (`pim.figures.waterfall_grid` on the delta arm is the natural next figure).

Provenance: `experiments/inverse_probe/scores/{othello_L-oth-20m,othello_L-oth-adjacent-20m,discworld_L-dw-noiseless-20m,discworld_L-dw-8ray-20m}_mirror128.json`
(every arm, every point), `logs/inverse_probe/`; scripts `experiments/inverse_probe/scripts/{othello,discworld}_inverse.py`.
The first Othello launch failed on a 4 GB broadcast in the retrieval search (its OOM path trips
the 2026-09-11 NVML mismatch); fixed as a one-hot matmul, log kept as `*.failed-nvml-oom.log`.
