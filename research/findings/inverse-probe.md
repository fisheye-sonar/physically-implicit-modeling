# Inverse probe — a learned state → latent map edits BOTH environments, better than any probe-derived write (2026-09-14)

**Status:** measured 2026-09-14 on twelve runs — Othello `L-oth-20m`, `L-oth-adjacent-flip-20m`,
`L-oth-adjacent-20m`, `L-oth-noflip-20m`; discworld `L-dw-20m`, `L-dw-noiseless-20m`,
`L-dw-smooth-20m`, `L-dw-8ray-20m`, `L-dw-8ray-tok-20m`, `L-dw-5ray-20m`, `L-dw-blink-20m`,
`L-dw-8ray-obs5-20m` (`experiments/inverse_probe/`; units `inverse_probe_oth`, `inverse_probe_dw`,
`inverse_probe_mirror`, `inverse_probe_batch2`, `inverse_probe_batch3`; 4–6 minutes of compute per
run). An INSTRUMENT experiment — it changes the write, not the environment — quoted beside the
canonical editors, never in their place. The interpretation of the Othello failures (readings 5–6)
is UNDER DISCUSSION with Sevan as of the evening of 2026-09-14; the numbers are final.

## The question and the bets

Every canonical editor writes through a probe fitted latent → state, then inverted (PI),
contrasted (ND) or descended (GS). Fit the OTHER direction — g: state → residual at point ℓ,
on the same probe corpus — and write g(target state) into the residual. Sevan's bet: it does not
edit, or edits poorly, in both environments. Mine: the delta form lands near ND on Othello and
beats PI but stays well short of Othello on discworld. Outcome: Sevan's bet loses in both
environments (standard Othello +0.88, every discworld instance +0.61 to +0.87); mine loses on
discworld (it reaches Othello's level on 5-ray and 8-ray). A later prediction of mine, that
`L-oth-noflip-20m` would fail, came true but is uninformative (reading 7).

## Method

g is the MIRROR of the canonical MLP probe (Sevan): one hidden layer of 128, the probes' own
200-epoch recipe, the same seeded 80/20 split by sequence, one g per residual point. Inputs:
Othello — the one-hot mine/theirs board (64 × 3 = 192); discworld — the FULL state, position
and velocity of both discs (8), in the run's canonical regression basis (frustum on every
instance but `L-dw-8ray-obs5-20m`, whose canonical block is cartesian). Outputs: the 512-d
residual at point ℓ. Four write forms at the edit position, scored on each run's canonical bench
with the canonical scorecards (Othello: symmetric-difference Edit Index and move fidelity;
discworld frame models: ray-zone Edit Index and the fidelity guard, the pre-dynamics target state
written for all dims; the token model †: the frame-set Edit Index the canonical token bench uses):

- **overwrite**          h′ = g(s_post) — the residual replaced by its conditional mean given the target state;
- **delta**              h′ = h + α (g(s_post) − g(s_pre)), α ∈ {0.25 … 3} — keeps what h carries beyond the state;
- **retrieval overwrite** h′ = m(s_post), m(s) = the mean residual of the k = 10 training frames whose state is nearest s (Hamming on Othello, Euclidean in standardised state units on discworld) — a lookup, no training;
- **retrieval delta**    h′ = h + α (m(s_post) − m(s_pre)).

Controls: overwrite with the state-free mean residual (inert or destructive everywhere:
−0.04 … −0.59); the canonical PI / ND / GS rows from each run's `scores.json` (Othello symdiff
block; discworld's regression block — ND has no regression form, hence n/a). Also g's held-out
R² per point (how much of the residual the state explains; a RANGE over points 1–8 because one g
is fitted per point) and, on Othello, whether the canonical linear probe reads the written
residual as s_post at the edited tile ("landed"). A wider map (1024 hidden, 40 epochs) was run
first on Othello and noiseless and gave the same picture a few hundredths higher; its files are
kept (`*_h1024e40.json`) and not quoted. The printer for the table is
`experiments/inverse_probe/scripts/table.py`.

## The table — best arm over residual points, Edit Index / fidelity guard (guard > 1 = degraded)

| run (bench) | unedited | **overwrite** | **delta** (best α) | **retrieval overwrite** | **retrieval delta** (best α) | canonical PI | canonical ND | canonical GS | state explains h (R², pts 1–8) |
|---|---|---|---|---|---|---|---|---|---|
| `L-oth-20m` (1000 flips) | −0.93 | +0.81 / 0.38 (pt 5) | **+0.88 / 0.23** (pt 4, α 3) | +0.04 / 2.54 | +0.15 / 1.34 | +0.82 / 0.30 | +0.72 / 0.31 | +0.83 / 0.28 | 0.68–0.88 |
| `L-oth-adjacent-flip-20m` (1000 flips) | −0.96 | +0.66 / 0.51 | **+0.81 / 0.35** | −0.36 / 4.07 | −0.31 / 2.46 | +0.35 / 0.83 | +0.32 / 0.61 | +0.14 / 1.75 | 0.66–0.97 |
| `L-oth-adjacent-20m` (1000 flips) | −0.96 | +0.00 / 1.48 (pt 3); guarded −0.03 / 0.74 | +0.50 / 1.12 (pt 1, α 2); guarded **+0.41 / 0.97** | −0.15 / 4.40 | −0.03 / 1.70 | +0.17 / 3.69 | +0.49 / 1.98 | −0.16 / 6.68 | 0.85–0.98 |
| `L-oth-noflip-20m` (1000 flips; control, see reading 7) | −0.98 | −0.55 / 2.58 | −0.48 / 4.17 | −0.63 / 2.49 | −0.68 / 2.58 | −0.31 / 3.71 | +0.22 / 2.54 | −0.52 / 4.25 | 0.86–0.95 |
| `L-dw-20m` (1000 teleports, frustum) | −0.70 | +0.55 / 0.42 | **+0.65 / 0.48** | +0.39 / 0.68 | +0.42 / 0.75 | +0.18 / 1.94 | n/a | −0.16 / 0.94 | 0.61–0.68 |
| `L-dw-noiseless-20m` (frustum) | −0.93 | +0.60 / 0.34 (pt 6) | **+0.66 / 0.46** (pt 5, α 3) | +0.40 / 0.65 | +0.38 / 0.76 | +0.23 / 1.54 | n/a | −0.08 / 1.07 | 0.29–0.35 |
| `L-dw-smooth-20m` (frustum) | −0.97 | **+0.67 / 0.26** | +0.62 / 0.34 | +0.30 / 0.80 | +0.24 / 0.90 | +0.11 / 2.62 | n/a | −0.02 / 1.16 | 0.60–0.74 |
| `L-dw-8ray-20m` (frustum) | −0.90 | +0.72 / 0.27 (pt 5) | **+0.83 / 0.28** (pts 1–2, α 3) | +0.63 / 0.43 | +0.68 / 0.50 | +0.26 / 0.99 | n/a | −0.03 / 0.90 | 0.37–0.57 |
| `L-dw-8ray-tok-20m` † (frame-set EI) | −0.75 | +0.65 / 0.29 | **+0.66 / 0.29** | +0.36 / 0.57 | +0.35 / 0.56 | +0.00 / 0.75 | n/a | −0.15 / 0.80 | 0.24–0.61 |
| `L-dw-5ray-20m` (frustum) | −0.89 | +0.83 / 0.20 (pt 4) | **+0.87 / 0.24** (pts 1, 3, α 2–3) | +0.69 / 0.39 | +0.75 / 0.44 | +0.24 / 0.94 | n/a | −0.06 / 0.86 | 0.46–0.63 |
| `L-dw-blink-20m` (frustum) | −0.92 | +0.52 / 0.44 | **+0.67 / 0.38** | +0.34 / 0.68 | +0.35 / 0.75 | +0.21 / 1.54 | n/a | −0.09 / 0.97 | 0.28–0.36 |
| `L-dw-8ray-obs5-20m` (cartesian) | −0.89 | +0.60 / 0.34 (pt 6) | **+0.61 / 0.40** (pt 5) | +0.45 / 0.59 | +0.40 / 0.68 | +0.19 / 1.60 | n/a | −0.07 / 1.10 | 0.43–0.49 |

Guards: every inverse-map arm on the eight discworld rows and on `L-oth-20m` /
`L-oth-adjacent-flip-20m` passes (fidelity < 1); on `L-oth-adjacent-20m` only the delta's
guarded arm does (+0.41 / 0.97 at point 1; every deeper point is destructive, guards 2–8); on
`L-oth-noflip-20m` nothing passes; Othello's retrieval forms never edit. Per point
(`scores/*_mirror128.json`): on standard Othello the effect is confined to points 3–6 and peaks at
4–5, where PI and ND edit; on noiseless it is positive from point 1 and flat from 3; on 8-ray
and 5-ray it is positive at EVERY point including the input embedding (5-ray overwrite +0.82 /
0.23 at point 0) and peaks for the delta at points 1–3; on obs5 point 0 is negative and the
effect grows to a plateau at points 4–7. Landing on Othello: standard and adjacent-flip ≥ 0.9
where they edit; adjacent 0.93–0.97 at points 1–8 while the output does not move; no-flip
0.00–0.03 at every point.

## Reading

1. **Sevan's bet loses in both environments, and the discworld half is the finding.** The
   regression rows of every discworld instance have never edited through a probe — PI only over
   the guard or at +0.26 at best, GS negative, on every instance and seed we have. A map from the
   full state alone, fitted on the same probe sequences with the probe's own architecture
   mirrored, edits all eight at +0.61 to +0.87 inside the guard, 5-ray and 8-ray at Othello's
   level. The residual was writable all along; the probe-derived write was the wrong write. This
   is the alignment result (`edit-direction-alignment.md` Result 2: the true Δ edits at +0.9,
   the position rows carry none of it) closed from the other side — the conditional mean
   E[h | state] carries the part of Δ the probe rows miss.
2. **It edits where the state explains little of the residual.** On noiseless g's held-out R²
   is 0.29–0.35, on blink 0.28–0.36, against 0.68–0.98 on Othello — most of discworld's
   residual is NOT a function of position and velocity — yet writing the conditional mean, or
   its difference, moves the output. The overwrite discards the unexplained 60–70% and still
   edits at +0.52 to +0.83 with the guard well under 1; the delta keeps it and edits a little
   better with a slightly worse guard. Editability does not require explaining the residual; it
   requires writing along the direction the state moves it.
3. **Retrieval works where states repeat and fails where they do not.** On 5-ray and 8-ray the
   ten nearest training frames are genuinely near and their mean residual edits at +0.69 and
   +0.63 — a training-free editor better than every canonical one on those runs; on every
   discworld instance it edits (+0.30 to +0.69); on Othello, where a 20-move board is
   essentially unique, the ten nearest boards differ in many tiles and retrieval is inert or
   destructive on all four instances. The learned map interpolates where the lookup cannot,
   and is never worse.
4. **The Othello result reproduces at probe width.** Delta +0.88 / 0.23 at point 4 — above
   PI (+0.82), ND (+0.72) and GS (+0.83) — with the same 128-unit hidden layer the canonical
   MLP probe uses. Width mattered only at the deeper points (6–8), where the 128-map explains
   less of the residual and its edits fall with it.
5. **The environment toggle survives the instrument change, and is sharper under it.** With the
   write held fixed, Othello's flipping toggle orders the runs: standard +0.88, adjacent-flip
   +0.81, adjacent +0.50 at guard 1.12 (guarded +0.41 / 0.97, point 1 only, destruction below).
   The canonical editors give the same order at a third of the range (+0.83 / +0.35 / ND +0.49
   at guard 1.98). Adjacent → adjacent-flip is the cleanest pair: the same placement rule, the
   same probe corpus size, colour used by legality in both, and switching recolouring ON takes
   the inverse map from not editing to +0.81. On discworld the toggles order the inverse map
   too — fewer rays edit better (5-ray +0.87 > 8-ray +0.83 > the rest +0.61 to +0.67), the
   token model matches its frame twin (+0.66 vs +0.83 is the frame-set EI's smaller range),
   obs5 is lowest (+0.61, cartesian) — but the range is narrow and every instance edits, so
   discworld's toggles are toggles of degree while Othello's flipping toggle is one of kind.
6. **Why adjacent does not edit is NOT settled; "lookup" is not the discriminator.** The
   adjacency finding's reading (`adjacency-ablation.md` reading 2) was that the decodable
   colour there is an input lookup — the placing move's parity — and the legality circuit
   consumes a different copy. The inverse map lands on adjacent at 0.93–0.97 with the output
   unmoved, which fits. But Sevan's objection stands: discworld position is ALSO a lookup (the
   observation floor equals the trained probe on every regression row; smooth's frame fixes
   position exactly) and the inverse map edits every discworld instance. So "lookup vs
   computed" cannot be what separates adjacent from the rest. The refinement proposed
   (unrecorded until this note, UNTESTED): the operative question is whether the fact the
   output needs is available from context positions OTHER than the one written. On discworld
   the current frame enters at the last position only, so any layer wanting the current
   position must read the stream we write into; on adjacent (and no-flip) a tile's colour is
   recoverable by attending to the token that placed it, so a write at the last position
   changes the copy the probe reads and leaves the copy attention can re-derive; with flipping
   on, colour depends on the whole later history and must be maintained in the residual, which
   puts the write on the path. The direct test — apply the inverse write at EVERY context
   position, each with its own edited board, on adjacent — has not been run (Sevan asked to
   hold it until the hypothesis is agreed). The account does not predict adjacent's partial
   +0.41 / 0.97 at point 1, which it can accommodate but not explain.
7. **`L-oth-noflip-20m` is a control with a known answer, not a test.** By the checkerboard
   theorem (`flip-ablation.md` §4, REGISTRY) colour on that instance equals square parity in
   every position and never enters legality, so (a) the edited board — one tile off parity —
   is outside the training distribution of g and of the model, (b) the canonical linear probe
   reads colour off occupancy and therefore reports the UNEDITED colour: landing is 0.00–0.03
   at every point, unlike adjacent's 0.95, and (c) there is nothing the output could move
   toward. Every arm is negative with guards 1–4.5, as are PI / ND / GS. My prediction that
   no-flip would fail was right for this reason and says nothing about reading 6; I had wrongly
   filed it as "a lookup like adjacent's".
8. **What this does to the project's negative.** "Discworld position is decodable but not
   editable" was true of the probe-derived writes only. The state-conditional mean is an
   editor built from the same data, architecture and held-out discipline as the probe, and it
   edits discworld's regression state on all eight instances. `adjacency-ablation.md` reading 3
   ("every discworld instance … not the copy the renderer reads") is contradicted on its
   discworld clause and carries a dated addendum. The open question moves: not whether the
   state is writable, but why a linear read-out's pseudo-inverse and a nonlinear read-out's
   gradient both miss the direction E[h | state] finds — the write directions those editors
   produce live in the probe's row space, which the alignment work measured to hold 1–3% of
   the true displacement on discworld — and, separately, what makes adjacent's residual copy
   of the board inert (reading 6).

Caveats: one seed of g and one α grid (0.25 … 3; several discworld deltas peak at the top of it,
so their best may be slightly under-read); k = 10 unswept; the inverse write sees the whole pre-
and post-edit STATE per case, more than PI / ND see (the tile flip / the teleport dims), which is
part of the construction and part of the caveat; the R² column is a range over the eight points
(one g per point), not one number; no waterfall yet for the discworld writes
(`pim.figures.waterfall_grid` on the 5-ray or 8-ray delta arm is the natural next figure).

Provenance: `experiments/inverse_probe/scores/{othello_<run>,discworld_<run>}_mirror128.json` for the
twelve runs above (every arm, every point, α and k recorded), `logs/inverse_probe/`; scripts
`experiments/inverse_probe/scripts/{othello_inverse,discworld_inverse,discworld_tokens_inverse,table}.py`.
The first Othello launch failed on a 4 GB broadcast in the retrieval search (its OOM path trips
the 2026-09-11 NVML mismatch); fixed as a one-hot matmul, log kept as `*.failed-nvml-oom.log`.
