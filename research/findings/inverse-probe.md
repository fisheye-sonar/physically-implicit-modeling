# Inverse probe — a learned state → latent map edits BOTH environments, better than any probe-derived write (2026-09-14)

**Status:** measured 2026-09-14 on twelve runs — Othello `L-oth-20m`, `L-oth-adjacent-flip-20m`,
`L-oth-adjacent-20m`, `L-oth-noflip-20m`; discworld `L-dw-20m`, `L-dw-noiseless-20m`,
`L-dw-smooth-20m`, `L-dw-8ray-20m`, `L-dw-8ray-tok-20m`, `L-dw-5ray-20m`, `L-dw-blink-20m`,
`L-dw-8ray-obs5-20m` (`experiments/inverse_probe/`; units `inverse_probe_oth`, `inverse_probe_dw`,
`inverse_probe_mirror`, `inverse_probe_batch2`, `inverse_probe_batch3`; 4–6 minutes of compute per
run). An INSTRUMENT experiment — it changes the write, not the environment — quoted beside the
canonical editors, never in their place. The adjacent failure was then put to two tests of
Sevan's design the same evening (section "The two tests" below); the numbers are final.

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
block; discworld's regression block — ND is n/a there by the registry's rule: one fixed direction with a
swept scalar is coherent only for a categorical target, so discworld's regression ND arms are computed into
`scores.json` but never tabulated). Also g's held-out
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
6. **Why adjacent does not edit — two hypotheses, two tests (see the section below).** The
   adjacency finding's reading (`adjacency-ablation.md` reading 2) was that the decodable
   colour there is an input lookup and the legality circuit consumes a different copy. Sevan's
   objection: discworld position is ALSO a lookup and edits, so "lookup vs computed" cannot be
   the discriminator. My refinement — that what matters is whether the fact is recoverable
   from context positions OTHER than the one written — was tested with Sevan's last-tile case
   set and is NOT supported. Sevan's alternative — that adjacent's output function does not
   extend to unreachable boards while standard Othello's does — survives the reconstruction
   control. Details and numbers below; the mechanism behind the generalisation gap is open.
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

## The two tests of the adjacent failure (2026-09-14 evening — Sevan's design)

**Test 1 — the last-placed disc.** The just-placed disc is the only tile whose square AND
colour enter the network through the position the write touches (colour there is fixed by the
position embedding at offset one). If the write fails on adjacent because the layers above the
write re-read the untouched context, editing THIS tile should succeed; if adjacent's function
simply does not extend to recoloured boards, it should fail like every other tile. Case set:
the 1000 canonical histories with the last-placed disc recoloured, kept where the legal set
changes (adjacent 864, standard 792; every case adds squares, none removes). All four write
forms at every point, and the three canonical editors RE-SEARCHED on these cases over the
run's own grids (ND 9 points × 14 α, PI 9 × 12, GS 5 layer sets × 6 α).

**Test 2 — the reconstruction control.** Overwrite with g(s_pre), the conditional mean of the
UNEDITED (reachable) board, no edit. If the output survives, it depends on nothing the board
fails to determine, and any degradation under g(s_post) is off-manifold extrapolation, not a
missing 3% of the residual (the claim I had made from the R² 0.97 / guard 1.5 pair, which
Sevan flagged as over-read). Reported as the fidelity guard computed against the PRE-edit legal
set (the ordinary guard's construction with the pre-edit truth; 1.0 = output unchanged; both
models are Bayes-optimal so the scale is the same on both). The raw RMSE from the unsteered
distribution is in the JSON as `drift_rmse` and is not quoted.

| last-tile cases | unedited | inverse overwrite | inverse delta | retrieval (either) | canonical ND | canonical PI | canonical GS |
|---|---|---|---|---|---|---|---|
| `L-oth-adjacent-20m` (864) | −1.00 | +0.37 / 0.44 (pt 1); +0.14 / 0.63 (pt 2); ≤ −0.08 / ≥ 1.26 from pt 3 | +0.41 / 0.51 (pt 1); +0.22 / 0.72 (pt 2); guards 2–7 from pt 3 | inert or destructive | +0.22 / 1.09; guarded **+0.19 / 0.69** (pt 1) | nothing positive (−0.19 / 3.66) | nothing positive (−0.92 / 5.21) |
| `L-oth-20m` (792) | −0.93 | +0.78 / 0.43 (pt 4) | **+0.86 / 0.25** (pt 4) | inert | +0.72 / 0.55 (pt 5) | +0.87 / 0.42 (pt 4) | +0.83 / 0.35 (layers ≥ 0) |

Landing on adjacent last-tile: 0.94–0.98 at points 1–8 for the overwrite. Per-point tables in
`scores/othello_<run>_mirror128_lasttile.json` (`canonical_on_cases` holds every ND / PI / GS arm).

| g(s_pre) overwrite, no edit — canonical cases (1000); guard vs the PRE-edit legal set, 1.0 = output unchanged | points 0–1 | points 2–4 | points 5–6 | points 7–8 |
|---|---|---|---|---|
| `L-oth-adjacent-20m` | 1.9–2.2 | 1.2–1.4 | 2.2–3.5 | 6.0–6.7 |
| `L-oth-20m` | 9.4–9.5 | 2.0–8.0 | 1.8–6.2 | 10–16 |

(The retrieval mean of the unedited board damages adjacent from point 3 on — guard 8.9–22 —
so ten nearest boards are not near enough on Othello even to reconstruct. Identical numbers on
the last-tile histories, as they must be: the control does not depend on the edit.)

**Reading.**

- **The positional / bypass hypothesis is not supported.** With everything about the tile
  inside the written stream, adjacent shows the same profile it shows on the canonical cases:
  a modest effect at the first point or two (+0.37–0.41, now inside the guard at 0.44–0.51
  where the canonical-case guard was 0.97), then destruction from point 3 while the read-out
  lands at ≥ 0.94. Deep-point writes were the ones the hypothesis said should now work; they
  do not. Standard Othello edits the just-placed disc as well as any tile (+0.86 / 0.25, and
  all three canonical editors ≥ +0.72 inside the guard). The fact's location changes collateral
  damage a little and editability not at all.
- **Adjacent's output IS a function of the board on reachable boards — the "unexplained 3%"
  claim is withdrawn.** Replacing adjacent's residual with g(s_pre) leaves the output within
  1.2–2.2× the trained model's own distance from truth at every point through 5. Standard
  Othello is damaged by the same replacement at points 0–3 and 6–8 (guard 5–16) and survives
  only at points 4–5 (guard 1.8–2.0) — exactly the points where its edits work. So on standard the
  board suffices for the output only mid-depth, and that is where the write lands; on adjacent
  the board suffices everywhere and the write lands nowhere. Sufficiency of the board for the
  output is necessary for the state→latent write to edit, not sufficient.
- **What stands is Sevan's framing: a generalisation gap.** Both models' outputs are functions
  of the board on the boards games produce. Standard Othello's function extends to boards no
  game produces — every edit in this programme writes such a board, and it moves the output at
  +0.86–0.88. Adjacent's does not: g(s_post) is a residual the linear probe reads as the
  recoloured board and the model reads as nothing in particular. Why one model's board function
  extends and the other's does not is the open question. Candidates, not claims: the INLP
  result (`adjacent-flip-ablation.md`: 80–280 colour copies per tile on adjacent vs 24–48 on
  standard) — g's extrapolation may move the probe-visible copies and not the ones downstream
  reads; or adjacent's legality is computed through features that coincide with the board only
  on-manifold (placement parity), so a recoloured board has no valid encoding to extrapolate
  to. A legality read-out fitted on-manifold and evaluated on g(s_post) would separate these.

Scripts: `othello_inverse.py --cases last-tile` (+ `--recon-only`), `othello_lasttile_gs.py`,
`lasttile_table.py`; driver `experiments/inverse_probe/drivers/lasttile.sh`; units
`inverse_probe_lasttile`, `inverse_probe_lasttile_gs`; scores
`scores/othello_{L-oth-adjacent-20m,L-oth-20m}_mirror128_{lasttile,recon}.json`; logs
`logs/inverse_probe/mirror128_*_lasttile.log`, `*_recon.log`, and (both GS runs, the unit's
quoting dropped the run name) `mirror128__lasttile_gs.log`.

## dw-blink by subset — the write does not care whether the frame ends a blink (2026-09-14 evening, Sevan's question)

Same g (mirrored map, fitted on the blink probe corpus WITHOUT a visibility input), same four
forms, the bench restricted to two populations of the 20k edits split, canonical PI / GS
re-searched on each (PI 9 points × 16 α, GS 5 layer sets × 10 α; the subset construction is
`experiments/blink_ablation/scripts/subset_editability.py`'s):

- **reappearance** — the edited object is hidden through the last context frame (EF−1) and
  visible at the edit frame: the write lands on the frame that ENDS its blink; 642 cases
  (all available), staleness 1–12 hidden frames (112 at the 12-frame cap);
- **visible** — neither object hidden at any frame from the start of the context through the
  end of the 15-step scored rollout; 630 cases (all available).

| `L-dw-blink-20m` | n | unedited | overwrite | delta | retrieval overwrite | retrieval delta | canonical PI | canonical GS | R² (pts 1–8) |
|---|---|---|---|---|---|---|---|---|---|
| all cases (canonical bench) | 1000 | −0.92 | +0.52 / 0.44 | +0.67 / 0.38 | +0.34 / 0.68 | +0.35 / 0.75 | +0.21 / 1.54 | −0.09 / 0.97 | 0.28–0.36 |
| reappearance | 642 | −0.85 | +0.53 / 0.39 | **+0.69 / 0.39** | +0.35 / 0.69 | +0.36 / 0.73 | +0.24 / 1.57 (guarded +0.09 / 0.92) | −0.04 / 0.97 | 0.28–0.36 |
| visible | 630 | −0.93 | +0.53 / 0.38 | **+0.67 / 0.43** | +0.34 / 0.67 | +0.34 / 0.77 | +0.21 / 1.56 (guarded −0.03 / 0.95) | −0.13 / 1.01 | 0.28–0.36 |

The two populations are indistinguishable on every column, and match the whole bench. Per
point the profiles coincide too (delta positive from point 1, peaking at points 3–6 at
+0.60–0.69 on both; overwrite negative at points 0–1 and +0.46–0.53 from point 4 on both).
Whether the model reads the edited object's position off the current frame (visible) or
carries it through a blackout of up to 12 frames and is about to see it again (reappearance),
the state-conditional mean moves the output by the same amount, and the canonical editors fail
by the same amount. The caveat that g has no visibility input — g(s) averages hidden and
visible frames at a state, ~84 % visible — would have biased the reappearance row down if it
mattered; it did not show. Scores `scores/discworld_L-dw-blink-20m_mirror128_sel-{reappearance,visible}.json`;
`discworld_inverse.py --select … --canonical-on-subset`; `blink_subset_table.py`; driver
`drivers/blink_subsets.sh`; unit `inverse_probe_blink_subsets`.

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
