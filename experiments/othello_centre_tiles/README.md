# othello_centre_tiles — are the four starting squares editable? (2026-09-11)

**Question (Sevan).** The centre four tiles (d4 e4 d5 e5) are occupied from move 0 and
recoloured most often; are they as editable as the rest of the board in standard Othello?

**Method.** Li's shipped 1001 never intervene on a centre tile (0 of 1001), so 98 cases were
synthesised from held-out test games with the bench's own recipe (a real prefix, one occupied
square flipped, rejected if the legal set is unchanged or empty, prefix lengths following the
shipped 1001), the square drawn from the centre four. `L-oth-20m`'s cached canonical probes,
the canonical editors and α grids, exactly as `master_eval` runs them. Nothing refitted, nothing
canonical changed. `scripts/centre_tiles.py [n]` → `scores/centre_tiles_L-oth-20m.json`.

**Result** (unedited −0.68 vs −0.71 on the whole bench):

| editor | best on centre tiles | at the whole-bench best arm | whole-bench best |
|---|---|---|---|
| PI | +0.64 / fid 0.20 (pt 5, α 2) | +0.52 / 0.29 (pt 4, α 3) | +0.61 / 0.24 |
| ND | +0.65 / 0.20 (pt 5, α 0.35) | +0.47 / 0.32 (pt 4, α 0.35) | +0.62 / 0.23 |
| GS | +0.57 / 0.27 (pt 2, α 0.05) | +0.56 / 0.28 (pt 0, α 0.05) | +0.65 / 0.21 |

The centre tiles edit as well as the board at large — PI and ND slightly better, GS slightly
worse — with the linear editors' best point one layer deeper (5 vs 4) than on the whole bench.
Nothing special about the most-recoloured squares.
