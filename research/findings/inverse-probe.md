# Inverse probe — a learned state → latent map edits Othello better than any probe-derived write (2026-09-14)

**Status:** Othello measured 2026-09-14 (`experiments/inverse_probe/`, unit `inverse_probe_oth`,
4 minutes of compute on `L-oth-20m`). Discworld held for Sevan's call. An INSTRUMENT
experiment (it changes the write, not the environment): quoted beside the canonical editors,
never in their place (`RESEARCH.md`: the independent variable is the environment).

## The question and the bets

Every canonical editor writes through a probe fitted latent → state, then inverted (PI),
contrasted (ND) or descended (GS). Fit the OTHER direction — g: state → residual at point ℓ,
on the same 20k-game probe corpus — and write g(target state). Sevan's bet: it does not edit,
or edits poorly. Mine: the delta form lands near ND's level, the overwrite costs fidelity.

## Method

g is an MLP (192 → 1024 → 512; one-hot mine/theirs board in; the residual at point ℓ out),
fitted on 943k training positions with the probes' own recipe (40 epochs, seeded 80/20 split
by game), one g per residual point. Three write forms at the edit position, on the canonical
1000-case bench with the canonical scorecards (symmetric-difference Edit Index, move fidelity):

- **overwrite** h′ = g(s_post) — the conditional mean of the residual given the board;
- **delta** h′ = h + α · (g(s_post) − g(s_pre)) — keeps what h carries beyond the board;
- **nearest neighbour** — both forms with g replaced by the mean residual of the 10 training
  boards nearest s (Hamming over 64 tiles): retrieval, no training.

Controls: overwrite with the state-free mean residual; the canonical PI / ND / GS rows. Also
g's held-out R² (how much of the residual the board explains) and "landed" (the canonical
linear probe reads the written residual as the target board at the flipped tile).

## Result

Unedited −0.933. Canonical on this run: PI +0.82 / 0.30, ND +0.72 / 0.31, GS +0.83 / 0.28.

| point | g R² (held out) | overwrite | mean-h control | nn overwrite | **delta, best α** | nn delta, best α |
|---|---|---|---|---|---|---|
| 0 | 0.43 | −0.28 / 1.96 | +0.13 / 2.92 | −0.24 / 2.18 | −0.58 / 1.24 (α 3) | −0.69 / 1.31 |
| 1 | 0.74 | −0.31 / 1.77 | −0.20 / 1.83 | −0.29 / 1.94 | −0.67 / 1.10 | −0.66 / 1.30 |
| 2 | 0.81 | −0.12 / 1.69 | −0.13 / 2.11 | −0.23 / 1.97 | −0.29 / 0.96 | −0.53 / 1.30 |
| 3 | 0.83 | +0.07 / 1.22 | −0.01 / 2.50 | −0.19 / 1.88 | +0.47 / 0.61 (α 3) | −0.33 / 1.22 |
| 4 | 0.85 | +0.57 / 0.56 | −0.04 / 3.40 | −0.09 / 1.81 | +0.86 / 0.38 (α 3) | −0.02 / 1.20 |
| **5** | 0.87 | **+0.90 / 0.22** | −0.05 / 3.74 | +0.02 / 2.11 | **+0.92 / 0.20 (α 1.5)** | +0.15 / 1.34 |
| 6 | 0.86 | +0.87 / 0.30 | −0.05 / 2.59 | +0.04 / 2.54 | +0.88 / 0.32 (α 2) | +0.03 / 1.42 |
| 7 | 0.84 | +0.52 / 1.29 | −0.05 / 2.17 | −0.01 / 4.30 | +0.60 / 1.21 | −0.07 / 2.76 |
| 8 | 0.92 | +0.25 / 3.30 | −0.05 / 2.17 | −0.00 / 4.15 | +0.32 / 5.61 | −0.07 / 5.43 |

The delta write at point 5, α 1.5: **+0.917 at fidelity 0.20**, above every canonical editor
on this run (best canonical +0.83 / 0.28); its α curve is a clean rise (α 1: +0.73, 1.5: +0.92,
2: +0.92, 3: +0.86) with the write 26% of the activation norm at the peak. The plain overwrite
at points 5–6 reaches +0.90 / 0.22 and +0.87 / 0.30 — replacing the whole residual with the
board's conditional mean edits nearly as well as the delta, at points where the board explains
86–87% of the residual's variance. Retrieval (10 nearest boards) never exceeds +0.15 and is
destructive everywhere; the mean-residual control is at the unedited floor or worse.

## Reading

1. **Sevan's bet loses on Othello, by a wide margin.** A map from the target board ALONE,
   learned from the probe corpus, is the best editor we have on this model: +0.92 / 0.20 against
   PI +0.82 / 0.30 and GS +0.83 / 0.28. The counterfactual history is not needed to find the
   write; the conditional mean of the residual given the board is enough.
2. **Where it works is where the probes work.** Points 4–6 — the same points at which PI and
   ND edit — are where g explains ≥ 85% of the residual and where the write lands in the
   output; at points 0–2 the read-out lands (82–99%) and the output does not follow, the same
   dissociation the probe-derived writes show early in the stack. Point 8 explains the most
   variance (0.92) and edits worst: the last residual is nearly the logits, and overwriting it
   with a mean destroys the move distribution (fidelity 3.3).
3. **Learning beats retrieval decisively.** Boards after 20 moves are essentially unique; the
   10 nearest training boards differ in many tiles and their mean residual is not the target's
   — the MLP interpolates where retrieval cannot. The state-free mean is inert, so the effect is
   the board's, not the mean's.
4. **What it says about the negative on discworld.** If the same construction edits discworld's
   8-ray model under the factorised target, then the discworld failure is a property of the
   probe-derived WRITE (a pseudo-inverse or a row contrast of a linear read-out), not of the
   representation; if it does not, the residual there carries too much that the target state
   does not determine (velocity, history), and the conditional mean cannot supply it — which is
   the alignment result (`edit-direction-alignment.md` Result 2) from the other side. That is the
   next run, on Sevan's go.

Provenance: `experiments/inverse_probe/scores/othello_L-oth-20m.json` (every arm),
`logs/inverse_probe/othello_L-oth-20m.log`; script `experiments/inverse_probe/scripts/othello_inverse.py`.
Caveats: one seed of g; the bench is the canonical one (fixed 20-move prefix); the α grid is
(0.25 … 3); hidden 1024 / 40 epochs unswept. The write is per case a function of (h, s_pre,
s_post) — an editor with more information than PI/ND (which see only the tile flip), which is
part of the point and part of the caveat.
