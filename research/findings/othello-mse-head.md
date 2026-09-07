# Objective ablation: MSE on the one-hot next move (L-oth-20m-mse)

**Date** 2026-09-05 · **Instance** `oth-uniform` (unchanged) · **Run**
`runs/objective_ablation/L-oth-20m-mse` (Transformer-L tokens, 8 × 512, 25.3M params, 780k
steps, matched recipe — identical to `L-oth-20m` except the loss) · **Driver**
`scripts/drivers/oth_mse.sh` · **Logs** `logs/objective_ablation/L-oth-20m-mse/` · **Chain**
training 19 h 46 min (L-oth-20m: 19 h 26 min), scoring + tables 9.5 min, distribution check
5.3 min · **Experiment** `experiments/othello_mse_head/`.

## Question

Othello's editability (GS-mine +0.65, PI/ND ≈ +0.6) is measured on a model trained with
cross-entropy on the next token; discworld's non-editability on models trained with MSE on
the next frame. How much of the gap is the *objective*? Retrain the Othello model with
everything identical except the loss: `mse_next_move_onehot`, the Brier score of the 61
head outputs against the one-hot next move. Same population minimiser (the conditional
next-move distribution), different gradient geometry — and a head that emits probability
*estimates* directly instead of logits.

**How the head is read.** The checkpoint carries `output_kind="raw"`; every scorer reads
its 60 move outputs as they are — no softmax, no clipping, no renormalisation
(`othello.data.move_probs`). Edit Index, `li_error` and fidelity are distribution-free;
`legal_mass` and the gate CE assume a distribution and are quoted next to the two
diagnostics that say whether the head is one (GOTCHAS 2026-09-04).

## Numbers (canonical scoring, EVAL_VERSION 2026-09-01.4; `L-oth-20m` beside it)

| | MSE head · L-oth-20m-mse | CE head · L-oth-20m |
|---|---|---|
| training loss (val, best) | Brier 0.013662 @ 770k | CE 2.02798 @ 775k |
| head sums to / negative mass (mean per position) | 1.0001 / 0.040 | 1 / 0 by construction |
| legal mass · top-1 legal (10k test games) | 0.9888 · 0.9984 | 0.9925 · 0.9980 |
| CE vs Bayes 2.0107 | 2.0482 | 2.0286 |
| Probe Skill mine LIN / MLP-128 (best point) | 0.961 / 0.960 (pt 6) | 0.975 / 0.976 (pt 7–8) |
| unedited Edit Index | −0.817 | −0.713 |
| **PI** best arm · EI / fidelity | pt4 · α3 · **+0.68** / 0.23 | pt4 · α3 · +0.61 / 0.24 |
| **ND** best arm · EI / fidelity | pt4 · α0.1 · **+0.74** / 0.18 | pt4 · α0.35 · +0.62 / 0.23 |
| **GS** best arm · EI / fidelity | pt4 · α0.05 · **+0.73** / 0.20 | pt0 · α0.05 · +0.65 / 0.21 |
| li_error_vs_post at the best arms PI / ND / GS | 0.14 / 0.10 / 0.13 | 0.11 / 0.10 / 0.05 |
| legal mass after the edit PI / ND / GS | 0.978 / 0.992 / 0.984 | 0.990 / 0.994 / 0.996 |

Per-point Probe Skill (LIN), points 0–8: MSE 0.342 0.797 0.867 0.918 0.946 0.960 0.961
0.958 0.958 · CE 0.342 0.808 0.881 0.928 0.953 0.967 0.974 0.975 0.975.

## What it says

1. **The objective is not what makes Othello editable.** With the loss swapped and nothing
   else, all three editors land at least as well as on the cross-entropy model: PI +0.68 vs
   +0.61, ND +0.74 vs +0.62, GS +0.73 vs +0.65, at the same fidelity (0.18–0.23 in both) and
   with the same after-edit legality (≥ 0.98 legal mass). The li_error at the best arms is
   the same 0.1–0.14 (Li et al.'s null is 2.68). Whatever separates Othello from discworld,
   it is not "CE gives you a steerable representation and MSE does not".
2. **The regressed head is a distribution to four decimals.** Trained only to minimise
   squared error against one-hot targets, the 60 move outputs sum to 1.0001 on average with
   0.04 of negative mass per position; legal mass 0.989 and top-1 legality 0.998 are within
   0.004 of the CE model. So the raw reading and the "made a distribution" reading cannot
   differ much (the distribution check below quantifies it), and `legal_mass` is a fair
   number on this model.
3. **The board is slightly *less* decodable, and peaks earlier.** Skill 0.961 vs 0.975, with
   the MSE model's best point at 6 and points 7–8 flat, where the CE model keeps improving to
   the last layer. A head that regresses probabilities needs the board no less — the gap is
   in how much of it the late residual stream keeps linearly available. It did not cost
   editability: the write point is 4 in both models.
4. **Editors respond at smaller α and in a narrower depth window.** ND's peak moved from
   α 0.35 (CE) to α 0.1, and the ND ridge is sharper (pt4: −0.02 → +0.74 → +0.64 → +0.27 for
   α 0.05/0.1/0.2/0.35); PI lands at points 3–4 only (point 5: +0.25 vs the CE model's
   +0.60); GS lands at points 0–4 (+0.70/+0.70/+0.73) and fails hard at 6–8 (−0.77/−0.71,
   where the CE model gives −0.10). The raw head's α-sensitivity is a units effect (its
   outputs are probabilities, the CE model's are logits), the depth window is not — it says
   the MSE model's steerable board representation lives in the middle of the stack.
5. **The unedited baseline is more negative** (−0.82 vs −0.71): the MSE model's unedited
   next-move estimate is more sharply that of the *pre*-edit board, i.e. it is not hedging
   between the two boards. Its GS best α (0.05) sits at the low edge of the GS grid, exactly
   as in the CE model — the GS grid should extend below 0.05 for both, an already-known
   sweep-edge case, not a new one.

## Raw vs clip-and-renormalise (experiments/othello_mse_head)

10,000 held-out test games, 589,564 positions (`scores/summary.md`, 5.3 min):

| raw output over the 60 move outputs | mean | p05 | p50 | p95 |
|---|---|---|---|---|
| sum | 1.0001 | 0.9983 | 0.9997 | 1.0032 |
| negative mass | 0.040 | 0.026 | 0.038 | 0.064 |
| min | −0.0046 | −0.0070 | −0.0043 | −0.0030 |
| max | 0.176 | 0.072 | 0.119 | 0.503 |
| L1 to its clip-and-renormalised version | 0.080 | 0.052 | 0.075 | 0.131 |

| read as | legal mass | top-1 legal | CE (Bayes 2.011) | unedited EI | PI pt4·α3 | ND pt4·α0.1 | GS pt4·α0.05 |
|---|---|---|---|---|---|---|---|
| `raw` (canonical) | 0.9888 | 0.9984 | 2.048 | −0.817 | +0.684 / 0.23 | +0.739 / 0.18 | +0.730 / 0.20 |
| `clipnorm` | 0.9509 | 0.9984 | 2.087 | −0.773 | +0.633 / 0.25 | +0.689 / 0.19 | +0.663 / 0.22 |

- The head is a distribution to within 0.2% on the sum and 0.04 of negative mass — the
  largest negative output is −0.007 (p05). Clipping and renormalising moves the vector by
  0.08 in L1, most of it renormalisation of the ≈ 0.04 of positive mass that the negatives
  were cancelling.
- **The negative outputs sit on illegal moves.** Under `clipnorm` legal mass *drops*
  0.989 → 0.951 and CE rises 2.048 → 2.087: the model puts small positive noise on
  illegal squares and small negative values on other illegal squares, and the two cancel in
  the raw sum. Clipping keeps the noise and throws the cancellation away. So the raw reading
  is the *more* faithful one about legality, and the canonical `legal_mass` (0.989) stands;
  quote the clipnorm value only as the "forced distribution" reading.
- Editability is robust to the reading: every Edit Index shifts by −0.04 to −0.07 under
  `clipnorm` (the renormalised estimates are less peaked), and PI +0.63 / ND +0.69 / GS +0.66
  still match or beat the CE model's +0.61 / +0.62 / +0.65. Fidelity is unchanged
  (0.19–0.25). A canonical re-score under `clipnorm` would not change the conclusion.

## Caveats

- One seed, one objective swap. The head's Brier score has no CE-model counterpart in the
  gates; a Brier of the CE model's softmax on the same test games would make the calibration
  comparison two-sided (cheap: `gates()` has the outputs).
- The α grids are the CE model's. ND's best arm at α 0.1 is interior to its grid
  (0.05 … 2.0), PI's at α 3 interior to its (0.25 … 5); only GS is edge-pinned, as before.
- Probe Skill on the MSE model is with the same 20k probe games and the same cached probe
  grid protocol (`runs/objective_ablation/L-oth-20m-mse/probes/`); the random-init floor is
  the shared (oth-uniform, transformer_l_tokens) baseline.
