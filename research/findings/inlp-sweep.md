# INLP sweep on discworld — per-variable copy counts order the runs the WRONG way for a redundancy story, and writing every copy adds +0.08–0.10 where the single probe is inert (2026-09-14)

**Status:** measured 2026-09-14 (evening) on eight run × target cells (`experiments/inlp_sweep/`; units
`inlp_sweep`, `inlp_sweep_5ray`, `inlp_sweep_rescore`; 9–23 min per cell). Single seed of the cascade
split; canonical 1000-case benches; eval 2026-09-12.2 canonical columns.

## Question and bets (Sevan)

Per residual point, how many orthogonal linear copies of each STATE VARIABLE does the residual hold —
measured, as in the Othello INLP (`adjacent-flip-ablation.md` §INLP), by ONE rank-1 deflation cascade per
variable, deflating that variable's own fitted direction until its held-out R² is exhausted — and does
writing all the copies at once, each copy's target shrunk toward the population mean by its R², edit
better than the canonical single-probe editors? Runs: `L-dw-noiseless-20m`, `L-dw-8ray-20m`,
`L-dw-smooth-20m`, `L-dw-8ray-tok-20m` on the full frustum state (8 variables: x, y, vx, vy of both
discs), and `L-dw-noiseless-20m`, `L-dw-8ray-20m`, `L-dw-5ray-20m` on `appearance-fac` (4 ordinal
variables: run centre and run length of each disc), plus `L-dw-5ray-20m` on the full state as the
factorised row's own baseline. Redundancy = dimensions removed per variable before the mean-over-variables
held-out R² falls below 0.4 (and 0.05; and exhaustion < 0.02).

Bets on record before any row landed (README): **Sevan** — 8-ray slightly LESS redundant than the other
regression runs; the appearance-fac variables MEANINGFULLY less redundant, most confidently on 5-ray.
**Claude** — 8-ray at least as redundant as noiseless and smooth; appearance-fac about EQUAL to position.
**Outcome:** 8-ray is ~4× more redundant than noiseless; the factorised variables are MORE redundant than
position on every run (5-ray fac the most redundant cell of all). Sevan's three predictions lose; Claude's
were right in direction and wrong in size on the factorised half.

## Method

`experiments/inlp_sweep/scripts/inlp_dw_sweep.py`. Residuals at each of the 9 points over 20k probe
sequences (canonical split by sequence, canonical standardisation); per variable a closed-form min-norm
least-squares cascade from moment matrices (float64, GPU), removing the fitted direction and refitting
until held-out R² < 0.02 or 480 iterations; a matched random-direction control (same count of random
directions inside the remaining subspace — flat at every cell: the copies are real). The write: every
variable's first K copies stacked into one matrix and solved JOINTLY — Δz = (AᵀΩA + λI)⁻¹AᵀΩ(t − r), rows
weighted by their R² (Ω), λ = 10⁻² × mean diagonal — with shrunk targets t_k = μ + R²_k(t − μ). Multi-output
least squares is separable, so at K = 1 the rows are the joint lstsq probe's and the write is PI in
z-space (verified: 8-ray pt 4 K = 1 +0.23 / 0.98 = canonical PI's guarded arm there; pt 1 −0.02 / 0.74 vs
PI −0.00 / 0.74). K ∈ {1, 2, 4, 8, 16, 32, 64, 128, all} × α ∈ {0.25 … 175}; canonical scorecards (ray-zone
Edit Index + guard; the token model's frame-set construction, †). Sevan caught the first write (a sum of
independent per-variable steps, the Othello script's form, which has no cross-talk when ONE tile is
written but disturbs the other variables' read-outs when eight are) — its results are superseded and kept
under `scores/_superseded/`; the unregularised joint pseudo-inverse blows up from K ≈ 8 (guards 3–4 at
every α) and was replaced by the weighted ridge. Cascades persisted under `probes/<run>/cascade_pt*.pt`.

## The table

| run (target) | unedited | dims removed per variable to mean R² < 0.4, pts 1–8 | to < 0.05 | to exhaustion | best guarded K-copy write, shrink (pt, K, α) | K = 1 at that point (≡ PI) | best unguarded | canonical PI | canonical ND | canonical GS |
|---|---|---|---|---|---|---|---|---|---|---|
| `L-dw-noiseless-20m` (full) | −0.93 | 9 / 7 / 7 / 6 / 6 / 5 / 5 / 5 | 24 / 15 / 13 / 13 / 13 / 12 / 12 / 12 | 30 / 19 / 17 / 16 / 16 / 15 / 15 / 15 | +0.03 / 1.00 (pt 4, K 8, α 12) | +0.21 / 1.99 | +0.27 / 1.66 (pt 1, K 8) | +0.23 / 1.54 | n/a | −0.08 / 1.07 |
| `L-dw-8ray-20m` (full) | −0.90 | 40 / 24 / 22 / 23 / 24 / 24 / 24 / 24 | 104 / 57 / 48 / 50 / 52 / 53 / 54 / 53 | 114 / 69 / 57 / 58 / 58 / 58 / 59 / 60 | **+0.34 / 0.93** (pt 1, K 142 = all, α 20) | −0.02 / 0.74 | +0.34 / 0.93 | +0.26 / 0.99 | n/a | −0.03 / 0.90 |
| `L-dw-smooth-20m` (full) | −0.97 | 29 / 29 / 27 / 25 / 24 / 24 / 24 / 24 | 69 / 56 / 51 / 50 / 48 / 48 / 47 / 47 | 74 / 62 / 58 / 56 / 55 / 54 / 54 / 54 | −0.11 / 0.98 (pt 3, K 64, α 8) | +0.11 / 2.40 | +0.11 / 2.59 (pt 4, K 1) | +0.11 / 2.62 | n/a | −0.02 / 1.16 |
| `L-dw-8ray-tok-20m` (full) † | −0.75 | 47 / 23 / 15 / 14 / 14 / 14 / 15 / 15 | 173 / 73 / 52 / 46 / 44 / 44 / 47 / 61 | 169 / 78 / 60 / 56 / 53 / 52 / 54 / 62 | +0.03 / 0.80 (pt 5, K 70 = all, α 35) | +0.00 / 0.75 | +0.03 / 0.80 | +0.00 / 0.75 | n/a | −0.15 / 0.80 |
| `L-dw-noiseless-20m` (appearance-fac) | −0.93 | 12 / 8 / 8 / 7 / 8 / 7 / 7 / 7 | 25 / 15 / 13 / 14 / 14 / 14 / 15 / 15 | 30 / 18 / 17 / 17 / 18 / 18 / 18 / 18 | +0.02 / 1.00 (pt 4, K 8, α 12) | +0.20 / 2.00 | +0.25 / 1.67 (pt 1, K 8) | +0.01 / 1.93 | +0.61 / 0.80 | +0.33 / 0.71 |
| `L-dw-8ray-20m` (appearance-fac) | −0.90 | 61 / 33 / 29 / 31 / 32 / 33 / 34 / 33 | 115 / 64 / 54 / 56 / 58 / 58 / 59 / 59 | 132 / 78 / 64 / 66 / 66 / 68 / 68 / 68 | +0.38 / 0.97 (pt 1, K 64, α 20) | −0.03 / 0.79 | +0.40 / 1.11 (pt 1, K 64) | +0.38 / 0.95 | +0.49 / 0.98 | +0.46 / 0.54 |
| `L-dw-5ray-20m` (full) | −0.89 | 59 / 32 / 27 / 28 / 28 / 27 / 27 / 26 | 172 / 82 / 65 / 64 / 65 / 62 / 62 / 60 | 177 / 92 / 72 / 71 / 71 / 69 / 69 / 68 | +0.36 / 1.00 (pt 0 — the degenerate embedding point; pts 1–5: **+0.31–0.34 / 0.82–0.92**, K 32–95) | +0.18 / 1.02 (pt 1: −0.66 / 0.86) | +0.38 / 1.10 (pt 0, K 1) | +0.24 / 0.94 | n/a | −0.06 / 0.86 |
| `L-dw-5ray-20m` (appearance-fac) | −0.91 | 98 / 48 / 40 / 42 / 43 / 42 / 40 / 39 | 183 / 91 / 74 / 74 / 74 / 71 / 71 / 70 | 211 / 109 / 88 / 87 / 88 / 85 / 85 / 84 | **+0.49 / 0.98** (pt 4, K 64, α 8) | +0.31 / 1.04 | +0.49 / 1.01 (pt 4, K 96 = all) | +0.51 / 0.70 | +0.55 / 0.58 | +0.60 / 0.46 |

ND is n/a on regression rows by the registry's rule (one fixed direction with a swept scalar is coherent only
for a categorical target). Figures: `experiments/inlp_sweep/outputs/inlp_r2_by_iteration.png` (all eight
cells) and `inlp_r2_by_iteration_<run>[_fac].png`, in the style of the Othello INLP figure. Per-variable
curves, per-point arms and the random controls are in `scores/inlp_<run>[_fac].json`.

## Reading

1. **Redundancy ordering (dims per variable to R² < 0.4, points 2–8):** 5-ray fac 39–48 > 8-ray fac 29–34 ≈
   5-ray 26–32 > smooth 24–29 ≈ 8-ray 22–24 > 8-ray token 14–23 > noiseless-fac 7–8 ≈ noiseless 5–7. Fewer
   rays means MORE orthogonal copies of each variable, not fewer; the observation-exact factor variables
   hold at least as many copies as position on every run and more on the coarse ones; the token model's
   state has as many copies to exhaustion as its frame twin (52–62) but decays 1.5× faster, i.e. fewer
   strong copies and a longer weak tail. Position is more redundant than velocity everywhere (8-ray pt 4:
   63–71 vs 42–57 copies). Point 1 is the outlier at every cell (2–4× the deeper points).
2. **The copy count predicts editability in NEITHER direction across models.** Editability by any write
   orders the runs 5-ray ≈ 8-ray > noiseless > smooth (canonical PI, the K-copy write, and the inverse map
   of `inverse-probe.md` all agree). Redundancy orders them 5-ray > smooth ≈ 8-ray > noiseless. Smooth is
   the case that breaks a story in either direction: as redundant as 8-ray, the least editable by every
   write. Othello (`adjacent-flip-ablation.md` §INLP) ran the OTHER way — the editable model had the fewest
   copies (30 vs 85–230 on the inert adjacent). So the number of linear copies is a symptom of code
   geometry (the parity lookup smeared through adjacent's residual; position spread thin when 5 rays barely
   constrain it), not a variable that sets editability.
3. **Within a model, the copies do what the redundancy story says — with a small ceiling.** Where the
   single joint probe (≡ PI) is inert, writing all copies helps: 8-ray pt 1 −0.02 → +0.34 / 0.93 (K = all),
   5-ray pts 1–5 −0.66…+0.27 → +0.31–0.34 / 0.82–0.92, adjacent Othello (earlier) inert → +0.47. Where the
   single probe already works, more copies add ≤ 0.05 (8-ray fac +0.38 = PI +0.38; 5-ray fac +0.49 vs PI
   +0.51 / 0.70 with a better guard). Where nothing works — noiseless, smooth, the token model,
   noiseless-fac — no K at any point clears the guard with more than +0.03. The gain over canonical PI is
   +0.08–0.10 on 8-ray and 5-ray, the same +0.06–0.09 the 2026-09-04 whole-probe cascade found, and every
   K-copy ceiling sits far below the state → latent write on the same runs (+0.83 / +0.87) and below the
   categorical editors on the factorised targets (ND +0.61 on noiseless-fac, GS +0.60 on 5-ray-fac).
4. **What this settles for the programme.** "Discworld's single-probe write fails because the code is
   redundant and one probe moves one copy" is true as far as it goes on the coarse-ray runs — and goes
   +0.08. It is false as an account of the cross-run ordering (reading 2) and false on noiseless and
   smooth, where writing every linear copy of every variable moves nothing. The block on the 128-ray runs is
   not in the number of copies; the inverse-probe finding locates it in the write direction (the
   state-conditional displacement, which the probe's row space and its copies do not span).

Caveats: single seed of the cascade split (probe-seed variance on this project is ≈ 0, run-seed SD
0.003–0.016 — `seed-variance.md`); the selection rule "best guarded arm" picked the degenerate embedding
point on 5-ray full (quoted with the points 1–5 numbers beside it); one ridge constant and one shrink rule
(the truncated pseudo-inverse at rtol 10⁻² gave +0.35 / 0.93 vs +0.33 / 0.96 on the 8-ray check); the
lstsq probes differ from the gradient-fitted canonical probe by 6–10 % in the K = 1 step (as before); the
token model is scored by the frame-set Edit Index and is not directly comparable to the frame rows.

Provenance: `experiments/inlp_sweep/{README.md, scripts/, drivers/, scores/, outputs/}`; cascades
`experiments/inlp_sweep/probes/<run>[_fac]/cascade_pt*.pt` (gitignored, on disk); logs `logs/inlp_sweep/`.
