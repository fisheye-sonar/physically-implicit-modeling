# The "saturation step" I extrapolated from was an overfitting turnaround, not convergence

**2026-08-24, ~01:00, from the 20M overnight run at step 240k (still running).**
Same architecture (Transformer L, 25.3M), same environment (Othello), same optimizer.
The *only* difference between the two runs below is the size of the game pool.

## What the sweep plan assumed

The compute budget rested on: "saturation was ~6k steps at 90k games and ~58k at 900k, so 20M
should saturate near 1-1.5M steps." That extrapolation is what turned a 31-day grid into a
4-day grid, and it is **wrong in kind, not just in magnitude**.

Those two numbers were not convergence points. They were **the step at which val loss turned
around and started climbing** — the overfitting minimum. Overfitting onset does scale with pool
size, but it is not the same quantity as convergence, and the 20M cell does not have one yet.

## The figure

`runs/scaling/loss_regime.png` — regenerate any time with
`python notebooks/experiments/editability/scaling/loss_regime_figure.py` (cheap, idempotent,
picks up whatever val passes the 20M run has written by then).


## Refinement at step 360k (added 05:00) — the power law is not perfectly clean

With 72 val passes the curve has begun to run **above** the power law fitted to steps 25k-200k,
and the gap grows monotonically: +0.0014 at 240k, +0.0021 at 280k, +0.0022 at 320k, +0.0031 at
360k. So "a clean power law with no knee" was too strong. The accurate statement is:

* The power law holds tightly over **25k-240k** (log-log R² 0.9945).
* Past ~250k the curve falls **slightly slower** than that law predicts.
* It is **still falling**, not plateaued: slope over the last 20 val passes is
  **−0.0028 per 100k steps at 13.2σ**. The apparent flatness in the last 8 passes alone is
  noise (1.8σ, range 0.0017).

**A likely cause is the constant learning rate, and this is a live decision.** Under a constant
LR a run settles at an SGD noise floor set by the LR, not by model capacity. The reference point
is in this repo already: the 14-epoch 900k run gained **2.0961 → 2.0841 = 0.0120 from annealing
alone**. If the same holds here, the constant-LR endpoint understates what these weights can do
by roughly that much.

Endpoint projections for step 780,000, which bracket the answer:

| method | projected val | excess |
|---|---|---|
| power law fit on all data | 2.0245 | +0.0153 |
| linear extrapolation of current slope (optimistic — curve is decelerating) | 2.0222 | +0.0130 |
| power law + observed upward deviation (my best guess) | ~2.028–2.030 | ~+0.019–0.021 |
| ...then a short anneal, if the 0.0120 reference transfers | ~2.017–2.020 | ~+0.008–0.011 |

**Not acting on this.** Constant LR was chosen deliberately so that checkpoint *k* means "trained
for *k* steps" and so the run can be extended without a second warmup; annealing mid-run would
destroy both properties. Whether to append a short anneal *after* step 780k — which preserves
them, since the constant-LR checkpoints are all already on disk — is Sevan's call.

## Evidence

Bayes floor for this generator is **2.0092** (`E[log |legal|]`); "excess" below is val − 2.0092.

### 900k games (`L90_theirs_othello`, 143 val passes, finished)

| step | train | val | val − train |
|---|---|---|---|
| 42,000 | 2.0961 | 2.0940 | −0.0021 |
| **58,000** | — | **2.0881** ← best | — |
| 82,000 | 2.0388 | 2.0940 | +0.0553 |
| 142,000 | 1.9781 | 2.1256 | +0.1475 |
| 282,000 | **1.9204** | 2.1790 | +0.2585 |

Best val **2.0881** (excess +0.0789) at step 58,000, then gives back **+0.0908** by the end.
Fitting a power law to steps ≥ 25k returns a **positive** exponent (+0.397): over that range it
is getting worse, not better.

**The smoking gun is the train loss: 1.9204 < 2.0092.** It has driven training loss *below the
true conditional entropy of the generator*. That is only possible by memorizing which of the
legal moves was actually drawn. This run is not a weaker world model of Othello; it is partly a
lookup table over 900k specific games.

### 20M games (`BIG20M_othello_L`, running, 48 val passes so far)

Excess over Bayes fits `18.78 · step^(−0.5263)`, **log-log R² = 0.9945** over steps 25k–240k
(44 points, max residual 0.063). Nearly a clean −1/2 exponent, no knee anywhere in a 10× span.
At step 60,000 the train–val gap is **−0.0134** — val still *below* train (dropout), i.e. no
overfitting has begun at all.

### Head to head, matched steps

| step | 20M excess | 900k excess | ratio |
|---|---|---|---|
| 50,000 | +0.0624 | +0.0812 | 1.30× |
| 100,000 | +0.0427 | +0.0917 | 2.15× |
| 150,000 | +0.0350 | +0.1227 | 3.51× |
| 200,000 | +0.0308 | +0.1517 | 4.93× |
| 240,000 | +0.0285 | +0.1656 | 5.80× |

The gap widens monotonically because the two runs are moving in opposite directions.
**At step 240k the 20M run is already 2.8× closer to Bayes than the 900k run ever got at its
own best step.**

## Consequences

1. **"Train the 20M cell to saturation" is not a well-posed instruction.** There is no knee.
   Under a −1/2 exponent, halving the excess costs 4× the steps: excess 0.020 at ~444k steps,
   0.015 at ~768k, 0.010 at ~1.66M, 0.005 at ~6.2M. A stopping point is a **budget decision**,
   not a fact to be discovered. The sweep needs a declared tolerance or a declared step budget.
2. **Cells are not comparable at "convergence".** The 900k cell's best model is a partly
   memorizing one; the 20M cell's is not. Matching cells by *steps* or by *epochs* compares
   different regimes. Matching by *excess over Bayes* may be the more honest axis — though it
   cannot be equalized upward, since 900k can never reach +0.028.
3. **Every Othello editability number so far comes from the memorizing regime.**
   `L90_theirs_othello`'s `best_model.pt` is the step-58,000 checkpoint, excess +0.0789. The
   +0.241 Edit Index was measured there. The 20M checkpoints are a materially better world
   model, and re-running the editability suite across them is the natural next test — it is
   also exactly what the log-spaced checkpoints were saved for.

## What this does NOT say

It does not say the editability conclusion is wrong. The environment-flip result
(Othello editable / discworld not) was measured with matched architecture, matched epochs and
matched pool size on both sides, so this regime issue applies to both arms equally. What it
does say is that the Othello arm was *not* tested at its best, and there is now a much better
model to test it on.
