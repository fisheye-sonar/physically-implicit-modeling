# 2026-08-21 — Nanda's linear-direction intervention, on their model and on ours

Scripts `othello_transfer/{linear_intervention,nanda_on_discworld}.py`. Nanda, Lee & Wattenberg,
*Emergent Linear Representations in World Models of Self-Supervised Sequence Models*
(arXiv:2309.00941), §4.1. Probes are ours (`othello_probe.fit_probe`, `hidden=None`), from the
run-5 cache: mine/theirs linear, **0.72%** held-out error. ~10 min total. **No models trained.**

Their method: `x' <- x + alpha * p_d`, where `p_d` is the linear probe's weight column for the
target direction at the intervened tile, added at **every** layer. One vector addition, no
gradients. Their Table 2: null **2.723**, Li's non-linear gradient method **0.12**, their linear
addition **0.10**.

## 1. It replicates, and their null baseline matches ours exactly

Our reproduced null is **2.723** — identical to their Table 2, confirming the same benchmark and
metric.

| arm | alpha | ‖Δx‖/‖x‖ | Li error ↓ | vs pre ↑ | Edit Index union | symdiff | legal mass |
|---|---|---|---|---|---|---|---|
| Nanda, add target direction | 0.12 | 0.120 | **0.108** | 1.678 | +0.593 | +0.748 | 0.990 |
| Nanda, add target direction | 0.18 | 0.180 | **0.062** | 2.717 | +0.603 | +0.854 | 0.996 |
| Nanda, add (target − current) | 0.12 | 0.120 | **0.026** | 2.947 | **+0.691** | **+0.895** | 0.998 |

α = 0.12 gives **0.108** against their published **0.10**. The `target − current` variant is the
best result anything has achieved on this benchmark — better than Li's 0.12, Nanda's 0.10, and our
own gradient editor's 0.016/+0.656 from run 5 on the Edit Index axis.

`w / x_std` beats the raw weight row (0.062 vs 0.116), as expected: our probe standardises its
input, so `w/σ` is the true raw-activation-space gradient.

**Their Figure 7 shape reproduces.** Writing at the first 1–5 residual points does nothing
(2.723 → 1.668); the error collapses only at 6+ (0.430 → 0.122 → 0.104). Their "a sufficient
number of layers need to be intervened."

## ⚠ CORRECTION (same day, later) — §2 below is WRONG, and the reason is mechanistic

`othello_transfer/single_layer.py`. Applying the injection at **one residual point** instead of all
nine, sweeping alpha at each point:

| point written | 0 | 1 | 2 | 3 | 4 | **5** | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| best Li error ↓ | 2.719 | 2.721 | 2.581 | 0.763 | 0.296 | **0.052** | 0.695 | 1.872 | 2.164 |
| Edit Index union | −0.829 | −0.823 | −0.579 | +0.404 | +0.559 | **+0.697** | +0.206 | −0.443 | −0.136 |

**At residual point 5 alone (alpha 1.5, write ratio 0.470): Li error 0.052, Edit Index +0.697,
legal mass 0.993.** That is the **highest Edit Index of any mechanism tested on this benchmark** —
above Nanda's +0.603, above our gradient editor's +0.656 — and a Li error well below the published
0.10 (Nanda) and 0.12 (Li).

**So the pseudoinverse is not a weak editor. The hypothesis in §Open below was right:** re-imposing
"hold all 64 tiles at their current read-out" at every layer makes each layer **undo the previous
layer's edit**, because the constraint is recomputed against the already-edited stream. Nanda's
fixed direction cannot do this — it does not depend on `x`.

The asymmetry is the evidence, and the single-layer control for Nanda's method supplies it:

| | best single point | all 9 points |
|---|---|---|
| Nanda addition | 0.236 (point 5) | **0.062** — multi-layer *helps* |
| our pseudoinverse | **0.052** (point 5) | 1.461 — multi-layer *hurts 28x* |

Both methods work only at **mid-depth** (points 3–6) and fail at both ends — too early and the write
is recomputed away, too late and too few blocks remain to propagate it.

**Consequences.**
1. The 2x3 table at the end of this note is wrong in one cell and is corrected below.
2. 2026-08-20's "our editor implementation is cleared" **stands unqualified**. I had started to
   walk it back; that walk-back was mistaken.
3. **Actionable on our own world — and immediately checked.** The obvious worry is that
   discworld's transformer editing suffers the same pathology. **It does not.**
   `transformers/transformer_world_state.ipynb` §4 (2026-08-04) already writes
   `h + (target − (Wh+b))W⁺` at **each residual point individually**, on all three windows, and finds
   it **inert at every one** (Edit Index −0.65…−0.68 = each model's own unsteered value, fidelity
   ratio 1.00). ⚠ I stated the opposite in the first version of this note; Sevan corrected it. The
   genuinely untried variant is narrower: 2026-08-04 took the **full jump** (α = 1), and never swept
   the step size — where Othello's single-point optimum is α = 1.5 with a ~50× spread.

---

## 1b. The one gap closed: single-point pseudoinverse on `W16` WITH the step size swept

`othello_transfer/pinv_alpha_discworld.py`. 2026-08-04's configuration held fixed (edits split,
N = 192, K = 15, probe fit by exact `lstsq` on 1200 test sequences per residual point, target = the
true post-edit positions), with **α** added as the only new axis: `h ← h₀ + α·(target − (Wh₀+b))W⁺`,
α ∈ 0.05…6.0, at each of the 5 residual points individually. 9 s.

**The anchor holds.** α = 1.0 — 2026-08-04's exact configuration — gives Edit Index **−0.683…−0.669**
across the five points, against that day's published **−0.68…−0.65**. Same measurement.

**The α axis does not rescue it.** Best cell over the whole 5 × 11 grid: point 2, α = 6.0, Edit Index
**−0.443** from an unsteered **−0.684**. It never crosses zero. Edit Index by α at the two live
points:

| α | 0.05 | 0.25 | 1.0 | 2.0 | 4.0 | 6.0 |
|---|---|---|---|---|---|---|
| point 1 | −0.684 | −0.683 | −0.676 | −0.654 | −0.575 | −0.493 |
| **point 2** (mid-depth) | −0.684 | −0.682 | −0.669 | −0.634 | −0.538 | **−0.443** |
| point 4 (last) | −0.684 | −0.683 | −0.681 | −0.677 | −0.667 | −0.656 |

**Monotonic, with no optimum** — the same signature as Nanda's addition on discworld, and the
opposite of Othello, where there is a sharp peak at α ≈ 0.12–0.18 (addition) / 1.5 (pseudoinverse).

⚠ **And the "best" cell is not an edit.** At point 2, α = 6.0 the write is ‖Δh‖/‖h‖ = **0.909** —
almost as large as the activation itself — and it leaves the probe read-out **15.96 sim units** from
the target, having overshot it by ~5× the original error of 3.19. Meanwhile collateral RMSE rises
0.1285 → **0.1662** (+29%) while target RMSE improves only 0.4848 → 0.4460 (−8%). At every α where
the write actually *lands* the read-out (α ≈ 1, probe error 0.000) the index is inert to three
decimals. So the honest statement is: **where the editor does what it is defined to do, it does
nothing; where it moves the output, it is no longer doing what it is defined to do.**

Depth matters here in the way Othello predicts and still does not help: points 1–2 (mid-depth on a
4-layer model, the analogue of Othello's points 3–5 of 9) are the only ones that move at all, and
points 0, 3 and 4 are flat across the whole α range.

**Conclusion: the full Othello recipe — single residual point, mid-depth, step size swept — has now
been applied to discworld and fails.** No part of the multi-layer pathology explains the discworld
negative.

---

## 2. Our pseudoinverse editor fails on their model — ⚠ SUPERSEDED by the correction above

`pim.editors.probe_steering.inject_state`, **unmodified** — the bridge is a four-line adapter
exposing `.linear` (our `WorldStateProbe` keeps that layer as `.net`), plus solving in the probe's
standardised space and mapping back, which is exact. Same probe, same model, same benchmark.

Full sweep, **15 alphas, write ratio 0.005 → 6.76**:

| alpha | 0.02 | 0.25 | **0.5** | 1.0 | 2.0 | 4.0 | 6.0 |
|---|---|---|---|---|---|---|---|
| Li error | 2.673 | 1.926 | **1.461** | 1.862 | 6.675 | 16.228 | 16.264 |
| Edit Index | −0.827 | −0.720 | **−0.275** | +0.037 | −0.021 | −0.041 | −0.043 |

**Never below 1.461 Li error; never above +0.037 Edit Index.** Against 0.062 / +0.603 for a single
vector addition along the *same probe's weight row*.

⚠ **Caveat.** `inject_state` is designed for a **regression** readout; here it is applied to a
3-way classifier, and the target — swap the intervened tile's mine/theirs logits, hold the other 63
tiles exactly — is a construction choice, not something the discworld code does. So this is not a
clean condemnation of the editor. What it *is*: a same-model, same-probe demonstration that the
minimum-norm null-space-preserving write is far weaker than an unconstrained push along the same
direction.

## 3. Nanda's method on discworld — it does not rescue it

`W16`, `4_fixed_refl_inview` test split, N = 512, edit frame 20. Probe = our linear **regression**
probe (its native setting), best R² 0.778 at point 2. Edit = **pure X displacement** of object 0 by
1.0 sim units, direction = that probe's weight row for the obj-0-x output, `/x_std`, unit-normed.
**Step 0 only**, no rollout, per Sevan.

| arm | Edit Index | target RMSE ↓ | ghost RMSE ↓ | collateral RMSE ↓ |
|---|---|---|---|---|
| unsteered | −0.6006 | 0.3113 | 0.3886 | 0.1549 |
| best (α = 1.0, all 5 points) | −0.1180 | **0.3753** | 0.3029 | **0.5608** |

**It never crosses zero, and the movement it does produce is degradation.** Target RMSE gets
*worse* (0.311 → 0.375) and collateral RMSE is **3.6× worse** (0.155 → 0.561) — the write drags the
*other* object. The Edit Index improves only because the output stops resembling the unedited world,
not because it comes to resemble the edited one. This is precisely the failure the zone metrics
exist to expose and that the index alone cannot see.

The α trend is **monotonic with no optimum**, unlike Othello, where there is a clear peak at
α ≈ 0.12–0.18 with collateral *improving* (legal mass 0.996). There is no regime here where the
edit lands cleanly.

⚠ **A depth caveat that this run cannot settle.** `W16` has 4 layers = **5 residual points**.
Othello-GPT has 9, and Nanda's Figure 7 (which we reproduced) shows the error only collapses once
**≥ 6** are written. We cannot write to 6 points because we do not have them. **This is a direct
argument for the planned architecture-transfer run** (`directions/othello-architecture-on-discworld.md`,
8 blocks / 9 points), and it means the discworld negative here is confounded with depth.

## The 2 × 3 table

| mechanism | Othello-GPT (their model) | discworld `W16` (our model) |
|---|---|---|
| Nanda linear-direction addition | **+0.603 ✓** (Li 0.062, all 9 points) | −0.118 ✗ (3.6× collateral) |
| our gradient editor (`_descend`) | **+0.656 ✓** (Li 0.016) | −0.194 ✗ |
| our pseudoinverse injection | **+0.697 ✓** (Li 0.052, point 5 only) | −0.66 ✗ |

**Corrected 2026-08-21 after the single-layer run.** All three mechanisms work on their model; all
three fail on ours. The editor-implication reading is withdrawn — the world reading is what
survives, and it is now supported by three independent write mechanisms rather than two.

*(An earlier version of this section offered two readings, the first of which — "the
pseudoinverse is a weak editor in its own right" — rested on the superseded §2 and is withdrawn.
Only the reading below survives.)*

**The world difference is what survives, and it is now supported by three independent write
mechanisms rather than two.** Nanda's addition, our gradient editor, and our pseudoinverse all
succeed on Othello-GPT and all fail on discworld `W16`. "Probe-derived writes fail here" is
therefore a statement about the world or the read-out, not about any one editor's implementation.

**The one qualification that does survive** is about *where* the write is applied, not about the
editor: on Othello the pseudoinverse must be written at a **single mid-depth residual point**, and
written at every point it destroys itself. That is an Othello fact. Discworld's numbers already come
from single-point writes (2026-08-04), so the pathology does not explain them.

## Open

- ~~A step-size sweep on the single-point `W16` write.~~ **Done, §1b above — it fails.**
- The depth confound above — settled only by the architecture-transfer run.
- Whether the Othello pseudoinverse failure survives a **regression** readout (its native case),
  which would remove the classifier-target caveat.
- Nanda's `target − current` variant is untested on discworld; it is the strongest arm on Othello
  and costs one line.
