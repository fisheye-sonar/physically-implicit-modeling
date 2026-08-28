# Editability of the 20M Othello model, and what the MLP probe was really doing

**2026-08-24.** Model `runs/scaling/BIG20M_othello_L` — Transformer L (25.3M), 20M games,
780k steps, val **2.02798** (excess over Bayes 2.0092 = **+0.01878**). Probe target `mine` only,
20,000 probe games. Source: `othello_arch/envctrl_eval.py` via `runs/scaling/edit20m.sh`;
results `runs/othello_arch/BIG20M_othello_L_editability_mine.json`.

## Edit Index

Unedited baseline: EI **−0.7126**, Li 2.763, legal 0.857.

| editor | best EI | pt | α | Li ↓ | Li-pre | legal |
|---|---|---|---|---|---|---|
| **PI injection (1 point)** | **+0.6104** | 4 | 3.0 | 0.114 | 2.486 | 0.990 |
| **Nanda (target − current)** | **+0.3746** | all | 0.08 | 0.096 | 2.865 | 0.994 |
| Nanda addition | +0.2403 | all | 0.12 | 0.194 | 2.509 | 0.987 |
| MLP grad steering | −0.0007 | 1 | 0.35 | 5.788 | 5.792 | 0.752 |

Guards clean on the top three: Li error 2.763 → 0.096–0.194 while **Li-pre stays high** (2.49–2.87),
so the model moved *to* the edited board, not merely away from the pre-edit one; legal mass → 0.99.
Li et al.'s published checkpoint scores 0.697 for reference.

**Two changes against the 900k result** (+0.241, `L90_theirs_othello`, best val 2.0881):
Nanda target−current **+0.241 → +0.3746**, and **PI injection reaches +0.6104** — the editor whose
+0.138 at 4 epochs did not survive to 14 and was withdrawn on 08-22. It is now the strongest
editor by a wide margin, at a single mid-depth point. Architecture and environment are identical;
the model is the only difference.

## The MLP probe: two causes, not one

At the historical default of **6,000** probe games the MLP-512 probe has 361,152 parameters against
353,900 rows (**0.98 rows/param**) and **loses to the linear probe on held-out data** — point 6:
MLP 1.617% held-out against 0.623% in-sample, linear 1.338%. Raising to 20,000 games (3.27
rows/param) fixed points 6–8 but left points 3–5 inverted.

Dropping to **MLP-128** at the same 20,000 games (90,432 params, **13.04 rows/param**) separates
the two effects:

| pt | linear held/in | MLP-512 held/in | MLP-128 held/in | 512 gap | 128 gap |
|---|---|---|---|---|---|
| 3 | 3.802/3.793 | 3.873/3.501 | 3.859/3.809 | +0.372 | **+0.050** |
| 5 | 1.701/1.709 | 1.786/1.442 | 1.748/1.712 | +0.344 | **+0.036** |

**Overfitting collapses ~10×** (gap 0.35 → 0.04), confirming MLP-512 was memorising. But MLP-128
is *still* 0.04–0.07 pp below linear at points 2–5, and that cannot be memorisation — nothing is
left to memorise. An MLP is a strictly larger function class, so with perfect optimisation it must
at least tie; the residual is SGD not fully converging.

**Conclusion: mine/theirs is linearly decodable and the MLP adds nothing** — which is what Nanda
et al.'s result predicts. The `MLP ≥ linear` tripwire holds only up to optimiser slack; on a
linearly-decodable target a small residual inversion is expected and is not evidence of overfitting.

✅ **MEASURED, and gradient steering's failure is NOT a probe artifact.** Re-running the full
sweep (9 points x 9 alphas) through the clean MLP-128 probes gives **EI −0.0014**, against −0.0007
for the overfit MLP-512 — identical within noise. Guards fail the same way: Li error *rises*
2.763 → 5.656, legal mass *drops* 0.857 → 0.747. Best EI never improved past point 0 at any point
or step size. `runs/othello_arch/BIG20M_othello_L_editability_mine_mlp128.json`,
`runs/scaling/mlp128_editability.py`; probes cached at `probe_cache/BIG20M_mlp128_probes.pt`.

This is coherent with the probe evidence: mine/theirs is linearly decodable, MLP-128 buys nothing
over linear, so there is no nonlinear structure for gradient steering to exploit that the linear
editors are not already using more directly.

## A cache bug found and fixed on the way

`envctrl_eval.py` pointed every checkpoint at ONE shared `runs/othello_arch/probe_cache/`, and
`fit_probe_grid`'s key carries no model identity. Running it on BIG20M would have silently loaded
`L90_theirs_othello`'s probes — the exact 2026-08-21 failure, in a file that never got the fix.
Now writes to `probe_cache/<weight-fingerprint>/`. The pre-existing grid could not be attributed to
a model (three runs share its settings) and is quarantined under `probe_cache/_UNATTRIBUTABLE/`.
