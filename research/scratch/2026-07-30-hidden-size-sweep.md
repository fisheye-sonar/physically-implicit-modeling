# Hidden-size sweep: how do the four affordances scale with latent capacity?

**Date:** 2026-07-30 · **Branch:** `michael_controls` · **Direction:** `hidden-size-sweep` (`[in-frame]`, sub-Q 1+2+3)
· **Status:** → **FLAG FOR PROMOTION** (clean confirmation that the §4 negative is capacity-independent; plus one
pre-registered hypothesis clearly refuted) · **Origin:** Michael's controls · **Author:** orchestrator.

## The question
Every result in this repo was measured at **one hidden size, `H = 256`**, chosen by default and never justified.
Does the picture change with capacity? Specifically: could the editability negative be an artifact of a latent with far
more dimensions (256) than the world has degrees of freedom (8)?

## Setup / provenance
Five GRUs, **one variable**: `hidden_size ∈ {8, 32, 128, 256, 512}`. All on `datasets/4_fixed_refl_inview`
(obs noise 0.2, position noise 0.04), 400 epochs, batch 256, AdamW lr 1e-3, wd 1e-4, seed 0, 1 layer.
Runs `runs/controls/H{8,32,128,256,512}`; registry `notebooks/experiments/editability/controls/CONTROL_RUNS.md`.
Metrics from `scripts/eval_controls.py` (2000 test sequences for probes, 64 edits, `ef=20`, `K=15`).
Notebook `notebooks/experiments/editability/controls/hidden_size_sweep.ipynb` (14 cells, 0 errors, 6 figures), PNGs in
`/tmp/hidden_size_sweep/`. Reference points: the world's true state is **8 numbers**; the observation is **128 rays**.

## Headline
**Capacity moves predictive quality and readability a great deal, and grabbability not at all.** Prediction saturates
by `H=128`. Linear readability rises *monotonically* with capacity — refuting the pre-registered guess that a squeezed
latent would be more linearly readable. And at **every** hidden size the probe-directed editors leave the ghost in
place (or remove it only by wrecking the rollout) while the decoder-gradient oracle clears it on the same model and
decoder.

## Results

**§1 Predictive quality (Fig 1).** Saturates by `H=128`; `H=512` buys nothing over `H=256`.
| | H=8 | H=32 | H=128 | H=256 | H=512 |
|---|---|---|---|---|---|
| parameters | 2,616 | 14,688 | 132,096 | 460,672 | 1,707,648 |
| next-step RMSE vs clean | 0.1495 | 0.1167 | 0.1054 | **0.1041** | 0.1042 |
*(copy-previous-frame 0.2139, observation noise floor 0.1539, random frame 0.3962 — shared, all five use dataset 4.)*

**§2 Recoverability — the refuted hypothesis (Fig 2).** I pre-registered that the **linear** readout would be *best at
small H* (a squeezed latent cannot afford a curved embedding). **Wrong, and monotonically so:**
| | H=8 | H=32 | H=128 | H=256 | H=512 |
|---|---|---|---|---|---|
| position R² (linear) | 0.175 | 0.305 | 0.754 | 0.828 | **0.855** |
| velocity R² (linear) | 0.002 | 0.092 | 0.369 | 0.471 | **0.531** |
A capacity-starved latent is not a cleaner one — it simply fails to represent the state. Note `H=8` sits *exactly* at
the world's true dimensionality and still reads position at R² 0.175: the 8 numbers are there in principle, but nothing
in a next-step prediction objective makes the model use its 8 dimensions that way.

**§3 Canonicality (Fig 2b) moves the other way**, which is the honest counterweight: fiber residual (fraction of ‖h‖
*not* a function of (pos, vel)) rises with capacity — linear 0.685 → 0.875, MLP **0.215 → 0.601**. Big-`H` states carry
proportionally much more non-physical content. So capacity trades canonicality for readability; the two standard §2/§3
metrics genuinely point in opposite directions along this axis.

**§4 Editability — the decisive result (Fig 3, Fig 5, Fig 6, Table 2), on the canonical metric set.**
Edit Index: **+1** = the output *is* the edited world, **−1** = the unedited world, **0** = equidistant from both.
A structural editor is only credited when it passes the **fidelity guard** (GT-traj RMSE ≤ unsteered's).
| | H=8 | H=32 | H=128 | H=256 | H=512 |
|---|---|---|---|---|---|
| unsteered (do nothing) | −0.52 | −0.62 | −0.68 | −0.68 | −0.68 |
| best structural editor passing the guard | *none pass* | *none pass* | −0.60 | −0.63 | −0.64 |
| decoder-gradient **oracle** | **+0.58** | **+0.76** | **+0.87** | **+0.94** | **+0.99** |

**No capacity makes the latent grabbable.** At `H ≥ 128` every probe-directed editor sits within 0.08 of "did
nothing". At `H = 8` and `H = 32` the MLP-probe gradient appears to reach ≈ 0 — but it gets there by **destroying
the observation**: its Target/Ghost/Collateral RMSEs exceed **1.0** (observation intensity is bounded in [0,1]) and
its fidelity ratio is up to **2.2×** unsteered. The Edit Index reports that correctly as ≈ 0, "neither world",
rather than as success — which is exactly what the retired `reach % of swap` metric got wrong here, scoring the
same edits at **400–440%**.

Meanwhile the oracle clears it on the *same model and decoder* at every `H`, so a state that renders the target
exists and rolls out fine — the failure is the **reachability of the edit map**, not capacity, and not the model.

**One clean new trend:** the oracle's Edit Index rises monotonically with capacity (**+0.58 → +0.99**). A larger
latent makes the target-rendering state *more precisely reachable by decoder optimisation*, even though it does
nothing for probe-directed reachability.

**But the oracle's win is a SINGLE-FRAME win (Fig 3b, added 2026-07-30 at Sevan's suggestion).** Tracking the Edit
Index at every rollout step shows the decoder-gradient edit does not *hold*: on `H=256` it goes
**+0.94 at step 0 → +0.26 at step 5 → +0.02 by step 14** — past 0 and into "neither world".
So even the one write mechanism that reaches the edited world does so for one frame and then decays. This is why
the trajectory view is not optional: a step-0 scorecard alone would have called this a success.

## A methodological addition — the fidelity guard
This sweep is what forced the **§4 metric redesign** (2026-07-30, now the canonical set in
`METRICS_AND_EDITORS.md` §4 / `scripts/editability_metrics.py`). At `H=8`/`H=32` the old ratio metrics reported
`reach` of **400–440% of the oracle observation** and ghost ratios of 0.725–0.899 — both of which read as success.
The same edits have **GT-traj RMSE up to 2.2× unsteered** and zone RMSEs above **1.0**: they were destroying the
observation, and intensity in the vacated rays fell as a side effect. A metric that scores *change* rather than
*correctness* inverts the conclusion exactly where the model is weakest.
The replacement fixes it two ways: the **Edit Index** scores an output far from *both* ground-truth worlds at
≈ 0 rather than at a large positive, and the **fidelity ratio** (`GT-traj RMSE(editor)/GT-traj RMSE(unsteered)`,
> 1 = the edit left the rollout further from the truth than doing nothing) gates any success claim.

## Caveats / open
- One seed per hidden size — the differences in §2/§3 are large and monotone, but no seed band was measured.
- GRU only; probes fit in-sample-per-model on held-out frames (70/30), not held-out sequences.
- `H=8`/`H=32` are weak enough that their editability numbers mostly measure fragility, not structure.

## Open questions for Sevan
- Artifact or signal? The capacity-independence of the §4 negative looks like the strongest form yet of that finding.
- The readability-vs-canonicality opposition along the capacity axis is a genuinely new observation — worth its own
  entry, or does it belong in `findings/state-geometry.md`?
