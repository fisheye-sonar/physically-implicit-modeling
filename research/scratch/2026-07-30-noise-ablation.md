# Noise ablation: which of our two noise sources is doing the work?

**Date:** 2026-07-30 · **Branch:** `michael_controls` · **Direction:** `noise-ablation` (`[in-frame]`, sub-Q 1+2+3)
· **Status:** → **FLAG FOR PROMOTION** (editability negative confirmed noise-independent; one clear positive on
observation noise as a regulariser; two pre-registered hypotheses refuted) · **Origin:** Michael's controls ·
**Author:** orchestrator.

## The question
Every dataset in this repo carries **two independent noise sources** and no result has ever separated them:
- **observation noise** (`obs_noise_std = 0.2`) corrupts the 1D scan — **sensing**. The world is exact; the view is not.
- **position noise** (`position_noise_std = 0.04`) adds Gaussian diffusion to the discs each step — **the world
  itself** is stochastic, so it is unpredictable however well it is sensed.

They are conceptually opposite. We have been reading every finding off the both-on cell of a 2×2 never filled in.

## Setup / provenance
Three new datasets matched to `4_fixed_refl_inview` in every respect except the two noise flags
(`datasets/9_obsnoise0_posnoise0`, `10_obsnoise0_posnoise004`, `11_obsnoise02_posnoise0`; 90k/10k/10k/10k,
`ef=20`, seed 0). Four GRUs at `H=256`, identical recipe (400 epochs, batch 256, AdamW lr 1e-3, wd 1e-4, seed 0):
`runs/controls/{N_obs0_pos0, N_obs0_pos004, N_obs02_pos0}` plus `H256` as the both-on cell (shared with the
hidden-size sweep). Registry `notebooks/experiments/editability/controls/CONTROL_RUNS.md`.
Notebook `notebooks/experiments/editability/controls/noise_ablation.ipynb` (14 cells, 0 errors, 5 figures), PNGs in
`/tmp/noise_ablation/`. Metrics from `scripts/eval_controls.py`.

> **Absolute RMSE is not comparable across cells** (the noise-free cells have a ~0 noise floor), so every predictive
> number is read against that cell's own baselines; Fig 1c normalises by each cell's own copy-previous-frame baseline.
> Each model is evaluated on **its own** dataset's edits split (the in-distribution choice), so the cells contain
> different scenes and Fig 5 gives every row its own GT column.

## Headline
**Neither noise source is what blocks editing** — the §4 negative holds in the fully deterministic, perfectly-sensed
world just as it does in the noisy one. The one substantive positive is the opposite of the pre-registered guess:
**observation noise acts as a regulariser that makes position *linearly* readable**, and makes the state markedly more
canonical under an MLP fiber map.

## Results

**§1 Predictive quality (Fig 1).** Each cell against its own baselines:
| cell | obs / pos noise | next-step RMSE | copy-prev | own noise floor |
|---|---|---|---|---|
| no noise | 0.0 / 0.00 | 0.0482 | 0.0940 | 0.0000 |
| world noise only | 0.0 / 0.04 | 0.0731 | 0.1028 | 0.0000 |
| sensing noise only | 0.2 / 0.00 | 0.0893 | 0.2109 | 0.1539 |
| both (repo standard) | 0.2 / 0.04 | 0.1041 | 0.2139 | 0.1539 |

**§2 Recoverability — both pre-registered predictions refuted (Fig 2, Table 1).**
| | no noise | world only | sensing only | both |
|---|---|---|---|---|
| position R² (linear) | 0.596 | 0.693 | **0.819** | **0.828** |
| velocity R² (linear) | 0.451 | 0.451 | 0.469 | 0.471 |

1. **Removing world noise does NOT buy velocity readability** (Δ = −0.002). I predicted it would, since velocity
   becomes exactly constant and therefore perfectly inferable. It does not — velocity readability is essentially
   invariant to *both* noise sources (0.451–0.471 across the whole 2×2).
2. **Observation noise is what makes position linearly readable** (0.596 → 0.819 with sensing noise on, world noise
   off; +0.22). This is the clearest positive in the ablation: **sensing noise acts as a regulariser** that pushes the
   position code into a linear one. A model with a perfect view has no reason to build one.

**§3 Canonicality — the two estimators disagree in sign, so both are reported (Fig 2b).**
| | no noise | world only | sensing only | both |
|---|---|---|---|---|
| fiber residual (linear) | 0.814 | 0.842 | 0.890 | 0.868 |
| fiber residual (MLP) | 0.605 | 0.670 | **0.469** | **0.477** |
Turning sensing noise off (world noise held on) moves the linear residual by −0.026 but the **MLP residual by +0.193**.
Under the MLP estimator — the one able to follow a curved embedding — **sensing noise makes the state markedly more
canonical**. Read as: observation noise reorganises the code into something an MLP can invert. Reporting only the
linear estimator here would have produced the opposite headline, which is worth remembering.

**§4 Editability — the negative is noise-independent (Fig 3, Fig 5, Table 2), on the canonical metric set.**
Edit Index: **+1** = the output *is* the edited world, **−1** = the unedited world, **0** = equidistant from both.
Structural editors are credited only when they pass the **fidelity guard** (GT-traj RMSE ≤ unsteered's).
| | no noise | world only | sensing only | both |
|---|---|---|---|---|
| unsteered (do nothing) | −0.85 | −0.80 | −0.70 | −0.68 |
| best structural editor passing the guard | −0.67 | −0.67 | −0.64 | −0.63 |
| decoder-gradient **oracle** | **+0.96** | **+0.91** | **+0.97** | **+0.94** |

In **every** cell — including the fully deterministic, perfectly-sensed world — probe-directed editors stay on the
unedited side of the index while the oracle crosses to the edited side on the same model and decoder.
**Stochasticity is not what stands between us and a grabbable object handle.**
*(The MLP-probe gradient reaches ≈ −0.12 in the noise-free cells, but fails the fidelity guard: its collateral RMSE
rises 0.116 → 0.756 and its fidelity ratio is 1.45. It is degrading the rollout toward "neither world", not
relocating an object — the failure mode the Edit Index was designed to catch.)*

**A new incidental result: belief inertia is governed by SENSING noise, not by the editors.** The oracle observation
(one frame of genuine teleport evidence, no editing at all) updates the belief very differently across cells:
| | no noise | world only | sensing only | both |
|---|---|---|---|---|
| oracle observation Edit Index | +0.06 | **+0.54** | **−0.40** | −0.08 |
With clean observations the model largely accepts one frame of teleport evidence; with sensing noise it mostly
does not. Suggestively, the *world-noise-only* cell accepts it furthest of all — a model trained where objects
genuinely jitter appears to hold a looser prior over motion and so accepts a large jump more readily. Worth a
dedicated test before treating that ordering as established (n = 1 seed per cell).

## Caveats / open
- One seed per cell; cells are matched statistically (64 random edits from matched generators), not sample-by-sample.
- `H=256`, GRU only.
- Position noise was varied only between 0.00 and 0.04 (the repo default); no intermediate or larger values.

## Open questions for Sevan
- Artifact or signal? The "observation noise is a linearising regulariser" result is the substantive positive and is
  clean (+0.22 position R², replicated across both position-noise settings).
- Does the noise-independence of the §4 negative get folded into `findings/editability.md`, or held with the
  hidden-size result as one combined "the negative is robust to capacity and to stochasticity" entry?
