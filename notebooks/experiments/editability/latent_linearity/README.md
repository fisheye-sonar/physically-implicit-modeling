# latent_linearity/ — the geometry of the edits that actually work

**Question:** the edits that succeed in this repo hand us ground-truth latent displacements. Do they agree on a
direction, and does that agreement imply structure in the latent representation?

This thread is about the **properties of the latent space that support editing of any kind** — not about editors.
No probe-derived write appears anywhere in it.

## Contents

| file | what it holds |
|---|---|
| [`latent_edit_directions.ipynb`](latent_edit_directions.ipynb) | the notebook: the four mechanisms, four architectures, and the alignment between their Δh |
| [`random_baseline.ipynb`](random_baseline.ipynb) | the **random-init control** on latent object-composition — is `[move A] + [move B] ≈ [move both]` learned, or does an untrained network do it too? |
| `composition_lib.py` | the composition measurement: edit set, the four counterfactual renders, the observation ceiling, and the metrics |
| [`LATENT_LINEARITY_RUNS.md`](LATENT_LINEARITY_RUNS.md) | the run registry — which checkpoint, which state object, and why that one |
| `edit_directions.py` | the one implementation of the mechanisms and the geometry metrics |
| `figures.py` | figure builders, built to hold N models and N mechanisms |

Figures are exported to `runs/latent_linearity/figures/` (gitignored).

## The four mechanisms

1. **Counterfactual Overwriting** — rewrite the whole history so the object always travelled to the target.
2. **Freeze-time Interp. TF @8** — append 8 rendered frames moving it there with the world frozen.
3. **Action Interface** — issue the teleport through an action channel the model was trained on.
4. **First Obs. TF** — show one post-edit observation.

1 and 2 apply to any model, so they carry Part 1 (all four architectures). 3 and 4 need a training distribution
that contained teleports, which in this repo means **GRUs only** — see the registry for the audit.

## What it currently says (2026-08-19; composition control added 2026-08-21)

- The two oracles agree on the displacement in **every** architecture: cos +0.59 … +0.91 (25°–54°), 4.0–5.5×
  the shuffled-pair control. Replicates `delta_h_analysis`'s GRU/RSSM numbers on an independent construction.
- On the action-conditioned GRU **all four mechanisms land**, and the trained action channel is the *closest*
  match to the counterfactual oracle of any pair measured: **+0.872, 29°**.
- Whether a single uncued post-edit frame persists is a fact about the training distribution:
  −0.00 (never saw a teleport) → +0.22 (teleports always cued by an action) → +0.53 (teleports seen uncued).
- The agreement brings no shared edit axis (cross-episode cosine ≈ 0 everywhere) and the direction stays at or
  below chance visibility to a linear position probe — except in the latent DiT's 64-d code, at 1.17× chance.

- **Latent object-composition is mostly architectural** (`random_baseline.ipynb`, 2026-08-21). By
  composed cosine a randomly initialised network of the identical config matches the trained one
  (+0.890 vs +0.904 linear; +0.853 vs +0.835 nonlinear — the untrained net is *higher*). Read
  against the **renderer's own** non-additivity (the two objects share rays), the trained *linear*
  model tracks that ceiling to within ±0.05 at every displacement scale while random init misses it
  by 0.03–0.10. `delta_h_analysis` §7's strong reading is dead; a ceiling-tracking claim survives.

Full numbers, caveats, and what would falsify them: each notebook's `Current results` /
`Summary` block.

## Reproducing

```bash
cd notebooks/experiments/editability/latent_linearity
jupyter nbconvert --to notebook --execute --inplace latent_edit_directions.ipynb   # ~35 s on one GPU
jupyter nbconvert --to notebook --execute --inplace random_baseline.ipynb          # ~15 s on one GPU
```

The notebook adopts the repo root as its working directory in cell [1], because
`scripts/eval_action_sweep.py` resolves `runs/` and `datasets/` relative to the process working directory
exactly as it does from the command line.
