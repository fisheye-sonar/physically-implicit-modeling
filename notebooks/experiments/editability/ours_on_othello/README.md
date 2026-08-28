# ours_on_othello — our transformer, our recipe, their world

**The mirror of [`../othello_transfer/`](../othello_transfer/).** That thread put *our probe and
our editor* on *their model*. This one trains *our architecture, with our discworld recipe*, on
*their world*.

| thread | model | world | asks |
|---|---|---|---|
| [`../othello_gpt/`](../othello_gpt/) | ours (`W16`) | ours | does **their method** work in our world? *(no — 2026-08-18)* |
| [`../othello_transfer/`](../othello_transfer/) | theirs | theirs | does **our implementation** work where the answer is published? *(yes — 2026-08-20/21)* |
| **`ours_on_othello/`** (here) | **ours, retrained** | **theirs** | is **our architecture** editable when the world is one where editing is known to work? |
| `directions/othello-architecture-on-discworld.md` | theirs | ours | run A — not run |

Those four are a 2×2. Two cells are filled and disagree, which is why the remaining two matter.
This is the cheap one: ~10 h against run A's ~5 days and 0.54 TB.

## What it can and cannot settle

- **Editable → strong.** Our architecture and training recipe are exonerated, and the discworld
  negative is about the world or our data. Run A becomes largely redundant.
- **Not editable, both gates passed → also informative.** The difference is architectural
  (256 vs 512, 4 vs 8 layers, RoPE vs learned positions, banded vs full), and run A becomes the
  priority with a sharp hypothesis rather than a fishing expedition.
- **It says nothing about whether discworld is the hard part.** Only run A does that.

## The two things that make the test interpretable

**1. The scale ladder.** Training at Li et al.'s 20M games would be 222× the unique sequences
`W16` ever saw, so a success there would be uninterpretable for our setting. Every rung therefore
runs the **same 105,600 optimiser steps** with the identical schedule and varies only the size of
the pool it samples from — 90k → 20M. Anything that moves across that row is data diversity, not
compute. See the registry.

**2. Held-out gates, not training loss.** Arm `M` sees 90k games 300 times, so training loss
reaches wherever memorisation can take it. Every gate is computed at `best_model.pt` on games
from a **disjoint index range**. This mirrors `W16`, which overfits discworld's training loss
(best val at epoch ~40 of 300) and is nevertheless a good predictor out of distribution.

⚠ **Top-1 accuracy has a hard ceiling well below 1.** Their generator draws uniformly from the
legal set, so the Bayes-optimal predictor scores `mean(1/|legal|)`. The Bayes rate is computed on
the same games and reported beside every accuracy; the meaningful quantity is **excess CE over
`bayes_ce`**.

## Files

| file | holds |
|---|---|
| [`ours_on_othello.ipynb`](ours_on_othello.ipynb) | the pipeline and every headline table |
| [`model.py`](model.py) | `OthelloTransformer` — our model with a token embedding in and move logits out |
| [`corpus.py`](corpus.py) | the nested scale ladder, index-seeded, with the disjointness assert |
| [`train.py`](train.py) | `scripts/train_transformer.py`'s recipe with cross-entropy |
| [`evaluate.py`](evaluate.py) | held-out gates, and the plumbing that points the shared pipeline at 5 residual points |
| [`OURS_ON_OTHELLO_RUNS.md`](OURS_ON_OTHELLO_RUNS.md) | the run registry — every code, every configuration, every cited number |

Outputs land in `runs/ours_on_othello/` (`corpus/`, `<run_code>/`, `probe_cache/`, `figures/`,
`results.json`); all gitignored.

## There is no shim

[`../othello_transfer/othello_shim.py`](../othello_transfer/othello_shim.py) exists because
minGPT had to be taught the seven names our editing code calls. Our model **has all seven
natively**, including `_run`'s `fn(layer, x) -> x` edit hook — so `transfer_pipeline`,
`othello_probe`, `othello_data` and `linear_intervention` drive it unchanged. `evaluate.attach()`
does two things and no more: tells them our model has 5 residual points rather than 9, and
repoints the probe cache so a 5-point grid can never be served from the 9-point one.

## Reproducing

```bash
cd notebooks/experiments/editability/ours_on_othello
python corpus.py 20000000                          # ~11 min on 32 cores, 1.2 GB
python train.py --rung M  --window 16               # ~18 min on one 5090
python train.py --rung M  --window 40
jupyter nbconvert --to notebook --execute --inplace ours_on_othello.ipynb
```

The notebook adopts the repo root as its working directory in cell [1], because
`othello_gpt/pipeline` resolves `runs/` and `datasets/` against the process working directory.
