# othello_gpt/ — the Othello-GPT method, ported to our transformer

Ports the probing and intervention methodology of **Li et al., *Emergent World
Representations: Exploring a Sequence Model Trained on a Synthetic Task*** (ICLR 2023,
[arXiv:2210.13382](https://arxiv.org/abs/2210.13382)) onto this repo's causal transformer world
model, and asks their question of our world: **is the world state a probe reads causally
responsible for what the model generates?**

Their paper is the strongest published claim that a *probe-derived write* to a sequence model's
activations succeeds. Every editor in this thread so far is a probe-derived write, and every one
fails. Running their method unchanged separates **"their method is better than ours"** from
**"their world is different from ours"** — very different consequences for the thread's central
negative.

> ### ⛔ This notebook is one half of a pair — read [`../othello_transfer/`](../othello_transfer/) with it
> This thread runs **their method on our model**. It cannot rule out the third explanation for the
> negative — that **our editor implementation is wrong** — because every editability number in the
> repo comes from that same code, including this one. The mirror thread
> [`../othello_transfer/`](../othello_transfer/) (opened 2026-08-20) runs **our probe and our editor,
> unmodified, on their model**, against their own 1001-case benchmark, and is the positive control
> that settles it. Its Table 4 carries all three columns side by side.
>
> `othello_probe.py` is **shared between the two threads** and is the single implementation of the
> paper's probe and write. It gained a 3-way classification head on 2026-08-20 for the mirror
> thread; the regression path this notebook uses is unchanged.

## Contents

| file | what it is |
|---|---|
| `othello_gpt_probing.ipynb` | **start here** — the Othello-GPT method port, top to bottom |
| `history_rewrite.ipynb` | the follow-up: apply the edit to the **whole observed history** instead of the latent. **This is the arm that works** |
| `othello_probe.py` | the paper's method: probe families, the §4.1 activation-gradient rule, the Figure 2C sequential multi-layer schedule |
| `pipeline.py` | experiment glue — load run, fit probes per residual point, assemble arms, score them |
| `history_edit.py` | the history rewrite: decode positions from the residual stream, translate one object's whole track, re-render, teacher-force |

## The two results side by side

Same model, same 256 episodes, same metrics:

| | Edit Index step 0 | step 14 | fidelity |
|---|---|---|---|
| unsteered (no intervention) | −0.684 | −0.439 | 1.000 |
| **latent write** — the paper's method, ported exactly | −0.538 | −0.428 | 0.994 |
| **history rewrite** — same edit, applied to the observed past | **+0.626** | **+0.351** | **0.674** |

The probe read-out lands in *both* cases. Only the second changes what the model generates, and
only the second persists. The history rewrite uses **no ground truth** — positions come from the
model's own read-out — but it does use the **renderer**, so it answers "will the dynamics honour a
consistent history?" rather than "can we find the right direction in latent space?".

Metrics come from `scripts/editability_metrics.py` and waterfalls from
`pim.figures.waterfall_grid`; neither is re-derived here.

## What is copied exactly, and what deviates

Copied: the probe families (linear vs a **one-hidden-layer** MLP), per-layer probe accuracy, the
update rule `x' ← x − α ∂L(p_θ(x), B')/∂x` on the **activation**, the sequential schedule over
residual points `L_s … L` at the last timestep, the hold-the-rest term with weight `β` (App. G),
and the null-intervention baseline.

Two deviations, both deliberate and both stated in the notebook's opening table:

1. **Regression, not 3-way classification.** Their board is 64 ternary tiles; our world state is
   continuous. Probe *shape* is unchanged; the loss and the reported metric become squared error
   and R².
2. **Held out by whole sequence, not by frame.** They split (activation, label) pairs 8:2 at
   random. Velocity is constant along a trajectory here, so that split leaks the identical label
   into training for every test frame — measured inflation **+0.34 R²** (`research/GOTCHAS.md`,
   2026-08-14). Matching the paper would inflate every probe number.

A third difference is an implementation choice, not a deviation: the activation update uses
**Adam** by default because our residual points differ in scale ~17× and no single raw-space `α`
converges at every depth. The paper explicitly sanctions this (App. G: "robust to different
configurations of optimizer, learning rate α, and number of steps"). The notebook runs the
literal plain-GD rule alongside it and reports where the two differ.

## Runs

No new world models. Uses `W2` / `W4` / `W16` from
[`../transformers/TRANSFORMER_RUNS.md`](../transformers/TRANSFORMER_RUNS.md) — `W16` is primary.
Probe and edit arms are defined in the notebook's own runs-and-arms table.

## Related

- [`../transformers/transformer_world_state.ipynb`](../transformers/transformer_world_state.ipynb)
  — establishes that this architecture has **two** state objects (carried buffer vs recomputed
  residual stream) and that they come apart. That is why "which layer is the world state" is a
  question here at all.
- [`../input_grad_steering/`](../input_grad_steering/) — probe-gradient steering on the **input**
  surface. This directory is the **activation** surface with the paper's exact schedule.
- [`../METRICS_AND_EDITORS.md`](../METRICS_AND_EDITORS.md) — canonical metric definitions.

## Reproducing

```bash
cd notebooks/experiments/editability/othello_gpt
python -m jupyter nbconvert --to notebook --execute --inplace othello_gpt_probing.ipynb
```

Roughly 3 minutes on one GPU: probe fitting dominates (20 probes per model), the interventions
themselves are ~0.2 s each at N = 256.
