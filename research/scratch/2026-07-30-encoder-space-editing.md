# Encoder-space editing: is the encoder output the writable interface the hidden state isn't?

**Date:** 2026-07-30 · **Branch:** `michael_controls` · **Direction:** `encoder-space-editing` (`[in-frame]`, sub-Q 3+2)
· **Status:** → **FLAG FOR PROMOTION** (nuanced: the *interface* matters and that is new, but the write is still
repainting rather than relocating) · **Origin:** Michael's controls · **Author:** orchestrator.

## The question
In this GRU every fact about the world enters the latent through exactly one channel:
`x_t = relu(W_enc·obs_t + b_enc)` → `h_t = GRUCell(x_t, h_{t-1})`. Every §4 editor to date writes to `h`, the
*accumulated* state, which the recurrence rejects. Michael's premise: `x` is the model's **input port** — the
representation the recurrence was trained to accept on every step — so a write there is in-distribution by
construction. Probe `x`, then edit `x`.

This is also the natural follow-up to `multistep_steering` §1b (freeze-time teacher forcing), the one mechanism that
works — but which works by feeding **externally rendered** observations, i.e. it needs the simulator. The question is
whether the win survives when the renderer is replaced by a latent write at the port.

## Setup / provenance
- Model `runs/controls/H256` (GRU H=256, `datasets/4_fixed_refl_inview`, obs noise 0.2, position noise 0.04,
  400 epochs). No retraining. Registry: `notebooks/experiments/editability/controls/CONTROL_RUNS.md`.
- Notebook `notebooks/experiments/editability/controls/encoder_editing.ipynb` (19 cells, 0 errors, 6 figures).
  PNGs in `/tmp/encoder_editing/`.
- Edit loop: freeze the world for `N` steps; each step take `x_base = relu(enc(decode(h)))` (the model's **own**
  belief re-encoded — no external render), edit it so an `x`-space probe reads the interpolated target, clamp to
  `≥0` (the encoder ends in a relu, so negative entries are unreachable), advance `h ← GRUCell(x_edited, h)`.
  Then unfreeze and free-run `K=15`. `ef=20`, 64 edits.
- Editors in `x`: readout injection, global-PCA projection, PCA geodesic, MLP-probe gradient. Brackets: the
  **freeze-time render oracle** (identical loop, `x_j = relu(enc(render(interp_j)))`) and **hidden-state one-shot
  injection** (the `h`-space baseline). Probes/estimators imported from `scripts/eval_controls.py` so the numbers are
  comparable with the other two controls notebooks.

## Headline
**Where you write genuinely matters — but the encoder port buys a *visible* change, not an object relocation.**
The same linear pseudoinverse edit that is inert on `h` moves the state measurably when applied at `x`. Every
probe-directed encoder write nevertheless stays on the *unedited* side of the Edit Index, and the best one gets
there partly by degrading the rollout. The freeze-time render oracle, through the identical loop and identical
port, crosses to the edited side — so a target encoder vector that moves the object demonstrably exists and the
probe-directed write simply cannot find it.

## Results

**§2/§3 — what lives at the port (Fig 2).** Confirms the pre-registered prediction exactly:
| | encoder output `x` | hidden state `h` |
|---|---|---|
| position R² (linear) | 0.657 | 0.828 |
| velocity R² (linear) | **0.005** | **0.474** |
`x` is an instantaneous function of one frame, so there is no velocity there at all. **You cannot write a velocity
at the encoder port because there is no velocity to write.** 61.7% of `x` entries are exactly 0 (post-relu).

**§4 — editability (Fig 3, Table 3), on the canonical metric set.** Edit Index: **+1** = the output *is* the world
where the edit happened, **−1** = the world where it did not, **0** = equidistant from both (ambiguous or garbage).
| method | space | Edit Index | % of the unsteered→oracle span | fidelity ratio |
|---|---|---|---|---|
| unsteered (no edit) | — | **−0.68** | 0% | 1.00 |
| hidden-state one-shot injection | `h` | **−0.67** | **1%** | 1.00 |
| readout injection (N=8) | `x` | **−0.43** | **21%** | — |
| MLP-probe gradient (N=8), best probe-directed | `x` | **−0.08** | **50%** | **1.15** |
| **freeze-time render oracle (N=8)** | `x` | **+0.52** | **100%** | **0.72** |

Zone RMSEs against ground truth at the edit frame (obs intensity; unsteered is the do-nothing reference):
| | Target | Ghost | Collateral |
|---|---|---|---|
| unsteered | 0.490 | 0.572 | 0.127 |
| best probe-directed (`x`) | 0.373 | 0.426 | **0.335** |
| render oracle (`x`) | **0.185** | **0.126** | 0.108 |

The best probe-directed write improves the target and ghost zones — and **triples the collateral error**
(0.127 → 0.335) while pushing the fidelity ratio to 1.15. That is the signature of repainting: it adds intensity
near the target and subtracts it near the ghost without moving an object, and the other object pays for it.

**Spreading the write helps, monotonically and for both:** the best probe-directed editor goes **−0.37 (N=1) →
−0.05 (N=12)**; the render oracle goes **−0.07 → +0.61** over the same range. Note the oracle at `N=1` is barely
better than doing nothing — the freeze-time win is *entirely* about spreading the evidence over frames.

**Fig 6 (the behind-the-scenes intermediates — Sevan's explicit ask) is the most informative panel.** During the
`N` frozen writes the **render oracle shows one coherent object translating**: a clean diagonal streak walking
from the ghost location to the target. The **probe-directed write shows a diffuse smear** — a new blob brightening
near the target while the old one stays put. That is the cross-fade signature, and it is exactly what the Edit
Index near 0 and the raised collateral error are reporting numerically.

## Reading
Two separable claims:
1. **New and positive:** the *edit interface* is a real variable, not a detail. The identical write mechanism moves
   from **1% to 21%** of the achievable range purely by being applied at the input port instead of the accumulated
   state, and the best probe-directed write reaches **50%**. Every prior §4 negative in this repo was measured only
   at `h`.
2. **Consistent with the standing negative:** it is still not a grabbable object handle. No probe-directed write
   crosses to the edited side of the index; the port carries no velocity, and the writes repaint rather than
   relocate (collateral error nearly triples). The freeze-time win remains dependent on
   *externally-rendered evidence* — the sharpest statement yet of the `multistep_steering` §1b caveat: what made
   freeze-time work was **supplying true observations**, not spreading the edit over time.

## Caveats / open
- One model, one hidden size, GRU only; RSSM has no comparable single input port (its encoder feeds the posterior).
- `N`-sweep run only for the oracle and the best probe-directed editor.
- The encoder-reachability clamp (`≥0`) is applied to all `x` writes; not ablated.
- Untested: editing at `x` with a **velocity-aware** target is impossible by construction, but editing at `x` while
  simultaneously constraining `h` is not — a natural next variant.

## Open questions for Sevan
- Artifact or signal? The interface effect (1% → 21–50% of the achievable span) is the new part; is it worth a `findings/` entry on its own,
  or does it fold into the existing "readable ≠ grabbable" entry as a refinement ("...and the failure is not merely
  where we were writing")?
- Worth re-running the *whole* §4 editor line-up at the encoder port on the models behind `object-individuation`?
