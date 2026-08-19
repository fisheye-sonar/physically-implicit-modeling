# The edits that work agree on a direction — in every architecture, and the learned action channel joins them

**Date:** 2026-08-19 · **Branch:** `latent_linearity` · sub-Q 3 (editability), bearing on sub-Q 1 (geometry) ·
**Author:** orchestrator, at Sevan's direction. **No models trained.**

Notebook `notebooks/experiments/editability/latent_linearity/latent_edit_directions.ipynb`
(+ `edit_directions.py`, `figures.py`, `LATENT_LINEARITY_RUNS.md`, `README.md`).
Figures: `runs/latent_linearity/figures/`. Metrics registered as `METRICS_AND_EDITORS.md` §5.

## The assignment

Extend `delta_h_analysis`'s ground-truth edit-direction analysis (2026-08-03, GRU + RSSM, two oracle mechanisms)
to all four architectures, then ask whether the two **learned** edit pathways — an action channel the model was
trained on, and a single post-edit observation shown to a model that has seen teleports — write the same latent
displacement as the training-free oracles. Explicitly *not* about editors, compositionality, or superposition.

## Setup

Δh is always `edited − matched control` — the same construction with the edit removed, carrying the **same
observation-noise draw**. Independent draws would inject a difference unrelated to the edit large enough to
dominate every cosine. Arms that consume the post-edit frame lead by one, so cross-mechanism comparisons put
everything at `ef+1`; the 1-vs-2 pair is reported at both alignments and the choice does not move it
(GRU +0.808 at `ef`, +0.824 at `ef+1`).

Part 1: dataset 4 `edits`, N=256, K=15, five (checkpoint, state object) pairs. Part 2:
`datasets/15_teleport_eval_single/eval.h5` (teleport-free world, one synthesised edit), N=256, three GRUs.

## Headline 1 — the two oracles agree in every architecture

`cos(Counterfactual Overwriting, Freeze-time Interp. TF @8)` at the edit frame, per episode then averaged:

| state object | cos | angle | enrichment of mean abs-cos over shuffled |
|---|---|---|---|
| DiT (pixel) · residual stream | **+0.910** | 25° | 5.5× |
| GRU · hidden state | **+0.808** | 36° | 5.2× |
| Transformer · residual stream | **+0.806** | 36° | 4.4× |
| Latent DiT · latent window | **+0.667** | 48° | 4.0× |
| RSSM · det+stoch state | **+0.593** | 54° | 4.4× |

Shuffled-pair control +0.00 ± 0.22 in every case. The GRU and RSSM numbers **replicate** `delta_h_analysis`
(+0.799 / +0.569) on an independently constructed Δh (edit-only rather than raw, corrected RSSM state chain), and
the result now holds on three architectures that notebook never touched. Both mechanisms land the edit on every
model (Table 2): counterfactual +0.60 … +0.66, freeze-time +0.10 (RSSM) / +0.52 … +0.65 (everywhere else),
against an unsteered −0.65 … −0.68, fidelity 0.57–0.83. Waterfalls Fig 3a–e confirm it in observation space on
four random episodes.

## Headline 2 — the trained action channel writes the oracle's displacement

`XG_A_H256` is the one checkpoint where all four mechanisms exist, and **all four land**: counterfactual +0.643 ·
freeze-time +0.563 · **action interface +0.645** · first-obs +0.216, unsteered −0.641. The action interface is
also the best at *holding* (+0.473 at step 14 vs the counterfactual's +0.409).

Their Δh mutually align at +0.72 … +0.87, and the tightest pair measured anywhere in the study is
**counterfactual overwrite vs the trained action channel, +0.872 (29°), 5.9× chance.** A pathway learned from
data and an oracle that rewrites the model's history arrive at nearly the same latent displacement.

## Headline 3 — persistence of one uncued frame is a fact about the training distribution

`First Obs. TF` at step 0 / step 14, three GRUs differing only in what they saw in training:

| model | step 0 | step 14 |
|---|---|---|
| never saw a teleport (`H256`) | **−0.002** | −0.095 |
| teleports always cued by an action (`XG_A_H256`) | **+0.216** | +0.162 |
| teleports seen uncued (`XG_C_H256`) | **+0.532** | +0.335 |

Sevan predicted at the outset that the action-conditioned model might not make an **uncued** teleport persist.
It does not — and the same recipe on the same data with the action input removed does. For `XG_A` every training
teleport arrived with an action, so an unexplained jump is evidence of noise; for `XG_C` an unexplained jump was
the only kind there was. Fig 11b shows the observer's first-obs column relocating and holding the object exactly
as the two oracles do.

## What did NOT appear, and it matters

- **No shared "an object moved" axis.** Cross-episode cosine between different edits' Δh: **+0.00 … +0.04** in
  every model and mechanism, chance 0. Replicates `delta_h_analysis` §5 (+0.011) on four more state objects.
- **The direction stays invisible to a linear position probe.** Row-space fraction ÷ chance: GRU 0.73× ·
  transformer 0.49× · pixel DiT 0.14× · RSSM 0.03×. **Exception: the latent DiT's 64-d carried code, 1.17×**
  (1.46× for first-obs) — the only state object above chance, and still nowhere near a handle. Worth a look:
  it is also by far the least linearly readable state (position R² linear 0.220 vs 0.74–0.86 elsewhere), so
  "less readable, more grabbable" is the shape of it, which is the `readable ≠ grabbable` trade in a new place.
- On the action model the action-induced Δh is **0.91× chance** in the probe's row space — the learned pathway's
  write is no more probe-visible than the oracles'.

## Magnitudes

2.4–6.4 × one ordinary dynamics step everywhere. Within the recurrent models Δh is about the size of the whole
state (‖Δh‖/‖h‖ 0.94–1.04; GRU counterfactual 0.95 vs `delta_h_analysis`'s 0.97 ✓); on the residual-stream views
it is 0.21–0.37 of the stream's norm, which is why the dynamics-step scale is the cross-architecture one.

## Two things found in the machinery

1. **`delta_h_analysis`'s `continue_from` double-advances the RSSM.** It round-trips the *prior* (imagined) state
   through `state_from_flat` and then calls `model.step`, which expects the **posterior** at `t−1` — so the
   deterministic core advances twice. This module keeps the posterior chain and applies `imagine` only at
   read-out. **Checked directly, and it is not the cause of the RSSM's weak freeze-time result**: run both paths
   on the same 256 episodes, the states differ by 6.5% of their norm and the freeze-time Edit Index reads
   **+0.097 (corrected) vs +0.091 (legacy)** at step 0, +0.169 vs +0.161 at step 14. Hypothesis raised and
   cleared; the RSSM's freeze-time weakness is real and still unexplained.
2. **The DiT family's `decode` is closer to the frame just consumed than to the next one** (k=−1 0.0985 vs k=0
   0.1088). Not misalignment: the k=0 value reproduces their published next-step RMSE against the clean render
   (0.1080 vs 0.1083 · 0.1088 vs 0.1089), and the frame-to-frame change is 0.1024 — the conditional-mean readout
   simply under-moves. Recorded in `GOTCHAS.md` so the next reader does not re-diagnose it as an off-by-one.

## Caveats

- One checkpoint per architecture, one seed, N=256, one world. Nothing separates architecture from checkpoint.
- The RSSM is the outlier in every measure, and its freeze-time arm barely edits (+0.097), so part of its low
  cosine is the displacement of an edit that did not land. Its 0.03× row-space fraction is a 30× *depletion* and
  deserves its own look.
- Part 2 is GRU-only: no teleport-trained RSSM, transformer or DiT exists (audit in the registry). Mechanisms 3
  and 4 are therefore untested for architecture-independence.
- The agreement in Headline 2 is correlational. A test that would make it causal: corrupt the action channel's
  write while holding its read-out accuracy fixed.

## Open questions for Sevan

- `★` on the three headlines, particularly Headline 2 (the learned pathway lands on the oracle's displacement) —
  it is the first evidence here that "train something that emits Δh" targets a well-defined object.
- Is the latent DiT's above-chance probe visibility worth pulling on? It is the only crack in a four-architecture
  negative and it comes with the least readable state, which is a suggestive pairing.
- Does the training-distribution result (Headline 3) belong in `object-individuation.md` as well as
  `editability.md`? It is about what the *input pathway* buys, which is that file's subject.
