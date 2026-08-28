# Direction: is the action channel's Δh *the* edit, or a correlate of one?

**Tag:** `[in-frame]` · **Sub-question:** 3 (editability) · **Status:** proposed · **Complexity:** medium
(one new training run for the architecture arm; the causal test needs no training).
Proposed by the agent 2026-08-19 out of `scratch/2026-08-19-latent-edit-directions.md`.

## The gap this closes

`latent_linearity` measured that on `XG_A_H256` the **trained action channel** writes a latent displacement
**+0.872 (29°)** from the one the counterfactual-overwrite oracle writes — the tightest agreement in that study.
That is a strong hint that a learned pathway and a history rewrite reach the *same* latent edit. But it is
**correlational**: both mechanisms produce the edited world, so their displacements could agree because each is
the state that renders the target, without the shared component being what does the work.

Two questions follow, and they are independent.

## Experiment A — the causal test (no training)

**Question.** Is the shared component of the two displacements the part that moves the generation?

**Method.** For each episode, decompose the action-induced `Δh_act` against the oracle's `Δh_cf`:
`Δh_act = Δh_∥ + Δh_⊥` (projection onto and orthogonal to `Δh_cf`). Then roll out from four states, all from
the same unsteered `h0` on the same episodes: `h0`, `h0 + Δh_act` (the whole write), `h0 + Δh_∥`,
`h0 + Δh_⊥`. Score all four on the canonical §4 set at step 0 and by step.

**Decision rule, stated in advance.** If `h0 + Δh_∥` recovers ≥ 80% of the full write's Edit Index gain over
unsteered while `h0 + Δh_⊥` recovers ≤ 20%, the shared component *is* the edit. If both recover a substantial
share, the displacement is not the right unit of analysis and the agreement in Headline 2 is a weaker result
than it reads. Report the fidelity ratio for every arm — an arm can gain index by degrading the output.

**Mandatory control.** A random direction of matched norm, and `Δh_cf` itself. Both are already implemented.

**Note.** This *is* a latent write, so it is subject to the whole thread's negative — it may simply fail, which
is itself informative and must be reported against `Δh_cf` (which by construction does work when applied as a
state overwrite).

## Experiment B — architecture-independence of the learned pathways

**Question.** Is "the trained action channel writes the oracle's displacement" a GRU fact?

**Method.** Train **one** teleport-action-conditioned transformer and **one** teleport-action-conditioned RSSM
on `datasets/7_cont_teleport`, matched to the `XG_A` recipe (400 epochs, batch 256, AdamW lr 1e-3, wd 1e-4,
seed 0, 10% val). The action-conditioning port is the same one `ActionGRUContinuousModel` uses: project
`[active, a1, a2]` per object through a small MLP and concatenate to the encoder input, defaulting to zeros so
every protocol method keeps its signature and the whole eval suite runs unchanged. Also train the matched
**observer** (`XG_C`-style, action input removed) for each, since Headline 3 needs the pair.

Then re-run `latent_linearity/latent_edit_directions.ipynb` Part 2 with the new checkpoints added to `SPECS2` —
a data change, not a code change; the figures are built to hold N models.

**What it decides.** Whether mechanisms 3 and 4 join mechanisms 1 and 2 as architecture-independent, and whether
the training-distribution result (−0.00 → +0.22 → +0.53) is a property of recurrence or of world models.

## Bootstrap

- Code: `notebooks/experiments/editability/latent_linearity/edit_directions.py`
  (`build_evidence`, `build_states`, `deltas`, `cosine_report`, `rowspace_report`), `figures.py`,
  `scripts/editability_metrics.py`, `scripts/eval_action_sweep.py` (`xg_data`, `xg_load`).
- Data: `datasets/15_teleport_eval_single/eval.h5` (teleport-free eval, edit synthesised at frame 20),
  `datasets/7_cont_teleport` (training).
- Checkpoints for Experiment A: `runs/action_sweep/XG_A_H256`.
- Registries: `notebooks/experiments/editability/latent_linearity/LATENT_LINEARITY_RUNS.md`,
  `notebooks/experiments/editability/action_hidden_size/ACTION_SWEEP_RUNS.md`.
- Conventions that must not be re-decided: Δh is `edited − matched control` with the **same noise draw**;
  compare only at a common frame alignment; every observation-space error is scored against `clean_obs`.

## Why it is worth doing

The editability thread's constructive endgame is a mechanism that emits Δh. `latent_linearity` showed the target
exists and that one learned pathway reaches it. Experiment A says whether that is the right target; Experiment B
says whether it is a fact about world models or about GRUs. Either answer moves the thread.
