# history_editing — is the un-edited part of the latent the *past*?

**Origin:** Sevan, 2026-08-13. **Branch:** `rogerio_controls`.
**Direction brief:** `research/directions/history-editing.md` ·
**Scratch note:** `research/scratch/2026-08-13-history-editing.md`

## The hypothesis

> *The reason our edits are failing is because the extra information in the latent world state, which
> we're not editing (outside the probe's row space), is information pertaining to the previous frames /
> history.*

The first hypothesis in the thread that names a **specific content** for the un-edited complement. Every
prior negative characterises that complement only geometrically (at/below chance in the probe row space;
~35–42% of `h` not a function of `(pos, vel)`; orthogonal to the true edit direction).

## Contents

| file | what it is |
|---|---|
| `gru_history_editing.ipynb` | GRU (`runs/controls/H256`). Lag probes vs a no-stored-history null → what explains the fiber residual → the **Latent history translation** editor with an `n`-sweep, matched against the identical content through the **observation channel** → row-space enrichment → waterfall. 14 code cells, 6 figures. |
| `transformer_history_editing.ipynb` | Transformer (`runs/transformers/W4`, span 13). The (residual point × window position) probe grid → an **all-position, all-layer activation write** with a landing diagnostic, depth/layer sweeps, a re-applied-each-step arm and a matched-norm control, against the same observation-channel twin → waterfall. 10 code cells, 3 figures. |
| `history_tools.py` | The pieces that must be identical across both notebooks: the single `waterfall_grid(...)` implementing the fixed 1-D spec from `CLAUDE.md`, and the lag-probe / subspace numerics. Pipeline logic stays in the notebooks. |

## Headline result

**The premise is right and its implication is wrong.** On the GRU the un-edited complement *is* recent
history — but **observation-shaped** history (past *observations* explain held-out R² ≈ 0.61–0.66 of the
fiber residual; past *positions* explain ≈ **0.00**). Writing a rigidly translated *position* history into
the state does nothing that a matched-norm **random** write does not also do (both −0.585 vs unsteered
−0.670), while the identical content through the observation channel reaches **+0.635**.

On the transformer — where the history *is* the carried state, with one representation per frame, readable
at R² ≈ 0.8 at every window position — the write **lands exactly** (probe readout error 3.289 → **0.000**)
and the prediction still barely moves (**−0.667 → −0.631**, fidelity 1.00), unchanged by writing further
back, at other depths, or re-applying it at every rollout step. The same content through observations:
**+0.681**.

So the discriminating question the brief pre-registered resolves onto its second branch: **the channel is
the barrier, not the content.**

## New code this thread added

* `pim/world_models/transformer/model.py` — `_run`'s `edit` argument now also accepts a **callable**
  `fn(layer_idx, x) -> x` invoked at every residual point, so a caller can write arbitrary positions at
  arbitrary layers (the tuple form wrote the last position only, which cannot express a history edit);
  plus `residual_stack(state)` exposing the `(n_layers+1, B, S, d_model)` probe/write surface.
  Both additive; the default forward path is asserted **bit-identical** in `tests/test_transformer.py`.
* `tests/test_transformer.py` — 6 new tests, including one pinning that **a constant offset is invisible
  to a pre-norm transformer** (LayerNorm's null space) — a live trap for anyone writing an activation edit,
  since it reads as a null result from the editor rather than from the model.

## Conventions this thread follows

Metrics and editor names come from `../METRICS_AND_EDITORS.md`; the §4 scorecard is computed by
`scripts/editability_metrics.py`. Both notebooks report the **same metric set in the same units**, so the
two architectures are comparable. Every claim about the generations ships with an observation-space
waterfall through the one `waterfall_grid(...)` helper.
