# Direction: Encoder-space editing — is the encoder output the writable interface the hidden state isn't?

**Tag:** `[in-frame]` · **Sub-question:** 3 (editability) + 2 (identifiability) · **Status:** active (2026-07-30) ·
**Complexity:** medium (no retraining; new edit interface + probe surface) · **Model:** GRU only.

Origin: Michael's controls, 2026-07-30. Companion to `multistep-steering.md`, whose freeze-time result this
direction tries to reproduce *without the simulator*.

## The premise (Michael's)

In the GRU every fact about the world reaches the latent through exactly one channel:

```
x_t = relu(W_enc · obs_t + b_enc)      ∈ R^H     ← the ONLY route from world to state
h_t = GRUCell(x_t, h_{t-1})
ô_{t+1} = W_dec · h_t + b_dec
```

Every §4 editor to date writes to `h`. But `h` is the *accumulated* state — a fixed point of the recurrence,
which the dynamics can and demonstrably do reject. `x` is different: it is the model's **input port**, the
representation the recurrence was trained to accept on every single step. A write to `x` is in-distribution by
construction in a way a write to `h` never is.

So: **probe `x`, then edit `x`.**

## Why this is the right follow-up to freeze-time

`multistep_steering` §1b found the one edit mechanism that works: **freeze-time teacher forcing** — freeze the
world, render the edited object interpolating pre→target over `N` frames, teacher-force those frames, unfreeze.
Lands the edit, clears the ghost, replicates on RSSM.

But it works by feeding **externally rendered observations** — it needs the ground-truth simulator's renderer to
manufacture the evidence. That is an oracle, not an edit interface. Interleaved latent steering (§1a), which uses
only the model's *own* decoded observations, fails.

Encoder-space injection sits exactly between them: it uses the model's own machinery (no renderer) but writes at
the **port where evidence normally arrives**, and it can be spread over `N` steps the same way freeze-time is.
The question is whether the freeze-time win survives when the renderer is replaced by a latent write.

## Hypotheses (state before running)

1. **Position is linearly readable from `x_t`, velocity is not.** `x_t` is an instantaneous function of `obs_t`
   alone — it has no memory — so velocity (which needs two frames) should be near-unrecoverable from `x` while
   being recoverable from `h`. A clean, cheap prediction that also *explains* any edit asymmetry we find.
2. **Encoder-space editing beats hidden-state editing.** The recurrence accepts `x`-writes it rejects as
   `h`-writes: ghost ratio drops below the ~1.0 floor that every structural `h`-editor is stuck at.
3. **Multi-step beats one-shot in encoder space**, mirroring the freeze-time `N`-sweep, and the intermediate
   decoded observations show a *coherent moving object*, not a cross-fade between two ghosts.
   *If instead the intermediates cross-fade — old object dimming while a new one brightens — that is the
   informative negative: the model has no object to move, only intensities to blend.*

## Design

**Model / data.** GRU `H=256` trained on `datasets/4_fixed_refl_inview` (`runs/controls/H256`; obs noise 0.2,
position noise 0.04). Edits split, `ef=20`, teleport of `edit_object`, `K=15` post-edit rollout steps, `N_EDIT=64`.

**The edit loop (`N` steps, world frozen).** At each step `j = 1..N`:
1. form the base encoder vector from the model's own current belief — `x_base = relu(enc(decode(h)))` — so no
   external render is ever used;
2. edit `x_base` so an `x`-space probe reads the interpolated position
   `p_j = p_pre + (j/N)·(p_target − p_pre)` for the edited object, the held `ef` position for the other;
3. advance the recurrence with the edited vector: `h ← GRUCell(x_edited, h)`;
4. record `decode(h)` — **these are the intermediate observations Sevan asked to see**.

Then unfreeze and free-run `K` steps. `N=1` is the one-shot encoder edit.

**Editors (all applied in `x`-space, against an `x`-space probe / `x`-space PCA bank).** Per
`METRICS_AND_EDITORS.md`: Readout injection (linear pseudoinverse), Global-PCA projection (POCS),
PCA geodesic (local tangent), MLP-probe gradient.

**Brackets (both required — this is what makes a negative interpretable).**
- **Freeze-time render oracle** — identical loop, but `x_j = relu(enc(render(interp_j)))` using the true
  renderer. This is `multistep_steering` §1b re-expressed in encoder space, and it is the **upper bracket**: it
  proves a target `x`-sequence exists that produces the edit. If the oracle works and the probe-directed editors
  do not, the failure is the *reachability of the edit map*, not the representation. (Same logic that made the
  §4 grabbability negative survive Sevan's predictor objection.)
- **Hidden-state one-shot injection** — the master `h`-space baseline, the thing encoder-space editing is
  supposed to beat.
- **Unsteered** and **GT (sim)** columns as always.

## Readouts

> **Metric note (added 2026-07-30, after this brief was written).** The §4 readouts below were pre-registered using
> the old ratio metrics (`reach % of swap`, `ghost ratio`, `selectivity`). Those were **retired** partway through
> this thread — they scored *change* rather than *correctness* and normalised by a model-dependent soft reference.
> The results were recomputed on the **canonical set** (`../../notebooks/experiments/editability/METRICS_AND_EDITORS.md` §4,
> implemented in `scripts/editability_metrics.py`): **Edit Index** in [−1,+1] plus **Target / Ghost / Collateral /
> Edit-frame / GT-traj RMSE** and the **fidelity ratio**. The hypotheses are left as originally written — they are
> the pre-registration — and map over as: "ghost ratio stays ≈ 1.0 for structural editors" ⇒ "the Edit Index stays
> near the unsteered (−1) end for structural editors".

1. **Predictive quality** — training loss curves; free-run RMSE vs rollout step against the repo's standard
   dashed baselines (copy-previous-frame, observation noise floor, random frame) from `pim/eval/baselines.py`.
2. **Recoverability, `x` vs `h`** — position and velocity R², **linear and MLP**, on matched held-out frames.
3. **Canonicality, `x` vs `h`** — fiber residual, **linear and MLP**.
4. **Editability** — per-step RMSE against the time-evolving post-edit GT (`clean_obs[ef+s]`, compared at the
   same step), ghost ratio, reach (% of swap), collateral, selectivity. `N`-sweep for the best editor.
5. **Waterfalls** — canonical `waterfall_grid` spec, plus the **behind-the-scenes** variant showing the `N`
   intermediate decoded observations in the shaded hidden band (as `multistep_steering` Fig 0a / Fig 2b).

## What would count as an answer

- **Encoder-space multi-step editing works** → we have the first *renderer-free* edit interface, and the
  object-handle story changes: the state is writable, just not at `h`. Sends us back to every prior negative to
  re-test at the encoder port.
- **Only the render oracle works** → the sharpest form of the freeze-time caveat: the win was never about
  spreading the edit over time, it was about **supplying true observations**. Editability requires the simulator.
- **Nothing works, including the oracle** → the `N`-step framing is wrong for this model; re-examine alignment.

## Deliverables

`notebooks/experiments/editability/controls/encoder_editing.ipynb`, PNGs to `/tmp/encoder_editing/`, and a dated
`research/scratch/2026-07-..-encoder-space-editing.md`. Registry: `notebooks/experiments/editability/controls/CONTROL_RUNS.md`.
Do NOT edit `findings/` or `RESEARCH.md`. Short, crisp notebook — Sevan's standing preference.
