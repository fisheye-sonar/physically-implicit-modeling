# 2026-08-13 — History editing: is the un-edited complement the *past*? → **FLAG FOR PROMOTION**

> `scratch/` is ungated — nothing here is "true" yet. Promotion to `findings/` is Sevan's call.

Direction: `directions/history-editing.md` · Notebooks:
`notebooks/experiments/editability/history_editing/{gru,transformer}_history_editing.ipynb` ·
Branch `rogerio_controls`. Both notebooks executed clean (0 errors; 6 and 3 figures).

## The question

Sevan's hypothesis: *the reason our edits fail is that the extra information in the latent — the part
outside the probe's row space, which we never edit — is information about the previous frames.* This is
the first hypothesis in the thread to name a **content** for the complement rather than describing it
geometrically. Two notebooks: the GRU (history compressed into one vector) and the transformer (history
*is* the carried state, one slot per frame — where the hypothesis has its best shot).

## Design: the one thing that makes it decisive

Every editor in this thread that works supplies a **velocity-consistent translated history** through
**observations**; every editor that fails writes one frame's position into the **latent**. So both
notebooks hold the content, the displacement `δ` and the number of history frames `n` fixed and vary
**only the channel**:

* **latent / activation channel** — pseudoinverse-inject so a probe reads `pos(frame) + δ` at every lag
  (GRU) or every window position × residual point (transformer).
* **observation channel** — teacher-force the model on the same `n+1` frames, *rendered*.

Interpretation was pre-registered in the brief before running.

## Results — GRU (`runs/controls/H256`, dataset 4, N=256 edits, N=1500 probe sequences)

**1. The past is readable, and only nonlinearly.** Linear probes read `pos(t−k)` at R² 0.828 (k=0) →
0.784 (k=20), essentially flat — **but so does the no-stored-history null** (read `(pos,v)` off `h`,
extrapolate back), to within **+0.0008 at every lag**. The **MLP** probe separates: direct **0.883** vs
learned null **0.737** at k=20 (**+0.146**), growing monotonically with lag. The imposed and learned nulls
agree to ≤0.006, so it is not extrapolation inefficiency. Calibration: the direct probe still sits below
the true-`(pos,v)` ceiling (0.991), so `h`'s knowledge of the past does not exceed what perfect knowledge
of the present implies.

**2. The complement is observation content, not past positions.** Linear fiber residual **0.856** of ‖h‖
(MLP 0.467). Regressed on predictors first residualised against the present `(pos,v)`, held-out R²:

| predictor | R² | shuffled control |
|---|---|---|
| past positions, 1 / 2 / 5 / 10 frames | −0.0001 … −0.0007 | ≈ same |
| **obs(t)** alone | **0.659** | −0.007 |
| obs(t−1) / obs(t−2) / obs(t−5) alone | 0.609 / 0.550 / 0.364 | ≈ 0 |
| past observations, 10 frames | 0.636 | −0.060 |

A **decaying trace of recent sensory frames**, not a record of where objects were.

**3. The editor.** Same content, same `n`, two channels (unsteered −0.670):

| n | latent | observation |
|---|---|---|
| 0 | −0.665 | +0.028 |
| 2 | −0.657 | +0.432 |
| 8 | −0.585 | **+0.635** |
| 16 | −0.477 (fid 1.06 ⚠) | **+0.671** (fid 0.61) |

**The decisive control: a matched-norm RANDOM direction scores −0.585, exactly the latent n=8 value.**
Every point of the latent arm's apparent gain is write *size*, not content. Past n=8 it only continues by
degrading (fid 1.36 at ×4). Observation arm: Target RMSE 0.488→0.104, Ghost 0.589→0.107, collateral flat.

**4. Why, structurally.** The stacked lag probe's **effective rank saturates at 8** — the `(pos,vel)` core
— however many lags are stacked (numeric rank 4/8/12/20/36/68; effective 4/7/8/8/8/9). Stacking lags adds
probe *outputs*, not usable *dimensions*. And `Δh_true` from the working observation edit lies in
`row(A_n)` at **0.49–0.60× chance** for every `n` — at or below chance, and *falling*. Demanding an
*inconsistent* history (δ at lag 0 only) forces the near-null trailing directions → a **5.5×‖h‖** write
that collapses the rollout to vertical stripes (fid 2.77).

## Results — Transformer (`runs/transformers/W4`, span 13, same dataset/edits)

**1. Every past frame is readable at every depth.** Probe grid mean linear R²: 0.591 (ℓ=0, encoder port),
**0.769 / 0.797 / 0.773 / 0.765** at ℓ=1…4 — peak mid-stack at ℓ=2, matching the published transformer
result. Flat across window position (only `j=0`, least context, dips). The write has somewhere to go.

**2. The write lands perfectly and the model ignores it.** Driving every one of the 5×13 sites to the
translated-history target moves the probe readout error **3.289 → 0.000 sim units** with a write of only
‖Δr‖/‖r‖ = **0.102**. Edit Index: **−0.667 → −0.631**, **fidelity 1.00**. Not a failed editor, not a
degraded rollout — a *perfectly executed* rewrite of the entire represented history that the model
declines to act on.

**3. Every obvious explanation fails.** history depth n=0→12: −0.643→−0.631, saturates by n=4 · single
residual point: −0.666…−0.647 · layers ≥1/≥2/≥3: −0.633/−0.643/−0.661 · **re-applied at every rollout
step: −0.631, identical** · scaled ×2/×4: −0.604/−0.557 still at fidelity 1.00 · **matched-norm random:
−0.661**. So there is a *small genuine content effect* (0.036 index points vs random's 0.006) — about 3%
of the distance to the observation result.

**4. Same content through observations:** +0.285 (n=0) → **+0.681** (n=8) → +0.677 (n=12), fidelity
0.82→0.60, Target 0.488→0.100, Ghost 0.588→0.101.

## Reading (mine, not established)

**The premise is right; its implication is wrong.** The un-edited complement *is* history — but
**observation-shaped** history, not a decodable record of past positions. The pre-registered fork resolves
onto its second branch: **the channel is the barrier, not the content.**

The transformer makes this as sharp as it can be made. The GRU could be excused — one vector, no per-frame
slots. The transformer *has* the slots, the probe reads each at R² ≈ 0.8, the write into all of them
succeeds **exactly**, and the world does not move. So a frame's representation is not a *handle* on that
frame: the probe finds a direction that **correlates** with position across the distribution, and writing
along it changes what a probe reads without changing what the network computes downstream.

This gives the thread's through-line — *no successful edit is free of dynamics* — a mechanism. The working
editors are not merely "using the dynamics"; they are the only ones writing **in the format the complement
is stored in**. It also explains, in one line, why `orthogonal_edits`' `∫gg′ = 0` argument bites: the
content that has to change is observation-shaped, and a position probe's row space is an 8-dimensional
`(pos, vel)` object that does not span it.

## New code (additive, default-off, tested)

`pim/world_models/transformer/model.py`: `_run`'s `edit` now also accepts a **callable**
`fn(layer_idx, x) -> x` applied at every residual point (the tuple form wrote the **last position only**,
which cannot express a history edit); plus `residual_stack(state)` exposing the
`(n_layers+1, B, S, d_model)` write surface. `tests/test_transformer.py` +6 tests (12 total), suite
**175 green**, ruff clean on all touched files.

**Trap worth knowing, now pinned by a test:** *a constant offset is invisible to a pre-norm transformer* —
adding the same value to every channel is LayerNorm's null space, so a naive "shift the residual stream"
write reads as a null result from the **editor** rather than from the **model**. It cost one debugging
cycle here.

## Follow-on, same day — full row space at H=8 (`full_rowspace_edit/h8_full_rowspace_edit.ipynb`)

Sevan: *using the 8-dim hidden state GRU, train two orthogonal linear probes (INLP style) both reading out
current position, and edit — since their row space will be the whole hidden size.* This attacks the same
negative from the opposite side: not "what is in the complement" but "what if there is no complement".

**The construction works exactly.** Two rank-4 probes, row spaces orthogonal to **7.9e-13**, together
spanning **8 of 8**. Reachable fraction of `Δh_true`: **0.5897** with probe 1 alone (chance 0.7071, so
**0.83× — below chance**, the familiar result) → **1.0000, exactly** with both.

**And nothing improves.** `cos(write, Δh_true)` = **+0.040 (88°)** with both probes, vs −0.011 (91°) with
probe 1 — the write direction is still orthogonal to the answer. At **identical displacement**
(‖Δh‖/‖h‖ = 0.791 = ‖Δh_true‖): full-rank injection **−0.277**, probe-1 injection −0.210, **random
direction −0.172**, unsteered −0.489, observation oracle **+0.529 at fidelity 0.81**. The full-row-space
write is **not better than noise of the same magnitude**, and all three degrade (fidelity 1.12–1.25).

**The literal ask is also degenerate.** The stacked map is square but has condition number **14,373**
(smallest singular value 8.8e-04); demanding both readouts equal the target needs ‖Δh‖/‖h‖ = **2137** →
Target RMSE 660, fidelity **261**, saturated garbage that the Edit Index reports as +0.001 (its
"equidistant *or* garbage" reading). Truncated-SVD variants walk it back without ever helping (k=4:
fidelity 1.58). *Mechanism:* the two probes disagree by **0.923 sim units** on real states (own errors vs
true position 3.02 / 3.20), so demanding exact agreement asks for a state more self-consistent than any the
model visits.

**Structural note worth keeping:** because INLP makes `row(W₁) ⊥ row(W₂)`, the min-norm probe-1 injection
**already** leaves probe 2's readout unchanged. So the second probe adds nothing unless it is asked for a
*different* value — and that is precisely the ill-posed request.

**Reading:** **reachability was never the binding constraint.** The row-space fraction is a valid ceiling
but was never the active one. What a probe gives is a direction that *correlates* with position across the
distribution; inverting it answers "which state would **read** as the target", not "which state **is** the
target". At H=8 probe 1 explains R² 0.17 and probe 2 only 0.09, so the inverse is dominated by what the
probe does not explain. Together with the history result: it is not that the editor cannot **reach** the
right state — a position readout does not **know where** it is.

**Caveat:** H=8 is a much weaker model (unsteered −0.489 vs −0.670; its own oracle only +0.529; visibly
blurry rollouts). All comparisons are internal to it. **The H=256 analogue is not run** — spanning that
state needs 64 rank-4 probes — and is the direct follow-on that would generalise beyond a model small
enough for two probes to exhaust. A ridge-regularised inverse is a cleaner editor than truncation, untested.

## Owed / not done

* One GRU, one transformer (`W4`), one seed, one dataset, **position only**, N=256 edits.
* §2's residual decomposition is **linear**; the MLP fiber residual (0.467) is the tighter target, and a
  nonlinear map from past frames would likely explain more.
* The transformer write uses **linear** probes only. Given §1's finding that the past is stored
  *nonlinearly* in the GRU, an MLP-gradient version of the all-position write is the obvious next arm.
* Observation arms are fed **clean** renders while the models were trained on noisy observations — an
  optimistic bias for the channel that already wins, so it does not threaten the conclusion; a
  noise-matched version is the tightening.
* The `n`-sweep saturation is *predicted* by this world's constant velocity. A world with acceleration or
  bounces would make past positions genuinely independent of `(pos,v)` — the natural follow-on if one
  wants the literal hypothesis tested where it could win.
