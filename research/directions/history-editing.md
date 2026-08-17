# Direction: History editing — is the un-edited part of the latent the *past*?

**Status:** **executed 2026-08-13** — both notebooks run clean; results in
`research/scratch/2026-08-13-history-editing.md` (**flagged for promotion**; the artifact-or-signal call and
the `done` mark are Sevan's). · **Tag:** `[in-frame]` ·
**Complexity:** medium (two notebooks, one small additive `pim/` change, no retraining)

> **Outcome against the pre-registered table below: row 4** — *"Rigid History Translation ≈ Pseudoinverse
> Injection while the observation-side window write lands"* → **the channel is the barrier, not the
> content.** With one refinement the brief did not anticipate: the complement **is** history (so the
> hypothesis's premise is right), but it is **observation-shaped** history — past *observations* explain
> held-out R² ≈ 0.61–0.66 of the GRU's fiber residual, past *positions* ≈ **0.00**.
**Thread dir:** `notebooks/experiments/editability/history_editing/`
**Branch:** `rogerio_controls`

## The hypothesis (Sevan's words)

> The reason our edits are failing is because the extra information in the latent world state, which
> we're not editing (outside the row space), is information pertaining to the previous frames /
> history.

This is the first hypothesis in the thread that names a *specific content* for the un-edited
complement. Every prior negative characterises the complement geometrically (at/below chance in the
probe row space, ~35–42% of `h` not a function of `(pos, vel)`, orthogonal to `Δh_true`) without
saying what is in it. "It is the past" is a testable answer.

## Why it is the right hypothesis to test now

Every mechanism in this thread that *works* supplies a **velocity-consistent translated history**
through the observation channel, and every mechanism that *fails* writes a **single frame's**
position into the latent:

| Works | Channel | Score |
|---|---|---|
| Counterfactual state overwrite (GRU) | rendered obs history | **+0.68** |
| DiT counterfactual **window** write, n=4 | 4 history frames | **+0.71** |
| Transformer history overwrite | obs buffer | **+0.63…+0.67** |
| Freeze-time teacher forcing | rendered obs, frozen time | **+0.52** |

| Fails | Channel | Score |
|---|---|---|
| Pseudoinverse (readout) injection | one frame's position, in `h` | **−0.66** (unsteered −0.68) |
| MLP-probe gradient / INLP / PCA variants | one frame's position, in `h` | −0.37…−0.51 |

The DiT result is the sharpest: **Render write @1 = +0.12 but the 4-frame velocity-consistent window
write = +0.71**, and PROGRESS records the conclusion that the +0.12 cap was *conflicting velocity
evidence*, not belief inertia. So "the edit fails because the history disagrees with it" already has
support from the observation side. This direction asks whether the same fix works from the **latent**
side — same content, different channel. That is the discriminating experiment:

- **Latent history edit works** → readout injection was only ever failing because it wrote an
  *inconsistent* history into a state that encodes a trajectory. The negative becomes a statement
  about edit *completeness*, not about latent grabbability.
- **Latent history edit fails while the observation-side version works** → the **channel** is what
  matters, not the content, and the thread's through-line (*no successful edit is free of dynamics*)
  hardens considerably.

## ⚠ The analytical wrinkle that shapes the design — read before building

The world is **ballistic between edits** (`speed_noise = direction_noise = 0`, open boundary), so

```
pos(t − k) = pos(t) − k · dt · v(t)        exactly
```

Two consequences, both load-bearing:

1. **A stacked lag probe cannot have rank 4(n+1).** Fitting `A_n : h → [pos_t, pos_{t−1}, …, pos_{t−n}]`
   by least squares gives block rows `A_k = A_pos − k·dt·A_vel`, so
   `row(A_n) = span(rows A_pos ∪ rows A_vel)` — **rank ≈ 8 for every n ≥ 1**. The "history row space"
   collapses onto the `(pos, vel)` row space. Measure the effective rank; do not assume it.
   Chance level for any subspace fraction therefore does **not** move with `n` past n=1 — report
   **enrichment = fraction / √(rank/H)**, never the raw fraction (`CLAUDE.md`).
2. **"Same δ at every lag" is algebraically a velocity-pinned position injection.**
   `A_k Δh = δ ∀k` ⟺ `A_pos Δh = δ` **and** `A_vel Δh = 0`. This is still a genuinely new editor —
   plain injection lets the velocity readout move as a min-norm side effect — but it must be *named*
   for what it does, and the `n`-sweep will **saturate**, not grow.

**Therefore the hypothesis only has empirical content if `h` carries history beyond what `(pos, vel)`
already implies.** The design below tests exactly that, and if it does not, that is itself the
result: the complement is not the past.

## What to measure

### GRU (`gru_history_editing.ipynb`, `runs/controls/H256`, dataset 4)

1. **Lag readability.** `h_t → pos(t−k)`, k = 0…20, linear + MLP, via
   `pim.extractors.fit_readability_probes` (held out **by sequence**). Report against two baselines
   that decide whether the curve means anything:
   - **ballistic reconstruction** — R² of `pos(t) − k·dt·v(t)` computed from the model's *own*
     decoded `(pos, vel)` readout. This is the null: "the past is inferable, not stored."
   - **shuffled-label floor.**
2. **Beyond-ballistic history.** Probe `h_t → obs(t−k)` (the actual noisy frames) and regress the
   **fiber residual** (`‖h − g(pos,vel)‖/‖h‖`, the canonical ~0.58 for `controls/H256`) onto past
   positions and past observations. This is the direct form of Sevan's hypothesis: *is the un-edited
   complement the past?* Answer it as a fraction of the residual explained.
3. **The editor — Rigid History Translation.** `A_n` stacked lag probe, target `+δ` at every lag,
   where `δ = tgt_pos − pre_pos` for the edited object. `n ∈ {0, 1, 2, 4, 8, 16}`.
   **Controls (each isolates one confound):**
   - `n = 0` = published Pseudoinverse Injection (the inert baseline).
   - **Inconsistent history** — `+δ` at lag 0 only, written through the *same* `A_n` pseudoinverse.
     Separates "consistent translated history" from "larger write subspace".
   - **Matched-norm random direction** — separates the effect from `‖Δh‖`.
   - **Scale sweep** α ∈ {0.5, 1, 2, 4} on the best `n`, gated on the fidelity ratio
     (`metric_corrected_edits` TEST 3: the index can be bought by degradation).
   - Ceilings: Unsteered, Counterfactual overwrite (+0.68), Freeze-time (+0.52).
4. **Row-space geometry.** Fraction of `Δh_true` (from the counterfactual oracle) inside `row(A_n)`,
   as **enrichment over the matched chance level**, with the measured effective rank on the axis.
5. **Waterfalls** over every arm (mandatory — `CLAUDE.md`), one `waterfall_grid(...)` helper.

### Transformer (`transformer_history_editing.ipynb`, `runs/transformers/W4`, span 13)

1. **Probe grid** — linear probe at (residual point ℓ × window position t) → `pos(t)`. Identifies the
   **earliest layer the position is decodable from**, which is where the multi-layer write starts.
   Plus the GRU-comparable lag view: `r[ℓ][last] → pos(last − k)`.
2. **The editor — all-position, all-layer injection.** At every residual point ℓ ≥ ℓ_start and every
   window position t, force the probe readout at t to `pos(t) + δ` (a rigid translation of the whole
   history). Re-applied at each subsequent layer because the residual stream is recomputed — that
   transience is the transformer's structural fact, not a failure.
   **Arms:** last-position/one-layer (= the published inert baseline) · all-positions/one-layer ·
   all-positions/all-layers · all-positions/all-layers **re-applied every rollout step** ·
   sweep over how far back the write reaches (`n_hist = 1, 2, 4, …, span`) ·
   oracle **history overwrite** (observation-level, +0.63) as the ceiling · Unsteered.
3. **Waterfalls** over the arms.

**Required `pim/` change (additive, default-off):** `TransformerModel._run`'s `edit` argument writes
`(layer, vector)` at the **last position only**. Generalise it to accept a **callable**
`fn(layer_idx, x) -> x` so a notebook can write arbitrary positions at arbitrary layers. The existing
tuple path must stay **bit-identical** (pin it with a test).

## Pre-registered interpretation (fixed BEFORE running)

| Outcome | Reading |
|---|---|
| Lag-R² **matches** the ballistic baseline, and the fiber residual is **not** explained by past frames | The complement is **not** the past. Hypothesis refuted; the editor arm becomes a test of velocity-pinning alone. |
| Lag-R² **beats** ballistic and the residual **is** partly explained by past frames | `h` stores genuine history. The editor arm then decides whether that history is *writable*. |
| Rigid History Translation ≫ Pseudoinverse Injection, fidelity ≤ 1 | **Edit completeness was the barrier.** Major positive; re-opens latent editing. |
| Rigid History Translation ≈ Pseudoinverse Injection while the observation-side window write lands | The **channel** is the barrier, not the content. Hardens the thread's central negative. |
| Index moves only with fidelity > 1 | Degradation, not editing (the `tangent_constrained_injection` failure mode). |

## Owed / scope limits to state in the notebooks

- One GRU (`controls/H256`), one transformer (`W4`, span 13), one seed, dataset 4, position only.
- Ballistic world — the lag-probe collapse is a property of *this* world; a world with acceleration
  or bounces would make history genuinely independent of `(pos, vel)`. Worth saying explicitly, since
  it is the obvious follow-on if the hypothesis is refuted for the ballistic reason.
- The transformer activation write is transient by construction; the re-applied-every-step arm is the
  only one that isolates "can the history content change the trajectory" from "the write washes out".
