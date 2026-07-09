# Canonical-State Editing — (pos,vel) fiber collapse + obs-driven probe (2026-06-24)

**Direction:** `research/directions/canonical-state-editing.md` (`[reframe]`, sub-Q2+Q3).
**Notebook:** `notebooks/experiments/editability/canonical_state_editing.ipynb` (executed on GPU, RTX 5090).
**PNGs:** `/tmp/canonical_state/fig1..fig5`.
**Model/data:** `runs/gru/3_dset3_gru_persistentids_inview_400epochs/best_model.pt` (val_loss 0.0236), `datasets/4_fixed_refl_inview` (2 obj, fixed refl, constant velocity, dt=1). H=256, states_tf=(10000,39,256). Velocities read DIRECTLY from HDF5 `velocities`, aligned `[:, :-1, :2, :]` (verified constant over time: temporal std 1.3e-8; mean |v| ≈ 0.05).

→ **FLAG FOR PROMOTION** (do not promote to findings/, do not mark direction done, do not edit RESEARCH.md — human's call).

---

## HEADLINE VERDICT

1. **The (pos,vel) decode fiber is NOT collapsed — `h` is non-canonical.** Even a *nonlinear* MLP `g(pos,vel)→h` leaves a **34.7% residual fraction** of `h` (R²=0.86 on `h`); a linear `g` leaves 88%. `h` carries a large component that is **not a function of the world's minimal sufficient statistic**. Adding velocity to position explains only +0.06 more of `h` (MLP). The remaining ~35% is history / non-canonical coordinates.

2. **Completing the target to (pos,vel) does NOT fix the ghost.** The joint 8-dim edit lands the probe readout *exactly* on (pos,vel) (posvel RMSE 0.000) yet **moves the observation essentially zero** (obs change 0.023, %gap-closed 1.4%, ghost ratio 0.99) — **identical to position-only**. Waterfalls: the bright streak stays at the *red ghost* location, never the green target. **This kills the velocity-incompleteness hypothesis: the ghost is non-canonicality / nonlinear-embedding, exactly as the brief predicted.**

3. **"Readable ≠ controllable", localized.** The **obs-objective** (gradient descent on `h` to match the GT obs) DOES move the observation to the target at step 0 (→target render 0.01 / 0.04 vs unsteered 0.28) — but the **probe-objective (Section C) does not move it at all.** Same target, opposite outcomes. That contrast *is* the readable≠controllable result.

4. **Obs-reaching lands wildly off-manifold and NOT on the canonical state, and does not stick.** The obs-driven `h*` has global off-manifold residual **15.7 / 8.7** (real states 1.75) and is **16.7 / 9.7** away from the real canonical state (which is at distance 0 by definition). The obs match reverts within a few rollout steps (→target s0 0.01 → s4 0.27). Pinning velocity with a 5-frame sequence helps a bit (slower revert, ghost s4 0.15 vs 0.31) but is still far off-manifold. **The model can be forced to *render* the target via an alien latent it never visits — it does not reach it through reality's state.**

---

## Section A — (pos,vel) probe + recoverability  [sub-Q2]

Linear probe `h_t → (pos,vel)` (8-dim), masked to both-objects-visible. Per-component R² (vs predict-mean):
- **Position: linear R² ≈ 0.84** (0.77–0.93). MLP lifts pos to **0.96**.
- **Velocity from `h_t` alone: weak.** Linear vel R² mean **0.48** (range 0.19–0.73); MLP barely changes it (0.47). So velocity is *partly* but not cleanly linearly readable from a single `h_t`, and the MLP gain is ~nil → it is **not a hidden nonlinear-but-present** code at single-frame.
- **Velocity is fundamentally a TEMPORAL feature.** Persistence pushed to temporal probes:
  - `dh = h_t − h_{t-1}`: linear vel R² 0.11 (bad), MLP 0.48.
  - 2-frame window `[h_{t-1}, h_t]`: linear 0.51, **MLP 0.76** (per-comp up to 0.91).
  - Best velocity recovery = **MLP on a 2-frame window (R²≈0.76)**, clearly beating single-frame.
- **Finding:** velocity is "weakly/partially linearly readable from `h_t`, substantially better from a 2-frame temporal window (esp. nonlinearly)". This makes sense: a GRU encodes velocity implicitly via the change in state, not as a clean instantaneous coordinate.

Fig 1 — `/tmp/canonical_state/fig1_recoverability.png`.

## Section B — Fiber-collapse metric `h ≈ g(pos,vel)`  [sub-Q2] — THE KEYSTONE

Regress `h ≈ g(input)` over all teacher-forced states; residual fraction `‖h−g‖/‖h‖` and R² on `h`:

| g model | resid frac | R²(h) |
|---|---|---|
| pos (4d) linear | 0.882 | 0.089 |
| pos (4d) MLP | 0.414 | 0.800 |
| pos,vel (8d) linear | 0.877 | 0.100 |
| **pos,vel (8d) MLP** | **0.347** | **0.859** |

- **Residual is large (35%) even for the best nonlinear `g(pos,vel)` → fiber NOT collapsed → `h` is non-canonical** (carries ~35% history/extra structure beyond the sufficient statistic).
- Incremental R²(h) from adding velocity to position: linear +0.012, MLP +0.059 → **velocity is a minor axis of `h`** (consistent with Section A: it's mostly implicit/temporal).
- **Linear→MLP residual drop = 0.47 (pos) / 0.53 (pos,vel) → the (pos,vel)→h embedding is strongly nonlinear/curved.** This is the geometric reason linear/min-norm edits leave the manifold (connects to the geodesic-walk curvature-barrier finding).

Fig 2 — `/tmp/canonical_state/fig2_fiber_collapse.png`.

## Section C — Joint (pos,vel) editing  [sub-Q3]

N=64 edits, edit_frame=20, N_rollout=15. Target = teleported pos + preserved original velocity (read from edits HDF5 `velocities[:, ef]`). Edited-state readouts: pos-only and pos,vel both hit their readouts exactly (RMSE 0.0); manifold-POCS variant trades readout error (0.32) for on-manifold (global resid 0.0).

Obs-space (step 0, direct edit; reference gap unsteered→target = 0.278):

| variant | →target | obs chg | ghost | %gap closed |
|---|---|---|---|---|
| unsteered | 0.278 | 0.000 | 1.00 | 0.0 |
| pos-only | 0.274 | 0.020 | 0.985 | 1.5 |
| **pos,vel** | **0.274** | **0.023** | **0.989** | **1.4** |
| pos,vel manifold | 0.267 | 0.100 | 0.930 | 3.8 |

- **Completing the target to (pos,vel) is indistinguishable from position-only — neither moves the object.** The ghost ratio stays ≈1 (the streak persists at the pre-edit location). Manifold variant moves the obs a little more but mostly by *scrambling* (waterfalls show it injecting noise, not relocating the object).
- **Conclusion: the ghost is NOT a velocity-incompleteness artifact.** Editing the complete sufficient statistic still ghosts → the failure is non-canonicality / curved embedding (Section B), exactly the brief's prediction.

Figs 3/4 — `fig3_C_obs_metrics.png`, `fig4a_C_scans.png`, `fig4b_C_waterfalls.png` (waterfalls show streak parked at red/ghost, not green/target).

## Section D — Observation-driven editing as a structure probe  [sub-Q3]

Adam on `h` minimizing decode/rollout-vs-GT-obs (oracle probe, NOT a deployable editor — uses GT obs as target). cudnn disabled for RNN backward in eval mode (forward identical). K_seq=5.

Endpoint geometry (real off-manifold resid 1.75; canonical at dist 0):

| state | posvel RMSE→GT | glob resid | dist→canon |
|---|---|---|---|
| probe pos,vel (C) | 0.000 | 1.86 | 3.14 |
| obs single | 4.95 | **15.73** | **16.75** |
| obs seq (K=5) | 2.85 | **8.73** | **9.65** |
| canonical (real) | 1.14 | 1.99 | 0.00 |

Stick test (→target render over rollout):

| variant | →tgt s0 | →tgt s4 | ghost s0 | ghost s4 |
|---|---|---|---|---|
| probe pos,vel | 0.274 | 0.297 | 0.989 | 0.995 |
| **obs single** | **0.011** | 0.275 | 0.087 | 0.312 |
| **obs seq** | **0.036** | 0.173 | 0.087 | 0.147 |
| canonical | 0.189 | 0.243 | 0.665 | 0.772 |

- Obs-objective reaches the target obs at step 0 (ghost killed: 0.09) but **lands 15.7 off-manifold and 16.7 from the canonical state** — an alien latent the model never visits.
- **It does not stick:** the obs reverts within ~4 steps (single → s4 0.28). Pinning velocity (sequence) helps — slower revert, ghost s4 0.15 vs 0.31 — but still far off-manifold.
- **Decisive contrast:** obs-objective moves the obs, probe-objective (Section C) does not → readable≠controllable, localized at this edit. The obs readout's null direction (toward the target rendering) exists in `h`-space but is *orthogonal to the probe-controlled subspace* and *off the visited manifold*.
- Note: even the *canonical* real state only gets →tgt to 0.19 (not 0) and has ghost 0.67 — because the GRU's own one-step prediction at a sharp teleport is imperfect; the obs-objective "beats" canonical on the obs metric precisely by going off-manifold (overfitting the single frame).

Fig 5 — `/tmp/canonical_state/fig5_D_obs_driven.png`.

---

## Caveats / open questions

- N=64 edit samples (matches reference notebooks); obs-space numbers are means. Stable across the 3 large-teleport exemplars shown.
- Velocity magnitudes are tiny (|v|≈0.05) so velocity R² is on a small-variance target; vel RMSE ≈ 0.03–0.05 in absolute terms regardless. The *relative* recoverability story (temporal > single-frame) is robust.
- "Canonical state" = teacher-forced `h` on the realized post-edit edits trajectory up to ef. It is the reality-consistent state, but the GRU's *own* rollout from it is imperfect at a hard teleport (val regime), so it's a soft reference, not a perfect oracle.
- The manifold-POCS (pos,vel) edit reaches global resid 0 but scrambles the obs — consistent with the geodesic-walk curvature-barrier result that the on-manifold readout target is unreachable in a fixed neighborhood.
- **Open:** does the ~35% non-canonical residual decompose into (a) recent observation-noise memory vs (b) genuine extra history? A probe of `h` against *past* obs/positions would localize it.
- **Open:** can a *temporal* editor (edit `h_t` AND `h_{t-1}` jointly, or edit through 2 steps) move the obs where the single-frame probe edit fails — since velocity lives in the 2-frame window?
- **Open:** the obs-objective finds an off-manifold latent that renders the target; is there a *constrained* obs-objective (project to manifold each step) that both moves the obs AND sticks? That would be the bridge between Sections C and D.

## One-line takeaway

The GRU hidden state is **predictively sufficient but non-canonical**: ~35% of `h` is not a (nonlinear) function of the world's minimal `(pos,vel)` statistic, velocity is encoded only temporally, and the `(pos,vel)→h` embedding is strongly curved. Consequently **completing the edit target to the full sufficient statistic does NOT fix editing** (ghost persists, object doesn't move), while an unconstrained obs-objective *can* render the target but only by jumping to an off-manifold, non-canonical state that doesn't stick. Readable ≠ controllable — confirmed and localized.
