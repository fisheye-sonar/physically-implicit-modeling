# Multi-step / freeze-time editing — does spreading the edit over frames beat the one-shot edit?

**Date:** 2026-07-16 · **Direction:** `research/directions/multistep-steering.md` (`[in-frame]`, sub-Q3 editability)
**Notebook:** `notebooks/experiments/editability/multistep_steering.ipynb` (ran synchronously, 0 error cells)
**PNGs:** `/tmp/multistep_steering/{fig1_1a_curves,fig2_1a_waterfalls,fig3_1b_Nsweep,fig4_1b_waterfalls,fig5_rssm_Nsweep}.png`
**Models:** GRU `runs/gru/3_dset3_gru_persistentids_inview_400epochs` (primary) + RSSM `runs/rssm/4_dset4_refined_best` (smaller). Data `datasets/4_fixed_refl_inview` edits split, `ef=20`, `N_EDIT=64`, `K=15` rollout. **No retraining.**

→ **FLAG FOR PROMOTION** — clear, consistent signal on both models: *spreading the edit over rendered frames (1b) works; spreading a latent push in closed loop (1a) does not.*

---

## Headline

- **1b Freeze-time teacher forcing = CLEAN WIN.** Rendering the edited object into the scene and teacher-forcing it in over `N` frames (vs a single teleport frame, `N=1`) monotonically **lands the edit** (RMSE→target ↓, ghost ↓) and **improves post-edit dynamics** without collateral blow-up — on **both** GRU and RSSM.
- **1a Interleaved latent steering = GHOST-ONLY TRADE (no clean win).** Closed-loop `push-h → decode → step` reduces the ghost only by dragging **both** objects (collateral explodes, overall obs fidelity gets *worse* than one-shot). The one-shot latent inject itself is essentially **inert** (barely changes the decoded obs) — the master's *readable ≠ controllable* again.
- **The contrast is the finding:** giving the model **observations** to absorb (1b, in-distribution) succeeds; re-injecting **off-manifold latents** (1a) does not. This favors an *observation-space* editing interface over a latent-space one.

## Key numbers (obs-space RMSE, intensity [0,1]; ghost = mean intensity in vacated pre-edit rays)

**GRU 1b freeze-time N-sweep** (N=1 is the teleport baseline):

| N | RMSE→target(0) | RMSE→GT-roll | ghost | edited_persist | unedited_track |
|---|---|---|---|---|---|
| 1 (teleport) | 0.181 | 0.236 | 0.333 | 2.72 | 2.11 |
| 3 | 0.156 | 0.216 | 0.215 | 2.37 | 1.76 |
| **5 (best RMSE→GT)** | 0.138 | **0.211** | 0.167 | 1.95 | 1.63 |
| 8 | 0.120 | 0.218 | 0.140 | 1.67 | 1.26 |
| 15 | **0.103** | 0.232 | **0.123** | **1.55** | 1.32 |
| *unsteered ref* | 0.278 | 0.291 | 0.546 | 3.32 | 1.12 |

- RMSE→target and ghost fall **monotonically** with N. RMSE→GT-roll (full-dynamics fidelity) has a **sweet spot at N≈5** then rises — more frozen frames place the object better but inject more velocity corruption. Best N=5 vs N=1: ΔRMSE→GT **−0.025**, Δghost **−0.166**, ΔRMSE→target −0.043, collateral controlled (+0.09).

**RSSM 1b freeze-time** (same trend, monotonic to N=15 within range): N=1 → best N=15: RMSE→target 0.258→**0.154**, RMSE→GT 0.277→**0.230**, ghost 0.485→**0.130**, edited_persist 3.16→1.80. ΔRMSE→GT **−0.047**, Δghost **−0.354**.
- RSSM det/stoch: position-probe RMSE **full=0.705, det-only=0.742, stoch-only=1.179** → the **deterministic h carries essentially all the position code**; the stochastic latent adds ~nothing to position readout.

**GRU 1a interleaved vs one-shot:**

| method | RMSE→target | RMSE→GT | ghost | collateral@0 | edited_persist |
|---|---|---|---|---|---|
| one-shot inject | 0.274 | 0.290 | 0.538 | 0.000 | 2.06 |
| interleave S32·η0.1 | 0.402 | 0.400 | **0.268** | 2.72 | 4.65 |
| interleave S16·η0.2 +manifold (best RMSE→GT of the interleaved) | 0.316 | 0.324 | 0.464 | 1.62 | 3.14 |

- One-shot is **inert**: ghost 0.538 ≈ unsteered 0.546, RMSE→GT 0.290 ≈ unsteered 0.291 — injecting the readout barely moves the decoded obs (collateral@0 is 0.0 only because `inject_state` sets the readout exactly, not because the obs changed). Interleaving forces obs change and *does* eat the ghost (down to 0.268 at S32) but at the cost of RMSE→GT ↑ and collateral 0→2.7. **Net: worse.** Manifold-projection helps a bit but still loses to one-shot on RMSE→GT.

## Interpretation

- **Freeze-time TF is also a deployable editor**, not just a diagnostic: we render the target scene ourselves (we know the edit), so no oracle GT obs is needed. That makes 1b the practical recommendation for editing this model: spread the teleport over ~3–8 frames.
- The **velocity artifact is real but not fatal**: `vel_err` stays *above* the unsteered baseline for every N (freeze reads as wrong velocity — interpolation velocity ≠ preserved velocity, and the held unedited object reads as zero-velocity), and it's what bends the GRU RMSE→GT curve back up past N≈5. It degrades the *dynamics* after the edit, not the *placement*.
- 1a's failure localizes the master result: the recurrent map rejects off-manifold latent injections even in closed loop; the re-assertion of the unedited object each step is exactly what corrupts it (holding it fixed drags it via the shared latent). Observations are the only reliable write-channel.

## Caveats / shakiness

- **Position-probe RMSE floor ≈ 0.74 sim-units** (linear probe on GRU h). All *position*-derived metrics (edited_persist, collateral, unedited_track, vel_err) carry that noise — treat their absolute values as soft. The **obs-space metrics (RMSE→target, RMSE→GT, ghost) are the trusted signal** and they tell the same story.
- `N_EDIT=64` samples — trends are monotonic and consistent across two architectures, so likely robust, but not a large N. Worth a rerun at N≈256 before promotion to `findings/`.
- Interleaved-rollout time-alignment: settle steps feed the model its own decoded obs, so sim-time is treated as frozen and post-settle rollout step 0 ↔ frame `ef`. Reasonable but an assumption.
- 1a grid is coarse ((S,η) ∈ {(8,.3),(16,.2),(32,.1)} + one manifold variant); a finer/smaller-η sweep might change the *degree* of 1a's failure but not its direction (more steps → more scrambling, monotone).

## Open questions

- Would ramping the **correct post-edit velocity** into the last frozen frame(s) (preview motion) remove the RMSE→GT sweet-spot ceiling and cut `vel_err`? This directly targets the one artifact that bends the curve back up.
- Should `N` **scale with teleport distance** (bigger jump → more frozen frames)? The sweet spot may be per-sample, not global.
- Why does GRU peak at N≈5 while RSSM keeps improving to N=15 — faster velocity inference in the GRU, or just probe/scale differences?
- Can a *hybrid* (freeze-time TF to place, then a light latent nudge to restore velocity) beat pure 1b on RMSE→GT?
