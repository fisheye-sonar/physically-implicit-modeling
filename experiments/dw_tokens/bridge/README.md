# dw_tokens_bridge — the token run read with the ORIGINAL discworld analysis

**Question.** `interface_ablation/L-dw-8ray-tok-20m` (dw-8ray frames as tokens through the
Othello Transformer-L) is scored canonically the Othello way (frame-set Edit Index,
`findings/interface-ablation.md`). What does it look like on the discworld axis — ray-zone
Edit Index over the 15-step free-run, zone RMSEs, fidelity against clean_obs — and what do
its waterfalls look like?

**How.** `scripts/adapter.py::TokenFrameAdapter` gives the token model the carried-state
surface the canonical discworld bench drives (`state_from_obs`, `decode(edit=)`, `advance`,
`predict_step`, `flat_state`, `rollout_with_edit`): frames in, tokens inside, frames out.
Decisions (2026-09-06): the reported frame is the **expected frame** Σ p_k·frame_k over the
421 real frames; the free-run feeds the **argmax token** back. The edit hook shapes one
prediction and later steps are recomputed unedited, exactly as for the regression
transformer. `scripts/score_bridge.py` then runs the canonical `load_bench` (192 teleports,
EF 20, K 15), `unsteered`, `pinv_arm`, `grad_steer_arm`, `edit_scorecard`, `fidelity_ratio`
with master_eval's own sweep settings, using the run's cached probes (loaded by their exact
key — a miss aborts). A second pass renders the **argmax frame** on the same best arms.
`scripts/waterfalls.py` draws both renderings through the canonical `waterfall_grid`
(unedited / PI best / GS best + signed-error columns, `random_samples` rows).

**Where.** `scores/bridge_L-dw-8ray-tok-20m.json` (the discworld block schema + an
`argmax_rendering` block per basis), `scores/summary.md` (beside the regression run and the
frame-set numbers), `outputs/waterfall_<run>_<basis>_<render>.png`. Driver
`drivers/bridge.sh`; logs `logs/dw_tokens/bridge/`. Nothing here is written into the run's
canonical `scores.json` or the master tables — fold-in is a separate decision.

**Canonical code touched.** None for the bridge itself (the adapter lives here).

**Status.** 2026-09-06 13:40 launched (unit `dw_tok_exp`, bridge then the observation-floor
experiment). Smoke on `runs/_pipeline_smoke/dw-tok-smoke`: `scores/*_smoke.*`,
`outputs/*_smoke.png`.

## Results (2026-09-06, 3.3 min; `scores/summary.md` has the full table)

| frustum basis | unedited EI | PI best · EI / fid | GS best · EI / fid |
|---|---|---|---|
| token model · ray-zone via bridge, expected frame | −0.894 | pos·pt6·α175 · +0.145 / 0.83 | pos·pt0·α0.5 · −0.271 / 0.87 |
| token model · ray-zone via bridge, argmax frame (same arms) | −0.932 | +0.222 / 1.02 | −0.386 / 1.04 |
| token model · frame-set (canonical) | −0.755 | pos·pt6·α175 · +0.004 / 0.75 | all·pt0·α0.5 · −0.097 / 0.78 |
| regression L-dw-8ray-20m · ray-zone (canonical) | −0.888 | pos·pt3·α175 · +0.297 / 1.11 | pos·pt0·α0.5 · −0.097 / 0.95 |

- **On the discworld axis the token model reads like the regression model.** Its unedited
  floor is the regression run's (−0.894 vs −0.888), PI's best is a large-α write at the top
  of the grid (+0.14 vs +0.30, both α = 175, write ratio 10–12) and GS is negative in both.
  The PI sweep has the same shape (α 1–5 nothing, α 20 −0.6, α 60 −0.1, α 175 +0.14).
- **The +0.14 is destruction, not steering, as the zone RMSEs say directly:** the target-zone
  RMSE falls 0.45 → 0.29 but the collateral-zone RMSE (the OTHER object's rays) rises
  0.10 → 0.28 and the ghost zone only drops to 0.32. The edit blurs the whole frame toward
  the mean rather than moving the object. The frame-set reading of the same arm (+0.004,
  p_post 0.03) says the same thing without needing the zone breakdown.
- **Argmax rendering scores higher and degrades more** (+0.22 at fidelity 1.02–1.14): a
  crisp frame that is wrong is further from the ground truth than a blur, and the guard
  reads > 1 accordingly. The expected frame is the fair reading, as decided.
- Waterfalls (`outputs/`): under the expected-frame rendering the free-run after the edit
  frame is a soft, greyed version of the unedited continuation for every arm; the argmax
  rendering shows crisp frames that persist the pre-edit configuration. No column shows the
  object at its teleport target. UNK inputs seen by the adapter: 0.
