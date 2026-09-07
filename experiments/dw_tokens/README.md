# dw_tokens — the interface ablation: discworld frames as tokens through the Othello model

Sub-experiments (one question, follow-ups nested): `bridge/` — the token run read with the
original discworld analysis + waterfalls; `obsfloor/` — one-hot observation-space decodability
floors. This folder holds the run-side analyses (CE by position, n-gram floor).

**Question.** `L-oth-20m-mse` showed the objective alone changes nothing on Othello
(MSE-on-one-hot as editable as CE). The other half: same environment, same data, same
stack, only the INTERFACE changes — dw-8ray frames become tokens of a 422-token vocabulary
(421 realisable 8-ray patterns + UNK), the input is an embedding table instead of
`Linear(8, 512)`, the output a softmax over frames trained with cross-entropy. If
tokenised dw-8ray becomes editable, the categorical interface is the lever; if it stays
where `L-dw-8ray-20m` is (PI +0.30, GS negative), the difference is the environment's
state structure, and the two ablations close both explanations.

**The run** — `runs/interface_ablation/L-dw-8ray-tok-20m` (canonical; `scripts/train.py
--env discworld --repr tokens --instance dw-8ray --arch transformer_l`, 780k steps, the
matched recipe; the run dir carries its `vocab.npz`). Data: `datasets/discworld/dw-8ray/tokens/`
(`scripts/make_discworld_tokens.py`, 0.9 min; every eval/edit frame occurs in training).
Driver `scripts/drivers/dw_tokens.sh` (train → wait for `experiments/dw_tokens/scorer_ready`
→ master_eval → tables); logs `logs/interface_ablation/L-dw-8ray-tok-20m/`.

**How it is scored — the Othello machinery with discworld state.** Probes: the ordinary
discworld regression probes (LIN / MLP-128, full state, cartesian + frustum) on the token
model's residual stream (`bench.fit_probes(encoder=…)`). Editors: PI (z-space) and GS on
those probes, writing the residual at the last context position through
`decode(idx, edit=hook)`, as Othello's arms do; ND n/a as for every discworld run. Score:
the next-frame distribution against the frames the edited / unedited worlds render at the
edit frame — Othello's `edit_index_legal` ("frame-set" construction, reported as
`edit_index`, marked † in the tables), `li_error`, `p_post`, `move_fidelity_ratio`
(`pim/environments/discworld/token_bench.py`; master_eval [3b]).

**Status.** DONE 2026-09-06 09:39. Train 12 h 09 min (val CE 0.4638 @ 780k), score + tables 15 min.
Result: NOT editable — PI +0.004 / GS −0.097 (frame-set EI; unedited −0.755), writes land in probe
space at α=1 with zero output change; decodability at the floor (LIN 0.968 = the frame lookup at
point 0; MLP 0.980 vs floor 0.968). The model learned the dynamics (CE 0.464 vs n-gram floors
0.737–0.654, `scores/ngram_floor_dw-8ray.json`; CE by position `scores/ce_by_position_*.json`).
With the objective ablation, neither loss nor interface explains Othello's editability.
Finding: `research/findings/interface-ablation.md`. Smoke artefacts: `scores/_smoke/`.
