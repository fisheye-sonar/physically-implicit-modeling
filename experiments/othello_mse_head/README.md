# othello_mse_head — does the objective matter? MSE-on-one-hot vs cross-entropy

**Question.** `L-oth-20m` (cross-entropy on the next token) is editable (GS-mine +0.65,
PI/ND ≈ +0.5); the discworld models (MSE on the next frame) are not. Is any of that the
*objective*? The cleanest test: retrain the Othello model with everything identical except
the loss — MSE regression of the 61 head outputs against the same one-hot next-move target
(`mse_next_move_onehot`, a Brier score). The population minimiser is the same conditional
distribution, the gradient geometry is not.

**The run** — `runs/objective_ablation/L-oth-20m-mse`, canonical (trained by
`scripts/train.py --env othello --arch transformer_l --objective mse_onehot`, scored by
`master_eval.ipynb`, in the master tables). Architecture (8 × 512 GPT, 61-way head), data
(oth-uniform 20M), recipe (780k steps, batch 256, lr 1e-3, wd 1e-4, clip 1, seed 0) are
`L-oth-20m`'s. Driver: `scripts/drivers/oth_mse.sh` (train → score → tables → this
experiment's distribution check). Logs: `logs/objective_ablation/L-oth-20m-mse/`.

**How the head is read.** The checkpoint carries `output_kind="raw"`; every scorer goes
through `data.move_probs(outputs, kind)` and reads the raw outputs as the estimates — no
softmax, no clipping, no renormalisation (minimal change; Edit Index, li_error and fidelity
are distribution-free). `legal_mass` and gate CE assume a distribution and are labelled
by `gates()["output_kind"]` plus `out_sum_mean` / `out_neg_mass_mean` (GOTCHAS 2026-09-04).

**This experiment's own script** — `scripts/distribution_check.py`: on 10k held-out test
games, how distribution-like are the raw outputs (sum, negative mass, min/max, L1 to the
clip-and-renormalised version, argmax legality), the gates under `raw` vs `clipnorm`, and
the run's canonical best arms (PI / ND / GS from `scores.json`) re-scored under both
kinds with the run's cached probes (a cache miss aborts — nothing is refit here). Output:
`scores/distribution_check_L-oth-20m-mse.json` + `scores/summary.md`.

**Status.** DONE 2026-09-05. Chain: training 19 h 46 min (best val Brier 0.013662 at
770k), master_eval 9.5 min, distribution check 5.3 min. Result: as editable as the CE model
(PI +0.68 / ND +0.74 / GS +0.73 vs +0.61 / +0.62 / +0.65), board slightly less decodable
(skill 0.961 vs 0.975), the raw head sums to 1.0001 with 0.04 negative mass that sits on
illegal moves (clipnorm reading: legal mass 0.951, every EI 0.04–0.07 lower, conclusion
unchanged). `scores/summary.md` has the tables; finding in
`research/findings/othello-mse-head.md`. Pipeline smoke on a 60-step model: `scores/_smoke/`.
