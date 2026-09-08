# Interface ablation: discworld frames as tokens through the Othello model (L-dw-8ray-tok-20m)

**Date** 2026-09-06 · **Instance** `dw-8ray` (unchanged; frames tokenised —
`datasets/discworld/dw-8ray/tokens/`, 422-token vocabulary = 421 realisable 8-ray patterns
+ UNK) · **Run** `runs/interface_ablation/L-dw-8ray-tok-20m` (`TransformerLTokens(422, 39)`:
the Othello architecture, embedding in, softmax + cross-entropy over frames out; 8 × 512,
25.67M params, 780k steps, matched recipe, 20M sequences) · **Driver**
`scripts/drivers/dw_tokens.sh` · **Logs** `logs/interface_ablation/L-dw-8ray-tok-20m/` ·
**Chain** tokenise 0.9 min, train 12 h 09 min, score + tables 15 min · **Experiment**
`experiments/dw_tokens/` (CE by position, n-gram floor).

## Question

`L-oth-20m-mse` showed the objective alone changes nothing on Othello (MSE-on-one-hot is as
editable as CE; `othello-mse-head.md`). This is the other half: same discworld instance,
same data, same stack, only the INTERFACE changes — every frame is one token, the input is
an embedding table instead of `Linear(8, 512)`, the output a distribution over frames trained
with CE. Exactly the Othello setup with a bigger vocabulary. If tokenised dw-8ray becomes
editable, the categorical interface is the lever; if it stays where `L-dw-8ray-20m` is, the
difference is the environment.

**Scoring — the Othello machinery with discworld state** (the user's design, 2026-09-05):
the same regression probes (LIN / MLP-128, full state, cartesian and frustum) on the token
model's residual stream, the same editors (PI z-space, GS; ND n/a), and Othello's scoring
form on the next-FRAME distribution: `edit_index_legal` with the frames the edited and the
unedited world render at the edit frame as the two (singleton) sets — the **frame-set Edit
Index**, +1 = the edited world's frame, −1 = the unedited one; `p_post` = mass on the edited
frame; guard `move_fidelity_ratio` (`pim/environments/discworld/token_bench.py`,
master_eval [3b]). † in the tables. It is the same axis and the same step-0 reading as the
ray-zone index of the regression rows, not the same formula; the ray-zone index on the
expected frame (`zone_edit_index_expected`) rides along as the bridge.

## Numbers (canonical scoring, EVAL_VERSION 2026-09-01.4; the regression run beside it)

| | token model · L-dw-8ray-tok-20m | regression · L-dw-8ray-20m |
|---|---|---|
| val loss (best) | CE 0.4638 @ 780k (still falling) | MSE 0.00575 @ 555k |
| Probe Skill LIN / MLP-128, frustum (best point) | 0.968 (pt 0) / 0.980 (pt 8) | 0.950 / 0.981 |
| random-init floor LIN / MLP, frustum | 0.968 (pt 0) / 0.968 | 0.955 / 0.975 |
| Probe Skill LIN / MLP-128, cartesian | 0.899 (pt 8) / 0.935 | 0.886 / 0.932 |
| random-init floor LIN / MLP, cartesian | 0.876 / 0.885 | 0.883 / 0.919 |
| Edit Index construction | **frame-set** (163 of 192 cases: the other 29 render the same frame in both worlds) | ray-zone (192) |
| unedited Edit Index | −0.755 (p_pre 0.81, p_post 0.02) | −0.888 |
| **PI** best arm · EI / fidelity | pos·pt6·α175 · **+0.004** / 0.75 (p_post 0.03, p_pre 0.03) | pos·pt3·α175 · +0.297 / 1.11 |
| **GS** best arm · EI / fidelity | all·pt0·α0.5 · **−0.097** / 0.78 (p_post 0.04, p_pre 0.18) | pos·pt0·α0.5 · −0.097 / 0.95 |
| zone Edit Index on the expected frame, PI best arm | +0.145 | (that IS the +0.297 column) |

Per-point Probe Skill (frustum): LIN 0.968 0.942 0.897 0.867 0.856 0.861 0.887 0.917 0.942
· MLP 0.968 0.972 0.975 0.976 0.976 0.976 0.977 0.979 0.980. Tripwire violations 0 in both
bases.

## What it says

1. **The interface is not what makes Othello editable either.** With the Othello model,
   the Othello objective and Othello's own scoring construction, tokenised dw-8ray does not
   edit: PI's best arm is +0.004 and GS's is −0.10, against a floor of −0.755. Read with
   `othello-mse-head.md` (Othello stays editable under MSE), neither the loss nor the
   categorical interface explains the gap between the environments. What is left is the
   environment: how its state is structured and how the next-frame computation uses it.
2. **The writes land in probe space and are inert to generation — the discworld signature
   again.** At α = 1 PI moves the probe read-out exactly onto the target (readout error
   0.409 → 0.000) and the next-frame distribution does not move at all (EI −0.76, p_pre
   0.81 at every point). Only writes 60–175× the exact jump (write ratio 10–12, read-out
   71 units off) change the output — by removing the mass from the unedited frame (0.81 →
   0.03) without putting any on the edited one (p_post ≤ 0.05 at every arm). **EI ≈ 0 here
   means mass on neither frame, a destroyed prediction, not a half-landed edit**; the guard
   reads 0.75 only because removing a wrong answer lowers the RMSE. GS does the same more
   gently (L0 α 0.5: p_pre 0.81 → 0.18, p_post 0.04).
3. **Same regime as the regression model, read on its own axis.** The ray-zone index on
   the token model's expected frame gives +0.145 at PI's best arm, next to the regression
   run's +0.297 at the identical α = 175 with fidelity 1.11 — both are the "moved away
   from the unedited world, did not arrive" reading. The frame-set construction is the
   honest one for a token model and it says so more plainly: p_post never rises.
4. **Decodability sits at the floor for both interfaces — and for the token model the
   floor is the frame itself.** LIN skill at point 0 (the embedding) equals the random-init
   floor exactly (0.968 frustum / 0.876 cartesian): a linear read of a one-hot frame is a
   lookup table, and the current 8-ray frame alone explains that much of the position in
   frustum coordinates. Deeper points first LOSE linear decodability (0.856 at point 4)
   and recover to 0.942 at point 8; the MLP rises to 0.980, +0.012 over its floor (the
   regression model: +0.006). Nothing about the token interface made the state more
   available to a probe than the observation already is — the dw-8ray reading from
   `ray-ablation.md` stands.
5. **The model did learn the dynamics, so this is not a failed training.** Its CE 0.464
   sits well below every frame n-gram floor (order 1 → 6: 0.737 → 0.654 on the same test
   split; `experiments/dw_tokens/scores/ngram_floor_dw-8ray.json`) and its top-1 accuracy
   0.860 beats the persistence rate (the frame is unchanged in 84.0% of transitions; the
   n-gram argmax never leaves it). CE falls along the sequence from 0.65 (t = 0) to 0.28
   (t = 38) as the history pins velocity (`ce_by_position_L-dw-8ray-tok-20m.json`). The
   8-ray observation is genuinely ambiguous about sub-ray position, so the Bayes floor is
   not zero and is not known; what is known is that the model integrates far more than six
   frames of history.

## Caveats

- Frame-set and ray-zone Edit Indices share an axis, not a formula; the table marks the
  token row † and the two are not quoted as one number. 29 of 192 cases are dropped
  because the teleport does not change the frame at the edit frame (8 rays, radius 1.0).
- PI's best arm is at the top of the α grid (175), as for the regression run; it is a
  destructive write either way, and extending the grid would not create a landing the
  smaller α do not show.
- One seed; one instance. The probe-space landing at α = 1 with zero output change is the
  strongest single observation and it is the same one the regression models gave.
