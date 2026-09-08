# adjacency_ablation — does editability return when colour is USED without the enclosure geometry? (2026-09-08)

**Question.** oth-noflip removed recolouring and, by the checkerboard theorem, made colour
causally irrelevant (not editable). oth-adjacent keeps "no recolouring" but changes the
placement rule to "touch one of your own discs (8-neighbourhood)", so legality depends on
colour again without any enclosure geometry. Prediction: editability comes back.

**Instance / run.** `datasets/othello/oth-adjacent/instance.json`; run
`runs/adjacency_ablation/L-oth-adjacent-20m` (Transformer-L tokens, 20M games, 780k steps, the
matched recipe); driver `scripts/drivers/oth_adjacent.sh`; logs `logs/adjacency_ablation/L-oth-adjacent-20m/`.
Canonical scoring by `master_eval.ipynb` (instance-aware via `corpus.rules_of`).

- `scripts/pilot_adjacent.py` — the pre-corpus gate (the no-flip lesson): recolourings that
  change the legal set >= 30%, no colour/parity theorem, diverse terminal boards. Result
  `scores/pilot_adjacent.json`: 68% change (mean differing set 2.4 squares), parity match
  45%, 19,999/20,000 distinct terminal boards, 1.13 passes/game, Bayes CE 2.43. PASS.
