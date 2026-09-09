# adjacency_ablation — does editability return when colour is USED without the enclosure geometry? (2026-09-08)

**Question.** oth-noflip removed recolouring and, by the checkerboard theorem, made colour
causally irrelevant (not editable). oth-adjacent keeps "no recolouring" but changes the
placement rule to "touch one of your own discs (8-neighbourhood)", so legality depends on
colour again without any enclosure geometry. Prediction: editability comes back.

**Status (2026-09-09): DONE — it does not.** Bayes-optimal model, legality provably colour-dependent, read-out lands 100%, Edit Index ≈ 0 (PI −0.05 / 2.31, ND +0.12 / 1.05, GS +0.00 / 5.73; extended α no better). Write-up `research/findings/adjacency-ablation.md`; `scripts/extended_alpha.py` → `scores/extended_alpha.json`.

**Instance / run.** `datasets/othello/oth-adjacent/instance.json`; run
`runs/adjacency_ablation/L-oth-adjacent-20m` (Transformer-L tokens, 20M games, 780k steps, the
matched recipe); driver `scripts/drivers/oth_adjacent.sh`; logs `logs/adjacency_ablation/L-oth-adjacent-20m/`.
Canonical scoring by `master_eval.ipynb` (instance-aware via `corpus.rules_of`).

- `scripts/pilot_adjacent.py` — the pre-corpus gate (the no-flip lesson): recolourings that
  change the legal set >= 30%, no colour/parity theorem, diverse terminal boards. Result
  `scores/pilot_adjacent.json`: 68% change (mean differing set 2.4 squares), parity match
  45%, 19,999/20,000 distinct terminal boards, 1.13 passes/game, Bayes CE 2.43. PASS.
