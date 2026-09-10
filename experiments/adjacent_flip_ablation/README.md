# adjacent_flip_ablation — adjacency placement WITH recolouring: does editability return when colour is used AND rewritten? (2026-09-09)

**Question.** oth-adjacent made colour causally relevant (a move must touch an own disc) but
kept "no recolouring", and the model was not editable: the probe read a lookup copy of colour
(the parity of the placing move), not the one legality consumes. oth-adjacent-flip keeps the
adjacency placement rule and turns recolouring back on — the placed disc flips the discs it
encloses, exactly as in Othello. Colour is then no longer decodable from the input, so the
model must track it. Prediction: the "used AND not input-decodable" cell is the editable one.

**Status (2026-09-09).** Instance registered; **pilot gate PASSED** (run locally, 20k games,
1.6 min): recolourings change the legal set 70.3% (mean differing set 2.7 squares); colour ==
square parity 0.497 (no parity theorem); 14,986/20,000 distinct terminal boards; flips/game
16.1 (0.27 per move) — the variant is not oth-adjacent; every game 60 moves; 3.9 passes/game
(vs 1.1 on oth-adjacent); one side wiped out in 0.01% of games; Bayes CE 2.296 (oth-adjacent
2.431, oth-uniform 2.011). No corpus or run yet.

⚠ **The WSL remote (i9-14900K) is not yet trusted for this chain.** Its first pilot attempt
crashed inside the board code with nondeterministic type errors that appear only under a
32-worker pool (2 and 8 workers pass; single process passes; the same code is clean here),
and the Windows WHEA log recorded 26 corrected machine-check errors that day ("Translation
Lookaside Buffer Error", Processor Core) — 8 of them in the 10 s in which the WSL VM died
under the test — on microcode 0x11D (BIOS F5, 2023-12) that predates Intel's 13th/14th-gen
stability fixes (0x129 / 0x12B / 0x12F). Silent corruption, not just crashes, is the risk. Do
not generate or train there until the BIOS/microcode is updated and a repeat of the 32-worker
generation runs clean with no new WHEA events.

**Instance / run.** `datasets/othello/oth-adjacent-flip/instance.json`; planned run
`runs/adjacent_flip_ablation/L-oth-adjacent-flip-20m` (Transformer-L tokens, 20M games, 780k
steps, the matched recipe); driver `drivers/oth_adjacent_flip.sh`; logs
`logs/adjacent_flip_ablation/L-oth-adjacent-flip-20m/`. Canonical scoring by
`master_eval.ipynb` (instance-aware via `corpus.rules_of`). Runs on the WSL remote.

- `scripts/pilot_adjacent_flip.py` — the pre-corpus gate: recolourings that change the legal
  set >= 30%, no colour/parity theorem, diverse terminal boards, AND flips/game > 0 (the
  variant is not oth-adjacent). Reports game length, passes, and how often one side is wiped
  out (a wiped-out player can never move again under adjacency). Result → `scores/pilot_adjacent_flip.json`.
