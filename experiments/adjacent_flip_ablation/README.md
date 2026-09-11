# adjacent_flip_ablation — adjacency placement WITH recolouring: does editability return when colour is used AND rewritten? (2026-09-09)

**Question.** oth-adjacent made colour causally relevant (a move must touch an own disc) but
kept "no recolouring", and the model was not editable: the probe read a lookup copy of colour
(the parity of the placing move), not the one legality consumes. oth-adjacent-flip keeps the
adjacency placement rule and turns recolouring back on — the placed disc flips the discs it
encloses, exactly as in Othello. Colour is then no longer decodable from the input, so the
model must track it. Prediction: the "used AND not input-decodable" cell is the editable one.

**Status (2026-09-11): DONE — editability returns, partially.** `runs/adjacent_flip_ablation/
L-oth-adjacent-flip-20m` (chain 2026-09-10 00:15 → 2026-09-11 04:22 PT on the WSL remote, unit
`oth_adjacent_flip`, 25.7 h training) is Bayes-optimal (CE 2.301 vs 2.295); trained skill
0.947 / 0.970 sits ABOVE the observation floor 0.888 / 0.914 (colour is computed); ND +0.24 /
fid 0.62, PI +0.17 / 0.85, GS +0.03 / 2.62 vs unedited −0.70 — above oth-adjacent (+0.12 /
−0.05 / 0.00), a third of standard Othello (+0.6). Write-up
`research/findings/adjacent-flip-ablation.md` (`observed`). Pilot gate passed 2026-09-09
(local) and again inside the chain (identical numbers). Next: the two case-level splits
(flipped-tile vs parity-tile cases; differing-set size) and an extended-α + landing sweep.

**Instance / run.** `datasets/othello/oth-adjacent-flip/instance.json`; planned run
`runs/adjacent_flip_ablation/L-oth-adjacent-flip-20m` (Transformer-L tokens, 20M games, 780k
steps, the matched recipe); driver `drivers/oth_adjacent_flip.sh`; logs
`logs/adjacent_flip_ablation/L-oth-adjacent-flip-20m/`. Canonical scoring by
`master_eval.ipynb` (instance-aware via `corpus.rules_of`). Runs on the WSL remote.

- `scripts/pilot_adjacent_flip.py` — the pre-corpus gate: recolourings that change the legal
  set >= 30%, no colour/parity theorem, diverse terminal boards, AND flips/game > 0 (the
  variant is not oth-adjacent). Reports game length, passes, and how often one side is wiped
  out (a wiped-out player can never move again under adjacency). Result → `scores/pilot_adjacent_flip.json`.

## Alignment, Haufe correction, ceiling (2026-09-11) — DONE

Self-contained; reuses `experiments/edit_direction_alignment/scripts/{othello_alignment,haufe_edit,common}.py`.
- `othello_alignment.py --run runs/adjacent_flip_ablation/L-oth-adjacent-flip-20m` →
  `../edit_direction_alignment/scores/othello_alignment_L-oth-adjacent-flip-20m_clean.json`: 42 clean
  cases; best pt 1 rows 0.051 / generic 0.012 / Haufe 0.094 / genH 0.036.
- `scripts/alignment_extras.py --run …` (all three Othello runs) → `scores/alignment_extras_<run>.json`:
  ND-direction cos² raw / Haufe / random, recoloured-vs-lookup split, prefix lengths, ceiling.
- `scripts/honesty_check.py --run …` (all three) → `scores/honesty_check_<run>.json`: true-counterfactual
  ceiling vs the run's canonical best PI / ND arms on the SAME clean cases, with mass on legal_post.
- `scripts/haufe_edit_adjacent_flip.py` → `scores/haufe_edit_adjacent_flip.json`: PI-haufe +0.291 / 0.53, ND-haufe +0.328 / 0.47.
- `scripts/honesty_check_v2.py --run …` (all three) → `scores/honesty_check_v2_<run>.json`: SUPERSEDES
  `honesty_check.py` — its legal-mass filter is toothless on adjacency instances; v2 screens on the
  model's rmse-to-own-uniform (≤ held-out p95) and classifies substitution vs swap (all exact boards are swaps).
Headline: least-aligned Othello model (4× generic raw, 2.6× Haufe). Ceilings on ORDINARY counterfactuals
+0.655 / +0.679 / +0.697 (flip / adjacent / standard), editors below them (ND 55% / 30% / 75% of ceiling
on the same cases) — a first-pass "+0.14 ceiling, dynamic range" claim was wrong and is withdrawn.
Write-up in `research/findings/adjacent-flip-ablation.md` (alignment section, corrected) and
`edit-direction-alignment.md` Result 7; corrected `GOTCHAS.md` entry. Logs: `logs/adjacent_flip_ablation/`.

## Presence edits (2026-09-11) — DONE

`scripts/presence_edit.py --run …` — rule-aware re-implementation of `flip_ablation/scripts/presence_probe_edit.py`
(dedicated 2-class presence probe per point, PI through it, 400 remove/add artificial boards, standard scorecard +
guard); probes under `probes/<run>/` (gitignored). Results (`scores/presence_edit_<run>.json`, `_ext_alpha` for α ≤ 20
at pts 2–3): standard Othello +0.447 / fid 0.37 (pt4 α3; reproduces 2026-09-07); oth-adjacent −0.042 / 4.02, guarded
−0.123 / 1.00; oth-adjacent-flip +0.060 / 1.40, guarded +0.049 / 0.74 (remove +0.19–0.25, add ≈ 0). Presence is not
editable on either adjacency model. Write-up: `research/findings/adjacent-flip-ablation.md` (presence section) and an
addendum in `flip-ablation.md`.
