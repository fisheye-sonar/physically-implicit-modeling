# flip_ablation — Othello without the flipping rule (runs/flip_ablation/L-oth-noflip-20m)

**Question.** Is Othello's editability a property of the flip dynamics (the board is a
non-trivial function of the move history) or of the categorical, board-like state itself?
`oth-noflip` keeps everything — legality by enclosure, passes, game end, uniform random
moves, the 20M-game index law, the model and recipe of `L-oth-20m` — and only stops
recolouring enclosed discs. The pilot (`scores/pilot_noflip.json`, 5k games/rule set) shows
the consequence: no passes ever, so a disc's colour is exactly the parity of the offset at
which its square was played. The board is syntactic; decodability will be ≈ 1 by
construction (a right-aligned linear observation probe reads it exactly); editability is
the experiment.

**What is canonical here.** The instance (`datasets/othello/oth-noflip/`), its synthesised
1001-case bench (`scripts/make_othello_edits.py`: Li's recipe and prefix-length mix from the
noflip TEST split), the run (`scripts/drivers/oth_noflip.sh`), scores via master_eval, rows
in the master tables. This folder holds the pilot and the like-for-like control: `L-oth-20m`
scored on a same-recipe synthesised FLIP bench, so the two runs are compared on benches of
identical construction (Li's shipped cases vs synthesised is otherwise a confound).

**Status.** 2026-09-06 evening: chain launched (unit `oth_noflip`): corpus → cases → train
(~20 h) → wait for the instance-aware scorer → score + tables. Control bench: DONE (below). Finding to follow in `research/findings/flip-ablation.md`.

## Control: L-oth-20m on a synthesised FLIP bench (2026-09-06, `scores/summary_control.md`)

1001 cases from the oth-uniform test split by the same recipe and prefix-length mix as
oth-noflip's bench; L-oth-20m's canonical best arms re-read on it:

| | synthesised flip bench | Li's shipped 1001 |
|---|---|---|
| unedited EI | −0.703 | −0.713 |
| PI pt4·α3 | +0.596 / 0.25 | +0.608 / 0.24 |
| ND pt4·α0.35 | +0.543 / 0.27 | +0.622 / 0.23 |
| GS pt0·α0.05 | +0.612 / 0.22 | +0.647 / 0.21 |

The bench construction moves the Edit Index by at most 0.08 (ND) and the floor by 0.01, so
the noflip run's synthesised bench is comparable to the shipped one; quote both when the
two runs are compared.
