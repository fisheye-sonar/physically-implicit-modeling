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

**Status.** DONE 2026-09-07 15:20. Corpus 3 h, train 19 h 21 min (val CE 1.6801, excess +0.0015 over the Bayes floor), score + tables 24 min. Result: NOT editable (PI −0.001 / ND +0.086 / GS +0.020, all at fidelity 2.6–4.0; unedited −0.823) while decodability is 1.000 at every point ≥ 1 — as it is for the random-init model and for a linear read of the right-aligned one-hot history. Finding: `research/findings/flip-ablation.md`.

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

## Correction (2026-09-07, after scoring): the checkerboard theorem

The user asked whether a no-flip game always ends in a checkerboard. It does — the SAME one,
every game: colour == parity of r+c is invariant under the enclosure rule from the standard
opening, and the legal set never depends on colour (60,000/60,000 positions equal an
occupancy-only rule). So the ablation removed not just the flip dynamics but every causal role
of colour; non-editability is by construction and the run is a "decodable ≠ used" control, not a
test of the flip dynamics. The next variant needs a non-checkerboard opening (colour-dependent
legality), pilot-checked BEFORE the 20M corpus. Details: `research/findings/flip-ablation.md`.

## Presence pilot (2026-09-07, `scripts/presence_pilot.py`, `scores/presence_pilot_*.json`)

The user's quicker test: edit PRESENCE (occupied vs empty) instead of colour on the trained
no-flip model. Presence is perfectly readable from the moves AND causally necessary for
legality (checkerboard theorem), so no retraining: the run's cached 3-way probes serve (blank
is one of the classes), PI swaps a tile between blank and its parity-consistent colour, and
the canonical scorecard applies. 400 cases (204 remove / 196 add), Li's length mix.

| no-flip model, PI | EI | fidelity | li_post (unedited 1.96) | remove-only | add-only |
|---|---|---|---|---|---|
| pt2 · α2 | +0.042 | 0.61 | 0.57 | −0.14 | +0.23 |
| **pt2 · α3** | **+0.296** | **0.67** | 0.75 | +0.18 | +0.41 |
| pt2 · α5 | +0.193 | 1.50 | 2.89 | +0.10 | +0.29 |
| pt4 · α5 | −0.074 | 1.29 | 1.88 | +0.04 | −0.20 |
| pt6 · α5 | −0.027 | 1.98 | 1.94 | +0.13 | −0.19 |
| unedited | −0.790 | 1.00 | 1.96 | | |

Presence edits partially LAND (fidelity < 1 at the best arm: the output moved toward the
edited world, not into destruction), best early (point 2), adding a disc easier than removing
one; colour edits on the same model sit at the floor until they destroy (−0.001 at fidelity
3.25). Controls below (flip model; random-init no-flip model).

Controls (same 400-case recipe, each model's own cached probes):

| model | presence decodable (LIN) | unedited EI | best PI presence arm | colour edits on the same model |
|---|---|---|---|---|
| no-flip, trained | 1.000 | −0.790 | pt2·α3 · **+0.296** / fid 0.67 | −0.001 / fid 3.25 (canonical) |
| flip, trained (L-oth-20m) | 1.000 | −0.695 | pt4·α1 · **+0.233** / fid 0.53 (add-only +0.37) | +0.608 / fid 0.24 (canonical) |
| no-flip, random init | 1.000 | +0.001 (output is noise: legal mass 0.12) | every arm +0.001 / fid 1.00 | — |

Reading: within ONE model (no-flip), two variables both linearly decodable at 1.000 from
random init onward split cleanly by causal role — colour (never used by the dynamics) is not
editable, presence (what legality depends on) is, partially, with the guard < 1. The flip
model's presence is editable to a similar degree; its colour far more. The random-init
control shows the presence signal is not an artefact of the edit construction. So "decodable
from the input" does not preclude editability; causal use of the probed variable is the
operative condition. Discworld (positions causally necessary, decodable, not editable) is
the case this does NOT explain by itself.

## Dedicated binary presence probe (2026-09-07, `scripts/presence_probe_edit.py`, `scores/presence_probe_*.json`, probes in `probes/`)

A fresh 2-class-per-tile linear probe (occupied / empty; canonical classification fit, held
out by sequence) at points 2, 4, 6 on each model, PI through it (swap the tile's two logits,
re-solve in z-space, the `linear_arm` pinv branch with 2 classes). Read-out landing checked.

| model | presence probe error | best PI presence arm · EI / fid | li_post (unedited) | remove / add EI | colour edits, same model |
|---|---|---|---|---|---|
| **no-flip, trained** | 0.001% | **pt2·α3 · +0.387 / 0.44** | 0.48 (1.96) | +0.15 / **+0.64** | −0.001 / 3.25 |
| flip, trained (L-oth-20m) | 0.000% | pt4·α3 · +0.447 / 0.37 | 0.31 (2.45) | **+0.60** / +0.28 | +0.608 / 0.24 |
| no-flip, random init | 0.000% | +0.001 / 1.00 (all arms) | 12.4 (12.4) | | |

The read-out lands (100%) at every α ≥ 1 on all three models — the write always succeeds in
probe space; only the trained models' OUTPUTS follow it. On the no-flip model presence is
editable at the flip model's level (adding a disc +0.64, cf. the flip model's colour +0.61),
while colour on the same model is not (−0.001, destroyed). Within one model, two variables both
linearly decodable at 1.000 from random init onward — and from the input — separate cleanly by
causal role. Decodability from the input does not preclude editability; causal use of the
probed variable is the operative condition. Discworld (position causally necessary,
decodable, not editable) is the case this leaves open.
