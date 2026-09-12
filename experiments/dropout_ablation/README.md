# dropout_ablation — is the redundancy of the colour code a regulariser effect or the rule's? (2026-09-11)

**Question.** Transformer-L is minGPT with dropout 0.1 on the embedding, attention and residual
paths. Dropout on the residual stream rewards writing a variable in several directions, so some
of every model's orthogonal colour copies (`adjacent-flip-ablation.md` §INLP: ~30 standard,
~50 flip, 85–232 adjacent) may be training noise rather than computation. The materialisation
theory predicts a MATERIALISED variable's copy count falls sharply without dropout while a
FUSED variable's does not — and, higher stakes, that oth-adjacent stays inert to single-probe
edits without dropout. If instead it becomes canonically editable, the adjacency negative was a
dropout artefact.

**Design.** `L-oth-adjacent-nodrop-390k`: identical to `adjacency_ablation/L-oth-adjacent-20m`
except `--dropout 0` and 390k steps (Othello editability is at ~95% of its 780k value by then,
`training-curve.md`); trained RESUMABLE (`scripts/train.py --resume`, added 2026-09-11) so it
can be extended to 780k. Driver `drivers/oth_adjacent_nodrop.sh` (waits for the oth-adjacent
data build on the remote, trains, scores). Then the same INLP + K-copy edits
(`experiments/adjacent_flip_ablation/scripts/inlp_othello.py`) and canonical scores, against the
dropout run. Second run when a GPU frees: standard Othello without dropout (the positive
prediction); third: the flip model.

**Status.** Launched 2026-09-11 evening on wsl-sevan (unit `oth_adjacent_nodrop`).
