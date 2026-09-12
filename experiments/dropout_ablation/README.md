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

**Status (2026-09-12): DONE.** Trained 21:47 → 10:39 PDT on wsl-sevan (12.5 h, 390k steps, best val 2.4357 at 385k; the
dropout run: 2.435). INLP + K-copy edits (`scores/inlp_othello_L-oth-adjacent-nodrop-390k.json`, figure
`outputs/inlp_dropout_vs_nodropout.png`): copies per tile pts 1–8 **283 / 227 / 187 / 161 / 150 / 148 / 146 / 162** vs 232 / 138 / 99 / 88 / 90 / 85 / 83 / 83 with dropout — MORE redundant
without the regulariser; K ≤ 16 edits inert at every point; best guarded +0.450 / fid 0.37 (pt1, K=128) vs
+0.472 / 0.47. The fused code is the adjacency rule's. Write-up: `research/findings/adjacent-flip-ablation.md`
§Dropout ablation; `inlp-redundancy.md` entry. Canonical scores: CE 2.436 (Bayes 2.433), legal mass 0.999; skill LIN 0.979 / MLP 0.981 (dropout run 0.988 / 0.990; observation floor 0.988); unedited -0.655; PI -0.302 / fid 1.07, ND +0.042 / 2.96, GS +0.001 / 7.76 (dropout run: PI −0.05 / 2.31, ND +0.12 / 1.05, GS +0.00 / 5.73); mine_signed skill 0.821 / 0.902, ND -0.029 / 2.66 — canonically inert, as with dropout.
⚠ Lab-GPU note: the INLP script's moment accumulation is now chunked — the desktop holds ~11 GB of the 5090 and the
un-chunked version OOM'd (surfacing as the NVML assert while the driver mismatch awaits a reboot).

**Extension (2026-09-12 12:04 →).** `drivers/oth_adjacent_nodrop_extend.sh`: the 390k resumable state copied into `L-oth-adjacent-nodrop-20m` (without scores/probes) and continued to 780k with `--resume`; the 390k dir remains the scored snapshot. Question for the 780k INLP: does the long, weak tail of colour copies (no-dropout's 150–280 vs dropout's 85–230) prune with more training, or is it the no-dropout regime's steady state?
