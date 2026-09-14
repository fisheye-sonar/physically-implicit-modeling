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

**Third arm (2026-09-13 00:35 →, queued behind the extension).** `L-oth-adjacent-drop03-390k`: the same
recipe at `--dropout 0.3` for 390k steps (resumable), asked for by Sevan after the no-dropout result — does
MORE dropout prune the colour copies or spread them further, and does single-probe editability move?
Driver `drivers/oth_adjacent_drop03.sh` (unit `oth_adjacent_drop03` on wsl-sevan): stage W waits for the
extension unit to exit, then trains and scores. 200-step smoke of the configuration: `runs/_smoke/L-oth-smoke-drop03`
on the remote (config records `dropout: 0.3`; stopped after the first validation pass to spare the shared GPU).
Then INLP + K-copy edits against the 0 / 0.1 runs.

**Extension DONE (2026-09-13 03:21).** `L-oth-adjacent-nodrop-20m` at 780k: INLP copies per tile **272 / 210 / 158 / 132 / 122 / 119 / 118 / 137**
(390k: 283 / 227 / 187 / 161 / 150 / 148 / 146 / 162; dropout 0.1: 232 / 138 / 99 / 88 / 90 / 85 / 83 / 83) — the tail prunes 10–20 % with doubled
training and stays 1.2–1.6× the dropout run's; K-copy edit curves unchanged (K ≤ 16 inert; best guarded +0.422 / 0.42 at pt 1, K=128); canonical
val 2.4351, skill 0.973 / 0.978, PI −0.206 / 1.22, ND +0.053 / 2.95, GS inert. Scores `scores/inlp_othello_L-oth-adjacent-nodrop-20m.json`; figures
`outputs/inlp_compare_copies_dropout0_390k_vs_780k.png`, `outputs/inlp_compare_r2_dropout0_390k_vs_780k.png` from `scripts/inlp_compare.py`
("label=score.json" per run — the same script will take the dropout-0.3 arm as a fourth line). INLP ran 96 min on the lab 5090 while it shared
the card with a discworld training job (26 min alone for the dropout run).

**Third arm DONE (2026-09-13 17:46; rescored at eval 2026-09-12.1 on the lab box).** Dropout 0.3, same optimum (val 2.4354).
Canonical (guarded, fid ≤ 1.1): **PI +0.172 / 1.04, ND +0.159 / 0.73** — the first oth-adjacent run on which canonical editors land
(dropout 0.1: ND +0.107 / 1.02; dropout 0: nothing). INLP copies 185 / 137 / 115 / 101 / 97 / 90 / 87 / 86, plateau-then-cliff cascades,
best guarded K-copy edit +0.445 / 0.52 (pt 3, K=64). `scripts/canonical_table.py` (four arms from scores.json), `scripts/inlp_compare.py`
(`outputs/inlp_compare_*_dropout_0_01_03.png`), `scripts/covariance_dim.py` (`scores/covariance_dim.json`: participation ratio of the
standardised residual covariance RISES with dropout, 44 → 81 → 84 at point 1 — dropout decorrelates the stream; it does not compress it).
⚠ Eval versions: the remote scores at `2026-09-01.4` (old bench); the lab box rescored every Othello run at `2026-09-12.1` on 2026-09-12
(new 1000-case bench, symmetric-difference headline). Every run trained on the remote is rescored here (`scripts/drivers/score_pending.sh`,
~4 min with pulled probes) before it enters a table; the remote drivers keep the remote's old notebook name `build_full_table.ipynb`.

**Fourth arm IN FLIGHT (2026-09-13 19:41 →).** `L-oth-adjacent-drop07-390k`, `--dropout 0.7`, `drivers/oth_adjacent_drop07.sh`, unit
`oth_adjacent_drop07`; ETA scored on the remote ≈ 09:30 PDT 2026-09-14, then rescore + INLP + covariance here.

**Fourth arm DONE (2026-09-14 09:37; rescored at 2026-09-12.1).** Dropout 0.7: val 2.4403 (0.008 above the floor — the first arm
short of optimal; eval-mode val sat at 2.66–2.75 for 110k steps before converging), skill 0.981 / 0.986, guarded PI −0.168 / 0.95,
ND +0.033 / 0.98 — editability falls back from the 0.3 peak. INLP copies 169 / 126 / 112 / 106 / 98 / 93 / 94 / 93, block-like
plateaus at every depth (half-R² 63 … 32), point-1 R² 0.72 (colour computed later), best guarded K-copy edit +0.391 / 0.69 (pt 4, K=64).
Five-way figures `outputs/inlp_compare_*_dropout_0_01_03_07.png`; `scores/covariance_dim.json` has all five arms (PR at pt 1: 44 / 40 /
81 / 84 / 76). Reading: editability vs dropout is an inverted U peaking at 0.3 — enough read-channel noise prunes the weak additive
tail; too much delays the colour computation and widens the redundant block until single-direction edits lose their handle again.
