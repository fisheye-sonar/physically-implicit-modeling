# probe_corpus_size — does it matter that the regression probes see 30k sequences and the categorical ones 200k? (2026-09-19)

**Question (Sevan).** The paper has to say that the regression probes (and the inverse map) are
fitted on 30k discworld sequences / 20k Othello games while the categorical discworld read-outs
get 200k sequences. Rather than argue it, measure it: refit on more data and see whether
decodability or editability moves. The alternative — moving the canonical recipe to 200k — would
bump the eval version, rescore every run and floor, and needs a streamed regression fit that does
not exist; this control costs a few GPU-hours and changes no canonical number.

**What is already known.** The two canonical recipes are matched in ROWS: 30k sequences × 39
frames = 1.17M, 20k games ≈ 1.18M. `findings/probe-capacity.md` (2026-09-02) fitted LIN and
MLP-128 on ~10M rows in both environments (250k sequences / 170k games, one residual point):
discworld 0.983 / 0.997, Othello 0.975 / 0.976 — the canonical values to three decimals. And
the split-to-split spread of the canonical fit is ±0.0006 (`findings/seed-variance.md`). So
decodability is not data-limited; what was never measured is EDITABILITY through instruments
fitted on more data.

**Method.** The canonical pipeline with the corpus size as the only knob, one corpus per
environment for every size (the large probe split), the run's canonical numbers beside it as the
reference, seed 0, the run's own bench, alpha grids and GS layers.
- `scripts/corpus_size_dw.py` — `L-dw-noiseless-20m`, cartesian: LIN + MLP-128 (200 epochs) and
  the PI / GS sweeps at 30k / 100k / 200k sequences of `probe_250k`; the inverse map + IM / IM-NN
  at 30k / 100k (at 200k its bank and fit hold a point's residuals twice: ~25 GB GPU, ~32 GB RAM).
- `scripts/corpus_size_oth.py` — `L-oth-20m`: the LINEAR grid and the PI / ND sweeps at every
  point at 20k / 60k / 100k games of `probe_large`; the inverse map + IM / IM-NN at 20k / 60k.
  No GS (its MLP grid is ~90 min at 20k games) and no 170k (one harvested point is ~41 GB).
Fits are cached in `probes/` (gitignored), results written after every size to `scores/`.

**Decision rule.** "Flat" = every size within the training-seed SD of the same quantity
(`experiments/paper_ci` ledger; ~0.005 on skill, ~0.01–0.03 on the Edit Index). If a quantity
moves by more than that with corpus size, it is reported as such and the recipe question reopens.

**Run.** Queue jobs `ctrl_corpus_dw` (lab only, ~3.5 h) and `ctrl_corpus_oth` (either host,
~1.5 h) in `experiments/paper_ci/queue/`, priority after the ray family. Smoke: `--smoke`.
Result: to be written into `research/findings/probe-capacity.md` as a dated entry when it lands.

## 2026-09-20 05:00 — Othello control: attempt 1 killed by the memory cap; what was kept

`ctrl_corpus_oth` (4090, unit cap 40 GB) was OOM-killed by its cgroup 4 minutes into the inverse-map fit at 60k games
(journal: `Failed with result 'oom-kill'`, 40.0 G peak). The Othello inverse-map path is DENSE: per residual point it
holds the harvested activations, the masked rows (3.5M × 512 floats = 7 GB at 60k) and their train/test copies, plus
the retrieval bank's copy — it fits at the canonical 20k, not at 60k. Written before the kill (`scores/oth_L-oth-20m.json`
on the 4090, synced back when the retry ends):

| games | skill LIN | PI | ND | IM | IM-NN |
|---|---|---|---|---|---|
| 20k (canonical) | 0.9747 | +0.819 / 0.29 | +0.748 / 0.34 | +0.810 / 0.38 | +0.036 / 2.14 |
| 60k | 0.9752 | +0.819 / 0.28 | +0.746 / 0.34 | — | — |

The retry is redefined to exactly that (`--sizes 20000 60000 --im-max-n 20000`; cache hits) so it cannot meet the cap
again. NOT run: 100k games, and the inverse map above 20k — the inverse map's corpus sensitivity is answered on
discworld (`ctrl_corpus_dw`, 30k vs 100k, whose dense path memmaps). `final_tables` no longer depends on the two
controls (they feed no table).

## 2026-09-20 05:40 — the reconciled definition (supersedes the retry described above)

Two sessions reacted to the 05:00 kill at once; this is the single definition that stands, written in
`experiments/paper_ci/plan.py::control_jobs` (the queue files are generated from it):

| job | host | what | attempts |
|---|---|---|---|
| `ctrl_corpus_oth` | lab | linear grid + PI / ND at 20k / 40k / 60k games (20k and 60k are recorded already and are skipped); inverse map + IM / IM-NN at 20k / 40k | 2 |
| `ctrl_corpus_dw` | lab | LIN + MLP-128 + PI / GS at 30k / 100k sequences; inverse map + IM / IM-NN at 30k / 60k | 2 |
| `ctrl_corpus_dw_200k` | lab | LIN + MLP-128 + PI / GS at 200k sequences (144 GB scratch) — its own job, ONE attempt, after `ctrl_corpus_dw`, so a memory kill at 200k cannot take the smaller sizes with it | 1 |

Kept from the other session's note: the controls are not dependencies of `final_tables` (they feed no table and pull
back only `scores/`), and the Othello inverse map is NOT fitted at 60k games. Added: a 40k-game point for both Othello
parts (the inverse map's corpus sensitivity is then measured on Othello too, at 2× the rows, inside the measured memory:
40.7 GB at 60k games ⇒ ~27 GB at 40k). Both scripts now take `--im-sizes` (the old `--im-max-n` still parses as an
alias) and SKIP every part already recorded in their scores file, so a retry, or a later job adding a size, is free.
The partial results and the fitted-probe cache of the killed run were pulled from the 4090 to the lab at 05:10.
Smokes of both rewritten scripts reproduce the 2026-09-19 smoke numbers exactly.
