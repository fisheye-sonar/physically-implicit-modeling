# master_eval_refactor — moving the scorer out of the notebook, gated on real runs (2026-09-19)

**Question (Sevan).** `notebooks/master_eval.ipynb` had grown to 1,251 code lines and 37
functions — unreadable by a person in an afternoon and refused outright by the notebook
tools (37.5k tokens against a 25k cap, so two owed one-line edits could not be made). It did no
metric math; what had accumulated was the `scores.json` block schema, the arm-selection rule,
an add-only-what-is-missing engine and the whole decodability-floor pipeline. Can all of that
move into `pim/` with **no number changing**, while a five-day scoring queue depends on it?

**What was done.** A pure move into `pim/scoring/` (REGISTRY §Evaluation has the module map).
The package was ASSEMBLED BY SLICING the cells' text, never retyped; the only edits are the
notebook globals `SETTINGS` / `RUNS` becoming parameters `s` / `runs`, and a `dry_run` guard in
each of the two loops. The notebook keeps cell [2] (SETTINGS + version rule) byte-identical
(sha256 of the cell source `f77f03837515e054` before and after) and is 174 code lines.

Deliberately NOT done (would change what `scores.json["best"]` means inside a replicate set
mid-queue): merging `best_arm` with `tables._best_by`, a fidelity guard on the selection,
re-keying Othello's `best` from the union to the symmetric-difference index.

## The gate — `scripts/`, results in `scores/`

| check | script | what it proves |
|---|---|---|
| verbatim audit | `scripts/assemble/` (rebuilds the package from git by slicing; `audit_loops.py` diffs the loops) | 33 functions moved: 16 byte-identical, 17 differ ONLY in their signature line / a threaded `s`, `runs` argument (21 changed lines, all listed); the three loops differ only by those arguments and the `dry_run` guards |
| call binding | (run in the session: `inspect.signature(...).bind` over every `ast.Call` to a package function) | all 79 calls between package functions bind against the callee's signature — a missed `s` cannot hide in a rarely-run branch |
| decision equivalence | `decisions_check.py` → `scores/decisions_check.json` | the OLD notebook loops, executed as written from git with every scorer / fitter / writer stubbed, and the NEW loops with `dry_run=True`, take identical decisions over the whole `runs/` tree under the queue's environment (`PIM_DW_BASES=frustum,cartesian PIM_SKIP_TOPICS=training_curve`) |
| parity on real runs | `parity_gate.py` → `scores/*.diff.txt` | an already-scored run rescored INTO SCRATCH through `pim.scoring`, every field diffed against (a) the pre-refactor notebook code run side by side on the same model, (b) the `scores.json` on disk. Tolerance 1e-6; bit-identical leaves are counted separately |
| unit tests | `tests/test_scoring_package.py` | the queue's contracts without a GPU: both environment hooks, the still-training guard, replicate target inheritance, what the driver calls missing, scorer dispatch, the block schema and the selection rule as it stands |

`noop_check.py` is the same dry run with a strict exit code (0 only if NOTHING is pending). On
today's tree it exits 1, correctly: 7 seed replicates inherit extra targets whose probes were
never fitted for them (only `appearance-fac` is), and two instances lack floors for extra
targets — the scorer attempts and skips these on every pass ("nothing added (probes not fitted
yet)"), by design. `decisions_check.py` shows the old code reports exactly the same items
(seven runs; two instances = three (instance, architecture) floor sets).

Fixtures: `ray_ablation/L-dw-5ray-20m__seed1` (discworld member: both bases, the inherited
categorical blocks — `appearance-fac` cached, `appearance` uncached and skipped —, IM / IM-NN)
and `adjacent_flip_ablation/L-oth-adjacent-flip-20m` (Othello: canonical block, `mine_signed`,
IM / IM-NN, gates).

## How to re-run

```bash
# from a clone with runs/ and datasets/ symlinked to the live tree (nothing is written there)
export PYTHONPATH=$PWD PIM_DW_BASES=frustum,cartesian PIM_SKIP_TOPICS=training_curve
CUDA_VISIBLE_DEVICES="" .pim/bin/python experiments/master_eval_refactor/scripts/decisions_check.py
systemd-run --user --unit=me_refactor_gate -p MemoryMax=30G --collect --working-directory=$PWD \
  /usr/bin/bash -c '.pim/bin/python -u experiments/master_eval_refactor/scripts/parity_gate.py <topic>/<run> [--old] > logs/master_eval_refactor/gate.log 2>&1'
```

`_nb.py` executes the notebook's OWN cell [2] for SETTINGS / `eval_version`, so the gate always
scores with what production will use; `OLD_COMMIT` (`c023401`) is the last commit with the
scorers inside the notebook.

## Results (2026-09-19, lab box, alongside `rep_dw-5ray_s2`'s training; unit `me_refactor_gate`, 34 min, peak 7.7 GB)

**GREEN — the move changes no number.**

| check | result |
|---|---|
| Othello fixture, OLD notebook code vs NEW `pim.scoring`, same model, same process | **14,610 scored leaves, 14,610 bit-identical**; 0 one-sided keys; SETTINGS snapshot identical. 3.8 min each |
| Discworld fixture, NEW vs the `scores.json` on disk (scored the same morning by the old notebook) | **17,103 scored leaves, 17,103 bit-identical** — frustum, cartesian, `appearance-fac`, every PI / ND / GS / IM / IM-NN arm, skills, per-dim R², tripwire report. 26.2 min. `appearance` (inherited, never fitted for the member): `SKIPPED — no cached probes`, as in production |
| Othello fixture, NEW vs disk (scored 2026-09-12, IM folded in 2026-09-15) | 8,484 bit-identical, 87 within 1e-6, **35 beyond (max 1.5e-5), all of them IM-NN** — the retrieval editor, never tabulated — plus 6,004 keys only in NEW (the case-level spread fields of 2026-09-18). **The OLD code differs from disk in exactly the same 6,039 entries** → drift since that file was written, not the move. Every PI / ND / GS / IM number on disk reproduces |
| decision equivalence, whole tree (32 runs) under the queue's environment | identical: 3 floor items + 7 run items on both sides, **0 full (re)scores**; all ten are skip-when-uncached (the 3 floor items are (instance, arch) pairs on 2 instances) |
| the real entry point: `nbconvert --execute` on the refactored notebook (one current run) | 5 code cells, 0 errors, `skip … (scored at …)`, live `scores.json` / `baselines.json` mtimes unchanged |
| neither fixture's `probes/` dir gained a file | 25 → 25, 31 → 31 |
| rebuild from git (`scripts/assemble/`) | reproduces the committed `pim/scoring/` byte for byte |
| tests / lint | 6 new tests pass (+ 19 neighbouring table / layout tests); `ruff check pim tests`: 47 hits before, 47 after, none in the new files |

So the old scorer IS bit-reproducible against itself on discworld and on every tabulated Othello
number; the 1e-6 tolerance was needed only for untabulated IM-NN arms against a week-old file,
where old and new code drift identically.

`settings` (a snapshot of SETTINGS at write time — it drifts whenever a run joins
`dw_extra_targets`) and `prediction` (folded in by `scripts/score_prediction.py`) are compared
apart from the scored numbers and never decide a verdict (`parity_gate.py::APART`).

Cut-over is the queue operator's: `CUTOVER.md`.
